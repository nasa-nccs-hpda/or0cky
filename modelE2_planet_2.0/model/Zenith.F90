module Zenith_mod
   implicit none
   private

   public :: calc_zenith_angle

contains

   subroutine calc_zenith_angle(clock, nStep, cosz, cosza)
!@sum calculate zenith angle for current time step
!@auth Gavin Schmidt (from RADIA)  (revised by T. Clune)
!@var nStep - number of timesteps over which to average
!@var cosz - array of zenith angles weighted by time
!@var cosza - weighted by incident light

      use CONSTANT, only : twopi, pi
      use ModelClock_mod
      use Rational_mod
      use BaseTime_mod, only: BaseTime, newBaseTime
      use Time_mod
      USE TimeInterval_mod

      use TimeConstants_mod, only: SECONDS_PER_DAY
      USE RAD_COM, only : useOrbit => orbit
      USE RAD_COSZ0, only : coszt, coszs
      use DOMAIN_DECOMP_1d, only: am_i_root
      use DOMAIN_DECOMP_1d, only: am_i_root, pack_data
      use resolution, only: IM, JM
      USE DOMAIN_DECOMP_ATM, only: grid                                                                                                                                                                            

      IMPLICIT NONE

      type (ModelClock), intent(in) :: clock
      integer, intent(in) :: nStep
      real*8, intent(inout) :: cosz(:,:)
      real*8, optional, intent(inout) :: cosza(:,:)

      real*8 :: rot1, rot2, drot
      type (Time) :: t1, t2
      type (TimeInterval) :: sPerDay
      real*8, parameter :: TINY = 1.d-10 ! radians

      t1 = clock%getCurrentTime()
      rot1 = useOrbit%getHourAngle(t1)

      t2 = t1
      call t2%add(nstep * clock%getDt())
      rot2 = useOrbit%getHourAngle(t2)

      drot = rot2 - rot1
      ! Choose the proper primary interval: [-pi, +pi)
      ! Drot should then always be "near zero".
      drot = -pi + modulo(drot+pi, 2*pi)
      if(abs(drot) > pi/4) call stop_model ("Time step is too large for accurate radiation.")

      if (drot < 0) then
         rot2 = rot1
         rot1 = rot1 + drot
      elseif (drot > 0) then
         rot2 = rot1 + drot
      else
         ! Introduce tiny delta to avoid divide-by-zero in COSZ for tidally locked case
         rot2 = rot2 + TINY
      end if

      !if (am_i_root()) write(55,*),'rot1, rot2: ', rot1 *180/pi, rot2*180/pi
      if (.not. present(cosza)) then
         call coszt (rot1, rot2, cosz)
      else
         call coszs (rot1, rot2, cosz, cosza)
      end if

   contains

      subroutine swap(a, b)
         real*8, intent(inout) :: a
         real*8, intent(inout) :: b
         real*8 :: tmp

         tmp = a
         a = b
         b = tmp
      end subroutine swap

   end subroutine calc_zenith_angle

end module Zenith_mod

