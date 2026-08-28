#include "rundeck_opts.h"

module atmcol_com
!@sum atmcol_com module containing 1D arrays communicated among column physics components.
!@+   This version is only for use within the scope of the tracer_3Dsource i-j loop, in
!@+   which it is assumed that pressure, temperature, density are held constant (and
!@+   that humidity receives an increment so small that its effect on virtual temperature
!@+   is negligible).
!@+   The filename ATMCOL_COMtr has the tr suffix to prevent merge collisions with
!@+   the module of the same name previously introduced on downstream branches.
  use resolution, only : lm
  use qusdef, only : nmom
  implicit none

!@var pl layer pressure (mb)
!@var plk PL**KAPA (-)
!@var airm the layer's pressure depth (mb)
!@var byam 1./airm (mb-1)
!@var ple pressure at layer edge (mb)
!@var plek PLE**KAPA (-)
  real*8, dimension(lm) :: pl,plk,airm,byam
  real*8, dimension(lm+1) :: ple,plek

! scaffolding in mks units, intended to supplant those in mb units
!@var pres layer pressure (Pa)
!@var dp layer pressure thickness (Pa)
!@var ma layer mass per area (kg/m2)
!@var byma 1/ma
!@var pedge pressure at layer edge (Pa)
  real*8, dimension(lm) :: pres,dp,ma,byma
  real*8, dimension(lm+1) :: pedge

!@var tl in-situ temperature (K)
!@var qv humidity (kg/kg): 1D version of atm_com:q
!@var qvmom moments of humidity Q: 2D version of somtq_com:qmom
!@var rhl relative humidity, or saturation water vapor mixing ratio (0-1)
!@var rhol density calculated from tl (kg/m3)
!@var rhotvl density calculated from virtual temperature (kg/m3)
!@var zl height above nominal sea level (m)
  real*8, dimension(lm) :: tl,qv,rhl,rhol,rhotvl,zl
  real*8, dimension(nmom,lm) :: qvmom

  interface update_qv
    module procedure update_qv_0d
    module procedure update_qv_1d
  end interface update_qv

  interface update_qvmom
    module procedure update_qvmom_1d
    module procedure update_qvmom_2d
  end interface update_qvmom

  contains

  subroutine update_qv_0d(l,q)
!@sum update_qv Calculate new values for qv and rhl for a given q
    use constant, only: lhe
    implicit none
    real*8, intent(in) :: q
    integer, intent(in) :: l
    real*8 :: qsat

    qv(l)=q
    rhl(l)=qv(l)/qsat(tl(l),lhe,pl(l))
  end subroutine update_qv_0d

  subroutine update_qv_1d(q)
!@sum update_qv Calculate new values for ql and rhl for a given q
    implicit none
    real*8, dimension(lm), intent(in) :: q
    integer :: l

    do l=1,lm
      call update_qv_0d(l,q(l))
    enddo
  end subroutine update_qv_1d

  subroutine update_qvmom_1d(l,qmom)
!@sum update_qvmom Fill in new values for qvmom for a given qmom
    use qusdef, only: nmom
    implicit none
    real*8, dimension(nmom), intent(in) :: qmom
    integer, intent(in) :: l
    integer :: m
    do m=1,nmom
      qvmom(m,l)=qmom(m)
    end do
  end subroutine update_qvmom_1d

  subroutine update_qvmom_2d(qmom)
!@sum update_qvmom Fill in new values for qvmom for a given qmom
    implicit none
    real*8, dimension(nmom,lm), intent(in) :: qmom
    integer :: l
    do l=1,lm
      call update_qvmom_1d(l,qmom(:,l))
    end do
  end subroutine update_qvmom_2d

end module atmcol_com

subroutine load_atmcol(i,j)
!@sum load_atmcol for fields which are shared across physics components,
!@+   copy 3D arrays into 1D column arrays
  use atmcol_com
  use constant, only : grav,bygrav,rgas,deltx
  use atm_com, only : ma_3d=>ma,byma_3d=>byma,pk,pek,pmid,pedn,pdsig,t,q,gz
  use somtq_com, only : qmom
!#ifdef TRACERS_ON
!  use tracer_com, only : trm,trm_col
!  use tracer_com, only : trmom,trmom_col
!#endif
  implicit none
  integer, intent(in) :: i,j

  ! various quantities not modified by tracer_3Dsource

  ! pl...byam are still in millibar units for now
  pl(:) = pmid(:,i,j)
  ple(:) = pedn(:,i,j)
  plk(:) = pk(:,i,j)
  plek(:) = pek(:,i,j)
  airm(:) = pdsig(:,i,j)
  byam(:) = 1./airm(:)

  ! mks units pressure and air mass
  pres(:) = pmid(:,i,j)*1d2
  pedge(:) = pedn(:,i,j)*1d2
  dp(:) = pdsig(:,i,j)*1d2
  ma(:) = ma_3d(:,i,j)
  byma(:) = byma_3d(:,i,j)

  ! temperature, humidity, density, height
  tl(:) = t(i,j,:)*plk(:)
  call update_qv(q(i,j,:))
  call update_qvmom(qmom(:,i,j,:))
  rhol(:) = pres(:)/(rgas*tl(:))
  rhotvl(:) = pres(:)/(rgas*tl(:)*(1d0+deltx*qv(:)))
  zl(:) = gz(i,j,:)*bygrav

end subroutine load_atmcol
