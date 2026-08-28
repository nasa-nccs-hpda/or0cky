#include "rundeck_opts.h"

      module AerParam_mod
!@sum This module reads, time-interpolates, and stores fields needed
!@+   by the radiation code in the prescribed-aerosol configuration
!@+   of modelE.  Subroutine updateAerosol2 provides the fields
!@+   used to calculate the direct radiative effects of aerosols.
!@+   Subroutine dCDNC_est provides a parameterized estimate of
!@+   changes of lower-tropospheric CDNC relative to 1850, as input
!@+   to prescriptions of aerosol indirect effects on cloud cover
!@+   and optical depth.
!@+   Dust and volcanic aerosols are ingested via the modules 
!@+   DustParam_mod VolcParam_mod below.
!@auth D. Koch, R. Ruedy
!@auth M. Kelley added comments, reprogrammed for netcdf input

      use timestream_mod, only : timestream
      implicit none
      save
      private

      public :: dCDNC_est
      public :: updateAerosol
      public :: updateAerosol2, get_aero_column
      public :: DRYM2G, aermix

      real*8, allocatable :: anssdd(:,:)
      real*8, allocatable :: mdpi(:,:,:)
      real*8, allocatable :: mdcur(:,:,:)
      real*8, allocatable :: md1850(:,:,:,:)

      character(len=3), dimension(6), parameter :: aernames=(/
!**** Sulfate
     &     'SUL',
!**** Sea Salt
     &     'SSA',
!**** Nitrate
     &     'NIT',
!**** Organic Carbon
     &     'OCA',
!**** Black Carbon from Fossil and bio fuel
     &     'BCA',
!**** Black Carbon from Biomass burning
     &     'BCB'
     &     /)

      real*8 :: DRYM2G(8) =
     &     (/4.667, 0.866, 4.448, 5.017, 9.000, 9.000, 1.000,1.000/)

C     Layer  1    2    3    4    5    6    7    8    9
      INTEGER :: La720=3 ! top low cloud level (aerosol-grid).
                         ! =3 for orig 9-level model
      REAL*8 , PARAMETER ::
     &     Za720=2635.                ! depth of low cloud region (m)
     &    ,byz_cm3 = 1.d-6 / Za720    ! 1d-6/depth in m (+conversion /m3 -> /cm3)
     &    ,byz_gcm3 = 1.d-3 * byz_cm3 ! g vs kg

      REAL*8, dimension(13) :: AERMIX=(/
C      Pre-Industrial+Natural 1850 Level  Industrial Process  BioMBurn
C      ---------------------------------  ------------------  --------
C       1    2    3    4    5    6    7    8    9   10   11   12   13
C      SNP  SBP  SSP  ANP  ONP  OBP  BBP  SUI  ANI  OCI  BCI  OCB  BCB
     + 1.0, 1.0, 1.0, 1.0, 2.5, 2.5, 1.9, 1.0, 1.0, 2.5, 1.9, 2.5, 1.9/)

      integer :: lma  ! ima,jma are now the same as the model grid im,jm
!@var a6jday optical depth for 6 aerosol types
      real*8, allocatable :: a6jday(:,:,:,:)
!@var plbaer pressures of layer interfaces in the aerosol files
      real*8, dimension(:), allocatable :: plbaer

!@var A6streams interface for reading and time-interpolating AERO files
!@+   See usage notes in timestream_mod
      type(timestream), dimension(6) :: A6streams

#ifdef OLD_BCdalbsn
!@var BCdepstream interface for reading and time-interpolating BC_dep file
      type(timestream), public :: BCdepstream
!@var depoBC,depoBC_1990 prescribed black carbon deposition (curr,1990)
!@+   for parameterization of the BC effect on snow albedo
      REAL*8, ALLOCATABLE, DIMENSION(:,:), public :: depoBC,depoBC_1990
#else
!@var BCdalbsnstream interface for reading and time-interpolating BC_dalbsn file
      type(timestream), public :: BCdalbsnstream
!@var BCdalbsn prescribed delta-albedo of snow (units = 1) on land/seaice from BC effects
      REAL*8, ALLOCATABLE, DIMENSION(:,:), public :: BCdalbsn
#endif

      contains

      subroutine dCDNC_EST(i,j,pland, dCDNC) !, table)
!@sum  finds change in cloud droplet number concentration since 1850
!@auth R. Ruedy
!@ver  1.0
      USE CONSTANT, only : pi
      implicit none
      integer, intent(in)  :: i,j ! grid indices
      real*8 , intent(in)  :: pland ! land fraction
      real*8 , intent(out) :: dCDNC ! CDNC(cur)-CDNC(1850)

      real*8, parameter, dimension(5) ::
C                TROPOSPHERIC AEROSOL PARAMETERS
C                  SO4     NO3    OCX    BCB   BCI
     &  f_act=(/ 1.0d0,  1.0d0, 0.8d0, 0.6d0, .8d0/), ! soluble fraction
     &  dens =(/1769d0, 1700d0,  1.d3,  1.d3, 1.d3/)  ! density

      real*8, parameter, dimension(2) ::
C                    Ocean         Land      ! r**3: r=.085,.052 microns
     &  radto3 =(/ 614.125d-24, 140.608d-24/),  ! used for SO4,NO3,OC,BC
     &  scl    =(/     162d0,       298d0/),  ! for Gultepe formula
     &  offset =(/     273d0,       595d0/)   ! for Gultepe formula

      integer it, n
      real*8  An,An0,cdnc(2),cdnc0(2),fbymass1

      do it=1,2  ! ocean, land
        An0 = anssdd(i,j)  !  aerosol number of sea salt and dust
        An  = An0          !  aerosol number of sea salt and dust
        do n=1,4
          fbymass1 =  F_act(n)*(.75d0/pi)/(dens(n)*radto3(it))
          An0 = An0 + mdpi (n,i,j)*fbymass1   ! +fact*tot_mass/part_mass
          An  = An  + mdcur(n,i,j)*fbymass1
        end do
        fbymass1 =  F_act(5)*(.75d0/pi)/(dens(5)*radto3(it))
        An  = An  + mdcur(5,i,j)*fbymass1

        if(An0.lt.1.) An0=1.
        if(An .lt.1.) An =1.
        cdnc0(it) = max( 20d0, scl(it)*log10(AN0)-offset(it))
        cdnc (it) = max( 20d0, scl(it)*log10(AN )-offset(it))
      end do

      dCDNC = (1-pland)*(cdnc(1)-cdnc0(1))+pland *(cdnc(2)-cdnc0(2))
      return
      end subroutine dCDNC_EST

      SUBROUTINE updateAerosol(JYEARA,JJDAYA)
      implicit none
      INTEGER, intent(in) :: jyeara,jjdaya

      call stop_model('updateAerosol: should not get here',255)

      RETURN
      END SUBROUTINE updateAerosol

      subroutine updateAerosol2(jYearA, jJDaya)
!@sum updateAerosol2 reads aerosol file(s) and calculates A6JDAY(lma,6,:,:)
!@+   (dry aerosol Tau) for current day, year.  On startup, it allocates
!@+   a6jday and plbaer, and reads plbaer.  Note that jYearA may be
!@+   negative, which is the Model E method for indicating that the
!@+   data for abs(jYearA) is to be used for all years (this matters
!@+   when time-interpolating between December and January).
! This version gives different results than the previous version, since
! the latter calculated the inter-month time interpolation weight using
!   XMI=(JJDAYA+JJDAYA+31-(JJDAYA+15)/61+(JJDAYA+14)/61)/61.D0
! and read_stream calculates it based on the midpoint dates of the
! current and following months.
! Note that the radiation code still assumes that all aerosols are on the
! same vertical grid.  A per-aerosol plbaer in calls to REPART in the
! radiation code would allow more flexibility.
      use domain_decomp_atm, only : grid
      use model_com, only : jdmidofm ! for md1850 month interp
      use timestream_mod, only : init_stream,read_stream
     &     ,reset_stream_properties,get_by_index,getname_firstfile
      use pario, only : par_open,par_close,read_dist_data,read_data
     &     ,get_dimlen,get_dimlens
      implicit none
c
!@var jyeara current year; negative if the same year is to be repeated
!@var jjdaya current day
      INTEGER, intent(in) :: jyeara,jjdaya
c
      INTEGER m,mi,mj,i,j,l,n,jyearx
      REAL*8 wtmi,wtmj
      REAL*8 xsslt ! ,xdust
      real*8 :: dp,mindp
      logical, save :: init = .false.
      logical :: cyclic
      integer :: dlens(7)
      character(len=32) :: fname1_ssa

      integer :: i_0,i_1,j_0,j_1

      integer :: fid
      real*8, allocatable :: aerarr(:,:,:),arr12(:,:,:,:)

      i_0 = grid%i_strt
      i_1 = grid%i_stop
      j_0 = grid%j_strt
      j_1 = grid%j_stop

      jyearx = abs(jyeara)
      
      if (.not. init) then
        init = .true.

        allocate( md1850 (4,i_0:i_1,j_0:j_1,0:12) )
        allocate(anssdd(i_0:i_1,j_0:j_1))
        allocate(mdpi(4,i_0:i_1,j_0:j_1))
        allocate(mdcur(5,i_0:i_1,j_0:j_1))

        cyclic = jyeara < 0

        ! Init the sea salt file first just to obtain lma,plbaer metadata
        n = 2
        call init_stream(grid,A6streams(n),
     &       'TAero_'//trim(aernames(n)),trim(aernames(n)),
     &       0d0,1d30,'linm2m',jyearx,jjdaya,cyclic=cyclic)
        call getname_firstfile(A6streams(n),fname1_ssa)

        fid = par_open(grid,trim(fname1_ssa),'read')

        !lma = get_dimlen(grid,fid,'lev')
        call get_dimlens(grid,fid,'plbaer',n,dlens)
        lma = dlens(1)-1

        allocate(A6JDAY(lma,6,i_0:i_1,j_0:j_1))

        allocate( plbaer(lma+1) )
        call read_data(grid,fid,'plbaer',plbaer,bcast_all=.true.)

        call par_close(grid,fid)

!**** For parameterized AIE, find level whose top is closest to 720 mb
        mindp = 1d30
        do La720=1,lma
          dp = abs(plbaer(La720)-720d0)
          if(dp > mindp) exit
          mindp = dp
        enddo
        La720 = La720 - 2

        allocate(arr12(grid%i_strt_halo:grid%i_stop_halo,
     &                 grid%j_strt_halo:grid%j_stop_halo,lma,12))
        allocate(aerarr(i_0:i_1,j_0:j_1,12))

        do n=1,6
          if(n.eq.2) cycle ! skip sea salt
          ! Initialize the stream to the year 1850 to extract
          ! the monthly climatology of 1850 aerosols for parameterized AIE
          ! The read_stream call below will jump to the current year.
          call init_stream(grid,A6streams(n),
     &         'TAero_'//trim(aernames(n)),trim(aernames(n)),
     &         0d0,1d30,'linm2m',1850,1,cyclic=.true.)
          do m=1,12
            call get_by_index(grid,A6streams(n),m,arr12(:,:,:,m))
          enddo
          aerarr = byz_cm3 *
     &         SUM(arr12(i_0:i_1,j_0:j_1,1:La720,:), DIM=3)
          select case (n)
          case (1)
            md1850(1,:,:,1:12) = aerarr
          case (3,4,5)
            md1850(n-1,:,:,1:12) = aerarr
          case (6)
            md1850(4,:,:,1:12) = md1850(4,:,:,1:12) + aerarr
          end select
          ! Need this call to allow year jumps for cyclic case
          call reset_stream_properties(grid,A6streams(n),cyclic=cyclic)
        enddo

        md1850(:,:,:,0) = md1850(:,:,:,12)
        deallocate(arr12,aerarr)

      endif ! end init

C**** read and time-interpolate
      allocate(aerarr(grid%i_strt_halo:grid%i_stop_halo,
     &                grid%j_strt_halo:grid%j_stop_halo,lma))
      do n=1,6
        call read_stream(grid,A6streams(n),jyearx,jjdaya,aerarr)
        do j=j_0,j_1
        do i=i_0,i_1
        do l=1,lma
          a6jday(l,n,i,j)=aerarr(i,j,l)
        enddo
        enddo
        enddo
      enddo
      deallocate(aerarr)

! remove AERMIX scalings
      DO J=J_0,J_1
      DO I=I_0,I_1
      DO N=1,6
      DO L=1,lma
        A6JDAY(L,N,I,J)=(1000.D0*DRYM2G(N))*A6JDAY(L,N,I,J)
      ENDDO
      ENDDO
      ENDDO
      ENDDO

C**** Calculate terms for aerosol indirect effect parameterization

      do i=1,13
        if(jjdaya.le.jdmidofm(i)) then
          wtmi = real(jdmidofm(i)-jjdaya,kind=8)/
     &               (jdmidofm(i)-jdmidofm(i-1))
          wtmj = 1d0-wtmi
          mi = i-1
          if(mi > 11) mi=0
          mj = mi+1
          exit
        endif
      enddo

!!!   xdust=.33/(2000.*4.1888*(.40d-6)**3)     ! f/[rho*4pi/3*r^3] (/kg)
      xsslt=1.d0/(2000.*4.1888*(.44d-6)**3) ! x/particle-mass (/kg)

      do j=J_0,J_1
      do i=I_0,I_1

C**** sea salt contribution to anssdd
c SUM to L=5 for low clouds only
        anssdd(i,j) = SUM(A6JDAY(1:La720,2,I,J))/(1000.D0*DRYM2G(2))
     &       *byz_cm3 * Xsslt

C**** SU4,NO3,OCX,BCB,BCI (reordered: no sea salt, no pre-ind BCI)
        mdpi(:,i,j) =
     &       WTMI*md1850(:,i,j,mi) + WTMJ*md1850(:,i,j,mj)
        mdcur(1,i,j) = SUM (A6JDAY(1:La720,1,I,J))*
     &       byz_gcm3/drym2g(1)
        mdcur(2,i,j) = SUM (A6JDAY(1:La720,3,I,J))*
     &       byz_gcm3/drym2g(3)
        mdcur(3,i,j) = SUM (A6JDAY(1:La720,4,I,J))*
     &       byz_gcm3/drym2g(4)
        mdcur(4,i,j) = SUM (A6JDAY(1:La720,6,I,J))*
     &       byz_gcm3/drym2g(6)
        mdcur(5,i,j) = SUM (A6JDAY(1:La720,5,I,J))*
     &       byz_gcm3/drym2g(5)
      end do
      end do

C Misc. comments that were in the previous version of this routine
C                TROPOSPHERIC AEROSOL COMPOSITIONAL/TYPE PARAMETERS
C                   SO4    SEA    ANT    OCX    BCI    BCB   *BCB  *BCB
C     DATA REFDRY/0.200, 1.000, 0.300, 0.300, 0.100, 0.100, 0.200,0.050/
C
C     DATA REFWET/0.272, 1.808, 0.398, 0.318, 0.100, 0.100, 0.200,0.050/
C
C     DATA DRYM2G/4.667, 0.866, 4.448, 5.018, 9.000, 9.000, 5.521,8.169/
C
CKoch DATA DRYM2G/5.000, 2.866, 8.000, 8.000, 9.000, 9.000, 5.521,8.169/
C
C     DATA RHTMAG/1.788, 3.310, 1.756, 1.163, 1.000, 1.000, 1.000,1.000/
C
CRH70 DATA WETM2G/8.345, 2.866, 7.811, 5.836, 9.000, 9.000, 5.521,8.169/
C
C     DATA Q55DRY/2.191, 2.499, 3.069, 3.010, 1.560, 1.560, 1.914,0.708/
C
C     DATA DENAER/1.760, 2.165, 1.725, 1.500, 1.300, 1.300, 1.300,1.300/
C
C     ------------------------------------------------------------------
C          DRYM2G(I) = 0.75/DENAER(I)*Q55DRY(I)/REFDRY(I)
C          WETM2G(I) = DRYM2G(I)*RHTMAG(I)
C          RHTMAG(I) = Rel Humidity TAU Magnification factor  at RH=0.70
C          REFWET(I) = Rel Humidity REFDRY Magnification      at RH=0.70
C     ------------------------------------------------------------------

      return
      end subroutine updateAerosol2

      subroutine get_aero_column (i,j,nlayrs,plb, ataulx)
!@sum Repartitions the aerosols to the current model grid
      integer, intent(in) :: i,j,nlayrs
      real*8 , intent(in) :: plb(nlayrs+1)
      real*8 , intent(out) :: ataulx(nlayrs,6)
      integer na

      do na=1,6 ! loop over 6 aerosol types
        call repart(a6jday(1,na,i,j),plbaer,lma+1,  ! in
     *              ataulx(1,na),    plb,nlayrs+1)  ! out,   in
      end do

      return
      end subroutine get_aero_column

      end module AerParam_mod

#ifdef OLD_BCdalbsn
      subroutine updBCd(year)
!@sum updBCd reads timeseries file for black carbon deposition
!@+   and interpolates depoBC to requested year.
!@auth R. Ruedy, M. Kelley
      use domain_decomp_atm, only : grid,getDomainBounds
      use timestream_mod, only : init_stream,read_stream
      use AerParam_mod, only: BCdepstream,depoBC,depoBC_1990
      implicit none
      integer, intent(in) :: year
c
      logical, save :: init = .false.
      integer :: i_0h,i_1h,j_0h,j_1h
      integer :: day

      day = 1 ! to pass a required argument

      if (.not. init) then
        init = .true.

        call getDomainBounds(grid, i_strt_halo=i_0h, i_stop_halo=i_1h,
     &                             j_strt_halo=j_0h, j_stop_halo=j_1h)
        allocate(depoBC     (i_0h:i_1h, j_0h:j_1h),
     &           depoBC_1990(i_0h:i_1h, j_0h:j_1h))

        call init_stream(grid,BCdepstream,'BC_dep','BC_dep',
     &       0d0,1d30,'none',year,day)
      endif

      call read_stream(grid,BCdepstream,year,day,depoBC)

      end subroutine updBCd
#else
      subroutine updBCdalbsn(year,day)
!@sum updBCdalbsn reads timeseries file for black carbon delta-snow-albedo
!@+   and interpolates to the requested day/year.   If the year is negative,
!@+   this is interpreted as indicating perpetual-year mode, as per the
!@+   convention for numerous radiation input files.
!@auth R. Ruedy, M. Kelley
      use domain_decomp_atm, only : grid,getDomainBounds
      use timestream_mod, only : init_stream,read_stream
      use AerParam_mod, only: BCdalbsnstream,BCdalbsn
      implicit none
      integer, intent(in) :: year,day
c
      logical, save :: init = .false.
      integer :: i_0h,i_1h,j_0h,j_1h
      logical :: cyclic
      integer :: absyr

      absyr = abs(year)
      if (.not. init) then
        init = .true.

        call getDomainBounds(grid, i_strt_halo=i_0h, i_stop_halo=i_1h,
     &                             j_strt_halo=j_0h, j_stop_halo=j_1h)
        allocate(BCdalbsn(i_0h:i_1h, j_0h:j_1h))
        BCdalbsn = 0.
        cyclic = year < 0
        call init_stream(grid,BCdalbsnstream,'BCdalbsn','BCdalbsn',
     &       -1d30,1d30,'linm2m',absyr,day,cyclic=cyclic)
      endif

      call read_stream(grid,BCdalbsnstream,absyr,day,BCdalbsn)
      BCdalbsn = BCdalbsn / 100d0 ! units conversion from % to 1

      end subroutine updBCdalbsn
#endif

      module DustParam_mod
!@sum This module reads, time-interpolates, and stores fields needed
!@+   by the radiation code in the prescribed-dust configuration
!@+   of modelE.   The logic follows that of AerParam_mod.
!@+   The interface routine is upddst2().
!@auth R. Miller original version
!@auth M. Kelley reprogrammed for new-style time-varying input
      use timestream_mod, only : timestream
      implicit none

!@var ddjday (kg/m2/layer) dust amount for each size class, layer, and column
!@+   for the current day
#if !defined(MININT_RADSW)
      real*8, dimension(:,:,:,:), allocatable :: ddjday
#else
      real*8, dimension(:,:,:,:), allocatable :: miox, mhos, macc

      real*8, dimension(:,:), allocatable :: eiox, siox, giox
      real*8, dimension(:,:), allocatable :: ehos, shos, ghos
      real*8, dimension(:,:), allocatable :: eacc, sacc, gacc
      real*8, dimension(:), allocatable :: qiox, qhos, qacc, deff
#endif

!@var {lmd,nsized} number of {layers, size classes} in DUSTaer input file
#if !defined(MININT_RADSW)
      integer :: lmd,nsized
#else
      integer :: hhz, ssx, eez, nsized
#endif

!@var DUSTaerstream interface for reading and time-interpolating DUSTaer files
!@+   See usage notes in timestream_mod
#if !defined(MININT_RADSW)
      type(timestream) :: DUSTaerstream
#else
      type(timestream) :: MIOX_stream
      type(timestream) :: MHOS_stream
      type(timestream) :: MACC_stream
#endif

!@var is_initialized whether the DUSTaer stream has been initialized
!@+   and various arrays allocated
      logical :: is_initialized=.false.

!@var plbdust nominal edge pressures of DUSTaer file layers
!@var {re,ro}dust radii of DUSTaer file size classes
#if !defined(MININT_RADSW)
      real*8, dimension(:), allocatable :: redust, rodust, plbdust
#else
      real*8, dimension(:), allocatable :: redust
#endif

      contains

      subroutine upddst2(jyeard,jjdayd)
      use domain_decomp_atm, only : grid
      use timestream_mod, only : init_stream,read_stream,
     &     getname_firstfile
      use pario, only : par_open,par_close
     &     ,get_dimlens,read_data
      implicit none
!@var jyeard, jjdayd year and day of the data to read into ddjday.
!@+   Note that jyeard may be negative, which is the Model E method for
!@+   indicating that the data for abs(jyeard) is to be used for all years.
      integer, intent(in) :: jyeard,jjdayd
!
      integer :: i_0,i_1,j_0,j_1,i,j,n,jyearx
      logical :: cyclic
#if !defined(MININT_RADSW)
      real*8, dimension(:,:,:,:), allocatable :: ddjday_transp
#endif
      integer :: fid,ndims,dlens(7)
      character(len=32) :: fname1_dust

      i_0 = grid%i_strt
      i_1 = grid%i_stop
      j_0 = grid%j_strt
      j_1 = grid%j_stop

      cyclic = jyeard < 0
      jyearx = abs(jyeard)

      if(.not. is_initialized) then
        is_initialized = .true.

#if !defined(MININT_RADSW)
        call init_stream(grid,DUSTaerstream,
     &       'DUSTaer','DUST',
     &       0d0,1d30,'linm2m',jyearx,jjdayd,cyclic=cyclic)


        ! read dust metadata
        call getname_firstfile(DUSTaerstream,fname1_dust)
        fid = par_open(grid,trim(fname1_dust),'read')
        call get_dimlens(grid,fid,'DUST',ndims,dlens)
        lmd    = dlens(3)
        nsized = dlens(4)
        allocate( plbdust(lmd+1), redust(nsized), rodust(nsized) )
        call read_data(grid,fid,'plbdust',plbdust,bcast_all=.true.)
        call read_data(grid,fid,'redust',redust,bcast_all=.true.)
        call read_data(grid,fid,'rodust',rodust,bcast_all=.true.)
        call par_close(grid,fid)

        allocate( ddjday(lmd,nsized,i_0:i_1,j_0:j_1) )
#else
#if defined(EXTMIX_VARYING)
       call init_stream(grid,MIOX_stream,'MINERALSaer','VIOX',
     &      0.0d0,1.0d30,'linm2m',jyearx,jjdayd,cyclic=cyclic)
       call init_stream(grid,MHOS_stream,'MINERALSaer','VHOS',
     &      0.0d0,1.0d30,'linm2m',jyearx,jjdayd,cyclic=cyclic)
       call init_stream(grid,MACC_stream,'MINERALSaer','VACC',
     &      0.0d0,1.0d30,'linm2m',jyearx,jjdayd,cyclic=cyclic)
#elif defined(EXTMIX_UNIFORM)
       call init_stream(grid,MIOX_stream,'MINERALSaer','UIOX',
     &      0.0d0,1.0d30,'linm2m',jyearx,jjdayd,cyclic=cyclic)
       call init_stream(grid,MHOS_stream,'MINERALSaer','UHOS',
     &      0.0d0,1.0d30,'linm2m',jyearx,jjdayd,cyclic=cyclic)
       call init_stream(grid,MACC_stream,'MINERALSaer','UACC',
     &      0.0d0,1.0d30,'linm2m',jyearx,jjdayd,cyclic=cyclic)
#endif

       ! read dust metadata
       call getname_firstfile(MIOX_stream,fname1_dust)
       fid = par_open(grid,trim(fname1_dust),'read')
       call get_dimlens(grid,fid,'VIOX',ndims,dlens)
       eez = 6
       hhz = dlens(3)
       ssx = dlens(4)
       allocate( eiox(eez,ssx), siox(eez,ssx), giox(eez,ssx) )
       allocate( ehos(eez,ssx), shos(eez,ssx), ghos(eez,ssx) )
       allocate( eacc(eez,ssx), sacc(eez,ssx), gacc(eez,ssx) )
       allocate( qiox(ssx), qhos(ssx), qacc(ssx), deff(ssx) )
       call read_data(grid,fid,'EIOX',eiox,bcast_all=.true.)
       call read_data(grid,fid,'SIOX',siox,bcast_all=.true.)
       call read_data(grid,fid,'GIOX',giox,bcast_all=.true.)
       call read_data(grid,fid,'EHOS',ehos,bcast_all=.true.)
       call read_data(grid,fid,'SHOS',shos,bcast_all=.true.)
       call read_data(grid,fid,'GHOS',ghos,bcast_all=.true.)
       call read_data(grid,fid,'EACC',eacc,bcast_all=.true.)
       call read_data(grid,fid,'SACC',sacc,bcast_all=.true.)
       call read_data(grid,fid,'GACC',gacc,bcast_all=.true.)
       call read_data(grid,fid,'QIOX',qiox,bcast_all=.true.)
       call read_data(grid,fid,'QHOS',qhos,bcast_all=.true.)
       call read_data(grid,fid,'QACC',qacc,bcast_all=.true.)
       call read_data(grid,fid,'REFF',deff,bcast_all=.true.)
       call par_close(grid,fid)

       ! maybe necessary in WRITER (in WRITET, whole upddst2 is used)
       nsized = ssx
       allocate( redust(nsized) )
       redust = deff

       allocate( miox(i_0:i_1,j_0:j_1,hhz,ssx) )
       allocate( mhos(i_0:i_1,j_0:j_1,hhz,ssx) )
       allocate( macc(i_0:i_1,j_0:j_1,hhz,ssx) )
#endif

      endif

#if !defined(MININT_RADSW)
      allocate( ddjday_transp(
     &     grid%i_strt_halo:grid%i_stop_halo,
     &     grid%j_strt_halo:grid%j_stop_halo,
     &     lmd,nsized) )

      call read_stream(grid,DUSTaerstream,jyearx,jjdayd,ddjday_transp)

      do j=j_0,j_1
      do i=i_0,i_1
        ddjday(:,:,i,j) = ddjday_transp(i,j,:,:)
      enddo
      enddo

      deallocate ( ddjday_transp )
#else
      call read_stream(grid,MIOX_stream,jyearx,jjdayd,miox)
      call read_stream(grid,MHOS_stream,jyearx,jjdayd,mhos)
      call read_stream(grid,MACC_stream,jyearx,jjdayd,macc)
#endif

      end subroutine upddst2

#if !defined(MININT_RADSW)
      subroutine get_dust_column (i,j,Nlayrs,PLB, DTAULX)
!@sum Repartitions the dust to the current model grid
      integer, intent(in) :: i,j,nlayrs
      real*8 , intent(in) :: plb(nlayrs+1)
      real*8 , intent(out) :: dtaulx(nlayrs,nsized)
      integer n

      do n=1,nsized
        call repart(ddjday(1,n,i,j),   PLBdust,lmd+1,  ! in
     *              dtaulx(1,n),       plb, nlayrs+1)  ! out,   in
      end do

      return
      end subroutine get_dust_column
#endif

      end module DustParam_mod

      module VolcParam_mod
!@sum This module reads, time-interpolates, and stores fields needed
!@+   by the radiation code in the prescribed-volcanic aerosol
!@+   configuration of modelE. (effective radius, optical depth)
!@+   The interface routine is updvol2().
!@auth R. Ruedy original version
!@auth M. Kelley reprogrammed for new-style time-varying input
!strm use timestream_mod, only : timestream
      implicit none

!@var vtau/vreff optical depth/effective radius of volcanic aerosols
      real*8, dimension(:,:,:), allocatable :: VTauTJK ! (time,lat,lev)
      real*8, dimension(:,:), allocatable :: VReffTJ ! (time,lat)
      real*8 vreffJ(46)                  ! (lat) reference grid
      REAL*8, save, allocatable :: gdata(:), VtauL(:), VtauJL(:,:)
      integer :: vmi,vme

!@var LMv,JMv number of layers, lat-zones in VOLCaer input file
      integer :: LMv,JMv    ,NVOLMON ! # of avail. months
!@var Latitude and Layer edges (km) of data file grid
      real*8, dimension(:), allocatable :: ELATVOL,HVOLKM

!@var VOLCaerstream interface for reading and time-interpolating VOLCaer files
!@+   See usage notes in timestream_mod
!strm type(timestream) :: VOLCaerstream

!@var hlbvolc nominal edge heights of VOLCaer file layers
!@var Reff effective radius of volc. aerosol - latitude dep
!strm real*8, dimension(:), allocatable :: HVOLKM,    volc

!@var is_initialized whether the VOLCaer stream has been initialized
!@+   and various arrays allocated
      logical :: is_initialized=.false.

      contains

      subroutine updvol2(jyearv,jjdayv,kyearvm)
      use domain_decomp_atm, only : grid,am_i_root
!strm use timestream_mod, only : init_stream,read_stream,
!strm&     getname_firstfile
!strm use pario, only : par_open,par_close,get_dimlens,read_data
      use pario, only : par_open,par_close,get_dimlens,read_data,
     &  read_attr,variable_exists,get_dimlen
      use filemanager, only : openunit, closeunit, is_fbsa 
      implicit none
!@var jyeard, jjdayd year and day of the data to read into ddjday.
!@+   Note that jyeard may be negative, which is the Model E method for
!@+   indicating that the data for abs(jyeard) is to be used for all years.
      integer, intent(in) :: jyearv,jjdayv
      integer, dimension(2), intent(in) :: kyearvm

!            for binary input files (obsolescent)
      CHARACTER*80 TITLE
      real*4, dimension(:,:), allocatable :: VTAUR4
      real*4, allocatable :: vtau4(:,:,:),vreff4(:,:),hv4(:),lat4(:)
      real*8, dimension(:), allocatable :: VTauT
      INTEGER :: I,J,K,L,M,N,N1,N2,NRFU,mi,mj,fid,idum
      REAL*8 XYYEAR,XYI,WMI,WMJ
      INTEGER, SAVE :: JVOLYI,JVOLYE
      REAL*8, SAVE :: E46LAT(47),TAULAT(46)
      INTEGER, SAVE :: NJ46

!strm logical :: cyclic
!strm real*8, dimension(:,:,:,:), allocatable :: ddjday_transp
!strm integer :: fid,ndims,dlens(7),jyearx
!strm character(len=32) :: fname1_volc
  
!strm cyclic = jyearv < 0      
!strm jyearx = abs(jyearv)     

      if(.not. is_initialized) then
        is_initialized = .true.

C-----------------------------------------------------------------------
CR(7)        Read Stratospheric Volcanic binary data
C            (NVOLMON months (years JVOLYI to JVOLYE) x JMv latitudes)
C            If jyearv<0 use the NVOLMON-month mean as background aerosol
C            ---------------------------------------------------------

        if (.not.is_fbsa('RADN7')) then 
          fid=par_open(grid,'RADN7','read')
          NVOLMON=get_dimlen(grid,fid,'date')
          JMv=get_dimlen(grid,fid,'lat')
          LMv=get_dimlen(grid,fid,'layers')
          ALLOCATE (VTauTJK(NVOLMON,JMv,LMv))
          ALLOCATE (VReffTJ(NVOLMON,JMv))
          ALLOCATE (HVOLKM(LMv+1),ELATVOL(JMv+1))
          ALLOCATE (hv4(LMv+1),LAT4(JMv+1))
          if(variable_exists(grid,fid,'zlayer'))then
            call read_data(grid,fid,'zlayer',HVOLKM,bcast_all=.true.)
          else
            call stop_model('missing zlayer in RADN7 file',255)
          endif
          if(variable_exists(grid,fid,'elat'))then
            call read_data(grid,fid,'elat',ELATVOL,bcast_all=.true.)
          else
            call stop_model('missing elat in RADN7 file',255)
          endif
          if(variable_exists(grid,fid,'Reff'))then
            call read_data(grid,fid,'Reff',VReffTJ,bcast_all=.true.)
          else
            call stop_model('missing Reff in RADN7 file',255)
          endif
          if(variable_exists(grid,fid,'tau'))then
            call read_data(grid,fid,'tau',VTauTJK,bcast_all=.true.)
          else
            call stop_model('missing tau in RADN7 file',255)
          endif
          call par_close(grid,fid)
 
        else !not a binary file  
          call openunit('RADN7',nrfu,.true.,.true.)
          READ (NRFU) TITLE ; rewind NRFU
          IF(TITLE(7:12).ne.'Header') 
     &      call stop_model('updvol2: use new type header file',255)
          READ (NRFU) TITLE,NVOLMON,JVOLYI,JVOLYE,JMv,LMv
          ALLOCATE (VTauTJK(NVOLMON,JMv,LMv))
          ALLOCATE (VTau4(NVOLMON,JMv,LMv))
          ALLOCATE (VReff4(NVOLMON,JMv))
          ALLOCATE (VReffTJ(NVOLMON,JMv))
          ALLOCATE (HVOLKM(LMv+1),ELATVOL(JMv+1))
          ALLOCATE (hv4(LMv+1),LAT4(JMv+1))
          READ (NRFU) TITLE,VTau4  ; VTauTJK = VTau4
          READ (NRFU) TITLE,VReff4 ; VReffTJ = VReff4
          READ (NRFU) TITLE,hv4    ; HVOLKM  = hv4
          READ (NRFU) TITLE,LAT4   ; ELATVOL = lat4
          call closeunit(nrfu)
          deallocate (VTau4,VReff4,hv4,LAT4)
        endif        
       
        if(jyearv == -1) then
          allocate (VTauT(NVOLMON)) 
          if (kyearvm(1)<jvolyi) then
            if (am_i_root()) print*,'kyearvm=',kyearvm,' jvolyi=',jvolyi
            call stop_model('Check kyearvm and jvolyi.',255)
          else
            vmi=(kyearvm(1)-jvolyi)*12+1
          endif
          if (kyearvm(2)>jvolye) then
            if (am_i_root()) print*,'kyearvm=',kyearvm,' jvolye=',jvolye
            call stop_model('Check kyearvm and jvolye.',255)
          else
            vme=(kyearvm(2)-jvolyi+1)*12
          endif
          do j=1,JMv
! Select between no weighting (old default) or weighting on AOD (new default
! as of Septermber 2022).
!          VReffTJ(1,J)=sum(VReffTJ(:,j))/nvolmon
            VTauT=sum(VTauTJK(:,j,:),2)
            VReffTJ(1,J)=sum(VReffTJ(vmi:vme,j)*VTauT(vmi:vme))
     &                  /sum(VTauT(vmi:vme))
            do k=1,LMv
              VTauTJK(1,J,K)=sum(VTauTJK(vmi:vme,j,k))/(vme-vmi+1)
            end do
          end do
        end if
C                   Set Grid-Box Edge Latitudes for Data Repartitioning
C                   ---------------------------------------------------
        NJ46=46+1
        DO J=2,46
          E46LAT(J)=-90.D0+(J-1.5D0)*180.D0/45
        END DO
        E46LAT(   1)=-90.D0
        E46LAT(NJ46)= 90.D0
        allocate (gdata(JMv),VtauL(LMv),VtauJL(NJ46,LMv))
      end if ! is_initialized
C                                          (Volcanic data)
C                                          -------------------------
      XYYEAR=JYEARV+JJDAYV/366.D0              ! time interpolation (lin)
      IF(XYYEAR < JVOLYI) XYYEAR=JVOLYI
      XYI=(XYYEAR-JVOLYI)*12.D0+1.D0
      IF(XYI > NVOLMON - .001D0) XYI=NVOLMON-.001D0
      MI=XYI
      WMJ=XYI-MI
      WMI=1.D0-WMJ
      MJ=MI+1
      DO 250 J=1,JMv
      GDATA(J)=WMI*VReffTJ(MI,J)+WMJ*VReffTJ(MJ,J)
  250 CONTINUE                                    ! hor. interpolation
      CALL RETERP(GDATA,ELATVol,JMv+1,vreffj,E46LAT,NJ46)
      DO 270 K=1,LMv
      DO 260 J=1,JMv
      GDATA(J)=WMI*VTauTJK(MI,J,K)+WMJ*VTauTJK(MJ,J,K)
  260 CONTINUE
      CALL RETERP(GDATA,ELATVOL,JMv+1,VtauJL(1,K),E46LAT,NJ46)
  270 CONTINUE
      return
      end subroutine updvol2

      subroutine get_volc_column (jlat,Nlayrs,HLB, vReff,VTAULX)
!@sum Vertically repartitions to the current model grid
      integer, intent(in) :: jlat,nlayrs
      real*8 , intent(in) :: hlb(nlayrs+1)
      real*8 , intent(out) :: vtaulx(nlayrs), vReff

      VtauL(:) = VtauJL(jlat,:)
      call repart(VtauL,hvolkm,LMv+1, VTAULX,hlb,nlayrs+1)

      vreff = vreffj(jlat)

      return
      end subroutine get_volc_column

      end module VolcParam_mod
