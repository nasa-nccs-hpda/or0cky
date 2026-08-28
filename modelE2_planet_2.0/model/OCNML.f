#include "rundeck_opts.h"
!@sum  Module OCNML contains the subroutines needed for "q-flux" runs that
!@+    prescribe the convergence of ocean heat transport in the mixed
!@+    layer, and predict ice cover and mixed-layer temperature.
!@auth G. Schmidt
!@auth M. Kelley restructuring, comments, netcdf-based input options
!@ver  1.0
      module ocnml
      use timestream_mod, only : timestream
      implicit none
      private

!@cont OSTRUC,OSOURC,
!@+    RUN_OCNML,PRECIP_OCNML,SET_GTEMP_OCNML,
!@+    alloc_mldepth,init_OCNML,daily_OCNML
      public ::
     &      init_ocnml,init_mldepth,init_qfluxes
     &     ,alloc_mldepth,alloc_qfluxes,alloc_ohtconv
     &     ,set_gtemp_ocnml,daily_ocnml,read_mldepth,read_mldepth_now
     &     ,precip_ocnml,run_ocnml,ostruc

      public :: tocean,z1o,z12o,fgeotherm
!@var Z1O current ocean mixed layer depth (m) updated once per day
!@var Z12O annual maximum ocean mixed layer depth (m)
!@var Z1OOLD value of Z1O during previous day (m)
      REAL*8, ALLOCATABLE, DIMENSION(:,:) :: Z1O,Z12O,Z1OOLD
!@var FGEOTHERM  geothermal heat flux (W/m2)
      REAL*8, ALLOCATABLE, DIMENSION(:,:) :: fgeotherm
!@var TOCEAN temperature of the ocean (C)
!@+   index 1 holds ML temp between the surface and Z1O
!@+         2 holds temperature between Z1O and Z12O
!@+         3 holds the temperature below Z12O
      REAL*8, ALLOCATABLE, DIMENSION(:,:,:) :: TOCEAN

!@var OTC annual mean OHT convergence in ML
      REAL*8, ALLOCATABLE, DIMENSION(:,:) :: OTC
!@var OTB,OTC Fourier expansion coefficients of annual cycle of OHT
!@+   convergence truncated to wavenumber 4 (OTA for sin, OTB for cos)
      REAL*8, ALLOCATABLE, DIMENSION(:,:,:) :: OTA,OTB
!@dbparam old_oht 0: new scheme, 1: old scheme
      integer :: old_oht=1

!@dbparam qfluxX multiplying factor for qfluxes
      REAL*8 :: qfluxX=1.

!@var Z1Ostream interface for reading and time-interpolating ML depth file
!@+   See usage notes in timestream_mod
      type(timestream) :: Z1Ostream

!@var oht_is_timestream - controlled by old_oht rundeck parameter
!!!!  logical :: oht_is_timestream=.false. ! for old scheme
      logical :: oht_is_timestream=.true.  ! for new scheme

!@var OHTstream interface for reading and time-interpolating OHT file
!@+   See usage notes in timestream_mod
      type(timestream) :: OHTstream

!@var ohtconv_now (W/m2) ocean heat flux convergence for current day
      real*8, allocatable, dimension(:,:), public :: ohtconv_now

!@dbparam z12o_max maximal mixed layer depth (m) for qflux model
      real*8, public :: z12o_max=-1,z12o_max_rsf=-1

!@dbparam year1_ohtconv first year of accumulation for ohtconv
      integer, public :: year1_ohtconv=0
!@var do_ohtconv whether to perform ohtconv accumulation (true if year1_ohtconv set)
      logical, public :: do_ohtconv=.false.
!@var acctime_ohtconv (s) number of seconds over which ohtconv has been accumulated
      real*8, dimension(12), public :: acctime_ohtconv=0.
!@var ohtconv (J/m2) implied time-integrated ocean heat transport convergence
!@+   climatology. It is constructed from the ML+seaice heat budget identity
!@+   hfinal - hinitial = hflx_surface + hsiconv + ohtconv
!@+   where ML = mixed-layer, hfinal and hinitial are the final and initial
!@+   ML+seaice heat contents for each month, hflx_surface is the time integrated
!@+   heat flux averaged over the open ocean and sea ice, including the latent
!@+   heat of freshwater input from precipitation and icebergs, and hsiconv is the
!@+   implied time integrated convergence of seaice heat from ice advection.
!@+
      real*8, allocatable, dimension(:,:,:), public :: ohtconv
!@var ocnht_sv (J/m2) instantaneous ML+seaice heat content, used for computing
!@+   time differences of heat content appearing in the ohtconv budget identity
      real*8, allocatable, dimension(:,:), public :: ocnht_sv

      contains

      subroutine alloc_qfluxes ! including geothermal fluxes
!@sum alloc_qfluxes allocate heat transport arrays belonging to the ocnml module
      use domain_decomp_atm, only : grid,getDomainBounds
      use pario, only : par_open,par_close,read_dist_data
      use Dictionary_mod, only : sync_param
      use filemanager, only : file_exists

      implicit none
!@dbparam geotherm_ocnX scaling factor for geothermal heat flux over ocean
      real*8 :: geotherm_ocnX = 0. ! reset to 1 IF geoth. heat flux is read in
      integer :: fid, i_0h,i_1h,j_0h,j_1h
      call getDomainBounds(grid,i_strt_halo=i_0h,i_stop_halo=i_1h)
      call getDomainBounds(grid,j_strt_halo=j_0h,j_stop_halo=j_1h)

      allocate(fgeotherm(i_0h:i_1h,j_0h:j_1h))
      fgeotherm = 1d0 ! default - to be scaled by geotherm_ocnX
      if(file_exists('GEOTHERM_LAND')) then
        fid = par_open(grid,'GEOTHERM_LAND','read')
        call read_dist_data(grid,fid,'fgeotherm',fgeotherm)
        call par_close(grid,fid)
        geotherm_ocnX = 1.
      endif
      call sync_param("geotherm_ocnX",geotherm_ocnX)
      fgeotherm = geotherm_ocnX * fgeotherm

      allocate(ohtconv_now(i_0h:i_1h,j_0h:j_1h))
      ohtconv_now(:,:) = 0.
      allocate(
     &         ota(i_0h:i_1h,j_0h:j_1h,4),
     &         otb(i_0h:i_1h,j_0h:j_1h,4),
     &         otc(i_0h:i_1h,j_0h:j_1h)
     &     )
      end subroutine alloc_qfluxes

      subroutine alloc_mldepth
!@sum alloc_mldepth allocate arrays belonging to the ocnml module
      use domain_decomp_atm, only : grid,getDomainBounds
      implicit none
      integer :: i_0h,i_1h,j_0h,j_1h
      call getDomainBounds(grid,i_strt_halo=i_0h,i_stop_halo=i_1h)
      call getDomainBounds(grid,j_strt_halo=j_0h,j_stop_halo=j_1h)
      allocate(tocean(3,i_0h:i_1h,j_0h:j_1h))
      allocate(
     &         z1o(i_0h:i_1h,j_0h:j_1h),
     &         z1oold(i_0h:i_1h,j_0h:j_1h),
     &         z12o(i_0h:i_1h,j_0h:j_1h)
     &     )
      end subroutine alloc_mldepth

      subroutine alloc_ohtconv
!@sum alloc_ohtconv allocate arrays belonging to the ocnml module
      use domain_decomp_atm, only : grid,getDomainBounds
      implicit none
      integer :: i_0h,i_1h,j_0h,j_1h
      call getDomainBounds(grid,i_strt_halo=i_0h,i_stop_halo=i_1h)
      call getDomainBounds(grid,j_strt_halo=j_0h,j_stop_halo=j_1h)
      allocate(ohtconv(i_0h:i_1h,j_0h:j_1h,12))
      ohtconv(:,:,:) = 0.
      allocate(ocnht_sv(i_0h:i_1h,j_0h:j_1h))
      ocnht_sv(:,:) = 0.
      end subroutine alloc_ohtconv

      subroutine read_mldepth_now(end_of_day,atmocn)
!@sum daily_ocnml updates mixed layer depth to the value for the current day
      use model_com, only : itime,itimei
      use model_com, only :  modelEclock
      use exchange_types, only : atmocn_xchng_vars
      implicit none
      logical, intent(in) :: end_of_day
      type(atmocn_xchng_vars) :: atmocn
!
      integer :: jyear,jday

c**** save previous day's mixed layer depth
      z1oold=z1o

      if(.not.(end_of_day.or.itime.eq.itimei)) return

      call modelEclock%get(year=jyear, dayOfYear=jday)

      call read_mldepth(atmocn,jyear,jday)

      end subroutine read_mldepth_now

      subroutine read_mldepth(atmocn,jyear,jday,save_old)
!@sum daily_ocnml updates mixed layer depth to the value for the current day
      use domain_decomp_atm, only : getDomainBounds,grid
      use timestream_mod, only : read_stream
      use exchange_types, only : atmocn_xchng_vars
      implicit none
      type(atmocn_xchng_vars) :: atmocn
      integer :: jyear,jday
      logical, optional :: save_old
!
      integer :: i,j, j_0,j_1,i_0,i_1

      if(present(save_old)) then
        if(save_old) z1oold=z1o
      endif

      call getDomainBounds(grid,j_strt=j_0,j_stop=j_1)
      i_0 = grid%i_strt
      i_1 = grid%i_stop

C**** interpolate the mixed layer depth z1o to the current day
      call read_stream(grid,Z1Ostream,jyear,jday,Z1O)
C**** limit z1o to the annual-maximum mixed layer depth z12o
      do j=j_0,j_1
      do i=i_0,atmocn%imaxj(j)
        z1o(i,j)=min( z12o(i,j) , z1o(i,j) )
      end do
      end do

      end subroutine read_mldepth

      subroutine daily_ocnml(end_of_day,atmocn,atmice)
!@sum daily_ocnml updates some fourier expansion terms for oht convergence
!@+   and some diagnostics
      use domain_decomp_atm, only : getDomainBounds,grid
      use diag_com, only : aij=>aij_loc,ij_toc2,ij_tgo2
      use constant, only : twopi,shw,rhows
      use Constant, only : daysPerYear
      use model_com, only :  modelEclock
      use exchange_types, only : atmocn_xchng_vars,atmice_xchng_vars
      use timestream_mod, only : read_stream
      implicit none
      logical, intent(in) :: end_of_day
      type(atmocn_xchng_vars) :: atmocn
      type(atmice_xchng_vars) :: atmice
      real*8 angle
      integer :: i,j, j_0,j_1,i_0,i_1
      integer :: jyear,jday
!@var SINANG,COSANG, [SN,CS]nANG sine, cosine of multiples n
!@+   of the time of year (time from 0 to 2*pi)
      REAL*8 :: SINANG=0.,SN2ANG=0.,SN3ANG=0.,SN4ANG=0.,
     *          COSANG=0.,CS2ANG=0.,CS3ANG=0.,CS4ANG=0.

      call modelEclock%get(year=jyear,dayOfYear=jday)

      call getDomainBounds(grid,j_strt=j_0,j_stop=j_1)
      i_0 = grid%i_strt
      i_1 = grid%i_stop

      if(oht_is_timestream) then
        ! year is ignored for climatological input
        call read_stream(grid,OHTstream,jyear,jday,ohtconv_now)
      elseif(daysPerYear > 0.) then
C**** Calculate sines and cosines of the time of year for
C**** obtaining OHT convergence from arrays OT[ABC]
      ! nb: runs in which daysPerYear is not the period of
      ! the OHT cycle will probably be reading OHT in a
      ! different style/format
        ANGLE=TWOPI*JDAY/daysPerYear
        SINANG=SIN(ANGLE)
        SN2ANG=SIN(2*ANGLE)
        SN3ANG=SIN(3*ANGLE)
        SN4ANG=SIN(4*ANGLE)
        COSANG=COS(ANGLE)
        CS2ANG=COS(2*ANGLE)
        CS3ANG=COS(3*ANGLE)
        CS4ANG=COS(4*ANGLE)
        do j=j_0,j_1
        do i=i_0,atmocn%imaxj(j)
          if(atmocn%focean(i,j).gt.0) then
            ohtconv_now(i,j) =
     &           (OTA(I,J,4)*SN4ANG+OTB(I,J,4)*CS4ANG
     &           +OTA(I,J,3)*SN3ANG+OTB(I,J,3)*CS3ANG
     &           +OTA(I,J,2)*SN2ANG+OTB(I,J,2)*CS2ANG
     &           +OTA(I,J,1)*SINANG+OTB(I,J,1)*COSANG+OTC(I,J))
          endif
        enddo
        enddo
      endif

      IF(end_of_day) THEN
C**** Only do this at end of the day
        DO J=J_0,J_1
        DO I=I_0,atmocn%IMAXJ(J)
          AIJ(I,J,IJ_TOC2)=AIJ(I,J,IJ_TOC2)+TOCEAN(2,I,J)
          AIJ(I,J,IJ_TGO2)=AIJ(I,J,IJ_TGO2)+TOCEAN(3,I,J)
        END DO
        END DO
C**** DO DEEP DIFFUSION IF REQUIRED
        CALL ODIFS
C**** RESTRUCTURE THE OCEAN LAYERS
        CALL OSTRUC(.TRUE.,atmocn,atmice)
      ENDIF

c**** set ml heat capacity array
      do j=j_0,j_1
      do i=i_0,atmocn%imaxj(j)
        if (atmocn%focean(i,j).gt.0) then
          atmocn%mlhc(i,j) = shw*(z1o(i,j)*rhows-atmice%fwsim(i,j))
        end if
      end do
      end do

      end subroutine daily_ocnml

      SUBROUTINE init_OCNML(iniOCEAN)
!@sum init_OCEAN performs misc ML initialization tasks
!@auth Original Development Team
!@ver  1.0
      USE DIAG_COM, only : npts,icon_OCE,conpt0
      IMPLICIT NONE
      LOGICAL :: QCON(NPTS), T=.TRUE. , F=.FALSE.
      LOGICAL, INTENT(IN) :: iniOCEAN  ! true if starting from ic.
c
      CHARACTER CONPT(NPTS)*10

C****   set conservation diagnostic for ocean heat
      CONPT=CONPT0
      QCON=(/ F, F, F, T, F, T, F, T, T, F, F/)
      CALL SET_CON(QCON,CONPT,"OCN HEAT","(10^6 J/M**2)   ",
     *     "(W/M^2)         ",1d-6,1d0,icon_OCE)

C****   initialise deep ocean arrays if required
      call init_ODEEP(iniOCEAN)

      RETURN
      END SUBROUTINE init_OCNML

      subroutine init_qfluxes
!@sum init_qfluxes reads entire OHT file if fourier format, otherwise
!@+   only initializes the reading
      USE FILEMANAGER
      USE Dictionary_mod
      USE DOMAIN_DECOMP_ATM, only : GRID, GETDOMAINBOUNDS, am_I_root
#ifndef CUBED_SPHERE
      USE DOMAIN_DECOMP_ATM, only : UNPACK_DATA, BROADCAST
#endif
      USE RESOLUTION, only : im,jm
      use pario, only : par_open, par_close, read_dist_data, read_data
      use pario, only : variable_exists
      use model_com, only :  modelEclock
      use timestream_mod, only : init_stream
      IMPLICIT NONE
!@var iu_OHT unit number for reading OHT
      INTEGER :: iu_OHT
      integer :: fid
      real*8, allocatable ::
     &     OTA_glob(:,:,:),OTB_glob(:,:,:),OTC_glob(:,:)
      real*8  :: z12o_max_param, z12o_max_oht
      integer :: jyear,jday

      call modelEclock%get(year=jyear, dayOfYear=jday)

      call sync_param( "qfluxX"   ,qfluxX)
      call sync_param( "old_oht"  ,old_oht)
      if(old_oht==1) oht_is_timestream = .false.

      if(oht_is_timestream) then
!****   if z12o_max is part of OHT, get it
        fid = par_open(grid,'OHT','read')
        if(variable_exists(grid,fid,'z12o_max')) then
          call read_data(grid,fid,'z12o_max',z12o_max_oht,
     *            bcast_all=.true.)
          call par_close(grid,fid)
        else
          z12o_max_oht = -1
        end if
!****   if z12o_max is also part of the rundeck, check for consistency
        if (is_set_param("z12o_max")) then
          call get_param("z12o_max", z12o_max_param)
          if (z12o_max_oht>0.and.z12o_max_oht.ne.z12o_max_param) then
            call stop_model('z12o_max inconsistent with OHT',255)
          endif
          z12o_max_oht = z12o_max_param
        endif
        call set_param("z12o_max", z12o_max_oht, 'o')
        z12o_max = z12o_max_oht
        if(z12o_max < 0) call stop_model('z12o_max not set',255)

        call init_stream(grid,OHTstream,'OHT','ohtconv',-1d30,1d30,
     &       'ppm',jyear,jday)

      else  ! read fourier coefficients and z12o_max
        if(is_fbsa('OHT')) then ! assuming GISS format
        !if(.true.) then         ! assuming GISS format
#ifdef CUBED_SPHERE
          call stop_model('use netcdf for OHT',255)
#else
          call openunit('OHT',iu_OHT,.true.,.true.)
          if(am_i_root()) then
            allocate(OTA_glob(im,jm,4))
            allocate(OTB_glob(im,jm,4))
            allocate(OTC_glob(im,jm))
            read (iu_OHT) OTA_glob,OTB_glob,OTC_glob,z12o_max
            write(6,*) "Read ocean heat transports from OHT"
          endif
          call unpack_data(grid,OTA_glob,OTA)
          call unpack_data(grid,OTB_glob,OTB)
          call unpack_data(grid,OTC_glob,OTC)
          call broadcast(grid,z12o_max)
          if(am_i_root()) then
            deallocate(OTA_glob,OTB_glob,OTC_glob)
          endif
          call closeunit (iu_OHT)
#endif
        else   ! assuming netcdf format
          fid = par_open(grid,'OHT','read')
          call read_dist_data(grid,fid,'ota',OTA)
          call read_dist_data(grid,fid,'otb',OTB)
          call read_dist_data(grid,fid,'otc',OTC)
          call read_data(grid,fid,'z12o_max',z12o_max,bcast_all=.true.)
          call par_close(grid,fid)
        endif
        call set_param("z12o_max", z12o_max, 'o')
      endif

      RETURN
      END SUBROUTINE init_qfluxes

      subroutine init_mldepth(atmocn)
!@sum init_mldepth open the ML depth file
      USE DOMAIN_DECOMP_ATM, only : GRID, GETDOMAINBOUNDS, am_I_root
      use TimeConstants_mod, only: INT_MONTHS_PER_YEAR
      use timestream_mod, only : init_stream,get_by_index
      use exchange_types, only : atmocn_xchng_vars
      use model_com, only :  modelEclock, kocean
      use dictionary_mod, only : get_param, is_set_param
      use Rational_mod, only : Real
      use BaseTime_mod, only: BaseTime
      implicit none
      type(atmocn_xchng_vars) :: atmocn
c
      INTEGER :: I,J,m
      real*8 z1ox(grid%i_strt_halo:grid%i_stop_halo,
     &            grid%j_strt_halo:grid%j_stop_halo)
      integer :: i_0,i_1, j_0,j_1
      integer :: fid
      real*8, allocatable ::
     &     OTA_glob(:,:,:),OTB_glob(:,:,:),OTC_glob(:,:)
      integer :: jyear,jday
      real*8 seconds
      type (BaseTime) :: t

      call modelEclock%get(year=jyear, dayOfYear=jday)

      call getDomainBounds(grid,j_strt=j_0,j_stop=j_1)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

C**** Read mixed layer depth climatology
      call init_stream(grid,Z1Ostream,'OCNML','z1o',0d0,1000d0,'linm2m',
     &     jyear,jday,msk=atmocn%focean)

      if(z12o_max < 0) then
        if(is_set_param('z12o_max')) then
          call get_param('z12o_max',z12o_max)
        else
          call stop_model('z12o_max not set',255)
        end if
      end if

      if(do_ohtconv) then
        t = modelEclock%getTimeInSecondsFromDate(year1_ohtconv,1,1,0)
        seconds = Real(t)
        if(seconds .gt. 0 .and. z12o_max .ne. z12o_max_rsf) then
          write(*,*) 'z12o_max was changed from',z12o_max_rsf,' to ',
     *     z12o_max
          call stop_model('z12o_max was changed',255)
        endif
      endif

C****   find and limit ocean ann max mix layer depths
      z12o = 0.
      do m=1,INT_MONTHS_PER_YEAR
        call get_by_index(grid,Z1Ostream,m,z1ox)
        do j=j_0,j_1
          do i=i_0,i_1
            z12o(i,j)=min( z12o_max , max(z12o(i,j),z1ox(i,j)) )
          end do
        end do
      end do
      IF (AM_I_ROOT())
     *     write(6,*)'Mixed Layer Depths limited to',z12o_max

      return
      end subroutine init_mldepth

      SUBROUTINE OSTRUC(QTCHNG,atmocn,atmice)
!@sum  OSTRUC restructures the ocean temperature profile as ML
!@sum         depths are changed (generally once a day)
!@auth Original Development Team
!@ver  1.0 (Q-flux ocean)
      USE DOMAIN_DECOMP_ATM, only : GRID,getdomainbounds
      use exchange_types, only : atmocn_xchng_vars,atmice_xchng_vars
      use constant, only : rhows
      IMPLICIT NONE
!@var QTCHNG true if TOCEAN(1) is changed (needed for qflux calculation)
      LOGICAL, INTENT(IN) :: QTCHNG
      type(atmocn_xchng_vars) :: atmocn
      type(atmice_xchng_vars) :: atmice
!
      INTEGER I,J
      REAL*8 :: DT2,T2ENT,T2WT,DWTRO,WTR1O,WTR2O,WTR2O_old
      INTEGER :: J_0, J_1, I_0, I_1
!****
!**** FLAND     LAND COVERAGE (1)
!****
!**** TOCEAN 1  OCEAN TEMPERATURE OF FIRST LAYER (C)
!****        2  MEAN OCEAN TEMPERATURE OF SECOND LAYER (C)
!****        3  OCEAN TEMPERATURE AT BOTTOM OF SECOND LAYER (C)
!****     RSI   RATIO OF OCEAN ICE COVERAGE TO WATER COVERAGE (1)
!****
!**** FWSIM     Fresh water sea ice amount (KG/M**2)
!****
!**** RESTRUCTURE OCEAN LAYERS
!****
      CALL GETdomainbounds(GRID,J_STRT=J_0,J_STOP=J_1)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

      DO J=J_0,J_1
      DO I=I_0,atmocn%IMAXJ(J)
        IF (atmocn%FOCEAN(I,J).LE.0.) CYCLE
        WTR2O_old = RHOWS*(Z12O(I,J)-Z1OOLD(I,J))
        WTR2O     = RHOWS*(Z12O(I,J)-Z1O(I,J))
        if(wtr2o_old .gt. 0.) then
          DWTRO=RHOWS*(Z1O(I,J)-Z1OOLD(I,J))
          IF(DWTRO.LE.0.) then  ! MIXED LAYER IS SHALLOWING
            DT2 = (TOCEAN(2,I,J)-TOCEAN(1,I,J))*DWTRO/WTR2O
          else                  ! MIXED LAYER IS DEEPENING
            DT2 = (TOCEAN(3,I,J)-TOCEAN(2,I,J))*DWTRO/wtr2o_old
            IF(QTCHNG) THEN     ! change TOCEAN(1) by entrainment of layer 2
              ! wtr2o/wtr2o_old is less than or equal to 1
              t2ent = tocean(2,i,j) +
     &             (tocean(2,i,j)-tocean(3,i,j))*(wtr2o/wtr2o_old)
              wtr1o = rhows*z1o(i,j)-atmice%fwsim(i,j)
              t2wt = dwtro/wtr1o
              tocean(1,i,j) = tocean(1,i,j)*(1d0-t2wt) + t2ent*t2wt
            ENDIF
          endif
          TOCEAN(2,I,J) = TOCEAN(2,I,J) + DT2
        endif

        if(min(wtr2o_old,wtr2o) .le. 0.) then
!**** MIXED LAYER DEPTH IS AT ITS MAXIMUM OR TEMP PROFILE IS UNIFORM
          TOCEAN(2,I,J)=TOCEAN(1,I,J)
          TOCEAN(3,I,J)=TOCEAN(1,I,J)
        endif
      END DO
      END DO
      RETURN
      END SUBROUTINE OSTRUC

      SUBROUTINE OSOURC (I,J
     *     ,ROICE,SMSI,TGW,WTRO,OTDT,RUN0,FODT,FIDT,RVRRUN
     *     ,RVRERUN,EVAPO,EVAPI,TFW,WTRW,RUN4O,ERUN4O
     *     ,RUN4I,ERUN4I,ENRGFO,ACEFO,ACEFI,ENRGFI)
!@sum  OSOURC applies fluxes to ocean in ice-covered and ice-free areas
!@+    ACEFO/I are the freshwater ice amounts,
!@+    ENRGFO/I is total energy (including a salt component)
!@auth Gary Russell
!@ver  1.0
      USE CONSTANT, only : shw,rhows
      USE SEAICE, only : ssi0,ei
      USE MODEL_COM, only : DTSRC
      use diag_com, only : aij=>aij_loc,ij_geotherm
      IMPLICIT NONE
!@var I,J gridpoint ID
      INTEGER, INTENT(IN) :: I,J
!@var TFW freezing temperature for water underlying ice (C)
      REAL*8, INTENT(IN) :: TFW

!@var TGW mixed layer temp.(C)
      REAL*8, INTENT(INOUT) :: TGW
      REAL*8, INTENT(IN) :: ROICE, SMSI, EVAPO, EVAPI, RUN0
      REAL*8, INTENT(IN) :: FODT, FIDT, RVRRUN, RVRERUN
      REAL*8, INTENT(OUT) :: ENRGFO, ACEFO, ENRGFI, ACEFI, RUN4O, RUN4I
     *     , ERUN4O, ERUN4I, WTRW, WTRO, OTDT
      REAL*8 EIW0, ENRGIW, WTRI1, EFIW, ENRGO0, EOFRZ, ENRGO
      REAL*8 WTRW0, ENRGW0, ENRGW, WTRI0, SMSI0
!@var emin min energy deficit required before ice forms (J/m^2)
      REAL*8, PARAMETER :: emin=-1d-10
C**** initiallize output
      ENRGFO=0. ; ACEFO=0. ; ACEFI=0. ; ENRGFI=0. ; WTRW=WTRO

C**** Calculate ML mass and heat convergence for this gridpoint
      WTRO=Z1O(I,J)*RHOWS
      OTDT=qfluxX*DTSRC*OHTCONV_NOW(I,J)
!**** add geothermal heat flux
      OTDT = OTDT + dtsrc*fgeotherm(i,j)

C**** Calculate previous ice mass (before fluxes applied)
      SMSI0=SMSI+RUN0+EVAPI

C**** Calculate extra mass flux to ocean, balanced by deep removal
      RUN4O=-EVAPO+RVRRUN  ! open ocean
      RUN4I=-EVAPI+RVRRUN  ! under ice
! force energy conservation
      ERUN4O=0. ! RUN4O*TGW*SHW ! corresponding heat flux at bottom (open)
      ERUN4I=0. ! RUN4I*TGW*SHW !                              (under ice)

C**** Calculate heat fluxes to ocean
      ENRGO  = FODT+OTDT+RVRERUN-ERUN4O ! in open water
      ENRGIW = FIDT+OTDT+RVRERUN-ERUN4I ! under ice

C**** Calculate energy in mixed layer (open ocean)
      ENRGO0=WTRO*TGW*SHW
      EOFRZ=WTRO*TFW*SHW
      IF (ENRGO0+ENRGO.GE.EOFRZ+emin) THEN
C**** FLUXES RECOMPUTE TGW WHICH IS ABOVE FREEZING POINT FOR OCEAN
        IF (ROICE.LE.0.) THEN
          TGW=TGW+ENRGO/(WTRO*SHW)
          RETURN
        END IF
      ELSE
C**** FLUXES COOL TGO TO FREEZING POINT FOR OCEAN AND FORM SOME ICE
C**** Solve TFW=(Eo-Eice)/(W-F)/SHW, w. mass of ice = F/(1-SSI0)
C**** and Eice=Ei(TFW,1d3*SSI0)*F/(1-SSI0)
C**** Conserve FW mass and energy (not salt)
        ACEFO=(ENRGO0+ENRGO-EOFRZ)/(Ei(TFW,1d3*SSI0)/(1.-SSI0)-TFW*SHW) ! fresh
        ENRGFO = ACEFO*Ei(TFW,1d3*SSI0)/(1.-SSI0) ! energy of frozen ice
        IF (ROICE.LE.0.) THEN
          TGW=TFW
          RETURN
        END IF
      END IF

         aij(i,j,ij_geotherm) = aij(i,j,ij_geotherm) + fgeotherm(i,j)
C****
      IF (ROICE.le.0) RETURN

C**** Calculate initial energy of mixed layer (before fluxes applied)
      WTRI0=WTRO-SMSI0       ! mixed layer mass below ice (kg/m^2)
      EIW0=WTRI0*TGW*SHW     ! energy of mixed layer below ice (J/m^2)

      WTRW0=WTRO-ROICE*SMSI0 ! initial total mixed layer mass
      ENRGW0=WTRW0*TGW*SHW   ! initial energy in mixed layer

C**** CALCULATE THE ENERGY OF THE WATER BELOW THE ICE AT FREEZING POINT
C**** AND CHECK WHETHER NEW ICE MUST BE FORMED
      WTRI1 = WTRO-SMSI         ! new mass of ocean (kg/m^2)
      EFIW = WTRI1*TFW*SHW ! freezing energy of ocean mass WTRI1
      IF (EIW0+ENRGIW .LE. EFIW+emin) THEN ! freezing case
C**** FLUXES WOULD COOL TGW TO FREEZING POINT AND FREEZE SOME MORE ICE
        ACEFI=(EIW0+ENRGIW-EFIW)/(Ei(TFW,1d3*SSI0)/(1.-SSI0)-TFW*SHW)  ! fresh
        ENRGFI = ACEFI*Ei(TFW,1d3*SSI0)/(1.-SSI0) ! energy of frozen ice
      END IF
C**** COMBINE OPEN OCEAN AND SEA ICE FRACTIONS TO FORM NEW VARIABLES
      WTRW = WTRO  -(1.-ROICE)* ACEFO        -ROICE*(SMSI+ACEFI)
      ENRGW= ENRGW0+(1.-ROICE)*(ENRGO-ENRGFO)+ROICE*(ENRGIW-ENRGFI)
      TGW  = ENRGW/(WTRW*SHW) ! new ocean temperature

      RETURN
      END SUBROUTINE OSOURC

      SUBROUTINE PRECIP_OCNML(atmocn,atmice,iceocn)
!@sum  PRECIP_OC driver for applying precipitation to ocean fraction
!@auth Original Development Team
!@ver  1.0
      USE CONSTANT, only : rhows,shw
      USE MODEL_COM, only : kocean
      USE DIAG_COM, only : itocean,itoice
      USE DIAG_COM, only : j_implm,j_implh,jreg
      USE DOMAIN_DECOMP_ATM, only : GRID,GETDOMAINBOUNDS,AM_I_ROOT
      USE EXCHANGE_TYPES, only : atmocn_xchng_vars,atmice_xchng_vars
     &     ,iceocn_xchng_vars
      USE SEAICE_COM, only : si_ocn
      IMPLICIT NONE
      type(atmocn_xchng_vars) :: atmocn
      type(atmice_xchng_vars) :: atmice
      type(iceocn_xchng_vars) :: iceocn
      REAL*8 TGW,PRCP,WTRO,ENRGP,ERUN4,POCEAN,POICE
     *     ,SMSI0,ENRGW,WTRW0,WTRW,RUN0,RUN4,ROICE,SIMELT,ESIMELT
      INTEGER I,J,JR
      INTEGER :: J_0,J_1, I_0,I_1
      REAL*8 :: FOCEAN

      REAL*8, DIMENSION(:,:), POINTER :: RSI,PREC,EPREC,
     &     RUNPSI,ERUNPSI,SRUNPSI,MELTI,EMELTI,SMELTI,
     &     FWSIM

      CALL GETDOMAINBOUNDS(GRID,J_STRT=J_0,J_STOP=J_1)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

      FWSIM => ATMICE%FWSIM
      RSI => SI_OCN%RSI
      PREC => ATMOCN%PREC
      EPREC => ATMOCN%EPREC
      RUNPSI => ICEOCN%RUNPSI
      ERUNPSI => ICEOCN%ERUNPSI
      SRUNPSI => ICEOCN%SRUNPSI
      MELTI => ICEOCN%MELTI
      EMELTI => ICEOCN%EMELTI
      SMELTI => ICEOCN%SMELTI

      DO J=J_0,J_1
      DO I=I_0,atmocn%IMAXJ(J)
        FOCEAN = ATMOCN%FOCEAN(I,J)
        ROICE=RSI(I,J)
        POICE=FOCEAN*RSI(I,J)
        POCEAN=FOCEAN*(1.-RSI(I,J))
        JR=JREG(I,J)
        IF (FOCEAN.gt.0) THEN
          PRCP=PREC(I,J)
          ENRGP=EPREC(I,J)
          RUN0=RUNPSI(I,J)-SRUNPSI(I,J) ! fresh water runoff
          SIMELT =(MELTI(I,J)-SMELTI(I,J))/FOCEAN
          ESIMELT=EMELTI(I,J)/FOCEAN

          TGW=TOCEAN(1,I,J)
          WTRO=Z1O(I,J)*RHOWS
          SMSI0=FWSIM(I,J)+ROICE*(RUN0-PRCP) ! initial ice

C**** Calculate effect of lateral melt of sea ice
          IF (SIMELT.gt.0) THEN
            TGW=TGW + (ESIMELT-SIMELT*TGW*SHW)/
     *           (SHW*(WTRO-SMSI0))
          END IF

C**** Additional mass (precip) is balanced by deep removal
          RUN4=PRCP
          ERUN4=0.              ! RUN4*TGW*SHW ! force energy conservation

          IF (POICE.LE.0.) THEN
            ENRGW=TGW*WTRO*SHW + ENRGP - ERUN4
            WTRW =WTRO
          ELSE
            WTRW0=WTRO -SMSI0
            ENRGW=WTRW0*TGW*SHW + (1.-ROICE)*ENRGP - ERUN4
            WTRW =WTRW0+ROICE*(RUN0-RUN4)
          END IF
          TGW=ENRGW/(WTRW*SHW)
          TOCEAN(1,I,J)=TGW
          CALL INC_AJ(I,J,ITOCEAN,J_IMPLM, RUN4*POCEAN)
          CALL INC_AJ(I,J,ITOCEAN,J_IMPLH,ERUN4*POCEAN)
          CALL INC_AJ(I,J,ITOICE ,J_IMPLM, RUN4*POICE)
          CALL INC_AJ(I,J,ITOICE ,J_IMPLH,ERUN4*POICE)
          CALL INC_AREG(I,J,JR,J_IMPLM, RUN4*FOCEAN)
          CALL INC_AREG(I,J,JR,J_IMPLH,ERUN4*FOCEAN)
          atmocn%MLHC(I,J)=WTRW*SHW    ! needed for underice fluxes
        END IF
      END DO
      END DO

      RETURN
C****
      END SUBROUTINE PRECIP_OCNML

      SUBROUTINE RUN_OCNML(atmocn,atmice,iceocn)
!@sum  OCEANS driver for applying surface fluxes to ocean fraction
!@auth Original Development Team
!@ver  1.0
!@calls OCNML:OSOURC
      USE CONSTANT, only : rhows,shw,tf
      USE MODEL_COM, only : kocean,dtsrc
      USE DIAG_COM, only : itocean,itoice
      USE DIAG_COM, only : jreg,j_implm,j_implh,j_oht
      USE SEAICE, only : ssi0,tfrez
      USE SEAICE_COM, only : si_ocn
#ifdef TRACERS_WATER
      use OldTracer_mod, only: trsi0
#endif
      USE DOMAIN_DECOMP_ATM, only : GRID,GETDOMAINBOUNDS,AM_I_ROOT
      USE EXCHANGE_TYPES, only : atmocn_xchng_vars,atmice_xchng_vars
     &     ,iceocn_xchng_vars
      IMPLICIT NONE
      type(atmocn_xchng_vars) :: atmocn
      type(atmice_xchng_vars) :: atmice
      type(iceocn_xchng_vars) :: iceocn
C**** grid box variables
      REAL*8 POCEAN, POICE, TFO
C**** prognostic variables
      REAL*8 TGW, WTRO, SMSI, ROICE
C**** fluxes
      REAL*8 EVAPO, EVAPI, FIDT, FODT, OTDT, RVRRUN, RVRERUN, RUN0
C**** output from OSOURC
      REAL*8 ERUN4I, ERUN4O, RUN4I, RUN4O, ENRGFO, ACEFO, ACEFI, ENRGFI,
     *     WTRW
      REAL*8 FOCEAN
      INTEGER I,J,JR
      INTEGER :: J_0,J_1, I_0,I_1

      REAL*8, DIMENSION(:,:), POINTER :: RSI,MSI,SNOWI,
     &     FLOWO,EFLOWO,GMELT,EGMELT,E0,SSS,GTEMP,GTEMP2,GTEMPR,
     &     RUNOSI,ERUNOSI,SRUNOSI,FWSIM
      REAL*8, DIMENSION(:,:,:), POINTER :: DMSI,DHSI,DSSI
#ifdef TRACERS_WATER
      REAL*8, DIMENSION(:,:,:,:), POINTER :: DTRSI
#endif

      FWSIM => ATMICE%FWSIM
      RSI => SI_OCN%RSI
      MSI => SI_OCN%MSI
      SNOWI => SI_OCN%SNOWI
      FLOWO => ATMOCN%FLOWO
      EFLOWO => ATMOCN%EFLOWO
      GMELT => ATMOCN%GMELT
      EGMELT => ATMOCN%EGMELT
      E0 => ATMOCN%E0
      SSS => ATMOCN%SSS
      GTEMP => ATMOCN%GTEMP
      GTEMP2 => ATMOCN%GTEMP2
      GTEMPR => ATMOCN%GTEMPR
      RUNOSI => ICEOCN%RUNOSI
      ERUNOSI => ICEOCN%ERUNOSI
      SRUNOSI => ICEOCN%SRUNOSI
      DMSI => ICEOCN%DMSI
      DHSI => ICEOCN%DHSI
      DSSI => ICEOCN%DSSI
#ifdef TRACERS_WATER
      DTRSI => ICEOCN%DTRSI
#endif

      CALL GETDOMAINBOUNDS(GRID,J_STRT=J_0,J_STOP=J_1)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

      DO J=J_0,J_1
      DO I=I_0,atmocn%IMAXJ(J)
        FOCEAN=ATMOCN%FOCEAN(I,J)
        JR=JREG(I,J)
        ROICE=RSI(I,J)
        POICE=FOCEAN*RSI(I,J)
        POCEAN=FOCEAN*(1.-RSI(I,J))
        IF (FOCEAN.gt.0) THEN
          TGW  =TOCEAN(1,I,J)
          EVAPO=atmocn%EVAPOR(I,J)
          EVAPI=atmice%EVAPOR(I,J)   ! evapor/dew at the ice surface (kg/m^2)
          FODT =E0(I,J)
          SMSI = 0.
          IF (ROICE.gt.0) SMSI =FWSIM(I,J)/ROICE
          TFO  =tfrez(sss(i,j))
C**** get ice-ocean fluxes from sea ice routine (no salt)
          RUN0=RUNOSI(I,J)-SRUNOSI(I,J) ! fw, includes runoff+basal
          FIDT=ERUNOSI(I,J)
c          SALT=SRUNOSI(I,J)
C**** get river runoff/iceberg melt flux
          RVRRUN = FLOWO(I,J)+ GMELT(I,J)
          RVRERUN=EFLOWO(I,J)+EGMELT(I,J)

C**** Calculate the amount of ice formation
          CALL OSOURC (I,J
     *           ,ROICE,SMSI,TGW,WTRO,OTDT,RUN0,FODT,FIDT,RVRRUN
     *           ,RVRERUN,EVAPO,EVAPI,TFO,WTRW,RUN4O,ERUN4O
     *           ,RUN4I,ERUN4I,ENRGFO,ACEFO,ACEFI,ENRGFI)

C**** Resave prognostic variables
          TOCEAN(1,I,J)=TGW
C**** Open Ocean diagnostics
          CALL INC_AJ(I,J,ITOCEAN,J_OHT  ,  OTDT*POCEAN)
          CALL INC_AJ(I,J,ITOCEAN,J_IMPLM, RUN4O*POCEAN)
          CALL INC_AJ(I,J,ITOCEAN,J_IMPLH,ERUN4O*POCEAN)
C**** Ice-covered ocean diagnostics
          CALL INC_AJ(I,J,ITOICE,J_OHT  ,  OTDT*POICE)
          CALL INC_AJ(I,J,ITOICE,J_IMPLM, RUN4I*POICE)
          CALL INC_AJ(I,J,ITOICE,J_IMPLH,ERUN4I*POICE)
C**** regional diagnostics
          CALL INC_AREG(I,J,JR,J_IMPLM,( RUN4O*POCEAN+ RUN4I*POICE))
          CALL INC_AREG(I,J,JR,J_IMPLH,(ERUN4O*POCEAN+ERUN4I*POICE))
          atmocn%MLHC(I,J)=SHW*WTRW

C**** Store mass and energy fluxes for formation of sea ice
C**** Note ACEFO/I is the freshwater amount of ice.
C**** Add salt for total mass
          DMSI(1,I,J)=ACEFO/(1.-SSI0)
          DMSI(2,I,J)=ACEFI/(1.-SSI0)
          DHSI(1,I,J)=ENRGFO
          DHSI(2,I,J)=ENRGFI
          DSSI(1,I,J)=SSI0*DMSI(1,I,J)
          DSSI(2,I,J)=SSI0*DMSI(2,I,J)
#ifdef TRACERS_WATER
C**** assume const mean tracer conc over freshwater amount
          DTRSI(:,1,I,J)=TRSI0()*ACEFO
          DTRSI(:,2,I,J)=TRSI0()*ACEFI
#endif

        END IF
      END DO
      END DO

      RETURN
C****
      END SUBROUTINE RUN_OCNML

      subroutine set_gtemp_ocnml(atmocn)
!@sum set_gtemp_sst copies sst into the ocean position in the gtemp array
      use constant, only : tf
      use domain_decomp_atm, only : grid,getDomainBounds
#ifdef SCM
      USE SCM_COM, only : SCMopt,SCMin
#endif
      USE EXCHANGE_TYPES, only : atmocn_xchng_vars,iceocn_xchng_vars
      implicit none
      type(atmocn_xchng_vars) :: atmocn
c
      integer :: i,j,j_0,j_1, i_0,i_1
      call getDomainBounds(grid,j_strt=j_0,j_stop=j_1)
      i_0 = grid%i_strt
      i_1 = grid%i_stop
      do j=j_0,j_1
      do i=i_0,atmocn%imaxj(j)
        if (atmocn%focean(i,j).gt.0) then
          atmocn%gtemp(i,j)=tocean(1,i,j)
          atmocn%gtemp2(i,j)=tocean(2,i,j)
          atmocn%gtempr(i,j) =tocean(1,i,j)+tf
#ifdef SCM
c         may specify ocean temperature for SCM
          if( SCMopt%Tskin )then
            atmocn%gtemp(I,J) = SCMin%Tskin - TF
            atmocn%gtempr(I,J) = SCMin%Tskin
          endif
#endif
        endif
      enddo
      enddo
      end subroutine set_gtemp_ocnml

      end module ocnml

      subroutine advsi_diag(atmocn,atmice)
!@sum advsi_diag calls sea ice diagnostic routine for kocean=1 case
!@+   and accumulates hsicnv for do_ohtconv=true
      use model_com, only : kocean,modelEclock
      use domain_decomp_atm, only : grid,getDomainBounds
      use ocnml, only : z1o,z12o
      use ocnml, only : do_ohtconv,ohtconv
      use exchange_types, only : atmocn_xchng_vars,atmice_xchng_vars
      implicit none
      type(atmocn_xchng_vars) :: atmocn
      type(atmice_xchng_vars) :: atmice
!
      integer i,j, j_0,j_1, i_0,i_1
      integer :: month
!
      if(kocean.ge.1) then
        call advsi_diag_ocnml(z1o,z12o,atmocn,atmice)
      elseif(do_ohtconv) then
        call modelEclock%get(month=month)
        call getDomainBounds(grid,i_strt=i_0,i_stop=i_1)
        call getDomainBounds(grid,j_strt=j_0,j_stop=j_1)
        do j=j_0,j_1
        do i=i_0,atmocn%imaxj(j)
          if(atmocn%focean(i,j).gt.0.) then
            ohtconv(i,j,month) = ohtconv(i,j,month)
     &           -  atmice%hsicnv(i,j)
          endif
        enddo
        enddo
      endif
      end subroutine advsi_diag


      SUBROUTINE CHECKO(SUBR)
!@sum  CHECKO Checks whether Ocean are reasonable
!@auth Original Development Team
      USE RESOLUTION, only : im,jm
      USE OCNML, only : tocean!,focean
      use fluxes, only : atmocn
      USE DOMAIN_DECOMP_ATM, only : GRID
      USE DOMAIN_DECOMP_1D, only : getDomainBounds
      IMPLICIT NONE

!@var SUBR identifies where CHECK was called from
      CHARACTER*6, INTENT(IN) :: SUBR
      LOGICAL QCHECKO
      INTEGER I,J
      integer :: J_0, J_1, J_0H, J_1H, I_0, I_1, I_0H, I_1H, njpol
C****
C**** Extrack useful local domain parameters from "grid"
C****
      call getDomainBounds(grid,
     *     J_STRT=J_0, J_STOP=J_1, J_STRT_HALO=J_0H, J_STOP_HALO=J_1H)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP
      I_0H = grid%I_STRT_HALO
      I_1H = grid%I_STOP_HALO
      njpol = grid%J_STRT_SKP-grid%J_STRT

C**** Check for NaN/INF in ocean data
      CALL CHECK3C(TOCEAN(:,I_0:I_1,J_0:J_1),3,I_0,I_1,J_0,J_1,NJPOL,
     &     SUBR,'toc')

      QCHECKO = .FALSE.
C**** Check for reasonable values for ocean variables
      DO J=J_0, J_1
        DO I=I_0, I_1
          IF (atmocn%FOCEAN(I,J).gt.0 .and. (TOCEAN(1,I,J).lt.-2. .or.
     *         TOCEAN(1,I,J).gt.50.)) THEN
            WRITE(6,*) 'After ',SUBR,': I,J,TOCEAN=',I,J,TOCEAN(1:3,I,J)
            QCHECKO = .TRUE.
          END IF
       END DO
      END DO
      IF (QCHECKO)
     *     call stop_model("CHECKO: Ocean variables out of bounds",255)

      END SUBROUTINE CHECKO

      subroutine def_rsf_ocnml(fid)
!@sum  def_rsf_ocean defines ocean array structure in restart files
!@auth M. Kelley
!@ver  beta
      use ocnml
      use domain_decomp_atm, only : grid
      use pario, only : defvar
      implicit none
      integer fid   !@var fid file id
      call defvar(grid,fid,tocean,'tocean(d3,dist_im,dist_jm)')
      call defvar(grid,fid,z1o,'z1o(dist_im,dist_jm)')
      return
      end subroutine def_rsf_ocnml

      subroutine new_io_ocnml(fid,iaction)
!@sum  new_io_ocean read/write ocean arrays from/to restart files
!@auth M. Kelley
!@ver  beta new_ prefix avoids name clash with the default version
      use model_com, only : ioread,iowrite
      use ocnml
      use domain_decomp_atm, only : grid
      use pario, only : write_dist_data,read_dist_data
      implicit none
      integer fid   !@var fid unit number of read/write
      integer iaction !@var iaction flag for reading or writing to file
      select case (iaction)
      case (iowrite)            ! output to restart file
        call write_dist_data(grid, fid, 'tocean', tocean, jdim=3)
        call write_dist_data(grid, fid, 'z1o', z1o)
      case (ioread)            ! input from restart file
        call read_dist_data(grid, fid, 'tocean', tocean, jdim=3)
        call read_dist_data(grid, fid, 'z1o', z1o)
      end select
      return
      end subroutine new_io_ocnml

      subroutine def_rsf_ohtconv(fid)
!@sum  def_rsf_ocean defines ohtconv array structure in restart files
!@auth M. Kelley
      use ocnml
      use domain_decomp_atm, only : grid
      use pario, only : defvar
      implicit none
      integer fid   !@var fid file id
      call defvar(grid,fid,z1o,'z1o(dist_im,dist_jm)')
      call defvar(grid,fid,acctime_ohtconv,'acctime_ohtconv(twelve)')
      call defvar(grid,fid,ohtconv,'ohtconv(dist_im,dist_jm,twelve)')
      call defvar(grid,fid,ocnht_sv,'ocnht_sv(dist_im,dist_jm)')
      call defvar(grid,fid,z12o_max,'z12o_max')
      return
      end subroutine def_rsf_ohtconv

      subroutine new_io_ohtconv(fid,iaction)
!@sum  new_io_ocean read/write ohtconv arrays from/to restart files
!@auth M. Kelley
      use model_com, only : ioread,iowrite
      use ocnml
      use domain_decomp_atm, only : grid
      use pario, only : write_dist_data,read_dist_data
      use pario, only : write_data,read_data
      implicit none
      integer fid   !@var fid unit number of read/write
      integer iaction !@var iaction flag for reading or writing to file
      select case (iaction)
      case (iowrite)            ! output to restart file
        call write_dist_data(grid, fid, 'z1o', z1o)
        call write_data(grid,fid,'acctime_ohtconv',acctime_ohtconv)
        call write_dist_data(grid,fid,'ohtconv',ohtconv)
        call write_dist_data(grid,fid,'ocnht_sv',ocnht_sv)
        call write_data(grid,fid,'z12o_max',z12o_max_rsf)
      case (ioread)            ! input from restart file
        call read_dist_data(grid, fid, 'z1o', z1o)
        call read_data(grid,fid,'acctime_ohtconv',acctime_ohtconv,
     &       bcast_all=.true.)
        call read_dist_data(grid,fid,'ohtconv',ohtconv)
        call read_dist_data(grid,fid,'ocnht_sv',ocnht_sv)
        call read_data(grid,fid,'z12o_max',z12o_max_rsf,
     &       bcast_all=.true.)
      end select
      return
      end subroutine new_io_ohtconv

      subroutine reset_odiag(idum)
! perform various operations at the beginning of a month
      use model_com, only: modelEclock,iyear1
      use ocnml
      implicit none
      integer :: idum ! not used
!
      integer :: year,dayOfYear

      if(.not. do_ohtconv) return ! nothing to do

      call modelEclock%get(year=year,dayOfYear=dayOfYear)

      ! zero out ohtconv if this is the first day
      ! of the first year of its accmulation period.
      if(dayOfYear==1 .and. year==year1_ohtconv) then
        acctime_ohtconv(:) = 0.
        ohtconv(:,:,:) = 0.
        z12o_max_rsf = z12o_max
      endif

      end subroutine reset_odiag

      subroutine finishmonth_ohtconv
! perform various operations at the end of a month (and year).
      use model_com, only: modelEclock,itime,itimei,yeari,yeare,
     &                     xlabel,lrunid
      use ocnml
      use domain_decomp_atm, only : grid,getDomainBounds,globalsum
      use geom, only : axyp,imaxj
      use resolution, only : im,jm
      use pario, only : par_open,par_close,par_enddef,defvar,
     &     write_dist_data, write_data
      use fluxes, only : atmocn ! should really be passed as arg
      implicit none
!
      integer :: dayOfYear,month,year
      real*8 :: corr,ohtsum,oasum,sumpos,sumneg,multfac
      real*8, dimension(:,:), allocatable :: annsum,oarea,ocnht,iceht,
     &     posarr
      real*8, dimension(:,:,:), allocatable :: ohtconv_out
      integer i,j,m,fid, j_0,j_1, i_0,i_1
      logical :: have_north_pole, have_south_pole
      logical :: addadj
      character(len=4) :: ye, yi, MLDmax

      call modelEclock%get(month=month,dayOfYear=dayOfYear,year=year)
      ! These minus 1 time shifts are because of subtleties in
      ! the sequencing of operations in the main time loop.
      dayOfYear = dayOfYear-1; if(dayOfYear==0) dayOfYear=365
      month = month-1; if(month==0) month=12

      ! First, determine the current ML+seaice heat content
      allocate(ocnht(grid%i_strt_halo:grid%i_stop_halo,
     &               grid%j_strt_halo:grid%j_stop_halo))
      ocnht = 0.
      call conserv_oce(ocnht)
      allocate(iceht(grid%i_strt_halo:grid%i_stop_halo,
     &               grid%j_strt_halo:grid%j_stop_halo))
      iceht = 0.
      call conserv_ohsi(iceht)
      ocnht = ocnht + iceht

      ! increment ohtconv with the current minus the initial heat content
      ohtconv(:,:,month) = ohtconv(:,:,month) + (ocnht-ocnht_sv)

      ! save the current heat content
      ocnht_sv = ocnht


      if(dayOfYear==365 .and. year==yeare) then
        ! End-of-year operations performed on ohtconv_out which is written
        ! to a separate file OHT.nc:
        ! 1. scale ohtconv J/m2 -> W/m2
        ! 2. adjust the annual area mean to zero
        allocate(annsum(grid%i_strt_halo:grid%i_stop_halo,
     &                  grid%j_strt_halo:grid%j_stop_halo))
        allocate(oarea(grid%i_strt_halo:grid%i_stop_halo,
     &                 grid%j_strt_halo:grid%j_stop_halo))
        allocate(ohtconv_out(grid%i_strt_halo:grid%i_stop_halo,
     &                       grid%j_strt_halo:grid%j_stop_halo,12))
        ohtconv_out(:,:,:) = 0.
        do m=1,12
          if(acctime_ohtconv(m).eq.0.) cycle
          ohtconv_out(:,:,m) = ohtconv(:,:,m)/acctime_ohtconv(m)
        enddo
        call getDomainBounds(grid,i_strt=i_0,i_stop=i_1)
        call getDomainBounds(grid,j_strt=j_0,j_stop=j_1)
        call getdomainbounds(grid,
     &          have_south_pole = have_south_pole,
     &          have_north_pole = have_north_pole)
        do j=j_0,j_1
        do i=i_0,imaxj(j)
          if(atmocn%focean(i,j).gt.0.) then
            oarea(i,j) = axyp(i,j)*atmocn%focean(i,j)
            annsum(i,j) = sum(ohtconv(i,j,:))*oarea(i,j)
          else
            oarea(i,j) = 0.
            annsum(i,j) = 0.
          endif
        enddo
        enddo
        if(have_south_pole) then
          j = 1
          oarea(2:im,j) = oarea(1,j)
          annsum(2:im,j) = annsum(1,j)
        endif
        if(have_north_pole) then
          j = jm
          oarea(2:im,j) = oarea(1,j)
          annsum(2:im,j) = annsum(1,j)
        endif
        call globalsum(grid,annsum,ohtsum,all=.true.)
        call globalsum(grid,oarea,oasum,all=.true.)
        addadj = .true.
        !addadj = .false.
        if(addadj) then
          corr = (ohtsum/oasum)/sum(acctime_ohtconv)   ! W/m2
          do m=1,12
            do j=j_0,j_1
            do i=i_0,imaxj(j)
              if(atmocn%focean(i,j).gt.0.) then
                ohtconv_out(i,j,m) = ohtconv_out(i,j,m) - corr
              endif
            enddo
            enddo
          enddo
          if(grid%gid.eq.0) then
            write(6,*) 'corr ',itime,corr
          endif
        else ! multiplicative adjustment
          allocate(posarr(grid%i_strt_halo:grid%i_stop_halo,
     &                    grid%j_strt_halo:grid%j_stop_halo))
          posarr = max(annsum, 0d0)
          call globalsum(grid,posarr,sumpos,all=.true.)
          sumneg = ohtsum - sumpos
          if(sumpos .gt. -sumneg) then ! multiply positive values by factor<1
            multfac = -sumneg/sumpos
            do m=1,12
            do j=j_0,j_1
            do i=i_0,imaxj(j)
              if(atmocn%focean(i,j).gt.0. .and.
     &           ohtconv(i,j,m).gt.0.) then
                ohtconv_out(i,j,m) = ohtconv_out(i,j,m)*multfac
              endif
            enddo
            enddo
            enddo
            if(grid%gid.eq.0) then
              write(6,*) 'multfacp ',multfac
            endif
          else ! multiply negative values by factor<1
            multfac = -sumpos/sumneg
            do m=1,12
            do j=j_0,j_1
            do i=i_0,imaxj(j)
              if(atmocn%focean(i,j).gt.0. .and.
     &             ohtconv(i,j,m).lt.0.) then
                ohtconv_out(i,j,m) = ohtconv_out(i,j,m)*multfac
              endif
            enddo
            enddo
            enddo
            if(grid%gid.eq.0) then
              write(6,*) 'multfacn ',multfac
            endif
          endif
        endif
        if(have_south_pole) then
          j = 1
          do m=1,12
            ohtconv_out(2:im,j,m) = ohtconv_out(1,j,m)
          enddo
        endif
        if(have_north_pole) then
          j = jm
          do m=1,12
            ohtconv_out(2:im,j,m) = ohtconv_out(1,j,m)
          enddo
        endif
        write (ye,'(I4.4)') yeare-1
        write (yi,'(I4.4)') year1_ohtconv
        write (MLDmax,'(I0)') nint(z12o_max)
        ! Write out the cumulative OHT as a separate file

        fid = par_open(grid,'OHT.'//xlabel(1:lrunid)//'.'//yi
     &              //'-'//ye//'.MXL'//trim(MLDmax)//'m.nc','create')
        call defvar(grid,fid,ohtconv_out,
     &       'ohtconv(dist_im,dist_jm,unlim_month)')
        call defvar(grid,fid,z12o_max,'z12o_max')
        call par_enddef(grid,fid)
        call write_dist_data(grid,fid,'ohtconv',ohtconv_out)
        call write_data(grid,fid,'z12o_max',z12o_max)
        call par_close(grid,fid)
      endif

      end subroutine finishmonth_ohtconv

      subroutine oceans_ohtconv(atmocn,atmice)
      use model_com, only : modelEclock,dtsrc
      use domain_decomp_atm, only : grid,getDomainBounds
      use ocnml, only : ohtconv,acctime_ohtconv
      use exchange_types, only : atmocn_xchng_vars,atmice_xchng_vars
      use seaice_com, only : si_ocn
      implicit none
      type(atmocn_xchng_vars) :: atmocn
      type(atmice_xchng_vars) :: atmice
!
      integer i,j, j_0,j_1, i_0,i_1
      integer :: month

      real*8, dimension(:,:), pointer :: rsi
      rsi => si_ocn%rsi

      call getDomainBounds(grid,i_strt=i_0,i_stop=i_1)
      call getDomainBounds(grid,j_strt=j_0,j_stop=j_1)

      ! increment surface energy budget

      call modelEclock%get(month=month)
      acctime_ohtconv(month) = acctime_ohtconv(month) + dtsrc
      do j=j_0,j_1
      do i=i_0,atmocn%imaxj(j)
        if(atmocn%focean(i,j).gt.0.) then
          ohtconv(i,j,month) = ohtconv(i,j,month) - (
     &             ( atmocn%e0(i,j)*(1d0-rsi(i,j)) +
     &               atmice%e0(i,j)*rsi(i,j) )
                   ! include river and iceberg energy
     &           +  (atmocn%eflowo(i,j) + atmocn%egmelt(i,j))
     &         )
        endif
      enddo
      enddo

      return
      end subroutine oceans_ohtconv

      subroutine precip_ohtconv(atmocn)
      use model_com, only : modelEclock
      use ocnml, only : ohtconv
      use domain_decomp_atm, only : grid,getDomainBounds
      use exchange_types, only : atmocn_xchng_vars
      implicit none
      type(atmocn_xchng_vars) :: atmocn
      integer i,j, j_0,j_1, i_0,i_1
      integer :: month

      call getDomainBounds(grid,i_strt=i_0,i_stop=i_1)
      call getDomainBounds(grid,j_strt=j_0,j_stop=j_1)
      call modelEclock%get(month=month)
      do j=j_0,j_1
      do i=i_0,atmocn%imaxj(j)
        if(atmocn%focean(i,j).gt.0.) then
          ohtconv(i,j,month) = ohtconv(i,j,month) - atmocn%eprec(i,j)
        endif
      enddo
      enddo

      return
      end subroutine precip_ohtconv

      subroutine new_io_ocdiag(fid,iaction)
      use ocnml
      use model_com, only : iowrite_single,itime,itimei
      implicit none
      integer fid,iaction

      if(do_ohtconv .and. iaction.eq.iowrite_single .and.
     &     itime.gt.1+itimei) then
        call finishmonth_ohtconv
      endif
      return
      end subroutine new_io_ocdiag

c
c the following are dummy routines
c
      subroutine def_rsf_ocdiag(fid)
      implicit none
      integer fid
      return
      end subroutine def_rsf_ocdiag
      subroutine set_ioptrs_ocnacc_default
      return
      end subroutine set_ioptrs_ocnacc_default
      subroutine set_ioptrs_ocnacc_extended
      return
      end subroutine set_ioptrs_ocnacc_extended
      subroutine def_meta_ocdiag(fid)
      implicit none
      integer fid
      return
      end subroutine def_meta_ocdiag
      subroutine write_meta_ocdiag(fid)
      implicit none
      integer fid
      return
      end subroutine write_meta_ocdiag


      SUBROUTINE conserv_OCE(OCEANE)
!@sum  conserv_OCE calculates ocean energy for Qflux ocean
!@auth Gavin Schmidt
      USE CONSTANT, only : shw,rhows
      USE RESOLUTION, only : im,jm
      USE GEOM, only : imaxj
      USE OCNML, only : tocean,z1o,z12o
      USE DOMAIN_DECOMP_ATM, only : GRID
      USE DOMAIN_DECOMP_1D, only : getDomainBounds
      use fluxes, only : atmocn
      IMPLICIT NONE
!@var OCEANE ocean energy (J/M^2)
      REAL*8, DIMENSION(grid%I_STRT_HALO:grid%I_STOP_HALO,
     &                  grid%J_STRT_HALO:grid%J_STOP_HALO) :: OCEANE
      INTEGER I,J
      integer :: J_0, J_1, I_0, I_1
      logical :: HAVE_NORTH_POLE, HAVE_SOUTH_POLE

C****
C**** Extract useful local domain parameters from "grid"
C****
      call getDomainBounds(grid, J_STRT = J_0, J_STOP = J_1,
     &          HAVE_SOUTH_POLE = HAVE_SOUTH_POLE,
     &          HAVE_NORTH_POLE = HAVE_NORTH_POLE)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

      DO J=J_0, J_1
      DO I=I_0,IMAXJ(J)
        IF (atmocn%FOCEAN(I,J).gt.0) THEN
          OCEANE(I,J)=(TOCEAN(1,I,J)*Z1O(I,J)
     *         +TOCEAN(2,I,J)*(Z12O(I,J)-Z1O(I,J)))*SHW*RHOWS
        ELSE
          OCEANE(I,J)=0
        END IF
      ENDDO
      ENDDO
      IF (HAVE_SOUTH_POLE) OCEANE(2:im,1) =OCEANE(1,1)
      IF (HAVE_NORTH_POLE) OCEANE(2:im,JM)=OCEANE(1,JM)
C****
      END SUBROUTINE conserv_OCE

      SUBROUTINE DUMMY_OCN
!@sum  DUMMY necessary entry points for non-dynamic/non-deep oceans
!@auth Gavin Schmidt
      ENTRY ODIFS
      ENTRY io_ocdiag
      ENTRY gather_odiags ()
      ENTRY diag_OCEAN
      ENTRY diag_OCEAN_prep

      ENTRY init_ODEEP(iniOCEAN)
      ENTRY alloc_ODEEP

      RETURN
      END SUBROUTINE DUMMY_OCN
