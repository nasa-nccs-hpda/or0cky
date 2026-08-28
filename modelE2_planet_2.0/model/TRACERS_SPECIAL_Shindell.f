#include "rundeck_opts.h"

      MODULE TRACER_SOURCES

#ifdef TRACERS_ON

      use timestream_mod, only : timestream

      IMPLICIT NONE
      SAVE

!@var GLTic vmr initial conditions for GLT tracer
      REAL*8, PARAMETER ::  GLTic = 1.d-9
!@dbparam GLToffset
      integer :: GLToffset = 0

#ifdef INTERACTIVE_WETLANDS_CH4
!@dbparam nn_or_zon approach to use for expanding wetlands 1=
!@+ zonal average, 0=nearest neighbor average
!@dbparam int_wet_dist to turn on/off interacive SPATIAL wetlands
!@dbparam ice_age if not = 0., allows no wetl emis for latitudes
!@+ poleward of +/- ice_age in degrees
!@dbparam ns_wet the number of source that is the wetlands src 
!@dbparam exclude_us_eu to exclude (=1) the U.S. and E.U. from 
!@+ interactive wetland distributiont
!@dbparam topo_lim upper limit on topography for int wetl dist
!@dbparam sat_lim lower limit on surf air temp for int wetl dist
!@dbparam gw_llim lower limit on ground wetness for int wetl dist
!@dbparam gw_ulim upper limit on ground wetness for int wetl dist
!@dbparam SW_lim lower limit on SW down flux for int wetl dist
!@param nra_ch4 number of running averages needed for int-wetlands
!@param nra_ncep # of ncep running averages needed for int-wetlands
!@param nday_ncep number of days in ncep running averages
!@param nday_ch4 number of days in running average
!@var maxHR_ch4 maximum number of sub-daily accumulations
!@var by_nday_ncep  1/real(nday_ncep)
!@var by_nday_ncep  1/real(nday_ch4)
!@var day_ncep daily NCEP temperature & precipitation 
!@var DRA_ch4 daily running average of model (prec,temp)
!@var avg_ncep running average (prec,temp) over nday_ncep days
!@var avg_model equivalent of avg_ncep, but based on model variables
!@var sum_ncep temp arrays for computing running average
!@var PRS_ch4 period running sum of model (prec,temp)
!@var HRA_ch4 hourly running average of model (prec,temp)
!@var iday_ncep current day (counter) of averaging period (prec,temp)
!@var iHch4 "hourly" index for averages of model (prec,temp)
!@var iDch4 "daily"  index for averages of model (prec,temp)
!@var i0ch4 ponter to current index in running sum of mode (prec,temp)
!@var first_ncep whether in the first ncep averaging period (prec,temp)
!@var first_mod whether in the first model averaging per.   (prec,temp)
!@var PTBA variable to hold the pressure, temperature, beta, and alpha
!@var PTBA1, PTBA2 for interpolations of PTBA
      integer, parameter :: nra_ch4 = 5, nra_ncep=2, n__prec=1,
     &  n__temp=2, n__SW=3, n__SAT=4, n__gwet=5, max_days=28, nncep=4
      integer :: int_wet_dist=0,exclude_us_eu=1,nn_or_zon=0,ns_wet=-1
      real*8 :: topo_lim = 205.d0, sat_lim=-9.d0, 
     & gw_ulim=100.d0, gw_llim=18.d0, SW_lim=27.d0, ice_age=0.d0
      integer, parameter, dimension(nra_ch4) :: 
     &                                  nday_ch4=(/28,14,14,14,28/)
      integer, parameter, dimension(nra_ncep):: nday_ncep=(/28,14/)
      real*8, dimension(nra_ch4)             :: by_nday_ch4
      real*8, dimension(nra_ncep)            :: by_nday_ncep
      integer, dimension(nncep)      :: ncep_units,jmon_nc
      logical :: wetl_first=.true.
      REAL*8, ALLOCATABLE, DIMENSION(:,:,:,:):: day_ncep
      REAL*8, ALLOCATABLE, DIMENSION(:,:,:,:):: DRA_ch4
      REAL*8, ALLOCATABLE, DIMENSION(:,:,:)  :: avg_model,PRS_ch4
      REAL*8, ALLOCATABLE, DIMENSION(:,:,:)  :: avg_ncep,sum_ncep
      REAL*8, ALLOCATABLE, DIMENSION(:,:,:,:):: HRA_ch4
      integer, dimension(nra_ncep) :: iday_ncep=0,i0_ncep=0,first_ncep=1
      INTEGER, ALLOCATABLE, DIMENSION(:,:,:) :: iHch4,iDch4,i0ch4,
     &                                          first_mod
      real*8, allocatable, dimension(:,:,:) ::  PTBA,PTBA1,PTBA2
      real*8, allocatable, dimension(:,:) :: add_wet_src
      integer :: maxHR_ch4
      type(timestream), dimension(nncep) :: wetlStream
#endif
#endif
      END MODULE TRACER_SOURCES


      subroutine alloc_tracer_sources(grid)
!@SUM  To alllocate arrays whose sizes now need to be determined
!@+    at run-time
!@auth G.Faluvegi
      use domain_decomp_atm, only : dist_grid, getDomainBounds
      use domain_decomp_atm, only : write_parallel
      USE Dictionary_mod, only : get_param, is_set_param
      use resolution, only : lm
      use model_com, only : DTsrc
      use TimeConstants_mod, only: SECONDS_PER_HOUR
      use tracer_sources
      use fluxes, only : NIsurf
      IMPLICIT NONE

      type (dist_grid), intent(in) :: grid
      integer :: ier, J_1H, J_0H, I_1H, I_0H
      logical :: init = .false.
      real*8 :: DTsrc_LOCAL
      integer :: NIsurf_LOCAL

      if(init)return
      init=.true.
    
      call getDomainBounds( grid , J_STRT_HALO=J_0H, J_STOP_HALO=J_1H )
      I_0H = grid%I_STRT_HALO
      I_1H = grid%I_STOP_HALO

#ifdef INTERACTIVE_WETLANDS_CH4
      ! here I want to define how many surface calls expected in
      ! each day, but real DTsrc and NIsurf are not available yet
      ! so use local copy from the database:

      DTsrc_LOCAL = DTsrc
      if(is_set_param("DTsrc"))call get_param("DTsrc",DTsrc_LOCAL)
      NIsurf_LOCAL = NIsurf
      if(is_set_param("NIsurf"))call get_param("NIsurf",NIsurf_LOCAL)

      maxHR_ch4=24*NIsurf_LOCAL*NINT(SECONDS_PER_HOUR/DTsrc_LOCAL)
#endif
 
#ifdef INTERACTIVE_WETLANDS_CH4
      allocate( first_mod(I_0H:I_1H,J_0H:J_1H,nra_ch4) )
      allocate( iHch4(I_0H:I_1H,J_0H:J_1H,nra_ch4) )
      allocate( iDch4(I_0H:I_1H,J_0H:J_1H,nra_ch4) )
      allocate( i0ch4(I_0H:I_1H,J_0H:J_1H,nra_ch4) )
      allocate( day_ncep(I_0H:I_1H,J_0H:J_1H,max_days,nra_ncep) )
      allocate( DRA_ch4(I_0H:I_1H,J_0H:J_1H,max_days,nra_ch4) )
      allocate( avg_model(I_0H:I_1H,J_0H:J_1H,nra_ch4) )
      allocate( PRS_ch4(I_0H:I_1H,J_0H:J_1H,nra_ch4) )
      allocate( avg_ncep(I_0H:I_1H,J_0H:J_1H,nra_ncep) )
      allocate( sum_ncep(I_0H:I_1H,J_0H:J_1H,nra_ncep) )
      allocate( HRA_ch4(I_0H:I_1H,J_0H:J_1H,maxHR_ch4,nra_ch4) )
      allocate( PTBA(I_0H:I_1H,J_0H:J_1H,nncep) )
      allocate( PTBA1(I_0H:I_1H,J_0H:J_1H,nncep) )
      allocate( PTBA2(I_0H:I_1H,J_0H:J_1H,nncep) )
      allocate( add_wet_src(I_0H:I_1H,J_0H:J_1H) )

      first_mod(:,:,:) = 1  
      iHch4(:,:,:) = 0
      iDch4(:,:,:) = 0
      i0ch4(:,:,:) = 0   
      day_ncep(:,:,:,:) =  0.d0
      DRA_ch4(:,:,:,:) =  0.d0 
      avg_model(:,:,:) = 0.d0 
      PRS_ch4(:,:,:) = 0.d0
      avg_ncep(:,:,:) = 0.d0
      sum_ncep(:,:,:) = 0.d0
      HRA_ch4(:,:,:,:) = 0.d0
      PTBA(:,:,:) = 0.d0
      PTBA1(:,:,:) = 0.d0
      PTBA2(:,:,:) = 0.d0
      add_wet_src(:,:) = 0.d0
#endif      
      
      return
      end subroutine alloc_tracer_sources 


      subroutine overwrite_GLT(i,j)
!@sum L=1 overwriting of generic linear tracer    
!@vers 2013/03/26
!@auth Greg Faluvegi
C****
C**** Right now, there is just one L=1 source that changes 
C**** linearly in time (at 1% increase per year)
      USE RESOLUTION, only : im,jm
      USE MODEL_COM, only: itime,itimei,DTsrc
      use TimeConstants_mod, only: SECONDS_PER_YEAR
      use atmcol_com, only: ma, byma
      use OldTracer_mod, only: trname, vol2mass
      USE TRACER_COM, only: trm_col,n_GLT
      USE TRACER_COM, only: nOverwrite
      USE TRACER_SOURCES, only: GLTic,GLToffset
      USE FLUXES, only : tr3Dsource
      
      IMPLICIT NONE
      integer, intent(in) :: i,j
      
!@var by_s_in_yr recip. of # seconds in a year
!@var new_mr mixing ratio to overwrite in L=1 this time step
!@var new_mass mass to overwrite in L=1 this time step

      REAL*8 bydtsrc, by_s_in_yr, new_mr, new_mass

      bydtsrc=1.d0/DTsrc
      by_s_in_yr = 1.d0/SECONDS_PER_YEAR

C initial source is an overwriting of GLTic pppv, then add
C 1% every year, linearly in time. (note: vol2mass should
C just be 1 for this tracer, but kept it in here, in case
C we change that.)
      new_mr = GLTic * (1.d0 +
     &(Itime-ItimeI+GLToffset)*DTsrc*by_s_in_yr*1.d-2) !pppv
      new_mass=new_mr*vol2mass(n_GLT)*ma(1) ! kg
      tr3Dsource(1,nOverwrite,n_GLT)=(new_mass-trm_col(1,n_GLT))*bydtsrc
      !i.e. tr3Dsource in kg/s 

      return
      end subroutine overwrite_GLT


      subroutine get_CH4_IC(icall)
      USE DOMAIN_DECOMP_ATM, only : GRID, getDomainBounds
      implicit none
      integer, intent(in) :: icall
!
      integer :: I,J, J_0, J_1, I_0, I_1

      call getDomainBounds(grid, J_STRT=J_0, J_STOP=J_1)
      call getDomainBounds(grid, I_STRT=I_0, I_STOP=I_1)

      do j=j_0,j_1
      do i=i_0,i_1
        call get_CH4_IC_column(icall,i,j)
      enddo
      enddo
      end subroutine get_CH4_IC

      subroutine get_CH4_IC_column(icall,i,j)
!@sum get_CH4_IC to generate initial conditions for methane.
!@vers 2013/03/26
!@auth Greg Faluvegi/Drew Shindell
      USE RESOLUTION, only : ls1=>ls1_nominal
      USE RESOLUTION, only : im,jm,lm
      USE MODEL_COM, only  : DTsrc
      USE GEOM, only       : lat2d_dg
      USE ATM_COM, only: MA3D=>MA
      use atmcol_com, only: ma
      USE CONSTANT, only: mair
      use OldTracer_mod, only: vol2mass
      USE TRACER_COM, only : trm, trm_col, n_CH4, nOverwrite
      USE FLUXES, only: tr3Dsource
      USE TRCHEM_Shindell_COM, only: CH4altT, CH4altX, ch4_init_shnh,
     *     fix_CH4_chemistry
 
      IMPLICIT NONE
      integer, intent(in) :: icall,i,j
 
!@var CH4INIT temp variable for ch4 initial conditions
!@var I,J,L dummy loop variables
!@param bymair 1/molecular wt. of air = 1/mair
!@var icall =1 (during run) =0 (first time)
      REAL*8, PARAMETER :: bymair = 1.d0/mair
      REAL*8 CH4INIT,bydtsrc
      INTEGER L

      bydtsrc=1.d0/DTsrc
C     First, the troposphere:
C       Initial latitudinal gradient for CH4:
      IF(LAT2D_DG(I,J) < 0.) THEN ! Southern Hemisphere
        CH4INIT=ch4_init_shnh(1)*vol2mass(n_CH4)*1.d-6
      ELSE                      ! Northern Hemisphere
        CH4INIT=ch4_init_shnh(2)*vol2mass(n_CH4)*1.d-6
      ENDIF
      select case(icall)
      case(0)                   ! initial conditions
        DO L=1,LS1-1
          trm(i,j,l,n_CH4) = MA3D(L,I,J)*CH4INIT
        END DO
      case(1)                   ! overwriting
        DO L=1,LS1-1
          tr3Dsource(l,nOverwrite,n_CH4) = (ma(L)*
     &         CH4INIT-trm_col(l,n_CH4))*bydtsrc
        END DO
      end select

C     Now, the stratosphere:
      do L=LS1,LM

c     Define stratospheric ch4 based on HALOE obs for tropics
c     and extratropics and scale by the ratio of initial troposphere
c     mixing ratios to 1.79 (observed):
        IF(LAT2D_DG(I,J) < 0.) THEN ! Southern Hemisphere
          CH4INIT=ch4_init_shnh(1)/1.79d0*vol2mass(n_CH4)*1.d-6
        ELSE                        ! Northern Hemisphere
          CH4INIT=ch4_init_shnh(2)/1.79d0*vol2mass(n_CH4)*1.d-6
        ENDIF
        IF(ABS(LAT2D_DG(I,J)) > 30.) THEN ! extratropics
          CH4INIT=CH4INIT*CH4altX(L)
        ELSE                              ! tropics
          CH4INIT=CH4INIT*CH4altT(L)
        END IF
        select case(icall)
        case(0) ! initial conditions
          trm(i,j,l,n_CH4) = MA3D(L,I,J)*CH4INIT
        case(1) ! overwriting
          tr3Dsource(l,nOverwrite,n_CH4) = (ma(L)*
     &    CH4INIT-trm_col(l,n_CH4))*bydtsrc
        end select
      end do ! l
      
      RETURN
      end subroutine get_CH4_IC_column
   
      
      subroutine interpolateAltitude()
      USE TRCHEM_Shindell_COM,only:LCOalt,PCOalt,
     &     ClOXaltIN, ClOxalt,
     &     BrOXaltIN, BrOxalt,
     &     HClAltIN, Hclalt,
     &     ClONO2altIN, ClONO2alt
      USE ATM_COM, only: pmidl00
      USE RESOLUTION, only : LM

C     Interpolate ClOx altitude-dependence to model resolution:
      CALL LOGPINT(LCOalt,PCOalt,ClOxaltIN,LM,PMIDL00,ClOxalt,.true.)
C     Interpolate BrOx altitude-dependence to model resolution:
      CALL LOGPINT(LCOalt,PCOalt,BrOxaltIN,LM,PMIDL00,BrOxalt,.true.)
C     Interpolate HCl altitude-dependence to model resolution:
      CALL LOGPINT(LCOalt,PCOalt,HClaltIN,LM,PMIDL00,HClalt,.true.)
C     Interpolate ClONO2 altitude-dependence to model resolution:
      CALL
     &    LOGPINT(LCOalt,PCOalt,ClONO2altIN,LM,PMIDL00,ClONO2alt,.true.)
      end subroutine interpolateAltitude
 
      SUBROUTINE LOGPINT(LIN,PIN,AIN,LOUT,POUT,AOUT,min_zero)
!@sum LOGPINT does vertical interpolation of column variable,
!@+   linearly in ln(P).
!@auth Greg Faluvegi
 
      IMPLICIT NONE

!@var LIN number of levels for initial input variable
!@var LOUT number of levels for output variable
!@var PIN pressures at LIN levels
!@var POUT pressures at LOUT levels
!@var AIN initial input column variable
!@var AOUT output (interpolated) column variable
!@var LNPIN natural log of PIN
!@var LNPOUT natural log of POUT
!@var min_zero if true, don't allow negatives
!@var slope slope of line used for extrapolations
 
      LOGICAL, INTENT(IN)                 :: min_zero
      INTEGER, INTENT(IN)                 :: LIN, LOUT
      REAL*8, INTENT(IN), DIMENSION(LIN)  :: PIN, AIN
      REAL*8, INTENT(IN),DIMENSION(LOUT)  :: POUT
      REAL*8, INTENT(OUT),DIMENSION(LOUT) :: AOUT 
      REAL*8, DIMENSION(LIN)              :: LNPIN
      REAL*8, DIMENSION(LOUT)             :: LNPOUT       
      INTEGER L1,L2
      REAL*8 slope
C      
      LNPIN(:) = LOG(PIN(:))                   ! take natural log
      LNPOUT(:)= LOG(POUT(:))                  ! of pressures
C      
      DO L1=1,LOUT
       IF (LNPOUT(L1)>LNPIN(1)) THEN        ! extrapolate
         slope=(AIN(2)-AIN(1))/(LNPIN(2)-LNPIN(1))
         AOUT(L1)=AIN(1)-slope*(LNPIN(1)-LNPOUT(L1))
       ELSE IF (LNPOUT(L1) < LNPIN(LIN)) THEN ! extrapolate
         slope=(AIN(LIN)-AIN(LIN-1))/(LNPIN(LIN)-LNPIN(LIN-1))
         AOUT(L1)=AIN(LIN)+slope*(LNPOUT(L1)-LNPIN(LIN))
       ELSE                                    ! interpolate
        DO L2=1,LIN-1
         IF(LNPOUT(L1) == LNPIN(L2)) THEN
           AOUT(L1)=AIN(L2)
         ELSE IF(LNPOUT(L1) == LNPIN(L2+1)) THEN
           AOUT(L1)=AIN(L2+1)
         ELSE IF(LNPOUT(L1) < LNPIN(L2) .and.
     &   LNPOUT(L1) > LNPIN(L2+1)) THEN
           AOUT(L1)=(AIN(L2)*(LNPIN(L2+1)-LNPOUT(L1)) 
     &     +AIN(L2+1)*(LNPOUT(L1)-LNPIN(L2)))/(LNPIN(L2+1)-LNPIN(L2))
         END IF
        END DO
       END IF
      END DO          
C      
C If necessary: limit interpolated array to positive numbers:
C
      IF(min_zero)AOUT(:)=MAX(0.d0,AOUT(:))
C           
      RETURN
      END SUBROUTINE LOGPINT


#ifdef INTERACTIVE_WETLANDS_CH4
      subroutine running_average(var_in,I,J,nicall,m)
!@sum running_average keeps a running average of the model variables
!@+ for use with the interactive wetlands CH4. Currently: 1st layer
!@+ ground temperature, precipitaion, downward SW rad flux, surface
!@+ air temperature, and ground wetness. 
!@+ I suppose I could generalized this in the future.
!@auth Greg Faluvegi
C
C**** Global variables:
c
      USE MODEL_COM, only: DTsrc
      use TimeConstants_mod, only: SECONDS_PER_HOUR, HOURS_PER_DAY
      USE TRACER_SOURCES, only: iH=>iHch4,iD=>iDch4,i0=>i0ch4,
     & first_mod,HRA=>HRA_ch4,DRA=>DRA_ch4,PRS=>PRS_ch4,
     & nday_ch4,by_nday_ch4,nra_ch4,maxHR_ch4,avg_model
C
      IMPLICIT NONE
c
C**** Local parameters and variables and arguments
C     
!@var var_in model variable which will be used in the average
!@var nicall number of times routine is called per main timesetep
!@var m the index of the variable in question
!@var temp just for holding current day average for use in avg_model
!@var nmax number of accumulations in one day
!@var bynmax reciprocal of nmax
      real*8, intent(IN) :: var_in, nicall
      real*8 temp, bynmax
      integer, intent(IN):: m, I, J
      integer n, nmax

      if(m > nra_ch4.or.m < 1)call stop_model('nra_ch4 problem',255)
      if(iH(I,J,m) < 0.or.iH(I,J,m) > maxHR_ch4) then
        write(6,*) 'IJM iH maxHR_ch4=',I,J,m,iH(I,J,m),maxHR_ch4
        call stop_model('iH or maxHR_ch4 problem',255)
      endif
      nmax=NINT(HOURS_PER_DAY*nicall*SECONDS_PER_HOUR/DTsrc)
      bynmax=1.d0/real(nmax)
      by_nday_ch4(:)=1.d0/real(nday_ch4(:))
      iH(I,J,m) = iH(I,J,m) + 1
      HRA(I,J,iH(I,J,m),m) = var_in
      ! do no more, unless it is the end of the day:

      if(iH(I,J,m) == nmax)then ! end of "day":
        iH(I,J,m) = 0
        if(first_mod(I,J,m) == 1)then ! first averaging period only
          iD(I,J,m) = ID(I,J,m) + 1 
          do n=1,nmax
            DRA(I,J,iD(I,J,m),m) = DRA(I,J,iD(I,J,m),m) + HRA(I,J,n,m)
          end do
          DRA(I,J,iD(I,J,m),m) = DRA(I,J,iD(I,J,m),m)*bynmax
          if(iD(I,J,m) == nday_ch4(m))then !end first period
            PRS(I,J,m) = 0.d0
            do n=1,nday_ch4(m)
              PRS(I,J,m) = PRS(I,J,m) + DRA(I,J,n,m)
            end do
            avg_model(I,J,m)= PRS(I,J,m) * by_nday_ch4(m)
            first_mod(I,J,m)=0
            iD(I,J,m)=0
            i0(I,J,m)=0
          end if
        else ! not first averaging period: update the running average
          i0(I,J,m) = i0(I,J,m) + 1 ! move pointer
          if(i0(I,J,m)  ==  nday_ch4(m)+1) i0(I,J,m)=1
          temp=0.d0
          do n=1,nmax
            temp = temp + HRA(I,J,n,m) 
          end do
          temp = temp * bynmax ! i.e. today's average
          PRS(I,J,m) = PRS(I,J,m) - DRA(I,J,i0(I,J,m),m)
          DRA(I,J,i0(I,J,m),m) = temp
          PRS(I,J,m) = PRS(I,J,m) + DRA(I,J,i0(I,J,m),m)
          avg_model(I,J,m)= PRS(I,J,m) * by_nday_ch4(m)
        end if
      end if

      END SUBROUTINE running_average


      subroutine read_ncep_for_wetlands(end_of_day)
!@sum reads NCEP precip and temperature data and the coefficients
!@+ used to parameterize CH4 wetlands emissions from these. Keeps
!@+ running average of these. Calculated the portion to add to
!@+ the CH4 source to be used later in subroutine alter_wetlands_source.
!@auth Greg Faluvegi based on Jean Lerner
      use model_com, only: modelEclock
      use timestream_mod, only : init_stream,read_stream
      use domain_decomp_atm, only: grid, getDomainBounds
      use TRCHEM_Shindell_COM, only: fix_CH4_chemistry
      use tracer_sources, only: nday_ncep,by_nday_ncep,first_ncep,
     &iday_ncep,day_ncep,i0_ncep,avg_ncep,sum_ncep,nra_ncep,max_days,
     &PTBA,PTBA1,PTBA2,nncep,ncep_units,jmon_nc,wetl_first,wetlStream

      implicit none

      integer :: n,m,i,j,k,day,year
      logical, intent(in) :: end_of_day
      character*10, dimension(nncep) :: ncep_files =
     & (/'PREC_NCEP ','TEMP_NCEP ','BETA_NCEP ','ALPHA_NCEP'/)
      real*8, dimension(nncep) :: ncepMins=(/0.d0,-1.d3,-1.d30,-1.d30/),
     &                            ncepMaxs=(/1.d3, 1.d3, 1.d30, 1.d30/)
      real*8,dimension(GRID%I_STRT_HALO:GRID%I_STOP_HALO
     *     ,GRID%J_STRT_HALO:GRID%J_STOP_HALO,nra_ncep)::day_ncep_tmp

      integer :: J_1, J_0, I_0, I_1

      call getDomainBounds(grid, J_STRT=J_0, J_STOP=J_1)
      call getDomainBounds(grid, I_STRT=I_0, I_STOP=I_1)

      if (fix_CH4_chemistry == 1) return ! don't bother if no CH4 chem

      by_nday_ncep(:)=1.d0/real(nday_ncep(:))

      do n=1,nra_ncep
       if(nday_ncep(n) > max_days .or. nday_ncep(n) < 1)
     & call stop_model('nday_ncep out of range',255)
      end do

      call modelEclock%get(year=year, dayOfYear=day)

      ! read the monthly data and interpolate to current day:
      if(wetl_first) then
        wetl_first=.false.
        do k = 1,nncep
          ! note these files are 12-months only, so no need to
          ! allow override of modelEclock year, and no need to set
          ! cyclic optional argument:
          call init_stream(grid,wetlStream(k),trim(ncep_files(k)),
     &    trim(ncep_files(k)),ncepMins(k),ncepMaxs(k),'linm2m',year,day)
        end do
      end if
      do k = 1,nncep
        call read_stream(grid,wetlStream(k),year,day,PTBA(:,:,k))
      end do

! Then update running averages of NCEP Precip(m=1) & Temp(m=2)

      IF(.not. end_of_day) return ! avoid redundant accumulation on restarts

      do m=1,nra_ncep
        if(m > nra_ncep)call stop_model('check on ncep index m',255)
        if(first_ncep(m) == 1)then !accumulate first nday_ncep(m) days
          iday_ncep(m) = iday_ncep(m) + 1
          day_ncep(I_0:I_1,J_0:J_1,iday_ncep(m),m)=
     &    PTBA(I_0:I_1,J_0:J_1,m)
          if(iday_ncep(m) == nday_ncep(m))then !end of averaging period
            sum_ncep(I_0:I_1,J_0:J_1,m)=0.d0
            do n=1,nday_ncep(m)
              sum_ncep(I_0:I_1,J_0:J_1,m)=
     &        sum_ncep(I_0:I_1,J_0:J_1,m)+day_ncep(I_0:I_1,J_0:J_1,n,m)
            end do
            first_ncep(m)= 0
            iday_ncep(m) = 0
            i0_ncep(m)   = 0
            avg_ncep(I_0:I_1,J_0:J_1,m)=
     &      sum_ncep(I_0:I_1,J_0:J_1,m)*by_nday_ncep(m)
          end if
        else                     ! no longer first averaging period
          i0_ncep(m) = i0_ncep(m) + 1
          if(i0_ncep(m) > nday_ncep(m)) i0_ncep(m) = 1
          day_ncep_tmp(I_0:I_1,J_0:J_1,m) = PTBA(I_0:I_1,J_0:J_1,m)
          sum_ncep(I_0:I_1,J_0:J_1,m)= sum_ncep(I_0:I_1,J_0:J_1,m) - 
     &    day_ncep(I_0:I_1,J_0:J_1,i0_ncep(m),m)
          day_ncep(I_0:I_1,J_0:J_1,i0_ncep(m),m) = 
     &    day_ncep_tmp(I_0:I_1,J_0:J_1,m)
          sum_ncep(I_0:I_1,J_0:J_1,m)= sum_ncep(I_0:I_1,J_0:J_1,m) + 
     &    day_ncep(I_0:I_1,J_0:J_1,i0_ncep(m),m)
          avg_ncep(I_0:I_1,J_0:J_1,m) = sum_ncep(I_0:I_1,J_0:J_1,m)*
     &    by_nday_ncep(m)
        endif
      end do ! m

      return
      end subroutine read_ncep_for_wetlands


      subroutine alter_wetlands_source(n,ns_wet)
!@sum alter_wetlands_source changes the magnitude of the CH4 wetlands
!@+ (+ Tundra) source based on B.Walter's parameterization comparing
!@+ 1st layer ground temperature from 1 week ago and precipitation 
!@+ from 2 weeks ago, such that:
!@+  CH4emis = CH4emis + (alpha*TempAnom + beta*PrecAnom)
!@+ It then optionally allows for change in the wetlands source IJ
!@+ distribution based on criteria for topography, surf. air temp.,
!@+ downward SW radiation, land fraction, and ground wetness.
!@+ You also have the option to exclude the U.S. and E.U. from 
!@+ wetland distribution changes.
!@auth Greg Faluvegi
      USE RESOLUTION, only : im,jm
      USE FLUXES, only : fland
      USE MODEL_COM, only: itime, modelEclock
      USE DOMAIN_DECOMP_ATM, only: GRID, getDomainBounds,
     &   broadcast, write_parallel
#ifdef CUBED_SPHERE
      USE DD2D_UTILS, only : 
#else
      USE DOMAIN_DECOMP_1D, only : 
#endif
     &     pack_data
      use OldTracer_mod, only: itime_tr0,trname
      USE Tracer_mod, only: ntsurfsrcmax
      USE TRACER_COM, only: sfc_src
      use TRACER_SOURCES, only: PTBA,nncep,first_ncep,avg_ncep,
     &   avg_model,nra_ncep,int_wet_dist,topo_lim,sat_lim,
     &   gw_ulim,gw_llim,SW_lim,exclude_us_eu,nra_ch4,first_mod,
     &   n__temp,n__sw,n__gwet,n__SAT,nn_or_zon,ice_age,add_wet_src
      use GEOM, only : lat2d_dg, lon2d_dg, imaxj
      use ghy_com, only : top_dev_ij,fearth
      USE TRCHEM_Shindell_COM, only: fix_CH4_chemistry

      implicit none
      
      integer, intent(in) :: n,ns_wet
      integer i,j,nt,iu,k
      character(len=300) :: out_line
      integer m,ii,ix,jj
      real*8 :: zm,zmcount
#ifdef CUBED_SPHERE
      real*8 :: src_glob(IM,IM,6),src_flatglob(1-im:2*im,1-im:2*im)
#else
      real*8, dimension(IM,JM) :: src_glob
#endif
      INTEGER :: J_1, J_0, J_0H, J_1H, I_0, I_1, J_0S,J_1S

      call getDomainBounds(grid, J_STRT=J_0, J_STOP=J_1,
     &     J_STRT_SKP=J_0S, J_STOP_SKP=J_1S)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

      if(ns_wet < 0 .or. ns_wet > ntsurfsrcmax)call stop_model
     & ('Must set ns_wet rundeck param to CH4 wetlands emis file',255)

! Don't alter the wetlands if:
      ! -->  the tracer is not supposed to be on yet:
      if (itime < itime_tr0(n)) return
      ! --> the methane is supposed to be fixed value:
      if (fix_CH4_chemistry == 1) return
      ! --> the first averaging period isn't done for ncep vars:
      do m=1,nra_ncep; if(first_ncep(m)==1) RETURN; end do

! Otherwise calculate the magnitude adjustments (not at poles or
! over water (/ice?) though. And not until the first averaging
! period is through (first_mod criterion):
      do m=1,nra_ncep
        loop_j: do j=J_0S,J_1S ! skipping poles
          loop_i: do i=I_0,imaxj(j)
            add_wet_src(i,j)=0.d0
            if(fearth(i,j) <= 0.) cycle loop_i
            if(first_mod(i,j,m)/=1.and.sfc_src(i,j,n,ns_wet)/=0.)
     &      add_wet_src(i,j) = add_wet_src(i,j) + PTBA(i,j,m+nra_ncep)*
     &      (avg_model(i,j,m) - avg_ncep(i,j,m))
          end do loop_i
        end do loop_j
      end do

! Now, determine the distribution (spatial) adjustments:
      if(int_wet_dist > 0)then ! if option is on

! Determine if there are new wetlands for given point,
! or remove existing wetlands:

       ! for nearest-neighbor all processors must know global source:
       if(nn_or_zon==0)then 
         call pack_data(grid,sfc_src(:,:,n,ns_wet),src_glob)
         call broadcast(grid,src_glob)
#ifdef CUBED_SPHERE
         ! Reorient the global array to the i-j index space of this
         ! processor. Implicit assumption (for now): wetlands exist on
         ! at least 2 faces; no data needed from the opposing face.
         call flatten_cube(src_glob,src_flatglob,im,grid%tile)
#endif
       else
#ifdef CUBED_SPHERE
         call stop_model('alter_wetlands_source: zonal mean '//
     &        'to be implemented using zonalmean_ij2ij',255)
#endif
       end if

       do j=J_0S,J_1S ! skipping poles
       do i=I_0,imaxj(j)
         if(fearth(i,j) <= 0.) cycle ! and boxes with no land

         ! if either {(point is not in U.S. or E.U.) .or. (it is but 
         ! the exclusion of U.S. and E.U. wetlands is OFF)} AND
         ! (the point has some land):

         if((exclude_us_eu == 0 .OR. .NOT.((lon2d_dg(i,j)
     &   >= -122.5.and.lon2d_dg(i,j) <= -72.5.and.lat2d_dg(i,j) >= 34.0
     &   .and.lat2d_dg(i,j) <= 46.5).or.(lon2d_dg(i,j) >= -12.5.and.
     &   lon2d_dg(i,j) <= 17.5.and.lat2d_dg(i,j) >= 37.5
     &        .and.lat2d_dg(i,j)<= 62.0))) .AND. (fland(i,j) > 0.))then

           ! if the topography slope, surface are temp., SW radiation,
           ! and ground wetness are within certain limits:

           if(top_dev_ij(i,j) < topo_lim .AND. avg_model(i,j,n__SAT)
     &     > sat_lim .AND. avg_model(i,j,n__SW) > SW_lim .AND.
     &     avg_model(i,j,n__gwet) > gw_llim .AND.
     &     avg_model(i,j,n__gwet) < gw_ulim) then

             ! if no wetlands there yet:

             if(sfc_src(i,j,n,ns_wet) == 0.)then
               zm=0.d0; zmcount=0.d0
               select case (nn_or_zon)
               case(1) ! use zonal average (of existing wetlands)
                 do ii=I_0, I_1  !EXCEPTIONAL CASE? 
                  if(sfc_src(ii,j,n,ns_wet) > 0.d0)then
                    zm=zm+sfc_src(ii,j,n,ns_wet)
                    zmcount=zmcount+1.d0
                  endif
                 enddo
               case(0) ! use nearest neighbor approach
                 ix=0
                 do while(zmcount == 0.)
                   ix=ix+1
#ifdef CUBED_SPHERE
                   if(ix>2*im)
     &                  call stop_model('ix>2*im int wetl dist',255)
                   do jj=j-ix,j+ix
                     if(jj<1-im .or. jj>2*im) cycle
                     do ii=i-ix,i+ix,1+(2*ix-1)*(1-abs((jj-j)/ix))
                       if(ii<1-im .or. ii>2*im) cycle
                       if(src_flatglob(ii,jj).le.0.) cycle
                       zm=zm+src_flatglob(ii,jj)
                       zmcount=zmcount+1.d0
                     enddo
                   enddo
#else
                   if(ix>im)call stop_model('ix>im int wetl dist',255)
                   do ii=i-ix,i+ix ;do jj=j-ix,j+ix
                     if(ii>0.and.ii<=im)then
                       if(jj>0.and.jj<=jm)then
                         if(src_glob(ii,jj) > 0.)then
                           zm=zm+src_glob(ii,jj)
                           zmcount=zmcount+1.d0
                         endif
                       endif
                     endif
                   enddo           ;enddo
#endif
                 enddo
               case default
                 call stop_model('problem with nn_or_zon',255)
               end select
               if(zmcount <= 0.)then
                 write(out_line,*)'zmcount for wetl src <= 0 @ IJ=',I,J
                 call write_parallel(trim(out_line),unit=6,crit=.true.)
                 add_wet_src(i,j)=-1.d0*sfc_src(i,j,n,ns_wet)
               else
                 add_wet_src(i,j)=zm/zmcount
               end if
             end if       ! end of no prescribed wetlands there
           else           ! no wetlands should be here
             add_wet_src(i,j)=-1.d0*sfc_src(i,j,n,ns_wet)
           end if         ! end wetlands criteria
         end if           ! end U.S./E.U./land criteria
       end do             ! i loop
       end do             ! j loop
      end if              ! Consider interactive welands distrbution?

! Limit wetlands source to be positive:
      do j=J_0,J_1  
        do i=I_0,imaxj(j)
          if((sfc_src(i,j,n,ns_wet)+add_wet_src(i,j))<0.)
     &    add_wet_src(i,j)=-1.d0*sfc_src(i,j,n,ns_wet)
        enddo
      enddo

! Optionally disallow emissions over glacier latitudes:
      if(ice_age /= 0.) then
        do j=J_0,J_1 
        do i=I_0,imaxj(j)
          if(abs(lat2d_dg(i,j))>abs(ice_age))
     &    add_wet_src(i,j)=-1.d0*sfc_src(i,j,n,ns_wet)
        enddo
        enddo
      endif

      return
      end subroutine alter_wetlands_source

      subroutine flatten_cube(arr6,flat5,n,face)
!@sum Places data from 5 of the 6 faces of a cube into the i-j index
!@+   space of one of the faces. Referenced to that face, the cube is
!@+   comprised of
!@+   (1) that face
!@+   (2) the 4 adjacent faces in the directions left, right, down, up
!@+   (3) the opposing face
!@+   Data from the local/adjacent faces are stored in
!@+   flat5(1-n:2*n,1-n:2*n) as follows:
!@+    local face: i, j = 1  :n  , 1  :n  (each face is n by n points)
!@+   to the left: i, j = 1-n:0  , 1  :n
!@+         right: i, j = 1+n:2*n, 1  :n
!@+          down: i, j = 1  :n  , 1-n:0
!@+            up: i, j = 1  :n  , 1+n:2*n
!@+   flat5 is set to zero where both i and j are either <1 or >n
!@+   Data from the opposing side can also be placed into the flat array
!@+   in a symmetric manner, but this is not necessary for current
!@+   applications (filling missing data using neighboring values).
!@auth M. Kelley
      implicit none
!@ var arr6  : input global array with 3d-cube indexing
!@ var flat5 : output global array with flattened-cube indexing
!@ var n     : faces are n by n gridpoints
!@ var face  : the cube face to be used as reference for i-j indexing
      integer :: n,face
      real*8, dimension(n,n,6) :: arr6
      real*8, dimension(1-n:2*n,1-n:2*n) :: flat5
      integer :: facem1,facep1,facem2,facep2
      facem1 = 1+mod(6+(face-1)-1,6)
      facep1 = 1+mod(6+(face-1)+1,6)
      facem2 = 1+mod(6+(face-1)-2,6)
      facep2 = 1+mod(6+(face-1)+2,6)
      flat5 = 0.
      flat5(1:n,1:n) = arr6(:,:,face)
      if(mod(face,2).eq.0) then
        flat5(1-n:0,1:n)   = arr6(:,:,facem1)           ! l
        flat5(n+1:2*n,1:n) = rotcw(arr6(:,:,facep2),n)  ! r
        flat5(1:n,1-n:0) = rotccw(arr6(:,:,facem2),n)   ! d
        flat5(1:n,n+1:2*n) = arr6(:,:,facep1)           ! u
      else
        flat5(1-n:0,1:n) = rotcw(arr6(:,:,facem2),n)    ! l
        flat5(n+1:2*n,1:n) = arr6(:,:,facep1)           ! r
        flat5(1:n,1-n:0) = arr6(:,:,facem1)             ! d
        flat5(1:n,n+1:2*n) = rotccw(arr6(:,:,facep2),n) ! u
      endif
      return
      contains
      function rotcw(arr,m)
      integer :: m
      real*8, dimension(m,m) :: arr,rotcw
      rotcw = transpose(arr(m:1:-1,:))
      end function rotcw
      function rotccw(arr,m)
      integer :: m
      real*8, dimension(m,m) :: arr,rotccw
      rotccw = transpose(arr(:,m:1:-1))
      end function rotccw
      end subroutine flatten_cube
#endif /* INTERACTIVE_WETLANDS_CH4 */


#ifdef SOLAR_ENERGETIC_PARTICLES
      subroutine update_ion_pairs_and_apex(y,d,e_of_d)
      !@sum update_ion_pairs_and_apex initialize or update the daily
      !@+ input of ion pair production rate by solar protons, and,
      !@+ annually, the apex geomagnetic coordinate transform
      ! Ion pair input is on its own geomagnetic lats vs. pressure level
      ! grid. Full year is read in and known by all processor domains.
      ! I.e. only reads once a year and on init calls, but new values
      ! passed to chemistry daily.
      !@auth Greg Faluvegi
      use rad_com, only: s0_day, s0_yr
      use TRCHEM_Shindell_COM, only: iprpIn, iprpToday, ionGlat, ionPlev
      use TRCHEM_Shindell_COM, only: LNionPlev, qdlat3D
      use domain_decomp_atm, only: write_parallel, grid, getDomainBounds
      use apex, only: apex_mka, apex_mall, ggrid, apex_beg_yr
      use shr_kind_mod, only : r8 => shr_kind_r8
      use resolution, only: LM
      use atm_com, only: PEDNL00, PMIDL00
      use geom, only : lon2d_dg,lat2d_dg

      implicit none

      include 'netcdf.inc'

      integer :: y, d, xyear, xday, fid, gid, pid, rc, ntimes
      integer :: nglats, nplevs, tdid, gdid, pdid, vid
      logical :: e_of_d
      character(len=300) :: out_line
      character(len=80) :: fname

      integer :: i,j,j_0,j_1,i_0,i_1,L
      real*8 :: Z

      ! Below used r8 notation for consistency with apex component.
      !@param altmax maximum altitude in km used for magnetic
      !@+ calculations. Seems it must be set much higher than the model
      !@+ top (which is about 89 km in 102L and 65 in the 40L)
      real(r8), parameter :: altmax = 1000._r8
      !@param hr reference height in km for the apex calculation
      real(r8), parameter :: hr = 0._r8
      !@param nvert apex_setup routine at the URL below states:
      !@+ Resolution parameter, corresponding to the maximum number of
      !@+ vertical grid increments when altmax=infinity. Points are
      !@+ spaced uniformly in 1/r between the Earth's surface and an
      !@+ altitude that is at least altmax...
      ! http://download.hao.ucar.edu/pub/maute/apex/apex.f90
      integer, parameter :: nvert = 40
      !@var dec_date decimal year
      !@var min_date decimal year of apex start year
      real(r8) :: dec_date, min_date
      integer :: nlat=0, nlon=0, nalt=0, ier
      integer, parameter :: mxlat=3*nvert+1, mxlon=5*nvert+1
      integer, parameter :: mxalt=nvert+1
      !@var gplat apex 3D grid latitudes (deg)
      !@var gplon apex 3D grid longitudes (deg)
      !@var gpalt apex 3D grid altitudes (km)
      real(r8) :: gplat(mxlat), gplon(mxlon), gpalt(mxalt)
      !@var qdlat quasi-dipole latitude at given location (deg)
      real(r8) :: qdlat
      ! The rest of these will be returned from the apex call but not
      ! used. So, I am declaring them but not describing them:
      real(r8), dimension(3) :: inputB,Bhat,d1,d2,d3,e1,e2,e3,f1,f2
      real(r8) :: bmag,si,malat,vmp,W,DD,Be3,sim,F,alon

      ! Current model dates passed in are allowed to override the
      ! the solar dates from the rad code, if solar dates are transient:
      xyear=s0_yr ; if(xyear == 0) xyear=y
      xday=s0_day ; if(xday == 0) xday=d

      ! Next line means section happens upon model initializations and
      ! every Jan 1st. Since that is so infrequent, let's reallocate
      ! the arrays and re-read the coordinates so we can use exact same
      ! code for these two cases. Since above xyear/xday settings could
      ! theoretically have xyear changing with time while xday is
      ! static, use the model day to determine new years (i.e. d passed
      ! in is currently dayOfYear from the model clock):
      if(.not. e_of_d .or. d == 1 ) then

        ! Determine current year's file:
        write(fname,'(a10,I4,a3)')'ION_PAIRS/',xyear,'.nc'

        ! open file:
        rc=nf_open(trim(fname),ncnowrit,fid)
        if(rc/=nf_noerr)call ionStop('opening the file',rc)

        ! get the dimension ids and lenghts:
        rc=nf_inq_dimid(fid,'time',tdid)
        if(rc/=nf_noerr)call ionStop('locating time dimension',rc)
        rc=nf_inq_dimid(fid,'glat',gdid)
        if(rc/=nf_noerr)call ionStop('locating glat dimension',rc)
        rc=nf_inq_dimid(fid,'plev',pdid)
        if(rc/=nf_noerr)call ionStop('locating plev dimension',rc)
        rc=nf_inq_dimlen(fid,tdid,ntimes)
        if(rc/=nf_noerr)call ionStop('determining time dim size',rc)
        if(ntimes/=365)call ionStop('checking for daily input',rc)
        rc=nf_inq_dimlen(fid,gdid,nglats)
        if(rc/=nf_noerr)call ionStop('determining glat dim size',rc)
        rc=nf_inq_dimlen(fid,pdid,nplevs)
        if(rc/=nf_noerr)call ionStop('determining plev dim size',rc)

        ! deallocate/allocate the ion pair input coordinates and
        ! arrays that hold the ion pair production rate input:

        if(allocated(ionPlev)) deallocate(ionPlev)
        if(allocated(LNionPlev)) deallocate(LNionPlev)
        if(allocated(ionGlat)) deallocate(ionGlat)
        if(allocated(iprpIn)) deallocate(iprpIn)
        if(allocated(iprpToday)) deallocate(iprpToday)
        allocate( ionPlev( nplevs ) )
        allocate( LNionPlev( nplevs ) )
        allocate( ionGlat( nglats ) )
        allocate( iprpIn( nglats , nplevs, ntimes ) )
        allocate( iprpToday( nglats , nplevs ) )
        ionPlev=0.d0
        LNionPlev=0.d0
        ionGlat=0.d0
        iprpIn=0.e0 ! single precision
        iprpToday=0.d0

        ! fill in the coordinate variables, to be used in the chemistry
        ! to locate the ion pair production rate in space:
        rc=nf_inq_varid(fid,'plev',pid)
        if(rc/=nf_noerr)call ionStop('locating plev variable',rc)
        rc=nf_get_vara_double(fid,pid,1,nplevs,ionPlev)
        LNionPlev(1:nplevs)=LOG(ionPlev(1:nplevs))
        if(rc/=nf_noerr)call ionStop('reading plev variable',rc)
        rc=nf_inq_varid(fid,'glat',gid)
        if(rc/=nf_noerr)call ionStop('locating glat variable',rc)
        rc=nf_get_vara_double(fid,gid,1,nglats,ionGlat)
        if(rc/=nf_noerr)call ionStop('reading glat variable',rc)

        ! read in a full year of the ion pair production rate variable:
        rc=nf_inq_varid(fid,'iprX',vid)
        if(rc/=nf_noerr)call ionStop('locating iprX variable',rc)
        rc=nf_get_vara_real(
     &     fid,vid,(/1,1,1/),(/nglats,nplevs,ntimes/),iprpIn )
        if(rc/=nf_noerr)call ionStop('reading iprX variable',rc)

        ! close the file; full year is in memory.
        ! Note, since these were not parallel reads, I think no need to
        ! broadcast the variable.
        rc=nf_close(fid)
        if(rc/=nf_noerr)call ionStop('closing the file',rc)

        ! heartbeat the PRT file when a new file read:
        write(out_line,*)'Year of ion pair prod rate values read from '
     &  //' file: '//trim(fname)
        call write_parallel(trim(out_line))

        ! -------------------------------------------------------
        ! also only on restarts and on start of new years, update
        ! the apex coordinates:

        ! Create a grid that is regular but covers the general model
        ! domain. Apex coords will be calculated on it and then in later
        ! call, model grid points located on this grid:
        call ggrid(nvert,-90._r8,90._r8,-180._r8,180._r8,0._r8,altmax,
     &           gplat,gplon,gpalt,mxlat,mxlon,mxalt,nlat,nlon,nalt)

        ! Use a decimal date from center of current year:
        dec_date = dble(xyear)+0.5d0
        ! If that date is before the first available date, use first
        ! available date (see Greg F. emails Aug 2-3 2021 for
        ! discussion). Not doing the same for dates after end of
        ! record at this time.
        min_date =  dble(apex_beg_yr)+0.5d0
        if (dec_date < min_date ) then
          dec_date = min_date
          write(out_line,'(a48,I4,a19,I4,a9)')
     &    'WARNING: Year for apex coordinates calculation (',xyear,
     &    ') too early. Using ',apex_beg_yr,' instead.'
          call write_parallel(trim(out_line))
        end if

        ! Create the vector arrays to be used to get apex coordinates,
        ! based on grid created by ggrid above.
        call apex_mka(dec_date,gplat,gplon,gpalt,nlat,nlon,nalt,ier)
        if (ier /= 0) call stop_model("apex_mka error",255)
        call getDomainBounds(grid, J_STRT=J_0, J_STOP=J_1)
        call getDomainBounds(grid, I_STRT=I_0, I_STOP=I_1)
        do j=j_0,j_1
          do i=i_0,i_1
            do L=1,LM
              ! if this were called often we could use
              ! Z=phi(i,j,L)*bygrav*1.d-3 in km. Here estimate
              ! nominally using a scale hieght of 7km:
              Z=7.d0*log(PEDNL00(1)/PMIDL00(L))

              ! call apex model to get the quasi-dipole latitude from
              ! current geographic lat, lon and altitude. All other
              ! output is ignored:
              call apex_mall(
     &         lat2d_dg(i,j), lon2d_dg(i,j), Z, hr, !input
     &         inputB,bhat,bmag,si,alon,malat,vmp,W, ! output
     &         DD,Be3,sim,d1,d2,d3,e1,e2,e3,qdlat,F,f1,f2,ier) ! output
              if (ier /= 0) call stop_model("apex_mall error",255)

              ! saving for chemistry use:
              qdlat3D(L,i,j)=qdlat
            end do
          end do
        end do
        ! heartbeat the PRT file when a new file read:
        write(out_line,*)'Updating apex coords for date: ',dec_date
        call write_parallel(trim(out_line))

      end if

      ! If this routine is called, it is either a new day or the model
      ! init phase, so no further conditions on slicing today's values:
      ! I believe CMIP specified not to interpolate in time.
      iprpToday(:,:) = dble( iprpIn(:,:,xday) )

      ! heartbeat the PRT file when daily values change:
      ! (southernmost glat and middle of plevs):
      write(out_line,*)'Sample ion pair prod rate for day',xday,
     &': ',iprpToday(1,32)
      call write_parallel(trim(out_line))

      end subroutine update_ion_pairs_and_apex


      subroutine ionStop(activityString,status)
      !@sum error handling for the netCDF/Fortran interface reads to
      !@+ ION_PAIRS file
      use domain_decomp_1d, only: am_i_root
      implicit none
      include 'netcdf.inc'
      character(len=*) :: activityString
      integer status
      if(am_i_root())
     &print*, 'ION_PAIR READING: model was '//trim(activityString)
     &//', encountered the NF error: ',trim(trim(nf_strerror(status)))
      call stop_model('ION_PAIR I/O error. See PRT message.',255)
      end subroutine ionStop
#endif /* SOLAR_ENERGETIC_PARTICLES */


#ifdef TRACERS_ACETONE
      module oceanEmissions
      use constant, only: undef
      implicit none
      private
      public :: oceanSpecies
      type oceanSpecies
        character*11 :: itsname='____unknown' ! name of species, which should match tracer sourceName
        real*8 :: KHS=undef ! Henry's law constant at standard conditions (pg 3. of Sander 1999)
        real*8 :: TDS=undef ! temperature dependence of soluability (pg 3. of Sander 1999)
        real*8 :: conc=undef ! sea water constant concentration mole/L
        real*8 :: Vb=undef ! liquid molar volume at boiling cm3 mol-1
        real*8 :: mw=undef ! molecular weight g mol-1
      end type oceanSpecies
      end module oceanEmissions


      module oceanEmissionsSpecies
      use oceanEmissions, only: oceanSpecies
      implicit none
      integer, parameter :: nOceanSpecies=1
      type(oceanSpecies), dimension(nOceanSpecies) :: species
      type(oceanSpecies), save :: acetone ! not 100% sure about "save"
      end module oceanEmissionsSpecies


      subroutine oceanEmissions_drv(i,j)
      !@sum calculate ocean source or sink of atmospheric constituents
      !@+ using two-film model of Liss and Slater (1974). Fill in
      !@+ sfc_src array.
      !@auth Greg Faluvegi (intial modelE implementation)

      use oceanEmissions, only: oceanSpecies
      use oceanEmissionsSpecies, only: nOceanSpecies,species,acetone
      use model_com, only: itime, dtsrc
      use fluxes, only: focean,atmocn,atmsrf
      use seaice_com, only : si_atm
      use constant, only: tf
      use TimeConstants_mod, only: SECONDS_PER_HOUR
      use OldTracer_mod, only: trname,itime_tr0
      use tracer_com, only: ntm, sfc_src, tracers, trm
      use tracer_mod, only: Tracer
      use TracerSurfaceSource_mod, only: itsOcean
      use trdiag_com, only: trcSurfByVol
      use trdiag_com, only: ijs_AcetOtoA, ijs_AcetAtoO, taijs=>taijs_loc

      implicit none

      integer, intent(IN) :: i,j
      integer :: n, nTracer, ns
      character*80 :: message
      class (Tracer), pointer :: trc
      real*8 :: TC, TK, TK0, DTR, KH0, U10, ka, kw, k600, SC, ra
      real*8 :: CW, CA, K, CD, ustar, waterToAir, airToWater, bydtsrc
      real*8 :: openOcean
      real*8, parameter :: ocean_thresh=0.1d0
      real*8, parameter :: SC600=600.d0
      real*8, parameter :: vk=0.40d0 ! von karman constant [dimensionless]
      real*8, parameter :: TMIN=-5.d0, TMAX=30.d0 ! deg C (initially tried -60 to 100)
      real*8, parameter :: sinkFraction=0.95d0 ! fraction L=1 removal allowed
      integer :: source_count

      bydtsrc=1.d0/dtsrc

      ! List nOCeanSpecies Ocean species for easier looping:
      species( 1)=acetone

      ! Prepare some information from this GCM gridbox:
      ! Get the surface temperature in deg C (over ocean only), with
      ! some limits:
      TC = atmocn%GTEMP(i,j)
      TC = MIN( MAX ( TC, TMIN ), TMAX )
      TK = TC+tf
      TK0 = tf+25.d0 ! i.e. 298.15 K. standard
      ! Get difference of recipricol of TK vs. standard TK0:
      DTR=((1.d0/TK)-(1.d0/TK0))
      ! Get the surface wind-speed (seems like atmocn would be better(?)
      ! but that causes reproducibility issue):
      U10=atmsrf%wsavg(i,j)   !TODO: is this 10m relevant wind? or 2m?

      tracers_loop: do nTracer=1,ntm ! loop over tracers

        ! skip if tracer not turned on yet, otherwise point to it:
        if(itime < itime_tr0(nTracer) ) cycle tracers_loop
        trc => tracers%getReference(trname(nTracer))

        source_count=0

        sources_loop: do ns=1,trc%ntSurfSrc ! loop over defined sources

          ! skip if not an ocean source:
          if(trc%surfaceSources(ns)%skipReason /= itsOcean)
     &      cycle sources_loop

          source_count=source_count+1 ! found an ocean source
          ! initialize the source so that, e.g. if in the next timestep
          ! ice encroaches or something else happens such that sfc_src
          ! is not filled in, the model won't remember the previous
          ! step's source:
          sfc_src(i,j,nTracer,ns)=0.d0

          ! ... for the same reason, I moved this check here (instead
          ! of a "return" statement near the top of the routine). I.e.
          ! REMEMBER to put all conditionals after the zeroing above...
          ! Skip for boxes with too little open ocean:
          openOcean=(1.d0-si_atm%rsi(i,j))*FOCEAN(i,j)
          if(openOcean < ocean_thresh) cycle sources_loop

          ! try to match tracer with defined species, otherwise skip:
          species_loop: do n=1,size(species)

            if(trim(trc%surfaceSources(ns)%sourceName) ==
     &         trim(species(n)%itsname)) then

              ! Begin calculations:

              ! Equation 22 of Johnson, 2010 (doi:10.5194/os-6-913-2010)
              ! 12.2d0 conversion and the KHS and TDS are explained and
              ! tabulated in version 3 of Sander 1999,
              ! (http://www.henrys-law.org/henry-3.0.pdf). KH0 should be
              ! dimensionless gas-over-liquid Henry's Law constant here:
              KH0=12.2d0/(TK*species(n)%KHS*EXP(species(n)%TDS*DTR))
              !TODO: seemed to me that GEOS-CHEM V9.X has the DTR backwards.
              ! Or perhaps do I? Consistently, they have a constant T where 
              ! TK is in above line. I am doing it in the order of the
              ! K_calcs_Johnson_OS.R program though...

              ! Calculate the Schmidt number for this species a function
              ! of temperature, following Johnson 2010:
              call getSchmidt(species(n),TC,SC)

              ! Get the water-side transfer velocity based on eq 28 of
              ! Johnson 2010 but updating the k600 from Nightingale et
              ! al 2000 (G.R.L., Discussion section):
              k600=(0.24d0*U10*U10 + 0.061d0*U10)
              kw=k600*(SC/SC600)**(-0.5)
              ! convert from cm hr-1 to m s-1:
              kw=kw/(1.d2*SECONDS_PER_HOUR)

              ! Get the air-side transfer velocity based on Jeffrey et
              ! al 2010 (TODO: Read). Here I think ka is already m s-1.
              cd=1.d-3*(0.61d0+0.063d0*U10) ! eq 11 Johnson
              ustar=U10*SQRT(cd) ! eq 13 Johnson
              ra=13.3d0*SQRT(SC)+cd**(-0.5)-5.d0+LOG(SC)/(2.d0*vk) ! eq 14 Johnson, demon
              ka=1.d-3+(ustar/ra) ! See Johnson supplument R code K_calcs_Johnson_OS.R

              ! Get the total transfer velocity from point of view of
              ! air from Johnson 2010 Eq 3 or Liss & Slater 1974 eq 9:
              ! TODO: confirm units of KH0 as I am doing m/s here unlike geos-chem:
              K=1.d0/((1.d0/kw) + (1.d0/(KH0*ka)))

              ! Get the conentration of species in water; converting from
              ! mole L-1 to kg m-3. In the conversion:
              ! factors of 1e2*1e2*1e2 in numerator [cm-3 --> m-3] cancel
              ! factors of 1e3*1e3 in denominator [g L-1 --> kg cm-3]
              ! which just leaves the molecular weight [g mole-1]:
              CW=species(n)%conc*species(n)%mw

              ! Canclulate water-to-air flux in kg m-2 s-1, prorated by
              ! ocean fraction:
              waterToAir=CW*K*openOcean

              ! Get the tracer concentrationin kg m-3:
              !TODO: Turns out we already have this in an array, but
              ! it's a diagnostic array, so think of any implications of that...
              CA=trcSurfByVol(i,j,nTracer)

              ! Canclulate air-to-water flux in kg m-2 s-1, prorated by
              ! ocean fraction (this doesn't go anywhere; it's just a
              ! sink from the atmosphere):
              ! TODO: understand the /KH0 bit.
              ! TODO: understand why goes-chem V9.X implements sink as exp decay
              airToWater=openOcean*CA*K/KH0

              ! In effort to avoid negative tracer, don't let the sink
              ! part pull all of tracer out of L=1:
              airToWater=MIN(airToWater,
     &              sinkFraction*trm(i,j,1,nTracer)*bydtsrc)

              ! diagnose source and sink separately; so far only for Acetone:
              if(trim(species(n)%itsname)=='OcnACTO_src') then
                taijs(i,j,ijs_AcetOtoA) = taijs(i,j,ijs_AcetOtoA) +
     &                                    waterToAir
                taijs(i,j,ijs_AcetAtoO) = taijs(i,j,ijs_AcetAtoO) +
     &                                    airToWater
              end if

              ! Save net flux density (kg m-2 s-1) to be applied
              ! outside this routine:
              sfc_src(i,j,nTracer,ns)=waterToAir-airToWater

              cycle sources_loop ! done with this tracer's source
              ! TODO: could that be made cycle tracers_loop?

            end if ! matching species to tracer source name

          end do species_loop

          write(message,*) 'Ocean species '//
     &    trim(trc%surfaceSources(ns)%sourceName)//' not found.'
          call stop_model(trim(message),255)

        end do sources_loop

        if(source_count > 1) then
          write(message,*) 'More than one ocean source found for '//
     &    trim(trc%surfaceSources(ns)%sourceName)
          call stop_model(trim(message),255)
        end if

      end do tracers_loop

      end subroutine oceanEmissions_drv


      subroutine getSchmidt(this,TC,SC)
      !@sub getSchmidt Obtain salinity-independent (for now)
      !@+ dimensionless Schmidt Number for passed-in species.
      !@auth Greg Faluvegi
      use oceanEmissions, only: oceanSpecies
      use constant, only: tf,rhows ! rhows is avg sfc density in kg m-3
      implicit none
      type(oceanSpecies), intent(IN) :: this
      real*8, intent(OUT) :: SC
      real*8, intent(IN) :: TC ! in deg C
      real*8 :: TK
      real*8 :: Dw ! diffusion coeff of Hayduk & Minhas 1982. See Johnson.
      real*8 :: vw ! kinematic viscocity of water
      real*8 :: estar ! epsilon star
      real*8 :: Ns ! Eta, dynamic viscocity of the solvent
      ! Hardy 1953 method dynamic viscocity, which is only temperature-
      ! dependent:
      TK=TC+tf
      Ns=1.787d0*1.052d0/(1.d0+(0.03338d0*TC)+(0.00018325d0*TC*TC))
      ! Johnson et al 2010 Eq 35:
      estar=(9.58d0/this%Vb)-1.12d0
      ! Johnson et al 2010 Eq 34:
      Dw=1.25d-8 * TK**(1.52) * Ns**estar * (this%Vb**(-0.19)-0.292d0)
      ! I believe Dw above is in cm2 s-1. We want the Schmidt number
      ! here to be dimensionless, so we want vw below also in cm2 s-1.
      ! Well, Ns is in centipoises or [0.01 g cm-1 s-1.]
      ! And rhows is in kg m-3.
      ! So, rhows*1d3*1.d-2*1.d-2*1.d-2 (or just 1d-3) would get to
      ! g cm-3. And Ns*1.d-2 would get to g cm-1 s-1. Which would get
      ! a ratio of cm2 s-1. So, conversion factors work out to
      ! a 1.d-2 on top and 1.d-3 on bottoms, or just a factor of 10.:
      vw=10.d0*Ns/rhows ! TODO: can we use non-constant density?
      SC=vw/Dw ! dimensionless.
      end subroutine getSchmidt


      subroutine init_oceanEmissions
      !@sum init_oceanEmissions initialize some properties for
      !@+ oceanEmissions_drv at startup
      !@auth Greg Faluvegi
      use oceanEmissionsSpecies, only : acetone
      use OldTracer_mod, only: tr_mm
      use tracer_com, only: n_acetone
      implicit none
      acetone%itsname='OcnACTO_src'
      acetone%KHS=2.7d1 ! pg 46 Sanders 1999 (Benkelberg et al 1995)
      acetone%TDS=5.3d3 ! pg 46 Sanders 1999 (Benkelberg et al 1995)
      acetone%conc=15.d-9 ! mole/L TODO: add justification
      acetone%mw=tr_mm(n_acetone)
      acetone%Vb=77.6d0 ! from compounds.dat of Johnson
      end subroutine init_oceanEmissions
#endif /* TRACERS_ACETONE */
