#include "rundeck_opts.h"

      module flammability_com
!@sum for routines to calculate flammability potential of surface 
!@+   vegetation. Optionally also altering tracer biomass sources.
!@auth Greg Faluvegi based on direction from Olga Pechony including
!@+ her document Flammability.doc
      use timestream_mod, only : timestream
!@ Modified by Keren Mezuman
!@var ij_flamV indicies for aij output 
      use ent_const, only : N_COVERTYPES

      implicit none
      save

! main variables:
!@var flammability unitless flammability coefficient
!@var vegetation density for purpose of flammability calculation
      real*8, allocatable, dimension(:,:) :: flammability,veg_density
!@param mfcc MODIS fire count calibration (goes with EPFC by veg type
!@+ params.) Units=fires/m2/yr when multiplied by the unitless flammability
      real*8, parameter :: mfcc=2.2d-5
      real*8, allocatable, dimension(:,:) :: EPFCByVegType 
! rest is for the running average:
!@var maxHR_prec maximum number of sub-daily accumulations
!@param nday_prec number of days in running average for prec
!@param nday_lai number of days in running average for Ent LAI
!@var DRAfl daily running average of model prec for flammability
!@var ravg_prec period running average of model prec for flammability
!@var PRSfl period running sum of model prec for flammability
!@var HRAfl hourly running average of model prec for flammability
!@var iHfl "hourly" index for averages of model prec
!@var iDfl "daily"  index for averages of model prec
!@var i0fl ponter to current index in running sum of model prec
!@var first_prec whether in the first model averaging per. for prec
      integer, parameter :: nday_prec=30, nday_lai=365 ! unhappy with this making a large I,J array

!===============================================================================
! rundeck controls of pyrE
!===============================================================================
!@dbparam whichEPFCs chooses EPFCByVegType calibration:
!@+ 1=GFED4s, 2=GFED3, 3=GFED2, 4=MOPITT
      integer :: whichEPFCs=1
!@dbparam allowFlammabilityReinit (default 1=YES) allows the
!@+ averaging period to start from the beginning (thus no emissions
!@+ for nday_prec days.) Override this with allowFlammabilityReinit=0
!@+ in the rundeck, and values from the AIC file should be used (if 
!@+ present.) 
      integer :: allowFlammabilityReinit = 1
!@dbparam limit_barren_flammability Threshold ratio over which bare soil should
!@+ become 100%, to avoid using unrealistic LAI and vegetation density in
!@+ deserts due to crops+pasture cover. Default value is 0.8 (80%).
      real*8 :: limit_barren_flammability=0.8d0
!@dbparam anthropogenic_fire_model If true, human ignition and suppression
!@+ is taken into account in the calculations, based on the population density.
      logical :: anthropogenic_fire_model=.true.
!@dbparam tuneFireCount Fire count tuning factor, currently based on MODIS.
!@+ Olga says: "While theoretically this is supposed to give the absolute number
!@+ of fire counts, this is not actually so, since (A) We don't know the real 
!@+ lightning flash rate and we don't know the real number of fire human-caused
!@+ ignitions and so we can not really calibrate the number of fire ignitions 
!@+ to reflect the true absolute values.  (B) We don't know the real fire counts
!@+ and we don't know if MODIS fire counts are any closer to them in absolute
!@+ values. And since we want to use the EPFCs derived relying on the MODIS fire
!@+ counts, we'll still need to calibrate the modeled fire counts to be
!@+ comparable with the MODIS absolute values." The following is that tuning
!@+ parameter:
      real*8 :: tuneFireCount=30.d0
!@dbparam tuneCtoGlightning Cloud-to-ground lightning tuning factor. A factor
!@+ of 0.1 is necessary to reduce overestimation compared to WWLLN data.
      real*8 :: tuneCtoGlightning=0.1d0
!@dbparam tuneBurntArea Burnt area tuning factor, currently based on GFED4s.
      real*8 :: tuneBurntArea=250.d0
!===============================================================================

      real*8, allocatable, dimension(:,:,:):: DRAfl,burnt_area
      real*8, allocatable, dimension(:,:)  :: ravg_prec,PRSfl,iHfl,iDfl,
     &                                        i0fl
      logical, allocatable, dimension(:,:):: first_prec
      real*8, allocatable, dimension(:,:,:):: HRAfl
      integer :: maxHR_prec
      real*8, allocatable, dimension(:,:,:):: DRAlai
      real*8, allocatable, dimension(:,:):: ravg_lai,PRSlai,iHlai,iDlai,
     &                                      i0lai
      logical, allocatable, dimension(:,:):: first_lai
      real*8, allocatable, dimension(:,:,:):: HRAlai
      integer :: maxHR_lai
      type(timestream):: popDensStream
      logical :: firstPopDensStream=.true.
      real*8, allocatable, dimension(:,:) :: populationDensity
      real*8, allocatable, dimension(:,:) :: nonSuppressFrac
      real*8, allocatable, dimension(:,:) :: saveFireCount

      end module flammability_com


      subroutine alloc_flammability(grid)
!@SUM  alllocates arrays whose sizes need to be determined
!@+    at run-time
!@auth Greg Faluvegi
#ifdef TRACERS_ON
      use TRACER_COM, only: ntm
#endif  /* TRACERS_ON */
      use ent_const, only : N_COVERTYPES
      use domain_decomp_atm, only: dist_grid, getDomainBounds
      use dictionary_mod, only : get_param, is_set_param
      use model_com, only: dtsrc
      use TimeConstants_mod, only: SECONDS_PER_HOUR, HOURS_PER_DAY
      use flammability_com, only: flammability,veg_density,
     & first_prec,iHfl,iDfl,i0fl,DRAfl,ravg_prec,PRSfl,HRAfl,
     & nday_prec,maxHR_prec,burnt_area,EPFCByVegType
      use flammability_com, only: 
     & first_lai,iHlai,iDlai,i0lai,DRAlai,ravg_lai,PRSlai,HRAlai,
     & nday_lai,maxHR_lai
      use flammability_com, only: populationDensity
      use flammability_com, only: nonSuppressFrac
      use flammability_com, only: saveFireCount

      implicit none

      real*8 :: DTsrc_LOCAL
      type (dist_grid), intent(in) :: grid
      integer :: ier, J_1, J_0, I_1, I_0

      call getDomainBounds(grid,J_STRT=J_0,J_STOP=J_1,
     &                          I_STRT=I_0,I_STOP=I_1)

      ! maxHR_prec is to be defined as the number of precip
      ! running-average steps that define one day. (The 1.d0
      ! in the line is left in to represent the number of calls 
      ! per DTsrc timestep. Same for Ent LAI:
      ! I believe the real DTsrc is not available yet so I use
      ! a local copy from the database:

      DTsrc_LOCAL = DTsrc
      if(is_set_param("DTsrc"))call get_param("DTsrc",DTsrc_LOCAL)
      maxHR_prec = NINT(HOURS_PER_DAY*1.d0*SECONDS_PER_HOUR/DTsrc_LOCAL)
      maxHR_lai  = NINT(HOURS_PER_DAY*1.d0*SECONDS_PER_HOUR/DTsrc_LOCAL)

      allocate( flammability(I_0:I_1,J_0:J_1) )
      allocate( veg_density (I_0:I_1,J_0:J_1) )
      allocate( first_prec  (I_0:I_1,J_0:J_1) )
      allocate( iHfl        (I_0:I_1,J_0:J_1) )
      allocate( iDfl        (I_0:I_1,J_0:J_1) )
      allocate( i0fl        (I_0:I_1,J_0:J_1) )
      allocate( DRAfl       (nday_prec,I_0:I_1,J_0:J_1) )
      allocate( ravg_prec   (I_0:I_1,J_0:J_1) )
#ifdef TRACERS_ON
      allocate( EPFCByVegType  (N_COVERTYPES,ntm) )
#endif  /* TRACERS_ON */
      allocate( burnt_area  (N_COVERTYPES,I_0:I_1,J_0:J_1) )
      allocate( PRSfl       (I_0:I_1,J_0:J_1) )
      allocate( HRAfl       (maxHR_prec,I_0:I_1,J_0:J_1) )
      allocate( first_lai   (I_0:I_1,J_0:J_1) )
      allocate( iHlai       (I_0:I_1,J_0:J_1) )
      allocate( iDlai       (I_0:I_1,J_0:J_1) )
      allocate( i0lai       (I_0:I_1,J_0:J_1) )
      allocate( DRAlai      (nday_lai,I_0:I_1,J_0:J_1) )
      allocate( ravg_lai    (I_0:I_1,J_0:J_1) )
      allocate( PRSlai      (I_0:I_1,J_0:J_1) )
      allocate( HRAlai      (maxHR_lai,I_0:I_1,J_0:J_1) )
      allocate( nonSuppressFrac(I_0:I_1,J_0:J_1) )
      allocate( populationDensity(I_0:I_1,J_0:J_1) )
      allocate( saveFireCount    (I_0:I_1,J_0:J_1) )

      return
      end subroutine alloc_flammability


      subroutine init_flammability
!@sum initialize flamability, veg density, etc. for fire model
!@auth Greg Faluvegi based on direction from Olga Pechony
      use model_com, only: Itime,ItimeI
#ifdef TRACERS_ON
      use OldTracer_mod, only: trname
      use TRACER_COM, only: ntm
      use flammability_com, only: whichEPFCs
#endif  /* TRACERS_ON */
      use ent_const, only: N_COVERTYPES
      use ent_pfts, only: ent_cover_names
      use dictionary_mod, only: sync_param
      use flammability_com, only: limit_barren_flammability
      use flammability_com, only: anthropogenic_fire_model
      use flammability_com, only: first_prec
     & ,allowFlammabilityReinit,DRAfl,hrafl,prsfl,i0fl,iDfl,iHfl
     & ,ravg_prec,burnt_area,EPFCByVegType
      use flammability_com, only: first_lai,DRAlai,HRAlai,PRSlai,i0lai
     & ,iDlai,iHlai,ravg_lai
      use domain_decomp_atm,only: am_i_root, readt_parallel
      use filemanager, only: openunit, closeunit, nameunit

      implicit none
      character*80 :: title,fname

      integer :: iu_data, n, v, tracerIndex
!!    EPFCByVegType tracer emissions per fire count as a
!!    function of ENT vegetation types

      call sync_param
     &("allowFlammabilityReinit",allowFlammabilityReinit)

      if( (Itime==ItimeI .and. allowFlammabilityReinit==1) .or. 
     &  allowFlammabilityReinit == -1 )then
        first_prec(:,:)=.true.
        burnt_area(:,:,:)=0.d0
        DRAfl=0.d0
        hrafl=0.d0
        prsfl=0.d0
        i0fl=0.d0
        iDfl=0.d0
        iHfl=0.d0
        ravg_prec=0.d0
        first_lai(:,:)=.true.
        DRAlai=0.d0
        HRAlai=0.d0
        PRSlai=0.d0
        i0lai=0.d0
        iDlai=0.d0
        iHlai=0.d0
        ravg_lai=0.d0
      end if

      call sync_param("limit_barren_flammability",
     &                 limit_barren_flammability)
      call sync_param("anthropogenic_fire_model",
     &                 anthropogenic_fire_model)
#ifdef TRACERS_ON
      !loop over tracers and have a select case for every tracer
      ! have multiple tracers at once e.g. BCB, M_BC1_BC, tomas one..
      ! make sure all the variables below are well defined at the
      ! beginning of the subroutine
      call sync_param("whichEPFCs",whichEPFCs)
      EPFCByVegType=0.d0
      select case(whichEPFCs)!kgC/#fire/(vegfrac in gridbox)
        case(1) ! GFED4s 2001-2009
          do n=1,ntm
          select case(trname(n))
            case('M_BC1_BC','BCB')
              do v=1,N_COVERTYPES
              select case(ent_cover_names(v))
                case('arid_shrub')
                  EPFCByVegType(v,n)=238.d0
                case('c3_grass_ann')
                  EPFCByVegType(v,n)=173.d0
                case('c3_grass_arct')
                  EPFCByVegType(v,n)=1159.d0
                case('c3_grass_per')
                  EPFCByVegType(v,n)=257.d0
                case('c4_grass')
                  EPFCByVegType(v,n)=726.d0
                case('cold_br_late')
                  EPFCByVegType(v,n)=767.d0
                case('cold_shrub')
                  EPFCByVegType(v,n)=357.d0
                case('decid_nd')
                  EPFCByVegType(v,n)=1844.d0
                case('drought_br')
                  EPFCByVegType(v,n)=1382.d0
                case('ever_br_late')
                  EPFCByVegType(v,n)=1434.d0
                case('ever_nd_late')
                  EPFCByVegType(v,n)=821.d0
                case default
                  EPFCByVegType(v,n)=0.d0
              end select
              end do
            case('M_OCC_OC','OCB')
              do v=1,N_COVERTYPES
              select case(ent_cover_names(v))
                case('arid_shrub')
                  EPFCByVegType(v,n)=1479.d0
                case('c3_grass_ann')
                  EPFCByVegType(v,n)=728.d0
                case('c3_grass_arct')
                  EPFCByVegType(v,n)=15551.d0
                case('c3_grass_per')
                  EPFCByVegType(v,n)=1504.d0
                case('c4_grass')
                  EPFCByVegType(v,n)=4339.d0
                case('cold_br_late')
                  EPFCByVegType(v,n)=3437.d0
                case('cold_shrub')
                  EPFCByVegType(v,n)=6562.d0
                case('decid_nd')
                  EPFCByVegType(v,n)=36753.d0
                case('drought_br')
                  EPFCByVegType(v,n)=10667.d0
                case('ever_br_late')
                  EPFCByVegType(v,n)=10941.d0
                case('ever_nd_late')
                  EPFCByVegType(v,n)=6537.d0
                case default
                  EPFCByVegType(v,n)=0.d0
              end select
              end do
            case('NOx') 
              do v=1,N_COVERTYPES
              select case(ent_cover_names(v))
                case('arid_shrub')
                  EPFCByVegType(v,n)=1009.d0
                case('c3_grass_ann')
                  EPFCByVegType(v,n)=690.d0
                case('c3_grass_arct')
                  EPFCByVegType(v,n)=1094.d0
                case('c3_grass_per')
                  EPFCByVegType(v,n)=908.d0
                case('c4_grass')
                  EPFCByVegType(v,n)=3152.d0
                case('cold_br_late')
                  EPFCByVegType(v,n)=1529.d0
                case('cold_shrub')
                  EPFCByVegType(v,n)=241.d0
                case('decid_nd')
                  EPFCByVegType(v,n)=1559.d0
                case('drought_br')
                  EPFCByVegType(v,n)=4835.d0
                case('ever_br_late')
                  EPFCByVegType(v,n)=4905.d0
                case('ever_nd_late')
                  EPFCByVegType(v,n)=1197.d0
                case default
                  EPFCByVegType(v,n)=0.d0
              end select
              end do
            case('CO')
              do v=1,N_COVERTYPES
              select case(ent_cover_names(v))
                case('arid_shrub')
                  EPFCByVegType(v,n)=39268.d0
                case('c3_grass_ann')
                  EPFCByVegType(v,n)=26761.d0
                case('c3_grass_arct')
                  EPFCByVegType(v,n)=251702.d0
                case('c3_grass_per')
                  EPFCByVegType(v,n)=41043.d0
                case('c4_grass')
                  EPFCByVegType(v,n)=117577.d0
                case('cold_br_late')
                  EPFCByVegType(v,n)=113392.d0
                case('cold_shrub')
                  EPFCByVegType(v,n)=105936.d0
                case('decid_nd')
                  EPFCByVegType(v,n)=481485.d0
                case('drought_br')
                  EPFCByVegType(v,n)=230829.d0
                case('ever_br_late')
                  EPFCByVegType(v,n)=249906.d0
                case('ever_nd_late')
                  EPFCByVegType(v,n)=146622.d0
                case default
                  EPFCByVegType(v,n)=0.d0
              end select
              end do
            case('Alkenes')
              do v=1,N_COVERTYPES
              select case(ent_cover_names(v))
                case('arid_shrub')
                  EPFCByVegType(v,n)=36.6d0
                case('c3_grass_ann')
                  EPFCByVegType(v,n)=25.1d0
                case('c3_grass_arct')
                  EPFCByVegType(v,n)=489.d0
                case('c3_grass_per')
                  EPFCByVegType(v,n)=38.8d0
                case('c4_grass')
                  EPFCByVegType(v,n)=110.d0
                case('cold_br_late')
                  EPFCByVegType(v,n)=106.d0
                case('cold_shrub')
                  EPFCByVegType(v,n)=104.d0
                case('decid_nd')
                  EPFCByVegType(v,n)=422.d0
                case('drought_br')
                  EPFCByVegType(v,n)=214.d0
                case('ever_br_late')
                  EPFCByVegType(v,n)=220.d0
                case('ever_nd_late')
                  EPFCByVegType(v,n)=137.d0
                case default
                  EPFCByVegType(v,n)=0.d0
              end select
              end do
            case('Paraffin')
              do v=1,N_COVERTYPES
              select case(ent_cover_names(v))
                case('arid_shrub')
                  EPFCByVegType(v,n)=18.5d0
                case('c3_grass_ann')
                  EPFCByVegType(v,n)=13.9d0
                case('c3_grass_arct')
                  EPFCByVegType(v,n)=226.d0
                case('c3_grass_per')
                  EPFCByVegType(v,n)=20.7d0
                case('c4_grass')
                  EPFCByVegType(v,n)=57.0d0
                case('cold_br_late')
                  EPFCByVegType(v,n)=69.8d0
                case('cold_shrub')
                  EPFCByVegType(v,n)=72.1d0
                case('decid_nd')
                  EPFCByVegType(v,n)=373.d0
                case('drought_br')
                  EPFCByVegType(v,n)=108.d0
                case('ever_br_late')
                  EPFCByVegType(v,n)=102.d0
                case('ever_nd_late')
                  EPFCByVegType(v,n)=89.1d0
                case default
                  EPFCByVegType(v,n)=0.d0
              end select
              end do
            case('SO2')
              do v=1,N_COVERTYPES
              select case(ent_cover_names(v))
                case('arid_shrub')
                  EPFCByVegType(v,n)=262.d0
                case('c3_grass_ann')
                  EPFCByVegType(v,n)=147.d0
                case('c3_grass_arct')
                  EPFCByVegType(v,n)=2315.d0
                case('c3_grass_per')
                  EPFCByVegType(v,n)=270.d0
                case('c4_grass')
                  EPFCByVegType(v,n)=795.d0
                case('cold_br_late')
                  EPFCByVegType(v,n)=555.d0
                case('cold_shrub')
                  EPFCByVegType(v,n)=878.d0
                case('decid_nd')
                  EPFCByVegType(v,n)=4168.d0
                case('drought_br')
                  EPFCByVegType(v,n)=1687.d0
                case('ever_br_late')
                  EPFCByVegType(v,n)=1438.d0
                case('ever_nd_late')
                  EPFCByVegType(v,n)=972.d0
                case default
                  EPFCByVegType(v,n)=0.d0
              end select
              end do
            case('NH3')
              do v=1,N_COVERTYPES
              select case(ent_cover_names(v))
                case('arid_shrub')
                  EPFCByVegType(v,n)=378.d0
                case('c3_grass_ann')
                  EPFCByVegType(v,n)=313.d0
                case('c3_grass_arct')
                  EPFCByVegType(v,n)=5065.d0
                case('c3_grass_per')
                  EPFCByVegType(v,n)=438.d0
                case('c4_grass')
                  EPFCByVegType(v,n)=1196.d0
                case('cold_br_late')
                  EPFCByVegType(v,n)=2101.d0
                case('cold_shrub')
                  EPFCByVegType(v,n)=2006.d0
                case('decid_nd')
                  EPFCByVegType(v,n)=10722.d0
                case('drought_br')
                  EPFCByVegType(v,n)=2340.d0
                case('ever_br_late')
                  EPFCByVegType(v,n)=2847.d0
                case('ever_nd_late')
                  EPFCByVegType(v,n)=2277.d0
                case default
                  EPFCByVegType(v,n)=0.d0
              end select
              end do
            case('CH4')
              do v=1,N_COVERTYPES
              select case(ent_cover_names(v))
                case('arid_shrub')
                  EPFCByVegType(v,n)=1351.d0
                case('c3_grass_ann')
                  EPFCByVegType(v,n)=1013.d0
                case('c3_grass_arct')
                  EPFCByVegType(v,n)=11574.d0
                case('c3_grass_per')
                  EPFCByVegType(v,n)=1494.d0
                case('c4_grass')
                  EPFCByVegType(v,n)=4113.d0
                case('cold_br_late')
                  EPFCByVegType(v,n)=5915.d0
                case('cold_shrub')
                  EPFCByVegType(v,n)=4505.d0
                case('decid_nd')
                  EPFCByVegType(v,n)=22977.d0
                case('drought_br')
                  EPFCByVegType(v,n)=8461.d0
                case('ever_br_late')
                  EPFCByVegType(v,n)=10477.d0
                case('ever_nd_late')
                  EPFCByVegType(v,n)=6863.d0
                case default
                  EPFCByVegType(v,n)=0.d0
              end select
              end do
          end select
          end do
        case default
          call stop_model('whichEPFCs unknown',255)
      end select
#endif  /* TRACERS_ON */

      return
      end subroutine init_flammability


      subroutine def_rsf_flammability(fid)
!@sum  def_rsf_flammability defines flammability array structure in 
!@+    restart files
!@auth Greg Faluvegi (directly from M. Kelley's def_rsf_lakes)
!@ver  beta
      use flammability_com
      use domain_decomp_atm, only : grid
      use pario, only : defvar
      implicit none
      integer fid   !@var fid file id

      call defvar(grid,fid,drafl,'drafl(nday_prec,dist_im,dist_jm)')
      call defvar(grid,fid,hrafl,'hrafl(maxHR_prec,dist_im,dist_jm)')
      call defvar(grid,fid,prsfl,'prsfl(dist_im,dist_jm)')
      call defvar(grid,fid,i0fl,'i0fl(dist_im,dist_jm)') ! real
      call defvar(grid,fid,iDfl,'iDfl(dist_im,dist_jm)') ! real
      call defvar(grid,fid,iHfl,'iHfl(dist_im,dist_jm)') ! real
      call defvar(grid,fid,first_prec,'first_prec(dist_im,dist_jm)')
      call defvar(grid,fid,ravg_prec,'ravg_prec(dist_im,dist_jm)')
      call defvar(grid,fid,saveFireCount,'saveFireCount(dist_im,
     &     dist_jm)')
      call defvar(grid,fid,burnt_area,'burnt_area(N_COVERTYPES,dist_im,
     &     dist_jm)')
      call defvar(grid,fid,dralai,'dralai(nday_lai,dist_im,dist_jm)')
      call defvar(grid,fid,hralai,'hralai(maxHR_lai,dist_im,dist_jm)')
      call defvar(grid,fid,prslai,'prslai(dist_im,dist_jm)')
      call defvar(grid,fid,i0lai,'i0lai(dist_im,dist_jm)') ! real
      call defvar(grid,fid,iDlai,'iDlai(dist_im,dist_jm)') ! real
      call defvar(grid,fid,iHlai,'iHlai(dist_im,dist_jm)') ! real
      call defvar(grid,fid,first_lai,'first_lai(dist_im,dist_jm)')
      call defvar(grid,fid,ravg_lai,'ravg_lai(dist_im,dist_jm)')

      return
      end subroutine def_rsf_flammability

      subroutine new_io_flammability(fid,iaction)
!@sum  new_io_flammability read/write arrays from/to restart files
!@auth Greg Faluvegi (directly from M. Kelley's new_io_lakes)
!@ver  beta new_ prefix avoids name clash with the default version
      use model_com, only : ioread,iowrite
      use domain_decomp_atm, only : grid
      use pario, only : write_dist_data,read_dist_data
      use flammability_com
      implicit none
!@var fid unit number of read/write
      integer fid   
!@var iaction flag for reading or writing to file
      integer iaction 
      select case (iaction)
      case (iowrite)            ! output to restart file
        call write_dist_data(grid, fid, 'drafl', drafl, jdim=3 )
        call write_dist_data(grid, fid, 'hrafl', hrafl, jdim=3 )
        call write_dist_data(grid, fid, 'prsfl', prsfl )
        call write_dist_data(grid, fid, 'i0fl', i0fl )
        call write_dist_data(grid, fid, 'iDfl', iDfl )
        call write_dist_data(grid, fid, 'iHfl', iHfl )
        call write_dist_data(grid, fid, 'first_prec', first_prec )
        call write_dist_data(grid, fid, 'ravg_prec', ravg_prec )
        call write_dist_data(grid, fid, 'saveFireCount',saveFireCount)
        call write_dist_data(grid, fid, 'burnt_area',burnt_area,jdim=3 )
        call write_dist_data(grid, fid, 'dralai', dralai, jdim=3 )
        call write_dist_data(grid, fid, 'hralai', hralai, jdim=3 )
        call write_dist_data(grid, fid, 'prslai', prslai )
        call write_dist_data(grid, fid, 'i0lai', i0lai )
        call write_dist_data(grid, fid, 'iDlai', iDlai )
        call write_dist_data(grid, fid, 'iHlai', iHlai )
        call write_dist_data(grid, fid, 'first_lai', first_lai )
        call write_dist_data(grid, fid, 'ravg_lai', ravg_lai )

      case (ioread)            ! input from restart file
        call read_dist_data(grid, fid, 'drafl', drafl, jdim=3 )
        call read_dist_data(grid, fid, 'hrafl', hrafl, jdim=3 )
        call read_dist_data(grid, fid, 'prsfl', prsfl )
        call read_dist_data(grid, fid, 'i0fl', i0fl )
        call read_dist_data(grid, fid, 'iDfl', iDfl )
        call read_dist_data(grid, fid, 'iHfl', iHfl )
        call read_dist_data(grid, fid, 'first_prec', first_prec )
        call read_dist_data(grid, fid, 'ravg_prec', ravg_prec )
        call read_dist_data(grid, fid, 'saveFireCount',saveFireCount)
        call read_dist_data(grid, fid, 'burnt_area',burnt_area,jdim=3 )
        call read_dist_data(grid, fid, 'dralai', dralai, jdim=3 )
        call read_dist_data(grid, fid, 'hralai', hralai, jdim=3 )
        call read_dist_data(grid, fid, 'prslai', prslai )
        call read_dist_data(grid, fid, 'i0lai', i0lai )
        call read_dist_data(grid, fid, 'iDlai', iDlai )
        call read_dist_data(grid, fid, 'iHlai', iHlai )
        call read_dist_data(grid, fid, 'first_lai', first_lai )
        call read_dist_data(grid, fid, 'ravg_lai', ravg_lai )
      end select
      return
      end subroutine new_io_flammability

      subroutine flammability_drv
!@sum driver routine for flammability potential of surface
!@+   vegetation calculation.
!@auth Greg Faluvegi based on direction from Olga Pechony
!@+ Later modified by Keren Mezuman
!@ver  1.0 
      use geom, only: axyp
      use model_com, only: dtsrc
      use atm_com, only : pedn
      use domain_decomp_atm,only: grid, getDomainBounds
      use flammability_com, only: limit_barren_flammability
      use flammability_com, only: anthropogenic_fire_model
      use flammability_com, only: tuneCtoGlightning
      use flammability_com, only: flammability,veg_density,ravg_prec,
     & iHfl,iDfl,i0fl,first_prec,HRAfl,DRAfl,PRSfl
     & ,saveFireCount,burnt_area
      use model_com, only: modelEclock
      use model_com, only: master_yr
      use lightning, only : CG_DENS 

      use fluxes, only: prec,atmsrf
      use constant, only: lhe
      use TimeConstants_mod, only: SECONDS_PER_DAY
      use diag_com, only: ij_flamm,ij_flamm_prec,ij_fvden,ij_fireCount
      use diag_com, only: ij_flamV
      use diag_com, only: aij=>aij_loc
      use flammability_com, only: ravg_lai,iHlai,iDlai,i0lai,
     & first_lai,HRAlai,DRAlai,PRSlai
      use ent_const, only : N_COVERTYPES
      use ent_pfts, only: ent_cover_names
      use ghy_com, only: fearth
      use diag_com, only: ij_ba_tree,ij_ba_shrub,ij_ba_grass
      use ent_com, only: entcells
      use ent_mod, only: ent_get_exports
#ifdef CACHED_SUBDD
      use subdd_mod, only : subdd_groups,subdd_ngroups,subdd_type
     &     ,inc_subdd,find_groups
#endif

      implicit none

#ifdef CACHED_SUBDD
      integer :: igrp,ngroups,grpids(subdd_ngroups),k
      type(subdd_type), pointer :: subdd
      real*8, dimension(grid%i_strt_halo:grid%i_stop_halo,
     &                  grid%j_strt_halo:grid%j_stop_halo) :: sddarr2d
#endif
      integer :: J_0, J_1, I_0, I_1, i, j, v
      integer :: xyear,xday
      real*8 :: qsat ! this is a function in UTILDBL.f
      real*8 :: tsurf,qsurf
      real*8 :: RH1,wsurf,fearth_axyp
      ! the 7.9 here was from running a year or two under 2005 conditions
      ! and seeing what was the maximum LAI returned by Ent. Therefore,
      ! under other climate conditions, the vegetation density may reach > 1.0. 
      ! Olga thought this would not cause any mathematical probems. 
      ! I am however limiting the flammability to 1.0, but normally its
      ! values seem to be much much lower than that anyway...
      ! This seemed to result in too much emissions, so trying 10.0, which
      ! is the MODIS-based number Olga used in her original offline VEG_DENS
      ! file: 
      real*8, parameter :: byLaiMax=1.d0/10.0d0 !! 7.9d0
      real*8 :: lai
!@var pvt fraction vegetation type for N_COVERTYPES (fraction)
!@var N_COVERTYPES = N_PFT + N_SOILCOV + N_OTHER = 16 + 2 + 0
      real*8, dimension(N_COVERTYPES):: pvt
      real*8 :: fracVegNonCrops, fracBare
!@var ba_sum Sum over all surface types of burnt area in a gridbox [m2]
      real*8 :: ba_sum
      real*8 :: prec2use

      call getDomainBounds(grid,J_STRT=J_0,J_STOP=J_1,
     &                          I_STRT=I_0,I_STOP=I_1)

      if (anthropogenic_fire_model) then
        call modelEclock%get(year=xyear, dayOfYear=xday)
        if (master_yr/=0) xyear=master_yr
        call readFlamPopDens(xyear, xday)
      endif

      do j=J_0,J_1
        do i=I_0,I_1

! get some input from Ent
          if (fearth(i,j)>0.d0) then
            call ent_get_exports( entcells(i,j),leaf_area_index=lai)
            call ent_get_exports(entcells(i,j),vegetation_fractions=pvt)
          else
            lai=0.d0
            pvt(:)=0.d0
          endif

! update the precipitation running average
          call prec_running_average(prec(i,j),ravg_prec(i,j), 
     &    iHfl(i,j),iDfl(i,j),i0fl(i,j),first_prec(i,j),HRAfl(:,i,j),
     &    DRAfl(:,i,j),PRSfl(i,j))

! update the LAI running average
! That is the lai from the last surface timestep only?
! (But Igor says this is OK as LAI is only computed once per day.)
          call lai_running_average(lai,ravg_lai(i,j), 
     &    iHlai(i,j),iDlai(i,j),i0lai(i,j),first_lai(i,j),HRAlai(:,i,j),
     &    DRAlai(:,i,j),PRSlai(i,j))

! calculate vegetation density
          if(.not.first_lai(i,j)) then
            veg_density(i,j) = ravg_lai(i,j)*byLaiMax*fearth(i,j)
          else
              ! for the first year of a run (since we don't have an annual 
              ! average yet, use the concurrent LAI):
            veg_density(i,j) =  lai*byLaiMax*fearth(i,j)
          end if

! calculate total vegetated and bare soil fractions
          fracVegNonCrops=0.d0
          fracBare=0.d0
          do v=1,N_COVERTYPES
            select case(ent_cover_names(v))
              case('bare_bright','bare_dark')
                fracBare = fracBare + pvt(v) * fearth(i,j) 
              case('c3_grass_ann','c3_grass_arct','c3_grass_per',
     &             'c4_grass','arid_shrub','cold_shrub',
     &             'cold_br_late','decid_nd','drought_br',
     &             'ever_br_late','ever_nd_late')
                fracVegNonCrops = fracVegNonCrops+pvt(v)*fearth(i,j)
            end select
          end do
          aij(i,j,ij_flamV)=aij(i,j,ij_flamV)+fracVegNonCrops
          if (fracBare + fracVegNonCrops > fearth(i,j)+0.00001d0) then
            call stop_model('cover types sum greater than fearth',255)
          endif

! restrict vegetation below the requested threshold
          if ((fracBare >= limit_barren_flammability).or.
     &        (fracVegNonCrops == 0.d0)) veg_density(i,j)=0.d0

          if(fearth(i,j)<=0.d0) cycle ! no flammability when there is no earth
!@var atmsrf contains atm-surf interaction
!@+ quantities averaged over all surface types
!@var WSAVG SURFACE WIND MAGNITUDE (m/s)
!@var wsurf surface wind velocity (m/s)
          wsurf=atmsrf%wsavg(i,j)
          tsurf=atmsrf%tsavg(i,j)![K]
          qsurf=atmsrf%qsavg(i,j)
!@var RH1 relative humidity in layer 1 (fraction)
          RH1=min(1.d0,qsurf/qsat(tsurf,lhe,pedn(1,i,j)))
!@var fearth soil covered land fraction (fraction)
!@var  axyp,byaxyp area of grid box (+inverse) (m^2)
          fearth_axyp=fearth(i,j)*axyp(i,j)

! The burnt area to be used for current flammability must be from the
! prevous time step, so calculated here instead of after calling step_ba.
          if(.not.first_prec(i,j)) then
            prec2use=ravg_prec(i,j)
          else
! use absolute precipitation while waiting for the full mean to accumulate
            prec2use=prec(i,j)
          endif
          ba_sum=sum(burnt_area(:,i,j))
          call calc_flammability(tsurf,SECONDS_PER_DAY
     &     *prec2use/dtsrc,RH1,veg_density(i,j),
     &     flammability(i,j),ba_sum,fearth_axyp)
          call calculate_fire_count(i,j)
            ! update the burnt area  average:
!@var burnt_area Burnt area (m^2)
!@var saveFireCount fire count rate (fire/m2/s)
          call step_ba(burnt_area(:,i,j),RH1,wsurf,
     &                 saveFireCount(i,j),pvt,fearth_axyp,i,j)

        end do
      end do

! update aij diagnostics
      aij(I_0:I_1,J_0:J_1,ij_flamm)=
     &  aij(I_0:I_1,J_0:J_1,ij_flamm)+flammability(:,:)
      where (first_prec)
        aij(I_0:I_1,J_0:J_1,ij_flamm_prec)=
     &    aij(I_0:I_1,J_0:J_1,ij_flamm_prec)+
     &    SECONDS_PER_DAY*prec(I_0:I_1,J_0:J_1)/dtsrc
      else where
        aij(I_0:I_1,J_0:J_1,ij_flamm_prec)=
     &    aij(I_0:I_1,J_0:J_1,ij_flamm_prec)+
     &    SECONDS_PER_DAY*ravg_prec(:,:)/dtsrc
      end where
      aij(I_0:I_1,J_0:J_1,ij_fvden)=
     &  aij(I_0:I_1,J_0:J_1,ij_fvden)+veg_density(:,:)
      aij(I_0:I_1,J_0:J_1,ij_fireCount)=
     &  aij(I_0:I_1,J_0:J_1,ij_fireCount)+saveFireCount(:,:)
      do v=1,N_COVERTYPES
        select case(ent_cover_names(v))
          case('arid_shrub','cold_shrub')
            aij(I_0:I_1,J_0:J_1,ij_ba_shrub)=
     &        aij(I_0:I_1,J_0:J_1,ij_ba_shrub)+burnt_area(v,:,:)
          case('c3_grass_ann','c3_grass_arct','c3_grass_per',
     &         'c4_grass')
            aij(I_0:I_1,J_0:J_1,ij_ba_grass)=
     &        aij(I_0:I_1,J_0:J_1,ij_ba_grass)+burnt_area(v,:,:)
          case('cold_br_late','decid_nd','drought_br',
     &         'ever_br_late','ever_nd_late')
            aij(I_0:I_1,J_0:J_1,ij_ba_tree)=
     &        aij(I_0:I_1,J_0:J_1,ij_ba_tree)+burnt_area(v,:,:)
        end select
      end do
  
#ifdef CACHED_SUBDD
****
**** Collect some high-frequency outputs
      call find_groups('aijh',grpids,ngroups)
      do igrp=1,ngroups
        subdd => subdd_groups(grpids(igrp))
        do k=1,subdd%ndiags
          select case (subdd%name(k))
          case ('FLAMM')
            sddarr2d(I_0:I_1,J_0:J_1) = flammability(:,:)
            call inc_subdd(subdd,k,sddarr2d)
          case ('FLAMM_prec')
            where (first_prec)
              sddarr2d(I_0:I_1,J_0:J_1) = 
     &          SECONDS_PER_DAY*prec(I_0:I_1,J_0:J_1)/dtsrc
            else where
              sddarr2d(I_0:I_1,J_0:J_1) = 
     &          SECONDS_PER_DAY*ravg_prec(:,:)/dtsrc
            end where
            call inc_subdd(subdd,k,sddarr2d)
          case ('FVDEN')
            sddarr2d(I_0:I_1,J_0:J_1) = veg_density(:,:)
            call inc_subdd(subdd,k,sddarr2d)
          case ('fireCount')
            sddarr2d(I_0:I_1,J_0:J_1) =saveFireCount(:,:)
            call inc_subdd(subdd,k,sddarr2d)
          case ('BA_grass')
            do v=1,N_COVERTYPES
              select case(ent_cover_names(v))
                case('c3_grass_ann','c3_grass_arct','c3_grass_per',
     &               'c4_grass')
                  sddarr2d(I_0:I_1,J_0:J_1) =burnt_area(v,:,:)
                  call inc_subdd(subdd,k,sddarr2d)
              end select
            end do
          case ('BA_shrub')
            do v=1,N_COVERTYPES
              select case(ent_cover_names(v))
                case('arid_shrub','cold_shrub')
                  sddarr2d(I_0:I_1,J_0:J_1) =burnt_area(v,:,:)
                  call inc_subdd(subdd,k,sddarr2d)
              end select
            end do
          case ('BA_tree')
            do v=1,N_COVERTYPES
              select case(ent_cover_names(v))
                case('cold_br_late','decid_nd','drought_br',
     &               'ever_br_late','ever_nd_late')
                  sddarr2d(I_0:I_1,J_0:J_1) =burnt_area(v,:,:)
                  call inc_subdd(subdd,k,sddarr2d)
              end select
            end do
          case ('f_ignCG')
            sddarr2d(I_0:I_1,J_0:J_1) =CG_DENS(:,:)*tuneCtoGlightning
            call inc_subdd(subdd,k,sddarr2d)
          end select
        enddo
      enddo
#endif
      return
      end subroutine flammability_drv

