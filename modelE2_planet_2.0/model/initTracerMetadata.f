#include "rundeck_opts.h"
!------------------------------------------------------------------------------
      subroutine setDefaultSpec(n, pTracer)
!------------------------------------------------------------------------------
      use Dictionary_mod, only: sync_param
      use RunTimeControls_mod, only: tracers_amp
      use RunTimeControls_mod, only: tracers_tomas
      use RunTimeControls_mod, only: tracers_aerosols_vbs
      use OldTracer_mod, only: trName, do_fire, do_aircraft
      use OldTracer_mod, only: set_do_fire, set_do_aircraft
      use OldTracer_mod, only: set_first_aircraft
      use OldTracer_mod, only: scale_aircraft, set_scale_aircraft
      use OldTracer_mod, only: set_do_rocket, set_first_rocket
      use OldTracer_mod, only: nBBsources, set_nBBsources
      use DOMAIN_DECOMP_ATM, only: am_i_root
      use TRACER_COM, only: tracers
      use TRACER_COM, only: set_ntsurfsrc, ntsurfsrc
      use Tracer_mod, only: ntsurfsrcmax
      use Tracer_mod, only: Tracer
      use Tracer_mod, only: findSurfaceSources
      use Tracer_mod, only: addSurfaceSource
      use TracerSurfaceSource_mod, only: itsMegan, itsCH4MGOL, itsOcean
      use filemanager, only : file_exists
#ifdef TRACERS_SPECIAL_Shindell
      use TRCHEM_Shindell_COM, only: use_rad_ch4
#endif
      implicit none

      integer, intent(in) :: n
      class (Tracer), pointer :: pTracer

      integer :: val

      call pTracer%insert('ntSurfSrc', 0)

!     The following section will check for rundeck file of
!     the form: trname_01, trname_02... and thereby define
!     the ntsurfsrc(n). If those files exist it reads
!     metadata to get information including the
!     source name (ssame-->{sname,lname,etc.}. ntsurfsrc(n)
!     get set to zero if those files aren't found.
!
!     general case:

      call findSurfaceSources(pTracer)

!     Next, check whether tracers have 3D aircraft source files/dirs:
      if(file_exists(trim(trname(n)//'_AIRC'))) then
        call set_do_aircraft(n, .true.)
        call set_first_aircraft(n, .true.)
      end if
!     and whether scalings were set up for those aircraft sources:
      if(do_aircraft(n))then
        if(file_exists(trim(trname(n)//'_AIRC_scale'))) then
          call set_scale_aircraft(n, .true.)
        end if
      end if
!     Also check whether tracers have 3D rocket source files/dirs:
      if(file_exists(trim(trname(n)//'_ROCKET'))) then
        call set_do_rocket(n, .true.)
        call set_first_rocket(n, .true.)
      end if

#ifdef DYNAMIC_BIOMASS_BURNING
!-------------------------------------------------------------------------------
!     allow some tracers to have biomass burning based on fire model:
!-------------------------------------------------------------------------------
      select case (trname(n))
        case('NOx','CO','HCHO','Alkenes','Paraffin',
     &       'BCB','OCB','NH3','SO2',
#ifdef TRACERS_AMP
     &       'M_BC1_BC','M_OCC_OC','M_ACC_SU','M_AKK_SU',
#endif
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
#ifdef TRACERS_dCO
     &       'd13Calke', 'd13CPAR',
#endif  /* TRACERS_dCO */
     &       'dC17O', 'dC18O', 'd13CO',
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
     &       'vbsAm2', 'vbsAm1', 'vbsAz',  'vbsAp1', 'vbsAp2',
     &       'vbsAp3', 'vbsAp4', 'vbsAp5', 'vbsAp6'
#ifdef TRACERS_TOMAS
     &       ,'AECOB_01','AOCOB_01' !BCB and OCB hygroscopities? Need to put emission into OB and IL.
#endif
     &       )
          call set_do_fire(n, .true.)
#ifdef TRACERS_SPECIAL_Shindell
        case('CH4') ! in here to avoid potential Lerner tracers conflict
          if(use_rad_ch4==0) call set_do_fire(n, .true.)
#endif
      end select
#endif /* DYNAMIC_BIOMASS_BURNING */

!-------------------------------------------------------------------------------
!     allow some tracers to have biomass burning sources that mix over
!     PBL layers (these become 3D sources no longer within ntsurfsrc(n)):
!-------------------------------------------------------------------------------
      val = nBBsources(n)
      call sync_param(trim(trname(n))//"_nBBsources",val)
      call set_nBBsources(n, val)
      if (nBBsources(n) > ntsurfsrc(n)) then
        call stop_model('nBBsources > ntsurfsrc',255)
      else
        call set_ntsurfsrc(n, ntsurfsrc(n)-nBBsources(n))
      endif

#ifdef DO_MEGAN
!-------------------------------------------------------------------------------
!     allow some tracers to have MEGAN-base vegetation emissions
!-------------------------------------------------------------------------------
!     These will be accounted among the surface sources. Tag any such
!     source as skipReason=itsMegan, so this can be used inside MEGAN to
!     fill in the sources and in the general tracer code to skip same
!     sources (e.g. when reading from files). KEEP THIS SECTION after
!     nBBsources were removed from ntsurfsrc(n) above.
!
!    sourceName will be used inside MEGAN to match sub-species!
!-------------------------------------------------------------------------------
      select case (trname(n))
      case ('Isoprene')
#ifdef ISOPRENE_MEGAN
        call addSurfaceSource(this=pTracer, skipReason=itsMegan,
     &  sourceName='MegISOP_src',
     &  sourceLname='MEGAN '//trim(trname(n)))
        ! Couple safety checks for Isprene:
#ifdef PS_BVOC
        call stop_model('DO_MEGAN + PS_BVOC conflict',255)
#endif /* PS_BVOC */
#ifdef BIOGENIC_EMISSIONS
        call stop_model('DO_MEGAN + BIOGENIC_EMISSIONS conflict',255)
#endif /* BIOGENIC_EMISSIONS */
#else
        continue
#endif /* ISOPRENE_MEGAN */
      case ('Acetone')
#ifdef ACETONE_MEGAN
        call addSurfaceSource(this=pTracer, skipReason=itsMegan,
     &  sourceName='MegACTO_src',
     &  sourceLname='MEGAN '//trim(trname(n)))
#else
        continue
#endif
      case ('Terpenes')
#ifdef TERPENES_MEGAN
        call addSurfaceSource(this=pTracer, skipReason=itsMegan,
     &    sourceName='MegMYRC_src', sourceLname='MEGAN Myrcene')
        call addSurfaceSource(this=pTracer, skipReason=itsMegan,
     &    sourceName='MegSABI_src', sourceLname='MEGAN Sabinene')
        call addSurfaceSource(this=pTracer, skipReason=itsMegan,
     &    sourceName='MegLIMO_src', sourceLname='MEGAN Limonene')
        call addSurfaceSource(this=pTracer, skipReason=itsMegan,
     &    sourceName='Meg3CAR_src', sourceLname='MEGAN 3-Carene')
        call addSurfaceSource(this=pTracer, skipReason=itsMegan,
     &    sourceName='MegOCIM_src',sourceLname='MEGAN t-beta-Ocimene')
        call addSurfaceSource(this=pTracer, skipReason=itsMegan,
     &    sourceName='MegBPIN_src', sourceLname='MEGAN beta-Pinene')
        call addSurfaceSource(this=pTracer, skipReason=itsMegan,
     &    sourceName='MegAPIN_src', sourceLname='MEGAN alpha-Pinene')
        call addSurfaceSource(this=pTracer, skipReason=itsMegan,
     &    sourceName='MegOMTP_src',
     &    sourceLname='MEGAN Other Monoterpenes')
        call addSurfaceSource(this=pTracer, skipReason=itsMegan,
     &    sourceName='MegFARN_src',
     &    sourceLname='MEGAN alpha-Farnesene')
        call addSurfaceSource(this=pTracer, skipReason=itsMegan,
     &    sourceName='MegBCAR_src',
     &    sourceLname='MEGAN beta-Caryophyllene')
        call addSurfaceSource(this=pTracer, skipReason=itsMegan,
     &    sourceName='MegOSQT_src',
     &    sourceLname='MEGAN Other Sesquiterpenes')
#else
        continue
#endif /* TERPENES_MEGAN */
      end select
#endif /* DO_MEGAN */

#ifdef WATER_MISC_GRND_CH4_SRC
      ! For CH4 (Shindell or Lerner) allow online lakes, ocean
      ! and misc. ground source, as set by Jean Lerner:
      select case (trname(n))
      case ('CH4')
        call addSurfaceSource(this=pTracer, skipReason=itsCH4MGOL,
     &  sourceName='OcnLkMiscG_src',
     &  sourceLname='Ocean+Lake+MiscGround source')
      end select
#endif

      select case (trname(n))
      case ('Acetone')
#ifdef ACETONE_OCEAN
        call addSurfaceSource(this=pTracer, skipReason=itsOcean,
     &  sourceName='OcnACTO_src',
     &  sourceLname='Ocean '//trim(trname(n)))
#else
        continue
#endif
      end select

! pyrE source always last (after nBBsources)
! any prognostic fire sources are saved in ntsurfsrc(n)+nBBsources(n)+1,
! so in practice tracers with prognostic fire sources can only have
! ntsurfsrcmax+nBBsources(n)-1 offline emissions input files
      if(do_fire(n) .and. 
     &   (ntsurfsrc(n)+nBBsources(n)+1 > ntsurfsrcmax))then
        write(6,*)trname(n),'ntsurfsrc+nBBsources+1 > max of ',
     &            ntsurfsrcmax
        call stop_model('ntsurfsrc+nBBsources+do_fire too large',13)
      end if

!-------------------------------------------------------------------------------
! add surface sources to tracers that don't follow the TRACERNAME_XX convention
!-------------------------------------------------------------------------------
      select case (trname(n))

#ifndef TRACERS_AEROSOLS_SOA
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
      (defined TRACERS_TOMAS)
      case ('OCII', 'M_OCC_OC', 'SOAgas') ! this handles OCT_src (terpene source)
        call addSurfaceSource(pTracer, "Terpenes_src",
     &                                   "Terpenes source")
#endif
#endif  /* TRACERS_AEROSOLS_SOA */

#ifdef TRACERS_SPECIAL_Shindell
      case('CH4') ! in case use_rad_ch4/=0 but the rundeck lists CH4_XX files
        if (use_rad_ch4/=0) call set_ntsurfsrc(n,0)
#endif

#ifdef TRACERS_SPECIAL_Lerner
      ! Note: Lerner CH4 and CO2 surface sources now handled
      ! automatically from the input files.
      case ('N2O')
        call addSurfaceSource(pTracer, "overwrite_at_surface",
     *        "Overwrite")
      case ('CFC11', 'Rn222')
        call addSurfaceSource(pTracer, "surface_src", "Surface Src")
      case ('O3')
        call addSurfaceSource(pTracer, "deposition_sink",
     *        "Deposition")
      case ('14CO2')
        call addSurfaceSource(pTracer, "surface_sink", "Surface Sink")
#endif  /* TRACERS_SPECIAL_Lerner */

#ifdef TRACERS_PASSIVE
      case ('SF6', 'SF6_c', 'nh5', 'nh50', 'aoanh', 'aoa',
     &     'e90', 'st8025', 'tape_rec', 'nh15')

        call addSurfaceSource(pTracer, "surface_src", "Surface Src")
#endif  /* TRACERS_PASSIVE */

#ifdef TRACERS_TOMAS
      case ('ANUM__01','ANUM__02','ANUM__03','ANUM__04','ANUM__05',
     &      'ANUM__06','ANUM__07','ANUM__08','ANUM__09','ANUM__10',
     &      'ANUM__11','ANUM__12','ANUM__13','ANUM__14','ANUM__15')
        call addSurfaceSource(pTracer, "SO4_src", "SO4 source")
        call addSurfaceSource(pTracer, "EC_src", "EC source")
        call addSurfaceSource(pTracer, "OC_src", "OC source")
#endif  /* TRACERS_TOMAS */

      end select

      end subroutine setDefaultSpec

!------------------------------------------------------------------------------
      subroutine initTracerMetadata()
!------------------------------------------------------------------------------
      use TRACER_COM, only: COUPLED_CHEM
      use TRACER_COM, only: emiss_over_model_top_at_LM
      use TRACER_COM, only: direct_inject_num, ex_volc_num
      use TRACER_COM, only: direct_inject_hr0
      use TRACER_COM, only: direct_inject_hr1
      use TRACER_COM, only: direct_inject_jday
      use TRACER_COM, only: direct_inject_ndays
      use TRACER_COM, only: direct_inject_year
      use TRACER_COM, only: direct_inject_pointlat
      use TRACER_COM, only: direct_inject_pointlon
      use TRACER_COM, only: direct_inject_rectlat0
      use TRACER_COM, only: direct_inject_rectlat1
      use TRACER_COM, only: direct_inject_rectlon0
      use TRACER_COM, only: direct_inject_rectlon1
      use TRACER_COM, only: direct_inject_bot
      use TRACER_COM, only: direct_inject_top
      use TRACER_COM, only: direct_inject_SO2
      use TRACER_COM, only: direct_inject_H2O
      use TRACER_COM, only: direct_inject_SU
#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS) ||\
    (defined TRACERS_AMP) || (defined TRACERS_TOMAS) 
      USE trdust_mod, only: imDust
#endif
#ifdef TRACERS_AMP
      use TRACER_COM, only: direct_inject_DD1
      use TRACER_COM, only: direct_inject_DD2
#endif
      use TRACER_COM, only: direct_inject_BC
      use TRACER_COM, only: direct_inject_OC
      use TRACER_COM, only: CO50_yield_from_CH4
      use Dictionary_mod, only: set_param, sync_param
      use RunTimeControls_mod, only: tracers_special_shindell
      use RunTimeControls_mod, only: tracers_drydep
      use RunTimeControls_mod, only: tracers_tomas
      use RunTimeControls_mod, only: tracers_water
      use RunTimeControls_mod, only: tracers_special_o18
      use RunTimeControls_mod, only: tracers_gasexch_ocean_cfc
      use RunTimeControls_mod, only: tracers_gasexch_ocean_co2
      use RunTimeControls_mod, only: tracers_gasexch_land_co2
      use RunTimeControls_mod, only: tracers_gasexch_gcc
      use RunTimeControls_mod, only: tracers_special_lerner
      use RunTimeControls_mod, only: tracers_passive
      use RunTimeControls_mod, only: tracers_aerosols_koch
      use RunTimeControls_mod, only: tracers_aerosols_vbs
      use RunTimeControls_mod, only: tracers_aerosols_seasalt
      use RunTimeControls_mod, only: tracers_aerosols_ocean
      use RunTimeControls_mod, only: tracers_nitrate
      use RunTimeControls_mod, only: tracers_dust
      use RunTimeControls_mod, only: tracers_dust_silt4
      use RunTimeControls_mod, only: tracers_dust_silt5
      use RunTimeControls_mod, only: tracers_hetchem
      use RunTimeControls_mod, only: tracers_cosmo
      use RunTimeControls_mod, only: tracers_radon
      use RunTimeControls_mod, only: tracers_minerals
      use RunTimeControls_mod, only: tracers_on
      use RunTimeControls_mod, only: tracers_air
      use RunTimeControls_mod, only: tracers_amp
      use OldTracer_mod, only: KH_298
      use OldTracer_mod, only: F0
      use OldTracer_mod, only: nGas
      use OldTracer_mod, only: nPart
      use OldTracer_mod, only: nWater
      use OldTracer_mod, only: tr_wd_type
      use OldTracer_mod, only: tr_mm
      use OldTracer_mod, only: tr_rkd
      use OldTracer_mod, only: trname
      use OldTracer_mod, only: initializeOldTracers
      use OldTracer_mod, only: set_needtrs
      use OldTracer_mod, only: set_mass2vol 
      use OldTracer_mod, only: set_vol2mass
      use OldTracer_mod, only: set_to_conc
      use OldTracer_mod, only: set_dowetdep
      use OldTracer_mod, only: set_dodrydep
      use OldTracer_mod, only: set_to_volume_MixRat
      use Tracer_mod, only: Tracer
#ifdef TRACERS_SPECIAL_Lerner
      use LernerTracersMetadata_mod
#endif
#ifdef TRACERS_PASSIVE
      use PassivetracersMetadata_mod
#endif
#ifdef TRACERS_SPECIAL_Shindell
      use ShindellTracersMetadata_mod
      use TRCHEM_Shindell_COM, only: use_rad_ch4
      use TRACER_SOURCES, only: GLToffset
      use ghgmod, only: save_dQ_for_NINT
#endif   
#ifdef TRACERS_TOMAS
      use TomasTracersMetadata_mod
#endif    
#ifdef TRACERS_AMP
      use AmpTracersMetadata_mod
#endif   
#ifdef TRACERS_AEROSOLS_Koch
      use KochTracersMetadata_mod
#endif   
#ifdef TRACERS_AEROSOLS_VBS
      use VbsTracersMetadata_mod
#endif   
#ifdef TRACERS_AEROSOLS_SEASALT
      use SeasaltTracersMetadata_mod
#endif   
#ifdef TRACERS_NITRATE
      use sharedTracersMetadata_mod, only:
     &  NH3_setSpec, NH4_setSpec
#endif
#ifdef TRACERS_RADON
      use sharedTracersMetadata_mod, only: Rn222_setSpec
      use MiscTracersMetadata_mod, only: Pb210_setSpec
#endif
#ifdef TRACERS_MINERALS
      use MineralsTracersMetadata_mod
#endif
#ifdef WATER_MISC_GRND_CH4_SRC
      use TRACER_COM, only: scale_CH4MGOL
#endif
      use MiscTracersMetadata_mod
      USE CONSTANT, only: mair
      USE TRACER_COM, only: ntm
      USE TRACER_COM, only: tracers
      use TimeConstants_mod, only : INT_HOURS_PER_DAY

      implicit none
      class (Tracer), pointer :: pTracer
      external setDefaultSpec
      integer :: i
      integer :: ex,ex2

      call sync_param( "COUPLED_CHEM", COUPLED_CHEM )
      call sync_param( "emiss_over_model_top_at_LM", 
     &                  emiss_over_model_top_at_LM )
#ifdef TRACERS_SPECIAL_Shindell
      call sync_param( "use_rad_ch4", use_rad_ch4 )
      call sync_param( "GLToffset", GLToffset )
      call sync_param( "save_dQ_for_NINT", save_dQ_for_NINT )
#endif
#ifdef WATER_MISC_GRND_CH4_SRC
      call sync_param( "scale_CH4MGOL", scale_CH4MGOL )
#endif

! simple aerosol and gas injections based on rundeck parameters

      call sync_param("direct_inject_num", direct_inject_num)
      call sync_param("ex_volc_num", ex_volc_num)
      if (direct_inject_num>0 .and. ex_volc_num>0)
     &  call stop_model('cannot use both ex_volc & direct_inject', 255)

      if (direct_inject_num+ex_volc_num>0) then
       allocate(direct_inject_hr0(direct_inject_num+ex_volc_num))
       allocate(direct_inject_hr1(direct_inject_num+ex_volc_num))
       allocate(direct_inject_jday(direct_inject_num+ex_volc_num))
       allocate(direct_inject_ndays(direct_inject_num+ex_volc_num))
       allocate(direct_inject_year(direct_inject_num+ex_volc_num))
       allocate(direct_inject_pointlat(direct_inject_num+ex_volc_num))
       allocate(direct_inject_pointlon(direct_inject_num+ex_volc_num))
       allocate(direct_inject_rectlat0(direct_inject_num+ex_volc_num))
       allocate(direct_inject_rectlat1(direct_inject_num+ex_volc_num))
       allocate(direct_inject_rectlon0(direct_inject_num+ex_volc_num))
       allocate(direct_inject_rectlon1(direct_inject_num+ex_volc_num))
       allocate(direct_inject_bot(direct_inject_num+ex_volc_num))
       allocate(direct_inject_top(direct_inject_num+ex_volc_num))
       allocate(direct_inject_SO2(direct_inject_num+ex_volc_num))
       allocate(direct_inject_H2O(direct_inject_num+ex_volc_num))
       allocate(direct_inject_SU(direct_inject_num))
#ifdef TRACERS_AMP
       allocate(direct_inject_DD1(direct_inject_num))
       allocate(direct_inject_DD2(direct_inject_num))
#endif
       allocate(direct_inject_BC(direct_inject_num))
       allocate(direct_inject_OC(direct_inject_num))

! set default emissions to full day length and zero amount
       direct_inject_hr0(:)=0
       direct_inject_hr1(:)=INT_HOURS_PER_DAY
       direct_inject_ndays(:)=1

       direct_inject_SO2(:)=0.d0
       direct_inject_H2O(:)=0.d0
       direct_inject_SU(:)=0.d0
#ifdef TRACERS_AMP
       direct_inject_DD1(:)=0.d0
       direct_inject_DD2(:)=0.d0
#endif
       direct_inject_BC(:)=0.d0
       direct_inject_OC(:)=0.d0

       direct_inject_rectlat0(:)=-999.d0 ! init as negative values so
       direct_inject_rectlat1(:)=-999.d0 ! model can tell if not used
       direct_inject_rectlon0(:)=-999.d0
       direct_inject_rectlon1(:)=-999.d0

       if (direct_inject_num>0) then
        call sync_param("direct_inject_hr0", direct_inject_hr0,
     &    direct_inject_num)
        call sync_param("direct_inject_hr1", direct_inject_hr1,
     &    direct_inject_num)
        call sync_param("direct_inject_jday", direct_inject_jday, 
     &    direct_inject_num)
        call sync_param("direct_inject_ndays", direct_inject_ndays,
     &    direct_inject_num)
        call sync_param("direct_inject_year", direct_inject_year,
     &    direct_inject_num)
       call sync_param("direct_inject_pointlat", direct_inject_pointlat,
     &    direct_inject_num)
       call sync_param("direct_inject_pointlon", direct_inject_pointlon,
     &    direct_inject_num)
        call sync_param("direct_inject_rectlat0",direct_inject_rectlat0,
     &    direct_inject_num)
        call sync_param("direct_inject_rectlat1",direct_inject_rectlat1,
     &    direct_inject_num)
        call sync_param("direct_inject_rectlon0",direct_inject_rectlon0,
     &    direct_inject_num)
        call sync_param("direct_inject_rectlon1",direct_inject_rectlon1,
     &    direct_inject_num)
        call sync_param("direct_inject_bot", direct_inject_bot,
     &    direct_inject_num)
        call sync_param("direct_inject_top", direct_inject_top,
     &    direct_inject_num)
        call sync_param("direct_inject_SO2", direct_inject_SO2,
     &    direct_inject_num)
        call sync_param("direct_inject_H2O", direct_inject_H2O,
     &    direct_inject_num)
        call sync_param("direct_inject_SU", direct_inject_SU,
     &    direct_inject_num)
#ifdef TRACERS_AMP
        call sync_param("direct_inject_DD1", direct_inject_DD1, 
     &    direct_inject_num)
        call sync_param("direct_inject_DD2", direct_inject_DD2,
     &    direct_inject_num)
#endif
        call sync_param("direct_inject_BC", direct_inject_BC,
     &    direct_inject_num)
        call sync_param("direct_inject_OC", direct_inject_OC,
     &    direct_inject_num)
       else
        direct_inject_num = ex_volc_num

        call sync_param("ex_volc_hr0", direct_inject_hr0,
     &    direct_inject_num)
        call sync_param("ex_volc_hr1", direct_inject_hr1,
     &    direct_inject_num)
        call sync_param("ex_volc_jday", direct_inject_jday, 
     &    direct_inject_num)
        call sync_param("ex_volc_year", direct_inject_year,
     &    direct_inject_num)
        call sync_param("ex_volc_lat", direct_inject_pointlat,
     &    direct_inject_num)
        call sync_param("ex_volc_lon", direct_inject_pointlon,
     &    direct_inject_num)
        call sync_param("ex_volc_bot", direct_inject_bot,
     &    direct_inject_num)
        call sync_param("ex_volc_top", direct_inject_top,
     &    direct_inject_num)
        call sync_param("ex_volc_SO2", direct_inject_SO2,
     &    direct_inject_num)
        call sync_param("ex_volc_H2O", direct_inject_H2O,
     &    direct_inject_num)
       endif

       do ex=1,direct_inject_num
         if (direct_inject_hr0(ex)<0 .or.
     &     direct_inject_hr0(ex)>INT_HOURS_PER_DAY)
     &     call stop_model('direct_inject_hr0 out of bounds', 255)
         if (direct_inject_hr1(ex)<0 .or.
     &     direct_inject_hr1(ex)>INT_HOURS_PER_DAY)
     &     call stop_model('direct_inject_hr1 out of bounds', 255)
         if (direct_inject_hr0(ex)>=direct_inject_hr1(ex))
     &     call stop_model('direct_inject_hr0>=direct_inject_hr1',
     &     255)

         if (direct_inject_jday(ex)<=0 .or.
     &     direct_inject_jday(ex)>366)
     &     call stop_model('direct_inject_jday(ex) out of bounds',
     &     255)
         if (direct_inject_ndays(ex)<1)
     &     call stop_model('direct_inject_ndays(ex)<1', 255)
         if (direct_inject_year(ex)<=0)
     &     call stop_model('direct_inject_year(ex)<=0', 255)

!        check whether point injection or distributed in rectangle
         if ((direct_inject_rectlat0(ex).ne.direct_inject_rectlat1(ex))
     &  .or. (direct_inject_rectlon0(ex).ne.direct_inject_rectlon1(ex)))
     &     then !injection across rectangular region
           if (direct_inject_rectlat0(ex)<=-90.d0 .or.
     &       direct_inject_rectlat0(ex)>90.d0)
     &       call stop_model(
     &       'direct_inject_rectlat0(ex) out of bounds', 255)
           if (direct_inject_rectlat1(ex)<=-90.d0 .or.
     &       direct_inject_rectlat1(ex)>90.d0)
     &       call stop_model(
     &       'direct_inject_rectlat1(ex) out of bounds', 255)
           if (direct_inject_rectlon0(ex)<=-180.d0 .or.
     &       direct_inject_rectlon0(ex)>180.d0)
     &       call stop_model(
     &       'direct_inject_rectlon0(ex) out of bounds', 255)
           if (direct_inject_rectlon1(ex)<=-180.d0 .or.
     &       direct_inject_rectlon1(ex)>180.d0)
     &       call stop_model(
     &       'direct_inject_rectlon1(ex) out of bounds', 255)
         else !point injection
           if (direct_inject_pointlat(ex)<=-90.d0 .or.
     &       direct_inject_pointlat(ex)>90.d0)
     &       call stop_model(
     &       'direct_inject_pointlat(ex) out of bounds', 255)
           if (direct_inject_pointlon(ex)<=-180.d0 .or.
     &       direct_inject_pointlon(ex)>180.d0)
     &       call stop_model(
     &       'direct_inject_pointlon(ex) out of bounds', 255)
         endif

!        disable multiple injections from having separate injection
!        hours within a single day (due to reliance on daily_tracer)
         if (direct_inject_hr0(ex)>0) then
           do ex2=1,direct_inject_num
             if (ex2==ex) cycle
             if ((direct_inject_jday(ex2)==direct_inject_jday(ex))
     &         .and. (direct_inject_hr0(ex2)/=direct_inject_hr0(ex)))
     &         call stop_model(
     &         'cannot have multiple direct_inject events occur'//
     &         ' at separate hours on same day', 255)
             if (direct_inject_jday(ex2)+direct_inject_ndays(ex2)-1
     &         ==direct_inject_jday(ex))
     &         call stop_model(
     &         'cannot have multiple direct_inject events occur'//
     &         ' at separate hours on same day', 255)
           enddo
         endif
         if (direct_inject_hr1(ex)<INT_HOURS_PER_DAY) then
           do ex2=1,direct_inject_num
             if (ex2==ex) cycle
             if ((direct_inject_jday(ex2)+direct_inject_ndays(ex2)-1
     &         ==direct_inject_jday(ex)+direct_inject_ndays(ex)-1)
     &         .and. (direct_inject_hr1(ex2)/=direct_inject_hr1(ex)))
     &         call stop_model(
     &         'cannot have multiple direct_inject events occur'//
     &         ' at separate hours on same day', 255)
             if (direct_inject_jday(ex2)
     &         ==direct_inject_jday(ex)+direct_inject_ndays(ex)-1)
     &         call stop_model(
     &         'cannot have multiple direct_inject events occur'//
     &         ' at separate hours on same day', 255)
           enddo
         endif

         if (direct_inject_bot(ex)>=direct_inject_top(ex))
     &     call stop_model(
     &     'direct_inject_bot(ex)>=direct_inject_top(ex)', 255)
         if (direct_inject_SO2(ex)<0.d0)
     &     call stop_model('direct_inject_SO2(ex)<0.d0', 255)
         if (direct_inject_H2O(ex)<0.d0)
     &     call stop_model('direct_inject_H2O(ex)<0.d0', 255)
         if (direct_inject_SU(ex)<0.d0)
     &      call stop_model('direct_inject_SU(ex)<0.d0', 255)
#ifdef TRACERS_AMP
         if (direct_inject_DD1(ex)<0.d0)
     &      call stop_model('direct_inject_DD1(ex)<0.d0', 255)
         if (direct_inject_DD2(ex)<0.d0)
     &      call stop_model('direct_inject_DD2(ex)<0.d0', 255)
#endif
         if (direct_inject_BC(ex)<0.d0)
     &      call stop_model('direct_inject_BC(ex)<0.d0', 255)
         if (direct_inject_OC(ex)<0.d0)
     &      call stop_model('direct_inject_OC(ex)<0.d0', 255)
       enddo
      endif

#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS) ||\
    (defined TRACERS_AMP) || (defined TRACERS_TOMAS) 
      call sync_param('imDUST',imDUST)
#endif

      call initializeOldTracers(tracers, setDefaultSpec)

! ***  BEGIN TRACER METADATA INITIALIZATION

#ifdef TRACERS_SPECIAL_Shindell
        if (tracers_special_shindell) then
          call SHINDELL_InitMetadata(pTracer)
        end if
#endif

#ifdef TRACERS_SPECIAL_Lerner
      if (tracers_special_lerner) then
        call Lerner_InitMetadata(pTracer)
        if (tracers_special_shindell) 
     &    call stop_model('contradictory tracer specs')
      end if
#endif

#ifdef TRACERS_PASSIVE
      if (tracers_passive) then
        call sync_param('CO50_yield_from_CH4', CO50_yield_from_CH4)
        call Passive_InitMetadata(pTracer)
      end if
#endif

      if (tracers_water) then
        call  Water_setSpec('Water')
      end if
     
#ifdef TRACERS_SPECIAL_O18
        if (tracers_special_o18) then
          call H2O18_setSpec('H2O18')
          call HDO_setSpec('HDO')
#ifdef TRACERS_WISO_O17
          call H2O17_setSpec('H2O17')
#endif
        end if
#endif

      if (tracers_gasexch_ocean_co2 .or. tracers_gasexch_land_co2
     &    .or. tracers_gasexch_gcc) then
        call  CO2n_setSpec('CO2n')
      end if
      
      if (tracers_gasexch_ocean_cfc) then
        call  CFCn_setSpec('CFCn')
      end if

#ifdef TRACERS_AEROSOLS_SEASALT
      if (tracers_aerosols_seasalt) then
        call Seasalt_InitMetadata(pTracer)
      end if
#endif

#ifdef TRACERS_AEROSOLS_Koch
      if (tracers_aerosols_koch) then
        call KOCH_InitMetadata(pTracer)
      end if
#endif

#ifdef TRACERS_AEROSOLS_VBS
      if (tracers_aerosols_vbs) then
        call VBS_InitMetadata(pTracer)
      end if
#endif

      if (tracers_aerosols_ocean) then
        call  OCocean_setSpec('OCocean') !Insoluble oceanic organic mass
      end if

      if (tracers_dust) then
        call  clay_setSpec('Clay')
        call  Silt1_setSpec('Silt1')
        call  Silt2_setSpec('Silt2')
        call  Silt3_setSpec('Silt3')
        if (tracers_dust_Silt4) call  Silt4_setSpec('Silt4')
        if (tracers_dust_Silt5) call  Silt5_setSpec('Silt5')
      end if

#ifdef TRACERS_NITRATE
      if (tracers_nitrate) then
        call  NH3_setSpec('NH3')
        call  NH4_setSpec('NH4')
        call  NO3p_setSpec('NO3p')
      end if
#endif

      if (tracers_hetchem) then
        call  SO4_d1_setSpec('SO4_d1')
        call  SO4_d2_setSpec('SO4_d2')
        call  SO4_d3_setSpec('SO4_d3')
        if (tracers_nitrate) then
          call  N_d1_setSpec('N_d1')
          call  N_d2_setSpec('N_d2')
          call  N_d3_setSpec('N_d3')
        end if
      end if

#ifdef TRACERS_RADON
      if (tracers_radon) then
        if (tracers_special_lerner) then
          call stop_model('contradictory tracer specs')
        else
          call  Rn222_setSpec('Rn222') ! conflicts with Lerner
          call  Pb210_setSpec('Pb210')
        endif
      end if

      if (tracers_cosmo) then
        call  Be7_setSpec('Be7')
        call  Be10_setSpec('Be10')
      end if
#endif  /* TRACERS_RADON */

#ifdef TRACERS_MINERALS
       if (tracers_minerals) then
         call Minerals_InitMetadata(pTracer)
       end if
#endif

      if (tracers_air) then
        call  air_setSpec('Air')
      end if

#ifdef TRACERS_AMP
      if (tracers_amp) then
        call AMP_InitMetadata(pTracer)
      end if
#endif

#ifdef TRACERS_TOMAS
      if (tracers_tomas) then
        call TOMAS_InitMetadata(pTracer)
      end if
#endif
      call init_source_distrib

! ***  END TRACER METADATA INITIALIZATION

      ! Generic tracer work
      ! All tracers must have been declared before reaching this point!!!
      ntm = tracers%size()

      call set_param("NTM",NTM,'o')
      call set_param("TRNAME",trName(),ntm,'o')
      call printTracerNames(trName())

      ! Generic tracer work
      do i = 1, ntm
        if (tracers_water) then
!**** Tracers that are soluble or are scavenged or are water => wet dep
          if (tr_wd_type(i).eq.nWater.or.tr_wd_type(i) .EQ. nPART .or.
     &      tr_RKD(i).gt.0) then
            call set_dowetdep(i, .true.)
          end if
        end if
        if (tracers_drydep) then
!**** If tracers are particles or have non-zero KH_298 or F0 do dry dep:
!**** Any tracers that dry deposits needs the surface concentration:
          if(KH_298(i).GT.0..OR.F0(i).GT.0..OR.tr_wd_type(i).eq.nPART) 
     &      then
            call set_dodrydep(i, .true.)
            call set_needtrs(i, .true.)
            if (tracers_water) then
              if (tr_wd_type(i).eq.nWATER) call stop_model
     &         ('A water tracer should not undergo dry deposition.',255)
            end if
          end if
        end if

        if (tracers_on) then
!**** Define the conversion from mass to volume units here
          call set_mass2vol(i, mair/tr_mm(i))
          call set_vol2mass(i, tr_mm(i)/mair)
          call set_to_conc(i, 0)
!**** Aerosol tracer output should be mass mixing ratio
          select case (tr_wd_TYPE(i))
          case (nGAS)
            call set_to_volume_MixRat(i, 1) !gas output to volume mixing ratio
          case (nPART, nWATER)
            call set_to_volume_MixRat(i, 0) ! aerosol/water output to mass mixing ratio
          case default
            call stop_model('tr_wd_type can only be nGAS, nPART, nWATER'
     &                      ,255)
          end select
       end if
      end do

      contains

      subroutine printTracerNames(tracerNames)
      use domain_decomp_atm, only: am_i_root

      character(len=*) :: tracerNames(:)
      integer :: i
      
      if (am_i_root()) then
        do i = 1, size(tracerNames)
          write(6,*) 'TRACER',i,trim(tracerNames(i))
        end do
      end if
      
      end subroutine printTracerNames

      end subroutine initTracerMetadata


!------------------------------------------------------------------------------
      subroutine init_source_distrib
      use dictionary_mod, only : is_set_param, get_param
      use oldtracer_mod, only: oldaddtracer, set_t_qlimit, findtracer,
     &  set_src_dist_base, set_src_dist_index
      use tracer_com, only: xyztr
      implicit none
      character(len=1024) :: list
      integer :: str_pos, i, nt, nt_orig, ndigits
      character(len=10) :: name, basename
      character(len=8) :: fm

      if (is_set_param('src_dist_tr')) then
        call src_dist_config
        call get_param('src_dist_tr', list)
        list=adjustl(list)
        do while (len_trim(list).gt.0)
          str_pos=index(list, ' ')
          name=list(1:str_pos-1)
          basename=name
          nt_orig=findtracer(basename)
          call set_t_qlimit(nt_orig, .false.)
          call set_src_dist_base(nt_orig, nt_orig)
          call set_src_dist_index(nt_orig, 1)
          ndigits=int(log10(size(xyztr, 1)*1d0))+1
          if (ndigits>7) call stop_model('too many digits',255)
          write(fm,'(a,i1.1,a,i1.1,a)') '(a,i',ndigits,'.',ndigits,')'
          do i=2, size(xyztr, 1)
            write(name, fm) trim(basename(1:8-ndigits)), i
            nt=oldaddtracer(name, basename)
            call set_src_dist_index(nt, i)
          end do
          list=adjustl(list(str_pos:))
        end do
      endif

      return
      end subroutine init_source_distrib
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
      subroutine laterInitTracerMetadata()
!------------------------------------------------------------------------------
      USE MODEL_COM, only: itime,master_yr
      use OldTracer_mod, only: itime_tr0,set_itime_tr0,trName
      use OldTracer_mod, only: to_volume_MixRat
      use OldTracer_mod, only: set_to_volume_MixRat
      use OldTracer_mod, only: to_conc
      use OldTracer_mod, only: set_to_conc
      USE TRACER_COM, only: NTM, tracers, syncProperty
      use TRACER_COM, only: nc_emis_use_ppm_interp
      use Dictionary_mod, only: sync_param,is_set_param,get_param
      use RAD_COM, only: diag_fc
#ifdef TRACERS_SPECIAL_O18
      use tracer_com, only: supsatfac
#endif
#ifdef TRACERS_WATER
      use TRDIAG_com, only: to_per_mil
#endif
#ifdef TRACERS_SPECIAL_Shindell
      use TRCHEM_Shindell_COM, only: tune_NOx
      use TRCHEM_Shindell_COM, only: tune_BVOC
#endif  /* TRACERS_SPECIAL_Shindell */
      use TRACER_COM, only: tune_BBsources
#ifdef TRACERS_AEROSOLS_Koch
      use aerosol_sources, only: tune_DMS
#endif  /* TRACERS_AEROSOLS_Koch */
#ifdef TRACERS_AEROSOLS_SEASALT
      use tracers_seasalt, only: tune_ss1, tune_ss2
#endif  /* TRACERS_AEROSOLS_SEASALT */
      use TRDIAG_COM, only: diag_rad,diag_aod_3d,diag_reff_3d,
     &     save_dry_aod
      use TRACER_COM, only: ntm ! should be available by this procedure call
#ifdef TRACERS_WATER
#ifdef TRDIAG_WETDEPO
      USE CLOUDS, ONLY : diag_wetdep
#endif
#endif /* TRACERS_WATER */
#ifdef TRACERS_PHOTOLYSIS
      use photolysis, only: rad_FL
#endif  /* TRACERS_PHOTOLYSIS */
#ifdef TRACERS_SPECIAL_Shindell
      USE TRCHEM_Shindell_COM,only:
     &     which_trop,allowSomeChemReinit,
     &     Lmax_rad_O3,Lmax_rad_CH4,
     &     use_rad_n2o,use_rad_cfc,cfc_rad95,PltOx,Tpsc_offset_N,
     &     Tpsc_offset_S,windowN2Ocorr,windowO2corr,
     &     reg1Power_SpherO2andN2Ocorr,reg1TopPres_SpherO2andN2Ocorr,
     &     reg2Power_SpherO2andN2Ocorr,reg2TopPres_SpherO2andN2Ocorr,
     &     reg3Power_SpherO2andN2Ocorr,reg3TopPres_SpherO2andN2Ocorr,
     &     reg4Power_SpherO2andN2Ocorr
#ifdef INTERACTIVE_WETLANDS_CH4
      USE TRACER_SOURCES, only:int_wet_dist,topo_lim,sat_lim,gw_ulim,
     &  gw_llim,sw_lim,exclude_us_eu,nn_or_zon,ice_age,nday_ch4,
     &  max_days,ns_wet,nra_ch4
#endif
#ifdef BIOGENIC_EMISSIONS
      use biogenic_emis, only: base_isopreneX
#endif
      use RAD_COM, only: O3_yr
      use TRCHEM_Shindell_COM, only: NOx_yr
      use TRCHEM_Shindell_COM, only: CO_yr
      use TRCHEM_Shindell_COM, only: VOC_yr
#endif /* TRACERS_SPECIAL_Shindell */
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS)  || (defined TRACERS_AEROSOLS_SEASALT)
      use TRACER_COM, only: aer_int_yr
      use TRACER_COM, only: SO2_int_yr
      use TRACER_COM, only: NH3_int_yr
      use TRACER_COM, only: BC_int_yr
      use TRACER_COM, only: OC_int_yr
#endif
#ifdef TRACERS_AMP
      USE AMP_AEROSOL, only: AMP_RAD_KEY
#endif
#if (defined TRACERS_AMP) || (defined OMA_TRAMPRAD)
      USE AMP_AEROSOL, only: LWaerosolCalcs,separate_h2so4p
#endif
#if (defined TRACERS_COSMO)
      USE COSMO_SOURCES, only: be7_src_param
#endif
#ifdef TRACERS_AEROSOLS_VBS
      USE AEROSOL_SOURCES, only: vbs_sets,vbs_conc
      USE AEROSOL_SOURCES, only: VBSemifactFF,VBSemifactBB
#endif  /* TRACERS_AEROSOLS_VBS */
      USE TRACER_COM, only: no_emis_over_ice
#ifdef TRACERS_MINERALS
      use trdust_mod, only: frIronOxideInAggregate,
     &     noAggregateByTotalFeox
#endif
      use Model_com, only: itime
      implicit none
      integer :: n,v,val,def

C**** 
C**** Set some documentary parameters in the database
C**** 
      def = itime
      call sync_param('default_itime_tr0',def)

      do n = 1, NTM
        call set_itime_tr0(n, def)
        val = itime_tr0(n)
        call sync_param(trim(trname(n))//'_itime_tr0',val)
        call set_itime_tr0(n, val)
      end do
      call syncProperty(tracers, "itime_tr0", set_itime_tr0,itime_tr0())
C**** Get to_volume_MixRat from rundecks if it exists
      call syncProperty(tracers, "to_volume_MixRat",
     &     set_to_volume_MixRat,to_volume_MixRat())
C**** Get to_conc from rundecks if it exists
      call syncProperty(tracers,"to_conc", set_to_conc, to_conc())

C**** Synchronise tracer related parameters from rundeck

      call sync_param("nc_emis_use_ppm_interp",nc_emis_use_ppm_interp)
 
#ifdef TRACERS_WATER
C**** Decide on water tracer conc. units from rundeck if it exists
      call sync_param("to_per_mil",to_per_mil,ntm)
#endif
#ifdef TRACERS_SPECIAL_Shindell
      call sync_param("tune_NOx",tune_NOx)
      call sync_param("tune_BVOC",tune_BVOC)
      call get_param("O3_yr", O3_yr, default=master_yr) ! duplicate of RAD_DRV
      if (is_set_param("NOx_yr")) then
        call get_param("NOx_yr",NOx_yr)
      else
        NOx_yr=O3_yr
      endif
      if (is_set_param("CO_yr")) then
        call get_param("CO_yr",CO_yr)
      else
        CO_yr=O3_yr
      endif
      if (is_set_param("VOC_yr")) then
        call get_param("VOC_yr",VOC_yr)
      else
        VOC_yr=O3_yr
      endif
#endif  /* TRACERS_SPECIAL_Shindell */
      call sync_param("tune_BBsources",tune_BBsources)
#ifdef TRACERS_AEROSOLS_Koch
      call sync_param("tune_DMS",tune_DMS)
#endif  /* TRACERS_AEROSOLS_Koch */
#ifdef TRACERS_AEROSOLS_SEASALT
      call sync_param("tune_ss1",tune_ss1)
      call sync_param("tune_ss2",tune_ss2)
#endif  /* TRACERS_AEROSOLS_SEASALT */
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
      (defined TRACERS_TOMAS) || (defined TRACERS_AEROSOLS_SEASALT)
C**** determine year of emissions
      if (is_set_param("aer_int_yr")) then
        call get_param("aer_int_yr",aer_int_yr)
      else
        aer_int_yr=master_yr
      endif
      if (is_set_param("SO2_int_yr")) then
        call get_param("SO2_int_yr",SO2_int_yr)
      else
        SO2_int_yr=aer_int_yr
      endif
      if (is_set_param("NH3_int_yr")) then
        call get_param("NH3_int_yr",NH3_int_yr)
      else
        NH3_int_yr=aer_int_yr
      endif
      if (is_set_param("BC_int_yr")) then
        call get_param("BC_int_yr",BC_int_yr)
      else
        BC_int_yr=aer_int_yr
      endif
      if (is_set_param("OC_int_yr")) then
        call get_param("OC_int_yr",OC_int_yr)
      else
        OC_int_yr=aer_int_yr
      endif
#endif
#ifdef TRACERS_AEROSOLS_VBS
      do v=1,vbs_sets
        call sync_param("VBSemifactFF",VBSemifactFF,vbs_conc(v)%nbins)
        call sync_param("VBSemifactBB",VBSemifactBB,vbs_conc(v)%nbins)
      enddo
#endif
#ifdef TRACERS_SPECIAL_O18
C**** set super saturation parameter for isotopes if needed
      call sync_param("supsatfac",supsatfac)
#endif
#ifdef TRACERS_ON
      CALL sync_param("diag_rad",diag_rad)
      CALL sync_param("diag_aod_3d",diag_aod_3d)
      CALL sync_param("diag_reff_3d",diag_reff_3d)
#if (defined TRACERS_WATER) && (defined TRDIAG_WETDEPO)
      CALL sync_param("diag_wetdep",diag_wetdep)
#endif
!     not params call sync_param("trans_emis_overr_day",trans_emis_overr_day)
!     not params call sync_param("trans_emis_overr_yr", trans_emis_overr_yr )
#endif /* TRACERS_ON */
#ifdef TRACERS_PHOTOLYSIS
      call sync_param("rad_FL",rad_fl)
#endif  /* TRACERS_PHOTOLYSIS */
#ifdef TRACERS_SPECIAL_Shindell
      call sync_param("allowSomeChemReinit",allowSomeChemReinit)
      call sync_param("which_trop",which_trop)
      call sync_param("Lmax_rad_O3",Lmax_rad_O3)
      call sync_param("Lmax_rad_CH4",Lmax_rad_CH4)
      call sync_param("use_rad_n2o",use_rad_n2o)
      call sync_param("use_rad_cfc",use_rad_cfc)
      call sync_param("PltOx",PltOx)
      call sync_param("Tpsc_offset_N",Tpsc_offset_N)
      call sync_param("Tpsc_offset_S",Tpsc_offset_S)
      call sync_param("reg1Power_SpherO2andN2Ocorr",
     &                 reg1Power_SpherO2andN2Ocorr)
      call sync_param("reg2Power_SpherO2andN2Ocorr",
     &                 reg2Power_SpherO2andN2Ocorr)
      call sync_param("reg3Power_SpherO2andN2Ocorr",
     &                 reg3Power_SpherO2andN2Ocorr)
      call sync_param("reg4Power_SpherO2andN2Ocorr",
     &                 reg4Power_SpherO2andN2Ocorr)
      call sync_param("reg1TopPres_SpherO2andN2Ocorr",
     &                 reg1TopPres_SpherO2andN2Ocorr)
      call sync_param("reg2TopPres_SpherO2andN2Ocorr",
     &                 reg2TopPres_SpherO2andN2Ocorr)
      call sync_param("reg3TopPres_SpherO2andN2Ocorr",
     &                 reg3TopPres_SpherO2andN2Ocorr)
      call sync_param("windowN2Ocorr",windowN2Ocorr)
      call sync_param("windowO2corr",windowO2corr)
#ifdef BIOGENIC_EMISSIONS
      call sync_param("base_isopreneX",base_isopreneX)
#endif
#ifdef INTERACTIVE_WETLANDS_CH4
      call sync_param("ice_age",ice_age)
      call sync_param("ns_wet",ns_wet)
      call sync_param("int_wet_dist",int_wet_dist)
      call sync_param("topo_lim",topo_lim)
      call sync_param("sat_lim",sat_lim)
      call sync_param("gw_ulim",gw_ulim)
      call sync_param("gw_llim",gw_llim)
      call sync_param("sw_lim",sw_lim)
      call sync_param("exclude_us_eu",exclude_us_eu)
      call sync_param("nn_or_zon",nn_or_zon)
      do n=1,nra_ch4
        if(nday_ch4(n) > max_days .or. nday_ch4(n) < 1)
     &       call stop_model('nday_ch4 out of range',255)
      end do
#endif

#endif /* TRACERS_SPECIAL_Shindell */
      call sync_param("diag_fc",diag_fc)

#if (defined TRACERS_AMP)
C**** Decide Radiative Mixing Rules - Volume - Core Shell - Maxwell Garnett, default Volume
      call sync_param("AMP_RAD_KEY",AMP_RAD_KEY)
#endif
#if (defined TRACERS_AMP) || (defined OMA_TRAMPRAD)
C**** Whether to calculate LW optical consts every call
c     (makes optical props. sensitive to aerosol size changes, but slows model)
      call sync_param("LWaerosolCalcs",LWaerosolCalcs)
C**** Whether to calculate H2SO4 aerosol properties as separate from amm sulf
c     (mostly matters for LW, where H2SO4 is more absorptive than amm sulf)
      call sync_param("separate_h2so4p",separate_h2so4p)
#endif
#ifdef TRACERS_MINERALS
      call sync_param('frIronOxideInAggregate', frIronOxideInAggregate)
      call sync_param('noAggregateByTotalFeox', noAggregateByTotalFeox)
#endif

#if (defined TRACERS_COSMO)
C**** get rundeck parameter for cosmogenic source factor
      call sync_param("be7_src_param", be7_src_param)
#endif
      call sync_param("no_emis_over_ice",no_emis_over_ice)

      end subroutine laterInitTracerMetadata

!------------------------------------------------------------------------------
      subroutine InitTracerMetadataAtmOcnCpler()
!------------------------------------------------------------------------------
      use Dictionary_mod, only: sync_param
#if (defined TRACERS_OCEAN) && !defined(TRACERS_OCEAN_INDEP)
! atmosphere copies atmosphere-declared tracer info to ocean
! so that the ocean can "inherit" it without referencing atm. code
      use ocn_tracer_com, only : tracerlist, ocn_tracer_entry,
     &     n_Water_ocn      => n_Water
      use oldtracer_mod, only: itime_tr0, ntrocn, t_qlimit,
     &            conc_from_fw, trdecay, vol2mass
      use tracer_com, only: n_water
      use trdiag_com, only: to_per_mil
#endif
      USE FLUXES, only : atmocn
      use OldTracer_mod, only: vol2mass
      USE TRACER_COM, only: ntm, gasex_index
      use OldTracer_mod, only: trw0
      implicit none
      integer :: n
#if (defined TRACERS_OCEAN) && !defined(TRACERS_OCEAN_INDEP)
      type(ocn_tracer_entry), pointer :: entry
#endif

#if (defined TRACERS_OCEAN) && !defined(TRACERS_OCEAN_INDEP)
! atmosphere copies atmosphere-declared tracer info to ocean module
! so that the ocean can "inherit" it without referencing atm. code
      n_Water_ocn = n_Water
      do n=1,ntm
        entry=>tracerlist%at(n)
        entry%itime_tr0    = itime_tr0(n)
        entry%ntrocn       = ntrocn(n)
        entry%to_per_mil   = to_per_mil(n)
        entry%t_qlimit     = t_qlimit(n)
        entry%conc_from_fw = conc_from_fw(n) 
        entry%trdecay      = trdecay(n)
        entry%trw0         = trw0(n)
#ifdef TRACERS_WATER
        entry%need_ic      = .true.
#endif
      enddo
#endif

! copy atmosphere-declared tracer info to atm-ocean coupler data
! structure for uses within ocean codes
      allocate(atmocn%trw0(ntm))
      do n=1,ntm
        atmocn%trw0(n) = trw0(n)
      enddo
      allocate(atmocn%vol2mass(gasex_index%getsize()))
      do n=1,gasex_index%getsize()
        atmocn%vol2mass(n) = vol2mass(gasex_index%at(n))
      enddo

      end subroutine InitTracerMetadataAtmOcnCpler

!------------------------------------------------------------------------------
      subroutine InitTracerDiagMetadata()
!------------------------------------------------------------------------------
      use TRACER_COM, only: remake_tracer_lists
      implicit none

C**** create tracer lists, needed for some diagnostic output decisions
      call remake_tracer_lists()

C**** Set some diags that are the same regardless
      call set_generic_tracer_diags

C**** Zonal mean/height diags
      call init_jls_diag

C**** lat/lon tracer sources, sinks and specials
      call init_ijts_diag

C**** lat/lon/height tracer specials
      call init_ijlts_diag

C**** Initialize conservation diagnostics
      call init_tracer_cons_diag

      end subroutine InitTracerDiagMetadata
