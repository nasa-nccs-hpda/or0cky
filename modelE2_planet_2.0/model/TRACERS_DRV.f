#include "rundeck_opts.h"

!@sum  TRACERS_DRV: tracer-dependent routines for air/water mass
!@+    and ocean tracers
!@+    Routines included:
!@+      Those that MUST EXIST for all tracers:
!@+        Diagnostic specs: init_tracer
!@+        Tracer initialisation + sources: tracer_ic, set_tracer_source
!@+        Entry points: daily_tracer
!@auth Jean Lerner/Gavin Schmidt
!=======================================================================
      integer function get_src_index(n)
!@var get_src_index If an emission file contains information for more than one
!@+                 tracer, first read tracer n_XXX, then set src_index=n_XXX.
!@+                 Note the order! Notable case is SO2/SO4
!@auth Kostas Tsigaridis
      use OldTracer_mod, only: trname
      use TRACER_COM, only: n_SO2
      implicit none
!@var n index of current tracer whose emissions index is being seeked
      integer, intent(in) :: n

      select case (trname(n))
        case ('SO4', 'M_ACC_SU', 'ASO4__01')
          get_src_index=n_SO2
        case ('M_AKK_SU')
          get_src_index=n_SO2
        case default
          get_src_index=n
      end select

      end function get_src_index
!=======================================================================
      real*8 function get_src_fact(n,is_bb,OA_not_OC)
!@var src_fact Factor to multiply aerosol emissions with. Default is 1.
!@+            Notable exceptions are SO2/SO4, where one file is being read
!@+            and distributed to both tracers,and organics, where emissions
!@+            of C are multiplied with OM/OC, and VBS tracers.
!@auth Kostas Tsigaridis
      use OldTracer_mod, only: trname
      use OldTracer_mod, only: tr_mm
      use OldTracer_mod, only: om2oc
      use TRACER_COM, only: n_M_AKK_SU
#ifdef TRACERS_AEROSOLS_VBS
      use aerosol_sources, only: VBSemifactFF,VBSemifactBB
#ifdef TRACERS_AMP
      use AMP_AEROSOL, only: vbs_conc
      use AERO_CONFIG, only: nmodes,mname
#else
      use aerosol_sources, only: vbs_conc
#endif
#endif  /* TRACERS_AEROSOLS_VBS */
      implicit none
!@var n index of current tracer whose emissions factor is being seeked
!@var is_bb true if the sector is biomass burning, false otherwise
!@var OA_not_OC true if the tracer has emissions in OA units, not OC
!@var so4_fraction mole fraction of so2 to be emitted as so4
      integer, intent(in) :: n
      logical, intent(in) :: is_bb
      logical, intent(in), optional :: OA_not_OC
      real*8, parameter :: so4_fraction=0.025d0
      real*8 :: akk_fraction
      logical :: OA_not_OC_local
      integer :: get_src_index
      integer :: i

      if (n_M_AKK_SU>0) then
        akk_fraction=0.01d0
      else
        akk_fraction=0.d0
      endif

      OA_not_OC_local=.false.
      if (present(OA_not_OC)) OA_not_OC_local=OA_not_OC

      select case (trname(n))
        case ('SO2')
          get_src_fact=1.d0-so4_fraction
        case ('SO4', 'M_ACC_SU', 'ASO4__01')
          get_src_fact=so4_fraction*tr_mm(n)/tr_mm(get_src_index(n))*
     &                 (1.d0-akk_fraction)
        case ('M_AKK_SU')
          get_src_fact=so4_fraction*tr_mm(n)/tr_mm(get_src_index(n))*
     &                 akk_fraction
        case ('OCII', 'OCIA', 'OCB', 'M_OCC_OC', 'M_BOC_OC', 'AOCOB_01')
          get_src_fact=1.d0
          if (.not.OA_not_OC_local) get_src_fact=get_src_fact*om2oc(n)
#ifdef TRACERS_AEROSOLS_VBS
        case ('vbsAm2', 'vbsAm1', 'vbsAz', 'vbsAp1', 'vbsAp2',
     &        'vbsAp3', 'vbsAp4', 'vbsAp5', 'vbsAp6',
     &        'M_OCC_OCM2','M_OCC_OCM1','M_OCC_OCM0',
     &        'M_OCC_OCP1','M_OCC_OCP2','M_OCC_OCP3',
     &        'M_OCC_OCP4','M_OCC_OCP5','M_OCC_OCP6')
#ifdef TRACERS_AMP
          do i=1,nmodes
            if (mname(i)=='OCC') then ! indices from OCC are needed here
              if (is_bb) then
                get_src_fact=VBSemifactBB(vbs_conc(i)%iaerinv(n))
              else
                get_src_fact=VBSemifactFF(vbs_conc(i)%iaerinv(n))
              endif
              exit
            endif
          enddo
#else
          if (is_bb) then ! same factor for all, so just use index 1 here
            get_src_fact=VBSemifactBB(vbs_conc(1)%iaerinv(n))
          else
            get_src_fact=VBSemifactFF(vbs_conc(1)%iaerinv(n))
          endif
#endif
          if (.not.OA_not_OC_local) get_src_fact=get_src_fact*om2oc(n)
#endif  /* TRACERS_AEROSOLS_VBS */
        case default
          get_src_fact=1.d0
      end select

      end function get_src_fact
!=======================================================================
      integer function tr_con_diag(vconpts, vqcon, vqsum)
!@sum tr_con_diag populate tracer conservation diagnostics
!@auth Kostas Tsigaridis
      use TRDIAG_COM, only: ntcons,conpts,npts_common,qcon,qsum
      implicit none
!@var vconpts local value of conpts
!@var vqcon local value of qcon
!@var vqsum local value of qsum
!@var g index to be assigned to the current diagnostic
!@var i local loop index
      character(len=*), intent(in) :: vconpts
      logical, intent(in), optional :: vqcon, vqsum
      integer :: g,i

      g=0
      do i=1,ntcons ! brute force, but only happens during initialization
        if (trim(conpts(i))=='') then
          g=npts_common+i
          exit
        endif
      enddo
      if (g==0) call stop_model('ntcons too small',255)

      tr_con_diag=g
      conpts(g-npts_common)=trim(vconpts)
      if (present(vqcon)) qcon(g)=vqcon
      if (present(vqsum)) qsum(g)=vqsum

      end function tr_con_diag
!=======================================================================
      integer function ijts_diag(sname,lname,units,ia,power,denom,
     *                           scalediv)
!@sum ijts_diag populate tracer 2d diagnostics
!@auth Kostas Tsigaridis
      use TRDIAG_COM, only: ktaijs,ia_ijts,sname_ijts,lname_ijts,
     &                      units_ijts,scale_ijts,
     &                      dname_ijts
      USE MODEL_COM, only: dtsrc
      implicit none
      character(len=*), intent(in) :: sname, lname, units
      integer, intent(in), optional :: ia
      integer, intent(in), optional :: power
      character(len=*), intent(in), optional :: denom
      real*8, intent(in), optional :: scalediv
      character*50 :: unit_string
!@var sname short name
!@var lname long name
!@var units units string
!@var ia accumulation index
!@var power exponent to scale the diagnostic (10**(-power))
!@var denom denominator to be applied
!@var scalediv value to divide scale_ijts with
!@var scdiv local copy of scalediv (as defined, or 1.d0 by default)
!@var k index to be assigned to the current diagnostic
!@var i local loop index
!@var pow local copy of power (as defined, or 0 by default)
      real*8 :: scdiv
      integer :: k,i,pow

      k=0
      do i=1,ktaijs ! brute force, but only happens during initialization
        if (trim(sname_ijts(i))=='') then
          k=i
          exit
        endif
      enddo
      if (k==0) call stop_model('ktaijs too small to fit '//sname,255)

      if (present(power)) then
        pow=power
      else
        pow=0
      endif

      if (present(scalediv)) then
        scdiv=scalediv
      else
        scdiv=1.d0
      endif

      ijts_diag=k
      if (present(ia)) ia_ijts(k)=ia
      sname_ijts(k)=sname
      lname_ijts(k)=lname
      units_ijts(k)=unit_string(pow, units)
      scale_ijts(k)=10.d0**(-pow)/scdiv
      if (present(denom)) dname_ijts(k)=denom

      end function ijts_diag
!=======================================================================
      integer function ijlt_diag(sname,lname,units,ia,power,denom)
!@sum ijlt_diag populate tracer 3d diagnostics
!@auth Kostas Tsigaridis
      use TRDIAG_COM, only: ktaijls,ia_ijlt,sname_ijlt,lname_ijlt,
     &                      units_ijlt,scale_ijlt,
     &                      dname_ijlt
      implicit none
      character(len=*), intent(in) :: sname, lname, units
      integer, intent(in), optional :: ia
      integer, intent(in), optional :: power
      character(len=*), intent(in), optional :: denom
      character*50 :: unit_string
!@var sname short name
!@var lname long name
!@var units units string
!@var ia accumulation index
!@var power exponent to scale the diagnostic (10**(-power))
!@var denom denominator to be applied
!@var k index to be assigned to the current diagnostic
!@var i local loop index
!@var pow local copy of power (as defined, or 0 by default)
      integer :: k,i,pow

      k=0
      do i=1,ktaijls ! brute force, but only happens during initialization
        if (trim(sname_ijlt(i))=='') then
          k=i
          exit
        endif
      enddo
      if (k==0) call stop_model('ktaijls too small to fit '//sname,255)

      if (present(power)) then
        pow=power
      else
        pow=0
      endif

      ijlt_diag=k
      if (present(ia)) ia_ijlt(k)=ia
      sname_ijlt(k)=sname
      lname_ijlt(k)=lname
      units_ijlt(k)=unit_string(pow, units)
      scale_ijlt(k)=10.d0**(-pow)
      if (present(denom)) dname_ijlt(k)=denom

      end function ijlt_diag
!=======================================================================
      subroutine init_tracer_cons_diag
!@sum init_tracer_cons_diag Initialize tracer conservation diagnostics
!@auth Gavin Schmidt
      use AbstractAttribute_mod, only: AbstractAttribute
      use Attributes_mod, only: assignment(=)
      use Attributes_mod, only: toPointer
      use Tracer_mod
      use TracerBundle_mod
      use TracerHashMap_mod
      use AttributeDictionary_mod, only: assignment(=)
      use TracerSurfaceSource_mod, only: TracerSurfaceSource
      USE TRACER_COM, only: ntm
      USE TRACER_COM, only: noverwrite
      USE TRACER_COM, only: nvolcanic
      USE TRACER_COM, only: nOther, nThermo
      use OldTracer_mod, only: ntm_power, dowetdep, dodrydep
      use OldTracer_mod, only: tr_wd_type, nPart
      use OldTracer_mod, only: nBBsources,trname,do_fire
      use TRACER_COM, only: nchemloss, nChemprod
      use TRACER_COM, only: nchemistry, nMicrophys
      use TRACER_COM, only: nbiomass
      use TRACER_COM, only: nAircraft
      use OldTracer_mod, only: do_aircraft
      use TRACER_COM, only: nRocket
      use OldTracer_mod, only: do_rocket
      use TRACER_COM, only: ntsurfsrc
      use TRACER_COM, only: tracers
      use TRACER_COM, only: n_SO2
      use Tracer_mod, only: Tracer
      use DIAG_COM, only: conpt0,npts
#ifdef TRACERS_TOMAS
      use TRACER_COM, only: n_AH2O, n_AECOB, n_AOCOB, n_ANUM
      use TRACER_COM, only: nSO4anum, nECanum, nOCanum
      use TRACER_COM, only: n_ASO4
#endif
#ifdef TRACERS_ON
      USE TRDIAG_COM
#endif /* TRACERS_ON */
      USE FLUXES, only : atmocn
      implicit none
      character*20 sum_unit(NTM),inst_unit(NTM)   ! for conservation
      character*50 :: unit_string
      character*10 :: conpt(npts)
#ifdef TRACERS_ON
      logical :: T=.TRUE. , F=.FALSE.
      logical :: Qf
      integer n,n_src,kk
      integer, pointer :: index=> null()
      class (AbstractAttribute), pointer :: pa
      class (Tracer), pointer :: pTracer,pTracerSrc
      type (TracerSurfaceSource), pointer :: sources(:)
#endif
      type (TracerIterator) :: iter
      interface
        integer function tr_con_diag(vconpts, vqcon, vqsum)
          character(len=*), intent(in) :: vconpts
          logical, intent(in), optional :: vqcon, vqsum
        end function tr_con_diag
      end interface

#ifdef TRACERS_ON

C**** To add a new conservation diagnostic:
C****       Set up a QCON, and call SET_TCON to allocate array numbers,
C****       set up scales, titles, etc.
C**** QCON denotes when the conservation diags should be accumulated
C**** QSUM says whether that diag is to be used in summation (if the
C****      routine DIAGTCB is used, this must be false).
C**** 1:NPTS+1 ==> INST,  DYN,   COND,   RAD,   PREC,   LAND,  SURF,
C****            FILTER,STRDG/OCEAN, DAILY, OCEAN1, OCEAN2,
C**** First 12 (npts_common) are standard for all tracers and GCM
C**** Later indices are configurable - you provide title and itcon
C**** index (which is used wherever you want to check point)
C**** For example, separate Moist convection/Large scale condensation
!      itcon_mc(n)=tr_con_diag('MOIST CONV',T)
!      itcon_ss(n)=tr_con_diag('LS COND',T)

#ifdef CUBED_SPHERE
      Qf = .false.  ! no SLP filter
#else
      Qf = .true.   ! SLP filter on
#endif

      qcon(1:npts_common)=(/T,                                !instant. (1)
     *                      T, T, F, F, T, T,Qf, T, F, F, F/) !2-12 (npts)
      qcon(npts_common+1:npts_common+ntcons)=F                !13-ktcon-1
      qsum(1:npts_common)=(/F,                                !instant. (1)
     *                      T, T, F, F, T, T,Qf, T, F, F, F/) !2-12 (npts)
      qsum(npts_common+1:npts_common+ntcons)=F                !13-ktcon-1
C**** this allows you to configure the common check points names.
      conpt=conpt0

      do n=1,NTM
        kt_power_inst(n)   = ntm_power(n)+2
        kt_power_change(n) = ntm_power(n)-4
      end do

C**** set some defaults
      itcon_AMP(:,:)=0
      itcon_AMPe(:)=0
      itcon_AMPm(:,:)=0
      itcon_surf(:,:)=0
      itcon_3Dsrc(:,:)=0
      itcon_decay(:)=0
      itcon_wt(:)=0
#ifdef TRACERS_WATER
      itcon_mc(:)=0
      itcon_ss(:)=0
#endif
#ifdef TRACERS_DRYDEP
      itcon_dd(:,:)=0
#endif
#ifdef TRACERS_TOMAS
      itcon_TOMAS(:,:)=0
      itcon_subcoag(:)=0
#endif

      iter = tracers%begin()
      do while (iter /= tracers%last())
        pTracer => iter%value()
        index => toPointer(pTracer%getReference('index'), index)
        n = index

! handle exceptions first (e.g. SO4 emissions are listed under SO2)
        select case (trim(pTracer%getName()))
        case ('SO4',
     &        'M_AKK_SU','M_ACC_SU',
     &        'ASO4__01','ASO4__02','ASO4__03','ASO4__04','ASO4__05',
     &        'ASO4__06','ASO4__07','ASO4__08','ASO4__09','ASO4__10',
     &        'ASO4__11','ASO4__12','ASO4__13','ASO4__14','ASO4__15')
          n_src = n_SO2
#ifdef TRACERS_TOMAS
        case ('AECIL_01','AECIL_02','AECIL_03','AECIL_04','AECIL_05',
     &        'AECIL_06','AECIL_07','AECIL_08','AECIL_09','AECIL_10',
     &        'AECIL_11','AECIL_12','AECIL_13','AECIL_14','AECIL_15',
     &        'AECOB_01','AECOB_02','AECOB_03','AECOB_04','AECOB_05',
     &        'AECOB_06','AECOB_07','AECOB_08','AECOB_09','AECOB_10',
     &        'AECOB_11','AECOB_12','AECOB_13','AECOB_14','AECOB_15')
          n_src = n_AECOB(1)
        case ('AOCIL_01','AOCIL_02','AOCIL_03','AOCIL_04','AOCIL_05',
     &        'AOCIL_06','AOCIL_07','AOCIL_08','AOCIL_09','AOCIL_10',
     &        'AOCIL_11','AOCIL_12','AOCIL_13','AOCIL_14','AOCIL_15',
     &        'AOCOB_01','AOCOB_02','AOCOB_03','AOCOB_04','AOCOB_05',
     &        'AOCOB_06','AOCOB_07','AOCOB_08','AOCOB_09','AOCOB_10',
     &        'AOCOB_11','AOCOB_12','AOCOB_13','AOCOB_14','AOCOB_15')
          n_src = n_AOCOB(1)
#endif  /* TRACERS_TOMAS */
        case default
          n_src = n
        end select
        pTracerSrc => tracers%getReference(trname(n_src))
        sources => pTracerSrc%surfaceSources

!-----
! diagnostics for all tracers, if they meet certain conditions
!-----
        conpt(8)="SRCS+SNKS"

#ifdef TRACERS_WATER
        if(dowetdep(n)) then
          itcon_mc(n)=tr_con_diag('MOIST CONV',T)
          itcon_ss(n)=tr_con_diag('LS COND',T)
        endif
#endif
#ifdef TRACERS_DRYDEP
        if(dodrydep(n)) then
          itcon_dd(n,1)=tr_con_diag('TURB DEP',T)
          if (tr_wd_type(n)==nPart) then
            itcon_dd(n,2)=tr_con_diag('GRAV SET',T)
          endif
        end if
#endif
        if(do_aircraft(n_src))then
          itcon_3Dsrc(nAircraft,n)=tr_con_diag('Aircraft src',T,T)
        endif
        if(do_rocket(n_src))then
          itcon_3Dsrc(nRocket,n)=tr_con_diag('Rocket src',T,T)
        endif
        if (nBBsources(n_src)>0 .or. do_fire(n_src)) then
          itcon_3Dsrc(nBiomass,n)=tr_con_diag('Biomass src',T,T)
        endif
        do kk=1,ntsurfsrc(n_src)
          itcon_surf(kk,n)=tr_con_diag(trim(sources(kk)%sourceLname),T)
        enddo

!-----
!     per-tracer diagnostics
!     Note: Do not include already registered surface sources
!-----
        select case (trim(pTracer%getName()))

        case ('CO2n')
          qcon(10) = .true.
          qsum(10) = .true.

        case ('Rn222')
          itcon_decay(n)=tr_con_diag('DECAY',T,T)

        case ('N2O')   ! two versions dependent on configuration
#ifdef TRACERS_SPECIAL_Lerner
c          itcon_surf(1,N)=tr_con_diag('Reset in L1',T)
          itcon_3Dsrc(nChemistry,n)=tr_con_diag('Strat. Chem.',T,T)
#endif
#ifdef TRACERS_SPECIAL_Shindell
          kt_power_change(n) = -14
          itcon_3Dsrc(nChemistry,n)=tr_con_diag('Chemistry',T,T)
          itcon_3Dsrc(nOverwrite,n)=tr_con_diag('Overwrite',T,T)
#endif

        case ('CFC11')
c          itcon_surf(1,N)=tr_con_diag('L1 Source',T)
          itcon_3Dsrc(nChemistry,n)=tr_con_diag('Strat. Chem.',T,T)

        case ('14CO2')
          itcon_surf(1,N)=tr_con_diag('Bombs and drift',T)


        case ('nh5','nh15','nh50','e90','CO50')
          kt_power_change(n) = -13
          itcon_decay(n)=tr_con_diag('DECAY',T,T)

        case ('aoa','aoanh')
          kt_power_change(n) = -13
          itcon_decay(n)=tr_con_diag('DECAY',T,T)
          kt_power_change(n) = -17
          itcon_3Dsrc(1,n)=tr_con_diag('L1 overwriting',T,T)


        case ('CH4')            ! two versions
#ifdef TRACERS_SPECIAL_Shindell
          kt_power_change(n) = -13
          itcon_3Dsrc(nChemistry,n)=tr_con_diag('Chemistry',T,T)
          itcon_3Dsrc(nOverwrite,n)=tr_con_diag('Overwrite',T,T)
#endif /* TRACERS_SPECIAL_Shindell */
#ifdef TRACERS_SPECIAL_Lerner
          itcon_3Dsrc(1,n)=tr_con_diag('Tropos. Chem.',T,T)
          itcon_3Dsrc(2,n)=tr_con_diag('Stratos. Chem.',T,T)
#endif /* TRACERS_SPECIAL_Lerner */

        case ('O3')
c          itcon_surf(1,N)=tr_con_diag('Deposition',T)
          itcon_3Dsrc(1,n)=tr_con_diag('Stratos. Chem.',T,T)
          itcon_3Dsrc(2,n)=tr_con_diag('Trop. Chem. Prod.',T,T)
          itcon_3Dsrc(3,n)=tr_con_diag('Trop. Chem. Loss',T,T)

        case ('Ox','N2O5','HNO3','H2O2','CH3OOH','HCHO','HO2NO2','PAN'
     *       ,'AlkylNit','ClOx','BrOx','HCl','HOCl','ClONO2','HBr'
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
#ifdef TRACERS_dCO
     *       ,'d13Calke','d13CPAR'
     *       ,'d17OPAN', 'd18OPAN', 'd13CPAN'
     *       ,'dMe17OOH', 'dMe18OOH', 'd13MeOOH'
     *       ,'dHCH17O', 'dHCH18O', 'dH13CHO'
#endif  /* TRACERS_dCO */
     *       ,'dC17O', 'dC18O', 'd13CO'
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
     *       ,'HOBr','BrONO2','CFC','NOx','CO','Isoprene','Alkenes'
     *       ,'Paraffin','Terpenes','Acetone') ! N2O done above
          select case (trim(pTracer%getName()))
            case ('N2O5','CH3OOH','HCHO','HO2NO2','PAN','AlkylNit','CFC'
#ifdef TRACERS_dCO
     *           ,'d17OPAN', 'd18OPAN', 'd13CPAN'
     *           ,'dMe17OOH', 'dMe18OOH', 'd13MeOOH'
     *           ,'dHCH17O', 'dHCH18O', 'dH13CHO'
#endif  /* TRACERS_dCO */
     *           ,'ClOx','BrOx','HCl','HOCl','ClONO2','HBr','HOBr'
     *           ,'BrONO2','NOx')
              kt_power_change(n) = -14
            case ('HNO3','H2O2','CO','Isoprene','Alkenes','Paraffin'
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
#if TRACERS_dCO
     *           ,'d13Calke','d13CPAR'
#endif  /* TRACERS_dCO */
     *           ,'dC17O', 'dC18O', 'd13CO'
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
     *           ,'Terpenes','Acetone')
              kt_power_change(n) = -13
            case default
              kt_power_change(n) = -12
          end select

          itcon_3Dsrc(nChemistry,n)=tr_con_diag('Chemistry',T,T)
          itcon_3Dsrc(nOverwrite,n)=tr_con_diag('Overwrite',T,T)
          select case(trim(pTracer%getName()))
          case ('NOx')
            itcon_3Dsrc(nOther,n)=tr_con_diag('Lightning',T,T)
#ifdef TRACERS_NITRATE
          case ('HNO3')
            itcon_3Dsrc(nThermo,n)=tr_con_diag('Thermodynamics',T,T)
#endif
          end select

#ifdef TRACERS_AEROSOLS_SOA
        case ('isopp1g','isopp1a','isopp2g','isopp2a',
     &        'apinp1g','apinp1a','apinp2g','apinp2a')
          itcon_3Dsrc(nChemistry,n)=tr_con_diag('Chemistry',T,T)
#endif  /* TRACERS_AEROSOLS_SOA */

        case ('GLT')
          kt_power_change(n) = -17
          itcon_3Dsrc(1,n)=tr_con_diag('L1 overwriting',T,T)

        case ('HTO')
          itcon_decay(n)=tr_con_diag('DECAY',T,T)

        case ('DMS')
          itcon_surf(1,n)=tr_con_diag('Ocean src',T)
          itcon_3Dsrc(nChemistry,n)=tr_con_diag('Chemistry',T,T)

        case ('MSA')
          itcon_3Dsrc(nChemistry,n)=tr_con_diag('Chemistry',T,T)

        case ('SO2')
          itcon_3Dsrc(nVolcanic,n)=tr_con_diag('Volcanic src',T,T)
          itcon_3Dsrc(nChemprod,n)=tr_con_diag('Chem. src',T,T)
          itcon_3Dsrc(nChemLoss,n)=tr_con_diag('Chem. sink',T,T)

        case ('SO4')
          itcon_3Dsrc(nChemistry,n)=tr_con_diag('Gas phase src',T,T)
          itcon_3Dsrc(nVolcanic,n)=tr_con_diag('Volcanic src',T,T)

        case ('BCII', 'BCIA', 'BCB', 'OCII', 'OCIA', 'OCB',
     &        'vbsGm2', 'vbsGm1', 'vbsGz',  'vbsGp1', 'vbsGp2',
     &        'vbsGp3', 'vbsGp4', 'vbsGp5', 'vbsGp6',
     &        'vbsAm2', 'vbsAm1', 'vbsAz',  'vbsAp1', 'vbsAp2',
     &        'vbsAp3', 'vbsAp4', 'vbsAp5', 'vbsAp6')
          select case(trim(pTracer%getName()))
          case ('vbsGm2', 'vbsGm1', 'vbsGz',  'vbsGp1', 'vbsGp2',
     &          'vbsGp3', 'vbsGp4', 'vbsGp5', 'vbsGp6')
            itcon_3Dsrc(nChemistry,n)=tr_con_diag('Aging source',T,T)
            itcon_3Dsrc(nChemLoss,n)=tr_con_diag('Aging loss',T,T)
            itcon_3Dsrc(nOther,n)=tr_con_diag('Part. loss',T,T)
          case ('vbsAm2', 'vbsAm1', 'vbsAz',  'vbsAp1', 'vbsAp2',
     &          'vbsAp3', 'vbsAp4', 'vbsAp5', 'vbsAp6')
            itcon_3Dsrc(nChemistry,n)=tr_con_diag('Part. source',T,T)
          case ('BCII', 'OCII')
            itcon_3Dsrc(nChemistry,n)=tr_con_diag('Aging loss',T,T)
          case ('BCIA', 'OCIA')
            itcon_3Dsrc(nChemistry,n)=tr_con_diag('Aging source',T,T)
          end select

        case ('SO4_d1', 'SO4_d2','SO4_d3','N_d1','N_d2','N_d3')
          itcon_3Dsrc(nChemistry,n)=tr_con_diag('Gas phase change',T,T)

        case ('H2SO4')
          itcon_3Dsrc(nMicrophys,n)=tr_con_diag('Microphysics',T,T)

        case ('NH4', 'NO3p', 'NH3')
          itcon_3Dsrc(nThermo,n)=tr_con_diag('Thermodynamics',T,T)

        case ('Be7', 'Be10')
          itcon_3Dsrc(1,n)=tr_con_diag('COSMO SRC',T,T)
          if (trim(pTracer%getName()).eq."Be7") then
            itcon_decay(n)=tr_con_diag('DECAY',T,T)
          end if

        case ('Pb210')
          itcon_3Dsrc(nChemistry,n)=tr_con_diag('RADIO SRC',T,T)
          itcon_decay(n)=tr_con_diag('DECAY',T,T)

        case ('H2O2_s')
          itcon_3Dsrc(1,n)=tr_con_diag('Gas phase src',T,T)
          itcon_3Dsrc(2,n)=tr_con_diag('Gas phase sink',T,T)

#ifndef TRACERS_WATER
        case ('seasalt1','seasalt2','OCocean'
     &         ,'Clay','Silt1','Silt2','Silt3','Silt4','Silt5'
     &         ,'ClayIlli' ,'ClayKaol','ClaySmec','ClayCalc','ClayQuar'
     &         ,'ClayFeld' ,'ClayHema','ClayGyps','ClayIlHe','ClayKaHe'
     &         ,'ClaySmHe' ,'ClayCaHe','ClayQuHe','ClayFeHe','ClayGyHe'
     &         ,'Sil1Quar' ,'Sil1Feld','Sil1Calc','Sil1Hema','Sil1Gyps'
     &         ,'Sil1Illi' ,'Sil1Kaol','Sil1Smec','Sil1QuHe','Sil1FeHe'
     &         ,'Sil1CaHe' ,'Sil1GyHe','Sil1IlHe','Sil1KaHe','Sil1SmHe'
     &         ,'Sil2Quar' ,'Sil2Feld','Sil2Calc','Sil2Hema','Sil2Gyps'
     &         ,'Sil2Illi' ,'Sil2Kaol','Sil2Smec','Sil2QuHe','Sil2FeHe'
     &         ,'Sil2CaHe' ,'Sil2GyHe','Sil2IlHe','Sil2KaHe','Sil2SmHe'
     &         ,'Sil3Quar' ,'Sil3Feld','Sil3Calc','Sil3Hema','Sil3Gyps'
     &         ,'Sil3Illi' ,'Sil3Kaol','Sil3Smec','Sil3QuHe','Sil3FeHe'
     &         ,'Sil3CaHe' ,'Sil3GyHe','Sil3IlHe','Sil3KaHe','Sil3SmHe'
     &         ,'Sil4Quar' ,'Sil4Feld','Sil4Calc','Sil4Hema','Sil4Gyps'
     &         ,'Sil4Illi' ,'Sil4Kaol','Sil4Smec','Sil4QuHe','Sil4FeHe'
     &         ,'Sil4CaHe' ,'Sil4GyHe','Sil4IlHe','Sil4KaHe','Sil4SmHe'
     &         ,'Sil5Quar' ,'Sil5Feld','Sil5Calc','Sil5Hema','Sil5Gyps'
     &         ,'Sil5Illi' ,'Sil5Kaol','Sil5Smec','Sil5QuHe','Sil5FeHe'
     &         ,'Sil5CaHe' ,'Sil5GyHe','Sil5IlHe','Sil5KaHe','Sil5SmHe')
          itcon_wt(n)=tr_con_diag('WET DEP',T)
#endif  /* not TRACERS_WATER */

c- Species including AMP  emissions - 2D sources and 3D sources
        case('M_AKK_SU','M_ACC_SU','M_OCC_OC','M_BC1_BC',
     *       'M_SSA_SS','M_SSC_SS','M_SSS_SS','M_DD1_DU','M_DD2_DU',
     *       'M_BOC_BC','M_BOC_OC',
     *       'M_NO3   ','M_NH4   ','M_H2O   ','M_DD1_SU',
     *       'M_DS1_SU','M_DS1_DU','M_DD2_SU',
     *       'M_DS2_SU','M_DS2_DU','M_SSA_SU',
     *       'M_OCC_SU','M_BC1_SU',
     *       'M_BC2_SU','M_BC2_BC','M_BC3_SU',
     *       'M_BC3_BC','M_DBC_SU','M_DBC_BC','M_DBC_DU',
     *       'M_BOC_SU',
     *       'M_BCS_SU','M_BCS_BC','M_MXX_SU','M_MXX_BC',
     *       'M_MXX_OC','M_MXX_DU','M_MXX_SS','M_OCS_SU',
     *       'M_OCS_OC','M_SSS_SU')
          select case (trim(pTracer%getName()))
          case ('M_NO3', 'M_NH4', 'M_H2O')
            itcon_3Dsrc(nThermo,n)=tr_con_diag('Thermodynamics',T,T)
          case default
            itcon_3Dsrc(nMicrophys,n)=tr_con_diag('Microphysics',T,T)
          end select
          select case (trim(pTracer%getName()))
            case ('M_SSA_SS','M_SSC_SS','M_SSS_SS','M_DD1_DU'
     *           ,'M_DD2_DU')
              itcon_surf(1,n)=tr_con_diag('Emission 2D AMP',T,T)
            case ('M_AKK_SU','M_ACC_SU',
     &            'M_BC1_BC','M_OCC_OC','M_BOC_BC','M_BOC_OC')
              select case (trim(pTracer%getName()))
              case ('M_AKK_SU','M_ACC_SU')
                itcon_3Dsrc(nVolcanic,n)=tr_con_diag('Volcanic src',T,T)
              end select
          end select
c Processes AMP Budget
          itcon_AMP(1,n)=tr_con_diag('P1 Nucleation',T,T)
          itcon_AMP(2,n)=tr_con_diag('P2 Coagulation',T,T)
          itcon_AMP(3,n)=tr_con_diag('P3 Condensation',T,T)
          itcon_AMP(4,n)=tr_con_diag('P4 Incloud',T,T)
          itcon_AMP(5,n)=tr_con_diag('P5 Intermode Loss',T,T)
          itcon_AMP(6,n)=tr_con_diag('P6 Mode Transf',T,T)
          itcon_AMP(7,n)=tr_con_diag('P7 AMP Budget',T,T)

        case ('N_AKK_1 ','N_ACC_1 ','N_DD1_1 ','N_DS1_1 ','N_DD2_1 '
     *       ,'N_DS2_1 ','N_SSA_1 ','N_SSC_1 ','N_OCC_1 ','N_BC1_1 '
     *       ,'N_BC2_1 ','N_BC3_1 ','N_DBC_1 ','N_BOC_1 ','N_BCS_1 '
     *       ,'N_MXX_1 ','N_OCS_1 ')

          kt_power_change(n) = 5
          kt_power_inst(n) = 3

          itcon_3Dsrc(nChemistry,n)=tr_con_diag('Gas phase change',T,T)
          itcon_AMPm(1,n)=tr_con_diag('Wet Diameter',T)
          itcon_AMPm(2,n)=tr_con_diag('Mode AktivPart',T)
          itcon_AMPm(3,n)=tr_con_diag('Dry Diameter',T)
c     Processes AMP Budget
          itcon_AMP(1,n)=tr_con_diag('P1 Nucleation',T,T)
          itcon_AMP(2,n)=tr_con_diag('P2 Coagulation',T,T)
          itcon_AMP(3,n)=tr_con_diag('P3 NOTHING',T,T)
          itcon_AMP(4,n)=tr_con_diag('P4 Intermode Coag',T,T)
          itcon_AMP(5,n)=tr_con_diag('P5 Intramode Tr',T,T)
          itcon_AMP(6,n)=tr_con_diag('P6 Mode Transf',T,T)
          itcon_AMP(7,n)=tr_con_diag('P7 AMP Budget',T,T)

#ifdef TRACERS_TOMAS

        case ('SOAgas')
!TOMAS - here needs lots of work~!
          itcon_3Dsrc(1,n)=tr_con_diag('Microphysics change',T,T)

       case('ASO4__01','ASO4__02','ASO4__03','ASO4__04','ASO4__05',
     *    'ASO4__06','ASO4__07','ASO4__08','ASO4__09','ASO4__10',
     *    'ASO4__11','ASO4__12','ASO4__13','ASO4__14','ASO4__15',
     *    'ANACL_01','ANACL_02','ANACL_03','ANACL_04','ANACL_05',
     *    'ANACL_06','ANACL_07','ANACL_08','ANACL_09','ANACL_10',
     *    'ANACL_11','ANACL_12','ANACL_13','ANACL_14','ANACL_15',
     *    'AECIL_01','AECIL_02','AECIL_03','AECIL_04','AECIL_05',
     *    'AECIL_06','AECIL_07','AECIL_08','AECIL_09','AECIL_10',
     *    'AECIL_11','AECIL_12','AECIL_13','AECIL_14','AECIL_15',
     *    'AECOB_01','AECOB_02','AECOB_03','AECOB_04','AECOB_05',
     *    'AECOB_06','AECOB_07','AECOB_08','AECOB_09','AECOB_10',
     *    'AECOB_11','AECOB_12','AECOB_13','AECOB_14','AECOB_15',
     *    'AOCIL_01','AOCIL_02','AOCIL_03','AOCIL_04','AOCIL_05',
     *    'AOCIL_06','AOCIL_07','AOCIL_08','AOCIL_09','AOCIL_10',
     *    'AOCIL_11','AOCIL_12','AOCIL_13','AOCIL_14','AOCIL_15',
     *    'AOCOB_01','AOCOB_02','AOCOB_03','AOCOB_04','AOCOB_05',
     *    'AOCOB_06','AOCOB_07','AOCOB_08','AOCOB_09','AOCOB_10',
     *    'AOCOB_11','AOCOB_12','AOCOB_13','AOCOB_14','AOCOB_15',
     *    'ADUST_01','ADUST_02','ADUST_03','ADUST_04','ADUST_05',
     *    'ADUST_06','ADUST_07','ADUST_08','ADUST_09','ADUST_10',
     *    'ADUST_11','ADUST_12','ADUST_13','ADUST_14','ADUST_15',
     *    'ANUM__01','ANUM__02','ANUM__03','ANUM__04','ANUM__05',
     *    'ANUM__06','ANUM__07','ANUM__08','ANUM__09','ANUM__10',
     *    'ANUM__11','ANUM__12','ANUM__13','ANUM__14','ANUM__15')
         
          itcon_3Dsrc(nMicrophys,n)=tr_con_diag('Microphysics',T,T)
c     Processes TOMAS Budget
          itcon_TOMAS(1,n)=tr_con_diag('Condensation',T)
          itcon_TOMAS(2,n)=tr_con_diag('Coagulation',T)
          itcon_TOMAS(3,n)=tr_con_diag('Nucleation',T)
          itcon_TOMAS(4,n)=tr_con_diag('Aqoxid SO4 MCV',T)
          itcon_TOMAS(5,n)=tr_con_diag('Aqoxid SO4 LGS',T)
          itcon_TOMAS(6,n)=tr_con_diag('Mk_Nk Fix',T)
          itcon_TOMAS(7,n)=tr_con_diag('Aeroupdate',T)
          itcon_subcoag(n)=tr_con_diag('subgrid coag',T)

       select case (trim(pTracer%getName()))

         case ('ASO4__01','ASO4__02','ASO4__03','ASO4__04','ASO4__05',
     *        'ASO4__06','ASO4__07','ASO4__08','ASO4__09','ASO4__10',
     *        'ASO4__11','ASO4__12','ASO4__13','ASO4__14','ASO4__15')

         itcon_3Dsrc(nVolcanic,n)=tr_con_diag('Volcanic src',T,T)

         case ('AECOB_01','AECOB_02','AECOB_03','AECOB_04','AECOB_05',
     *        'AECOB_06','AECOB_07','AECOB_08','AECOB_09','AECOB_10',
     *        'AECOB_11','AECOB_12','AECOB_13','AECOB_14','AECOB_15',
     *        'AECIL_01','AECIL_02','AECIL_03','AECIL_04','AECIL_05',
     *        'AECIL_06','AECIL_07','AECIL_08','AECIL_09','AECIL_10',
     *        'AECIL_11','AECIL_12','AECIL_13','AECIL_14','AECIL_15')

          itcon_3Dsrc(nChemistry,n)=tr_con_diag('ECOB Aging',T,T)

         case ('AOCOB_01','AOCOB_02','AOCOB_03','AOCOB_04','AOCOB_05',
     *        'AOCOB_06','AOCOB_07','AOCOB_08','AOCOB_09','AOCOB_10',
     *        'AOCOB_11','AOCOB_12','AOCOB_13','AOCOB_14','AOCOB_15',
     *        'AOCIL_01','AOCIL_02','AOCIL_03','AOCIL_04','AOCIL_05',
     *        'AOCIL_06','AOCIL_07','AOCIL_08','AOCIL_09','AOCIL_10',
     *        'AOCIL_11','AOCIL_12','AOCIL_13','AOCIL_14','AOCIL_15')

          itcon_3Dsrc(nChemistry,n)=tr_con_diag('OCOB Aging',T,T)

c     - Species including TOMAS  emissions - 2D sources and 3D sources
         case('ANACL_01','ANACL_02','ANACL_03','ANACL_04','ANACL_05',
     *        'ANACL_06','ANACL_07','ANACL_08','ANACL_09','ANACL_10',
     *        'ANACL_11','ANACL_12','ANACL_13','ANACL_14','ANACL_15')

          itcon_surf(1,n)=tr_con_diag('2D src',T)

         case('ANUM__01','ANUM__02','ANUM__03','ANUM__04','ANUM__05',
     *        'ANUM__06','ANUM__07','ANUM__08','ANUM__09','ANUM__10',
     *        'ANUM__11','ANUM__12','ANUM__13','ANUM__14','ANUM__15')

          itcon_3Dsrc(nSO4anum,n)=tr_con_diag('SO4 3D src',T,T)
          itcon_3Dsrc(nECanum,n)=tr_con_diag('EC 3D src',T,T)
          itcon_3Dsrc(nOCanum,n)=tr_con_diag('OC 3D src',T,T)
          itcon_surf(1,n)=tr_con_diag('2D src by SO4',T)
          itcon_surf(2,n)=tr_con_diag('2D src by EC',T)
          itcon_surf(3,n)=tr_con_diag('2D src by OC',T)
          itcon_surf(4,n)=tr_con_diag('2D src by SS',T)
          itcon_surf(5,n)=tr_con_diag('2D src by DU',T)

         case('ADUST_01','ADUST_02','ADUST_03','ADUST_04','ADUST_05',
     *        'ADUST_06','ADUST_07','ADUST_08','ADUST_09','ADUST_10',
     *        'ADUST_11','ADUST_12','ADUST_13','ADUST_14','ADUST_15')
          itcon_surf(1,n)=tr_con_diag('2D src',T)

       end select
#endif /* TRACERS_TOMAS */
        end select

        scale_inst(n)   = 10d0**(-kt_power_inst(n))
        scale_change(n) = 10d0**(-kt_power_change(n))
#ifdef TRACERS_TOMAS
        if(n.ge.n_ANUM(1).and.n.lt.n_AH2O(1))THEN
        inst_unit(n) = unit_string(kt_power_inst(n),  '# m-2)')
        sum_unit(n)  = unit_string(kt_power_change(n),'# m-2 s-1)')
        else
        inst_unit(n) = unit_string(kt_power_inst(n),  'kg m-2)')
        sum_unit(n)  = unit_string(kt_power_change(n),'kg m-2 s-1)')
        endif
#else
        inst_unit(n) = unit_string(kt_power_inst(n),  'kg m-2)')
        sum_unit(n)  = unit_string(kt_power_change(n),'kg m-2 s-1)')
#endif

        CALL SET_TCON(QCON,pTracer%getName(),QSUM,inst_unit(n),
     *       sum_unit(n),scale_inst(n),scale_change(n), N,CONPTs,CONPT)
        qcon(npts_common+1:) = .false. ! reset to defaults for next tracer
        qsum(npts_common+1:) = .false. ! reset to defaults for next tracer
        qcon(10)  = .false.     ! reset to defaults for next tracer
        qsum(10)  = .false.     ! reset to defaults for next tracer
        conpts=''
        conpt=conpt0

        call iter%next()
      end do

      natmtrcons = tracers%size()

#ifdef TRACERS_OCEAN
      atmocn%natmtrcons = natmtrcons
#endif

#endif /* TRACERS_ON */

      return
      end subroutine init_tracer_cons_diag

      subroutine init_jls_diag
!@sum init_jls_diag Initialise zonal mean/height tracer diags
!@auth Gavin Schmidt
      use Tracer_mod, only: Tracer
      use TracerSurfaceSource_mod, only: TracerSurfaceSource
      use TRACER_COM, only: ntm
      USE DOMAIN_DECOMP_ATM, only: AM_I_ROOT
      use TimeConstants_mod, only: SECONDS_PER_DAY
      USE MODEL_COM, only: dtsrc
      use TRACER_COM, only: coupled_chem
      use TRACER_COM, only: n_SO2, nAircraft, nbiomass, nchemistry
      use TRACER_COM, only: nMicrophys, nChemprod
      use TRACER_COM, only: nOther, nOverwrite, nVolcanic, nChemloss
      use TRACER_COM, only: ntsurfsrc, tracers, aqchem_list
      use OldTracer_mod, only: do_aircraft
      use TRACER_COM, only: nRocket
      use OldTracer_mod, only: do_rocket
#ifdef TRACERS_TOMAS
      use TRACER_COM, only: n_ANUM, n_AECOB, n_AOCOB
      use TRACER_COM, only: nSO4anum, nECanum, nOCanum
#endif
      USE DIAG_COM
#ifdef TRACERS_ON
      USE TRDIAG_COM
#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS)
      use trdust_mod, only: nDustEmjl, nDustEm2jl, nDustEv1jl,
     &   nDustEv2jl, nDustWthjl, imDust
#endif
#if (defined TRACERS_WATER) && (defined TRDIAG_WETDEPO)
      USE CLOUDS, ONLY : diag_wetdep
#endif
#endif /* TRACERS_ON */
      use OldTracer_mod, only: trname, ntm_power, src_dist_index,
     &                         nBBsources
      implicit none
      integer k,n,kk,ltop,n_src
      character*50 :: unit_string
      class (Tracer), pointer :: pTracer
      type (TracerSurfaceSource), pointer :: sources(:)
      type (TracerSurfaceSource), pointer :: SO2sources(:)
      type (TracerSurfaceSource), pointer :: AECOB01sources(:)
      type (TracerSurfaceSource), pointer :: AOCOB01sources(:)
      character(len=7) :: tend_units
      character(len=7) :: number_flux_units
      character(len=7) :: mass_flux_units

      tend_units = 'kg kg-1 s-1'
      number_flux_units = '# m-2 s-1'
      mass_flux_units = 'kg m-2 s-1'

C**** Please note that short names for diags i.e. sname_jls are used
C**** in special ways and MUST NOT contain spaces, commas or % signs.
C**** Underscores and minus signs are allowed.

C**** Define a max layer for some optionally trop/strat tracers
      LTOP = LM

#ifdef TRACERS_ON
C**** Tracer sources and sinks
C**** Defaults for jls (sources, sinks, etc.)
C**** These need to be 'hand coded' depending on circumstances
      do k=1,ktajls             ! max number of sources and sinks
        jgrid_jls(k) = 1
        jwt_jls(k) = jls_mass_weighted ! normal case: mass weighting
        ia_jls(k) = ia_src
        scale_jls(k) = 1./DTsrc
      end do
      jls_grav=0
#ifdef TRACERS_WATER
C**** set defaults for some precip/wet-dep related diags
      jls_prec(:,:)=0
#endif

      k = 0
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS)
      pTracer => tracers%getReference('SO2')
      SO2sources => pTracer%surfaceSources
#endif
#ifdef TRACERS_TOMAS
      pTracer => tracers%getReference('AECOB_01')
      AECOB01sources => pTracer%surfaceSources
      pTracer => tracers%getReference('AOCOB_01')
      AOCOB01sources => pTracer%surfaceSources
#endif
      do n=1,NTM
        if (src_dist_index(n)/=0) cycle

!=============================================!
! emissions for all tracers, if they have any !
!=============================================!

! handle exceptions first (e.g. SO4 emissions are listed under SO2)
      select case (trname(n))
      case ('SO4',
     &      'M_AKK_SU','M_ACC_SU',
     &      'ASO4__01','ASO4__02','ASO4__03','ASO4__04','ASO4__05',
     &      'ASO4__06','ASO4__07','ASO4__08','ASO4__09','ASO4__10',
     &      'ASO4__11','ASO4__12','ASO4__13','ASO4__14','ASO4__15')
        n_src = n_SO2
#ifdef TRACERS_TOMAS
      case ('AECIL_01','AECIL_02','AECIL_03','AECIL_04','AECIL_05',
     &      'AECIL_06','AECIL_07','AECIL_08','AECIL_09','AECIL_10',
     &      'AECIL_11','AECIL_12','AECIL_13','AECIL_14','AECIL_15',
     &      'AECOB_01','AECOB_02','AECOB_03','AECOB_04','AECOB_05',
     &      'AECOB_06','AECOB_07','AECOB_08','AECOB_09','AECOB_10',
     &      'AECOB_11','AECOB_12','AECOB_13','AECOB_14','AECOB_15')
        n_src = n_AECOB(1)
      case ('AOCIL_01','AOCIL_02','AOCIL_03','AOCIL_04','AOCIL_05',
     &      'AOCIL_06','AOCIL_07','AOCIL_08','AOCIL_09','AOCIL_10',
     &      'AOCIL_11','AOCIL_12','AOCIL_13','AOCIL_14','AOCIL_15',
     &      'AOCOB_01','AOCOB_02','AOCOB_03','AOCOB_04','AOCOB_05',
     &      'AOCOB_06','AOCOB_07','AOCOB_08','AOCOB_09','AOCOB_10',
     &      'AOCOB_11','AOCOB_12','AOCOB_13','AOCOB_14','AOCOB_15')
        n_src = n_AOCOB(1)
#endif  /* TRACERS_TOMAS */
      case default
        n_src = n
      end select
      pTracer => tracers%getReference(trname(n_src))
      sources => pTracer%surfaceSources

! aqueous chemistry sources and sinks
      if (allocated(aqchem_list)) then
      if (any(n.eq.aqchem_list)) then
        k = k + 1
        jls_incloud(1,n) = k
        sname_jls(k) = trim(trname(n))//'_mc_cloud_aqchem'
        lname_jls(k) = trim(trname(n))//' mc cloud aqchem'
        jls_ltop(k) = LM
        jls_power(k) = 0
        units_jls(k) = unit_string(jls_power(k),tend_units)

        k = k + 1
        jls_incloud(2,n) = k
        sname_jls(k) = trim(trname(n))//'_ss_cloud_aqchem'
        lname_jls(k) = trim(trname(n))//' ss cloud aqchem'
        jls_ltop(k) = LM
        jls_power(k) = 0
        units_jls(k) = unit_string(jls_power(k),tend_units)
      endif
      endif

! surface emissions
      do kk=1,ntsurfsrc(n_src)
        k = k + 1
        jls_source(kk,n) = k
        sname_jls(k) = trim(trname(n))//'_'//
     &                 trim(sources(kk)%sourceName)
        lname_jls(k) = trim(trname(n))//' '//
     &                 trim(sources(kk)%sourceLname)
        jls_ltop(k) = 1
        jls_power(k) = ntm_power(n)+11
        select case(trname(n))
        case ('ANUM__01','ANUM__02','ANUM__03','ANUM__04','ANUM__05',
     *    'ANUM__06','ANUM__07','ANUM__08','ANUM__09','ANUM__10',
     *    'ANUM__11','ANUM__12','ANUM__13','ANUM__14','ANUM__15')
          units_jls(k) = unit_string(jls_power(k),number_flux_units)
        case default
          units_jls(k) = unit_string(jls_power(k),mass_flux_units)
        end select
        jwt_jls(k) = jls_not_mass_weighted
      end do

! aircraft emissions
      if(do_aircraft(n_src)) then
        k = k + 1
        jls_3Dsource(nAircraft,n) = k
        sname_jls(k) = trim(trname(n))//'_aircraft_src'
        lname_jls(k) = trim(trname(n))//' aircraft source'
        jls_ltop(k) = LM
        jls_power(k) = -2
        units_jls(k) = unit_string(jls_power(k),tend_units)
      end if

! rocket emissions
      if(do_rocket(n_src)) then
        k = k + 1
        jls_3Dsource(nRocket,n) = k
        sname_jls(k) = trim(trname(n))//'_rocket_src'
        lname_jls(k) = trim(trname(n))//' rocket source'
        jls_ltop(k) = LM
        jls_power(k) = -2
        units_jls(k) = unit_string(jls_power(k),tend_units)
      end if

! biomass burning emissions
      if (nBBsources(n_src) .gt. 0) then
        k = k + 1
        jls_3Dsource(nBiomass,n) = k
        sname_jls(k) = trim(trname(n))//'_biomass_src'
        lname_jls(k) = trim(trname(n))//' biomass source'
        jls_ltop(k) = LM
        jls_power(k) = -2
        units_jls(k) = unit_string(jls_power(k),tend_units)
      endif

!=============================!
! Tracer-specific diagnostics !
!=============================!
      select case (trname(n))

c      case ('SF6','SF6_c','nh5','nh50','e90','st8025','tape_rec','aoa','aoanh','nh15')
c        call layer1_init_jls(k,n,trname(n))
c      case ('CFCn')
c        call layer1_init_jls(k,n,trname(n))
      case ('CO2n')
        call CO2n_init_jls(k,n,'CO2n')
      case ('Rn222')
        call Rn222_init_jls(k,n,'Rn222')
      case ('N2O')
        call N2O_init_jls(k,n,'N2O')
      case ('CFC11')   !!! should start April 1
        k = k + 1
        jls_3Dsource(1,n) = k
        sname_jls(k) = 'Stratos_chem_change_'//trim(trname(n))
        lname_jls(k) = 'CHANGE OF '//trim(trname(n))//
     &                 ' BY CHEMISTRY IN STRATOS'
        jls_ltop(k) = lm
        jls_power(k) = -3
        units_jls(k) = unit_string(jls_power(k),tend_units)

      case ('CH4')
#ifdef TRACERS_SPECIAL_Shindell
        k = k + 1
        jls_3Dsource(nChemistry,n) = k
        sname_jls(k) = 'chemistry_source_of_'//trim(trname(n))
        lname_jls(k) = 'CHANGE OF '//trim(trname(n))//' BY CHEMISTRY'
        jls_ltop(k) = LTOP
        jls_power(k) = 0
        units_jls(k) = unit_string(jls_power(k),tend_units)
        k = k + 1
        jls_3Dsource(nOverwrite,n) = k
        sname_jls(k) = 'overwrite_source_of_'//trim(trname(n))
        lname_jls(k) = 'CHANGE OF '//trim(trname(n))//' BY OVERWRITE'
        jls_ltop(k) = LM
        jls_power(k) = 0
        units_jls(k) = unit_string(jls_power(k),tend_units)
#else
        k = k + 1
        jls_3Dsource(1,n) = k
        sname_jls(k) = 'Tropos_Chem_change_'//trim(trname(n))
        lname_jls(k) = 'CHANGE OF '//trim(trname(n))//
     &                 ' BY CHEMISTRY IN TROPOSPHERE'
        jls_ltop(k) = lm
        jls_power(k) = -1
        units_jls(k) = unit_string(jls_power(k),tend_units)
        k = k + 1
        jls_3Dsource(2,n) = k
        sname_jls(k) = 'Stratos_Chem_change_'//trim(trname(n))
        lname_jls(k) = 'CHANGE OF '//trim(trname(n))//
     &                 ' BY CHEMISTRY IN STRATOS'
        jls_ltop(k) = lm
        jls_power(k) = -1
        units_jls(k) = unit_string(jls_power(k),tend_units)
#endif

      case ('O3')
       k = k + 1
        jls_3Dsource(1,n) = k
        sname_jls(k) = 'Strat_Chem_change_'//trim(trname(n))
        lname_jls(k) = 'Change of O3 by Chemistry in Stratos'
        jls_ltop(k) = lm
        jls_power(k) = 1
        units_jls(k) = unit_string(jls_power(k),tend_units)
       k = k + 1
        jls_3Dsource(2,n) = k
        sname_jls(k) = 'Trop_Chem_Prod_change_'//trim(trname(n))
        lname_jls(k) = 'Change of O3 by Chem Prod. in Troposphere'
        jls_ltop(k) = lm
        jls_power(k) = 1
        units_jls(k) = unit_string(jls_power(k),tend_units)
       k = k + 1
        jls_3Dsource(3,n) = k
        sname_jls(k) = 'Trop_Chem_Loss_change_'//trim(trname(n))
        lname_jls(k) = 'Change of O3 by Chem Loss in Troposphere'
        jls_ltop(k) = lm
        jls_power(k) = 1
        units_jls(k) = unit_string(jls_power(k),tend_units)

#ifdef TRACERS_PASSIVE
      case ('nh5','nh15','nh50','aoa','aoanh','e90','CO50')
        k = k + 1
        jls_decay(n) = k   ! decay loss
        sname_jls(k) = 'Decay_of_'//trim(trname(n))
        lname_jls(k) = 'LOSS OF '//trim(trname(n))//' BY DECAY'
        jls_ltop(k) = LM
        jls_power(k) = 0
        units_jls(k) = unit_string(jls_power(k),'kg s-1')
#endif

#ifdef TRACERS_WATER
C**** generic ones for many water tracers
      case ('Water', 'H2O18', 'HDO', 'HTO', 'H2O17' )
       k = k + 1
        jls_isrc(1,n) = k
        sname_jls(k) = 'Evap_'//trim(trname(n))
        lname_jls(k) = 'EVAPORATION OF '//trim(trname(n))
        jls_ltop(k) = 1
        jls_power(k) = ntm_power(n)+4
        scale_jls(k) = SECONDS_PER_DAY/DTsrc
        units_jls(k) = unit_string(jls_power(k),'mm day-1')
        jwt_jls(k) = jls_not_mass_weighted
       k = k + 1
        jls_isrc(2,n) = k
        sname_jls(k) = 'Ocn_Evap_'//trim(trname(n))
        lname_jls(k) = 'OCEAN EVAP OF '//trim(trname(n))
        jls_ltop(k) = 1
        jls_power(k) = ntm_power(n)+4
        scale_jls(k) = SECONDS_PER_DAY/DTsrc
        units_jls(k) = unit_string(jls_power(k),'mm day-1')
        jwt_jls(k) = jls_not_mass_weighted
       k = k + 1
        jls_prec(1,n)=k
        sname_jls(k) = 'Precip_'//trim(trname(n))
        lname_jls(k) = 'PRECIPITATION OF '//trim(trname(n))
        jls_ltop(k) = 1
        jls_power(k) = ntm_power(n)+4
        scale_jls(k) = SECONDS_PER_DAY/DTsrc
        units_jls(k) = unit_string(jls_power(k),'mm day-1')
        jwt_jls(k) = jls_not_mass_weighted
       k = k + 1
        jls_prec(2,n)=k
        sname_jls(k) = 'Ocn_Precip_'//trim(trname(n))
        lname_jls(k) = 'OCEAN PRECIP OF '//trim(trname(n))
        jls_ltop(k) = 1
        jls_power(k) = ntm_power(n)+4
        scale_jls(k) = SECONDS_PER_DAY/DTsrc
        units_jls(k) = unit_string(jls_power(k),'mm day-1')
        jwt_jls(k) = jls_not_mass_weighted

C**** special one unique to HTO
      if (trname(n).eq."HTO") then
       k = k + 1
        jls_decay(n) = k   ! special array for all radioactive sinks
        sname_jls(k) = 'Decay_of_'//trim(trname(n))
        lname_jls(k) = 'LOSS OF '//TRIM(trname(n))//' BY DECAY'
        jls_ltop(k) = lm
        jls_power(k) = ntm_power(n)+8
        scale_jls(k) = 1./DTsrc
        units_jls(k) = unit_string(jls_power(k),tend_units)
      end if
#endif

      case ('GLT')
        k = k + 1
        jls_3Dsource(1,n) = k
        sname_jls(k) = 'L1_overwrite_source_'//trim(trname(n))
        lname_jls(k) = trim(trname(n))//' L1 overwrite source'
        jls_ltop(k) = 1
        jls_power(k) = -5
        units_jls(k) = unit_string(jls_power(k),tend_units)

      case ('HCl','HOCl','ClONO2','HBr','HOBr','BrONO2','CFC',
     &      'BrOx','ClOx','Alkenes','Paraffin','Isoprene','CO',
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
#if TRACERS_dCO
     *      'd13Calke','d13CPAR',
     *      'd17OPAN', 'd18OPAN', 'd13CPAN',
     *      'dMe17OOH', 'dMe18OOH', 'd13MeOOH',
     *      'dHCH17O', 'dHCH18O', 'dH13CHO',
#endif  /* TRACERS_dCO */
     *      'dC17O', 'dC18O', 'd13CO',
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
     &      'N2O5','HNO3','H2O2','CH3OOH','HCHO','HO2NO2','PAN',
     &      'AlkylNit','Ox','NOx','Terpenes','Acetone')
        k = k + 1
        jls_3Dsource(nChemistry,n) = k
        sname_jls(k) = 'chemistry_source_of_'//trim(trname(n))
        lname_jls(k) = 'CHANGE OF '//trim(trname(n))//' BY CHEMISTRY'
        jls_ltop(k) = LM
        select case(trname(n))
        case ('Ox')
          jls_power(k) = 1
        case default
          jls_power(k) = -1
        end select
        units_jls(k) = unit_string(jls_power(k),tend_units)
        select case(trname(n))
        case ('Alkenes','Paraffin','Isoprene','CO','N2O5','HNO3',
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
#if TRACERS_dCO
     *      'd13Calke','d13CPAR',
     *      'd17OPAN', 'd18OPAN', 'd13CPAN',
     *      'dMe17OOH', 'dMe18OOH', 'd13MeOOH',
     *      'dHCH17O', 'dHCH18O', 'dH13CHO',
#endif  /* TRACERS_dCO */
     *      'dC17O', 'dC18O', 'd13CO',
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
     &  'H2O2','CH3OOH','HCHO','HO2NO2','PAN','AlkylNit','Ox',
     &  'Terpenes','Acetone','NOx','BrOx','ClOx')
          k = k + 1
          jls_3Dsource(nOverwrite,n) = k
          sname_jls(k) = 'overwrite_source_of_'//trim(trname(n))
          lname_jls(k) =
     &    'CHANGE OF '//trim(trname(n))//' BY OVERWRITE'
          jls_ltop(k) = LM
          jls_power(k) = -1
          units_jls(k) = unit_string(jls_power(k),tend_units)
        case ('CFC')  ! L=1 overwrite only.
          k = k + 1
          jls_3Dsource(nOverwrite,n) = k
          sname_jls(k) = 'overwrite_source_of_'//trim(trname(n))
          lname_jls(k) =
     &    'CHANGE OF '//trname(n)//' BY OVERWRITE'
          jls_ltop(k) = 1 ! L=1 overwrite only
          jls_power(k) = -1
          units_jls(k) = unit_string(jls_power(k),tend_units)
        end select
        select case(trname(n))
        case('NOx')
          k = k + 1
          jls_3Dsource(nOther,n) = k
          sname_jls(k) = 'lightning_source_of_'//trim(trname(n))
          lname_jls(k) = 'CHANGE OF '//trim(trname(n))//' BY LIGHTNING'
          jls_ltop(k) = LM
          jls_power(k) = -2
          units_jls(k) = unit_string(jls_power(k),tend_units)
        end select

#ifdef TRACERS_AEROSOLS_SOA
      case ('isopp1g','isopp2g','apinp1g','apinp2g')
c put in chemical production
        k = k + 1
        jls_3Dsource(nChemistry,n) = k
        sname_jls(k) = 'chemistry_source_of_'//trim(trname(n))
        lname_jls(k) = 'CHANGE OF '//trim(trname(n))//' BY CHEMISTRY'
        jls_ltop(k) = LM
        jls_power(k) = -1
        units_jls(k) = unit_string(jls_power(k),tend_units)

      case ('isopp1a','isopp2a','apinp1a','apinp2a')
c put in chemical production
        k = k + 1
        jls_3Dsource(nChemistry,n) = k
        sname_jls(k) = 'chemistry_source_of_'//trim(trname(n))
        lname_jls(k) = 'CHANGE OF '//trim(trname(n))//' BY CHEMISTRY'
        jls_ltop(k) = LM
        jls_power(k) = -1
        units_jls(k) = unit_string(jls_power(k),tend_units)
c gravitational settling of SOA
        k = k + 1
        jls_grav(n) = k
        sname_jls(k) = 'grav_sett_of_'//trim(trname(n))
        lname_jls(k) = 'Gravitational Settling of '//trim(trname(n))
        jls_ltop(k) = LM
        jls_power(k) = -2
        units_jls(k) = unit_string(jls_power(k),tend_units)
#endif  /* TRACERS_AEROSOLS_SOA*/

      case ('DMS')
        k = k + 1
        jls_isrc(1,n) = k
        sname_jls(k) = 'Ocean_source_of_'//trim(trname(n))
        lname_jls(k) = 'DMS ocean source'
        jls_ltop(k) = 1
        jls_power(k) =0
        units_jls(k) = unit_string(jls_power(k),mass_flux_units)
        jwt_jls(k) = jls_not_mass_weighted
C
        k = k + 1
        jls_3Dsource(1,n) = k
        sname_jls(k) = 'Chemical_sink_of_'//trim(trname(n))
        lname_jls(k) = 'DMS chemical loss'
        jls_ltop(k) =LM
        jls_power(k) =0
        units_jls(k) = unit_string(jls_power(k),tend_units)

       case ('MSA')
c put in chemical production of MSA
        k = k + 1
        jls_3Dsource(1,n) = k
        sname_jls(k) = 'chemistry_source_of_'//trim(trname(n))
        lname_jls(k) = 'Chemical production of MSA'
        jls_ltop(k) = LM
        jls_power(k) = -1
        units_jls(k) = unit_string(jls_power(k),tend_units)
c gravitational settling of MSA
        k = k + 1
        jls_grav(n) = k
        sname_jls(k) = 'grav_sett_of_'//trim(trname(n))
        lname_jls(k) = 'Gravitational Settling of MSA'
        jls_ltop(k) = LM
        jls_power(k) = -3
        units_jls(k) = unit_string(jls_power(k),tend_units)

       case ('SO2')
c volcanic production of SO2
        k = k + 1
        jls_3Dsource(nVolcanic,n) = k
        sname_jls(k) = trim(trname(n))//'_volcanic_src'
        lname_jls(k) = trim(trname(n))//' volcanic source'
        jls_ltop(k) = LM
        jls_power(k) = 0
        units_jls(k) = unit_string(jls_power(k),tend_units)
c put in chemical production of SO2
        k = k + 1
        jls_3Dsource(nChemprod,n) = k
        sname_jls(k) = 'dms_source_of_'//trim(trname(n))
        lname_jls(k) = 'production of SO2 from DMS'
        jls_ltop(k) = LM
        jls_power(k) =  1
        units_jls(k) = unit_string(jls_power(k),tend_units)
c put in chemical sink of SO2
        k = k + 1
        jls_3Dsource(nChemloss,n) = k
        sname_jls(k) = 'chem_sink_of_'//trim(trname(n))
        lname_jls(k) = 'chemical sink of SO2'
        jls_ltop(k) = LM
        jls_power(k) =  1
        units_jls(k) = unit_string(jls_power(k),tend_units)
        case ('SO4')
c gas phase source of SO4
        k = k + 1
        jls_3Dsource(nChemistry,n) = k
        sname_jls(k) = 'gas_phase_source_of_'//trim(trname(n))
        lname_jls(k) = trim(trname(n))//' gas phase source'
        jls_ltop(k) = LM
        jls_power(k) = 1
        units_jls(k) = unit_string(jls_power(k),tend_units)
c volcanic source of SO4
        k = k + 1
        jls_3Dsource(nVolcanic,n) = k
        sname_jls(k) = trim(trname(n))//'_volcanic_src'
        lname_jls(k) = trim(trname(n))//' volcanic source'
        jls_ltop(k) = LM
        jls_power(k) = 1
        units_jls(k) = unit_string(jls_power(k),tend_units)
c gravitational settling of SO4
        k = k + 1
        jls_grav(n) = k
        sname_jls(k) = 'grav_sett_of_'//trim(trname(n))
        lname_jls(k) = 'Gravitational Settling of '//trim(trname(n))
        jls_ltop(k) = LM
        jls_power(k) = -3
        units_jls(k) = unit_string(jls_power(k),tend_units)

        case ('SO4_d1', 'SO4_d2', 'SO4_d3')
c gas phase source
        k = k + 1
        jls_3Dsource(1,n) = k
        sname_jls(k) = 'gas_phase_source_of_'//trim(trname(n))
        lname_jls(k) = trim(trname(n))//' gas phase source'
        jls_ltop(k) = LM
        jls_power(k) = 1
        units_jls(k) = unit_string(jls_power(k),tend_units)
c gravitational settling
        k = k + 1
        jls_grav(n) = k
        sname_jls(k) = 'grav_sett_of_'//trim(trname(n))
        lname_jls(k) = 'Gravitational Settling of '//trim(trname(n))
        jls_ltop(k) = LM
        jls_power(k) = -3
        units_jls(k) = unit_string(jls_power(k),tend_units)

        case ('Be7')
c cosmogenic source from file
        k = k + 1
        jls_3Dsource(1,n) = k
        sname_jls(k) = 'Cosmogenic_src_of_'//trim(trname(n))
        lname_jls(k) = 'Be7 cosmogenic src'
        jls_ltop(k) = lm
        jls_power(k) = -28
        units_jls(k) = unit_string(jls_power(k),tend_units)
c radioactive decay
        k = k + 1
        jls_decay(n) = k   ! special array for all radioactive sinks
        sname_jls(k) = 'Decay_of_'//trim(trname(n))
        lname_jls(k) = 'Loss of Be7 by decay'
        jls_ltop(k) = lm
        jls_power(k) = -28
        units_jls(k) = unit_string(jls_power(k),tend_units)
c gravitational settling
        k = k + 1
        jls_grav(n) = k   ! special array grav. settling sinks
        sname_jls(k) = 'Grav_Settle_of_'//trim(trname(n))
        lname_jls(k) = 'Loss of Be7 by grav settling'
        jls_ltop(k) = lm
        jls_power(k) = -28
        units_jls(k) = unit_string(jls_power(k),tend_units)

        case ('Be10')
c cosmogenic source from file/same as Be7
        k = k + 1
        jls_3Dsource(1,n) = k
        sname_jls(k) = 'Cosmogenic_src_of_'//trim(trname(n))
        lname_jls(k) = 'Be10 cosmogenic src'
        jls_ltop(k) = lm
        jls_power(k) = -28  !may need changing around
        units_jls(k) = unit_string(jls_power(k),tend_units)
c gravitational settling
        k = k + 1
        jls_grav(n) = k   ! special array grav. settling sinks
        sname_jls(k) = 'Grav_Settle_of_'//trim(trname(n))
        lname_jls(k) = 'Loss of Be10 by grav settling'
        jls_ltop(k) = lm
        jls_power(k) = -28  !may need changing around
        units_jls(k) = unit_string(jls_power(k),tend_units)

        case ('Pb210')
c source of Pb210 from Rn222 decay
        k = k + 1
        jls_3Dsource(nChemistry,n) = k
        sname_jls(k) = 'Radioactive_src_of_'//trim(trname(n))
        lname_jls(k) = 'Pb210 radioactive src'
        jls_ltop(k) = lm
        jls_power(k) =-26   ! -10  !may need to be changed
        units_jls(k) = unit_string(jls_power(k),tend_units)
c radioactive decay
        k = k + 1
        jls_decay(n) = k   ! special array for all radioactive sinks
        sname_jls(k) = 'Decay_of_'//trim(trname(n))
        lname_jls(k) = 'Loss of Pb210 by decay'
        jls_ltop(k) = lm
        jls_power(k) =-26   ! -10  !may need to be changed
        units_jls(k) = unit_string(jls_power(k),tend_units)
c gravitational settling
        k = k + 1
        jls_grav(n) = k   ! special array grav. settling sinks
        sname_jls(k) = 'Grav_Settle_of_'//trim(trname(n))
        lname_jls(k) = 'Loss of Pb210 by grav settling'
        jls_ltop(k) = lm
        jls_power(k) = -28
        units_jls(k) = unit_string(jls_power(k),tend_units)

        case ('H2O2_s')
c gas phase source and sink of H2O2
        k = k + 1
        jls_3Dsource(1,n) = k
        sname_jls(k) = 'gas_phase_source_of_'//trim(trname(n))
        lname_jls(k) = trim(trname(n))//' gas phase source'
        jls_ltop(k) = LM
        jls_power(k) = 2
        units_jls(k) = unit_string(jls_power(k),tend_units)
        k = k + 1
        jls_3Dsource(2,n) = k
        sname_jls(k) = 'gas_phase_sink_of_'//trim(trname(n))
        lname_jls(k) = trim(trname(n))//' gas phase sink'
        jls_ltop(k) = LM
        jls_power(k) = 2
        units_jls(k) = unit_string(jls_power(k),tend_units)
c photolysis rate
        k = k + 1
        jls_phot = k
        sname_jls(k) = 'photolysis_rate_of_'//trim(trname(n))
        lname_jls(k) = 'photolysis rate of '//trim(trname(n))
        jls_ltop(k) =LM
        jls_power(k) =-9
        units_jls(k) = unit_string(jls_power(k),'/kg/s')
      case ('vbsGm2', 'vbsGm1', 'vbsGz',  'vbsGp1', 'vbsGp2',
     &      'vbsGp3', 'vbsGp4', 'vbsGp5', 'vbsGp6')
        k = k + 1
        jls_3Dsource(nChemistry,n) = k
        sname_jls(k) = trim(trname(n))//'_aging_source'
        lname_jls(k) = trim(trname(n))//' aging source'
        jls_ltop(k) = LM
        jls_power(k) = -1
        units_jls(k) = unit_string(jls_power(k),tend_units)
        k = k + 1
        jls_3Dsource(nChemloss,n) = k
        sname_jls(k) = trim(trname(n))//'_aging_loss'
        lname_jls(k) = trim(trname(n))//' aging loss'
        jls_ltop(k) = LM
        jls_power(k) = -1
        units_jls(k) = unit_string(jls_power(k),tend_units)
        k = k + 1
        jls_3Dsource(nOther,n) = k
        sname_jls(k) = trim(trname(n))//'_partitioning'
        lname_jls(k) = trim(trname(n))//' partitioning'
        jls_ltop(k) = LM
        jls_power(k) = -1
        units_jls(k) = unit_string(jls_power(k),tend_units)

      case ('BCII', 'BCIA', 'BCB', 'OCII', 'OCIA', 'OCB',
     &      'vbsAm2', 'vbsAm1', 'vbsAz',  'vbsAp1', 'vbsAp2',
     &      'vbsAp3', 'vbsAp4', 'vbsAp5', 'vbsAp6')
        select case(trname(n))
        case ('BCII', 'BCB', 'OCII', 'OCB',
     &        'vbsAm2', 'vbsAm1', 'vbsAz',  'vbsAp1', 'vbsAp2',
     &        'vbsAp3', 'vbsAp4', 'vbsAp5', 'vbsAp6')
          select case(trname(n))
          case ('vbsAm2', 'vbsAm1', 'vbsAz',  'vbsAp1', 'vbsAp2',
     &          'vbsAp3', 'vbsAp4', 'vbsAp5', 'vbsAp6')
            k = k + 1
            jls_3Dsource(nChemloss,n) = k
            sname_jls(k) = trim(trname(n))//'_partitioning'
            lname_jls(k) = trim(trname(n))//' partitioning'
            jls_ltop(k) = LM
            jls_power(k) = -1
            units_jls(k) = unit_string(jls_power(k),tend_units)
          end select
        case ('BCIA', 'OCIA')
          k = k + 1
          jls_3Dsource(nChemistry,n) = k
          sname_jls(k) = 'Aging_source_of_'//trim(trname(n))
          lname_jls(k) = trim(trname(n))//' aging source'
          jls_ltop(k) = LM
          jls_power(k) = -1
          units_jls(k) = unit_string(jls_power(k),tend_units)
        end select
        k = k + 1
        jls_grav(n) = k
        sname_jls(k) = 'grav_sett_of_'//trim(trname(n))
        lname_jls(k) = 'Gravitational Settling of '//trim(trname(n))
        jls_ltop(k) = LM
        jls_power(k) = -2
        units_jls(k) = unit_string(jls_power(k),tend_units)

#ifdef TRACERS_TOMAS
       case('ASO4__01','ASO4__02','ASO4__03','ASO4__04','ASO4__05',
     *    'ASO4__06','ASO4__07','ASO4__08','ASO4__09','ASO4__10',
     *    'ASO4__11','ASO4__12','ASO4__13','ASO4__14','ASO4__15',
     *    'ANACL_01','ANACL_02','ANACL_03','ANACL_04','ANACL_05',
     *    'ANACL_06','ANACL_07','ANACL_08','ANACL_09','ANACL_10',
     *    'ANACL_11','ANACL_12','ANACL_13','ANACL_14','ANACL_15',
     *    'AECIL_01','AECIL_02','AECIL_03','AECIL_04','AECIL_05',
     *    'AECIL_06','AECIL_07','AECIL_08','AECIL_09','AECIL_10',
     *    'AECIL_11','AECIL_12','AECIL_13','AECIL_14','AECIL_15',
     *    'AECOB_01','AECOB_02','AECOB_03','AECOB_04','AECOB_05',
     *    'AECOB_06','AECOB_07','AECOB_08','AECOB_09','AECOB_10',
     *    'AECOB_11','AECOB_12','AECOB_13','AECOB_14','AECOB_15',
     *    'AOCIL_01','AOCIL_02','AOCIL_03','AOCIL_04','AOCIL_05',
     *    'AOCIL_06','AOCIL_07','AOCIL_08','AOCIL_09','AOCIL_10',
     *    'AOCIL_11','AOCIL_12','AOCIL_13','AOCIL_14','AOCIL_15',
     *    'AOCOB_01','AOCOB_02','AOCOB_03','AOCOB_04','AOCOB_05',
     *    'AOCOB_06','AOCOB_07','AOCOB_08','AOCOB_09','AOCOB_10',
     *    'AOCOB_11','AOCOB_12','AOCOB_13','AOCOB_14','AOCOB_15',
     *    'ADUST_01','ADUST_02','ADUST_03','ADUST_04','ADUST_05',
     *    'ADUST_06','ADUST_07','ADUST_08','ADUST_09','ADUST_10',
     *    'ADUST_11','ADUST_12','ADUST_13','ADUST_14','ADUST_15')
        k = k + 1
        jls_3Dsource(nMicrophys,n) = k
        sname_jls(k) = 'Microphysics_src_of_'//trim(trname(n))
        lname_jls(k) = trim(trname(n))//'Microphysics src'
        jls_ltop(k) = LM
        jls_power(k) = 0
        units_jls(k) = unit_string(jls_power(k),tend_units)
        k = k + 1
        jls_grav(n) = k
        sname_jls(k) = 'grav_sett_of_'//trim(trname(n))
        lname_jls(k) = 'Gravitational Settling of '//trim(trname(n))
        jls_ltop(k) = LM
        jls_power(k) = -2
        units_jls(k) = unit_string(jls_power(k),tend_units)

        select case (trname(n))

        case ('ASO4__01','ASO4__02','ASO4__03','ASO4__04','ASO4__05',
     *       'ASO4__06','ASO4__07','ASO4__08','ASO4__09','ASO4__10',
     *       'ASO4__11','ASO4__12','ASO4__13','ASO4__14','ASO4__15')

c volcanic source of SO4
        k = k + 1
        jls_3Dsource(nVolcanic,n) = k
        sname_jls(k) = trim(trname(n))//'_volcanic_src'
        lname_jls(k) = trim(trname(n))//' volcanic source'
        jls_ltop(k) = LM
        jls_power(k) = 0
        units_jls(k) = unit_string(jls_power(k),tend_units)
c industrial source
        case ('ANUM__01','ANUM__02','ANUM__03','ANUM__04','ANUM__05',
     *    'ANUM__06','ANUM__07','ANUM__08','ANUM__09','ANUM__10',
     *    'ANUM__11','ANUM__12','ANUM__13','ANUM__14','ANUM__15')
c SO4
        k = k + 1
        jls_3Dsource(nSO4anum,n) = k
        sname_jls(k) = 'SO4_source_of_'//trim(trname(n))
        lname_jls(k) = trim(trname(n))//' SO4 source'
        jls_ltop(k) = LM
        jls_power(k) = 10
        units_jls(k) = unit_string(jls_power(k),'#/kg/s')
        k = k + 1
        jls_3Dsource(nECanum,n) = k
        sname_jls(k) = 'EC_source_of_'//trim(trname(n))
        lname_jls(k) = trim(trname(n))//'EC source'
        jls_ltop(k) = LM
        jls_power(k) = 10
        units_jls(k) = unit_string(jls_power(k),'#/kg/s')
        k = k + 1
        jls_3Dsource(nOCanum,n) = k
        sname_jls(k) = 'OC_source_of_'//trim(trname(n))
        lname_jls(k) = trim(trname(n))//'OC source'
        jls_ltop(k) = LM
        jls_power(k) = 10
        units_jls(k) = unit_string(jls_power(k),'#/kg/s')
        k = k + 1
        jls_3Dsource(nMicrophys,n) = k
        sname_jls(k) = 'Microphysics_src_of_'//trim(trname(n))
        lname_jls(k) = trim(trname(n))//'Microphysics src'
        jls_ltop(k) = LM
        jls_power(k) = 10
        units_jls(k) = unit_string(jls_power(k),'#/kg/s')
c industrial source
        do kk=1,ntsurfsrc(n_ANUM(1))
          k = k + 1
          jls_source(kk,n) = k
          sname_jls(k) = trim(trname(n))//'_'//
     &                   trim(sources(kk)%sourceName)
          lname_jls(k) = trim(trname(n))//' '//
     &                   trim(sources(kk)%sourceLname)
          jls_ltop(k) = 1
          jls_power(k) =10
          units_jls(k) = unit_string(jls_power(k),'#/m2/s')
          jwt_jls(k) = jls_not_mass_weighted
        enddo
        k = k + 1
        jls_isrc(1,n) = k
        sname_jls(k) = 'NACL_source_of_'//trim(trname(n))
        lname_jls(k) = trim(trname(n))//'ANACL source'
        jls_ltop(k) = 1
        jls_power(k) =10
        units_jls(k) = unit_string(jls_power(k),'#/m2/s')
        jwt_jls(k) = jls_not_mass_weighted
        k = k + 1
        jls_isrc(2,n) = k
        sname_jls(k) = 'Dust_source_of_'//trim(trname(n))
        lname_jls(k) =  trim(trname(n))//'ADUST source'
        jls_ltop(k) = 1
        jls_power(k) =1
        units_jls(k) = unit_string(jls_power(k),'#/m2/s')
        jwt_jls(k) = jls_not_mass_weighted
        k = k + 1
        jls_grav(n) = k
        sname_jls(k) = 'grav_sett_of_'//trim(trname(n))
        lname_jls(k) = 'Gravitational Settling of '//trim(trname(n))
        jls_ltop(k) = LM
        jls_power(k) = 10
        units_jls(k) = unit_string(jls_power(k),'#/m2/s')

      case ('ANACL_01','ANACL_02','ANACL_03','ANACL_04','ANACL_05',
     *    'ANACL_06','ANACL_07','ANACL_08','ANACL_09','ANACL_10',
     *    'ANACL_11','ANACL_12','ANACL_13','ANACL_14','ANACL_15')
        k = k + 1
        jls_isrc(1,n) = k
        sname_jls(k) = 'Ocean_source_of_'//trim(trname(n))
        lname_jls(k) = 'Ocean source of '//trim(trname(n))
        jls_ltop(k) = 1
        jls_power(k) =0
        units_jls(k) = unit_string(jls_power(k),mass_flux_units)
        jwt_jls(k) = jls_not_mass_weighted

      case ('AECOB_01','AECOB_02','AECOB_03','AECOB_04','AECOB_05',
     *    'AECOB_06','AECOB_07','AECOB_08','AECOB_09','AECOB_10',
     *    'AECOB_11','AECOB_12','AECOB_13','AECOB_14','AECOB_15',
     *    'AECIL_01','AECIL_02','AECIL_03','AECIL_04','AECIL_05',
     *    'AECIL_06','AECIL_07','AECIL_08','AECIL_09','AECIL_10',
     *    'AECIL_11','AECIL_12','AECIL_13','AECIL_14','AECIL_15')
        k = k + 1
        jls_3Dsource(1,n) = k
        sname_jls(k) = 'Aging_loss_of_'//trim(trname(n))
        lname_jls(k) = trim(trname(n))//' aging loss'
        jls_ltop(k) = LM
        jls_power(k) = 1
        units_jls(k) = unit_string(jls_power(k),tend_units)

      case ('AOCOB_01','AOCOB_02','AOCOB_03','AOCOB_04','AOCOB_05',
     *    'AOCOB_06','AOCOB_07','AOCOB_08','AOCOB_09','AOCOB_10',
     *    'AOCOB_11','AOCOB_12','AOCOB_13','AOCOB_14','AOCOB_15',
     *    'AOCIL_01','AOCIL_02','AOCIL_03','AOCIL_04','AOCIL_05',
     *    'AOCIL_06','AOCIL_07','AOCIL_08','AOCIL_09','AOCIL_10',
     *    'AOCIL_11','AOCIL_12','AOCIL_13','AOCIL_14','AOCIL_15')
        k = k + 1
        jls_3Dsource(1,n) = k
        sname_jls(k) = 'Aging_loss_of_'//trim(trname(n))
        lname_jls(k) = trim(trname(n))//' aging loss'
        jls_ltop(k) = LM
        jls_power(k) = 1
        units_jls(k) = unit_string(jls_power(k),tend_units)

! TOMAS  : should I exclude aerosol water??

        case('ADUST_01','ADUST_02','ADUST_03','ADUST_04','ADUST_05',
     *    'ADUST_06','ADUST_07','ADUST_08','ADUST_09','ADUST_10',
     *    'ADUST_11','ADUST_12','ADUST_13','ADUST_14','ADUST_15')

        k = k + 1
        jls_isrc(1,n) = k
        sname_jls(k) = 'Dust_source_of_'//trim(trname(n))
        lname_jls(k) = trim(trname(n))//' dust source'
        jls_ltop(k) = 1
        jls_power(k) =0
        units_jls(k) = unit_string(jls_power(k),mass_flux_units)
        jwt_jls(k) = jls_not_mass_weighted
        end select

#endif /* TRACERS_TOMAS*/

      case ('seasalt1', 'seasalt2', 'OCocean')
c ocean source
        k = k + 1
        jls_isrc(1,n) = k
        sname_jls(k) = trim(trname(n))//'_ocean_src'
        lname_jls(k) = trim(trname(n))//' ocean source'
        jls_ltop(k) = 1
        jls_power(k) = 1
        units_jls(k) = unit_string(jls_power(k),mass_flux_units)
        jwt_jls(k) = jls_not_mass_weighted
c gravitational settling
        k = k + 1
        jls_grav(n) = k
        sname_jls(k) = trim(trname(n))//'_grav_sett'
        lname_jls(k) = trim(trname(n))//' gravitational settling'
        jls_ltop(k) = LM
        select case (trname(n))
        case ('seasalt1', 'OCocean')
          jls_power(k) = -2
        case ('seasalt2')
          jls_power(k) =0
        end select
        units_jls(k) = unit_string(jls_power(k),tend_units)

#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS)
        CASE('Clay','Silt1','Silt2','Silt3','Silt4','Silt5','ClayIlli'
     &         ,'ClayKaol','ClaySmec','ClayCalc','ClayQuar','ClayFeld'
     &         ,'ClayHema','ClayGyps','ClayIlHe','ClayKaHe','ClaySmHe'
     &         ,'ClayCaHe','ClayQuHe','ClayFeHe','ClayGyHe','Sil1Quar'
     &         ,'Sil1Feld','Sil1Calc','Sil1Hema','Sil1Gyps','Sil1Illi'
     &         ,'Sil1Kaol','Sil1Smec','Sil1QuHe','Sil1FeHe','Sil1CaHe'
     &         ,'Sil1GyHe','Sil1IlHe','Sil1KaHe','Sil1SmHe','Sil2Quar'
     &         ,'Sil2Feld','Sil2Calc','Sil2Hema','Sil2Gyps','Sil2Illi'
     &         ,'Sil2Kaol','Sil2Smec','Sil2QuHe','Sil2FeHe','Sil2CaHe'
     &         ,'Sil2GyHe','Sil2IlHe','Sil2KaHe','Sil2SmHe','Sil3Quar'
     &         ,'Sil3Feld','Sil3Calc','Sil3Hema','Sil3Gyps','Sil3Illi'
     &         ,'Sil3Kaol','Sil3Smec','Sil3QuHe','Sil3FeHe','Sil3CaHe'
     &         ,'Sil3GyHe','Sil3IlHe','Sil3KaHe','Sil3SmHe','Sil4Quar'
     &         ,'Sil4Feld','Sil4Calc','Sil4Hema','Sil4Gyps','Sil4Illi'
     &         ,'Sil4Kaol','Sil4Smec','Sil4QuHe','Sil4FeHe','Sil4CaHe'
     &         ,'Sil4GyHe','Sil4IlHe','Sil4KaHe','Sil4SmHe','Sil5Quar'
     &         ,'Sil5Feld','Sil5Calc','Sil5Hema','Sil5Gyps','Sil5Illi'
     &         ,'Sil5Kaol','Sil5Smec','Sil5QuHe','Sil5FeHe','Sil5CaHe'
     &         ,'Sil5GyHe','Sil5IlHe','Sil5KaHe','Sil5SmHe')

        k=k+1
          jls_isrc(nDustEmjl,n)=k
          lname_jls(k)='Emission of '//TRIM(trname(n))
          sname_jls(k)=TRIM(trname(n))//'_emission'
          jls_ltop(k)=1
          jls_power(k)=1
          units_jls(k)=unit_string(jls_power(k),mass_flux_units)
          jwt_jls(k) = jls_not_mass_weighted
        IF ( imDust == 0 .or. imDust >= 3 ) THEN
          k=k+1
          jls_isrc(nDustEm2jl,n)=k
          lname_jls(k)='Cubic emission of '//TRIM(trname(n))
          sname_jls(k)=TRIM(trname(n))//'_emission2'
          jls_ltop(k)=1
          jls_power(k)=1
          units_jls(k)=unit_string(jls_power(k),mass_flux_units)
          jwt_jls(k) = jls_not_mass_weighted
        END IF
#ifndef TRACERS_DRYDEP
        k=k+1
          jls_isrc(nDustTurbjl,n)=k
          lname_jls(k)='Turbulent deposition of '//TRIM(trname(n))
          sname_jls(k)=TRIM(trname(n))//'_turb_depo'
          jls_ltop(k)=1
          jls_power(k)=1
          units_jls(k)=unit_string(jls_power(k),mass_flux_units)
          jwt_jls(k) = jls_not_mass_weighted
#endif
        k=k+1
          jls_grav(n)=k
          lname_jls(k)='Gain by gravitational settling of '
     &         //TRIM(trname(n))
          sname_jls(k)=TRIM(trname(n))//'_grav_sett'
          jls_ltop(k)=Lm
          jls_power(k)=1
          units_jls(k)=unit_string(jls_power(k),tend_units)
#ifndef TRACERS_WATER
        k=k+1
          jls_wet(n)=k
          lname_jls(k)='Loss by wet deposition of '//TRIM(trname(n))
          sname_jls(k)=TRIM(trname(n))//'_wet_depo'
          jls_ltop(k)=Lm
          jls_power(k)=1
          units_jls(k)=unit_string(jls_power(k),'kg s-1')
#endif
#endif /* TRACERS_DUST || TRACERS_MINERALS */

C**** Here are some more examples of generalised diag. configuration
c      n = n_dust
c        k = k + 1
c        jls_grav(n) = k   ! special array grav. settling sinks
c        sname_jls(k) = 'Grav_Settle_of_'//trname(n)
c        lname_jls(k) = 'LOSS OF DUST BY SETTLING'
c        jls_ltop(k) = lm
c        jls_power(k) = -11
c        units_jls(k) = unit_string(jls_power(k),tend_units)
      end select

#if (defined TRACERS_WATER) && (defined TRDIAG_WETDEPO)
c**** additional wet deposition diagnostics
      IF (diag_wetdep == 1) THEN
        k=k+1
        jls_trdpmc(1,n)=k
        lname_jls(k)='MC Condensation of '//TRIM(trname(n))
        sname_jls(k)=TRIM(trname(n))//'_cond_mc'
        jls_ltop(k)=Lm
        jls_power(k)=1
        units_jls(k)=unit_string(jls_power(k),tend_units)
        k=k+1
        jls_trdpmc(2,n)=k
        lname_jls(k)='Evaporated '//TRIM(trname(n))//' in MC Downdrafts'
        sname_jls(k)=TRIM(trname(n))//'_downeva_mc'
        jls_ltop(k)=Lm
        jls_power(k)=1
        units_jls(k)=unit_string(jls_power(k),tend_units)
        k=k+1
        jls_trdpmc(3,n)=k
        lname_jls(k)='Condensed '//TRIM(trname(n))//' in MC CLW'
        sname_jls(k)=TRIM(trname(n))//'_conclw_mc'
        jls_ltop(k)=Lm
        jls_power(k)=1
        units_jls(k)=unit_string(jls_power(k),tend_units)
        k=k+1
        jls_trdpmc(4,n)=k
        lname_jls(k)='Precipitated '//TRIM(trname(n))//' by MC'
        sname_jls(k)=TRIM(trname(n))//'_precip_mc'
        jls_ltop(k)=Lm
        jls_power(k)=1
        units_jls(k)=unit_string(jls_power(k),tend_units)
        k=k+1
        jls_trdpmc(5,n)=k
        lname_jls(k)='Reevaporated '//TRIM(trname(n))//' from MC Precip'
        sname_jls(k)=TRIM(trname(n))//'_reevap_mc'
        jls_ltop(k)=Lm
        jls_power(k)=1
        units_jls(k)=unit_string(jls_power(k),tend_units)
        k=k+1
        jls_trdpmc(6,n)=k
        lname_jls(k)='MC Washout of '//TRIM(trname(n))
        sname_jls(k)=TRIM(trname(n))//'_washout_mc'
        jls_ltop(k)=Lm
        jls_power(k)=1
        units_jls(k)=unit_string(jls_power(k),tend_units)
        k=k+1
        jls_trdpls(1,n)=k
        lname_jls(k)='LS Washout of '//TRIM(trname(n))
        sname_jls(k)=TRIM(trname(n))//'_washout_ls'
        jls_ltop(k)=Lm
        jls_power(k)=1
        units_jls(k)=unit_string(jls_power(k),tend_units)
        k=k+1
        jls_trdpls(2,n)=k
        lname_jls(k)='Precipitated '//TRIM(trname(n))//' by LS'
        sname_jls(k)=TRIM(trname(n))//'_precip_ls'
        jls_ltop(k)=Lm
        jls_power(k)=1
        units_jls(k)=unit_string(jls_power(k),tend_units)
        k=k+1
        jls_trdpls(3,n)=k
        lname_jls(k)='Condensed '//TRIM(trname(n))// ' in LS CLW'
        sname_jls(k)=TRIM(trname(n))//'_conclw_ls'
        jls_ltop(k)=Lm
        jls_power(k)=1
        units_jls(k)=unit_string(jls_power(k),tend_units)
        k=k+1
        jls_trdpls(4,n)=k
        lname_jls(k)='Reevaporated '//TRIM(trname(n))//' from LS Precip'
        sname_jls(k)=TRIM(trname(n))//'_reevap_ls'
        jls_ltop(k)=Lm
        jls_power(k)=1
        units_jls(k)=unit_string(jls_power(k),tend_units)
        k=k+1
        jls_trdpls(5,n)=k
        lname_jls(k)='Evaporated '//TRIM(trname(n))//' from LS CLW'
        sname_jls(k)=TRIM(trname(n))//'_clwevap_ls'
        jls_ltop(k)=Lm
        jls_power(k)=1
        units_jls(k)=unit_string(jls_power(k),tend_units)
        k=k+1
        jls_trdpls(6,n)=k
        lname_jls(k)='LS Condensation of '//TRIM(trname(n))
        sname_jls(k)=TRIM(trname(n))//'_cond_ls'
        jls_ltop(k)=Lm
        jls_power(k)=1
        units_jls(k)=unit_string(jls_power(k),tend_units)
      END IF
#endif

      end do

C**** Additional Special JL diagnostics
C**** (not necessary associated with a particular tracer)
#ifdef TRACERS_SPECIAL_Shindell
        k = k + 1
        jls_ClOcon=k
        sname_jls(k) = 'ClO_conc'
        lname_jls(k) = 'ClO concentration'
        jls_ltop(k)  = LTOP
        jls_power(k) = -11
        scale_jls(k) = 1.
        units_jls(k) = unit_string(jls_power(k),'V/V air')
        k = k + 1
        jls_H2Ocon=k
        sname_jls(k) = 'H2O_conc'
        lname_jls(k) = 'H2O concentration'
        jls_ltop(k)  = LTOP
        jls_power(k) = -7
        scale_jls(k) = 1.
        units_jls(k) = unit_string(jls_power(k),'V/V air')
        k = k + 1
        jls_H2Ochem=k
        sname_jls(k) = 'H2O_chem'
        lname_jls(k) = 'H2O change due to chemistry'
        jls_ltop(k)  = LTOP
        jls_power(k) = -4
        scale_jls(k) = 1./DTsrc
        units_jls(k) = unit_string(jls_power(k),tend_units)
        k = k + 1
        jls_Oxp=k
        sname_jls(k) = 'Ox_chem_prod'
        lname_jls(k) = 'Ox production due to chemistry'
        jls_ltop(k)  = LM
        jls_power(k) = 2
        scale_jls(k) = 1.d0/DTsrc
        units_jls(k) = unit_string(jls_power(k),tend_units)
        k = k + 1
        jls_Oxd=k
        sname_jls(k) = 'Ox_chem_dest'
        lname_jls(k) = 'Ox destruction due to chemistry'
        jls_ltop(k)  = LM
        jls_power(k) = 2
        scale_jls(k) = 1.d0/DTsrc
        units_jls(k) = unit_string(jls_power(k),tend_units)
        k = k + 1
        jls_OxpT=k
        sname_jls(k) = 'trop_Ox_chem_prod'
        lname_jls(k) = 'Troposphere Ox prod by chemistry'
        jls_ltop(k)  = LM
        jls_power(k) = 2
        scale_jls(k) = 1.d0/DTsrc
        units_jls(k) = unit_string(jls_power(k),tend_units)
        k = k + 1
        jls_OxdT=k
        sname_jls(k) = 'trop_Ox_chem_dest'
        lname_jls(k) = 'Troposphere Ox dest by chemistry'
        jls_ltop(k)  = LM
        jls_power(k) = 2
        scale_jls(k) = 1.d0/DTsrc
        units_jls(k) = unit_string(jls_power(k),tend_units)
        k = k + 1
        jls_COp=k
        sname_jls(k) = 'CO_chem_prod'
        lname_jls(k) = 'CO production due to chemistry'
        jls_ltop(k)  = LM
        jls_power(k) = 1
        scale_jls(k) = 1.d0/DTsrc
        units_jls(k) = unit_string(jls_power(k),tend_units)
        k = k + 1
        jls_COd=k
        sname_jls(k) = 'CO_chem_dest'
        lname_jls(k) = 'CO destruction due to chemistry'
        jls_ltop(k)  = LM
        jls_power(k) = 1
        scale_jls(k) = 1.d0/DTsrc
        units_jls(k) = unit_string(jls_power(k),tend_units)
#ifdef TRACERS_ACETONE
        k = k + 1
        jls_AcetP=k
        sname_jls(k) = 'Acetone_chem_prod'
        lname_jls(k) = 'Acetone production due to chemistry'
        jls_ltop(k)  = LM
        jls_power(k) = 1
        scale_jls(k) = 1.d0/DTsrc
        units_jls(k) = unit_string(jls_power(k),tend_units)
        k = k + 1
        jls_AcetD=k
        sname_jls(k) = 'Acetone_chem_dest'
        lname_jls(k) = 'Acetone destruction due to chemistry'
        jls_ltop(k)  = LM
        jls_power(k) = 1
        scale_jls(k) = 1.d0/DTsrc
        units_jls(k) = unit_string(jls_power(k),tend_units)
#endif
        k = k + 1
        jls_OHcon=k
        sname_jls(k) = 'OH_conc'
        lname_jls(k) = 'OH concentration'
        jls_ltop(k)  = LTOP
        jls_power(k) = 5
        scale_jls(k) = 1.
        units_jls(k) = unit_string(jls_power(k),'molecules cm-3')
c
        k = k + 1
        jls_H2Omr=k
        sname_jls(k) = 'H2O_mr'
        lname_jls(k) = 'H2O mixing ratio (weighted by daylight)'
        jls_ltop(k)  = LTOP
        jls_power(k) = -4
        scale_jls(k) = 1.
        units_jls(k) = unit_string(jls_power(k),'parts/vol')
c
        k = k + 1
        jls_day=k
        sname_jls(k) = 'daylight'   ! not output
        lname_jls(k) = 'Daylight weighting'
        jls_ltop(k)  = 1
        jls_power(k) = 0
        scale_jls(k) = 100.
        units_jls(k) = unit_string(jls_power(k),'%')
c
        k = k + 1
        jls_N2O5sulf=k
        sname_jls(k) = 'N2O5_sulf'
        lname_jls(k) = 'N2O5 sulfate sink'
        jls_ltop(k)  = LTOP
        jls_power(k) = -2
        units_jls(k) = unit_string(jls_power(k),tend_units)
c
        k = k + 1
        jls_O3vmr=k
        sname_jls(k) = 'O3_VMR'
        lname_jls(k) = 'O3 volume mixing ratio'
        jls_ltop(k)  = LTOP
        jls_power(k) = -8 ! simply to match Ox_CONCENTRATION diag
        scale_jls(k) = 1.
        units_jls(k) = unit_string(jls_power(k),'V/V air')

#endif  /* TRACERS_SPECIAL_Shindell */

#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS)
c Oxidants
        if (coupled_chem.le.0) then
#ifndef TRACERS_SPECIAL_Shindell
          k = k + 1
          jls_OHconk = k
          sname_jls(k) = 'OH_conc'
          lname_jls(k) = 'OH Concentration'
          jls_ltop(k) = LM
          jls_power(k) =5
          scale_jls(k) = 1.
          units_jls(k) = unit_string(jls_power(k),'molecules cm-3')
#endif

          k = k + 1
          jls_HO2con = k
          sname_jls(k) = 'HO2_conc'
          lname_jls(k) = 'HO2 Concentration'
          jls_ltop(k) =LM
          jls_power(k) =7
          scale_jls(k) =1.
          units_jls(k) = unit_string(jls_power(k),'molecules cm-3')

          k = k + 1
          jls_NO3 = k
          sname_jls(k) = 'NO3_conc'
          lname_jls(k) = 'NO3 Concentration'
          jls_ltop(k) =LM
          jls_power(k) =5
          scale_jls(k) =1.
          units_jls(k) = unit_string(jls_power(k),'molecules cm-3')
        endif
#endif  /* TRACERS_AEROSOLS_Koch || TRACERS_AMP || TRACERS_TOMAS */

#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS)
      k = k + 1
      jls_spec(nDustEv1jl)=k
      lname_jls(k)='No. dust events'
      sname_jls(k)='no_dust_ev1'
      jls_ltop(k)=1
      scale_jls(k)=SECONDS_PER_DAY/Dtsrc
      units_jls(k)='1/d'
      jwt_jls(k) = jls_not_mass_weighted
      k = k + 1
      jls_spec(nDustEv2jl)=k
      lname_jls(k)='No. dust events above threshold wind'
      sname_jls(k)='no_dust_ev2'
      jls_ltop(k)=1
      scale_jls(k)=SECONDS_PER_DAY/Dtsrc
      units_jls(k)='1/d'
      jwt_jls(k) = jls_not_mass_weighted
      k = k + 1
      jls_spec(nDustWthjl)=k
      lname_jls(k)='Threshold velocity for dust emission'
      sname_jls(k)='wtrsh'
      jls_ltop(k)=1
      scale_jls(k)=1.
      units_jls(k)='m s-1'
      jwt_jls(k) = jls_not_mass_weighted
#endif

      if (k.gt. ktajls) then
        if (AM_I_ROOT()) write (6,*)
     &   'tjl_defs: Increase ktajls=',ktajls,' to at least ',k
         print *,'k should be ',k
        call stop_model('ktajls too small',255)
      end if
#endif /* TRACERS_ON */

      return

      contains

      subroutine CO2n_init_jls(k,n,name)
      integer, intent(inout) :: k
      integer, intent(in) :: n
      character(len=*), intent(in) :: name
      k = k + 1
      jls_isrc(1,n) = k
      sname_jls(k) = 'Ocean_Gas_Exchange_'//trim(trname(n))
      lname_jls(k) = trim(trname(n))//' Ocean/Atmos. Gas Exchange'
      jls_ltop(k) = 1
      jls_power(k) = 3
      units_jls(k) = unit_string(jls_power(k),mass_flux_units)
      jwt_jls(k) = jls_not_mass_weighted
      end subroutine CO2n_init_jls

      subroutine Rn222_init_jls(k,n,name)
      integer, intent(inout) :: k
      integer, intent(in) :: n
      character(len=*), intent(in) :: name
      k = k + 1
      jls_decay(n) = k          ! special array for all radioactive sinks
      sname_jls(k) = 'Decay_of_'//trim(trname(n))
      lname_jls(k) = 'LOSS OF '//trim(trname(n))//' BY DECAY'
      jls_ltop(k) = lm
      jls_power(k) = -26
      units_jls(k) = unit_string(jls_power(k),tend_units)
      end subroutine Rn222_init_jls
      
      subroutine N2O_init_jls(k,n,name)
      integer, intent(inout) :: k
      integer, intent(in) :: n
      character(len=*), intent(in) :: name
#ifdef TRACERS_SPECIAL_Shindell
      k = k + 1
      jls_3Dsource(nChemistry,n) = k
      sname_jls(k) = 'chemistry_source_of_'//trim(trname(n))
      lname_jls(k) = 'CHANGE OF '//trim(trname(n))//' BY CHEMISTRY'
      jls_ltop(k) = LM
      jls_power(k) = -1
      units_jls(k) = unit_string(jls_power(k),tend_units)
      k = k + 1
      jls_3Dsource(nOverwrite,n) = k
      sname_jls(k) = 'overwrite_source_of_'//trim(trname(n))
      lname_jls(k) =
     &     'CHANGE OF '//trim(trname(n))//' BY OVERWRITE'
      jls_ltop(k) = 1           ! really L=1 overwrite only
      jls_power(k) = -1
      units_jls(k) = unit_string(jls_power(k),tend_units)
#endif
#ifdef TRACERS_SPECIAL_Lerner
      k = k + 1
      jls_3Dsource(nChemistry,n) = k
      sname_jls(k) = 'Stratos_chem_change_'//trim(trname(n))
      lname_jls(k) = 'CHANGE OF '//trim(trname(n))//
     &               ' BY CHEMISTRY IN STRATOS'
      jls_ltop(k) = lm
      jls_power(k) = -1
      units_jls(k) = unit_string(jls_power(k),tend_units)
#endif
      end subroutine N2O_init_jls

      end subroutine init_jls_diag

      subroutine init_ijts_diag
!@sum init_ijts_diag Initialise lat/lon tracer diags
!@auth Gavin Schmidt
      use Tracer_mod, only: Tracer
      use TracerSurfaceSource_mod, only: TracerSurfaceSource
      USE DOMAIN_DECOMP_ATM, only: AM_I_ROOT
      use TimeConstants_mod, only: SECONDS_PER_DAY
      USE MODEL_COM, only: dtsrc
      use TRACER_COM, only: ntm, n_SO2, nAircraft, nbiomass, nchemistry
      use TRACER_COM, only: nOther, nOverwrite, nVolcanic, nChemloss
      use TRACER_COM, only: nChemprod, nThermo, nMicrophys
      use TRACER_COM, only: ntsurfsrc, tracers, aqchem_list
      use OldTracer_mod, only: do_aircraft
      use TRACER_COM, only: nRocket
      use OldTracer_mod, only: do_rocket
#ifdef TRACERS_TOMAS
      use TRACER_COM, only: n_AOCOB, n_ANUM, n_AECOB
      use TRACER_COM, only: nSO4anum, nECanum, nOCanum
#endif
      USE DIAG_COM
#ifdef TRACERS_ON
      USE TRDIAG_COM
#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS)
      use trdust_mod, only: nDustEmij, nDustEm2ij, nDustEv1ij
     &   ,nDustEv2ij, nDustWthij, imDust, nSubClays
#endif
#if (defined TRACERS_WATER) && (defined TRDIAG_WETDEPO)
      USE CLOUDS, ONLY : diag_wetdep
#endif
      use RAD_COM, only: diag_fc
#endif /* TRACERS_ON */
#ifdef TRACERS_AMP
      use tracer_com, only: n_N_AKK_1
#endif
      use OldTracer_mod, only: has_chemistry
      use OldTracer_mod, only: has_overwrite
      use OldTracer_mod, only: trname, ntm_power, dodrydep,
     &          src_dist_index,nBBsources,do_fire
      use rad_com, only: nradfrc
      use RunTimeControls_mod, only: tracers_aerosols_seasalt,
     &     tracers_aerosols_koch, tracers_nitrate
      implicit none

      interface
        subroutine set_diag_aod(n,n_subclasses)
        integer, intent(in) :: n
        integer, optional, intent(in) :: n_subclasses
        end subroutine set_diag_aod
      end interface
      interface
        subroutine set_diag_rf(n,n_subclasses)
        integer, intent(in) :: n
        integer, optional, intent(in) :: n_subclasses
        end subroutine set_diag_rf
      end interface

      integer n,n1,kr,n_src
      character*50 :: unit_string
      CHARACTER*17 :: cform
      class (Tracer), pointer :: pTracer
      type (TracerSurfaceSource), pointer :: sources(:)

      interface
        integer function ijts_diag(sname,lname,units,ia,power,denom,
     *                             scalediv)
          character(len=*), intent(in) :: sname, lname, units
          integer, intent(in), optional :: ia
          integer, intent(in), optional :: power
          character(len=*), intent(in), optional :: denom
          real*8, intent(in), optional :: scalediv
        end function ijts_diag
      end interface

#ifdef TRACERS_ON
C**** Defaults for ijts (sources, sinks, etc.)
      ia_ijts=ia_src ! default
      sname_ijts=''

      ijts_fc(:,:)=0
      ijts_3Dsource(:,:)=0
      ijts_aq(:,:)=0
      ijts_isrc(:,:)=0
      ijts_gasex(:,:)=0
      denom_ijts(:) = 0
#ifdef TRACERS_AMP
      ijts_AMPe(:)=0
      ijts_AMPp(:,:)=0
      ijts_AMPpdf(:,:)=0
      ijts_tauint=0
      ijts_taustratint=0
      ijts_reffstratint=0
      ijts_2Dreff(:)=0
#endif
#ifdef TRACERS_TOMAS
      ijts_TOMAS(:,:)=0
      ijts_subcoag(:)=0
#endif
C**** This needs to be 'hand coded' depending on circumstances
      do n=1,NTM
        if (src_dist_index(n)/=0) cycle

!=============================================!
! emissions for all tracers, if they have any !
!=============================================!

! handle exceptions first (e.g. SO4 emissions are listed under SO2)
      select case (trname(n))
      case ('SO4',
     &      'M_AKK_SU','M_ACC_SU',
     &      'ASO4__01','ASO4__02','ASO4__03','ASO4__04','ASO4__05',
     &      'ASO4__06','ASO4__07','ASO4__08','ASO4__09','ASO4__10',
     &      'ASO4__11','ASO4__12','ASO4__13','ASO4__14','ASO4__15')
        n_src = n_SO2
#ifdef TRACERS_TOMAS
      case ('AECIL_01','AECIL_02','AECIL_03','AECIL_04','AECIL_05',
     &      'AECIL_06','AECIL_07','AECIL_08','AECIL_09','AECIL_10',
     &      'AECIL_11','AECIL_12','AECIL_13','AECIL_14','AECIL_15',
     &      'AECOB_01','AECOB_02','AECOB_03','AECOB_04','AECOB_05',
     &      'AECOB_06','AECOB_07','AECOB_08','AECOB_09','AECOB_10',
     &      'AECOB_11','AECOB_12','AECOB_13','AECOB_14','AECOB_15')
        n_src = n_AECOB(1)
      case ('AOCIL_01','AOCIL_02','AOCIL_03','AOCIL_04','AOCIL_05',
     &      'AOCIL_06','AOCIL_07','AOCIL_08','AOCIL_09','AOCIL_10',
     &      'AOCIL_11','AOCIL_12','AOCIL_13','AOCIL_14','AOCIL_15',
     &      'AOCOB_01','AOCOB_02','AOCOB_03','AOCOB_04','AOCOB_05',
     &      'AOCOB_06','AOCOB_07','AOCOB_08','AOCOB_09','AOCOB_10',
     &      'AOCOB_11','AOCOB_12','AOCOB_13','AOCOB_14','AOCOB_15')
        n_src = n_AOCOB(1)
#endif  /* TRACERS_TOMAS */
      case default
        n_src = n
      end select
      pTracer => tracers%getReference(trname(n_src))
      sources => pTracer%surfaceSources

! aqueous chemistry sources and sinks
      if (allocated(aqchem_list)) then
        if (any(n.eq.aqchem_list)) then
          do kr=1,5
      select case(kr)
      case(1)
        ijts_aq(kr,n)=ijts_diag(trim(trname(n))//'_aqchem',
     *                           trim(trname(n))//' aqchem',
     *                           'kg m-2 s-1', power=-15,
     *                           scalediv=dtsrc)
      case(2)
        ijts_aq(kr,n)=ijts_diag(trim(trname(n))//'_aqchem_MC_cloud',
     *                           trim(trname(n))//' aqchem MC cloud',
     *                           'kg m-2 s-1', power=-15,
     *                           scalediv=dtsrc)
      case(3)
        ijts_aq(kr,n)=ijts_diag(trim(trname(n))//'_aqchem_MC_precip',
     *                          trim(trname(n))//' aqchem MC precip',
     *                          'kg m-2 s-1', power=-15,
     *                          scalediv=dtsrc)
      case(4)
        ijts_aq(kr,n)=ijts_diag(trim(trname(n))//'_aqchem_LS_cloud',
     *                          trim(trname(n))//' aqchem LS cloud',
     *                          'kg m-2 s-1', power=-15,
     *                          scalediv=dtsrc)
      case(5)
        ijts_aq(kr,n)=ijts_diag(trim(trname(n))//'_aqchem_LS_precip',
     *                          trim(trname(n))//' aqchem LS precip',
     *                          'kg m-2 s-1', power=-15,
     *                          scalediv=dtsrc)
      case default
        continue ! nothing to do
      end select
          enddo
        endif
      endif

! surface emissions
      do kr=1,ntsurfsrc(n_src)
        select case (trname(n))
        case ('ANUM__01','ANUM__02','ANUM__03','ANUM__04','ANUM__05',
     &        'ANUM__06','ANUM__07','ANUM__08','ANUM__09','ANUM__10',
     &        'ANUM__11','ANUM__12','ANUM__13','ANUM__14','ANUM__15')
          ijts_source(kr,n)=
     *      ijts_diag(trim(trname(n))//'_'//
     *                  trim(sources(kr)%sourceName),
     *                  trim(trname(n))//' '//
     *                  trim(sources(kr)%sourceLname),
     *                  '# m-2 s-1', power=-15,
     *                  scalediv=dtsrc)
        case default
          ijts_source(kr,n)=
     *      ijts_diag(trim(trname(n))//'_'//
     *                  trim(sources(kr)%sourceName),
     *                  trim(trname(n))//' '//
     *                  trim(sources(kr)%sourceLname),
     *                  'kg m-2 s-1', power=-15,
     *                  scalediv=dtsrc)
        end select
      end do

! aircraft emissions
      if(do_aircraft(n_src))then
        ijts_3Dsource(nAircraft,n)=
     *    ijts_diag(trim(trname(n))//'_aircraft_src',
     *              trim(trname(n))//' aircraft source',
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)
      end if

! rocket emissions
      if(do_rocket(n_src))then
        ijts_3Dsource(nRocket,n)=
     *    ijts_diag(trim(trname(n))//'_rocket_src',
     *              trim(trname(n))//' rocket source',
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)
      end if

! biomass burning emissions
      if (nBBsources(n_src) .gt. 0 .or. do_fire(n_src)) then
        ijts_3Dsource(nBiomass,n)=
     *    ijts_diag(trim(trname(n))//'_biomass_src',
     *              trim(trname(n))//' biomass source',
     *              'kg m-2 s-1', power=-12,
     *              scalediv=dtsrc)
      endif

!============================================!
! Chemical source (+) or sink (-) of tracers !
!============================================!
      if (has_chemistry(n)) then
        ijts_3Dsource(nChemistry,n)=
     *    ijts_diag(trim(trname(n))//'_chem',
     *              trim(trname(n))//' chemistry',
     *              'kg m-2 s-1', power=-12,
     *              scalediv=dtsrc)
      endif

!======================!
! Overwrite of tracers !
!======================!
      if (has_overwrite(n)) then
        ijts_3Dsource(nOverwrite,n)=
     *    ijts_diag(trim(trname(n))//'_overw',
     *              trim(trname(n))//' overwrite',
     *              'kg m-2 s-1', power=-12,
     *              scalediv=dtsrc)
      endif

!=============================!
! Tracer-specific diagnostics !
!=============================!
      select case (trname(n))

        case ('CFCn','CO2n')
!NR: this should be for Lerner tracer CO2 not for CO2n
!         ijts_isrc(1,n)=
!    *      ijts_diag(trim(trname(n))//'_ocean_source',
!    *                trim(trname(n))//' ocean source',
!    *                'kg m-2 s-1', power=-12,
!    *                scalediv=dtsrc)
          ijts_isrc(1,n)=
     *      ijts_diag(trim(trname(n))//'_GASX',
     *                trim(trname(n))//' Land+Ocean',
     *                'kg m-2 s-1', power=-11,
     *                scalediv=dtsrc)
          ijts_gasex(1,n)= ! Gas Exchange Coefficient (piston velocity) (open ocean only)
     *      ijts_diag('O_Piston_Veloc_'//trim(trname(n)),
     *                trim(trname(n))//' Ocean Piston Velocity',
     *                'm s-1', ia=ia_srf, power=-5,
     *                denom='ocnfr')
          ijts_gasex(2,n)= ! Gas Exchange Solubility coefficient
     *      ijts_diag('O_Solubility_'//trim(trname(n)),
     *                trim(trname(n))//' Ocean Solubility',
     *                'mol m-3 uatm-1', ia=ia_srf, power=-5,
     *                denom='ocnfr')
          ijts_gasex(3,n)= ! Gas exchange
     *      ijts_diag('O_Gas_Exchange_'//trim(trname(n)),
     *                trim(trname(n))//' Ocean Gas Exchange',
     *                'mol m-2 yr-1',
     *                denom='ocnfr')

        case ('SF6','SF6_c')
          ijts_source(1,n)=
     *      ijts_diag(trim(trname(n))//'_GRID_SOURCE_LAYER_1',
     *                trim(trname(n))//' Layer 1 SOURCE',
     *                'kg m-2 s-1', power=-15,
     *                scalediv=dtsrc)

        case ('nh5','nh50','nh15')
          ijts_source(1,n)=
     *      ijts_diag(trim(trname(n))//'_NH_Mid_SOURCE_LAYER_1',
     *                trim(trname(n))//' Layer 1 SOURCE',
     *                'kg m-2 s-1', power=-12,
     *                scalediv=dtsrc)

        case ('e90')
          ijts_source(1,n)=
     *      ijts_diag(trim(trname(n))//'_SOURCE_LAYER_1',
     *                trim(trname(n))//' Layer 1 SOURCE',
     *                'kg m-2 s-1', power=-12,
     *                scalediv=dtsrc)

        case ('tape_rec')
          ijts_source(1,n)=
     *      ijts_diag(trim(trname(n))//'_SOURCE_UTLS',
     *                trim(trname(n))//' UTLS SOURCE',
     *                'kg m-2 s-1', power=-12,
     *                scalediv=dtsrc)

        case ('aoa','aoanh')
          ijts_3Dsource(nOverwrite,n)=
     *      ijts_diag(trim(trname(n))//'_overwrite',
     *                trim(trname(n))//' overw',
     *                'kg m-2 s-1', power=-15,
     *                scalediv=dtsrc)

c      case ('Rn222')
c        ijts_source(1,n)=
c     *    ijts_diag(trim(trname(n))//'_SOURCE_Layer_1',
c     *              trim(trname(n))//' L 1 SOURCE',
c     *              'kg m-2 s-1', power=-21,
c     *              scalediv=dtsrc)

c      case ('N2O')
c#ifdef TRACERS_SPECIAL_Lerner
c        ijts_source(nOverwrite,n)=
c     *    ijts_diag(trim(trname(n))//'_CHANGE_IN_L_1',
c     *              trim(trname(n))//' CHANGE IN L 1',
c     *              'kg m-2 s-1', power=-12,
c     *              scalediv=dtsrc)
c#endif

      case ('CFC11')
        ijts_source(1,n)=
     *    ijts_diag(trim(trname(n))//'_SOURCE_LAYER_1',
     *              trim(trname(n))//' L 1 SOURCE',
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)

      case ('14CO2')
        ijts_source(1,n)=
     *    ijts_diag(trim(trname(n))//'_L1_Sink',
     *              trim(trname(n))//' L 1 SINK',
     *              'kg m-2 s-1', power=-21,
     *              scalediv=dtsrc)

      case ('NOx','CO','Isoprene','Alkenes','Paraffin',
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
#ifdef TRACERS_dCO
     *'d13Calke','d13CPAR',
     *'d17OPAN', 'd18OPAN', 'd13CPAN',
     *'dMe17OOH', 'dMe18OOH', 'd13MeOOH',
     *'dHCH17O', 'dHCH18O', 'dH13CHO',
#endif  /* TRACERS_dCO */
     *'dC17O', 'dC18O', 'd13CO',
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
     &'ClOx','BrOx','HCl','HOCl','ClONO2','HBr','HOBr','BrONO2',
     &'CFC','H2O2','CH3OOH','Ox','N2O5','HNO3','HCHO','Terpenes',
     &'HO2NO2','PAN','AlkylNit','Acetone')

        select case(trname(n))
        case('NOx','CO','Isoprene','Alkenes','Paraffin',
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
#ifdef TRACERS_dCO
     *  'd13Calke','d13CPAR',
     *  'd17OPAN', 'd18OPAN', 'd13CPAN',
     *  'dMe17OOH', 'dMe18OOH', 'd13MeOOH',
     *  'dHCH17O', 'dHCH18O', 'dH13CHO',
#endif  /* TRACERS_dCO */
     *  'dC17O', 'dC18O', 'd13CO',
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
     &  'CFC','H2O2','CH3OOH','Ox','N2O5','HNO3','HCHO',
     &  'Terpenes','HO2NO2','PAN','AlkylNit','Acetone')
        select case(trname(n))
        case('NOx')
          ijts_3Dsource(nOther,n)=
     *      ijts_diag(trim(trname(n))//'_lightning',
     *                trim(trname(n))//' lightning',
     *                'kg m-2 s-1', power=-12,
     *                scalediv=dtsrc)
        case('HNO3')
          ijts_3Dsource(nThermo,n)=
     *      ijts_diag(trim(trname(n))//'_thermo',
     *                trim(trname(n))//' thermodynamics',
     *                'kg m-2 s-1', power=-12,
     *                scalediv=dtsrc)
        case('Ox')
          if (nradfrc>0) then
            ijts_fc(1,n)=
     *        ijts_diag('swf_tp_'//trim(trname(n)),
     *                  trim(trname(n))//' tropopause SW rad forc',
     *                  'W m-2', power=-2,
     *                  ia=ia_rad_frc)
            ijts_fc(2,n)=
     *        ijts_diag('lwf_tp_'//trim(trname(n)),
     *                  trim(trname(n))//' tropopause LW rad forc',
     *                  'W m-2', power=-2,
     *                  ia=ia_rad_frc)
            ijts_fc(3,n)=
     *        ijts_diag('swf_toa_'//trim(trname(n)),
     *                  trim(trname(n))//' TOA SW rad forc',
     *                  'W m-2', power=-2,
     *                  ia=ia_rad_frc)
            ijts_fc(4,n)=
     *        ijts_diag('lwf_toa_'//trim(trname(n)),
     *                  trim(trname(n))//' TOA LW rad forc',
     *                  'W m-2', power=-2,
     *                  ia=ia_rad_frc)
#ifdef AUX_OX_RADF_TROP
#ifndef AUXILIARY_OX_RADF
            call stop_model
     &      ('AUX_OX_RADF_TROP needs AUXILIARY_OX_RADF',255)
#endif
#endif
#ifdef AUXILIARY_OX_RADF
            if(trname(n)=='Ox')then
              ijts_auxfc(1)=
     *          ijts_diag('swfauxtp_'//trim(trname(n)),
     *                    trim(trname(n))//' AUX tropp SW rad forc',
     *                    'W m-2', power=-2,
     *                    ia=ia_rad_frc)
              ijts_auxfc(2)=
     *          ijts_diag('lwfauxtp_'//trim(trname(n)),
     *                    trim(trname(n))//' AUX tropp LW rad forc',
     *                    'W m-2', power=-2,
     *                    ia=ia_rad_frc)
              ijts_auxfc(3)=
     *          ijts_diag('swfauxtoa_'//trim(trname(n)),
     *                    trim(trname(n))//' AUX TOA SW rad forc',
     *                    'W m-2', power=-2,
     *                    ia=ia_rad_frc)
              ijts_auxfc(4)=
     *          ijts_diag('lwfauxtoa_'//trim(trname(n)),
     *                    trim(trname(n))//' AUX TOA LW rad forc',
     *                    'W m-2', power=-2,
     *                    ia=ia_rad_frc)
            endif
#endif /* AUXILIARY_OX_RADF */
          endif
        end select
      end select

      case ('CH4')
#ifndef TRACERS_SPECIAL_Shindell /* NOT */
        ijts_3Dsource(1,n)=
     *    ijts_diag(trim(trname(n))//'_trop_chem',
     *              trim(trname(n))//' Tropospheric Chemistry',
     *              'kg m-2 s-1', power=-12,
     *              scalediv=dtsrc)
        ijts_3Dsource(2,n)=
     *    ijts_diag(trim(trname(n))//'_strat_chem',
     *              trim(trname(n))//' Stratospheric Chemistry',
     *              'kg m-2 s-1', power=-12,
     *              scalediv=dtsrc)
#endif

      case ('O3')
        ijts_source(1,n)=
     *    ijts_diag(trim(trname(n))//'_deposition_L1',
     *              trim(trname(n))//' deposition, layer 1',
     *              'kg m-2 s-1', power=-12,
     *              scalediv=dtsrc)
        ijts_3Dsource(1,n)=
     *    ijts_diag(trim(trname(n))//'_strat_chem',
     *              trim(trname(n))//' Stratospheric Chemistry',
     *              'kg m-2 s-1', power=-12,
     *              scalediv=dtsrc)
        ijts_3Dsource(2,n)=
     *    ijts_diag(trim(trname(n))//'_trop_chem_prod',
     *              trim(trname(n))//' Tropo. Chem. Production',
     *              'kg m-2 s-1', power=-12,
     *              scalediv=dtsrc)
        ijts_3Dsource(3,n)=
     *    ijts_diag(trim(trname(n))//'_trop_chem_loss',
     *              trim(trname(n))//' Tropo. Chem. Loss',
     *              'kg m-2 s-1', power=-12,
     *              scalediv=dtsrc)

#ifdef TRACERS_WATER
      case ('Water', 'H2O18', 'H2O17', 'HDO', 'HTO' )
          ! nothing I can think of....
#endif

      case ('NH3', 'NH4', 'NO3p')
        ijts_3Dsource(nThermo,n)=
     *    ijts_diag(trim(trname(n))//'_thermo',
     *              trim(trname(n))//' thermodynamics',
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)

        select case (trname(n))
        case ('NO3p')
          call set_diag_aod(n)
          if (diag_fc==2) call set_diag_rf(n)
        end select

      case ('BCB', 'OCB', 'BCIA', 'OCIA', 'isopp1a')
        call set_diag_aod(n)
        if (diag_fc==2) call set_diag_rf(n)

      case ('SO2')
        ijts_3Dsource(nVolcanic,n)=
     *    ijts_diag(trim(trname(n))//'_volcanic_src',
     *              trim(trname(n))//' volcanic source',
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)
        ijts_3Dsource(nChemprod,n)=
     *    ijts_diag(trim(trname(n))//'_source_from_DMS',
     *              trim(trname(n))//' source from DMS',
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)
        ijts_3Dsource(nChemloss,n)=
     *    ijts_diag(trim(trname(n))//'_chem_sink',
     *              trim(trname(n))//' Chemical sink',
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)

      case ('vbsAm2', 'vbsAm1', 'vbsAz',  'vbsAp1', 'vbsAp2',
     &      'vbsAp3', 'vbsAp4', 'vbsAp5', 'vbsAp6')
        ijts_3Dsource(nChemistry,n)=
     *    ijts_diag(trim(trname(n))//'_partitioning',
     *              trim(trname(n))//' partitioning',
     *              'kg m-2 s-1', power=-12,
     *              scalediv=dtsrc)

        select case(trname(n))
        case ('vbsAm2')
          call set_diag_aod(n)
          if (diag_fc==2) call set_diag_rf(n)
        end select

      case ('DMS')
        ijts_isrc(1,n)=
     *    ijts_diag(trim(trname(n))//'_ocean_src',
     *              trim(trname(n))//' ocean source',
     *              'kg m-2 s-1', power=-12,
     *              scalediv=dtsrc)

      case ('SO4')
c put in production of SO4 from gas phase
        ijts_3Dsource(nChemistry,n)=
     *    ijts_diag(trim(trname(n))//'_gas_phase_source',
     *              trim(trname(n))//' gas phase source',
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)
        ijts_3Dsource(nVolcanic,n)=
     *    ijts_diag(trim(trname(n))//'_volcanic_src',
     *              trim(trname(n))//' volcanic source',
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)

        call set_diag_aod(n)
        if (diag_fc==2) call set_diag_rf(n)

      case ('vbsGm2', 'vbsGm1', 'vbsGz',  'vbsGp1', 'vbsGp2',
     &      'vbsGp3', 'vbsGp4', 'vbsGp5', 'vbsGp6')
        ijts_3Dsource(nChemistry,n)=
     *    ijts_diag(trim(trname(n))//'_aging_src',
     *              trim(trname(n))//' aging source',
     *              'kg m-2 s-1', power=-12,
     *              scalediv=dtsrc)
        ijts_3Dsource(nChemloss,n)=
     *    ijts_diag(trim(trname(n))//'_aging_loss',
     *              trim(trname(n))//' aging loss',
     *              'kg m-2 s-1', power=-12,
     *              scalediv=dtsrc)
        ijts_3Dsource(nOther,n)=
     *    ijts_diag(trim(trname(n))//'_partitioning',
     *              trim(trname(n))//' partitioning',
     *              'kg m-2 s-1', power=-12,
     *              scalediv=dtsrc)

#ifdef TRACERS_AMP
      case ('H2SO4')
        ijts_3Dsource(nMicrophys,n)=
     *    ijts_diag('AMP_src_'//trim(trname(n)),
     *              'AMP_src_'//trim(trname(n)),
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)

      case ('M_NO3   ','M_NH4   ','M_H2O   ','M_AKK_SU','N_AKK_1 ',!AKK
     *    'M_ACC_SU','N_ACC_1 ','M_DD1_SU','M_DD1_DU','N_DD1_1 ',!ACC,DD1
     *    'M_DS1_SU','M_DS1_DU','N_DS1_1 ','M_DD2_SU','M_DD2_DU',!DS1,DD2
     *    'N_DD2_1 ','M_DS2_SU','M_DS2_DU','N_DS2_1 ','M_SSA_SU',!DD2,DS2,SSA
     *    'M_SSA_SS','N_SSA_1 ','M_SSC_SS','N_SSC_1',            !SSA,SSC
     *    'M_OCC_SU','M_OCC_OC','N_OCC_1 ','M_BC1_SU','M_BC1_BC',!OCC,BC1
     *    'N_BC1_1 ','M_BC2_SU','M_BC2_BC','N_BC2_1 ','M_BC3_SU',!BC1,BC2,BC3
     *    'M_BC3_BC','N_BC3_1 ','M_DBC_SU','M_DBC_BC','M_DBC_DU',!BC3,DBC
     *    'N_DBC_1 ','M_BOC_SU','M_BOC_BC','M_BOC_OC','N_BOC_1 ',!DBC,BOC
     *    'M_BCS_SU','M_BCS_BC','N_BCS_1 ','M_MXX_SU','M_MXX_BC',!BCS,MXX
     *    'M_MXX_OC','M_MXX_DU','M_MXX_SS','N_MXX_1 ','M_OCS_SU',
     *    'M_OCS_OC','N_OCS_1 ','M_SSS_SS','M_SSS_SU',
     * 'M_ACC_OCM2','M_ACC_OCM1','M_ACC_OCM0','M_ACC_OCP1','M_ACC_OCP2',
     * 'M_ACC_OCP3','M_ACC_OCP4','M_ACC_OCP5','M_ACC_OCP6',
     * 'M_DD1_OCM2','M_DD1_OCM1','M_DD1_OCM0','M_DD1_OCP1','M_DD1_OCP2',
     * 'M_DD1_OCP3','M_DD1_OCP4','M_DD1_OCP5','M_DD1_OCP6',
     * 'M_DS1_OCM2','M_DS1_OCM1','M_DS1_OCM0','M_DS1_OCP1','M_DS1_OCP2',
     * 'M_DS1_OCP3','M_DS1_OCP4','M_DS1_OCP5','M_DS1_OCP6',
     * 'M_DD2_OCM2','M_DD2_OCM1','M_DD2_OCM0','M_DD2_OCP1','M_DD2_OCP2',
     * 'M_DD2_OCP3','M_DD2_OCP4','M_DD2_OCP5','M_DD2_OCP6',
     * 'M_DS2_OCM2','M_DS2_OCM1','M_DS2_OCM0','M_DS2_OCP1','M_DS2_OCP2',
     * 'M_DS2_OCP3','M_DS2_OCP4','M_DS2_OCP5','M_DS2_OCP6',
     * 'M_SSA_OCM2','M_SSA_OCM1','M_SSA_OCM0','M_SSA_OCP1','M_SSA_OCP2',
     * 'M_SSA_OCP3','M_SSA_OCP4','M_SSA_OCP5','M_SSA_OCP6',
     * 'M_SSC_OCM2','M_SSC_OCM1','M_SSC_OCM0','M_SSC_OCP1','M_SSC_OCP2',
     * 'M_SSC_OCP3','M_SSC_OCP4','M_SSC_OCP5','M_SSC_OCP6',
     * 'M_OCC_OCM2','M_OCC_OCM1','M_OCC_OCM0','M_OCC_OCP1','M_OCC_OCP2',
     * 'M_OCC_OCP3','M_OCC_OCP4','M_OCC_OCP5','M_OCC_OCP6',
     * 'M_BC1_OCM2','M_BC1_OCM1','M_BC1_OCM0','M_BC1_OCP1','M_BC1_OCP2',
     * 'M_BC1_OCP3','M_BC1_OCP4','M_BC1_OCP5','M_BC1_OCP6',
     * 'M_BC2_OCM2','M_BC2_OCM1','M_BC2_OCM0','M_BC2_OCP1','M_BC2_OCP2',
     * 'M_BC2_OCP3','M_BC2_OCP4','M_BC2_OCP5','M_BC2_OCP6',
     * 'M_OCS_OCM2','M_OCS_OCM1','M_OCS_OCM0','M_OCS_OCP1','M_OCS_OCP2',
     * 'M_OCS_OCP3','M_OCS_OCP4','M_OCS_OCP5','M_OCS_OCP6',
     * 'M_BOC_OCM2','M_BOC_OCM1','M_BOC_OCM0','M_BOC_OCP1','M_BOC_OCP2',
     * 'M_BOC_OCP3','M_BOC_OCP4','M_BOC_OCP5','M_BOC_OCP6',
     * 'M_BCS_OCM2','M_BCS_OCM1','M_BCS_OCM0','M_BCS_OCP1','M_BCS_OCP2',
     * 'M_BCS_OCP3','M_BCS_OCP4','M_BCS_OCP5','M_BCS_OCP6',
     * 'M_MXX_OCM2','M_MXX_OCM1','M_MXX_OCM0','M_MXX_OCP1','M_MXX_OCP2',
     * 'M_MXX_OCP3','M_MXX_OCP4','M_MXX_OCP5','M_MXX_OCP6')
        select case (trname(n))
        case ('M_NO3', 'M_NH4', 'M_H2O')
          ijts_3Dsource(nThermo,n)=
     *      ijts_diag(trim(trname(n))//'_thermo',
     *                trim(trname(n))//' thermodynamics',
     *                'kg m-2 s-1', power=-15,
     *                scalediv=dtsrc)
        case default
          ijts_3Dsource(nMicrophys,n)=
     *      ijts_diag('AMP_src_'//trim(trname(n)),
     *                'AMP_src_'//trim(trname(n)),
     *                'kg m-2 s-1', power=-15,
     *                scalediv=dtsrc)
        end select
        ijts_AMPp(1,n)=
     *    ijts_diag('P1_Nucl_'//trim(trname(n)),
     *              'P1_Nucl_'//trim(trname(n)),
     *              ' ', power=-11)
        ijts_AMPp(2,n)=
     *    ijts_diag('P2_Coag_'//trim(trname(n)),
     *              'P2_Coag_'//trim(trname(n)),
     *              ' ', power=-11)
        ijts_AMPp(3,n)=
     *    ijts_diag('P3_Cond_'//trim(trname(n)),
     *              'P3_Cond_'//trim(trname(n)),
     *              ' ', power=-11)
        ijts_AMPp(4,n)=
     *    ijts_diag('P4_Incld_NIMC_'//trim(trname(n)),
     *              'P4_Incld_NIMC_'//trim(trname(n)),
     *              ' ', power=-11)
        ijts_AMPp(5,n)=
     *    ijts_diag('P5_IMLoss_NIAC_'//trim(trname(n)),
     *              'P5_IMLoss_NIAC_'//trim(trname(n)),
     *              ' ', power=-11)
        ijts_AMPp(6,n)=
     *    ijts_diag('P6_Mode_Trans_'//trim(trname(n)),
     *              'P6_Mode_Trans_'//trim(trname(n)),
     *              ' ', power=-11)
        ijts_AMPp(7,n)=
     *    ijts_diag('P7_Total_Change_'//trim(trname(n)),
     *              'P7_Total_Change_'//trim(trname(n)),
     *              ' ', power=-11)
#endif

#ifdef TRACERS_TOMAS
      case ('SOAgas')

c put in production of SO4 from gas phase
        do kr=1,ntsurfsrc(n)
          ijts_source(kr,n)=
     *      ijts_diag(trim(trname(n))//'_terpenes_src',
     *                trim(trname(n))//' terpenes source',
     *                'kg m-2 s-1', power=-15,
     *                scalediv=dtsrc)
        enddo

      case ('H2SO4')

c put in production of SO4 from gas phase
        ijts_3Dsource(nMicrophys,n)=
     *    ijts_diag('Microphysics_chg_'//trim(trname(n)),
     *              'Microphysics change'//trim(trname(n)),
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)

      case('ASO4__01','ASO4__02','ASO4__03','ASO4__04','ASO4__05',
     *    'ASO4__06','ASO4__07','ASO4__08','ASO4__09','ASO4__10',
     *    'ASO4__11','ASO4__12','ASO4__13','ASO4__14','ASO4__15',
     *    'ANACL_01','ANACL_02','ANACL_03','ANACL_04','ANACL_05',
     *    'ANACL_06','ANACL_07','ANACL_08','ANACL_09','ANACL_10',
     *    'ANACL_11','ANACL_12','ANACL_13','ANACL_14','ANACL_15',
     *    'AECIL_01','AECIL_02','AECIL_03','AECIL_04','AECIL_05',
     *    'AECIL_06','AECIL_07','AECIL_08','AECIL_09','AECIL_10',
     *    'AECIL_11','AECIL_12','AECIL_13','AECIL_14','AECIL_15',
     *    'AECOB_01','AECOB_02','AECOB_03','AECOB_04','AECOB_05',
     *    'AECOB_06','AECOB_07','AECOB_08','AECOB_09','AECOB_10',
     *    'AECOB_11','AECOB_12','AECOB_13','AECOB_14','AECOB_15',
     *    'AOCIL_01','AOCIL_02','AOCIL_03','AOCIL_04','AOCIL_05',
     *    'AOCIL_06','AOCIL_07','AOCIL_08','AOCIL_09','AOCIL_10',
     *    'AOCIL_11','AOCIL_12','AOCIL_13','AOCIL_14','AOCIL_15',
     *    'AOCOB_01','AOCOB_02','AOCOB_03','AOCOB_04','AOCOB_05',
     *    'AOCOB_06','AOCOB_07','AOCOB_08','AOCOB_09','AOCOB_10',
     *    'AOCOB_11','AOCOB_12','AOCOB_13','AOCOB_14','AOCOB_15',
     *    'ADUST_01','ADUST_02','ADUST_03','ADUST_04','ADUST_05',
     *    'ADUST_06','ADUST_07','ADUST_08','ADUST_09','ADUST_10',
     *    'ADUST_11','ADUST_12','ADUST_13','ADUST_14','ADUST_15',
     *    'ANUM__01','ANUM__02','ANUM__03','ANUM__04','ANUM__05',
     *    'ANUM__06','ANUM__07','ANUM__08','ANUM__09','ANUM__10',
     *    'ANUM__11','ANUM__12','ANUM__13','ANUM__14','ANUM__15')

        ijts_3Dsource(nMicrophys,n)=
     *    ijts_diag('Microphysics_chg_'//trim(trname(n)),
     *              'Microphysics change'//trim(trname(n)),
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)
        ijts_TOMAS(1,n)=
     *    ijts_diag('MP1_Cond_'//trim(trname(n)),
     *              'MP1_Cond_'//trim(trname(n)),
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)
        ijts_TOMAS(2,n)=
     *    ijts_diag('MP2_Coag_'//trim(trname(n)),
     *              'MP2_Coag_'//trim(trname(n)),
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)
        ijts_TOMAS(3,n)=
     *    ijts_diag('MP3_Nucl_'//trim(trname(n)),
     *              'MP3_Nucl_'//trim(trname(n)),
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)
        ijts_TOMAS(4,n)=
     *    ijts_diag('MP4_Aqoxid_MC_'//trim(trname(n)),
     *              'MP4_Aqoxid_MC_'//trim(trname(n)),
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)
        ijts_TOMAS(5,n)=
     *    ijts_diag('MP5_Aqoxid_LS_'//trim(trname(n)),
     *              'MP5_Aqoxid_LS_'//trim(trname(n)),
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)
        ijts_TOMAS(6,n)=
     *    ijts_diag('MP6_Mk_Nk_Fix_'//trim(trname(n)),
     *              'MP6_Mk_Nk_Fix_'//trim(trname(n)),
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)
        ijts_TOMAS(7,n)=
     *    ijts_diag('MP7_Aeroupdate_'//trim(trname(n)),
     *              'MP7_Aeroupdate_'//trim(trname(n)),
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)
        ijts_subcoag(n)=
     *    ijts_diag('Subgrid_coag_'//trim(trname(n)),
     *              'Subgrid_coag_'//trim(trname(n)),
     *              'kg m-2 s-1', power=-15,
     *              scalediv=dtsrc)

        select case(trname(n))
        case ('ASO4__01','ASO4__02','ASO4__03','ASO4__04','ASO4__05',
     *       'ASO4__06','ASO4__07','ASO4__08','ASO4__09','ASO4__10',
     *       'ASO4__11','ASO4__12','ASO4__13','ASO4__14','ASO4__15')

          ijts_3Dsource(nVolcanic,n)=
     *      ijts_diag(trim(trname(n))//'_volcanic_src',
     *                trim(trname(n))//' volcanic source',
     *                'kg m-2 s-1', power=-15,
     *                scalediv=dtsrc)

        case ('ANUM__01','ANUM__02','ANUM__03','ANUM__04','ANUM__05',
     *    'ANUM__06','ANUM__07','ANUM__08','ANUM__09','ANUM__10',
     *    'ANUM__11','ANUM__12','ANUM__13','ANUM__14','ANUM__15')

          ijts_3Dsource(nSO4anum,n)=
     *      ijts_diag(trim(trname(n))//'_SO4_3D_src',
     *                trim(trname(n))//' SO4 3D source',
     *                '# m-2 s-1', power=10,
     *                scalediv=dtsrc)
          ijts_3Dsource(nECanum,n)=
     *      ijts_diag(trim(trname(n))//'_EC_3D_src',
     *                trim(trname(n))//' EC 3D source',
     *                '# m-2 s-1', power=10,
     *                scalediv=dtsrc)
          ijts_3Dsource(nOCanum,n)=
     *      ijts_diag(trim(trname(n))//'_OC_3D_src',
     *                trim(trname(n))//' OC 3D source',
     *                '# m-2 s-1', power=10,
     *                scalediv=dtsrc)
          ijts_isrc(1,n)=
     *      ijts_diag(trim(trname(n))//'_NACL_src',
     *                trim(trname(n))//' NACL source',
     *                '# m-2 s-1', power=10,
     *                scalediv=dtsrc)
          ijts_isrc(2,n)=
     *      ijts_diag(trim(trname(n))//'_DUST_src',
     *                trim(trname(n))//' DUST source',
     *                '# m-2 s-1', power=10,
     *                scalediv=dtsrc)

        case ('ANACL_01','ANACL_02','ANACL_03','ANACL_04','ANACL_05',
     *        'ANACL_06','ANACL_07','ANACL_08','ANACL_09','ANACL_10',
     *        'ANACL_11','ANACL_12','ANACL_13','ANACL_14','ANACL_15')
          ijts_isrc(1,n)=
     *      ijts_diag(trim(trname(n))//'_emission',
     *                trim(trname(n))//' Ocean source',
     *                'kg m-2 s-1', power=-12,
     *                scalediv=dtsrc)

        case('ADUST_01','ADUST_02','ADUST_03','ADUST_04','ADUST_05',
     *    'ADUST_06','ADUST_07','ADUST_08','ADUST_09','ADUST_10',
     *    'ADUST_11','ADUST_12','ADUST_13','ADUST_14','ADUST_15')

          ijts_isrc(1,n)=
     *      ijts_diag(trim(trname(n))//'_emission',
     *                trim(trname(n))//' Emission',
     *                'kg m-2 s-1', power=-12,
     *                scalediv=dtsrc)
        end select

        select case(trname(n))
        case('ASO4__01','ANACL_01','AECOB_01','AECIL_01',
     &       'AOCOB_01','AOCIL_01','ADUST_01')

        call set_diag_aod(n)
        if (diag_fc==2) then
          call set_diag_rf(n)
        else if (diag_fc==1) then
          select case (trname(n))
            case ('ASO4__01')
              call set_diag_rf(n)
          end select
        endif

      end select

#endif
      case ('H2O2_s')
        ijts_3Dsource(nChemistry,n)=
     *    ijts_diag(trim(trname(n))//'_gas_phase_source',
     *              trim(trname(n))//' gas phase source',
     *              'kg m-2 s-1', power=-10,
     *              scalediv=dtsrc)
        ijts_3Dsource(nChemloss,n)=
     *    ijts_diag(trim(trname(n))//'_gas_phase_sink',
     *              trim(trname(n))//' gas phase sink',
     *              'kg m-2 s-1', power=-10,
     *              scalediv=dtsrc)

      case ('seasalt1', 'seasalt2', 'OCocean')
        ijts_isrc(1,n)=
     *    ijts_diag(trim(trname(n))//'_ocean_src',
     *              trim(trname(n))//' ocean source',
     *              'kg m-2 s-1', power=-12,
     *              scalediv=dtsrc)

#ifdef TRACERS_AEROSOLS_SEASALT
        select case (trname(n))
        case ('seasalt1','seasalt2')
          call set_diag_aod(n)
          if (diag_fc==2) then
            call set_diag_rf(n)
          else if (diag_fc==1) then
c**** this call to set_diag_rf sets up the indexing of the summary
c**** radiative forcing diagnostics for OMA, when diag_fc=1. It only
c**** works when seasalt1 is present and is the first radiatively active
c**** aerosol in the list of tracers.
            select case (trname(n))
              case ('seasalt1')
                call set_diag_rf(n)
            end select
          endif
          
        end select
#endif

#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS)
      CASE('Clay','Silt1','Silt2','Silt3','Silt4','Silt5','ClayIlli'
     &      ,'ClayKaol','ClaySmec','ClayCalc','ClayQuar','ClayFeld'
     &      ,'ClayHema','ClayGyps','ClayIlHe','ClayKaHe','ClaySmHe'
     &      ,'ClayCaHe','ClayQuHe','ClayFeHe','ClayGyHe','Sil1Quar'
     &      ,'Sil1Feld','Sil1Calc','Sil1Hema','Sil1Gyps','Sil1Illi'
     &      ,'Sil1Kaol','Sil1Smec','Sil1QuHe','Sil1FeHe','Sil1CaHe'
     &      ,'Sil1GyHe','Sil1IlHe','Sil1KaHe','Sil1SmHe','Sil2Quar'
     &      ,'Sil2Feld','Sil2Calc','Sil2Hema','Sil2Gyps','Sil2Illi'
     &      ,'Sil2Kaol','Sil2Smec','Sil2QuHe','Sil2FeHe','Sil2CaHe'
     &      ,'Sil2GyHe','Sil2IlHe','Sil2KaHe','Sil2SmHe','Sil3Quar'
     &      ,'Sil3Feld','Sil3Calc','Sil3Hema','Sil3Gyps','Sil3Illi'
     &      ,'Sil3Kaol','Sil3Smec','Sil3QuHe','Sil3FeHe','Sil3CaHe'
     &      ,'Sil3GyHe','Sil3IlHe','Sil3KaHe','Sil3SmHe','Sil4Quar'
     &      ,'Sil4Feld','Sil4Calc','Sil4Hema','Sil4Gyps','Sil4Illi'
     &      ,'Sil4Kaol','Sil4Smec','Sil4QuHe','Sil4FeHe','Sil4CaHe'
     &      ,'Sil4GyHe','Sil4IlHe','Sil4KaHe','Sil4SmHe','Sil5Quar'
     &      ,'Sil5Feld','Sil5Calc','Sil5Hema','Sil5Gyps','Sil5Illi'
     &      ,'Sil5Kaol','Sil5Smec','Sil5QuHe','Sil5FeHe','Sil5CaHe'
     &      ,'Sil5GyHe','Sil5IlHe','Sil5KaHe','Sil5SmHe')
        ijts_isrc(nDustEmij,n)=
     *    ijts_diag(trim(trname(n))//'_emission',
     *              trim(trname(n))//' emission',
     *              'kg m-2 s-1', power=-13,
     *              scalediv=dtsrc)
        IF ( imDust == 0 .or. imDust >= 3 ) THEN
          ijts_isrc(nDustEm2ij,n)=
     *      ijts_diag(trim(trname(n))//'_emission2',
     *                trim(trname(n))//' cubic emission',
     *                'kg m-2 s-1', power=-13,
     *                scalediv=dtsrc)
        END IF
#ifndef TRACERS_DRYDEP
        ijts_isrc(nDustTurbij,n)=
     *    ijts_diag(trim(trname(n))//'_turb_depo',
     *              trim(trname(n))//' turbulent deposition',
     *              'kg m-2 s-1', power=-13,
     *              scalediv=dtsrc)
#endif
#ifndef TRACERS_WATER
        ijts_wet(n)=
     *    ijts_diag(trim(trname(n))//'_wet_depo',
     *              trim(trname(n))//' wet deposition',
     *              'kg m-2 s-1', power=-13,
     *              scalediv=dtsrc)
#endif
        SELECT CASE (trname(n))
        CASE ('Clay','ClayIlli','ClayKaol','ClaySmec','ClayCalc'
     &         ,'ClayQuar','ClayFeld' ,'ClayHema','ClayGyps','ClayIlHe'
     &         ,'ClayKaHe','ClaySmHe' ,'ClayCaHe','ClayQuHe','ClayFeHe'
     &         ,'ClayGyHe')

          call set_diag_aod( n, nSubClays )
          if (diag_fc==2) then
            call set_diag_rf( n, nSubClays )
          else if ( diag_fc == 1 .and. .not. ( tracers_aerosols_seasalt
     &           .or. tracers_aerosols_koch .or. tracers_nitrate ) )
     &           then
c**** This sets up, in a somewhat disguised way, the diagnostic outputs
c**** of summary dust radiative forcing for simulations with standalone
c**** dust/minerals in the case of diag_fc=1.
            select case ( trname( n ) )
            case ('Clay', 'ClayIlli')
              call set_diag_rf( n, nSubClays )
            end select
          end if

        CASE('Silt1','Silt2','Silt3','Silt4','Silt5','Sil1Quar'
     &         ,'Sil1Feld','Sil1Calc','Sil1Hema','Sil1Gyps','Sil1Illi'
     &         ,'Sil1Kaol','Sil1Smec','Sil1QuHe','Sil1FeHe','Sil1CaHe'
     &         ,'Sil1GyHe','Sil1IlHe','Sil1KaHe','Sil1SmHe','Sil2Quar'
     &         ,'Sil2Feld','Sil2Calc','Sil2Hema','Sil2Gyps','Sil2Illi'
     &         ,'Sil2Kaol','Sil2Smec','Sil2QuHe','Sil2FeHe','Sil2CaHe'
     &         ,'Sil2GyHe','Sil2IlHe','Sil2KaHe','Sil2SmHe','Sil3Quar'
     &         ,'Sil3Feld','Sil3Calc','Sil3Hema','Sil3Gyps','Sil3Illi'
     &         ,'Sil3Kaol','Sil3Smec','Sil3QuHe','Sil3FeHe','Sil3CaHe'
     &         ,'Sil3GyHe','Sil3IlHe','Sil3KaHe','Sil3SmHe','Sil4Quar'
     &         ,'Sil4Feld','Sil4Calc','Sil4Hema','Sil4Gyps','Sil4Illi'
     &         ,'Sil4Kaol','Sil4Smec','Sil4QuHe','Sil4FeHe','Sil4CaHe'
     &         ,'Sil4GyHe','Sil4IlHe','Sil4KaHe','Sil4SmHe','Sil5Quar'
     &         ,'Sil5Feld','Sil5Calc','Sil5Hema','Sil5Gyps','Sil5Illi'
     &         ,'Sil5Kaol','Sil5Smec','Sil5QuHe','Sil5FeHe','Sil5CaHe'
     &         ,'Sil5GyHe','Sil5IlHe','Sil5KaHe','Sil5SmHe')
          call set_diag_aod(n)
          if (diag_fc==2) call set_diag_rf(n)

        END SELECT
#endif  /* TRACERS_DUST || TRACERS_MINERALS */

      end select

#if (defined TRACERS_WATER) && (defined TRDIAG_WETDEPO)
c**** additional wet deposition diagnostics
      IF (diag_wetdep == 1) THEN
        ijts_trdpmc(1,n)=
     *    ijts_diag(trim(trname(n))//'_cond_mc',
     *              'MC Condensation of '//trim(trname(n)),
     *              'kg m-2 s-1', power=-13,
     *              scalediv=dtsrc)
        ijts_trdpmc(2,n)=
     *    ijts_diag(trim(trname(n))//'_downeva_mc',
     *              'Evaporated '//trim(trname(n))//' in MC Downdrafts',
     *              'kg m-2 s-1', power=-13,
     *              scalediv=dtsrc)
        ijts_trdpmc(3,n)=
     *    ijts_diag(trim(trname(n))//'_conclw_mc',
     *              'Condensed '//trim(trname(n))//' in MC CLW',
     *              'kg m-2 s-1', power=-13,
     *              scalediv=dtsrc)
        ijts_trdpmc(4,n)=
     *    ijts_diag(trim(trname(n))//'_precip_mc',
     *              'Precipitated '//trim(trname(n))//' by MC',
     *              'kg m-2 s-1', power=-13,
     *              scalediv=dtsrc)
        ijts_trdpmc(5,n)=
     *    ijts_diag(trim(trname(n))//'_reevap_mc',
     *              'Reevaporated '//trim(trname(n))//' from MC Precip',
     *              'kg m-2 s-1', power=-13,
     *              scalediv=dtsrc)
        ijts_trdpmc(6,n)=
     *    ijts_diag(trim(trname(n))//'_washout_mc',
     *              'MC Washout of '//trim(trname(n)),
     *              'kg m-2 s-1', power=-13,
     *              scalediv=dtsrc)
        ijts_trdpls(1,n)=
     *    ijts_diag(trim(trname(n))//'_washout_ls',
     *              'LS Washout of '//trim(trname(n)),
     *              'kg m-2 s-1', power=-13,
     *              scalediv=dtsrc)
        ijts_trdpls(2,n)=
     *    ijts_diag(trim(trname(n))//'_precip_ls',
     *              'Precipitated '//trim(trname(n))//' by LS',
     *              'kg m-2 s-1', power=-13,
     *              scalediv=dtsrc)
        ijts_trdpls(3,n)=
     *    ijts_diag(trim(trname(n))//'_conclw_ls',
     *              'Condensed '//trim(trname(n))//' in LS CLW',
     *              'kg m-2 s-1', power=-13,
     *              scalediv=dtsrc)
        ijts_trdpls(4,n)=
     *    ijts_diag(trim(trname(n))//'_reevap_ls',
     *              'Reevaporated '//trim(trname(n))//' from LS Precip',
     *              'kg m-2 s-1', power=-13,
     *              scalediv=dtsrc)
        ijts_trdpls(5,n)=
     *    ijts_diag(trim(trname(n))//'_clwevap_ls',
     *              'Evaporated '//trim(trname(n))//' from LS CLW',
     *              'kg m-2 s-1', power=-13,
     *              scalediv=dtsrc)
        ijts_trdpls(6,n)=
     *    ijts_diag(trim(trname(n))//'_cond_ls',
     *              'LS Condensation of '//trim(trname(n)),
     *              'kg m-2 s-1', power=-13,
     *              scalediv=dtsrc)
      END IF
#endif
      end do

C**** Additional Special IJ diagnostics
C**** (not necessary associated with a particular tracer)
#ifdef BC_ALB
c BC impact on grain size
c         k = k + 1
c         ijts_alb(2,n) = k
c         ia_ijts(k) = ia_rad????
c         lname_ijts(k) = 'BC impact on grain size'
c         sname_ijts(k) = 'grain_BC'
c         ijts_power(k) = -9
c         units_ijts(k) = unit_string(ijts_power(k),' ')
c         scale_ijts(k) = 10.**(-ijts_power(k))
c BC impact on albedo
      if (nradfrc>0) then
        ijts_alb(1)=
     *    ijts_diag('alb_BC',
     *              'BC impact on snow albedo of land/seaice',
     *              '%', ia=ia_rad_frc,
     *              scalediv=1.d-2, denom='sunlit_snow_freq')

c SW forcing from albedo change
        ijts_alb(2)=
     *    ijts_diag('swf_BCALB',
     *              'BCalb SW radiative forcing',
     *              'W m-2', power=-2, ia=ia_rad_frc)
      endif

#endif
#ifdef TRACERS_SPECIAL_Shindell
#ifdef BIOGENIC_EMISSIONS
      ijs_isoprene=
     *  ijts_diag('Int_isop',
     *            'Interactive isoprene source',
     *            'kg m-2 s-1', power=-10,
     *            scalediv=dtsrc)
#endif
      ijs_NO2_1030=
     *  ijts_diag('NO2_1030',
     *            'NO2 10:30 trop col',
     *            'molecules cm-2', power=15,
     *            denom='NO2_1030c')
      ijs_NO2_1030c=
     *  ijts_diag('NO2_1030c',
     *            'count NO2 10:30 trop col',
     *            'number of accum')
      ijs_NO2_1330=
     *  ijts_diag('NO2_1330',
     *            'NO2 13:30 trop col',
     *            'molecules cm-2', power=15,
     *            denom='NO2_1330c')
      ijs_NO2_1330c=
     *  ijts_diag('NO2_1330c',
     *            'count NO2 13:30 trop col',
     *            'number of accum')
      ijs_O3mass=
     *  ijts_diag('O3_Total_Mass',
     *            'Total Column Ozone (not Ox) Mass',
     *            'kg m-2', power=-4)
#ifdef ACETONE_OCEAN
      ijs_AcetOtoA= ijts_diag('Acetone_Ocean_To_Air',
     *            'Rate of Acetone flux from Ocean to Air',
     *            'kg m-2 s-1', power=-15)! , scalediv=dtsrc)
      ijs_AcetAtoO= ijts_diag('Acetone_Air_To_Ocean',
     *            'Rate of Acetone flux from Air to Ocean',
     *            'kg m-2 s-1', power=-15)! , scalediv=dtsrc)
#endif /* ACETONE_OCEAN */
#endif  /* TRACERS_SPECIAL_Shindell */
#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS)
      ijts_spec(nDustEv1ij)=
     *  ijts_diag('no_dust_ev1',
     *            'No. dust events',
     *            'd-1',
     *            scalediv=dtsrc/SECONDS_PER_DAY)
      ijts_spec(nDustEv2ij)=
     *  ijts_diag('no_dust_ev2',
     *            'No. dust events above threshold wind',
     *            'd-1',
     *            scalediv=dtsrc/SECONDS_PER_DAY)
      ijts_spec(nDustWthij)=
     *  ijts_diag('wtrsh',
     *            'Threshold velocity for dust emission',
     *            'm s-1')
#endif

#ifdef TRACERS_AMP

        ijts_tauint=
     &    ijts_diag('tau_2D',
     &              'aerosol optical depth (vert.+spec. integrated)',
     &              ' ', power=0,ia=ia_rad)

        ijts_taustratint=
     &    ijts_diag('tau_2D_strat',
     &        'aerosol optical depth, stratosphere (vert.+spec. int.)',
     &        ' ', power=0,ia=ia_rad)

        ijts_reffstratint=  
     &    ijts_diag('Reff_2D_strat',
     &    'aerosol effective radius, stratospheric mean (AOD-weighted)',
     &    'um', power=0, denom='tau_2D_strat',ia=ia_rad)

      do n=1,NTM
        pTracer => tracers%getReference(trname(n))
        sources => pTracer%surfaceSources
        select case(trname(n))
        case('M_AKK_SU','M_ACC_SU','Water')
          ijts_3Dsource(nVolcanic,n)=
     *      ijts_diag(trim(trname(n))//'_volcanic_src',
     *                trim(trname(n))//' volcanic source',
     *                'kg m-2 s-1', power=-15,
     *                scalediv=dtsrc)
c- interactive sources diagnostic
        case('M_DD1_DU','M_SSA_SS','M_SSC_SS','M_DD2_DU','M_SSS_SS')
          ijts_isrc(1,n)=
     *      ijts_diag('Emission_'//trim(trname(n)),
     *                'Emission_'//trim(trname(n)),
     *                'kg m-2 s-1', power=-15,
     *                scalediv=dtsrc)

        case('N_AKK_1 ','N_ACC_1 ','N_DD1_1 ','N_DS1_1 ','N_DD2_1 ',
     *       'N_DS2_1 ','N_SSA_1 ','N_SSC_1 ','N_OCC_1 ','N_BC1_1 ',
     *       'N_BC2_1 ','N_BC3_1 ','N_DBC_1 ','N_BOC_1 ','N_BCS_1 ',
     *       'N_MXX_1 ','N_OCS_1 ')

        ijts_2Dreff(n)=
     &    ijts_diag('Reff_2D_'//trim(trname(n)),
     &          trim(trname(n))//' effective radius (AOD-wgt vert avg)',
     &          'um', power=0,ia=ia_rad,
     &          denom='ext_band6_'//trim(trname(n)))

        call set_diag_aod(n)
        if (diag_fc==2) then
          call set_diag_rf(n)
        else if (diag_fc==1) then
          select case (trname(n))
            case ('N_AKK_1')
              call set_diag_rf(n)
          end select
        endif

        end select
      end do

c - Tracer independent Diagnostic (stays here if 2D, moves to ijlt if 3D)
         ijts_AMPe(1)=
     *    ijts_diag('PM1',
     *              'PM1 Mixing ratio',
     *              'kg kg-1', power=-9, ia=ia_src)
         ijts_AMPe(2)=
     *    ijts_diag('PM2p5',
     *              'PM2p5 Mixing ratio',
     *              'kg kg-1', power=-9, ia=ia_src)
         ijts_AMPe(3)=
     *    ijts_diag('PM10',
     *              'PM10 Mixing ratio',
     *              'kg kg-1', power=-9, ia=ia_src)

#endif  /* TRACERS_AMP */

c
c Append some denominator fields if necessary
c
      if(any(dname_ijts.eq.'clrsky')) then
        ijts_clrsky=
     *    ijts_diag('clrsky',
     *              'CLEAR SKY FRACTION',
     *              '%',
     *              ia=ia_rad,
     *              scalediv=1.d-2)
      endif

      if(any(dname_ijts.eq.'ocnfr')) then
        ijts_pocean=
     *    ijts_diag('ocnfr',
     *              'OCEAN FRACTION',
     *              '%',
     *              scalediv=1.d-2)
      endif

      if(any(dname_ijts.eq.'sunlit_snow_freq')) then ! snow albedo weight
        ijts_sunlit_snow=
     *    ijts_diag('sunlit_snow_freq',
     *              'SUNLIT SNOW FREQUENCY',
     *              '%',
     *              ia=ia_rad_frc,
     *              scalediv=1.d-2)
      endif

c find indices of denominators
      call FindStrings(dname_ijts,sname_ijts,denom_ijts,ktaijs)

#endif /* TRACERS_ON */

      return
      end subroutine init_ijts_diag

#ifdef TRACERS_ON
      subroutine FindStrings(StringsToFind,ListOfStrings,Indices,n)
!@sum FindStrings finds the positions of a list of strings in a 2nd list.
!     Needs optimization.
      use mdiag_com, only : sname_strlen
      implicit none
      integer :: n
      character(len=sname_strlen), dimension(n) ::
     &     StringsToFind,ListOfStrings
      integer, dimension(n) :: Indices
      integer :: k,kk
      logical :: found
      do k=1,n
        if(len_trim(StringsToFind(k)).gt.0) then
          found = .false.
          do kk=1,n
            if(trim(ListOfStrings(kk)).eq.trim(StringsToFind(k))) then
              Indices(k) = kk
              found = .true.
              exit
            endif
          enddo
          if(.not.found) then
            write(6,*) 'FindStrings: string '//
     &           trim(StringsToFind(k))//' not found'
            call stop_model('FindStrings: string not found',255)
          endif
        endif
      enddo
      end subroutine FindStrings
#endif

#ifdef TRACERS_ON
      subroutine set_diag_aod(n,n_subclasses)
!@sum set_diag_aod saves extinction, scattering and asymmetry parameter diags
!@auth Dorothy Koch, modified by Kostas Tsigaridis
      use OldTracer_mod, only: trname
      use mdiag_com, only : sname_strlen
      USE TRDIAG_COM, only: diag_rad,ijts_tau,ijts_sqex,ijts_sqsc
     &                     ,ijts_sqcb
     &                     ,ijts_tausub,ijts_sqexsub,ijts_sqscsub
     &                     ,ijts_sqcbsub,save_dry_aod,MaxBand
      USE DIAG_COM, only: ia_rad
#if defined(USE_PLANET_RAD) && defined(GISS_RAD_OFF)
      USE planet_rad, ONLY: l_aerosol_diag, n_band_diag
#endif
      implicit none

      integer, intent(in) :: n
!@var n_subclasses optional argument for the number of sub classes of a given
!@+  tracer (>= 1)
      integer, optional, intent(in) :: n_subclasses
      character*50 :: unit_string
!@param sascs short name of all-sky/clear-sky selector
!@param lascs long name of all-sky/clear-sky selector
!@var s index of sascs and lascs
!@var kr index of solar bands
!@var skr value of kr as a string
!@var sn1 value of n1 as a string
      character(len=sname_strlen), parameter :: dname='clrsky'
      character(len=10), parameter, dimension(3) ::
     &  sascs=(/'    ','CS_ ','DRY_'/),
     &  lascs=(/'         ','clear sky','dry aeros'/)
      integer :: k,kr,s,n1,n_sub
      character(len=1) :: skr,sn1

      interface
        integer function ijts_diag(sname,lname,units,ia,power,denom,
     *                             scalediv)
          character(len=*), intent(in) :: sname, lname, units
          integer, intent(in), optional :: ia
          integer, intent(in), optional :: power
          character(len=*), intent(in), optional :: denom
          real*8, intent(in), optional :: scalediv
        end function ijts_diag
      end interface

      n_sub=1
      if (present(n_subclasses)) n_sub=n_subclasses

! aerosol optical depth and related diagnostics
#if defined(USE_PLANET_RAD) && defined(GISS_RAD_OFF)
      if (l_aerosol_diag) then

      if (n_band_diag > MaxBand)
     &  call stop_model('Increase MaxBand in TRDIAG_COM', 255)
#endif

      do s=1,size(sascs)
        if (trim(sascs(s)).eq.'DRY_' .and. save_dry_aod.eq.0) cycle
        IF (diag_rad /= 1) THEN
! aerosol optical depth for band6
          do n1=1,n_sub
            if (n_sub == 1) then
              sn1=' '
            else
              sn1=char(48+n1)
            end if

            if (trim(sascs(s))=='CS_') then
              k=ijts_diag('tau_'//trim(sascs(s))//trim(trname(n))//
     *                      trim(sn1),
     *                    trim(trname(n))//trim(sn1)//' '//
     *                      trim(lascs(s))//' aerosol optical depth',
     *                    ' ', power=-2, ia=ia_rad, denom=trim(dname))
            else if (trim(sascs(s))=='DRY_') then
              k=ijts_diag('tau_'//trim(sascs(s))//trim(trname(n))//
     *                      trim(sn1),
     *                    trim(trname(n))//trim(sn1)//' '//
     *                      trim(lascs(s))//' aerosol optical depth',
     *                    ' ', power=-2, ia=ia_rad)
            else
              k=ijts_diag('tau_'//trim(sascs(s))//trim(trname(n))//
     *                      trim(sn1),
     *                    trim(trname(n))//' aerosol optical depth',
     *                    ' ', power=-2, ia=ia_rad)
            endif

            if (n_sub == 1) then
              ijts_tau(s,n)=k
            else
              ijts_tausub(s,n,n1)=k
            end if
          end do                ! n1
        ELSE
#if defined(USE_PLANET_RAD) && defined(GISS_RAD_OFF)
          DO kr=1,n_band_diag
#else
          DO kr=1,6
#endif
            write (skr,'(i1)') kr

            do n1=1,n_sub
              if (n_sub == 1) then
                sn1=' '
              else
                sn1=char(48+n1)
              end if

! extinction aerosol optical depth in six solar bands
              if (trim(sascs(s))=='CS_') then
                k=ijts_diag('ext_'//trim(sascs(s))//'band'//skr//'_'//
     *                        trim(trname(n))//trim(sn1),
     *                      trim(trname(n))//trim(sn1)//' '//
     *                        trim(lascs(s))//' SW extinction band '//
     *                        skr,
     *                      ' ', power=-4, ia=ia_rad, denom=trim(dname))
              else if (trim(sascs(s))=='DRY_') then
                k=ijts_diag('ext_'//trim(sascs(s))//'band'//skr//'_'//
     *                        trim(trname(n))//trim(sn1),
     *                      trim(trname(n))//trim(sn1)//' '//
     *                        trim(lascs(s))//' SW extinction band '//
     *                        skr,
     *                      ' ', power=-4, ia=ia_rad)
              else
                k=ijts_diag('ext_'//trim(sascs(s))//'band'//skr//'_'//
     *                        trim(trname(n))//trim(sn1),
     *                      trim(trname(n))//' SW extinction band '//
     *                        skr,
     *                      ' ', power=-4, ia=ia_rad)
              endif

              if (n_sub == 1) then
                ijts_sqex(s,kr,n)=k
              else
                ijts_sqexsub(s,kr,n,n1)=k
              end if

! scattering aerosol optical depth in six solar bands
              if (trim(sascs(s))=='CS_') then
                k=ijts_diag('sct_'//trim(sascs(s))//'band'//skr//'_'//
     *                        trim(trname(n))//trim(sn1),
     *                      trim(trname(n))//trim(sn1)//' '//
     *                        trim(lascs(s))//' SW scattering band '//
     *                        skr,
     *                      ' ', power=-4, ia=ia_rad, denom=trim(dname))
              else
                k=ijts_diag('sct_'//trim(sascs(s))//'band'//skr//'_'//
     *                        trim(trname(n))//trim(sn1),
     *                      trim(trname(n))//' SW scattering band '//
     *                        skr,
     *                      ' ', power=-4, ia=ia_rad)
              endif

              if (n_sub == 1) then
                ijts_sqsc(s,kr,n)=k
              else
                ijts_sqscsub(s,kr,n,n1)=k
              end if

! scattering asymmetry factor in six solar bands
              if (trim(sascs(s))=='CS_') then
                k=ijts_diag('asf_'//trim(sascs(s))//'band'//skr//'_'//
     *                        trim(trname(n))//trim(sn1),
     *                      trim(trname(n))//trim(sn1)//' '//
     *                        trim(lascs(s))//
     *                        ' SW asymmetry factor band '//skr,
     *                      ' ', power=-2, ia=ia_rad, denom=trim(dname))
              else if (trim(sascs(s))=='DRY_') then
                k=ijts_diag('asf_'//trim(sascs(s))//'band'//skr//'_'//
     *                        trim(trname(n))//trim(sn1),
     *                      trim(trname(n))//trim(sn1)//' '//
     *                        trim(lascs(s))//
     *                        ' SW asymmetry factor band '//skr,
     *                      ' ', power=-2, ia=ia_rad)
              else
                k=ijts_diag('asf_'//trim(sascs(s))//'band'//skr//'_'//
     *                        trim(trname(n))//trim(sn1),
     *                      trim(trname(n))//
     *                        ' SW asymmetry factor band '//skr,
     *                      ' ', power=-2, ia=ia_rad)
              endif

              if (n_sub == 1) then
                ijts_sqcb(s,kr,n)=k
              else
                ijts_sqcbsub(s,kr,n,n1)=k
              end if
            end do              ! n1
          END DO                ! kr
        END IF
      enddo                     ! s
#if defined(USE_PLANET_RAD) && defined(GISS_RAD_OFF)
      end if
#endif

      end subroutine set_diag_aod


      subroutine set_diag_rf(n,n_subclasses)
!@sum set_diag_rf saves shortwave and longwave forcing, for all-sky and
!@+               clear-sky, at surface and TOA
!@auth Kostas Tsigaridis
      use OldTracer_mod, only: trname
      use RunTimeControls_mod, only: tracers_amp, tracers_tomas,
     &     tracers_dust, tracers_minerals, tracers_aerosols_seasalt,
     &     tracers_aerosols_koch,tracers_nitrate
      use mdiag_com, only : sname_strlen,lname_strlen
      USE TRDIAG_COM, only: ijts_fc,ijts_fcsub
      USE DIAG_COM, only: ia_rad_frc
      use RAD_COM, only: nradfrc,diag_fc
      implicit none

      integer, intent(in) :: n
!@var n_subclasses optional argument for the number of sub classes of a given
!@+  tracer (>= 1)
      integer, optional, intent(in) :: n_subclasses
      character*50 :: unit_string
!@param sascs short name of all-sky/clear-sky selector
!@param lascs long name of all-sky/clear-sky selector
!@param stoasrf short name of toa/surf selector
!@param ltoasrf long name of toa/surf selector
!@param sswlw short name of swf/lwf selector
!@param lswlw long name of swf/lwf selector
!@var s index of sascs and lascs
!@var l index of stoasrf and ltoasrf
!@var f index of sswlw and lswlw
!@var i combined index of s,l,f
!@var kr index of solar bands
!@var skr value of kr as a string
!@var sn1 value of n1 as a string
      character(len=sname_strlen) :: sname
      character(len=lname_strlen) :: lname
      character(len=sname_strlen), parameter :: dname='clrsky'
      character(len=10), parameter, dimension(2) ::
     &  sascs=(/'   ','CS_'/),lascs=(/'         ','clear sky'/),
     &  stoasrf=(/'     ','surf_'/),ltoasrf=(/'TOA    ','surface'/),
     &  sswlw=(/'swf_','lwf_'/),lswlw=(/'shortwave','longwave '/)
      integer :: k,kr,s,l,f,i,n1,n_sub
      character(len=1) :: skr,sn1
      character(len=20) :: spcname ! following MAX_LEN_NAME=20 for trname
!@var flag to determine whether dust/minerals are the sole radiative tracers
      logical :: l_dust_standalone

      interface
        integer function ijts_diag(sname,lname,units,ia,power,denom,
     *                             scalediv)
          character(len=*), intent(in) :: sname, lname, units
          integer, intent(in), optional :: ia
          integer, intent(in), optional :: power
          character(len=*), intent(in), optional :: denom
          real*8, intent(in), optional :: scalediv
        end function ijts_diag
      end interface

      n_sub=1
      if (present(n_subclasses)) n_sub=n_subclasses

      l_dust_standalone = .false.
      l_dust_standalone = ( tracers_dust .or. tracers_minerals ) .and.
     &     .not. (tracers_aerosols_seasalt .or. tracers_aerosols_koch
     &     .or. tracers_nitrate )

! radiative forcing and related diagnostics

      if (diag_fc==2) then
        spcname=trim(trname(n))
      else if (diag_fc==1) then
        if (tracers_amp) then
          spcname='AMP'
        elseif (tracers_tomas) then
          spcname='TOMAS'
        else if ( l_dust_standalone ) then
          spcname='OMAdust'
        else
          spcname='OMA'
        endif
      endif

      if (nradfrc>0) then
        do s=1,size(sascs)
        do l=1,size(stoasrf)
        do f=1,size(sswlw)
          i=(s-1)*size(stoasrf)*size(sswlw)+(l-1)*size(sswlw)+f
          do n1=1,n_sub
            if (n_sub == 1) then
              sn1=' '
            else
              if ( diag_fc == 2 ) then
                sn1=char(48+n1)
              else if ( diag_fc == 1 .and. l_dust_standalone ) then
                sn1=' '
              end if
            end if

            sname = trim(sswlw(f))
            lname = trim(spcname)//trim(sn1)//' '//trim(lswlw(f))
            if (trim(sascs(s))=='CS_') then
              sname = trim(sname)//trim(sascs(s))
              lname = trim(lname)//' '//trim(lascs(s))
            endif
            if (trim(stoasrf(l))=='surf_') then
              sname = trim(sname)//trim(stoasrf(l))
              lname = trim(lname)//' '//trim(ltoasrf(l))
            endif
            sname = trim(sname)//trim(spcname)//trim(sn1)
            lname = trim(lname)//' radiative forcing'

            if (trim(sascs(s))=='CS_') then
              k=ijts_diag(trim(sname), trim(lname),
     *                    'W m-2', power=-2, ia=ia_rad_frc,
     *                    denom=trim(dname))
            else
              k=ijts_diag(trim(sname), trim(lname),
     *                    'W m-2', power=-2, ia=ia_rad_frc)
            endif

            if (n_sub == 1) then
              ijts_fc(i,n)=k
            else
              ijts_fcsub(i,n,n1)=k
c**** for standalone dust/minerals
              if ( diag_fc == 1 .and. l_dust_standalone ) exit
            end if
          end do ! n1
        enddo ! f
        enddo ! l
        enddo ! s
      endif

      return
      end subroutine set_diag_rf
#endif  /* TRACERS_ON */

      subroutine init_ijlts_diag
!@sum init_ijlts_diag Initialise lat/lon/height tracer diags
!@auth Gavin Schmidt
      USE DOMAIN_DECOMP_ATM, only: AM_I_ROOT
      use OldTracer_mod, only: trname,max_len_name
      USE TRACER_COM, only: ntm
      use rad_com, only: nraero_aod,ntrix_aod
      use ghgmod, only: save_dQ_for_NINT
#ifdef TRACERS_ON
      USE TRDIAG_COM
#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS)
      use trdust_mod, only: nSubClays
#endif
#endif /* TRACERS_ON */
      USE MODEL_COM, only: dtsrc
      USE DIAG_COM
#ifdef SOA_DIAGS
      use tracers_soa, only: issoa
#endif  /* SOA_DIAGS */
      implicit none
      integer n,i
      character*50 :: unit_string
      character(len=max_len_name) :: trname_curr
      character*1 :: clay_num
      integer :: iclay

      interface
        integer function ijlt_diag(sname,lname,units,ia,power,denom)
          character(len=*), intent(in) :: sname, lname, units
          integer, intent(in), optional :: ia
          integer, intent(in), optional :: power
          character(len=*), intent(in), optional :: denom
        end function ijlt_diag
      end interface

#ifdef TRACERS_ON
      ir_ijlt = ir_log2  ! default
      ia_ijlt = ia_src   ! default
      denom_ijlt(:) = 0
#ifdef TRACERS_AMP
      ijlt_AMPm(:,:)=0
#endif

C**** use this routine to set 3D tracer-related diagnostics.

C**** some tracer specific 3D arrays
      if (nraero_aod>0) then
      if (diag_aod_3d>0 .and. diag_aod_3d<5) then ! valid values are 1-4
        allocate(ijlt_3Dtau(nraero_aod))    ; ijlt_3Dtau = 0
        allocate(ijlt_3DtauCS(nraero_aod))  ; ijlt_3DtauCS = 0
        allocate(ijlt_3DtauDRY(nraero_aod))  ; ijlt_3DtauDRY = 0
        allocate(ijlt_3Daaod(nraero_aod))   ; ijlt_3Daaod = 0
        allocate(ijlt_3DaaodCS(nraero_aod)) ; ijlt_3DaaodCS = 0
        allocate(ijlt_3DaaodDRY(nraero_aod)) ; ijlt_3DaaodDRY = 0
#ifdef TRACERS_AMP
        allocate(ijlt_3Dreff(nraero_aod)) ; ijlt_3Dreff = 0
#endif

        iclay=0
        do n=1,nraero_aod
          trname_curr=trim(trname(ntrix_aod(n)))
          select case (trname(ntrix_aod(n)))
            case ('Clay','ClayIlli','ClayKaol','ClaySmec','ClayCalc'
     &           ,'ClayQuar','ClayFeld','ClayHema','ClayGyps'
     &           ,'ClayIlHe','ClayKaHe','ClaySmHe','ClayCaHe'
     &           ,'ClayQuHe','ClayFeHe','ClayGyHe')
              iclay=iclay+1
              write(clay_num, '(i1)') iclay
              trname_curr=trim(trname_curr)//trim(clay_num)
#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS)
              if (iclay .eq. nSubClays) iclay = 0
#endif
          end select

          if (diag_aod_3d==1 .or. diag_aod_3d==3) then
            ijlt_3Dtau(n)=
     &        ijlt_diag(ia=ia_rad,
     &                  sname='tau_3D_'//trim(trname_curr),
     &                  lname=trim(trname_curr)//' tau',
     &                  units=' ', power=-2)
            ijlt_3Daaod(n)=
     &        ijlt_diag(ia=ia_rad,
     &                  sname='aaod_3D_'//trim(trname_curr),
     &                  lname=trim(trname_curr)//' aaod',
     &                  units=' ', power=-2)
#ifdef TRACERS_AMP
            if (diag_reff_3d > 0) then

                ijlt_3Dreff(n)=
     &            ijlt_diag(ia=ia_rad,
     &                     sname='Reff_3D_'//trim(trname_curr),
     &                     lname=trim(trname_curr)//' effective radius',
     &                     units='um', power=0,
     &                     denom='tau_3D_'//trim(trname_curr))
            endif ! diag_reff_3d>0
#endif
          endif ! diag_aod_3d = 1 or 3

          if (diag_aod_3d==2 .or. diag_aod_3d==3) then
            ijlt_3DtauCS(n)=
     &        ijlt_diag(ia=ia_rad,
     &                  sname='tau_3D_CS_'//trim(trname_curr),
     &                  lname=trim(trname_curr)//' CS tau',
     &                  units=' ', power=-2,
     &                  denom='clrsky2d')
            ijlt_3DaaodCS(n)=
     &        ijlt_diag(ia=ia_rad,
     &                  sname='aaod_3D_CS_'//trim(trname_curr),
     &                  lname=trim(trname_curr)//' CS aaod',
     &                  units=' ', power=-2,
     &                  denom='clrsky2d')
          endif ! diag_aod_3d = 2 or 3

          if (diag_aod_3d==4 .or. diag_aod_3d==3) then
          if (save_dry_aod>0) then
            ijlt_3DtauDRY(n)=
     &        ijlt_diag(ia=ia_rad,
     &                  sname='tau_3D_DRY_'//trim(trname_curr),
     &                  lname=trim(trname_curr)//' DRY tau',
     &                  units=' ', power=-2)
            ijlt_3DaaodDRY(n)=
     &        ijlt_diag(ia=ia_rad,
     &                  sname='aaod_3D_DRY_'//trim(trname_curr),
     &                  lname=trim(trname_curr)//' DRY aaod',
     &                  units=' ', power=-2)
          endif ! save_dry_aod>0
          endif ! diag_aod_3d = 4 or 3
        enddo ! nraero_aod
      else if (diag_aod_3d<0 .and. diag_aod_3d>-5) then ! if negative, save total
        allocate(ijlt_3Dtau(1))    ; ijlt_3Dtau = 0
        allocate(ijlt_3DtauCS(1))  ; ijlt_3DtauCS = 0
        allocate(ijlt_3DtauDRY(1))  ; ijlt_3DtauDRY = 0
        allocate(ijlt_3Daaod(1))   ; ijlt_3Daaod = 0
        allocate(ijlt_3DaaodCS(1)) ; ijlt_3DaaodCS = 0
        allocate(ijlt_3DaaodDRY(1)) ; ijlt_3DaaodDRY = 0

        if (diag_aod_3d==-1 .or. diag_aod_3d==-3) then
          ijlt_3Dtau(1)=
     &      ijlt_diag(ia=ia_rad,
     &                sname='tau_3D',
     &                lname='tau',
     &                units=' ', power=-2)
          ijlt_3Daaod(1)=
     &      ijlt_diag(ia=ia_rad,
     &                sname='aaod_3D',
     &                lname='aaod',
     &                units=' ', power=-2)

#ifdef TRACERS_AMP
          if (diag_reff_3d > 0) then
            ijlt_3Dreff_allspec = 0

            ijlt_3Dreff_allspec=
     &          ijlt_diag(ia=ia_rad,
     &           sname='Reff_3D_allspec',
     &           lname='effective radius (all species weighted by AOD)',
     &           units='um', power=0,
     &           denom='tau_3D')
          endif ! diag_reff_3d>0
#endif
        endif ! diag_aod_3d = -1 or -3

        if (diag_aod_3d==-2 .or. diag_aod_3d==-3) then
          ijlt_3DtauCS(1)=
     &      ijlt_diag(ia=ia_rad,
     &                sname='tau_3D_CS',
     &                lname='CS tau',
     &                units=' ', power=-2,
     &                denom='clrsky2d')
          ijlt_3DaaodCS(1)=
     &      ijlt_diag(ia=ia_rad,
     &                sname='aaod_3D_CS',
     &                lname='CS aaod',
     &                units=' ', power=-2,
     &                denom='clrsky2d')
        endif ! diag_aod_3d = -2 or -3

        if (diag_aod_3d==-4 .or. diag_aod_3d==-3) then
          if (save_dry_aod>0) then
            ijlt_3DtauDRY(1)=
     &        ijlt_diag(ia=ia_rad,
     &                  sname='tau_3D_DRY',
     &                  lname='DRY tau',
     &                  units=' ', power=-2)
            ijlt_3DaaodDRY(1)=
     &        ijlt_diag(ia=ia_rad,
     &                  sname='aaod_3D_DRY',
     &                  lname='DRY aaod',
     &                  units=' ', power=-2)
          endif ! save_dry_aod>0
        endif ! diag_aod_3d = -4 or -3
      endif ! 0<diag_aod_3d<5 or 0>diag_aod_3d>-5

      endif ! nraero_aod>0

#if (defined TRACERS_AEROSOLS_Koch && defined TRACERS_NITRATE) || \
    (defined TRACERS_AMP)
C**** output fraction of sulfate that contains NH4
      ijlt_fracso4hasnh4=
     &  ijlt_diag(ia=ia_rad,
     &            sname='fracSO4hasNH4',
     &            lname='fraction of SO4 with at least 1 NH4',
     &            units=' ')
#endif /* (Koch && NITRATE)||AMP */

      do n=1,NTM
        select case(trname(n))

#ifdef SAVE_AEROSOL_3DMASS_FOR_NINT
        CASE('Clay','Silt1','Silt2','Silt3','Silt4','Silt5', 'isopp1a'
     $       ,'isopp2a','apinp1a','apinp2a','OCB','OCII','OCIA','BCB'
     $       ,'BCII' ,'BCIA', 'SO4','MSA','NO3p','NH4','seasalt1'
     $       ,'seasalt2','SO4_d1','SO4_d2','SO4_d3','N_d1','N_d2'
     $       ,'N_d3')
        ijlt_3Dmass(n)=
     &    ijlt_diag(sname='Mass_3D_'//trim(trname(n)),
     &              lname=trim(trname(n))//' mass',
     &              units='kg m-2', power=-5)
#endif /* define io parameters for 3Dmass diagnostic (Ron) */

#ifdef TRACERS_AMP
c- 3D diagnostic per mode
        CASE('N_AKK_1 ','N_ACC_1 ','N_DD1_1 ','N_DS1_1 ','N_DD2_1 ',
     *       'N_DS2_1 ','N_SSA_1 ','N_SSC_1 ','N_OCC_1 ','N_BC1_1 ',
     *       'N_BC2_1 ','N_BC3_1 ','N_DBC_1 ','N_BOC_1 ','N_BCS_1 ',
     *       'N_MXX_1 ','N_OCS_1 ')
          ijlt_AMPm(1,n)=
     &      ijlt_diag(sname='DIAM_'//trim(trname(n)),
     &                lname=trim(trname(n))//' DIAM',
     &                units='m', power=-2,denom=trim(trname(n)))
          ijlt_AMPm(2,n)=
     &      ijlt_diag(sname='ACTI3D_'//trim(trname(n)),
     &                lname=trim(trname(n))//' ACTI',
     &                units='#', power=-2)
          ijlt_AMPm(3,n)=
     &      ijlt_diag(sname='DIAM_DRY_'//trim(trname(n)),
     &                lname=trim(trname(n))//' DIAM_DRY',
     &                units='m', power=-2,denom=trim(trname(n)))
#endif
        end select
      end do

C**** 3D tracer-related arrays but not attached to any one tracer

#ifdef TRACERS_SPECIAL_Shindell
        ijlt_OHvmr=
     &    ijlt_diag(sname='OH_vmr',
     &              lname='OH mixing ratio',
     &              units='V/V air', power=-10)
        ijlt_OHconc=
     &    ijlt_diag(sname='OH_conc',
     &              lname='OH concentration',
     &              units='molecules cm-3', power=5)
        ijlt_NO3=
     &    ijlt_diag(sname='NO3_conc',
     &              lname='NO3 concentration',
     &              units='molecules cm-3', power=5)
        ijlt_HO2=
     &    ijlt_diag(sname='HO2_conc',
     &              lname='HO2 concentration',
     &              units='molecules cm-3', power=7)
        ijlt_JO1D=
     &    ijlt_diag(sname='JO1D',
     &              lname='Ox to O1D photolysis rate',
     &              units='s-1')
        ijlt_JNO2=
     &    ijlt_diag(sname='JNO2',
     &              lname='NO2 photolysis rate',
     &              units='s-1')
        ijlt_JH2O2=
     &    ijlt_diag(sname='JH2O2',
     &              lname='H2O2 photolysis rate',
     &              units='s-1', power=2)
        ijlt_O3ppbv=
     &    ijlt_diag(sname='O3_vmr',
     &              lname='O3 not Ox volume mixing ratio',
     &              units='ppbv')
        ijlt_O3cmatm=
     &    ijlt_diag(sname='O3_cm_atm',
     &              lname='O3 not Ox in cm-atm units',
     &              units='cm-atm')
        ijlt_COp=
     &    ijlt_diag(sname='COprod',
     &              lname='CO production rate',
     &              units='mole m-3 s-1')
        ijlt_COd=
     &    ijlt_diag(sname='COdest',
     &              lname='CO destruction rate',
     &              units='mole m-3 s-1')
        ijlt_Oxp=
     &    ijlt_diag(sname='Oxprod',
     &              lname='Ox production rate',
     &              units='mole m-3 s-1')
        ijlt_Oxd=
     &    ijlt_diag(sname='Oxdest',
     &              lname='Ox destruction rate',
     &              units='mole m-3 s-1')
        ijlt_NOxd=
     &    ijlt_diag(sname='NOxdest',
     &              lname='NOx destruction rate',
     &              units='mole m-3 s-1')

        ijlt_CH4d=
     &    ijlt_diag(sname='CH4dest',
     &              lname='CH4 destruction rate',
     &              units='mole m-3 s-1')
        ijlt_OxpHO2=
     &    ijlt_diag(sname='OxpHO2',
     &              lname='Ox prod rate via HO2+NO',
     &              units='mole m-3 s-1')
        ijlt_OxpCH3O2=
     &    ijlt_diag(sname='OxpCH3O2',
     &              lname='Ox prod rate via CH3O2+NO',
     &              units='mole m-3 s-1')
        ijlt_OxpRO2=
     &    ijlt_diag(sname='OxpRO2',
     &              lname='Ox prod rate via RO2+NO',
     &              units='mole m-3 s-1')
        ijlt_OxlOH=
     &    ijlt_diag(sname='OxlOH',
     &              lname='Ox loss rate via OH',
     &              units='mole m-3 s-1')
        ijlt_OxlHO2=
     &    ijlt_diag(sname='OxlHO2',
     &              lname='Ox loss rate via HO2',
     &              units='mole m-3 s-1')
        ijlt_OxlALK=
     &    ijlt_diag(sname='OxlALK',
     &              lname='Ox loss rate via Alkenes',
     &              units='mole m-3 s-1')
        ijlt_pO1D=
     &    ijlt_diag(sname='pO1d',
     &              lname='O1D production from ozone',
     &              units='mole m-3 s-1')
        ijlt_pOH=
     &    ijlt_diag(sname='pOH',
     &              lname='OH production from O1D+H2O',
     &              units='mole m-3 s-1')
        ijlt_NOxLgt=
     &    ijlt_diag(sname='NOx_Lightn',
     &              lname='NOx production from Lightning',
     &              units='kg(N) m-2 s-1', power=-15)
        ijlt_NOvmr=
     &    ijlt_diag(sname='NO_vmr',
     &              lname='NO mixing ratio',
     &              units='V/V air', power=-10) ! to match NOx
        ijlt_NO2vmr=
     &    ijlt_diag(sname='NO2_vmr',
     &              lname='NO2 mixing ratio',
     &              units='V/V air', power=-10) ! to match NOx
      if(save_dQ_for_NINT==1) then
        ijlt_dQ=
     &    ijlt_diag(sname='dQ',
     &              lname='total chem water change',
     &              units='kg/kg air')
        ijlt_dQoh=
     &    ijlt_diag(sname='dQoh',
     &              lname='CH4+OH chem water change',
     &              units='kg/kg air')
        ijlt_dQo1d=
     &    ijlt_diag(sname='dQo1d',
     &              lname='CH4+O1D chem water change',
     &              units='kg/kg air')
        ijlt_dQcl=
     &    ijlt_diag(sname='dQcl',
     &              lname='CH4+Cl chem water change',
     &              units='kg/kg air')
        ijlt_dQsf3=
     &    ijlt_diag(sname='dQsf3',
     &              lname='photolysis related chem water change',
     &              units='kg/kg air')
      end if
#ifdef TRACERS_ACETONE
        ijlt_JacetA=
     &    ijlt_diag(sname='JacetA',
     &              lname='Acetone photolysis rate, Xsection Acet-a',
     &              units='s-1')
        ijlt_JacetB=
     &    ijlt_diag(sname='JacetB',
     &              lname='Acetone photolysis rate, Xsection Acet-b',
     &              units='s-1')
#endif /* TRACERS_ACETONE */
#endif /* TRACERS_SPECIAL_Shindell */

#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) || \
    (defined TRACERS_TOMAS)
        ijlt_prodSO4aq=
     &    ijlt_diag(sname='SO4aqSrc3D',
     &              lname='SO4 aqueous chem source 3D',
     &              units='kg m-2 s-1', power=-15) ! to match ijts 2D
        ijlt_prodSO4gs=
     &    ijlt_diag(sname='SO4gasSrc3D',
     &              lname='SO4 gas phase source 3D',
     &              units='kg m-2 s-1', power=-15) ! to match ijts 2D
#endif /* Koch or AMP or Tomas */

#ifdef TRACERS_NITRATE
        ijlt_aH2O=
     &    ijlt_diag(sname='aerosol_H2O',
     &              lname='aerosol H2O',
     &              units='ug m-3')
        ijlt_apH=
     &    ijlt_diag(sname='aerosol_pH',
     &              lname='aerosol pH',
     &              units=' ')
#endif  /* TRACERS_NITRATE */

#ifdef SOA_DIAGS
        ijlt_soa_changeL_isoprene=
     &    ijlt_diag(sname='SOA_changeL_isoprene',
     &              lname='changeL of isoprene',
     &              units='ug m-3')
        ijlt_soa_changeL_terpenes=
     &    ijlt_diag(sname='SOA_changeL_terpenes',
     &              lname='changeL of terpenes',
     &              units='ug m-3')
        ijlt_soa_voc2nox=
     &    ijlt_diag(sname='SOA_voc2nox',
     &              lname='VOC/NOx ratio',
     &              units='ppbC ppb-1')
        ijlt_soa_pcp=
     &    ijlt_diag(sname='SOA_pcp',
     &              lname='Total non-volatile SOA-absorbing mass',
     &              units='ug m-3')
        ijlt_soa_aerotot=
     &    ijlt_diag(sname='SOA_aerotot',
     &              lname='PCP plus SOA',
     &              units='ug m-3 per MW')
        ijlt_soa_aerotot_gas=
     &    ijlt_diag(sname='SOA_aerotot_gas',
     &              lname='Gas-phase semivolatile potential SOA',
     &              units='ug m-3 per MW')
        ijlt_soa_xmf_isop=
     &    ijlt_diag(sname='SOA_xmf_isop',
     &              lname='Molar fraction of isoprene SOA',
     &              units='fraction')
        ijlt_soa_xmf_apin=
     &    ijlt_diag(sname='SOA_xmf_apin',
     &              lname='Molar fraction of a-pinene SOA',
     &              units='fraction')
        ijlt_soa_zcoef_isop=
     &    ijlt_diag(sname='SOA_zcoef_isop',
     &              lname='Activity coefficient for isoprene SOA',
     &              units=' ')
        ijlt_soa_zcoef_apin=
     &    ijlt_diag(sname='SOA_zcoef_apin',
     &              lname='Activity coefficient for a-pinene SOA',
     &              units=' ')
        ijlt_soa_meanmw=
     &    ijlt_diag(sname='SOA_meanmw',
     &              lname='Mean organic aerosol molecular weight',
     &              units='g mol-1')
        ijlt_soa_iternum=
     &    ijlt_diag(sname='SOA_iternum',
     &              lname='Total iterations for SOA calculations',
     &              units='count')
        ijlt_soa_m0=
     &    ijlt_diag(sname='SOA_M0',
     &              lname='Final M0 value',
     &              units='ug m-3')
      do i=1,nsoa
          ijlt_soa_y0_ug_g(i)=
     &      ijlt_diag(sname='SOA_y0_ug_'//trim(trname(issoa(i)-1)),
     &                lname='y0_ug of '//trim(trname(issoa(i)-1)),
     &                units='ug m-3')
          ijlt_soa_y0_ug_a(i)=
     &      ijlt_diag(sname='SOA_y0_ug_'//trim(trname(issoa(i))),
     &                lname='y0_ug of '//trim(trname(issoa(i))),
     &                units='ug m-3')
          ijlt_soa_y_ug_g(i)=
     &      ijlt_diag(sname='SOA_y_ug_'//trim(trname(issoa(i)-1)),
     &                lname='y_ug of '//trim(trname(issoa(i)-1)),
     &                units='ug m-3')
          ijlt_soa_y_ug_a(i)=
     &      ijlt_diag(sname='SOA_y_ug_'//trim(trname(issoa(i))),
     &                lname='y_ug of '//trim(trname(issoa(i))),
     &                units='ug m-3')
          ijlt_soa_changeL_g_before(i)=
     &      ijlt_diag(sname='SOA_changeL_before_'//
     &                      trim(trname(issoa(i)-1)),
     &                lname='changeL of '//trim(trname(issoa(i)-1))//
     &                      ' before SOA',
     &                units='ug m-3')
          ijlt_soa_changeL_a_before(i)=
     &      ijlt_diag(sname='SOA_changeL_before_'//
     &                      trim(trname(issoa(i))),
     &                lname='changeL of '//trim(trname(issoa(i)))//
     &                      ' before SOA',
     &                units='ug m-3')
          ijlt_soa_changeL_g_after(i)=
     &      ijlt_diag(sname='SOA_changeL_after_'//
     &                      trim(trname(issoa(i)-1)),
     &                lname='changeL of '//trim(trname(issoa(i)-1))//
     &                      ' after SOA',
     &                units='ug m-3')
          ijlt_soa_changeL_a_after(i)=
     &      ijlt_diag(sname='SOA_changeL_after_'//
     &                      trim(trname(issoa(i))),
     &                lname='changeL of '//trim(trname(issoa(i)))//
     &                      ' after SOA',
     &                units='ug m-3')
          ijlt_soa_apartmass(i)=
     &      ijlt_diag(sname='SOA_apartmass_'//trim(trname(issoa(i))),
     &                lname='Effective apartmass of '//
     &                       trim(trname(issoa(i))),
     &                units=' ')
          ijlt_soa_kpart(i)=
     &      ijlt_diag(sname='SOA_kpart_'//trim(trname(issoa(i))),
     &                lname='Partitioning coefficient of '//
     &                      trim(trname(issoa(i))),
     &                units='m3 ug-1')
          ijlt_soa_kp(i)=
     &      ijlt_diag(sname='SOA_kp_'//trim(trname(issoa(i))),
     &                lname='Final partitioning coefficient of '//
     &                      trim(trname(issoa(i))),
     &                units='m3 ug-1')
          ijlt_soa_soamass(i)=
     &      ijlt_diag(sname='SOA_soamass_'//trim(trname(issoa(i))),
     &                lname='Potential SOA mass of '//
     &                      trim(trname(issoa(i))),
     &                units='ug m-3')
          ijlt_soa_partfact(i)=
     &      ijlt_diag(sname='SOA_partfact_'//trim(trname(issoa(i))),
     &                lname='Final partfact value of '//
     &                      trim(trname(issoa(i))),
     &                units=' ')
          ijlt_soa_evap(i)=
     &      ijlt_diag(sname='SOA_evap_'//trim(trname(issoa(i))),
     &                lname='Evaporation of '//trim(trname(issoa(i))),
     &                units='ug m-3')
          ijlt_soa_cond(i)=
     &      ijlt_diag(sname='SOA_cond_'//trim(trname(issoa(i)-1)),
     &                lname='Condensation of pre-existing '//
     &                      trim(trname(issoa(i)-1)),
     &                units='ug m-3')
          ijlt_soa_chem(i)=
     &      ijlt_diag(sname='SOA_chem_'//trim(trname(issoa(i)-1)),
     &                lname='Condensation of same-step produced '//
     &                      trim(trname(issoa(i)-1)),
     &                units='ug m-3')
      enddo
#endif  /* SOA_DIAGS */

#ifdef TRACERS_TOMAS

        ijlt_ccn_01=
     &    ijlt_diag(sname='CCN_01_SS',
     &              lname='CCN 0.1%',
     &              units='cm-3')
        ijlt_ccn_02=
     &    ijlt_diag(sname='CCN_02_SS',
     &              lname='CCN 0.2%',
     &              units='cm-3')
        ijlt_ccn_03=
     &    ijlt_diag(sname='CCN_03_SS',
     &              lname='CCN 0.3%',
     &              units='cm-3')

#endif /* TRACERS_TOMAS */

c
c Append some denominator fields if necessary
c
c nothing is using this as a denominator yet, but some fields _could_.
c      if(any(dname_ijlt(1:k).eq.'airmass')) then
        ijlt_airmass=
     &    ijlt_diag(sname='airmass',
     &              lname='Air Mass',
     &              units='kg/m2/layer')
c      endif

      if(any(dname_ijlt.eq.'clrsky2d')) then
        ijlt_clrsky2d=
     &    ijlt_diag(ia=ia_rad,
     &              sname='clrsky2d',
     &              lname='CLEAR SKY FRACTION',
     &              units='%', power=2)

      endif

c find indices of denominators
! call FindStrings(dname_ijlt,sname_ijlt,denom_ijlt,ktaijls)

#endif /* TRACERS_ON */

      return
      end subroutine init_ijlts_diag

      function get_atmco2()
      USE GHGMOD, only : xnow
      use runtimecontrols_mod, only: constco2
      use dictionary_mod, only: sync_param
      use model_com, only: modelEclock
      use ghgmod, only : CO2_trend
      implicit none
      real*8 :: get_atmco2
      real*8, save :: atmco2=-1.
      integer :: year, dayOfYear
      real*8 :: tnow

      if (constco2) then
        if (atmco2<0.) then    ! uninitialized
          atmco2=280.
          call sync_param("atmCO2", atmco2)
        endif
        if ( atmco2 > 0.d0 ) then
          get_atmco2=atmco2
        else
          call modelEclock%get(year=year, dayOfYear=dayOfYear)
          tnow = year + (dayOfYear-0.999d0)/366.d0
          call CO2_trend(get_atmco2, tnow)
        endif
      else
        get_atmco2=xnow(1)
      endif
      return
      end function get_atmco2

      SUBROUTINE tracer_IC
!@sum tracer_IC initializes tracers when they are first switched on
!@vers 2013/03/27
!@auth Jean Lerner
      USE DOMAIN_DECOMP_ATM, only: AM_I_ROOT,readt_parallel,
     &     readt8_column, skip_parallel
      USE Dictionary_mod, only : get_param, is_set_param
#ifdef TRACERS_ON
      USE FLUXES, only : atmocn,atmice,atmglas,atmlnd,atmsrf,asflx
#ifdef GLINT2
      USE FLUXES, only : atmglas_hp
#endif
      USE CONSTANT, only: mair,rhow,grav,tf,avog,rgas,loschmidt_constant
      use TimeConstants_mod, only: SECONDS_PER_DAY
      USE resolution,ONLY : Im,Jm,Lm,Ls1=>ls1_nominal
      USE ATM_COM, only : q,qcl,qci,pedn,lm_req
      use model_com, only: modelEclock
      USE MODEL_COM, only: itime,dtsrc,itimeI
      USE ATM_COM, only: pmidl00
      USE DOMAIN_DECOMP_ATM, only : GRID,getDomainBounds,write_parallel
      USE SOMTQ_COM, only : qmom,mz,mzz
      use OldTracer_mod, only: trname, itime_tr0, vol2mass, tr_mm
      use OldTracer_mod, only: trsi0, needtrs
      use OldTracer_mod, only: set_itime_tr0
      USE TRACER_COM, only: NTM, trm, trmom, tracers
#ifdef TRACERS_TOMAS
      USE TRACER_COM, only:
     *     n_ASO4,n_AOCOB,n_ASO4,n_ANUM,nbins
      USE TOMAS_AEROSOL, only: sqrt_xk_xk1
#endif
#ifdef TRACERS_WATER
      use OldTracer_mod, only: trw0, tr_wd_type, nWATER
      USE TRACER_COM,only: trwm,n_HDO,n_H2O18, n_OCII
      USE LANDICE, only : ace1li,ace2li
#ifndef TRACERS_ATM_ONLY
      USE LANDICE_COM, only : trsnowli,trlndi
      USE LAKES_COM, only : trlake
      USE GHY_COM, only : tr_w_ij,tr_wsn_ij
#endif
      USE LANDICE_COM, only : snowli
      USE SEAICE_COM, only : si_atm,si_ocn
      USE LAKES_COM, only : mwl,mldlk,flake
      USE GHY_COM, only : w_ij,wsn_ij,nsn_ij,fr_snow_ij,fearth
      USE FLUXES, only : flice,focean
#endif
      USE GEOM, only: axyp,lat2d_dg,lonlat_to_ij,lat2d,lon2d
      USE ATM_COM, only: MA,byMA  ! Air mass of each box (kg m-2)
      USE PBLCOM, only: npbl
#ifdef TRACERS_SPECIAL_Lerner
      USE LINOZ_CHEM_COM, only: tlt0m,tltzm,tltzzm
      USE PRATHER_CHEM_COM, only: nstrtc
#endif
      USE FILEMANAGER, only: openunit,closeunit,nameunit,is_fbsa
#ifdef TRACERS_SPECIAL_Shindell
      use rad_com, only : chem_tracer_save,plb0
      use ghgmod
      use constant, only : byavog
      use tracer_com, only: n_N2O, n_CH4, n_CFC
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
#ifdef TRACERS_dCO
      use tracers_dCO, only: dalke_IC_fact
      use tracers_dCO, only: dPAR_IC_fact
      use tracers_dCO, only: dPAN_IC_fact
      use tracers_dCO, only: dMeOOH_IC_fact
      use tracers_dCO, only: dHCHO_IC_fact
#endif  /* TRACERS_dCO */
      use tracers_dCO, only: dC17O_IC_fact
      use tracers_dCO, only: dC18O_IC_fact
      use tracers_dCO, only: d13CO_IC_fact
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
      USE TRCHEM_Shindell_COM,only: ch4icx,
     &  OxIC,COIC,byO3MULT,fix_CH4_chemistry,
     &  ICfact_N,ICfact_COt,ICfact_COs,ICfact_Oth
     &  ,use_rad_n2o,use_rad_cfc,use_rad_ch4
     &  ,ClOxalt,BrOxalt,ClONO2alt,HClalt,N2OICX,CFCIC
     &  ,ICfact_N2O,ICfact_CFC,fact_cfc
#ifdef INTERACTIVE_WETLANDS_CH4
      USE TRACER_SOURCES, only:first_mod,first_ncep,avg_model,avg_ncep,
     & PRS_ch4,sum_ncep
#endif
      USE TRACER_SOURCES, only:GLTic
#endif /* TRACERS_SPECIAL_Shindell */
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS)
      use OldTracer_mod, only: om2oc
      USE AEROSOL_SOURCES, only: DMSinput
#ifndef TRACERS_AEROSOLS_SOA
      USE AEROSOL_SOURCES, only: OCT_src
#endif  /* TRACERS_AEROSOLS_SOA */
      USE AEROSOL_SOURCES, only: SO2_src_3D,iso2volcano
      use geom, only : byaxyp
#endif
#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS) ||\
    (defined TRACERS_AMP)  || (defined TRACERS_TOMAS)
      USE trdust_mod,ONLY : hbaij,ricntd
      use trdust_drv, only: tracer_ic_soildust
#endif
      use OldTracer_mod, only: trli0
#ifdef TRACERS_AMP
      use TRACER_COM, only:n_M_OCC_OC
#else
      use TRACER_COM, only:n_OCII
#endif
#endif /* TRACERS_ON */
      use oldtracer_mod, only: src_dist_base, src_dist_index
      use tracer_com, only: xyztr
      use pario, only : par_open,par_close,read_dist_data
      use filemanager, only: file_exists
      IMPLICIT NONE
      real*8,parameter :: d18oT_slope=0.45,tracerT0=25
      INTEGER i,n,l,j,iu_data,ipbl,it,lr,m,ls,lt,ipatch
      CHARACTER*80 title
      CHARACTER*300 out_line
      REAL*8 CFC11ic,conv
      REAL*8 :: trinit =1., tmominit=0.
      real*8 tracerTs

#ifdef TRACERS_ON
      REAL*8, DIMENSION(GRID%I_STRT_HALO:GRID%I_STOP_HALO,
     &                  GRID%J_STRT_HALO:GRID%J_STOP_HALO,lm) ::
     *                                CO2ic
      REAL*8, DIMENSION(GRID%I_STRT:GRID%I_STOP,
     &                  GRID%J_STRT:GRID%J_STOP,lm) ::
     *                                ic14CO2
      REAL*4, DIMENSION(jm,lm)    ::  N2Oic   !each proc. reads global array
      REAL*8, DIMENSION(GRID%J_STRT_HALO:GRID%J_STOP_HALO,lm) ::
     *                                                      CH4ic
#ifdef TRACERS_SPECIAL_Lerner
      REAL*8, DIMENSION(GRID%I_STRT_HALO:GRID%I_STOP_HALO,
     &                  GRID%J_STRT_HALO:GRID%J_STOP_HALO) :: icCFC
      REAL*8 stratm,xlat,pdn,pup
#endif
      real*8 :: get_atmco2

!@param bymair 1/molecular wt. of air = 1/mair
!@param byjm 1./JM
      REAL*8, PARAMETER :: bymair = 1.d0/mair, byjm =1.d0/JM
#ifdef TRACERS_SPECIAL_Shindell
!@var ghgCmAtm array of same shape as rad code/ghgmod ulgas for
!@+ returning gas amounts by layer in cm-atm units from getgas calls
!@var ghgplb bottom layer pressures like used in rad code/ghgmod
      real*8, dimension(lxghg,13) :: ghgCmAtm
      real*8, dimension(lxghg+1) :: ghgplb
!@var CMATMtoKG for converting cm-atm units e.g. from rad code/ghgmod
!@+   to KG e.g. trm()
      real*8 :: CMATMtoKG
!@var jlat46 lat index relative to the rad code 72x46 grid
!@var ilon72 lon index relative to the rad code 72x46 grid
      integer :: jlat46,ilon72
!@var imonth dummy index for choosing the right month
!@var ICfactor varying factor for altering initial conditions
!@var dICfactor varying factor for altering initial conditions of dCO tracers
      INTEGER imonth, J2
      REAL*8 ICfactor,dICfactor
!@var PRES local nominal pressure for vertical interpolations
      REAL*8, DIMENSION(LM) :: PRES
#endif
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP)  ||\
    (defined TRACERS_TOMAS) || (defined TRACERS_AEROSOLS_SEASALT)
      include 'netcdf.inc'
#endif
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP)  ||\
    (defined TRACERS_TOMAS) || (defined TRACERS_AEROSOLS_SEASALT)
      integer start(3),count(3),status,ncidu,id1
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP)  ||\
    (defined TRACERS_TOMAS)
      INTEGER ii,jj,ir,mm,iuc,mmm,ll,iudms
      INTEGER iuc2,lmax
#endif
#endif
#if defined (TRACERS_AEROSOLS_Koch) || defined (TRACERS_AMP) ||\
    defined (TRACERS_TOMAS)
      ! 1-deg volc. emiss
      real*8 :: volc_lons(360),volc_lats(180),
     &     volc_pup(360,180),volc_emiss(360,180)
      real*8 :: x1d(lm),amref(lm),pednref(lm+1),amsum
      real*8, allocatable, dimension(:,:) :: psref
      integer :: iu_ps,file_id,vid,ilon,jlat,volc_ij(2)
#endif
#ifdef TRACERS_TOMAS
      integer k
#endif
      INTEGER J_0, J_1, I_0, I_1
      LOGICAL HAVE_SOUTH_POLE, HAVE_NORTH_POLE
      integer :: lat_val
#endif /* TRACERS_ON */
      character(len=:), allocatable :: name

#ifdef TRACERS_ON
C****
C**** Extract useful local domain parameters from "grid"
C****
      call getDomainBounds(grid, J_STRT=J_0,       J_STOP=J_1,
     *               HAVE_SOUTH_POLE=HAVE_SOUTH_POLE,
     *               HAVE_NORTH_POLE=HAVE_NORTH_POLE)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP


#ifdef TRACERS_SPECIAL_Shindell
      PRES(1:LM)=PMIDL00(1:LM)
#endif
      do n=1,ntm
      if (itime.eq.itime_tr0(n)) then

C**** set some defaults for air mass tracers
      trm(:,J_0:J_1,:,n) = 0.
      trmom(:,:,J_0:J_1,:,n) = 0.

#ifdef TRACERS_WATER
C**** set some defaults for water tracers
      trwm(:,J_0:J_1,:,n)=0. ! cloud liquid water
#ifndef TRACERS_ATM_ONLY
      trlake(n,:,:,J_0:J_1)=0.
      si_atm%trsi(n,:,:,J_0:J_1)=0.
#endif
      if(si_ocn%grid%im_world .ne. im) then
        call stop_model(
     &       'TRACER_IC: tracers in sea ice are no longer on the '//
     &       'atm. grid - please move the si_ocn references',255)
      endif
#ifndef TRACERS_ATM_ONLY
      si_ocn%trsi(n,:,:,J_0:J_1)=0.
      trlndi(n,:,J_0:J_1,:)=0.
      trsnowli(n,:,J_0:J_1,:)=0.
      tr_w_ij(n,:,:,:,J_0:J_1)=0.
      tr_wsn_ij(n,:,:,:,J_0:J_1)=0.
#endif
#endif
      name=trname(n)
      if (src_dist_index(n)/=0) name=trname(src_dist_base(n))
      select case (name)

        case default
          write(6,*) 'In TRACER_IC:',name,' does not exist '
          call stop_model("TRACER_IC",255)

        case ('Air')
          do l=1,lm
          do j=J_0,J_1
            trm(:,j,l,n) = MA(l,:,j)
          end do; enddo

        case ('SF6','SF6_c','nh5','nh15','nh50','e90','CO50',
     &        'aoa','aoanh','st8025','tape_rec')
         ! defaults ok

        case ('Be7', 'Be10', 'Pb210', 'Rn222')
          ! defaults ok

        case ('CO2')
          call openunit('CO2_IC',iu_data,.true.,.true.)
          CALL READT_PARALLEL(grid,iu_data,NAMEUNIT(iu_data),CO2IC,0)
          call closeunit(iu_data)
          do l=1,lm         !ppmv==>ppmm
          do j=J_0,J_1
            trm(:,j,l,n) = co2ic(:,j,l)*MA(l,:,j)*1.54d-6
          enddo; enddo

        case ('N2O')
#ifdef TRACERS_SPECIAL_Lerner
          call openunit('N2O_IC',iu_data,.true.,.true.)
C**** ESMF: Each processor reads the global array: N2Oic
          read (iu_data) title,N2Oic     ! unit is PPMM/(M*AXYP)
          call closeunit(iu_data)
          if (AM_I_ROOT()) write(6,*) title,' read from N2O_IC'
          do l=1,lm         !ppmv==>ppmm
          do j=J_0,J_1
            trm(:,j,l,n) = MA(l,:,j)*N2Oic(j,l)
          enddo; enddo
#endif


#ifdef TRACERS_SPECIAL_Shindell
         if(use_rad_n2o <= 0)then
           ! N2O initial conditions from input file:
           do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
             trm(i,j,l,n) = N2OICX(i,j,l)*ICfact_N2O
           end do   ; end do   ; end do
         else
           ! N2O initial conditions from GHGMOD (rad code):
           ghgplb(LM+1+1:LM+1+lm_req)=plb0(1:lm_req)
           do j=J_0,J_1
             do i=I_0,I_1
               ghgplb(1:LM+1)=pedn(1:LM+1,i,j)
               call get_72x46ij(lon2d(i,j),lat2d(i,j),ilon72,jlat46)
               call getgas(i,j,jlat46,ghgplb,ghgCmAtm)
               ! Units of 1.d1 factor below are: 1d3(mole/Kmole)*1d-4(m-2/cm-2)
               CMATMtoKG=1.d1*loschmidt_constant*byavog*tr_mm(n_N2O)
               trm(i,j,1:LM,n) = ghgCmAtm(1:LM,6) * CMATMtoKG
             end do
           end do
         end if
#endif

        case ('CFC11')   !!! should start April 1
          CFC11ic = 268.D-12*136.5/29.029    !268 PPTV
          do l=1,lm
          do j=J_0,J_1
            trm(:,j,l,n) = MA(l,:,j)*CFC11ic
          enddo; enddo
#ifdef TRACERS_SPECIAL_Lerner
C****
C**** Read in first layer distribution; This is used up to about 100 mb
C****
      call openunit('CFCic_Lerner',iu_data,.true.,.true.)
      CALL READT_PARALLEL(grid,iu_data,NAMEUNIT(iu_data),icCFC,0)
      call closeunit(iu_data)
C**** Fill in the tracer; above 100 mb interpolate linearly with P to 0 at top
      stratm = 101.9368
      DO J=J_0,J_1
      DO I=I_0,I_1
        PUP = STRATM*GRAV
        DO LS=LM,1,-1
          PDN = PUP + MA(ls,I,J)*GRAV
          IF(PDN.GT.10000.d0)  GO TO 450
          trm(I,J,LS,N) =
     *      MA(ls,I,J)*icCFC(i,j)*.5*(PUP+PDN)/10000.d0
          PUP = PDN
        enddo
  450   CONTINUE
        trm(I,J,LS,N) = MA(ls,I,J)*icCFC(i,j)*
     *    (1.-.5*(10000.-PUP)*(10000.-PUP)/(10000.*(PDN-PUP)))
        DO LT=1,LS-1
          trm(I,J,LT,N) = MA(lt,I,J)*icCFC(i,j)
        enddo
      enddo; enddo
#endif


        case ('14CO2')   !!! this tracer is supposed to start 10/16
#ifdef TRACERS_SPECIAL_Lerner
          call get_14CO2_IC(ic14CO2)
          do l=1,lm         !ppmv==>ppmm
          do j=J_0,J_1
            trm(:,j,l,n) = MA(l,:,j)*ic14CO2(:,j,l)*1.d-18
          enddo; enddo
#endif

        case ('CH4')
#ifdef TRACERS_SPECIAL_Shindell
         if(use_rad_ch4 <= 0)then
           ! CH4 initial conditions from file
           select case (fix_CH4_chemistry)
           case default
             call get_CH4_IC(0) ! defines trm(:,:,:,n_CH4) within
           case(-1) ! ICs from file...
             call get_CH4_IC(0) ! defines trm(:,:,:,n_CH4) within
             do l=ls1,lm; do j=J_0,J_1; do i=I_0,I_1
               trm(I,J,L,n) = CH4ICX(I,J,L)
             end do   ; end do   ; end do
           end select
#ifdef INTERACTIVE_WETLANDS_CH4
           first_mod(:,:,:)=1
           first_ncep(:)=1
           avg_model(:,:,:)=0.d0
           avg_ncep(:,:,:)=0.d0
           PRS_ch4(:,:,:)=0.d0
           sum_ncep(:,:,:)=0.d0
#endif
         else
           ! CH4 initial conditions from GHGMOD (rad code):
           ghgplb(LM+1+1:LM+1+lm_req)=plb0(1:lm_req)
           do j=J_0,J_1
             do i=I_0,I_1
               ghgplb(1:LM+1)=pedn(1:LM+1,i,j)
               call get_72x46ij(lon2d(i,j),lat2d(i,j),ilon72,jlat46)
               call getgas(i,j,jlat46,ghgplb,ghgCmAtm)
               ! Units of 1.d1 factor below are: 1d3(mole/Kmole)*1d-4(m-2/cm-2)
               CMATMtoKG=1.d1*loschmidt_constant*byavog*tr_mm(n_CH4)
               trm(i,j,1:LM,n) = ghgCmAtm(1:LM,7) * CMATMtoKG /
     &                                              CH4X_RADoverCHEM
             end do
           end do
         end if
         ! should be able to remove this next block once radiation comes
         ! after tracer 3d source (chemistry) in sequence:
         do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
           ! Units of 1.d1 factor below are: 1d3(mole/Kmole)*1d-4(m-2/cm-2)
           chem_tracer_save(2,L,I,J)=trm(I,J,L,n)
     &          *avog/(tr_mm(n)*1.d1*loschmidt_constant) ! to atm*cm
         end do   ; end do   ; end do
#endif /* TRACERS_SPECIAL_Shindell */
#ifdef TRACERS_SPECIAL_Lerner
          call get_wofsy_gas_IC(name,CH4ic)
          do l=1,lm         !ppbv==>ppbm
          do j=J_0,J_1
            trm(:,j,l,n) = MA(l,:,j)*CH4ic(j,l)*0.552d-9
          enddo; enddo
#endif

        case ('O3')
          do l=1,lm
          do j=J_0,J_1
            trm(:,j,l,n) = MA(l,:,j)*20.d-9*vol2mass(n)
          enddo; enddo
#ifdef TRACERS_SPECIAL_Lerner
          do l=lm,lm+1-nstrtc,-1
          lr = lm+1-l
            do j=J_0,J_1
            do i=I_0,I_1
            if (tlt0m(i,j,lr,5) /= 0.) then
            trm(i,j,l,n) =
     *          tlt0m(i,j,lr,1)*MA(l,i,j)*vol2mass(n)
            trmom(mz,i,j,l,n)  =
     *          tltzm(i,j,lr,1)*MA(l,i,j)*vol2mass(n)
            trmom(mzz,i,j,l,n)  =
     *         tltzzm(i,j,lr,1)*MA(l,i,j)*vol2mass(n)
            end if
            end do
            end do
          end do
#endif

#ifdef TRACERS_WATER
      case ('Water', 'H2O18', 'HDO', 'HTO', 'H2O17')

C**** initial atmospheric conc. needs to be defined for each tracer
        select case (name)
        case ('Water')
          trinit=1.
C**** for gradients defined on air mass
          tmominit = 1.
C**** for gradients defined on water mass (should be an option?)
c     tmominit = 0.
        case ('H2O18')        ! d18O=-80
          trinit=0.92d0*trw0(n)
          tmominit = trinit
        case ('H2O17')        ! d18O=-43.15  (D17O=0)
          trinit=.95685d0*trw0(n)
          tmominit = trinit
        case ('HDO')   ! dD=-630
          trinit=0.37d0*trw0(n)
          tmominit = trinit
        case ('HTO')
          trinit=0.
          tmominit = trinit
        end select

        do l=1,lm
        do j=J_0,J_1
          do i=I_0,I_1
            trm(i,j,l,n) =  q(i,j,l)*MA(l,i,j)*trinit
            trwm(i,j,l,n)= (qcl(i,j,l)+qci(i,j,l))*MA(l,i,j)*trinit
            trmom(:,i,j,l,n) = qmom(:,i,j,l)*MA(l,i,j)*tmominit
            if (src_dist_index(n)/=0) then
              trm(i, j, l, n)=trm(i, j, l, n)*
     &                                 xyztr(src_dist_index(n), i, j)
              trwm(i, j, l, n)=trwm(i, j, l, n)*
     &                                 xyztr(src_dist_index(n), i, j)
              trmom(:, i, j, l, n)=trmom(:, i, j, l, n)*
     &                                 xyztr(src_dist_index(n), i, j)
            endif
          end do
        end do
        end do
        if (HAVE_SOUTH_POLE) then
           do i=2,im
              trm(i,1,:,n) =  trm(1,1,:,n) !poles
              trwm(i, 1,:,n)= trwm(1, 1,:,n) !poles
              trmom(:,i, 1,:,n)=0.
           enddo
        endif
        if (HAVE_NORTH_POLE) then
           do i=2,im
              trm(i,jm,:,n) = trm(1,jm,:,n) !poles
              trwm(i,jm,:,n)= trwm(1,jm,:,n) !poles
              trmom(:,i,jm,:,n)=0.
          enddo
        endif
        if (name.eq."HTO") then ! initialise bomb source
          do l=ls1-1,ls1+1      ! strat. source lat 44 N - 56 N
          do j=J_0,J_1
          do i=I_0,I_1
            if(nint(lat2d_dg(i,j)).ge.44.and.nint(lat2d_dg(i,j)).le.56)
     *           trm(i,j,l,n)= q(i,j,l)*MA(l,i,j)*1d10*1d-18
          end do
          end do
          end do
        end if

#ifndef TRACERS_ATM_ONLY
        call init_single_seaice_tracer(si_atm,n,trsi0(n))
        call init_single_seaice_tracer(si_ocn,n,trsi0(n))

        do j=J_0,J_1
          do i=I_0,I_1
            tracerTs=trw0(n)
#ifdef TRACERS_SPECIAL_O18
c Define a simple d18O based on Tsurf for GIC, put dD on meteoric water line
            if(name.eq."H2O18") tracerTs=TRW0(n_H2O18)*(1.+1d-3*
     *           ((atmsrf%tsavg(i,j)-(tf+tracerT0))*d18oT_slope))
            if(name.eq."HDO") tracerTs=TRW0(n_HDO)*(1.+(1d-3*
     *         (((atmsrf%tsavg(i,j)-(tf+tracerT0))*d18oT_slope)*8+1d1)))
#endif
C**** lakes
            if (flake(i,j).gt.0) then
              trlake(n,1,i,j)=tracerTs*mldlk(i,j)*rhow*flake(i,j)
     *             *axyp(i,j)
              if (mwl(i,j)-mldlk(i,j)*rhow*flake(i,j)*axyp(i,j).gt.1d-10
     *             *mwl(i,j)) then
                trlake(n,2,i,j)=tracerTs*mwl(i,j)-trlake(n,1,i,j)
              else
                trlake(n,2,i,j)=0.
              end if
              atmocn%gtracer(n,i,j)=trw0(n)
            else !if (focean(i,j).eq.0) then
              trlake(n,1,i,j)=trw0(n)*mwl(i,j)
              trlake(n,2,i,j)=0.
c            else
c              trlake(n,1:2,i,j)=0.
            end if
c**** ice
            if (si_atm%msi(i,j).gt.0) then
              atmice%gtracer(n,i,j)=trsi0(n)
            end if
c**** landice
            if (flice(i,j).gt.0) then
              trlndi(n,i,j,:)=trli0(n)*(ace1li+ace2li)	! calls trli0_s()
              trsnowli(n,i,j,:)=trli0(n)*snowli(i,j,:)
              do ipatch=1,ubound(atmglas,1)
#ifdef GLINT2
                atmglas_hp(ipatch)%gtracer(n,i,j)=trli0(n)
#endif
                atmglas(ipatch)%gtracer(n,i,j)=trli0(n)
              enddo
            else
              trlndi(n,i,j,:)=0.
              trsnowli(n,i,j,:)=0.
              do ipatch=1,ubound(atmglas,1)
#ifdef GLINT2
                atmglas_hp(ipatch)%gtracer(n,i,j)=0.
#endif
                atmglas(ipatch)%gtracer(n,i,j)=0.
              enddo
            end if
c**** earth
            !!!if (fearth(i,j).gt.0) then
            if (focean(i,j) < 1.d0) then
              conv=rhow         ! convert from m to kg m-2
              tr_w_ij  (n,:,:,i,j)=tracerTs*w_ij (:,:,i,j)*conv
              tr_wsn_ij(n,1:nsn_ij(1,i,j),1,i,j)=
     &             tracerTs*wsn_ij(1:nsn_ij(1,i,j),1,i,j)
     &             *fr_snow_ij(1,i,j)*conv
              tr_wsn_ij(n,1:nsn_ij(2,i,j),2,i,j)=
     &             tracerTs*wsn_ij(1:nsn_ij(2,i,j),2,i,j)
     &             *fr_snow_ij(2,i,j)*conv
              !trsnowbv(n,2,i,j)=trw0(n)*snowbv(2,i,j)*conv
              atmlnd%gtracer (n,i,j)=trw0(n)
            else
              tr_w_ij  (n,:,:,i,j)=0.
              tr_wsn_ij(n,:,:,i,j)=0.
              !trsnowbv(n,1,i,j)=0.
              !trsnowbv(n,2,i,j)=0.
              atmlnd%gtracer(n,i,j)=0.
            end if
          end do
          end do
#endif
#ifdef TRACERS_SPECIAL_O18
          if (AM_I_ROOT()) then
            if(name.eq."H2O18") write(6,'(A52,f6.2,A15,f8.4,A18)')
     *            "Initialized trlake tr_w_ij tr_wsn_ij using Tsurf at"
     *           ,tracerT0,"degC, 0 permil",d18oT_slope
     *           ,"permil d18O/degC"
          endif
#endif

#endif /* TRACERS_WATER */

#ifdef TRACERS_SPECIAL_Shindell
        case ('Ox')
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
            trm(I,J,L,n) = OxIC(I,J,L)
            ! should be able to remove next line once rad code is after tr3dsource code:
            chem_tracer_save(1,L,I,J)=OxIC(I,J,L)*byO3MULT
          end do   ; end do   ; end do

        case ('NOx')
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
            trm(i,j,l,n) = MA(l,i,j)*1.d-11*ICfact_N
            if(PRES(L).lt.10.)trm(i,j,l,n)=trm(i,j,l,n)*3.d2
          end do; end do; end do

        case ('ClOx')
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
            trm(i,j,l,n) =
     &      MA(l,i,j)*vol2mass(n)*1.d-11*ClOxalt(l)
          end do; end do; end do

        case ('BrOx')
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
            trm(i,j,l,n) =
     &      MA(l,i,j)*vol2mass(n)*1.d-11*BrOxalt(l)
          end do; end do; end do

        case ('HCl')
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
            trm(i,j,l,n) =
     &      MA(l,i,j)*vol2mass(n)*1.d-11*HClalt(l)
          end do; end do; end do

        case ('ClONO2')
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
            trm(i,j,l,n) =
     &      MA(l,i,j)*vol2mass(n)*1.d-11*ClONO2alt(l)
          end do; end do; end do

        case ('N2O5')
          do l=1,lm; do j=J_0,J_1; do i=i_0,i_1
            trm(i,j,l,n) = MA(l,i,j)*1.d-12*ICfact_N
          end do; end do; end do

        case ('HNO3')
          do l=1,lm; do j=J_0,J_1; do i=i_0,i_1
            trm(i,j,l,n) = MA(l,i,j)*1.d-10*ICfact_N
            if(PRES(L).lt.50.and.PRES(L).gt.10.)
     &      trm(i,j,l,n)=trm(i,j,l,n)*1.d2
          end do; end do; end do
#endif /* TRACERS_SPECIAL_Shindell */

        case ('H2O2')
          do l=1,lm; do j=J_0,J_1; do i=i_0,i_1
            trm(i,j,l,n) = MA(l,i,j)*5.d-10
          end do; end do; end do

#ifdef TRACERS_SPECIAL_Shindell
        case ('GLT')
          do l=1,lm; do j=J_0,J_1; do i=i_0,i_1
            trm(i,j,l,n) = GLTic*vol2mass(n)*MA(l,i,j)
          end do; end do; end do

        case ('CH3OOH',
#ifdef TRACERS_dCO
     *        'dMe17OOH', 'dMe18OOH', 'd13MeOOH',
     *        'dHCH17O', 'dHCH18O', 'dH13CHO',
#endif  /* TRACERS_dCO */
     &        'HCHO')
          select case (trname(n))
#ifdef TRACERS_dCO
            case ('dMe17OOH', 'dMe18OOH', 'd13MeOOH')
              dICfactor=dMeOOH_IC_fact
            case ('dHCH17O', 'dHCH18O', 'dH13CHO')
              dICfactor=dHCHO_IC_fact
#endif  /* TRACERS_dCO */
            case default
              dICfactor=1.d0
          end select
          do l=1,lm; do j=J_0,J_1; do i=i_0,i_1
            trm(i,j,l,n) = MA(l,i,j)*1.d-11*dICfactor
          end do; end do; end do

        case ('HO2NO2')
          do l=1,lm; do j=J_0,J_1; do i=i_0,i_1
            trm(i,j,l,n) = MA(l,i,j)*1.d-12*ICfact_N
          end do; end do; end do

        case ('CO'
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
     *       ,'dC17O','dC18O','d13CO'
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
     *       )
          select case (trname(n))
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
            case ('dC17O')
              dICfactor=dC17O_IC_fact
            case ('dC18O')
              dICfactor=dC18O_IC_fact
            case ('d13CO')
              dICfactor=d13CO_IC_fact
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
            case default
              dICfactor=1.d0
          end select
          do l=1,lm
            if(L.le.LS1-1) then
              ICfactor=ICfact_COt ! troposphere
            else
              ICfactor=ICfact_COs ! stratosphere
            end if
            do j=J_0,J_1; do i=I_0,I_1
              trm(I,J,L,n) = COIC(I,J,L)*ICfactor*dICfactor
            end do   ; end do
          end do

        case ('PAN'
#ifdef TRACERS_dCO
     *       ,'d17OPAN','d18OPAN','d13CPAN'
#endif  /* TRACERS_dCO */
     *       )
          select case (trname(n))
#ifdef TRACERS_dCO
            case ('d17OPAN','d18OPAN','d13CPAN')
              dICfactor=dPAN_IC_fact
#endif  /* TRACERS_dCO */
            case default
              dICfactor=1.d0
          end select
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
            trm(i,j,l,n) = MA(l,i,j)*vol2mass(n)*4.d-11*
     &                     ICfact_Oth*dICfactor
          end do; end do; end do

        case ('Isoprene')
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
            trm(i,j,l,n) =
     &      MA(l,i,j)*vol2mass(n)*0.d-11*ICfact_Oth
          end do; end do; end do

        case ('AlkylNit')
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
            trm(i,j,l,n) =
     &      MA(l,i,j)*vol2mass(n)*2.d-10*ICfact_Oth
          end do; end do; end do

        case('Alkenes'
#ifdef TRACERS_dCO
     *      ,'d13Calke'
#endif  /* TRACERS_dCO */
     *       )
          select case (trname(n))
#ifdef TRACERS_dCO
            case ('d13Calke')
              dICfactor=dalke_IC_fact
#endif  /* TRACERS_dCO */
            case default
              dICfactor=1.d0
          end select
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
            trm(i,j,l,n) = MA(l,i,j)*vol2mass(n)*4.d-10*
     &                     ICfact_Oth*dICfactor
          end do; end do; end do

        case('Paraffin'
#ifdef TRACERS_dCO
     *      ,'d13CPAR'
#endif  /* TRACERS_dCO */
     *       )
          select case (trname(n))
#ifdef TRACERS_dCO
            case ('d13CPAR')
              dICfactor=dPAR_IC_fact
#endif  /* TRACERS_dCO */
            case default
              dICfactor=1.d0
          end select
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
            trm(i,j,l,n) = MA(l,i,j)*vol2mass(n)*5.d-10*
     &                     ICfact_Oth*dICfactor
          end do; end do; end do

        case('Terpenes','Acetone'
#ifdef TRACERS_AEROSOLS_SOA
     &      ,'isopp1g','isopp1a','isopp2g','isopp2a'
     &      ,'apinp1g','apinp1a','apinp2g','apinp2a'
#endif
#ifdef TRACERS_AEROSOLS_VBS
     *      ,'vbsGm2', 'vbsGm1', 'vbsGz',  'vbsGp1', 'vbsGp2'
     *      ,'vbsGp3', 'vbsGp4', 'vbsGp5', 'vbsGp6'
     *      ,'vbsAm2', 'vbsAm1', 'vbsAz',  'vbsAp1', 'vbsAp2'
     *      ,'vbsAp3', 'vbsAp4', 'vbsAp5', 'vbsAp6'
#endif  /* TRACERS_AEROSOLS_VBS */
#ifdef TRACERS_AEROSOLS_OCEAN
     &      ,'OCocean'
#endif
     &      )
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
            trm(i,j,l,n) =
     &      MA(l,i,j)*vol2mass(n)*0.d0*5.d-14*ICfact_Oth
          end do; end do; end do
#endif /* TRACERS_SPECIAL_Shindell */

        case ('CO2n')
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
             !units: [am]=kg_air/m2, [axyp]=m2, [tr_mm]=kg_CO2,
             !       [bymair]=1/kg_air, [atmCO2]=ppmv=10^(-6)kg_CO2/kg_air
             !       [vol2mass]=(gr,CO2/moleCO2)/(gr,air/mole air)
             trm(i,j,l,n) = MA(l,i,j)*vol2mass(n)
     .                    * get_atmCO2()*1.d-6
             atmocn%gtracer(n,i,j) = vol2mass(n)
     .                    * get_atmCO2() * 1.d-6      !initialize gtracer
          end do; end do; end do
        case ('CFCn')
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
            if(l.ge.LS1) then
              trm(i,j,l,n) = MA(l,i,j)*vol2mass(n)*2.d-13
            else
              trm(i,j,l,n) = MA(l,i,j)*vol2mass(n)*1.d-13
            end if
          end do; end do; end do

#ifdef TRACERS_SPECIAL_Shindell
        case ('CFC')
          if(use_rad_cfc.le.0)then
            ! CFC initial conditions from input file:
            do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
              trm(I,J,L,n) = CFCIC(I,J,L)*ICfact_CFC
            end do   ; end do   ; end do
          else
            ! CFC initial conditions from GHGMOD (rad code):
            ghgplb(LM+1+1:LM+1+lm_req)=plb0(1:lm_req)
            do j=J_0,J_1
              do i=I_0,I_1
                ghgplb(1:LM+1)=pedn(1:LM+1,i,j)
                call get_72x46ij(lon2d(i,j),lat2d(i,j),ilon72,jlat46)
                call getgas(i,j,jlat46,ghgplb,ghgCmAtm)
                ! Units of 1.d1 factor below are: 1d3(mole/Kmole)*1d-4(m-2/cm-2)
                CMATMtoKG=1.d1*loschmidt_constant*byavog*tr_mm(n_CFC)
                trm(i,j,1:LM,n) = (ghgCmAtm(1:LM,8) + ghgCmAtm(1:LM,9))
     &                            * CMATMtoKG * fact_cfc
              end do
            end do
          end if
#endif /* TRACERS_SPECIAL_Shindell */

        case ('BrONO2','HBr','HOBr')
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
            if(l.ge.LS1) then
              trm(i,j,l,n) = MA(l,i,j)*vol2mass(n)*2.d-13
            else
              trm(i,j,l,n) = MA(l,i,j)*vol2mass(n)*1.d-13
            end if
          end do; end do; end do

        case ('HOCl')
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
            if(l.ge.LS1) then
              trm(i,j,l,n) = MA(l,i,j)*vol2mass(n)*5.d-11
            else
              trm(i,j,l,n) = MA(l,i,j)*vol2mass(n)*1.d-11
            end if
          end do; end do; end do

        case('DMS')
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
            trm(i,j,l,n) = MA(l,i,j)*vol2mass(n)*5.d-13
          end do; end do; end do

#ifndef TRACERS_TOMAS
        case('MSA', 'SO2', 'SO4', 'SO4_d1', 'SO4_d2', 'SO4_d3',
     *         'N_d1','N_d2','N_d3','NH3','NH4','NO3p',
     *         'BCII', 'BCIA', 'BCB', 'OCII', 'OCIA', 'OCB', 'H2O2_s',
     *         'seasalt1', 'seasalt2',
     *         'M_NO3   ','M_NH4   ','M_H2O   ','N_AKK_1 ',
     *         'N_ACC_1 ','M_DD1_SU','N_DD1_1 ',
     *         'M_DS1_SU','M_DS1_DU','N_DS1_1 ','M_DD2_SU','M_DD2_DU',
     *         'N_DD2_1 ','M_DS2_SU','M_DS2_DU','N_DS2_1 ','M_SSA_SU',
     *         'M_OCC_SU','N_OCC_1 ','M_BC1_SU','N_SSA_1 ','N_SSC_1 ',
     *         'N_BC1_1 ','M_BC2_SU','M_BC2_BC','N_BC2_1 ','M_BC3_SU',
     *         'M_BC3_BC','N_BC3_1 ','M_DBC_SU','M_DBC_BC','M_DBC_DU',
     *         'N_DBC_1 ','M_BOC_SU','M_BOC_BC','M_BOC_OC','N_BOC_1 ',
     *         'M_BCS_SU','M_BCS_BC','N_BCS_1 ','M_MXX_SU','M_MXX_BC',
     *         'M_MXX_OC','M_MXX_DU','M_MXX_SS','N_MXX_1 ','M_OCS_SU',
     *         'M_OCS_OC','N_OCS_1 ','H2SO4',
     *         'M_AKK_SU','M_ACC_SU','M_DD1_DU',
     *         'M_SSA_SS','M_SSC_SS','M_BC1_BC','M_OCC_OC',
     *         'M_SSS_SS','M_SSS_SU',
     * 'M_ACC_OCM2','M_ACC_OCM1','M_ACC_OCM0','M_ACC_OCP1','M_ACC_OCP2',
     * 'M_ACC_OCP3','M_ACC_OCP4','M_ACC_OCP5','M_ACC_OCP6',
     * 'M_DD1_OCM2','M_DD1_OCM1','M_DD1_OCM0','M_DD1_OCP1','M_DD1_OCP2',
     * 'M_DD1_OCP3','M_DD1_OCP4','M_DD1_OCP5','M_DD1_OCP6',
     * 'M_DS1_OCM2','M_DS1_OCM1','M_DS1_OCM0','M_DS1_OCP1','M_DS1_OCP2',
     * 'M_DS1_OCP3','M_DS1_OCP4','M_DS1_OCP5','M_DS1_OCP6',
     * 'M_DD2_OCM2','M_DD2_OCM1','M_DD2_OCM0','M_DD2_OCP1','M_DD2_OCP2',
     * 'M_DD2_OCP3','M_DD2_OCP4','M_DD2_OCP5','M_DD2_OCP6',
     * 'M_DS2_OCM2','M_DS2_OCM1','M_DS2_OCM0','M_DS2_OCP1','M_DS2_OCP2',
     * 'M_DS2_OCP3','M_DS2_OCP4','M_DS2_OCP5','M_DS2_OCP6',
     * 'M_SSA_OCM2','M_SSA_OCM1','M_SSA_OCM0','M_SSA_OCP1','M_SSA_OCP2',
     * 'M_SSA_OCP3','M_SSA_OCP4','M_SSA_OCP5','M_SSA_OCP6',
     * 'M_SSC_OCM2','M_SSC_OCM1','M_SSC_OCM0','M_SSC_OCP1','M_SSC_OCP2',
     * 'M_SSC_OCP3','M_SSC_OCP4','M_SSC_OCP5','M_SSC_OCP6',
     * 'M_OCC_OCM2','M_OCC_OCM1','M_OCC_OCM0','M_OCC_OCP1','M_OCC_OCP2',
     * 'M_OCC_OCP3','M_OCC_OCP4','M_OCC_OCP5','M_OCC_OCP6',
     * 'M_BC1_OCM2','M_BC1_OCM1','M_BC1_OCM0','M_BC1_OCP1','M_BC1_OCP2',
     * 'M_BC1_OCP3','M_BC1_OCP4','M_BC1_OCP5','M_BC1_OCP6',
     * 'M_BC2_OCM2','M_BC2_OCM1','M_BC2_OCM0','M_BC2_OCP1','M_BC2_OCP2',
     * 'M_BC2_OCP3','M_BC2_OCP4','M_BC2_OCP5','M_BC2_OCP6',
     * 'M_OCS_OCM2','M_OCS_OCM1','M_OCS_OCM0','M_OCS_OCP1','M_OCS_OCP2',
     * 'M_OCS_OCP3','M_OCS_OCP4','M_OCS_OCP5','M_OCS_OCP6',
     * 'M_BOC_OCM2','M_BOC_OCM1','M_BOC_OCM0','M_BOC_OCP1','M_BOC_OCP2',
     * 'M_BOC_OCP3','M_BOC_OCP4','M_BOC_OCP5','M_BOC_OCP6',
     * 'M_BCS_OCM2','M_BCS_OCM1','M_BCS_OCM0','M_BCS_OCP1','M_BCS_OCP2',
     * 'M_BCS_OCP3','M_BCS_OCP4','M_BCS_OCP5','M_BCS_OCP6',
     * 'M_MXX_OCM2','M_MXX_OCM1','M_MXX_OCM0','M_MXX_OCP1','M_MXX_OCP2',
     * 'M_MXX_OCP3','M_MXX_OCP4','M_MXX_OCP5','M_MXX_OCP6')
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
            trm(i,j,l,n) = MA(l,i,j)*vol2mass(n)*5.d-14
          end do; end do; end do
#endif
#ifdef TRACERS_TOMAS
        case('SO2','NH3','NH4','H2SO4','SOAgas','H2O2_s')
          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
            trm(i,j,l,n) =MA(l,i,j)*vol2mass(n)*1.d-30
          end do; end do; end do

       case('ASO4__01','ASO4__02','ASO4__03','ASO4__04','ASO4__05',
     *    'ASO4__06','ASO4__07','ASO4__08','ASO4__09','ASO4__10',
     *    'ASO4__11','ASO4__12','ASO4__13','ASO4__14','ASO4__15',
     *    'ANACL_01','ANACL_02','ANACL_03','ANACL_04','ANACL_05',
     *    'ANACL_06','ANACL_07','ANACL_08','ANACL_09','ANACL_10',
     *    'ANACL_11','ANACL_12','ANACL_13','ANACL_14','ANACL_15',
     *    'AECIL_01','AECIL_02','AECIL_03','AECIL_04','AECIL_05',
     *    'AECIL_06','AECIL_07','AECIL_08','AECIL_09','AECIL_10',
     *    'AECIL_11','AECIL_12','AECIL_13','AECIL_14','AECIL_15',
     *    'AECOB_01','AECOB_02','AECOB_03','AECOB_04','AECOB_05',
     *    'AECOB_06','AECOB_07','AECOB_08','AECOB_09','AECOB_10',
     *    'AECOB_11','AECOB_12','AECOB_13','AECOB_14','AECOB_15',
     *    'AOCIL_01','AOCIL_02','AOCIL_03','AOCIL_04','AOCIL_05',
     *    'AOCIL_06','AOCIL_07','AOCIL_08','AOCIL_09','AOCIL_10',
     *    'AOCIL_11','AOCIL_12','AOCIL_13','AOCIL_14','AOCIL_15',
     *    'AOCOB_01','AOCOB_02','AOCOB_03','AOCOB_04','AOCOB_05',
     *    'AOCOB_06','AOCOB_07','AOCOB_08','AOCOB_09','AOCOB_10',
     *    'AOCOB_11','AOCOB_12','AOCOB_13','AOCOB_14','AOCOB_15',
     *    'ADUST_01','ADUST_02','ADUST_03','ADUST_04','ADUST_05',
     *    'ADUST_06','ADUST_07','ADUST_08','ADUST_09','ADUST_10',
     *    'ADUST_11','ADUST_12','ADUST_13','ADUST_14','ADUST_15',
     *    'AH2O__01','AH2O__02','AH2O__03','AH2O__04','AH2O__05',
     *    'AH2O__06','AH2O__07','AH2O__08','AH2O__09','AH2O__10',
     *    'AH2O__11','AH2O__12','AH2O__13','AH2O__14','AH2O__15')

          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
                trm(i,j,l,n) =MA(l,i,j)*1.d-20
          end do; end do; end do

      CASE('ANUM__01','ANUM__02','ANUM__03','ANUM__04','ANUM__05',
     *    'ANUM__06','ANUM__07','ANUM__08','ANUM__09','ANUM__10',
     *    'ANUM__11','ANUM__12','ANUM__13','ANUM__14','ANUM__15')

           k=n-n_ANUM(1)+1

          do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
                trm(i,j,l,n) =MA(l,i,j)*7.d-20
     &               /sqrt_xk_xk1(k) !MA(l,i,j)*vol2mass(n)*5.d-14
          end do; end do; end do
#endif

#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS)
        CASE('Clay','Silt1','Silt2','Silt3','Silt4','Silt5','ClayIlli'
     &         ,'ClayKaol','ClaySmec','ClayCalc','ClayQuar','ClayFeld'
     &         ,'ClayHema','ClayGyps','ClayIlHe','ClayKaHe','ClaySmHe'
     &         ,'ClayCaHe','ClayQuHe','ClayFeHe','ClayGyHe','Sil1Quar'
     &         ,'Sil1Feld','Sil1Calc','Sil1Hema','Sil1Gyps','Sil1Illi'
     &         ,'Sil1Kaol','Sil1Smec','Sil1QuHe','Sil1FeHe','Sil1CaHe'
     &         ,'Sil1GyHe','Sil1IlHe','Sil1KaHe','Sil1SmHe','Sil2Quar'
     &         ,'Sil2Feld','Sil2Calc','Sil2Hema','Sil2Gyps','Sil2Illi'
     &         ,'Sil2Kaol','Sil2Smec','Sil2QuHe','Sil2FeHe','Sil2CaHe'
     &         ,'Sil2GyHe','Sil2IlHe','Sil2KaHe','Sil2SmHe','Sil3Quar'
     &         ,'Sil3Feld','Sil3Calc','Sil3Hema','Sil3Gyps','Sil3Illi'
     &         ,'Sil3Kaol','Sil3Smec','Sil3QuHe','Sil3FeHe','Sil3CaHe'
     &         ,'Sil3GyHe','Sil3IlHe','Sil3KaHe','Sil3SmHe','Sil4Quar'
     &         ,'Sil4Feld','Sil4Calc','Sil4Hema','Sil4Gyps','Sil4Illi'
     &         ,'Sil4Kaol','Sil4Smec','Sil4QuHe','Sil4FeHe','Sil4CaHe'
     &         ,'Sil4GyHe','Sil4IlHe','Sil4KaHe','Sil4SmHe','Sil5Quar'
     &         ,'Sil5Feld','Sil5Calc','Sil5Hema','Sil5Gyps','Sil5Illi'
     &         ,'Sil5Kaol','Sil5Smec','Sil5QuHe','Sil5FeHe','Sil5CaHe'
     &         ,'Sil5GyHe','Sil5IlHe','Sil5KaHe','Sil5SmHe')
          ! defaults ok
          hbaij=0D0
          ricntd=0D0
#endif

      end select

C**** Initialise pbl profile if necessary
      if (needtrs(n)) then
        do ipatch=1,size(asflx)
        do j=J_0,J_1
        do ipbl=1,npbl
#ifdef TRACERS_WATER
          if(tr_wd_type(n).eq.nWATER)THEN
            asflx(ipatch)%trabl(ipbl,n,:,j) =
     &           trinit*asflx(ipatch)%qabl(ipbl,:,j)
            if (src_dist_index(n)/=0) asflx(ipatch)%trabl(ipbl,n,:,j) =
     &              asflx(ipatch)%trabl(ipbl,n,:,j)*
     &                                 xyztr(src_dist_index(n), :, j)
          ELSE
            asflx(ipatch)%trabl(ipbl,n,:,j) =
     &           trm(:,j,1,n)*byMA(1,:,j)
          END IF
#else
            asflx(ipatch)%trabl(ipbl,n,:,j) =
     &         trm(:,j,1,n)*byMA(1,:,j)
#endif
        end do
        end do
        end do
      end if

      write(out_line,*) ' Tracer ',trname(n),' initialized at itime='
     *     ,itime
      call write_parallel(trim(out_line))

      end if
      end do
#endif /* TRACERS_ON */
C****
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS)
c read in DMS source
      DMSinput(:,:,:)= 0.d0
      if (file_exists('DMS_SEA')) then
        if(is_fbsa('DMS_SEA')) then
          call openunit('DMS_SEA',iudms,.true.,.true.)
          do mm=1,12
            call readt_parallel(grid,iudms,nameunit(iudms),
     *           DMSinput(:,:,mm),0)
          end do
          call closeunit(iudms)
        else
          iudms = par_open(grid,'DMS_SEA','read')
          call read_dist_data(grid,iudms,'DMSwater',DMSinput)
          call par_close(grid,iudms)
        endif
      endif
 901  FORMAT(3X,3(I4),E11.3)

c read in SO2 emissions
c volcano - continuous
C    Initialize:
      if (iso2volcano>0) then
      so2_src_3D(:,:,:,iso2volcano)= 0.d0
c read lat-lon netcdf file and convert lat,lon,pres to i,j,l.
c NOTE: the input file specifies integrals over its gridboxes.
      ALLOCATE(  psref(grid%i_strt:grid%i_stop,
     &                 grid%j_strt:grid%j_stop) )
      iu_ps = par_open(grid,'PSREF','read')
      call read_dist_data(grid,iu_ps,'prsurf',psref)
      call par_close(grid,iu_ps)
      status = nf_open('SO2_VOLCANO',nf_nowrite,file_id)
      status = nf_inq_varid(file_id,'lon',vid)
      status = nf_get_var_double(file_id,vid,volc_lons)
      status = nf_inq_varid(file_id,'lat',vid)
      status = nf_get_var_double(file_id,vid,volc_lats)
      status = nf_inq_varid(file_id,'Pres_CONTmax',vid)
      status = nf_get_var_double(file_id,vid,volc_pup)
      status = nf_inq_varid(file_id,'VOLC_CONT',vid)
      status = nf_get_var_double(file_id,vid,volc_emiss)
      status = nf_close(file_id)
      do jlat=1,ubound(volc_lats,1)
        do ilon=1,ubound(volc_lons,1)
          if(volc_emiss(ilon,jlat) <= 0.) cycle
          call lonlat_to_ij(
     &         (/volc_lons(ilon),volc_lats(jlat)/),volc_ij)
          ii = volc_ij(1); jj = volc_ij(2)
          if(jj<j_0 .or. jj>j_1) cycle
          if(ii<i_0 .or. ii>i_1) cycle
          Call CALC_VERT_AMP (psref(ii,jj),lm, amref,x1d,pednref,x1d)
          lmax = 1
          do while(pednref(lmax) > volc_pup(ilon,jlat))
            lmax = lmax + 1
          enddo
          amsum = sum(amref(1:lmax))
          do ll=1,lmax ! add source between surf and max height
            so2_src_3d(ii,jj,ll,iso2volcano)=
     &        so2_src_3d(ii,jj,ll,iso2volcano)
     &          +(amref(ll)/amsum)*byaxyp(ii,jj)*
     &           volc_emiss(ilon,jlat)/(SECONDS_PER_DAY*30.4d0)/12.d0
          enddo
        enddo
      enddo
      deallocate(psref)
      endif ! iso2volcano>0
#endif
! ---------------------------------------------------
#ifndef TRACERS_AEROSOLS_SOA
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS)
c Terpenes
      OCT_src(:,:,:)=0.d0
      if (file_exists('Terpenes_01')) then
        if(is_fbsa('Terpenes_01')) then
          call openunit('Terpenes_01',iuc,.true.,.true.)
          call skip_parallel(iuc)
          do mm=1,12
            call readt_parallel(grid,iuc,nameunit(iuc),
     &                          OCT_src(:,:,mm),0)
          end do
          call closeunit(iuc)
        else ! netcdf
          iuc = par_open(grid,'Terpenes_01','read')
          call read_dist_data(grid,iuc,'Terpenes',OCT_src)
          call par_close(grid,iuc)
        endif
c units are mg Terpene/m2/month
        do i=I_0,I_1; do j=J_0,J_1; do mm=1,12
! 10% of terpenes end up being SOA
#ifdef TRACERS_TOMAS
          OCT_src(i,j,mm)=OCT_src(i,j,mm)*0.1d0
     +                    *om2oc(n_AOCOB(1))
#else
#ifdef TRACERS_AMP
          if (n_M_OCC_OC>0) then
            OCT_src(i,j,mm)=OCT_src(i,j,mm)*0.1d0
     +                      *om2oc(n_M_OCC_OC)
          endif
#else
#ifndef SULF_ONLY_AEROSOLS
          OCT_src(i,j,mm)=OCT_src(i,j,mm)*0.1d0
     +                    *om2oc(n_OCII)
#endif  /* SULF_ONLY_AEROSOLS */
#endif
#endif
        end do; end do; end do
      endif
#endif
#endif  /* TRACERS_AEROSOLS_SOA */
! ---------------------------------------------------
#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS) ||\
    (defined TRACERS_AMP) || (defined TRACERS_TOMAS)
c **** reads in files for dust/mineral tracers
      call tracer_ic_soildust
#endif

      end subroutine tracer_IC


      subroutine daily_tracer(end_of_day)
!@sum daily_tracer is called once a day for tracers
!@+   SUBROUTINE tracer_IC is called from daily_tracer to allow for
!@+     tracers that 'turn on' on different dates.
!@auth Jean Lerner
C**** Note this routine must always exist (but can be a dummy routine)
      USE RESOLUTION, only : lm
      Use ATM_COM,    Only: MA,T
      use model_com, only: modelEclock
      USE MODEL_COM, only:itime
      USE FLUXES, only : fearth0,focean,flake0
      USE SOMTQ_COM, only : tmom,mz
      USE DOMAIN_DECOMP_ATM, only : grid, getDomainBounds,
     & write_parallel
      USE RAD_COM, only: o3_yr, ghg_yr
#ifdef TRACERS_VOLCEXP
      use GEOM, only: byaxyp
      USE AEROSOL_SOURCES, only: so2_src_3d,iso2volcanoexpl
      USE timestream_mod, only: init_stream,read_stream
      USE tracer_com, only: SO2_volc_stream,SO2_vphe_stream
#endif
#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS) ||\
    (defined TRACERS_AMP) || (defined TRACERS_TOMAS) 
      use pario, only : par_open,par_close,read_dist_data
      use timestream_mod, only : timestream
      USE timestream_mod, only: init_stream,read_stream
      USE trdust_mod, only: imDust
      USE trdust_mod, only: nDustBins
      USE trdust_mod, only: nAerocomDust
      USE trdust_mod, only: d_dust
#endif
#ifdef PRESC_BB_INJ
      USE timestream_mod, only: init_stream,read_stream
      USE tracer_com, only: BB_injbot_stream
      USE tracer_com, only: BB_injmami_stream
      USE tracer_com, only: BB_injtop_stream
      USE tracer_com, only: BB_inj_bot
      USE tracer_com, only: BB_inj_mami
      USE tracer_com, only: BB_inj_top
      USE tracer_com, only: fire_src_3d_fact
#endif  /* PRESC_BB_INJ */
#ifdef WATER_MISC_GRND_CH4_SRC
      use tracer_com, only: scale_CH4MGOL
#endif
      use GEOM, only: lat_to_j
      use GEOM, only: lon_to_i
      USE ATM_COM, only: byMA
      USE GEOM, only: axyp,byaxyp,dlat
      USE RESOLUTION, only : jm
#ifdef TRACERS_SPECIAL_Shindell
      use TRCHEM_Shindell_COM, only: NOx_yr
      use TRCHEM_Shindell_COM, only: CO_yr
      use TRCHEM_Shindell_COM, only: VOC_yr
#endif  /* TRACERS_SPECIAL_Shindell */
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS)
      use TRACER_COM, only: aer_int_yr
      use TRACER_COM, only: SO2_int_yr
      use TRACER_COM, only: NH3_int_yr
      use TRACER_COM, only: BC_int_yr
      use TRACER_COM, only: OC_int_yr
      use TRACER_COM, only: direct_inject_num
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
#ifdef TRACERS_AMP
      use TRACER_COM, only: direct_inject_DD1
      use TRACER_COM, only: direct_inject_DD2
#endif
      use TRACER_COM, only: direct_inject_BC
      use TRACER_COM, only: direct_inject_OC
      use TRACER_COM, only: nVolcanic,nBiomass
      USE AEROSOL_SOURCES, only: so2_src_3d,iso2directinj,H2O_src_3d
      USE AEROSOL_SOURCES, only: su_src_3d,bc_src_3d,oc_src_3d
#ifdef TRACERS_AMP
      USE AEROSOL_SOURCES, only: dd1_src_3d,dd2_src_3d
#endif
      USE apply3d, only : apply_tracer_3Dsource
      USE FLUXES, only: tr3Dsource
#endif
#ifdef CUBED_SPHERE
      USE tracer_com, only: AIRCstreams
      USE tracer_com, only: ROCKETstreams
#endif
      use TracerSurfaceSource_mod, only: itsCH4MGOL
      USE CONSTANT, only: grav
      use RunTimeControls_mod, only: tracers_amp
      use RunTimeControls_mod, only: tracers_tomas
      use RunTimeControls_mod, only: tracers_aerosols_soa
      use TimeConstants_mod, only: SECONDS_PER_DAY, HOURS_PER_DAY
      use OldTracer_mod, only: trname, itime_tr0, MAX_LEN_NAME
      use OldTracer_mod, only: nBBsources,vol2mass
      use TRACER_COM, only: tracers, set_ntsurfsrc
      USE TRACER_COM, only: daily_z
      USE ATM_COM, only: zatmo
      USE CONSTANT, only: bygrav
      USE TRACER_COM, only: n_CO2n
      USE TRACER_COM, only: NTM,n_SO4,n_SO2,n_M_ACC_SU,n_M_AKK_SU,
     & n_CH4,n_Isoprene,sfc_src,ntsurfsrc,
     & trans_emis_overr_yr,trans_emis_overr_day
      use TRACER_COM, only: ntm_chem_beg,ntm_chem_end
#ifdef TRACERS_TOMAS
      use TRACER_COM, only: n_ASO4,nbins
#endif
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
      use TRACERS_dCO, only: dCO_fact
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
#ifdef TRACERS_SPECIAL_Lerner
      use tracer_com, only: n_O3,n_CO2,n_CH4
      USE TRACERS_MPchem_COM, only: STRATCHEM_SETUP
      USE LINOZ_CHEM_COM, only: LINOZ_SETUP, Linoz_daily
#endif
#ifdef TRACERS_PHOTOLYSIS
      use photolysis, only: rad_FL,read_FL
#endif  /* TRACERS_PHOTOLYSIS */
#ifdef TRACERS_SPECIAL_Shindell
      USE TRCHEM_Shindell_COM, only: tune_NOx
      USE TRCHEM_Shindell_COM, only: tune_BVOC
#ifdef EESC_BASED_CLTOT_BRTOT
      USE TRCHEM_Shindell_COM, only: ch3cl,ch3br,ccl4,f114,hcfc141b,
     & hcfc22,halon2402,halon1301,f113,halon1211,ch3ccl3,f12,f11
      USE TRCHEM_Shindell_COM, only: ch3cl_stream,ch3br_stream,
     & ccl4_stream,f114_stream,hcfc141b_stream,hcfc22_stream,
     & halon2402_stream,halon1301_stream,f113_stream,halon1211_stream,
     & ch3ccl3_stream,f12_stream,f11_stream
      USE timestream_mod, only: init_stream,read_stream_ijless
#endif /* EESC_BASED_CLTOT_BRTOT */
#endif /* TRACERS_SPECIAL_Shindell */
#ifdef TRACERS_COSMO
      USE COSMO_SOURCES, only : variable_phi
#endif
      use Tracer_mod, only: Tracer, readSurfaceSources
      IMPLICIT NONE
      INTEGER n,last_month,kk,nread,xday,xyear,ns
      LOGICAL, INTENT(IN) :: end_of_day
      data last_month/-1/
      INTEGER J_0, J_1, I_0, I_1,I,J,Ipoint,Jpoint,ll,lmax,lmin
      INTEGER I0_rect,I1_rect,J0_rect,J1_rect,Irect,Jrect
#ifdef TRACERS_TOMAS
      integer km, najl_num,naij_num,k
      real*8 :: scalesize(nbins+nbins) !temporal emission mass fraction
#endif
#ifdef TRACERS_VOLCEXP
      real*8, dimension(GRID%I_STRT_HALO:GRID%I_STOP_HALO,
     &                  GRID%J_STRT_HALO:GRID%J_STOP_HALO)
     &     :: SO2_volc_emis_expl, Plume_hei_volc_emis_expl !  volc emiss
#endif
#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS) ||\
    (defined TRACERS_AMP) || (defined TRACERS_TOMAS) 
!@var dust_bin1_stream reading of offline emissions, when needed, based on imDust
!@var dust_bin2_stream reading of offline emissions, when needed, based on imDust
!@var dust_bin3_stream reading of offline emissions, when needed, based on imDust
!@var dust_bin4_stream reading of offline emissions, when needed, based on imDust
      integer :: ib,fid
      type(timestream) :: dust_bin1_stream
      type(timestream) :: dust_bin2_stream
      type(timestream) :: dust_bin3_stream
      type(timestream) :: dust_bin4_stream
      real*8, dimension(GRID%I_STRT:GRID%I_STOP,
     &                  GRID%J_STRT:GRID%J_STOP) :: work_aerocom ! dust emissions
      real*8, dimension(GRID%I_STRT_HALO:GRID%I_STOP_HALO,
     &                  GRID%J_STRT_HALO:GRID%J_STOP_HALO) :: work_sum ! tmp array
#endif
      integer :: ex
      real*8 :: topo,llz
      real*8, dimension(lm) :: dz,totz
      real*8 :: inj_mult,area_wgt,day_frac,time_fact,area_wgt_denom
      class (Tracer), pointer :: pTracer
C****
      integer :: year, month, dayOfYear
      character(len=MAX_LEN_NAME) :: tmpString
      logical :: isChemTracer
#ifdef EESC_BASED_CLTOT_BRTOT
      integer :: dd,yy
#endif

      call modelEclock%get(year=year, month=month,
     *     dayOfYear=dayOfYear)
CC****
C**** Extract useful local domain parameters from "grid"
C****

      call getDomainBounds(grid, J_STRT=J_0, J_STOP=J_1)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

      if(end_of_day.or.(daily_z(I_0,J_0,1)==0.d0)) then
         Call COMPUTE_GZ (MA,T,TMOM(MZ,:,:,:), DAILY_Z)
        daily_z = daily_z/grav
      endif

#ifdef TRACERS_SPECIAL_Shindell
#ifdef EESC_BASED_CLTOT_BRTOT
      ! 'year' from the modelEclock unless override by ghg_yr:
      dd=dayofyear
      if(ghg_yr .ne. 0) then
        yy = ghg_yr
      else
        yy = year
      end if
      if(.not. end_of_day) then ! synonym for model init phase
        call init_stream(grid,    ch3cl_stream,'EESC','ch3cl'
     &       ,0d0,1d30,'none',yy,dd,ijless=.true.)
        call init_stream(grid,    ch3br_stream,'EESC','ch3br'
     &       ,0d0,1d30,'none',yy,dd,ijless=.true.)
        call init_stream(grid,     ccl4_stream,'EESC','ccl4'
     &       ,0d0,1d30,'none',yy,dd,ijless=.true.)
        call init_stream(grid,     f114_stream,'EESC','f114'
     &       ,0d0,1d30,'none',yy,dd,ijless=.true.)
        call init_stream(grid, hcfc141b_stream,'EESC','hcfc141b'
     &       ,0d0,1d30,'none',yy,dd,ijless=.true.)
        call init_stream(grid,   hcfc22_stream,'EESC','hcfc22'
     &       ,0d0,1d30,'none',yy,dd,ijless=.true.)
        call init_stream(grid,halon2402_stream,'EESC','halon2402'
     &       ,0d0,1d30,'none',yy,dd,ijless=.true.)
        call init_stream(grid,halon1301_stream,'EESC','halon1301'
     &       ,0d0,1d30,'none',yy,dd,ijless=.true.)
        call init_stream(grid,     f113_stream,'EESC','f113'
     &       ,0d0,1d30,'none',yy,dd,ijless=.true.)
        call init_stream(grid,halon1211_stream,'EESC','halon1211'
     &       ,0d0,1d30,'none',yy,dd,ijless=.true.)
        call init_stream(grid,  ch3ccl3_stream,'EESC','ch3ccl3'
     &       ,0d0,1d30,'none',yy,dd,ijless=.true.)
        call init_stream(grid,      f12_stream,'EESC','f12'
     &       ,0d0,1d30,'none',yy,dd,ijless=.true.)
        call init_stream(grid,      f11_stream,'EESC','f11'
     &       ,0d0,1d30,'none',yy,dd,ijless=.true.)
      endif
      call read_stream_ijless(grid,    ch3cl_stream,yy,dd,ch3cl)
      call read_stream_ijless(grid,    ch3br_stream,yy,dd,ch3br)
      call read_stream_ijless(grid,     ccl4_stream,yy,dd,ccl4)
      call read_stream_ijless(grid,     f114_stream,yy,dd,f114)
      call read_stream_ijless(grid, hcfc141b_stream,yy,dd,hcfc141b)
      call read_stream_ijless(grid,   hcfc22_stream,yy,dd,hcfc22)
      call read_stream_ijless(grid,halon2402_stream,yy,dd,halon2402)
      call read_stream_ijless(grid,halon1301_stream,yy,dd,halon1301)
      call read_stream_ijless(grid,     f113_stream,yy,dd,f113)
      call read_stream_ijless(grid,halon1211_stream,yy,dd,halon1211)
      call read_stream_ijless(grid,  ch3ccl3_stream,yy,dd,ch3ccl3)
      call read_stream_ijless(grid,      f12_stream,yy,dd,f12)
      call read_stream_ijless(grid,      f11_stream,yy,dd,f11)
#endif /* EESC_BASED_CLTOT_BRTOT */
#endif /* TRACERS_SPECIAL_Shindell */

#ifdef TRACERS_VOLCEXP
! Reading explosive volcano emissions for SO2
      if(.not. end_of_day) then ! synonym for model init phase
        ! initialize the file handle

        call init_stream(grid,SO2_volc_stream,'SO2_VOLCANO_EXPL','SO2'
     &       ,0d0,1d30,'linm2m',year,dayofyear)

        call init_stream(grid,SO2_vphe_stream,'SO2_VOLCANO_EXPL',
     &       'Plume_height',0d0,1d30,'linm2m',year,dayofyear)
      endif

      call read_stream(grid,SO2_volc_stream,year,dayofyear,
     &                 SO2_volc_emis_expl)
      call read_stream(grid,SO2_vphe_stream,year,dayofyear,
     &                 Plume_hei_volc_emis_expl)

      so2_src_3d(:,:,:,iso2volcanoexpl) = 0.d0

      DO J=J_0,J_1
      DO I=I_0,I_1

        so2_volc_emis_expl(i,j) = so2_volc_emis_expl(i,j)*byaxyp(i,j) ! kg -> kg/m2
        if(so2_volc_emis_expl(i,j) <= 0.d0) cycle
          lmax = 1
          do while(daily_z(i,j,lmax) < Plume_hei_volc_emis_expl(i,j))
            lmax = lmax + 1
          enddo
            lmax = lmax + 1 ! adding one layer as to not have plume height identical to mixing height
            lmin=max(1,lmax - lmax/3)
          do ll=lmin,lmax ! add source into the upper 1/3 of the plume
                          ! conversion kt d-1 into kg s-1
          if (lmax <= 2) then
            so2_src_3d(i,j,1,iso2volcanoexpl)=
     &        so2_src_3d(i,j,1,iso2volcanoexpl)
     &                + so2_volc_emis_expl(i,j)/SECONDS_PER_DAY*1.d6
          else
            so2_src_3d(i,j,ll,iso2volcanoexpl)=
     &        so2_src_3d(i,j,ll,iso2volcanoexpl)
     &                + (1./(float(lmax-lmin)+1))
     &                * so2_volc_emis_expl(i,j)/SECONDS_PER_DAY*1.d6
          endif
          enddo
        enddo
      enddo
#endif

#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS) ||\
    (defined TRACERS_AMP) || (defined TRACERS_TOMAS) 
! prescribed AeroCom dust emissions
      if ( imDust == 1 .or. imDust == 3 .or. imDust == 5 ) then

        if(.not. end_of_day) then ! synonym for model init phase
        ! initialize the file handle
          call init_stream(grid,dust_bin1_stream,'dust_bins','dust_bin1'
     &         ,0d0,1d30,'linm2m',year,dayofyear)
          call init_stream(grid,dust_bin2_stream,'dust_bins','dust_bin2'
     &         ,0d0,1d30,'linm2m',year,dayofyear)
          call init_stream(grid,dust_bin3_stream,'dust_bins','dust_bin3'
     &         ,0d0,1d30,'linm2m',year,dayofyear)
          call init_stream(grid,dust_bin4_stream,'dust_bins','dust_bin4'
     &         ,0d0,1d30,'linm2m',year,dayofyear)
        endif ! end_of_day

c     The units of the prescribed dust emission fluxes are kg/day per
c     model grid cell in the AeroCom input files. The read in mass
c     fluxes have to be divided by the grid cell areas and the number of
c     seconds per day, since the subroutine for dust dust emission,
c     through which the fluxes go, expects the fluxes to be in kg/m^2/s.
        call read_stream(grid,dust_bin1_stream,year,dayofyear,
     &                   work_aerocom)
        d_dust(i_0:i_1,j_0:j_1,1)=
     &    work_aerocom/axyp(i_0:i_1,j_0:j_1)/SECONDS_PER_DAY
        call read_stream(grid,dust_bin2_stream,year,dayofyear,
     &                   work_aerocom)
        d_dust(i_0:i_1,j_0:j_1,2)=
     &    work_aerocom/axyp(i_0:i_1,j_0:j_1)/SECONDS_PER_DAY
        call read_stream(grid,dust_bin3_stream,year,dayofyear,
     &                   work_aerocom)
        d_dust(i_0:i_1,j_0:j_1,3)=
     &    work_aerocom/axyp(i_0:i_1,j_0:j_1)/SECONDS_PER_DAY
        call read_stream(grid,dust_bin4_stream,year,dayofyear,
     &                   work_aerocom)
        d_dust(i_0:i_1,j_0:j_1,4)=
     &    work_aerocom/axyp(i_0:i_1,j_0:j_1)/SECONDS_PER_DAY

        if ( imDust == 3 ) then
!c******** normalize AeroCom size distribution to be used as factors for model
!c         calculated dust emission flux
          work_sum=sum(d_dust, dim=3)
          do ib = 1,nAerocomDust
            where (work_sum>0.d0)
              d_dust(:,:,ib)=d_dust(:,:,ib)/work_sum
            elsewhere
              d_dust(:,:,ib)=0.d0
            endwhere
          enddo
        elseif ( imDust == 5 ) then
          do ib = nAerocomDust+1,nDustBins
            d_dust(:,:,ib)=d_dust(:,:,nAerocomDust)
          enddo
        endif

      endif ! imDust
#endif

#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS)
! simple aerosol and gas injections based on rundeck parameters
        Ipoint=-1
        Jpoint=-1
        H2O_src_3d(:,:,:)=0.d0
        SU_src_3d(:,:,:)=0.d0
#ifdef TRACERS_AMP
        DD1_src_3d(:,:,:)=0.d0
        DD2_src_3d(:,:,:)=0.d0
#endif
        BC_src_3d(:,:,:)=0.d0
        OC_src_3d(:,:,:)=0.d0
        if (iso2directinj>0) so2_src_3d(:,:,:,iso2directinj)=0.d0

      if (direct_inject_num>0) then
        do ex=1,direct_inject_num

! check timing, note: not coded to inject continuously across new year
!                     so set up 2 adjacent injections for this case
         if (direct_inject_jday(ex)>dayOfYear) cycle
         if (direct_inject_jday(ex)+direct_inject_ndays(ex)-1
     &     <dayOfYear) cycle
         if (direct_inject_year(ex)/=year) cycle

! check location

         if ((direct_inject_rectlat0(ex).ne.direct_inject_rectlat1(ex))
     &  .or. (direct_inject_rectlon0(ex).ne.direct_inject_rectlon1(ex)))
     &   then
!          rectangular injection region
           J0_rect=lat_to_j(direct_inject_rectlat0(ex))
           J1_rect=lat_to_j(direct_inject_rectlat1(ex))
           I0_rect=lon_to_i(direct_inject_rectlon0(ex))
           I1_rect=lon_to_i(direct_inject_rectlon1(ex))
           if (j_0>J1_rect .or. j_1<J0_rect) cycle
           if (i_0>I1_rect .or. i_1<I0_rect) cycle

!          set up to weight by column area relative to injection area mean
!          column. This calculation is an approximation, due to injection
!          regions potentially being distributed among multiple processors
!          (so can't rely on axyp(i,j), etc)
           area_wgt_denom = 0.d0
           Do J=J0_rect,J1_rect
             area_wgt_denom = area_wgt_denom
     &         +cos((J-.5*(jm + 1))*dlat) / (J1_rect-J0_rect+1.d0)
           ENDDO
         else
!          point injection (single column)
           jpoint=lat_to_j(direct_inject_pointlat(ex))
           if (jpoint<j_0 .or. jpoint>j_1) cycle
           ipoint=lon_to_i(direct_inject_pointlon(ex))
           if (ipoint<i_0 .or. ipoint>i_1) cycle
         endif
 
! set to distribute injection across its duration, reducing daily emissions
! on partial injection days if these exist
         if ((direct_inject_jday(ex)==dayOfYear)
     &     .and. (direct_inject_ndays(ex)==1)) then
           day_frac=(direct_inject_hr1(ex)-direct_inject_hr0(ex))
     &       /HOURS_PER_DAY
         else if (direct_inject_jday(ex)==dayOfYear) then
           day_frac=(HOURS_PER_DAY-direct_inject_hr0(ex))/HOURS_PER_DAY
         else if (direct_inject_jday(ex)==
     &     dayOfYear+direct_inject_ndays(ex)-1) then
           day_frac=direct_inject_hr1(ex)/HOURS_PER_DAY
         else
           day_frac = 1.d0
         endif

         time_fact=day_frac/(direct_inject_ndays(ex)-2.d0
     &     + (HOURS_PER_DAY-direct_inject_hr0(ex))/HOURS_PER_DAY
     &     + direct_inject_hr1(ex)/HOURS_PER_DAY )

! find layers edges and locate lmin and lmax

         DO J=J_0,J_1
         DO I=I_0,I_1
           if (ipoint>0 .and. jpoint>0) then ! If point inj, only one column
             if (I.ne.Ipoint) cycle
             if (J.ne.Jpoint) cycle
           else ! If rectangular inj, inject into any column within region
             if (I<I0_rect .or. I>I1_rect) cycle
             if (J<J0_rect .or. J>J1_rect) cycle

             area_wgt = cos((J-.5*(jm + 1))*dlat) ! more mass to larger columns
     &         / area_wgt_denom
           endif

           topo=zatmo(i,j)*bygrav
           dz(1)=(daily_z(i,j,1)-topo)*2.d0
           totz(1)=dz(1)+topo ! altitude of top of layer
           do ll=2,lm
             dz(ll)=(daily_z(i,j,ll)-totz(ll-1))*2.d0
             totz(ll)=totz(ll-1)+dz(ll)
           enddo
           do lmin=2,lm ! start from 2, to avoid mixing layer
             if (totz(lmin)>=direct_inject_bot(ex)) exit
           enddo
           do lmax=lmin,lm
             if (totz(lmax)>=direct_inject_top(ex)) exit
           enddo
           if (lmin==lmax) then
             dz(lmin)=direct_inject_top(ex)-direct_inject_bot(ex)
           else
             dz(lmin)=totz(lmin)-direct_inject_bot(ex)
             dz(lmax)=direct_inject_top(ex)-totz(lmax-1)
           endif
           llz=sum(dz(lmin:lmax))

! set emissions
           do ll=lmin,lmax
             inj_mult=dz(ll)/llz/SECONDS_PER_DAY*time_fact
     &         *1.d9*byaxyp(i,j)

             if (ipoint<0 .or. jpoint<0) then !if rect inj, distribute across area
               inj_mult=inj_mult*area_wgt
     &           /(J1_rect-J0_rect+1.d0)/(I1_rect-I0_rect+1.d0)
             endif

             so2_src_3d(i,j,ll,iso2directinj)=
     &         so2_src_3d(i,j,ll,iso2directinj)+
     &         direct_inject_SO2(ex)*inj_mult ! kg m-2 s-1
             H2O_src_3d(i,j,ll)=H2O_src_3d(i,j,ll)+
     &         direct_inject_H2O(ex)*inj_mult*byMA(ll,i,j) ! kg kg-1 s-1
             SU_src_3d(i,j,ll)=SU_src_3d(i,j,ll)+
     &         direct_inject_SU(ex)*inj_mult ! kg m-2 s-1
#ifdef TRACERS_AMP
             DD1_src_3d(i,j,ll)=DD1_src_3d(i,j,ll)+
     &         direct_inject_DD1(ex)*inj_mult ! kg m-2 s-1
             DD2_src_3d(i,j,ll)=DD2_src_3d(i,j,ll)+
     &         direct_inject_DD2(ex)*inj_mult ! kg m-2 s-1
#endif
             BC_src_3d(i,j,ll)=BC_src_3d(i,j,ll)+
     &         direct_inject_BC(ex)*inj_mult ! kg m-2 s-1
             OC_src_3d(i,j,ll)=OC_src_3d(i,j,ll)+
     &         direct_inject_OC(ex)*inj_mult ! kg m-2 s-1
           enddo

! write confirmation of injection to .PRT file (deleted w/ restart)

         if (((ipoint>0 .and. jpoint>0) .or.
     &    (I.eq.I0_rect .and. J.eq.J0_rect))
     &    .and. (dayOfYear==direct_inject_jday(ex))) then ! report 1x per inj
          write(*,'(a,i4,a,/,es12.4,a,/,es12.4,a,/'//
#ifdef TRACERS_AMP
     &     ',es12.4,a,/,es12.4,a,/'//
#endif
     &     ',es12.4,a,/,es12.4,a,/,es12.4,a,/,a,es12.4,a,es12.4,a)')
     .     "direct_inject injection #", ex, " initiated, ",
     .     direct_inject_SO2(ex),' Tg SO2, ',
     .     direct_inject_SU(ex),' Tg SO4, ',
#ifdef TRACERS_AMP
     .     direct_inject_DD1(ex),' Tg fine dust, ',
     .     direct_inject_DD2(ex),' Tg coarse dust, ',
#endif
     .     direct_inject_BC(ex),' Tg BC, ',
     .     direct_inject_OC(ex),' Tg OC, & ',
     .     direct_inject_H2O(ex),' Tg H2O',' distributed between ',
     .     direct_inject_bot(ex),' and ',direct_inject_top(ex),
     .     'meters altitude'
          if (ipoint>0 .and. jpoint>0) then
            write(*,'(a,es12.4,a,es12.4,a)')
     .      ' at coordinates ',direct_inject_pointlat(ex),
     .      ' deg lat,',direct_inject_pointlon(ex),' deg lon'
          else
           write(*,'(a,es12.4,a,es12.4,a,es12.4,a,es12.4,a)')
     .     ' at coordinates ',direct_inject_rectlat0(ex),
     .     ' to ',direct_inject_rectlat1(ex),' deg lat, ',
     .     direct_inject_rectlon0(ex),' to ',direct_inject_rectlon1(ex),
     .     ' deg lon'
          endif
         endif

         ENDDO !I
         ENDDO !J
        enddo ! ex
      endif
#endif

#ifdef PRESC_BB_INJ
! Read in bottom and top of biomass burning emissions injection heights
! This is similar to explosive volcanoes in '#ifdef TRACERS_VOLCEXP'
! Possible injection height parameters 'APB', 'MAMI', and 'APT'
! are in GFAS emissions files identified in the rundeck by 'INJ_HGT'.

      if(.not. end_of_day) then ! synonym for model init phase
! initialize the file handle

! Altitude of plume bottom
        call init_stream(grid,BB_injbot_stream,'INJ_HGT','APB'
     &       ,0d0,1d30,'linm2m',year,dayofyear)
! Mean altitude of maximum injection
        call init_stream(grid,BB_injmami_stream,'INJ_HGT','MAMI'
     &       ,0d0,1d30,'linm2m',year,dayofyear)
! Altitude of plume top
        call init_stream(grid,BB_injtop_stream,'INJ_HGT','APT'
     &       ,0d0,1d30,'linm2m',year,dayofyear)
      endif

      call read_stream(grid,BB_injbot_stream,year,dayofyear,
     &                 BB_inj_bot)
      call read_stream(grid,BB_injmami_stream,year,dayofyear,
     &                 BB_inj_mami)
      call read_stream(grid,BB_injtop_stream,year,dayofyear,
     &                 BB_inj_top)

      fire_src_3d_fact(:,:,:)=0.d0

      DO J=J_0,J_1
        DO I=I_0,I_1
          if (BB_inj_top(i,j).le.1.d0) cycle ! not zero, to capture tiny values

! Fill-in APB with APT-2*(APT-MAMI) when no APB data exist
          if (BB_inj_bot(i,j)==0.d0) then
            BB_inj_bot(i,j)=BB_inj_top(i,j)-
     &        2.d0*(BB_inj_top(i,j)-BB_inj_mami(i,j))
          endif ! else nothing, data exists

! Find model levels for current plume bottom and top, adopted from direct_inject
          topo=zatmo(i,j)*bygrav
          dz(1)=(daily_z(i,j,1)-topo)*2.d0
          totz(1)=dz(1)+topo ! altitude at top of layer
          do ll=2,lm
            dz(ll)=(daily_z(i,j,ll)-totz(ll-1))*2.d0
            totz(ll)=totz(ll-1)+dz(ll)
          enddo
          do lmin=1,lm
            if (totz(lmin)>=BB_inj_bot(i,j)) exit
          enddo
          do lmax=lmin,lm
            if (totz(lmax)>=BB_inj_top(i,j)) exit
          enddo
          if (lmin==lmax) then
            if (BB_inj_top(i,j) > BB_inj_bot(i,j)) then
              dz(lmin)=BB_inj_top(i,j)-BB_inj_bot(i,j)
            endif ! else nothing, use dz unmodified
          else
            dz(lmin)=totz(lmin)-BB_inj_bot(i,j)
            dz(lmax)=BB_inj_top(i,j)-totz(lmax-1)
          endif
          llz=sum(dz(lmin:lmax))

! calculate emissions factor
          do ll=lmin,lmax
            fire_src_3d_fact(ll,i,j)=dz(ll)/llz
          enddo

        ENDDO ! i
      ENDDO ! j
#endif  /* PRESC_BB_INJ */

#ifdef TRACERS_SPECIAL_Lerner
      if (.not. end_of_day) then
C**** Initialize tables for linoz
        if (itime.ge.itime_tr0(n_O3)) then  
              call linoz_setup(n_O3)
              call linoz_daily(modelEclock%getYear(),
     &        modelEclock%getDayofYear())
              call linoz_STRATL
        endif 

C**** Initialize tables for Prather StratChem tracers
        call stratchem_setup
      else 
C**** Update linoz tables each day
          call linoz_daily(modelEclock%getYear(),
     &    modelEclock%getDayofYear())
          call linoz_STRATL
      end if  ! not end of day

C**** Prather StratChem tracers change each month
      if (modelEclock%getMonth().NE.last_month) then
        call STRTL  ! one call does all based on n_MPtable_max
        last_month = modelEclock%getMonth()
      end if 

#endif

#ifdef TRACERS_COSMO
      if (variable_phi .eq. 0) then
         call read_Be_source_noAlpha
         print*, "called old version of Be source"
      end if

      if (variable_phi .eq. 1) then
         call read_Be_source
         print*, "called new version of Be source"
      end if

      if (variable_phi .eq. 2) then
         if ((dayOfYear .eq. 1) .or. (.not. end_of_day)) then
            call update_annual_phi
            print*, "called update_annual_phi"
         end if
      end if

      if (variable_phi .eq. 3) then
         call update_daily_phi
         print*, "called update_daily_phi"
      end if
#endif

!===============================================================================
! Chemistry/OMA/MATRIX/TOMAS case, where surface emissions are of type
! TRACERNAME_XX. Also (partially) LERNER:
!===============================================================================
#ifdef TRACERS_PHOTOLYSIS
C**** Next line for fastj photon fluxes to vary with time:
      if(rad_FL.gt.0) call READ_FL(end_of_day)
#endif  /* TRACERS_PHOTOLYSIS */
#if (defined TRACERS_SPECIAL_Shindell) || (defined TRACERS_AEROSOLS_Koch) ||\
    (defined TRACERS_AMP) || (defined TRACERS_TOMAS) ||\
    (defined TRACERS_SPECIAL_Lerner) || (defined TRACERS_GASEXCH_GCC) ||\
    (defined TRACERS_PASSIVE)

      !! xday is used by multiple sources below
      xday=dayOfYear

#ifdef TRACERS_SPECIAL_Shindell
#ifdef SOLAR_ENERGETIC_PARTICLES
      call update_ion_pairs_and_apex(year,dayOfYear,end_of_day)
#endif /* SOLAR_ENERGETIC_PARTICLES */
#endif /* TRACERS_SPECIAL_Shindell */

!-------------------------------------------------------------------------------
! tracers loop
!-------------------------------------------------------------------------------
      do n=1,ntm
        pTracer => tracers%getReference(trname(n))

        if ((n>=ntm_chem_beg).and.(n<=ntm_chem_end)) then
          isChemTracer=.true. ! careful: only for this n loop
        else
          isChemTracer=.false.
        end if
!**** Allow overriding of ozone precursor transient emissions date:
! for now, tying this to O3_yr becasue Gavin
! didn't want a new parameter, also not allowing
! day overriding yet, because of that.
#ifdef TRACERS_SPECIAL_Shindell
        if (isChemTracer) then
          trans_emis_overr_yr=o3_yr
          if(trans_emis_overr_yr > 0)then
            xyear=trans_emis_overr_yr
          else
            xyear=year
          endif

          select case (trname(n))
          case ('NOx')
            if (NOx_yr > 0) xyear=NOx_yr
          case ('CO')
            if (CO_yr > 0) xyear=CO_yr
          case ('Alkenes', 'Paraffin')
            if (VOC_yr > 0) xyear=VOC_yr
          end select
        else
#endif

! allow overriding of transient aerosol emissions date
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS)
          xyear=year
          if(aer_int_yr > 0) xyear=aer_int_yr
          select case (trname(n))
          case ('SO2', 'SO4', 'M_ACC_SU', 'M_AKK_SU', 'ASO4__01')
            if (SO2_int_yr > 0) xyear=SO2_int_yr
          case ('NH3')
            if (NH3_int_yr > 0) xyear=NH3_int_yr
          case ('BCII', 'BCB', 'M_BC1_BC', 'M_BOC_BC', 'AECOB_01')
            if (BC_int_yr > 0) xyear=BC_int_yr
          case ('OCII', 'OCB', 'M_OCC_OC', 'M_BOC_OC', 'AOCOB_01',
     &          'vbsAm2', 'vbsAm1', 'vbsAz', 'vbsAp1', 'vbsAp2',
     &          'vbsAp3', 'vbsAp4', 'vbsAp5', 'vbsAp6')
            if (OC_int_yr > 0) xyear=OC_int_yr
          end select
#endif
#ifdef TRACERS_SPECIAL_Shindell
        end if
#endif
#ifdef TRACERS_SPECIAL_Lerner
        ! no overriding for the moment for Lerner:
        xyear=year
#endif
#ifdef TRACERS_GASEXCH_GCC
        xyear=year
#endif

        ! define nread per tracer
        ! only include offline files here, so ignore e.g. do_fire(n)
        nread=ntsurfsrc(n)+nBBsources(n)

        select case (trname(n))
        case ('OCII','M_OCC_OC','SOAgas') ! Koch/AMP/TOMAS cases
          if (.not.tracers_aerosols_soa) nread=nread-1
        case ('SO4','M_AKK_SU','M_ACC_SU',
     &        'ANUM__01','ANUM__02','ANUM__03','ANUM__04','ANUM__05',
     &        'ANUM__06','ANUM__07','ANUM__08','ANUM__09','ANUM__10',
     &        'ANUM__11','ANUM__12','ANUM__13','ANUM__14','ANUM__15',
     &        'ASO4__01','ASO4__02','ASO4__03','ASO4__04','ASO4__05',
     &        'ASO4__06','ASO4__07','ASO4__08','ASO4__09','ASO4__10',
     &        'ASO4__11','ASO4__12','ASO4__13','ASO4__14','ASO4__15')
          nread=0
        case ('vbsAm2', 'vbsAm1', 'vbsAz', 'vbsAp1', 'vbsAp2',
     &        'vbsAp3', 'vbsAp4', 'vbsAp5', 'vbsAp6')
#ifdef TRACERS_PASSIVE
        case ('Rn222', 'CO2', 'N2O', 'CFC11', '14CO2', 'CH4', 'O3')
          nread=0
        case ('SF6', 'SF6_c', 'nh5', 'nh50', 'e90',
     &        'st8025', 'aoa', 'aoanh', 'tape_rec', 'nh15')
          nread=0 ! regional sources calculated in the code, not via a file
#endif  /* TRACERS_PASSIVE */
        end select

#ifdef TRACERS_SPECIAL_Lerner
        ! for the moment, for Lerner tracers, only allow CO2 and CH4:
        select case (trname(n))
        case ('CO2','CH4')
          nread=ntsurfsrc(n)+nBBsources(n)
        case ('Rn222','N2O','CFC11','14CO2','O3')
          nread=0
        end select
#endif

!-------------------------------------------------------------------------------
! read surface sources of all tracers
!-------------------------------------------------------------------------------
        call readSurfaceSources(pTracer,n,nread,xyear,xday,
     &                          itime,itime_tr0(n),sfc_src,isChemTracer)
!-------------------------------------------------------------------------------

! post-read calculations
        select case (trname(n))
        case ('CH4')
#ifdef WATER_MISC_GRND_CH4_SRC
          do ns=1,ntsurfsrc(n)
            if(pTracer%surfaceSources(ns)%skipReason==itsCH4MGOL) then
              sfc_src(I_0:I_1,J_0:J_1,n,ns)=
     &          scale_CH4MGOL * (
     &          1.698d-12*fearth0(I_0:I_1,J_0:J_1) + ! incl. 5.3558e-5 from Jean
     &          5.495d-11*flake0(I_0:I_1,J_0:J_1)  + ! incl. 17.330e-4 from Jean
     &          1.141d-12*focean(I_0:I_1,J_0:J_1)    ! incl. 3.5997e-5 from Jean
     &                          )
              exit ! Found. Should be only one source.
            end if
          end do
#endif
#ifdef TRACERS_SPECIAL_Shindell
#ifdef INTERACTIVE_WETLANDS_CH4
          if(nread>0) call read_ncep_for_wetlands(end_of_day)
#endif

       case ('NOx') ! use : for sources, to include BB
         sfc_src(:,J_0:J_1,n,:)=tune_NOx*sfc_src(:,J_0:J_1,n,:)

       case ('Isoprene', 'Terpenes')
         sfc_src(:,J_0:J_1,n,1:ntsurfsrc(n))=
     &     tune_BVOC*sfc_src(:,J_0:J_1,n,1:ntsurfsrc(n))
#endif  /* TRACERS_SPECIAL_Shindell */

        case ('M_OCC_OC', 'OCII')
          if (.not.tracers_aerosols_soa) then
            if (ntsurfsrc(n)>1) then
              sfc_src(:,J_0:J_1,n,ntsurfsrc(n):
     &                            ntsurfsrc(n)+nBBsources(n))=
     &        sfc_src(:,J_0:J_1,n,ntsurfsrc(n)-1:
     &                            ntsurfsrc(n)+nBBsources(n)-1)
            endif ! else there is only the terpene source
            sfc_src(:,J_0:J_1,n,ntsurfsrc(n))=0.d0 ! this will become terpene sources
          endif

#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
        case ('dC17O')
          do ns=1,nread
            sfc_src(:,J_0:J_1,n,ns)=
     &        sfc_src(:,J_0:J_1,n,ns)*dCO_fact%dC17O_emis(ns)
          enddo
        case ('dC18O')
          do ns=1,nread
            sfc_src(:,J_0:J_1,n,ns)=
     &        sfc_src(:,J_0:J_1,n,ns)*dCO_fact%dC18O_emis(ns)
          enddo
        case ('d13CO')
          do ns=1,nread
            sfc_src(:,J_0:J_1,n,ns)=
     &        sfc_src(:,J_0:J_1,n,ns)*dCO_fact%d13CO_emis(ns)
          enddo
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
        end select

      end do ! ntm
!-------------------------------------------------------------------------------
! end tracers loop
!-------------------------------------------------------------------------------

! define ntsurfsrc for sulfate, AFTER reading and AFTER the tracers loop
      if (n_SO4>0) call set_ntsurfsrc(n_SO4,ntsurfsrc(n_SO2))
      if (n_M_ACC_SU>0) call set_ntsurfsrc(n_M_ACC_SU, ntsurfsrc(n_SO2))
      if (n_M_AKK_SU>0) call set_ntsurfsrc(n_M_AKK_SU, ntsurfsrc(n_SO2))
#ifdef TRACERS_TOMAS
        call set_ntsurfsrc(n_ASO4(1),ntsurfsrc(n_SO2))
#endif

#endif /* TRACERS_SPECIAL_Shindell || TRACERS_AEROSOLS_Koch || TRACERS_AMP || TRACERS_TOMAS || TRACERS_SPECIAL_Lerner || TRACERS_GASEXCH_GCC || TRACERS_PASSIVE */
!===============================================================================
! End of Chemistry/OMA/MATRIX/TOMAS/Some Lerner case
!===============================================================================

C**** Initialize tracers here to allow for tracers that 'turn on'
C**** at the start of any day
      call tracer_IC

#ifdef TRACERS_GASEXCH_GCC
      return
#endif
      if ((n_co2n>0).and.end_of_day) call adjust_co2

      return
      end subroutine daily_tracer

!gas exchange CO2 case reset trm here
!for the constCO2 case just reset to atmCO2 which is defined in the rundeck
!for the variable case (presently default) reset to the value scaled by
!the xnow value.
      subroutine adjust_co2
      use resolution, only : lm, psf
      use domain_decomp_atm, only : grid, globalsum, am_i_root
      use tracer_com, only: trm, trmom, n_co2n
      use oldtracer_mod, only: vol2mass
      use geom, only: axyp
      use constant, only: grav
      use model_com, only : nstep=>itime
      implicit none

      integer i,j,l,i_0, i_1, j_0, j_1, n
      real*8 :: sarea,
     &          trm_vert(GRID%I_STRT_HALO:GRID%I_STOP_HALO,
     &                   GRID%J_STRT_HALO:GRID%J_STOP_HALO),
     &          trm_glbavg,factor,atm_glbavg
      real*8 :: get_atmco2

      n=n_CO2n
      i_0=grid%i_strt
      i_1=grid%i_stop
      j_0=grid%j_strt
      j_1=grid%j_stop

         !area weighted tracer global average
      do j=J_0,J_1 ; do i=I_0,I_1
        trm_vert(i,j) = sum(trm(i,j,1:lm,n))*axyp(i,j) !@PL trm in kg CO2/m2, needs kg CO2 for factor
      enddo; enddo

      CALL GLOBALSUM(grid,axyp,    sarea,     all=.true.)
      CALL GLOBALSUM(grid,trm_vert,trm_glbavg,all=.true.)

         !total atm mass
      atm_glbavg = PSF*sarea*100.d0/grav

         !current concentration to new concentration
      factor = get_atmCO2()*atm_glbavg/trm_glbavg *vol2mass(n)*1.d-6
      if(AM_I_ROOT( ))then
        write(*,'(a,i5,8e12.4)')
     .           "TRACER_DRV, factor", nstep,factor,
     .            atm_glbavg,vol2mass(n),
     .            get_atmCO2(),sarea,
     .            PSF,grav,
     .            trm_glbavg/(atm_glbavg*vol2mass(n)*1.d-6)
      endif
#ifdef TRACERS_GASEXCH_CO2_DO_NOT_ADJUST
      return
#endif
      do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
        trm(i,j,l,n) = factor*trm(i,j,l,n)
      enddo; enddo; enddo

      if (factor .lt. 1.d0) then ! adjust moments
        do l=1,lm; do j=J_0,J_1; do i=I_0,I_1
          trmom(:,i,j,l,n)=factor*trmom(:,i,j,l,n)
             !?? do we need trmom=0 for the atmco2 case?
        enddo; end do; end do
      end if

      return
      end subroutine adjust_co2


#ifdef TRACERS_ON
      SUBROUTINE set_tracer_2Dsource
!@sum tracer_source calculates non-interactive sources for tracers
!@vers 2013/03/26
!@auth Jean Lerner/Gavin Schmidt
      USE MODEL_COM, only: itime,dtsrc,nday
      use SpecialIO_mod, only: write_parallel
      use TracerSurfaceSource_mod, only: TracerSurfaceSource,itsMEGAN
      use Attributes_mod
      use AttributeDictionary_mod
      use TracerBundle_mod
      use Tracer_mod, only: Tracer
      use model_com, only: modelEclock
      use RESOLUTION, only: im
      USE DOMAIN_DECOMP_ATM, only : GRID, GLOBALSUM,AM_I_ROOT
     *   ,globalmax, getDomainBounds
      USE FLUXES, only: fland,flice,focean,atmsrf
      USE SEAICE_COM, only : si_atm
      USE GHY_COM, only : fearth
      USE CONSTANT, only: tf,bygrav,mair,pi,teeny
      USE LAKES_COM, only : flake
      use OldTracer_mod, only: vol2mass
      use OldTracer_mod, only: trname
      use OldTracer_mod, only: tr_mm
      use OldTracer_mod, only: itime_tr0
      use OldTracer_mod, only: nBBsources
      use OldTracer_mod, only: do_fire
      use TimeConstants_mod, only: SECONDS_PER_DAY, INT_DAYS_PER_YEAR,
     &           SECONDS_PER_HOUR, HOURS_PER_DAY, INT_MONTHS_PER_YEAR
      USE ATM_COM, only: MA  ! Air mass of each box (kg m-2)
      USE TRACER_COM, only: ntm
#ifndef SKIP_TRACER_SRCS
      USE FLUXES, only: trsource
#endif
      use TRACER_COM, only: tracers
      USE RESOLUTION, only : pmtop,psf
      USE GEOM, only: axyp,areag,lat2d_dg,lon2d_dg,imaxj,lat2d
      USE QUSDEF
      USE TRACER_COM, only: sfc_src
      use TRACER_COM, only: n_isoprene, n_SO2, no_emis_over_ice
      use TRACER_COM, only: trm, ntsurfsrc, rnsrc
      use TRACER_COM, only: tracers
      use OldTracer_mod, only: itime_tr0, vol2mass, trname
#ifdef TRACERS_TOMAS
      use TRACER_COM, only: n_AH2O, n_AECOB
      use TRACER_COM, only: n_ANUM, n_AECIL, n_AOCIL, n_AOCOB
      use TRACER_COM, only: nbins, n_ASO4
#endif
#if (defined INTERACTIVE_WETLANDS_CH4) && (defined TRACERS_SPECIAL_Shindell)
      USE TRACER_SOURCES, only: ns_wet,add_wet_src
#endif
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS)
#ifndef TRACERS_AEROSOLS_SOA
      USE AEROSOL_SOURCES, only: OCT_src
#endif  /* TRACERS_AEROSOLS_SOA */
#endif
#if defined(TRACERS_RADON) || defined(TRACERS_SPECIAL_Lerner)
      use TRACER_COM, only: set_ntsurfsrc, ntsurfsrc
      USE tracers_radon_com, only: rn_src
#endif
#if (defined TRACERS_SPECIAL_Shindell)
      USE RAD_COM,  only : cosz1,cosz_day
#endif
#ifdef TRACERS_AEROSOLS_OCEAN
      use TRACERS_SEASALT, only: read_seawifs_chla
#endif  /* TRACERS_AEROSOLS_OCEAN */
#ifdef TRACERS_AMP
      USE AERO_SETUP, only : RECIP_PART_MASS
      USE TRDIAG_COM, only : taijs=>taijs_loc,ijts_AMPe
     &                       ,ijts_tauint
     &                       ,ijts_taustratint,ijts_reffstratint
     &                       ,ijts_2Dreff
#endif
#ifdef TRACERS_TOMAS
      USE TOMAS_EMIS, only : scalesizeSO4,scalesizeCARBO30
      USE TOMAS_AEROSOL, only: sqrt_xk_xk1
#endif
      use TracerHashMap_mod, only:
     &     TracerIterator, operator(/=)
      use Attributes_mod
      use AbstractAttribute_mod
      USE FILEMANAGER, only: openunit,closeunit
      USE Dictionary_mod, only: sync_param
#ifdef KLOVENSKI_DEV
      !=== temp do not push ===
#ifdef DO_MEGAN
      USE megan, only: acc_vcmax, acc_betadL
      use subdd_mod, only : inc_subdd
#endif
      !=== temp do not push ===
#endif
      implicit none
      integer :: i,j,ns,ns_isop,l,ky,n
      REAL*8 :: sarea,steppy,base,steppd,x,airm,anngas,
     *  tmon,bydt,tnew,fice
      REAL*8 :: sarea_prt(GRID%I_STRT_HALO:GRID%I_STOP_HALO,
     &                    GRID%J_STRT_HALO:GRID%J_STOP_HALO)
#ifdef TRACERS_SPECIAL_Shindell
c      real*8 :: factj(GRID%J_STRT_HALO:GRID%J_STOP_HALO)
c      real*8 :: nlight, max_COSZ1, fact0
#endif
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS)
!@var src_index source index for the current tracer
!@var src_fact source factor for the current tracer
      integer :: src_index,get_src_index
      real*8 :: src_fact
      interface
        real*8 function get_src_fact(n,is_bb,OA_not_OC)
          integer, intent(in) :: n
          logical, intent(in) :: is_bb
          logical, intent(in), optional :: OA_not_OC
        end function get_src_fact
      end interface
#endif

#ifdef TRACERS_TERP
!@param orvoc_fact Fraction of ORVOC added to Terpenes, for SOA production (Griffin et al., 1999)
      real*8, parameter :: orvoc_fact=0.32d0
      real*8 :: max_isop_flux
#endif  /* TRACERS_TERP */
      integer :: i_ocmip, iu_data
      real*8  :: factor
      real*8  :: trsource_prt(GRID%I_STRT_HALO:GRID%I_STOP_HALO,
     &                        GRID%J_STRT_HALO:GRID%J_STOP_HALO)
      real*8, dimension(NTM) :: trsource_glbavg
!@var ocmip_cfc: CFC-11 emissions estimated from OCMIP surf.conc.
      !60years (1939--1998) OCMIP surfc. concentr. converted to
      !global averaged emission rates
      !each value corresponds to the annual value
!      REAL*8, DIMENSION(:), allocatable, save :: ocmip_cfc
      INTEGER I_0, I_1, J_0, J_1, J_1S, J_0S
      class (Tracer), pointer :: pTracer
      integer :: index
      type (TracerSurfaceSource), pointer :: sources(:)
#ifdef TRACERS_TOMAS
      integer :: k, kn
      real*8 :: tot_emis(GRID%I_STRT:GRID%I_STOP,
     &     GRID%J_STRT:GRID%J_STOP)
#endif
      integer :: year, month, dayOfYear,hour,localTimeIndex

      type (TracerIterator) :: iter
      class (AbstractAttribute), pointer :: pa

      call modelEclock%get(year=year, month=month,
     *     dayOfYear=dayOfYear, hour=hour)
C****
C**** Extract useful local domain parameters from "grid"
C****
      call getDomainBounds(grid, J_STRT=J_0, J_STOP=J_1,
     &     I_STRT=I_0, I_STOP=I_1, J_STRT_SKP=J_0S, J_STOP_SKP=J_1S)

      bydt = 1./DTsrc
#ifdef TRACERS_TOMAS
#ifndef SKIP_TRACER_SRCS
        do k=1,nbins
           trsource(:,J_0:J_1,1,n_ANUM(k))=0.
           trsource(:,J_0:J_1,2,n_ANUM(k))=0.
           trsource(:,J_0:J_1,3,n_ANUM(k))=0.
        enddo
#endif
#endif
#ifdef DO_MEGAN
      ! Outside of tracer loop, call MEGAN-based biogenic emissions.
      ! Emissions will be experienced by any tracers with
      ! skipReason=itsMegan defined in their tracer source object. The
      ! source short name also must match one of the MEGAN-defined
      ! species. Call will fill sfc_src, to be added to trsource below.
      ! Let's skip the poles.
      do j=J_0S,J_1S
        do i=I_0,imaxj(j)
          ! Do we have to zero the polar boxes for 2:IM ??
          call biogenicEmissions_drv(i,j)
        end do
      end do
#ifdef KLOVENSKI_DEV
      !=== temp do not push ===
#ifdef CACHED_SUBDD
      call inc_subdd('vcmax_',acc_vcmax,2,.false.,
     & units='umol m-2 s-1',
     & long_name='MEGAN debug Vcmax variable')
      call inc_subdd('btran_',acc_betadL,2,.false.,
     & units='unknown',
     & long_name='MEGAN debug btran_megan variable')
#endif
      !=== temp do not push ===
#endif
#endif /* DO_MEGAN */
#ifdef TRACERS_ACETONE
      ! Outside of tracer loop, call routine to fill an ocean source
      ! for any tracer with skipReason=itsOcean defined in their source
      ! object and with source short name matching one defined in the
      ! routine (currently only acetone). Source can be positive or
      ! negative! Call will fill sfc_src, to be added to trsource below.
      do j=J_0S,J_1S
        do i=I_0,imaxj(j)
          ! Do we have to zero the polar boxes for 2:IM ??
          call oceanEmissions_drv(i,j)
        end do
      end do
#endif /* TRACERS_ACETONE */

C**** All sources are saved as kg s-1
      iter = tracers%begin()
      do while (iter /= tracers%last())
        pTracer => iter%value()
        pa => pTracer%getReference('index')
        index = pa
        n = index
        pTracer => tracers%getReference(trname(n))
        sources => pTracer%surfaceSources

        if (itime.lt.itime_tr0(n)) then
          call iter%next()
          cycle
        end if

#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS)
      src_index=get_src_index(n)
      src_fact=get_src_fact(n,.false.)
#endif

      select case (trim(pTracer%getName()))

      case default
!     write(6,*) ' Sources for ',trim(pTracer%getName()),' are not in this routine'
C****
C**** Surface Sources of SF6 and CFCn (Same grid as CFC11)
C****
      case ('SF6','CFC11','CFCn','SF6_c')
#ifndef SKIP_TRACER_SRCS
        trsource(:,:,:,n)=0
#endif
C**** SF6 source increases each year by .3pptv/year
C**** SF6_c source is constant, same as first year SF6, but always
C**** CFCn source increases each year so that the glbavg is from obs
C**** CFC source is the same each year
C**** Distribute source over ice-free land
        steppy = 1./(SECONDS_PER_DAY*INT_DAYS_PER_YEAR)
        if (trim(pTracer%getName()).eq.'SF6' .or.
     *      trim(pTracer%getName()).eq.'CFCn' .or.
     *      trim(pTracer%getName()).eq.'SF6_c') then
C         Make sure index KY=1 in year that tracer turns on
          ky = 1 + (itime-itime_tr0(n))/(nday*INT_DAYS_PER_YEAR)
          if (trim(pTracer%getName()).eq.'SF6_c') ky = 1
          base = (0.3d-12)*vol2mass(n) !pptm
          x = base*ky
          airm = (psf-pmtop)*100.*bygrav*AREAG !(kg/m**2 X m**2 = kg)
          anngas = x*airm
        else if (trim(pTracer%getName()).eq.'CFC11') then
          anngas = 310.d6
        endif

c Could the masks for latlon rectangles be precomputed at
c initialization and their areas saved? (Or do whenever
c fearth changes.)

C**** Source over United States and Canada
        call regional_src(n, .37d0*anngas*steppy,
     &                    -70.d0, -125.d0, 50.d0, 30.d0)
C**** Source over Europe and Russia
        call regional_src(n, .37d0*anngas*steppy,
     &                    45.d0, -10.d0, 65.d0, 36.1d0) ! 0.1 deg offset avoids overlap with Middle East
C**** Source over Far East
        call regional_src(n, .13d0*anngas*steppy,
     &                    150.d0, 120.d0, 45.d0, 20.d0)
C**** Source over Middle East
        call regional_src(n, .05d0*anngas*steppy,
     &                    75.d0, 30.d0, 35.9d0, 15.d0) ! 0.1 deg offset avoids overlap with Europe
C**** Source over South America
        call regional_src(n, .04d0*anngas*steppy,
     &                    -40.d0, -50.d0, -22.5d0, -23.5d0)
C**** Source over South Africa
        call regional_src(n, .02d0*anngas*steppy,
     &                    30.d0, 25.d0, -24.d0, -28.d0)
C**** Source over Australia and New Zealand
        call regional_src(n, .02d0*anngas*steppy,
     &                    150.5d0, 149.5d0, -33.5d0, -34.5d0)

        if (trim(pTracer%getName()).eq.'CFCn') then
          !print out global average for each time step before weighing
          !in the OCMIP values
          sarea  = 0.
          trsource_glbavg(n)=0.
          sarea_prt(:,:)  = 0.
          trsource_prt(:,:) = 0.
          do j=J_0,J_1
            do i=I_0,I_1
             factor = axyp(i,j)*fearth(i,j)
             sarea_prt(i,j)= FACTOR
#ifndef SKIP_TRACER_SRCS
             trsource_prt(i,j) = trsource(i,j,1,n)*FACTOR
#endif
            enddo
          enddo

          CALL GLOBALSUM(grid, sarea_prt,    sarea, all=.true.)
          CALL GLOBALSUM(grid,trsource_prt,trsource_glbavg(n),
     &                                                all=.true.)

          trsource_glbavg(n)=trsource_glbavg(n)/sarea 
          !weight trsource by ocmip_cfc global average
          !number of steps/year=INT_DAYS_PER_YEAR*SECONDS_PER_DAY/dtsrc
          !                    =365*86400/1800 =17520
!         if (.not.allocated(ocmip_cfc)) then
!           !read in OCMIP based CFC-11 global emissions
!           !=sum(dC/dt) for each hemisphere
!           !these are *annual global averages* and need to be
!           !converted to our timestep value
!           allocate(ocmip_cfc(67))
!           print*, 'opening file=OCMIP_cfc.dat'
!           call openunit('OCMIP_cfc',iu_data,.false.,.true.)
!           do i=1,67
!             read(iu_data,'(5x,e12.4)')ocmip_cfc(i)
!           enddo
!           call closeunit(iu_data)
!         endif
!         i_ocmip=(itime-itime_tr0(n))/INT_DAYS_PER_YEAR/
!    &            int(SECONDS_PER_DAY/dtsrc)+1
!         if (mod(itime,INT_DAYS_PER_YEAR*int(SECONDS_PER_DAY/dtsrc))
!    &        .eq. 0.) then
!           write(6,'(a,2i5)'),'TRACERS_DRV, new year: itime, i_ocmip=',
!    &                         itime,i_ocmip
!         endif
!#ifndef SKIP_TRACER_SRCS
!          do j=J_0,J_1 ! TNL
!            do i=1,72
!               trsource(i,j,1,n) = trsource(i,j,1,n)*
!     &           (ocmip_cfc(i_ocmip)/(INT_DAYS_PER_YEAR*
!     &           SECONDS_PER_DAY/dtsrc)) / trsource_glbavg(n)
!            enddo
!          enddo
!#endif

          !recompute global average after weighting in OCMIP
          sarea  = 0.
          trsource_glbavg(n)=0.
          sarea_prt(:,:)  = 0.
          trsource_prt(:,:) = 0.
          do j=J_0,J_1
            do i=I_0,I_1
             factor = axyp(i,j)*fearth(i,j)
             sarea_prt(i,j)= FACTOR
#ifndef SKIP_TRACER_SRCS
             trsource_prt(i,j) = trsource(i,j,1,n)*FACTOR
#endif
            enddo
          enddo

          CALL GLOBALSUM(grid, sarea_prt,    sarea, all=.true.)
          CALL GLOBALSUM(grid,trsource_prt,trsource_glbavg(n),
     &                                                 all=.true.)

          trsource_glbavg(n)=trsource_glbavg(n)/sarea

        endif  ! pTracer==CFCn

#ifdef TRACERS_PASSIVE
       case ('nh5','nh15','nh50','e90')
         trsource(:,:,:,n)=0.d0
#endif

#if defined(TRACERS_RADON) || defined(TRACERS_SPECIAL_Lerner)
C****
C**** Surface Sources for Radon-222
C****
      case ('Rn222')
#ifndef SKIP_TRACER_SRCS
        call set_ntsurfsrc(n,1)
        trsource(:,J_0:J_1,:,n)=0.d0
C**** ground source
        steppd=1.d0/SECONDS_PER_DAY
        do j=J_0,J_1
          do i=I_0,I_1
            if (rnsrc.eq.0) then !standard source
C**** source from ice-free land
              if(atmsrf%tsavg(i,j).lt.tf) then !composite surface air temperature
                trsource(i,j,1,n)=
     &            1.0d-16*steppd*axyp(i,j)*fearth(i,j)
              else  ! 1 atom/cm^2/s
                trsource(i,j,1,n)=
     &            3.2d-16*steppd*axyp(i,j)*fearth(i,j)
              end if
            else if (rnsrc.eq.1) then !Conen and Robertson
              trsource(i,j,1,n)=
     &          3.2d-16*steppd*axyp(i,j)*fearth(i,j)
c add code to implement Conen and Robertson - linear decrease in Rn222
c   emission from 1 at 30N to 0.2 at 70N and 0.2 north of 70N
              if (nint(lat2d_dg(i,j)).gt.30 .and.
     &            nint(lat2d_dg(i,j)).lt.70) then
                trsource(i,j,1,n)=trsource(i,j,1,n)
     &            *(1.d0-(lat2d_dg(i,j)-30.d0)/40.d0*0.8d0)
              else if (nint(lat2d_dg(i,j)).ge.70) then
                trsource(i,j,1,n)=
     &            0.2*trsource(i,j,1,n)
              endif
            else if (rnsrc.eq.2) then !Schery and Wasiolek
c Schery source
              trsource(i,j,1,n)=rn_src(i,j,month)
            endif
            if (rnsrc.le.1) then
C**** source from ice-free ocean
              trsource(i,j,1,n)=trsource(i,j,1,n)
     &          +1.6d-18*steppd*axyp(i,j)
     &          *(1.-fland(i,j))*(1.-si_atm%rsi(i,j))
            endif
          enddo                 !i
        enddo                   !j
#endif
#endif

#ifdef TRACERS_SPECIAL_Lerner
C****
C**** Sources and sinks for CO2 and CH4 (kg s-1)
C****
      case ('CO2','CH4')
        do ns=1,ntsurfsrc(n) 
          do j=J_0,J_1
            trsource(I_0:I_1,j,ns,n)=
     &      sfc_src(I_0:I_1,j,n,ns)
          end do
        end do

C****
C**** Sources and sinks for N2O:
C**** First layer is set to a constant 462.2 ppbm. (300 PPB V)
C****
      case ('N2O')
      do j=J_0,J_1
        trsource(:,j,1,n) = (MA(1,:,j)*462.2d-9-trm(:,j,1,n))*bydt
      end do
C****
C**** Linoz Deposition from layer 1
C****
      case ('O3')
#ifndef LINOZ_TRDRYDEP
      call linoz_depo(1,n)
#endif 
#endif
C****
C**** Sources and sinks for 14CO2
C**** NOTE: This tracer is supposed to start on 10/16
C**** Decay is a function of the number of months since itime_tr0
C**** The tracer is reset to specific values in layer 1 only if
C****   this results in a sink
C****
      case ('14CO2')
#ifndef SKIP_TRACER_SRCS
      tmon = (itime-itime_tr0(n))*INT_MONTHS_PER_YEAR/
     &       (nday*INT_DAYS_PER_YEAR)
      trsource(:,J_0:J_1,1,n) = 0.
      do j=J_0,J_1
      do i=I_0,I_1
         if (lat2d(i,j).lt.0.) then
               tnew = MA(1,i,j)*(4.82d-18*tr_mm(n)/mair)*
     *          (44.5 + tmon*(1.02535d0 - tmon*
     *                  (2.13565d-2 - tmon*8.61853d-5)))
         else
               tnew = MA(1,i,j)*(4.82d-18*tr_mm(n)/mair)*
     *          (73.0 - tmon*(0.27823d0 + tmon*
     *                  (3.45648d-3 - tmon*4.21159d-5)))
         endif
         if (tnew.lt.trm(i,j,1,n))
     *     trsource(i,j,1,n) = (tnew-trm(i,j,1,n))*bydt
      end do
      end do
#endif

C****
C**** No non-interactive surface sources of Water
C****
      case ('Water')
#ifndef SKIP_TRACER_SRCS
        trsource(:,J_0:J_1,:,n)=0.d0
#endif

#ifdef TRACERS_SPECIAL_Shindell
      case ('Ox','NOx','ClOx','BrOx','N2O5','HNO3','H2O2','CH3OOH',
     &      'HCHO','HO2NO2','CO','PAN','AlkylNit','Alkenes','Paraffin',
#ifdef TRACERS_ACETONE
     &      'Acetone',
#endif
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
#ifdef TRACERS_dCO
     *      'd13Calke','d13CPAR',
     *      'd17OPAN','d18OPAN','d13CPAN',
     *      'dMe17OOH', 'dMe18OOH', 'd13MeOOH',
     *      'dHCH17O', 'dHCH18O', 'dH13CHO',
#endif  /* TRACERS_dCO */
     *      'dC17O', 'dC18O', 'd13CO',
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
     &      'HCl','HOCl','ClONO2','HBr','HOBr','BrONO2','N2O','CFC')
#ifdef CALCULATE_FLAMMABILITY
        if(do_fire(n))then
          call dynamic_biomass_burning(n,ntsurfsrc(n)+nBBsources(n)+1)
        endif
#endif  /* CALCULATE_FLAMMABILITY */
        do ns=1,ntsurfsrc(n); do j=J_0,J_1
          trsource(I_0:I_1,j,ns,n)=
     &    sfc_src(I_0:I_1,j,n,ns)
        end do ; end do
#ifdef TRACERS_TERP
      case ('Terpenes')
        do ns=1,ntsurfsrc(n)
          if(ns==1) then
            do j=J_0,J_1
              trsource(I_0:I_1,j,ns,n)=
     &        sfc_src(I_0:I_1,j,n,ns)
            end do
! If no orvoc file provided, scale up the terpenes one instead.
! 0.4371 is the ratio of orvoc/isoprene emissions in the Lathiere et al. (2005) results
            if(ntsurfsrc(n)==1) then ! no orvoc file exists
              call globalmax(grid,
     &             maxval(sfc_src(I_0:I_1,J_0:J_1,n_Isoprene,
     &                            1:ntsurfsrc(n_Isoprene))),
     &             max_isop_flux)
              if (max_isop_flux <= 0.d0) call stop_model(
     &          'Offline isoprene sources are needed', 255)
              do ns_isop=1,ntsurfsrc(n_Isoprene) ! use all Isoprene sources for orvoc scaling
                do j=J_0,J_1
                  trsource(I_0:I_1,j,ns,n)=trsource(I_0:I_1,j,ns,n)+
     &            orvoc_fact*0.4371*
     &            sfc_src(I_0:I_1,j,n_Isoprene,ns_isop)
                end do
              end do
            end if
          else ! use the orvoc file
            do j=J_0,J_1
              trsource(I_0:I_1,j,ns,n)=orvoc_fact*
     &        sfc_src(I_0:I_1,j,n,ns)
            end do
          endif
        end do
#endif  /* TRACERS_TERP */
      case ('CH4')
#ifdef CALCULATE_FLAMMABILITY
        if(do_fire(n))then
          call dynamic_biomass_burning(n,ntsurfsrc(n)+nBBsources(n)+1)
        endif
#endif  /* CALCULATE_FLAMMABILITY */
        do ns=1,ntsurfsrc(n); do j=J_0,J_1
          trsource(I_0:I_1,j,ns,n)=
     &    sfc_src(I_0:I_1,j,n,ns)
        end do ; end do
#ifdef INTERACTIVE_WETLANDS_CH4
        if(ntsurfsrc(n) > 0) then
          call alter_wetlands_source(n,ns_wet)
          do j=J_0,J_1
            trsource(I_0:I_1,j,ns_wet,n)=trsource(I_0:I_1,j,ns_wet,n)+
     &      add_wet_src(I_0:I_1,j)
          enddo
        endif
#endif
#if !defined(PS_BVOC) && !defined(BIOGENIC_EMISSIONS)
      ! Isoprene sources to be emitted only during sunlight, and
      ! weighted by cos of solar zenith angle. Unless it is a
      ! MEGAN source, where daylight already handled:
      case ('Isoprene')
        do ns=1,ntsurfsrc(n)
          if(pTracer%surfaceSources(ns)%skipReason==itsMEGAN) then
            do j=J_0,J_1; do i=I_0,I_1
              trsource(i,j,ns,n)=sfc_src(i,j,n,ns)
            end do ; end do
          else ! not MEGAN
            do j=J_0,J_1; do i=I_0,I_1
              if(COSZ1(i,j)>0.)then
                trsource(i,j,ns,n)=(COSZ1(i,j)/(COSZ_day(i,j)+teeny))*
     &          sfc_src(i,j,n,ns)
              else
                trsource(i,j,ns,n)=0.d0
              endif
            end do  ; end do
          end if
        end do
#endif /* not PS_BVOC and not BIOGENIC_EMISSIONS */
#endif /* TRACERS_SPECIAL_Shindell */

#ifdef TRACERS_AEROSOLS_OCEAN
      case ('OCocean')
        call read_seawifs_chla(month) ! CHECK this has to be called once per month, not every timestep
#endif  /* TRACERS_AEROSOLS_OCEAN */

#ifdef TRACERS_PASSIVE
      case ('CO50')
        do ns=1,ntsurfsrc(n)
          trsource(I_0:I_1,J_0:J_1,ns,n)=
     &      sfc_src(I_0:I_1,J_0:J_1,n,ns)*axyp(I_0:I_1,J_0:J_1)
        end do
#endif  /* TRACERS_PASSIVE */

#ifndef TRACERS_AEROSOLS_SOA
#ifdef TRACERS_TOMAS
        case ('SOAgas')
!OCT_src is kg/m2/month? or kg/m2/sec?? 
        do j=J_0,J_1; do i=I_0,I_1
           trsource(i,j,ntsurfsrc(n),n)=OCT_src(i,j,month)
         end do; enddo
#endif
#endif  /* TRACERS_AEROSOLS_SOA */
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS)
      case ('SO2', 'SO4', 'M_ACC_SU', 'M_AKK_SU',
     &      'BCII', 'BCB', 'OCII', 'OCB',
     &      'vbsAm2', 'vbsAm1', 'vbsAz', 'vbsAp1', 'vbsAp2',
     &      'vbsAp3', 'vbsAp4', 'vbsAp5', 'vbsAp6',
     &      'M_BC1_BC', 'M_OCC_OC', 'M_BOC_BC', 'M_BOC_OC',
     &      'M_OCC_OCM2','M_OCC_OCM1','M_OCC_OCM0',
     &      'M_OCC_OCP1','M_OCC_OCP2','M_OCC_OCP3',
     &      'M_OCC_OCP4','M_OCC_OCP5','M_OCC_OCP6',
     &      'ASO4__01','AOCOB_01','AECOB_01')

#ifdef CALCULATE_FLAMMABILITY
        if(do_fire(n))then
          call dynamic_biomass_burning(n,ntsurfsrc(n)+nBBsources(n)+1)
        endif
#endif  /* CALCULATE_FLAMMABILITY */

#ifndef TRACERS_AEROSOLS_SOA
        select case (trim(pTracer%getName()))
        case ('OCII', 'M_OCC_OC')
          sfc_src(:,J_0:J_1,src_index,ntsurfsrc(n))=
     &      OCT_src(:,J_0:J_1,month)/src_fact
        end select
#endif  /* TRACERS_AEROSOLS_SOA */

        do ns=1,ntsurfsrc(src_index)
          trsource(:,J_0:J_1,ns,n)=
     &      sfc_src(:,J_0:J_1,src_index,ns)
     &      *src_fact

#ifdef TRACERS_TOMAS
!ntsurfsrc(3) for number

!     ns=1 : SO4 number
!     ns=2 : EC number
!     ns=3 : OC number
        tot_emis(:,J_0:J_1)=0.0
          if(n.eq.n_ASO4(1))then
             tot_emis(:,J_0:J_1)= trsource(:,J_0:J_1,ns,n_ASO4(1))

             do k=1,nbins
                trsource(:,J_0:J_1,ns,n_ASO4(k))=
     &              tot_emis(:,J_0:J_1)*scalesizeSO4(k)

                trsource(:,J_0:J_1,1,n_ANUM(k))=
     &           trsource(:,J_0:J_1,1,n_ANUM(k)) +
     &               trsource(:,J_0:J_1,ns,n_ASO4(k))
     &               /sqrt_xk_xk1(k)    
              enddo


          elseif(n.eq.n_AECOB(1))then

             tot_emis(:,J_0:J_1)= trsource(:,J_0:J_1,ns,n_AECOB(1))

             do k=1,nbins
                trsource(:,J_0:J_1,ns,n_AECOB(k))=
     &               tot_emis(:,J_0:J_1)*scalesizeCARBO30(k)*0.8

                trsource(:,J_0:J_1,ns,n_AECIL(k))=
     &               tot_emis(:,J_0:J_1)*scalesizeCARBO30(k)*0.2

                trsource(:,J_0:J_1,2,n_ANUM(k))=
     &           trsource(:,J_0:J_1,2,n_ANUM(k)) +
     &             ( trsource(:,J_0:J_1,ns,n_AECOB(k))+
     &                 trsource(:,J_0:J_1,ns,n_AECIL(k)))
     &             /sqrt_xk_xk1(k)  
             enddo
          elseif(n.eq.n_AOCOB(1))then

             tot_emis(:,J_0:J_1)= trsource(:,J_0:J_1,ns,n_AOCOB(1))

             do k=1,nbins
                trsource(:,J_0:J_1,ns,n_AOCOB(k))=
     &               tot_emis(:,J_0:J_1)*scalesizeCARBO30(k)*0.5

                trsource(:,J_0:J_1,ns,n_AOCIL(k))=
     &               tot_emis(:,J_0:J_1)*scalesizeCARBO30(k)*0.5

                trsource(:,J_0:J_1,3,n_ANUM(k))=
     &           trsource(:,J_0:J_1,3,n_ANUM(k)) +
     &              ( trsource(:,J_0:J_1,ns,n_AOCOB(k))+
     &                trsource(:,J_0:J_1,ns,n_AOCIL(k)))
     &            /sqrt_xk_xk1(k)  
             enddo
          endif

#endif
        enddo ! ns
#if (defined TRACERS_NITRATE) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS)
      case ('NH3')
#ifdef CALCULATE_FLAMMABILITY
        if(do_fire(n))then
          call dynamic_biomass_burning(n,ntsurfsrc(n)+nBBsources(n)+1)
        endif
#endif  /* CALCULATE_FLAMMABILITY */
        do ns=1,ntsurfsrc(n)
          trsource(:,J_0:J_1,ns,n)=sfc_src(:,J_0:J_1,n,ns)
        enddo

#endif /* TRACERS_NITRATE || TRACERS_AMP || TRACERS_TOMAS */
#endif /* (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) || (defined TRACERS_TOMAS) || (defined TRACERS_GASEXCH_GCC) */

! Addition of surface emission sources for GCC
#ifdef TRACERS_GASEXCH_GCC
      case ('CO2n')
        do ns=1,ntsurfsrc(n)
          do j=J_0,J_1
#ifdef GCC_ZERO_EMISSION
            trsource(I_0:I_1,j,ns,n)=sfc_src(I_0:I_1,j,n,ns)
     &                              * 0.0
!    &                              *axyp(I_0:I_1,j)   !should be commented out for tracersE3 runs
#else
            trsource(I_0:I_1,j,ns,n)=sfc_src(I_0:I_1,j,n,ns)
!    &                              *axyp(I_0:I_1,j)   !should be commented out for tracersE3 runs
#endif
          enddo
        enddo
#endif
      end select

#ifndef SKIP_TRACER_SRCS
      ! The diurnal cycle application:
      do ns=1,ntsurfsrc(n)                    ! loop over sources
        if(sources(ns)%applyDiurnalCycle)then ! does source have a diurnal cycle defined?
          ! this might not work if there aren't an equal number of timesteps each hour:
          if(MOD(SECONDS_PER_HOUR,dtsrc).ne.0.d0)then
            call write_parallel('Diurnal emissions steps/hr problem')
            call stop_model('Problem w/ emissions diurnal cycle 2',255)
          end if
          do j=J_0,J_1                        ! loop horizontal space
            do i=I_0,imaxj(j)
              ! intending here for localTimeIndex an integer index ranging from 1 to INT_HOURS_PER_DAY
              localTimeIndex=(hour+1)
     &            +NINT((i-(IM+1)/2.)*HOURS_PER_DAY/float(IM))
              if(localTimeIndex>HOURS_PER_DAY)
     &            localTimeIndex=localTimeIndex-HOURS_PER_DAY
              if(localTimeIndex<1)
     &            localTimeIndex=localTimeIndex+HOURS_PER_DAY
              trsource(i,j,ns,n)=trsource(i,j,ns,n)*
     &            sources(ns)%diurnalCycle(localTimeIndex)
            end do     ! i
          end do       ! j
        end if         ! this source has a diurnal cycle
      end do           ! sources

      ! Optionally set sources to zero over (>90%) ice:
      if(no_emis_over_ice > 0)then
        do j=J_0,J_1
          do i=I_0,imaxj(j)
            fice=flice(i,j)+si_atm%rsi(i,j)*(focean(i,j)+flake(i,j))
            if(fice > 0.9d0) trsource(i,j,:,n)=0.d0
          end do
        end do
      end if
#endif

      call iter%next()
      end do ! n - main tracer loop

      END SUBROUTINE set_tracer_2Dsource

      subroutine regional_src(n,source,lon_e,lon_w,lat_n,lat_s)
!@sum Assign regional 2d sources
!@auth Kostas Tsigaridis, based on old Lerner code
        use DOMAIN_DECOMP_ATM, only : globalsum,grid,getDomainBounds
        use GEOM, only: axyp,byaxyp
        use GHY_COM, only : fearth
#ifndef SKIP_TRACER_SRCS
        use FLUXES, only: trsource
#endif
        implicit none
        integer, intent(in) :: n
        real*8, intent(in) :: source, lon_e, lon_w, lat_n, lat_s
        real*8 :: sarea_prt(grid%I_STRT_HALO:grid%I_STOP_HALO,
     &                      grid%J_STRT_HALO:grid%J_STOP_HALO)
        real*8 :: sarea
        integer :: i_0,i_1,j_0,j_1
        integer :: i,j

        call getDomainBounds(grid, I_STRT=I_0, I_STOP=I_1,
     &                             J_STRT=J_0, J_STOP=J_1)

        call get_latlon_mask(lon_w,lon_e,lat_s,lat_n,sarea_prt)
        do j=j_0,j_1; do i=i_0,i_1
          sarea_prt(i,j) = sarea_prt(i,j)*axyp(i,j)*fearth(i,j)
        enddo; enddo
        call globalsum(grid, sarea_prt, sarea, all=.true.)
#ifndef SKIP_TRACER_SRCS
        do j=j_0,j_1; do i=i_0,i_1
            trsource(i,j,1,n) = trsource(i,j,1,n) +
     &         source*sarea_prt(i,j)*byaxyp(i,j)/sarea
        enddo; enddo
#endif
      end subroutine regional_src

      subroutine get_latlon_mask(lon_w,lon_e,lat_s,lat_n,latlon_mask)
!@sum Set mask array to 1 for all cells overlapping a lat-lon rectangle
!@auth Kelley
      use domain_decomp_atm, only : getDomainBounds,grid
#ifdef CUBED_SPHERE
      use geom, only : lon2d_dg,lat2d_dg
#else
      use geom, only : lon_to_i,lat_to_j
#endif
      implicit none
      real*8 :: lon_w,lon_e,lat_s,lat_n
      real*8, dimension(grid%i_strt_halo:grid%i_stop_halo,
     &                  grid%j_strt_halo:grid%j_stop_halo) ::
     &     latlon_mask
      integer :: i,j, i_0,i_1,j_0,j_1
      integer :: ie,iw,js,jn

      call getDomainBounds(grid, I_STRT=I_0, I_STOP=I_1,
     &     J_STRT=J_0, J_STOP=J_1)

      latlon_mask(:,:) = 0d0

#ifdef CUBED_SPHERE
      do j=j_0,j_1
      do i=i_0,i_1
        if(lon2d_dg(i,j) >= lon_w .and.
     &     lon2d_dg(i,j) <= lon_e .and.
     &     lat2d_dg(i,j) >= lat_s .and.
     &     lat2d_dg(i,j) <= lat_n) then
          latlon_mask(i,j) = 1d0
        endif
      enddo
      enddo
c The above approach to defining masks does not work for latlon
c rectangles whose lon and/or lat lengths are less than the
c gridsize.  Solution: if a rectangle is narrow in either
c dimension, trace a discretized path from its SW to its NE
c corner and call lonlat_to_ij for each latlon point along the
c path, setting the mask to 1 for the returned i,j that are in
c the local domain.  Or, trace all 4 edges of the rectangle.
c      if(narrow rectangle) then
c        lon = lon_w
c        lat = lat_s
c        do point=1,npoints_traverse
c          lon = lon + (lon_e-lon_w)/npoints_traverse
c          lat = lat + (lat_n-lat_s)/npoints_traverse
c          call lonlat_to_ij((/lon,lat/),ij)
c          if(i,j in local domain) then
c            latlon_mask(i,j) = 1d0
c          endif
c        enddo
c      endif
#else
c latlon grid
      ie = lon_to_i(lon_e)
      iw = lon_to_i(lon_w)
      jn = lat_to_j(lat_n)
      js = lat_to_j(lat_s)
      do j=max(js,j_0),min(jn,j_1)
        latlon_mask(iw:ie,j) = 1d0
      enddo
#endif
      return
      end subroutine get_latlon_mask

#ifdef TRACERS_SPECIAL_Lerner
      subroutine lernerchem_prep
      use OldTracer_mod, only: itime_tr0
      USE TRACER_COM, only: n_CH4
      USE MODEL_COM, only: itime
      implicit none

C****CH4
      if(itime.ge.itime_tr0(n_CH4)) then
        call Trop_chem_CH4_prep
      end if
      end subroutine lernerchem_prep

      subroutine calculate_and_apply_lerner(i,j)
      use OldTracer_mod, only: itime_tr0
      USE TRACER_COM, only: n_CH4, n_O3, n_N2O, n_CFC11
      use TRACER_COM, only: nTropCH4, nStratCH4
      use TRACER_COM, only: nTropO3P, nTropO3L, nStratO3
      use TRACER_COM, only: nChemistry
      USE MODEL_COM, only: itime
      USE apply3d, only : apply_tracer_3Dsource
      implicit none
      integer, intent(in) :: i,j
      INTEGER ns,xday   ; real*8 now
      REAL*8 factor 

C****CH4
      if(itime.ge.itime_tr0(n_CH4)) then
        call Trop_chem_CH4(i,j,nTropCH4,n_CH4)
        call apply_tracer_3Dsource(i,j,nTropCH4,n_CH4)
        call Strat_chem_Prather(i,j,nStratCH4,n_CH4)
        call apply_tracer_3Dsource(i,j,nStratCH4,n_CH4,.false.)
      end if
C****O3
      if(itime.ge.itime_tr0(n_O3)) then
        call Trop_chem_O3(i,j,nTropO3P,nTropO3L,n_O3)
        call apply_tracer_3Dsource(i,j,nTropO3P,n_O3,.false.)
        call apply_tracer_3Dsource(i,j,nTropO3L,n_O3,.false.)
        call Strat_chem_O3(i,j,nStratO3,n_O3)
        call apply_tracer_3Dsource(i,j,nStratO3,n_O3,.false.)
      end if
C****N2O
      if(itime.ge.itime_tr0(n_N2O)) then
        call Strat_chem_Prather(i,j,nChemistry,n_N2O)
        call apply_tracer_3Dsource(i,j,nChemistry,n_N2O,.FALSE.)
      end if
C****CFC11
      if(itime.ge.itime_tr0(n_CFC11)) then
        call Strat_chem_Prather(i,j,nChemistry,n_CFC11)
        call apply_tracer_3Dsource(i,j,nChemistry,n_CFC11,.FALSE.)
      end if
C****

      end subroutine calculate_and_apply_lerner
#endif

#ifdef TRACERS_PASSIVE
      subroutine calculate_and_apply_passive(i,j)
      use TimeConstants_mod, only: SECONDS_PER_DAY
      use OldTracer_mod, only: itime_tr0
      USE MODEL_COM, only: itime
      USE RESOLUTION, only : lm
      use model_com, only: modelEclock
      USE TRACER_COM, only: n_nh5, n_nh50, n_nh15, n_e90, n_CO50
      USE TRACER_COM, only: n_aoa, n_aoanh, n_st8025, n_tape_rec
      use TRACER_COM, only: nChemistry
      use TRACER_COM, only: nOverwrite
      use TRACER_COM, only: CO50_yield_from_CH4
      use OldTracer_mod, only: vol2mass
      USE FLUXES, only: tr3Dsource
      USE ATM_COM, only: pmidl00
      USE TRACER_COM, only: trm_col
      USE MODEL_COM, only: dtsrc
      USE apply3d, only : apply_tracer_3Dsource
      USE ATM_COM, only: LTROPO
      use atmcol_com, only: ma   ! layer mass (kg/m2)
      USE GEOM, only: axyp, lat2d_dg 
      USE CONSTANT, only : pi 
      implicit none
      integer, intent(in) :: i,j
      integer :: l
      integer :: year, month, dayOfYear
      INTEGER ns,xday   ; real*8 now
      REAL*8 factor 
      real*8 :: bydt,expdecst8025
      integer :: layer80hPa


C**** NH5, NH50 and NH15: idealized loss tracers set over NH midlatitudes (30N-50N)

      bydt = 1./dtsrc 

      if(n_nh5 > 0) then
      if(itime.ge.itime_tr0(n_nh5)) then

        if (nint(lat2d_dg(i,j)).gt.30 .and.
     &      nint(lat2d_dg(i,j)).lt.50) then
          tr3Dsource(1,nOverwrite,n_nh5) = (ma(1)*100.0d-9 !ma ~ kg/m2 
     *          -trm_col(1,n_nh5))*bydt  !tr3Dsource ~ kg/m2/s 
        endif
        call apply_tracer_3Dsource(i,j,nOverwrite,n_nh5,.FALSE.)

      endif
      endif

      if(n_nh15 > 0) then
      if(itime.ge.itime_tr0(n_nh15)) then

        if (nint(lat2d_dg(i,j)).gt.30 .and.
     &      nint(lat2d_dg(i,j)).lt.50) then
          tr3Dsource(1,nOverwrite,n_nh15) = (ma(1)*100.0d-9
     *          -trm_col(1,n_nh15))*bydt
        endif
        call apply_tracer_3Dsource(i,j,nOverwrite,n_nh15,.FALSE.)

      endif
      endif

      if(n_nh50 > 0) then
      if(itime.ge.itime_tr0(n_nh50)) then

        if (nint(lat2d_dg(i,j)).gt.30 .and.
     &      nint(lat2d_dg(i,j)).lt.50) then
          tr3Dsource(1,nOverwrite,n_nh50) = (ma(1)*100.0d-9
     *          -trm_col(1,n_nh50))*bydt
        endif
        call apply_tracer_3Dsource(i,j,nOverwrite,n_nh50,.FALSE.)

      endif
      endif

C**** E90: idealized loss tracer set over entire surface layer

      if(n_e90 > 0) then
      if(itime.ge.itime_tr0(n_e90)) then

        tr3Dsource(1,nOverwrite,n_e90) = (ma(1)*100.0d-9
     *          -trm_col(1,n_e90))*bydt
        call apply_tracer_3Dsource(i,j,nOverwrite,n_e90,.FALSE.)

      endif
      endif

! CO50 tracer, like CO with a 50-day lifetime. Call it after masterchem.
      if (n_CO50>0) then
        tr3Dsource(:,nChemistry,n_CO50)=
     &    CO50_yield_from_CH4*1760.d-9   ! Assume 1760 vmr CH4
     &    *vol2mass(n_CO50) ! convert to mmr
     &    *MA(:)            ! convert to kg m-2
     &    *3.73d-9          ! lifetime of 8.5 years (first order loss [s-1])
        call apply_tracer_3Dsource(i,j,nChemistry,n_CO50,.FALSE.)
      endif

C****AOANH and AOA: Two mean age tracers are defined, one with respect to the NH midlatitude surface and one with respect to the Earth's surface.  In lieu of the "clock-tracer" implementation (see GLT tracer), here we solve for the mean age as the solution to d(G)dt=1, where G is the mean age and d/dt is the advective derivative. Zero boundary conditions are enforced over the source region (either NH midlatitude surface layer (30N-50N) or the entire Earth's surface). Units are in days.


      if(n_aoanh > 0) then
      if(itime.ge.itime_tr0(n_aoanh)) then

        tr3Dsource(:,nChemistry,n_aoanh) = ma(:)/SECONDS_PER_DAY
        call apply_tracer_3Dsource(i,j,nChemistry,n_aoanh,.FALSE.)
        if (nint(lat2d_dg(i,j)).ge.30 .and.
     &      nint(lat2d_dg(i,j)).le.50) then
          tr3Dsource(1,nOverwrite,n_aoanh) = -trm_col(1,n_aoanh)*bydt
          call apply_tracer_3Dsource(i,j,nOverwrite,n_aoanh,.FALSE.)
        endif
     
      endif
      endif

      if(n_aoa > 0) then
      if(itime.ge.itime_tr0(n_aoa)) then

        tr3Dsource(:,nChemistry,n_aoa) = ma(:)/SECONDS_PER_DAY
        call apply_tracer_3Dsource(i,j,nChemistry,n_aoa,.FALSE.)
        tr3Dsource(1,nOverwrite,n_aoa) = -trm_col(1,n_aoa)*bydt
        call apply_tracer_3Dsource(i,j,nOverwrite,n_aoa,.FALSE.)

      endif
      endif

C****ST8025: An idealized loss tracer with a stratospheric source (fixed concentration above 80 mb) and idealized exponential decay in the troposphere.  Can be used to evaluate stratosphere-troposphere-exchange.


      if(n_st8025 > 0) then 
      if(itime.ge.itime_tr0(n_st8025)) then

! decay needs to be calculated here instead of using trdecay, because no decay
! should happen between 80 hPa and the tropopause
        expdecst8025 = exp(-dtsrc/(25.d0*SECONDS_PER_DAY))
        layer80hPa = minloc(abs(pmidl00-80.d0), dim=1)
        do l=1,lm
          if (l >= layer80hPa) then
            tr3Dsource(l,nOverwrite,n_st8025) = (ma(l)*200.0d-9
     &          -trm_col(l,n_st8025))*bydt
          elseif(l.le.ltropo(i,j)) then    ! decay in the troposphere
            tr3Dsource(l,nChemistry,n_st8025) =
     &          (trm_col(l,n_st8025)*expdecst8025
     &          -trm_col(l,n_st8025))*bydt
          endif ! else no tendency between tropopause and 80 hPa, do nothing
        enddo
        call apply_tracer_3Dsource(i,j,nOverwrite,n_st8025,.FALSE.)
        call apply_tracer_3Dsource(i,j,nChemistry,n_st8025,.FALSE.)

      endif
      endif

C****TAPE_REC: An idealized oscillating tracer in the tropical lower stratosphere whose period mimics the seasonal cycle of water vapor.  Can be used to evaluate vertical ascent in the lower stratosphere.

      if(n_tape_rec > 0) then
      if(itime.ge.itime_tr0(n_tape_rec)) then

C**** Get current model time
      call modelEclock%get(year=year, dayOfYear=dayOfYear)

       do l=1,lm
         if (nint(pmidl00(l)).ge.99. .and.
     &         nint(pmidl00(l)).lt.110.) then
           if (nint(lat2d_dg(i,j)).gt.-10 .and.
     &         nint(lat2d_dg(i,j)).lt.10) then
               xday=dayOfYear
               factor=(1.d0 + sin(2*(xday*pi/365.d0)-pi))*10.0d-9
               tr3Dsource(l,nOverwrite,n_tape_rec) = (ma(l)*factor
     *        -trm_col(l,n_tape_rec))*bydt
            endif
         endif
       enddo
       call apply_tracer_3Dsource(i,j,nOverwrite,n_tape_rec,.FALSE.)
  
      endif
      endif

      end subroutine calculate_and_apply_passive

#endif

#ifdef TRACERS_COSMO
      subroutine calculate_and_apply_cosmo(i,j)
      use RESOLUTION, only: LM
      use OldTracer_mod
      use TRACER_COM, only: n_Be7, n_Be10
      USE FLUXES, only: tr3Dsource
      USE MODEL_COM, only: itime
      use atmcol_com, only: ma   ! layer mass (kg/m2)
      USE apply3d, only : apply_tracer_3Dsource
      USE COSMO_SOURCES, only: be7_src_3d, be10_src_3d
      implicit none
      integer, intent(in) :: i,j
      INTEGER l  ! real*8 now

C****Be7
c cosmogenic src 
      if (itime.ge.itime_tr0(n_Be7)) then
        do l=1,lm
          tr3Dsource(l,nChemistry,n_Be7) = ma(l)*
     &         be7_src_3d(i,j,l)
        enddo
        call apply_tracer_3Dsource(i,j,nChemistry,n_Be7)
      endif

C****Be10
c cosmogenic src
      if (itime.ge.itime_tr0(n_Be10)) then
        do l=1,lm
          tr3Dsource(l,nChemistry,n_Be10) = ma(l)*
     &         be10_src_3d(i,j,l)
        enddo
        call apply_tracer_3Dsource(i,j,nChemistry,n_Be10)
      endif
C****

      end subroutine calculate_and_apply_cosmo
#endif

#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS) 
      subroutine apply_volcanic_emissions(i,j)
      use OldTracer_mod, only: trname, tr_wd_type
      use TRACER_COM, only: ntm
      use TRACER_COM, only: nVolcanic
#ifdef TRACERS_WATER
      use TRACER_COM, only: nWater
      USE TRACER_COM, only: trm_col
#endif
      use MODEL_COM, only: itime, modelEClock
      use TimeConstants_mod, only : HOURS_PER_DAY
      USE FLUXES, only: tr3Dsource
      USE apply3d, only : apply_tracer_3Dsource
      use TRACER_COM, only: direct_inject_num
      use TRACER_COM, only: direct_inject_year
      use TRACER_COM, only: direct_inject_jday
      use TRACER_COM, only: direct_inject_ndays
      use TRACER_COM, only: direct_inject_hr0
      use TRACER_COM, only: direct_inject_hr1
      USE AEROSOL_SOURCES, only: iso2directinj
      USE AEROSOL_SOURCES, only: so2_src_3d,nso2src_3d,H2O_src_3d
      USE AEROSOL_SOURCES, only: su_src_3d
#ifdef TRACERS_AMP
      USE AEROSOL_SOURCES, only: dd1_src_3d,dd2_src_3d
#endif
      USE ATMCOL_COM, only: qv
      USE ATMCOL_COM, only: update_qv
      USE MODEL_COM, only: dtsrc
#ifdef TRACERS_TOMAS
      use TRACER_COM, only: nbins
      use TRACER_COM, only: n_ASO4
      use TRACER_COM, only: n_ANUM
      use TRACER_COM, only: nSO4anum
      USE TOMAS_AEROSOL, only: sqrt_xk_xk1
      use TOMAS_EMIS, only: scalesizeSO4_vol
#endif

      implicit none
      integer :: year, dayOfYear, hour
      integer, intent(in) :: i,j
      integer :: n,nn,k
!@var src_fact source factor for the current tracer
      integer :: get_src_index,bb_i,bb_e
      integer ex
      real*8 :: src_fact,hour_fact
      interface
        real*8 function get_src_fact(n,is_bb,ibb)
          integer, intent(in) :: n
          logical, intent(in) :: is_bb
          logical, intent(in), optional :: ibb
        end function get_src_fact
      end interface

C**** Ensure direct injections only occur in specified hours
C     Cannot handle multiple injections having the same day as
C     a partial injection day (since tracer sources are daily)

      call modelEclock%get(year=year, dayOfYear=dayOfYear, hour=hour)

      hour_fact=1.d0
      do ex=1,direct_inject_num
        if ((direct_inject_hr0(ex)>0)
     &    .or.(direct_inject_hr1(ex)<HOURS_PER_DAY)) then

          if (direct_inject_year(ex)==year) then
            if ((direct_inject_jday(ex)==dayOfYear) 
     &        .and. (direct_inject_ndays(ex)==1)) then
              if ((hour<direct_inject_hr0(ex)) 
     &          .or. (hour>=direct_inject_hr1(ex))) then
                hour_fact=0.d0
              else
                hour_fact=HOURS_PER_DAY
     &            /(direct_inject_hr1(ex)-direct_inject_hr0(ex))
              endif
            else if (direct_inject_jday(ex)==dayOfYear) then
              if (hour<direct_inject_hr0(ex)) then
                hour_fact=0.d0
              else
                hour_fact=HOURS_PER_DAY
     &            /(HOURS_PER_DAY-direct_inject_hr0(ex))
              endif
            else if (direct_inject_jday(ex)==
     &        dayOfYear+direct_inject_ndays(ex)-1) then
              if (hour>=direct_inject_hr1(ex)) then
                hour_fact=0.d0
              else
                hour_fact=HOURS_PER_DAY/direct_inject_hr1(ex)
              endif
            endif
          endif
        endif
      enddo

C**** All sources are saved as kg s-1

      do n=1,ntm
        src_fact=get_src_fact(n,.false.)

        select case(trname(n))
        case ('SO2','SO4','M_ACC_SU','M_AKK_SU','ASO4__01','Water',
     &    'M_DD1_DU','M_DD2_DU')
          select case(trname(n))
          case ('SO2','SO4','M_ACC_SU','M_AKK_SU')
            do k=1,nso2src_3d
              if (k/=iso2directinj) then
                tr3Dsource(:,nVolcanic,n)=
     &            tr3Dsource(:,nVolcanic,n)+
     &            so2_src_3d(i,j,:,k)*src_fact
              endif
            enddo
            select case(trname(n))
            case ('SO2')
              if (iso2directinj>0)
     &          tr3Dsource(:,nVolcanic,n)= !no src_fact since direct inj
     &            tr3Dsource(:,nVolcanic,n)+
     &            so2_src_3d(i,j,:,iso2directinj)*hour_fact
            case ('M_ACC_SU','SO4')
              tr3Dsource(:,nVolcanic,n)= ! direct / no src_fact
     &          tr3Dsource(:,nVolcanic,n)+su_src_3d(i,j,:)
     &          *hour_fact
            end select
            call apply_tracer_3Dsource(i,j,nVolcanic,n)

          case ('Water') ! apply changes to q. do not call update_qvmom,
                         ! since the concentration always increases.
            call update_qv(qv(:)+H2O_src_3d(i,j,:)*dtsrc*hour_fact)
#ifdef TRACERS_WATER
C**** Add water to relevant tracers as well
            trm_col(:,n)=trm_col(:,n)*
     &                   (qv(:)+H2O_src_3d(i,j,:)*dtsrc*hour_fact)/qv(:)
#endif
#ifdef TRACERS_TOMAS
          case ('ASO4__01')
            do k=1,nbins
              tr3Dsource(:,nVolcanic,n_ASO4(k))=
     &          sum(so2_src_3d(i,j,:,:),2)*scalesizeSO4_vol(k)*src_fact
              tr3Dsource(:,nSO4anum,n_ANUM(k))=
     &          tr3Dsource(:,nVolcanic,n_ASO4(k))/sqrt_xk_xk1(k)
!              call apply_tracer_3Dsource(i,j,nVolcanic,n_ASO4(k))
!              call apply_tracer_3Dsource(i,j,nVolcanic,n_ANUM(k))
            enddo
#endif  /* TRACERS_TOMAS */
#ifdef TRACERS_AMP
C**** Option to add fine/coarse dust as proxy for volcanic ash (MATRIX only)
          case ('M_DD1_DU')
            tr3Dsource(:,nVolcanic,n)=
     &        tr3Dsource(:,nVolcanic,n)+dd1_src_3d(i,j,:) !direct / no src_fact
     &        *hour_fact
            call apply_tracer_3Dsource(i,j,nVolcanic,n)
          case ('M_DD2_DU')
            tr3Dsource(:,nVolcanic,n)=
     &        tr3Dsource(:,nVolcanic,n)+dd2_src_3d(i,j,:) !direct / no src_fact
     &        *hour_fact
            call apply_tracer_3Dsource(i,j,nVolcanic,n)
#endif
          end select
        case default
          ! do nothing, no emissions
        end select
      enddo

C*****
      end subroutine apply_volcanic_emissions
#endif

#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_SPECIAL_Shindell) || (defined TRACERS_TOMAS)
      subroutine apply_biomass_burning_emissions(i,j)
#ifdef TRACERS_TOMAS
      use RESOLUTION, only: LM
#endif
      use OldTracer_mod, only: trname
      use OldTracer_mod, only: do_fire
      use OldTracer_mod, only: nBBsources
      USE TRACER_COM, only: ntm, sfc_src
      use TRACER_COM, only: ntsurfsrc
      use TRACER_COM, only: nBiomass
      use TRACER_COM, only: tune_BBsources
      use TRACER_COM, only: direct_inject_num
      use TRACER_COM, only: direct_inject_year
      use TRACER_COM, only: direct_inject_jday
      use TRACER_COM, only: direct_inject_ndays
      use TRACER_COM, only: direct_inject_hr0
      use TRACER_COM, only: direct_inject_hr1
      use model_com, only: modelEclock
      use TimeConstants_mod, only: HOURS_PER_DAY
      USE AEROSOL_SOURCES, only: bc_src_3d,oc_src_3d
      USE FLUXES, only: tr3Dsource
      use atmcol_com, only: ma   ! layer mass (kg/m2)
      USE apply3d, only : apply_tracer_3Dsource
      USE PBLCOM, only: dclev
#ifdef TRACERS_TOMAS
      ! MK todo:  move volcanic operations to volcanosrc routine,
      ! because bb sources may be added in a different way in future.
      use TRACER_COM, only: nbins
      use TRACER_COM, only: n_ANUM
      use TRACER_COM, only: n_ASO4
      use TRACER_COM, only: n_AOCOB
      use TRACER_COM, only: nSO4anum
      USE TOMAS_AEROSOL, only: sqrt_xk_xk1
      USE TOMAS_EMIS, only : scalesizeSO4_vol,scalesizeSO4_bio
#endif
#ifdef PRESC_BB_INJ
      USE RAD_COM,  only : cosz1,cosz_day
      USE tracer_com, only: fire_src_3d_fact
      use RESOLUTION, only: LM
#endif  /* PRESC_BB_INJ */

      implicit none
      integer, intent(in) :: i,j
      INTEGER n,l,blay,ex
!@var src_index source index for the current tracer
!@var src_fact source factor for the current tracer
      integer :: src_index,get_src_index,bb_i,bb_e
      real*8 :: src_fact, hour_fact

      interface
        real*8 function get_src_fact(n,is_bb,ibb)
          integer, intent(in) :: n
          logical, intent(in) :: is_bb
          logical, intent(in), optional :: ibb
        end function get_src_fact
      end interface
!@var blsrc (m2/s) tr3Dsource (kg s-1) in boundary layer,
!@+                per unit of air mass (kg/m2)
      real*8 :: blsrc
#ifdef TRACERS_TOMAS
      integer :: k
      real*8, dimension(NBINS,LM) :: TOMAS_bio
#endif
      integer :: year, dayOfYear, hour

C**** All sources are saved as kg s-1
      do n=1,ntm
      src_index=get_src_index(n)
      src_fact=get_src_fact(n,.true.)

      select case (trname(n))

      case default

      case ('Alkenes', 'CO', 'NOx','HCHO','Paraffin','CH4',
#ifdef TRACERS_ACETONE
     &      'Acetone',
#endif
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
#ifdef TRACERS_dCO
     *      'd13Calke','d13CPAR',
#endif  /* TRACERS_dCO */
     *      'dC17O', 'dC18O', 'd13CO',
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
     &      'NH3', 'SO2', 'SO4', 'BCII', 'BCB', 'OCII', 'OCB',
     &      'vbsAm2', 'vbsAm1', 'vbsAz',  'vbsAp1', 'vbsAp2',
     &      'vbsAp3', 'vbsAp4', 'vbsAp5', 'vbsAp6',
     &      'M_ACC_SU', 'M_AKK_SU',
     &      'M_BC1_BC', 'M_OCC_OC', 'M_BOC_BC', 'M_BOC_OC',
     &      'M_OCC_OCM2','M_OCC_OCM1','M_OCC_OCM0',
     &      'M_OCC_OCP1','M_OCC_OCP2','M_OCC_OCP3',
     &      'M_OCC_OCP4','M_OCC_OCP5','M_OCC_OCP6',
     &      'ASO4__01','AECOB_01','AOCOB_01')

C**** Ensure direct biomass injections only occur in specified hours,
C     as for volcanic direct injections

        call modelEclock%get(year=year, dayOfYear=dayOfYear, hour=hour)

        hour_fact=1.d0
        do ex=1,direct_inject_num
          if ((direct_inject_hr0(ex)>0)
     &      .or.(direct_inject_hr1(ex)<HOURS_PER_DAY)) then

            if (direct_inject_year(ex)==year) then
              if ((direct_inject_jday(ex)==dayOfYear) 
     &          .and. (direct_inject_ndays(ex)==1)) then
                if ((hour<direct_inject_hr0(ex)) 
     &            .or. (hour>=direct_inject_hr1(ex))) then
                  hour_fact=0.d0
                else
                  hour_fact=HOURS_PER_DAY
     &              /(direct_inject_hr1(ex)-direct_inject_hr0(ex))
                endif
              else if (direct_inject_jday(ex)==dayOfYear) then
                if (hour<direct_inject_hr0(ex)) then
                  hour_fact=0.d0
                else
                  hour_fact=HOURS_PER_DAY
     &              /(HOURS_PER_DAY-direct_inject_hr0(ex))
                endif
              else if (direct_inject_jday(ex)==
     &          dayOfYear+direct_inject_ndays(ex)-1) then
                if (hour>=direct_inject_hr1(ex)) then
                  hour_fact=0.d0
                else
                  hour_fact=HOURS_PER_DAY/direct_inject_hr1(ex)
                endif
              endif
            endif
          endif
        enddo

C**** 3D biomass source
        if(do_fire(src_index) .or. nBBsources(src_index) > 0) then
          bb_i=ntsurfsrc(src_index)+1 ! index of first BB source
          bb_e=ntsurfsrc(src_index)+nBBsources(src_index) ! index last BB source
          if(do_fire(src_index)) then ! pyrE sources always last
            if (nBBsources(src_index)==0) then
              bb_e=bb_i
            else
              bb_e=bb_e+1
            endif
          endif

          blsrc=tune_BBsources*src_fact*
     &          sum(sfc_src(i,j,src_index,bb_i:bb_e))
#ifdef PRESC_BB_INJ
! Time-varying prescribed injection heights from file
          do l=1,LM ! fire_src_3d_fact is zero outside plume bottom and top
            tr3Dsource(l,nBiomass,n)=blsrc*fire_src_3d_fact(l,i,j)
          end do
          if ((sum(fire_src_3d_fact(:,i,j)) .eq. 0.d0) .and.
     &        (sum(sfc_src(i,j,src_index,bb_i:bb_e)) .ne. 0.d0)) then
            tr3Dsource(1,nBiomass,n)=blsrc
          endif
!CONTINUE FROM HERE: Why this does not sum up properly?
#else  /* .not. PRESC_BB_INJ */
! Standard plume distribution through boundary layer
          blay=int(dclev(i,j)+0.5d0)
          blsrc=blsrc/sum(MA(1:blay))
          do l=1,blay
            tr3Dsource(l,nBiomass,n)=blsrc*MA(l)
          end do
#endif  /* PRESC_BB_INJ */
#ifdef DIURN_BB
! This is a first test of diurnal cycle scaling for CO.
! Meant for the August 2017 Canada case study when CO emissions dominate everything else.
! Totally just a test.
          if(COSZ1(i,j)>0.d0)then
            injsrc=injsrc*(COSZ1(i,j)/(COSZ_day(i,j)+teeny))
          else
            injsrc=0.d0
          endif
#endif  /* DIURN_BB */
        end if 
! add direct injections (direct_inj function) of BC and OC here
        select case (trname(n))
          case ('M_BC1_BC','BCB')
            tr3Dsource(:,nBiomass,n)= ! direct / no src_fact
     &        tr3Dsource(:,nBiomass,n)+bc_src_3d(i,j,:)
     &        *hour_fact
          case ('M_OCC_OC','OCB')
            tr3Dsource(:,nBiomass,n)= ! direct / no src_fact
     &        tr3Dsource(:,nBiomass,n)+oc_src_3d(i,j,:)
     &        *hour_fact
         end select
#ifndef TRACERS_TOMAS
        call apply_tracer_3Dsource(i,j,nBiomass,n)
#else
! only apply the non-TOMAS tracers here. This assumes that TOMAS tracers
! are last, and ASO4(1) is the first TOMAS tracer.
        if(n<n_ASO4(1)) call apply_tracer_3Dsource(i,j,nBiomass,n)
#endif

#ifdef TRACERS_TOMAS

!Initialize 
        select case (trname(n))
        case ('ASO4__01')

          do k=1,nbins
            TOMAS_bio(k,:)=
     &        tr3Dsource(:,nBiomass,n_ASO4(1))*scalesizeSO4_bio(k)
          enddo
       
          do k=1,nbins
            tr3Dsource(:,nBiomass,n_ASO4(k))=TOMAS_bio(k,:)
            tr3Dsource(:,nSO4anum,n_ANUM(k))=
     &        tr3Dsource(:,nSO4anum,n_ANUM(k))
     &       +tr3Dsource(:,nBiomass,n_ASO4(k))/sqrt_xk_xk1(k)
          enddo

        end select
#endif

      end select

      end do
      end subroutine apply_biomass_burning_emissions
#endif

#if (defined TRACERS_SPECIAL_Shindell) || (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) || (defined TRACERS_TOMAS) || (defined TRACERS_GASEXCH_GCC)

      subroutine emissions_3d_prep
      use OldTracer_mod, only: trname
      USE TRACER_COM, only: ntm
      use model_com, only: modelEclock
      use ATM_COM, only: phi
      use OldTracer_mod, only: do_aircraft
      use TRACER_COM, only: nAircraft,AIRCstreams,AIRCsrc,AIRSstreams
      use OldTracer_mod, only: do_rocket
      use TRACER_COM, only: nRocket,ROCKETstreams,ROCKETsrc
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
      use TRACERS_dCO, only: dCO_fact
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
      implicit none
      INTEGER n,xday
!@var src_index source index for the current tracer
!@var src_fact source factor for the current tracer
      integer :: src_index,get_src_index
      integer :: year, dayOfYear

C**** Get current model time
      call modelEclock%get(year=year, dayOfYear=dayOfYear)

      !  Aircraft Sources Here: All Tracers! (formerly just hardcoded set allowed)
      do n=1,ntm 
        src_index=get_src_index(n)
        if(do_aircraft(src_index)) then
          xday=dayOfYear
          call get_3d_tracer
     &     (n,nAircraft,trim(trname(src_index))//'_AIRC',year,xday,
     &      phi,AIRCstreams(n),AIRCsrc(:,:,:,n),AIRSstreams(n))
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
          select case (trname(n))
          case ('dC17O')
            AIRCsrc(:,:,:,n)=AIRCsrc(:,:,:,n)*dCO_fact%dC17O_airc
          case ('dC18O')
            AIRCsrc(:,:,:,n)=AIRCsrc(:,:,:,n)*dCO_fact%dC18O_airc
          case ('d13CO')
            AIRCsrc(:,:,:,n)=AIRCsrc(:,:,:,n)*dCO_fact%d13CO_airc
          end select
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
        end if
        if(do_rocket(src_index)) then
          xday=dayOfYear
          call get_3d_tracer
     &     (n,nRocket,trim(trname(src_index))//'_ROCKET',year,xday,
     &      phi,ROCKETstreams(n),ROCKETsrc(:,:,:,n))
        end if
      end do
      end subroutine emissions_3d_prep

      subroutine apply_3d_emissions(i,j)
      USE TRACER_COM, only: ntm
      use OldTracer_mod, only: do_aircraft
      use TRACER_COM, only: nAircraft
      use TRACER_COM, only: AIRCsrc
      use OldTracer_mod, only: do_rocket
      use TRACER_COM, only: nRocket
      use TRACER_COM, only: ROCKETsrc
#ifdef TRACERS_TOMAS
      use TRACER_COM, only: n_AECOB
#endif
      USE apply3d, only : apply_tracer_3Dsource
      use fluxes, only: tr3Dsource
      implicit none
      integer, intent(in) :: i,j
!
      INTEGER n,l
!@var src_index source index for the current tracer
!@var src_fact source factor for the current tracer
      integer :: src_index,get_src_index

      !  3d sources here: All tracers! (formerly just hardcoded set allowed)
      do n=1,ntm 
        src_index=get_src_index(n)
        if(do_aircraft(src_index)) then
          tr3Dsource(:,nAircraft,n) = AIRCsrc(i,j,:,n)
#ifdef TRACERS_TOMAS
          ! TOMAS has to apply this among tracers in its own section below.
          if(n /= n_AECOB(1))
     &    call apply_tracer_3Dsource(i,j,nAircraft,n)
#else
          call apply_tracer_3Dsource(i,j,nAircraft,n)
#endif
        end if
        if(do_rocket(src_index)) then
          tr3Dsource(:,nRocket,n) = ROCKETsrc(i,j,:,n)
          call apply_tracer_3Dsource(i,j,nRocket,n)
        end if
      end do
      end subroutine apply_3d_emissions
#endif

#ifdef TRACERS_SPECIAL_Shindell
      subroutine calculate_and_apply_chemistry(i,j)
      use RESOLUTION, only: LM
      use OldTracer_mod, only: tr_mm
      USE TRACER_COM, only: trm_col
      use TRACER_COM, only: n_CFC, n_CH4
      use TRACER_COM, only: n_N2O
      use TRACER_COM, only: ntm_chem_beg, ntm_chem_end
      use TRACER_COM, only: n_NOx, nChemistry
      use TRACER_COM, only: nOther, nOverwrite
      USE CONSTANT, only : byavog
      USE FLUXES, only: tr3Dsource
      USE MODEL_COM, only: itime,dtsrc,itimeI
      USE apply3d, only : apply_tracer_3Dsource
      USE Dictionary_mod, only : get_param, is_set_param
      use TRACER_COM, only: n_GLT
#ifdef TRACERS_AEROSOLS_SOA
      USE TRACERS_SOA, only: n_soa_i,n_soa_e
#endif  /* TRACERS_AEROSOLS_SOA */


      implicit none
      integer, intent(in) :: i,j

      INTEGER n,l

C**** Update General Linear Tracer:
C Applying non-chemistry 3D sources, so they can be "seen" by chemistry:
C (Note: using this method, tracer moments are changed just like they
C are done for chemistry.  It might be better to do it like surface
C sources are done? -- GSF 11/26/02)
c
      call overwrite_GLT(i,j)
      call apply_tracer_3Dsource(i,j,nOverwrite,n_GLT)

      call get_lightning_NOx(i,j)
      call apply_tracer_3Dsource(i,j,nOther,n_NOx)

C**** Call the model CHEMISTRY and OVERWRITEs:

      call masterchem(i,j)   ! does chemistry and over-writing.
                             ! tr3Dsource defined within, for both processes

C**** Apply chemistry and overwrite changes:
      do n=ntm_chem_beg, ntm_chem_end
        call apply_tracer_3Dsource(i,j,nChemistry,n)
        call apply_tracer_3Dsource(i,j,nOverwrite,n)
      end do

      end subroutine calculate_and_apply_chemistry
#endif

#ifdef TRACERS_HETCHEM
      subroutine calculate_and_apply_hetchem(i,j)
      use TRACER_COM, only: n_N_d1,n_N_d2,n_N_d3
      use TRACER_COM, only: nChemistry
      USE apply3d, only : apply_tracer_3Dsource
      implicit none
      integer, intent(in) :: i,j

c calculation of heterogeneous reaction rates: SO2 on dust 
      CALL SULFDUST(i,j)

#ifdef TRACERS_NITRATE
       call apply_tracer_3Dsource(i,j,nChemistry,n_N_d1) ! NO3 chem prod on dust
       call apply_tracer_3Dsource(i,j,nChemistry,n_N_d2) ! NO3 chem prod on dust
       call apply_tracer_3Dsource(i,j,nChemistry,n_N_d3) ! NO3 chem prod on dust
#endif  /* TRACERS_NITRATE */

      end subroutine calculate_and_apply_hetchem
#endif  /* TRACERS_HETCHEM */

      subroutine apply_aerosol_gas_chem(i,j)
      use TRACER_COM, only: n_DMS,n_MSA,n_SO2
      use TRACER_COM, only: n_SO4_d1,n_SO4_d2,n_SO4_d3,n_SO4,n_H2SO4
      use TRACER_COM, only: n_H2O2_s
      use TRACER_COM, only: nChemistry,nChemprod,nChemloss
      USE apply3d, only : apply_tracer_3Dsource
      implicit none

      integer, intent(in) :: i,j

! if any index is zero below, nothing happens; no need to check before calling
      call apply_tracer_3Dsource(i,j,nChemistry,n_DMS)    ! DMS chem sink
      call apply_tracer_3Dsource(i,j,nChemistry,n_MSA)    ! MSA chem source
      call apply_tracer_3Dsource(i,j,nChemprod,n_SO2)     ! SO2 chem source
      call apply_tracer_3Dsource(i,j,nChemloss,n_SO2)     ! SO2 chem sink

      call apply_tracer_3Dsource(i,j,nChemistry,n_SO4_d1) ! SO4 chem prod on dust
      call apply_tracer_3Dsource(i,j,nChemistry,n_SO4_d2) ! SO4 chem prod on dust
      call apply_tracer_3Dsource(i,j,nChemistry,n_SO4_d3) ! SO4 chem prod on dust
      call apply_tracer_3Dsource(i,j,nChemistry,n_SO4)    ! SO4 chem source
!H2SO4 chem prod is zero for TOMAS (H2SO4_chem will be used in TOMAS_DRV)
!But it still calls to save the diagnostics. 
      call apply_tracer_3Dsource(i,j,nChemistry,n_H2SO4)  ! SO4 chem source

      call apply_tracer_3Dsource(i,j,nChemprod,n_H2O2_s)  ! H2O2 chem source
      call apply_tracer_3Dsource(i,j,nChemLoss,n_H2O2_s)  ! H2O2 chem sink

      end subroutine apply_aerosol_gas_chem

#ifdef TRACERS_TOMAS
      subroutine calculate_and_apply_tomas(i,j)
      USE DOMAIN_DECOMP_ATM, only : am_i_root
      use RESOLUTION, only: LM
      use OldTracer_mod, only: do_aircraft
      use TimeConstants_mod, only: SECONDS_PER_DAY
      USE TRACER_COM, only: trm_col
      use TRACER_COM, only: n_DMS, n_H2O2_s
      use TRACER_COM, only: n_NH3, n_NH4
      use TRACER_COM, only: n_SO2
      use TRACER_COM, only: nAircraft, nBiomass, nThermo, nMicrophys
      use TRACER_COM, only: nVolcanic, nChemprod
      use TRACER_COM, only: nSO4anum, nECanum, nOCanum
      use TRACER_COM, only: nbins
      use TRACER_COM, only: n_AOCIL, n_ANUM
      use TRACER_COM, only: n_AECOB, n_AOCOB, n_ASO4, n_H2SO4, n_SOAgas
      use TRACER_COM, only: nChemistry, n_AECIL, ntm_tomas
      USE FLUXES, only: tr3Dsource
      USE MODEL_COM, only: dtsrc
      USE apply3d, only : apply_tracer_3Dsource
      USE TOMAS_AEROSOL, only : trm_preemis
      USE TOMAS_AEROSOL, only: sqrt_xk_xk1
      USE TOMAS_EMIS, only : scalesizeCARBO100,scalesizeCARBO30
      implicit none
      integer, intent(in) :: i,j
!
      INTEGER n,l
!@var src_index source index for the current tracer
!@var src_fact source factor for the current tracer
      integer :: src_index,get_src_index
      integer :: k,kn,jc,tracnum
      real*8, dimension (NBINS,LM) :: TOMAS_bio,TOMAS_air

! EC/OC aging
      do k=1,nbins
        call calc_and_apply_expo_decay(i,j,1.5d0,n_AECOB(k),n_AECIL(k))
        call calc_and_apply_expo_decay(i,j,1.5d0,n_AOCOB(k),n_AOCIL(k))
      enddo

! save trm_col before emissions for internal TOMAS usage
      do l=1,lm
        trm_preemis(:,l)=trm_col(l,:)
      enddo
   
! EC from biomass burning. Keep the two loops separate.
      do k=1,nbins
        TOMAS_bio(k,:)=
     &    tr3Dsource(:,nBiomass,n_AECOB(1))*scalesizeCARBO100(k)
      enddo

      do k=1,nbins
        tr3Dsource(:,nBiomass,n_AECOB(k))=TOMAS_bio(k,:)*0.8d0
        tr3Dsource(:,nBiomass,n_AECIL(k))=TOMAS_bio(k,:)*0.2d0
        tr3Dsource(:,nECanum,n_ANUM(k))=TOMAS_bio(k,:)/sqrt_xk_xk1(k)

        call apply_tracer_3Dsource(i,j,nBiomass, n_AECOB(k))
        call apply_tracer_3Dsource(i,j,nBiomass, n_AECIL(k))
        call apply_tracer_3Dsource(i,j,nECanum, n_ANUM(k))
      enddo

! EC from aircraft. Keep the two loops separate.
      TOMAS_air(:,:)=0.d0

      if(do_aircraft(n_AECOB(1)))then
        do k=1,nbins
          TOMAS_air(k,:)=
     &      tr3Dsource(:,nAircraft,n_AECOB(1))*scalesizeCARBO30(k)
        enddo
      endif

      if(do_aircraft(n_AECOB(1))) then
        do k=1,nbins
          tr3Dsource(:,nAircraft,n_AECOB(k))=TOMAS_air(k,:)*0.8d0
          tr3Dsource(:,nAircraft,n_AECIL(k))=TOMAS_air(k,:)*0.2d0
          tr3Dsource(:,nECanum,n_ANUM(k))=TOMAS_air(k,:)/sqrt_xk_xk1(k)

          call apply_tracer_3Dsource(i,j,nAircraft,n_AECOB(k))
          call apply_tracer_3Dsource(i,j,nAircraft,n_AECIL(k))
          call apply_tracer_3Dsource(i,j,nECanum, n_ANUM(k))
        enddo
      endif

! sulfate from volcanic and biomass burning.
      do k=1,nbins
        call apply_tracer_3Dsource(i,j,nVolcanic,n_ASO4(k))
        call apply_tracer_3Dsource(i,j,nBiomass, n_ASO4(k))
        call apply_tracer_3Dsource(i,j,nSO4anum, n_ANUM(k)) 
      enddo

! OC from biomass burning. Keep the two loops separate.
      do k=1,nbins
         TOMAS_bio(k,:)=
     &     tr3Dsource(:,nBiomass,n_AOCOB(1))*scalesizeCARBO100(k)
      enddo

      do k=1,nbins
         tr3Dsource(:,nBiomass,n_AOCOB(k))=TOMAS_bio(k,:)*0.5d0
         tr3Dsource(:,nBiomass,n_AOCIL(k))=TOMAS_bio(k,:)*0.5d0
         tr3Dsource(:,nOCanum,n_ANUM(k))=TOMAS_bio(k,:)/sqrt_xk_xk1(k)
 
         call apply_tracer_3Dsource(i,j,nBiomass, n_AOCOB(k))
         call apply_tracer_3Dsource(i,j,nBiomass, n_AOCIL(k))
         call apply_tracer_3Dsource(i,j,nOCanum, n_ANUM(k))
      enddo
       
! call TOMAS code
      call subgridcoag_drv(i,j,dtsrc)
      call TOMAS_DRV(i,j)
!      if(am_i_root()) print*,'exit TOMAS DRV'

! the following loop assumes that n_ASO4(1) is the first TOMAS tracer and the last
! nbins ones are all water tracers.
      DO n=1,ntm_TOMAS-nbins ! exclude h2o
        call apply_tracer_3Dsource(i,j,nMicrophys,n_ASO4(1)+n-1)! Aerosol Mirophysics
      ENDDO

      call apply_tracer_3Dsource(i,j,nThermo,n_NH3) !simple equilibrium model in TOMAS
      call apply_tracer_3Dsource(i,j,nThermo,n_NH4) !simple equilibrium model in TOMAS
      call apply_tracer_3Dsource(i,j,nMicrophys,n_H2SO4) ! H2SO4 chem prod
      call apply_tracer_3Dsource(i,j,nChemistry,n_SOAgas) ! SOAgas chem prod

      end subroutine calculate_and_apply_tomas
#endif /* TRACERS_TOMAS */

#ifdef TRACERS_AEROSOLS_Koch
      subroutine calculate_and_apply_oma(i,j)
      use RESOLUTION, only: LM
      use OldTracer_mod, only: trname
      use TRACER_COM, only: ntm, trm_col
      use TRACER_COM, only: n_BCIA, n_BCII
      use TRACER_COM, only: n_DMS, n_H2O2_s, n_MSA
      use TRACER_COM, only: n_OCIA, n_OCII
      use TRACER_COM, only: n_SO4, n_SO4_d1, n_SO4_d2, n_SO4_d3
      use TRACER_COM, only: n_SO2
      use TRACER_COM, only: nChemistry
      use TRACER_COM, only: nChemloss, nChemprod
      use TRACER_COM, only: nOther
      USE FLUXES,     only: tr3Dsource
      USE MODEL_COM,  only: dtsrc
      USE apply3d, only : apply_tracer_3Dsource
#ifdef TRACERS_AEROSOLS_VBS
      use TRACER_COM, only: n_BCB, n_isopp1a, n_isopp2a, n_apinp1a,
     &                      n_apinp2a, n_NH4, n_NO3p
      use CONSTANT, only : gasc,mair
      use atmcol_com, only: tl   ! layer temperature (K)
      use atmcol_com, only: pl   ! layer pressure (mb)
      use atmcol_com, only: ma   ! layer mass (kg/m2)
      use AEROSOL_SOURCES, only: oxid,vbs_sets,vbs_conc
      use TRACERS_VBS, only: vbs_tracers,vbs_conditions,vbs_calc
#endif  /* TRACERS_AEROSOLS_VBS */
      use TimeConstants_mod, only: SECONDS_PER_DAY

      implicit none
      integer, intent(in) :: i,j

#ifdef TRACERS_AEROSOLS_VBS
!      type(vbs_tracers) :: vbs_tr_old ! concentrations, ug m-3
      type(vbs_conditions) :: vbs_cond ! current box conditions (meteo+chem)
!@var kg2ugm3 factor to convert kilograms gridbox-1 to ug m-3
      real*8 :: kg2ugm3
      integer :: l,v
#endif /* TRACERS_AEROSOLS_VBS */
      integer :: n

      do n=1,NTM

      select case (trname(n))
        case ('BCII')
          call calc_and_apply_expo_decay(i,j,1.d0,n_BCII,n_BCIA) ! efold time of 1 days

#ifdef TRACERS_AEROSOLS_VBS
        case ('vbsAm2') ! This handles all VBS tracers
        do l=1,lm
          call get_oxidants(i,j,l) ! get oxidant concentrations
          kg2ugm3=1.d9*(1.d2*pl(l))*mair/(ma(l)*gasc*tl(l))
          vbs_cond%dt=dtsrc
          vbs_cond%OH=oxid%OH
          vbs_cond%temp=tl(l)
          vbs_cond%nvoa=(trm_col(l,n_BCII)
     &                  +trm_col(l,n_BCIA)
     &                  +trm_col(l,n_BCB)
#ifdef TRACERS_AEROSOLS_SOA
     &                  +trm_col(l,n_isopp1a)
     &                  +trm_col(l,n_isopp2a)
     &                  +trm_col(l,n_apinp1a)
     &                  +trm_col(l,n_apinp2a)
#endif /* TRACERS_AEROSOLS_SOA */
#ifdef TRACERS_AEROSOLS_OCEAN
     &                  +trm_col(l,n_ococean)
#endif  /* TRACERS_AEROSOLS_OCEAN */
     &                  +trm_col(l,n_msa)
     &                  +trm_col(l,n_so4)
#ifdef TRACERS_NITRATE
     &                  +trm_col(l,n_nh4)
     &                  +trm_col(l,n_no3p)
#endif
     &                  )*kg2ugm3
          vbs_conc(1)%gas=trm_col(l,vbs_conc(1)%igas)*kg2ugm3
          vbs_conc(1)%aer=trm_col(l,vbs_conc(1)%iaer)*kg2ugm3

          call vbs_calc(vbs_conc(1),vbs_cond)

          tr3Dsource(l,nChemprod,vbs_conc(1)%igas)=
     &      vbs_conc(1)%chem_prod/kg2ugm3/vbs_cond%dt
          tr3Dsource(l,nChemloss,vbs_conc(1)%igas)=
     &      vbs_conc(1)%chem_loss/kg2ugm3/vbs_cond%dt
          tr3Dsource(l,nOther,vbs_conc(1)%igas)=
     &      -vbs_conc(1)%partition/kg2ugm3/vbs_cond%dt ! partitioning
          tr3Dsource(l,nOther,vbs_conc(1)%iaer)=
     &      vbs_conc(1)%partition/kg2ugm3/vbs_cond%dt
!     &      (vbs_conc(1)%gas-vbs_tr_old%gas)/kg2ugm3/vbs_cond%dt
!      if (sum(vbs_tr_old%gas)+sum(vbs_tr_old%aer) /= 0.) then
!        print '(a,3e)','KOSTAS gas',
!     &                 sum(vbs_tr_old%gas),
!     &                 sum(vbs_conc(1)%gas),
!     &                 sum(vbs_tr_old%gas)+sum(vbs_tr_old%aer)
!        print '(a,3e)','KOSTAS aer',
!     &                 sum(vbs_tr_old%aer),
!     &                 sum(vbs_conc(1)%aer),
!     &                 sum(vbs_conc(1)%gas)+sum(vbs_conc(1)%aer)
!        print '(a,3e)','KOSTAS bud',
!     &                 sum(vbs_conc(1)%chem_prod),
!     &                 sum(vbs_conc(1)%chem_loss),
!     &                 sum(vbs_conc(1)%partition)
!      endif
        enddo

        do v=1,vbs_conc(1)%nbins
          call apply_tracer_3Dsource(i,j,nChemprod,vbs_conc(1)%igas(v))  ! aging source
          call apply_tracer_3Dsource(i,j,nChemloss,vbs_conc(1)%igas(v))  ! aging loss
          call apply_tracer_3Dsource(i,j,nOther,vbs_conc(1)%igas(v))     ! partitioning
          call apply_tracer_3Dsource(i,j,nOther,vbs_conc(1)%iaer(v))     ! partitioning
        enddo
#else
        case ('OCII')
          call calc_and_apply_expo_decay(i,j,1.6d0,n_OCII,n_OCIA) ! efold time of 1.6 days
#endif /* TRACERS_AEROSOLS_VBS */
      end select

      enddo

      end subroutine calculate_and_apply_oma
#endif

#ifdef TRACERS_AMP
      subroutine calculate_and_apply_matrix(i,j)
      use OldTracer_mod, only: trname
      use TRACER_COM, only: n_DMS, n_H2O2_s, n_SO2
      use TRACER_COM, only: n_NH3
      use TRACER_COM, only: n_H2SO4
      use TRACER_COM, only: n_vbsGm2,n_vbsGm1,n_vbsGz,n_vbsGp1,n_vbsGp2,
     &                      n_vbsGp3,n_vbsGp4,n_vbsGp5,n_vbsGp6,nOther
      use TRACER_COM, only: nChemistry, nThermo, nMicrophys
      use TRACER_COM, only: ntmAMPi, ntmAMPe
      USE apply3d, only : apply_tracer_3Dsource
#ifdef  TRACERS_SPECIAL_Shindell
      use TRACER_COM, only: n_HNO3
#endif 

      implicit none
      integer, intent(in) :: i,j
!
      INTEGER n

      call MATRIX_DRV(i,j)

      DO n=ntmAMPi,ntmAMPe
        select case(trname(n))
        case ('M_NO3','M_NH4','M_H2O')
          call apply_tracer_3Dsource(i,j,nThermo,n) ! Aerosol Thermodynamics
        case default
          call apply_tracer_3Dsource(i,j,nMicrophys,n) ! Aerosol Microphysics
        end select
      ENDDO

      call apply_tracer_3Dsource(i,j,nThermo,n_NH3)    ! NH3
      call apply_tracer_3Dsource(i,j,nMicrophys,n_H2SO4) ! H2SO4 chem prod
#ifdef  TRACERS_SPECIAL_Shindell
      call apply_tracer_3Dsource(i,j,nThermo,n_HNO3)    ! HNO3 change due to thermodynamics
#endif
      call apply_tracer_3Dsource(i,j,nOther, n_vbsGm2)
      call apply_tracer_3Dsource(i,j,nOther, n_vbsGm1)
      call apply_tracer_3Dsource(i,j,nOther, n_vbsGz)
      call apply_tracer_3Dsource(i,j,nOther, n_vbsGp1)
      call apply_tracer_3Dsource(i,j,nOther, n_vbsGp2)
      call apply_tracer_3Dsource(i,j,nOther, n_vbsGp3)
      call apply_tracer_3Dsource(i,j,nOther, n_vbsGp4)
      call apply_tracer_3Dsource(i,j,nOther, n_vbsGp5)
      call apply_tracer_3Dsource(i,j,nOther, n_vbsGp6)

      end subroutine calculate_and_apply_matrix
#endif

#ifdef TRACERS_NITRATE
      subroutine calculate_and_apply_nitrate(i,j)
      use RESOLUTION, only: LM
      USE CONSTANT,   only: mair,gasc
      use ATMCOL_COM, only: tl   ! layer temperature (K)
      use ATMCOL_COM, only: rhl  ! layer relative humidity (0-1)
      use ATMCOL_COM, only: pl   ! layer pressure (mb)
      use ATMCOL_COM, only: ma   ! layer mass (kg/m2)
      USE TRACER_COM, only: trm_col
      use TRACER_COM, only: n_SO4
      use TRACER_COM, only: n_HNO3,n_NO3p
      use TRACER_COM, only: n_NH3,n_NH4
      use TRACER_COM, only: ntm_dust, n_soilDust
      use TRACER_COM, only: n_seasalt1,n_seasalt2
      use TRACER_COM, only: nThermo
      use RunTimeControls_mod, only: tracers_special_shindell
      use RunTimeControls_mod, only: tracers_dust, tracers_minerals
      use RunTimeControls_mod, only: tracers_aerosols_seasalt
      use TRACER_COM, only: coupled_chem
      USE AEROSOL_SOURCES, only: off_HNO3
      USE MODEL_COM, only : dtsrc
      use TRDIAG_COM, only: taijls=>taijls_loc,ijlt_aH2O,ijlt_apH
      USE FLUXES, only: tr3Dsource
      USE apply3d, only : apply_tracer_3Dsource
#ifdef TRACERS_SPECIAL_Shindell
      use TRCHEM_Shindell_COM, only: topLevelOfChemistry
#else
      use AEROSOL_SOURCES, only: off_HNO3
#endif
      implicit none
!@var AVOL Convert kg m-2 to kg m-3
!@var ASO4 Aerosol sulfate [ug m-3]
!@var ANO3 Aerosol nitrate [ug m-3]
!@var ANH4 Aerosol ammonium [ug m-3]
!@var AH2O Aerosol water [ug m-3]
!@var ApH Aerosol pH
!@var GNH3 Gaseous ammonia [ug m-3]
!@var GHNO3 Gaseous nitric acid [ug m-3]
      real*8 :: AVOL,ASO4,ANO3,ANH4,DUST,SALT,AH2O,ApH,SSH2O
      real*8 :: GNH3,GHNO3,RHD,RHC

      integer, intent(in) :: i,j
!
      integer :: l,lm_nitrate, nd

#ifdef TRACERS_SPECIAL_Shindell
      lm_nitrate = topLevelOfChemistry
#else
      lm_nitrate = LM
#endif
      do l=1,lm_nitrate
        AVOL=ma(l)/mair*1000.d0*gasc*tl(l)/(pl(l)*100.d0)

        ASO4=trm_col(l,n_SO4)*1.d9/AVOL
        ANO3=trm_col(l,n_NO3p)*1.d9/AVOL
        ANH4=trm_col(l,n_NH4)*1.d9/AVOL
        GNH3=trm_col(l,n_NH3)*1.d9/AVOL
        if (tracers_special_shindell.and.coupled_chem==1) then
          GHNO3=trm_col(l,n_HNO3)*1.d9/AVOL
        else
          GHNO3=off_HNO3(i,j,l)*1.d9/AVOL
        endif
        DUST=0.d0
        if (tracers_dust .or. tracers_minerals) then
          do nd = n_soilDust,n_soilDust+ntm_dust-1
            DUST = DUST + trm_col( l, nd )
          end do
          DUST = DUST * 1.d9/AVOL
        end if
        if (tracers_aerosols_seasalt) then
          SALT=trm_col(l,n_seasalt1)*1.d9/AVOL
     &        +trm_col(l,n_seasalt2)*1.d9/AVOL
        else
          SALT=0.d0
        endif

        call AERO_THERMO(ASO4,ANO3,ANH4,DUST,SALT,AH2O,ApH,SSH2O,
     &                   GNH3,GHNO3,tl(l),rhl(l),RHD,RHC,
     &                   .false.,66)

! no need to update tr3Dsource for ASO4, since it does not change in eqsam.
        tr3Dsource(l,nThermo,n_NO3p)=(ANO3*1.d-9*AVOL-
     &                                trm_col(l,n_NO3p))/dtsrc
        tr3Dsource(l,nThermo,n_NH4)= (ANH4*1.d-9*AVOL-
     &                                trm_col(l,n_NH4))/dtsrc
! no need to update tr3Dsource for DUST, since it does not change in eqsam.
! no need to update tr3Dsource for SALT, since it does not change in eqsam.
! water is not affected, the aerosol amount is only a diagnostic in terms of mass
        tr3Dsource(l,nThermo,n_NH3)= (GNH3*1.d-9*AVOL-
     &                                trm_col(l,n_NH3))/dtsrc
        if (tracers_special_shindell.and.coupled_chem==1) then
          tr3Dsource(l,nThermo,n_HNO3)=(GHNO3*1.d-9*AVOL-
     &                                  trm_col(l,n_HNO3))/dtsrc
        endif

! save aerosol water (ug/m3) and aerosol pH (dimensionless)
        taijls(I,J,L,ijlt_aH2O)=taijls(I,J,L,ijlt_aH2O)+AH2O
        taijls(I,J,L,ijlt_apH)=taijls(I,J,L,ijlt_apH)+ApH
      enddo

      if (tracers_special_shindell.and.coupled_chem==1) then
        call apply_tracer_3Dsource(i,j,nThermo,n_HNO3) ! HNO3 change
      endif
      call apply_tracer_3Dsource(i,j,nThermo,n_NO3p) ! NO3p change
      call apply_tracer_3Dsource(i,j,nThermo,n_NH4)  ! NH4 change
      call apply_tracer_3Dsource(i,j,nThermo,n_NH3)  ! NH3 change

      return

      end subroutine calculate_and_apply_nitrate
#endif

      SUBROUTINE tracer_3Dsource
!@sum tracer_3Dsource calculates interactive sources for tracers
!@+   All sources are saved as kg/m2/s
!@+   Please note that if the generic routine 'apply_tracer_3Dsource'
!@+   is used, all diagnostics and moments are updated automatically.
      USE DOMAIN_DECOMP_ATM, only : GRID, getDomainBounds
      use RESOLUTION, only: LM
      use tracer_com, only : trm,trm_col
      use tracer_com, only : trmom,trmom_col
      USE TRACER_COM, only: ntm,n_Pb210
     & ,mchem,mtrace,coupled_chem
     & ,n_Ox,n_SO2,n_H2O2,n_HNO3,n_NH3
      USE MODEL_COM,  only: itime,dtsrc,itimeI
#ifdef CACHED_SUBDD
#ifdef TRACERS_DRYDEP_DIAG_SUBDD
      use subdd_mod, only : inc_subdd
#endif
#endif
#ifndef SKIP_TRACER_SRCS
      USE FLUXES, only: tr3Dsource
#endif
#ifdef TRACERS_DRYDEP
#ifdef TRACERS_DRYDEP_DIAG_SUBDD
      USE FLUXES, only: vd_glob_for
     & ,vd_glob_cro,vd_glob_gra
     & ,vd_glob_shr,egs_glob_for
     & ,egs_glob_cro,egs_glob_gra
     & ,egs_glob_shr,egcut_wet_glob
#endif
#endif
#ifdef TRACERS_SPECIAL_Shindell
      use rad_com, only: clim_interact_chem
      use somtq_com, only : qmom
      use atm_com, only : q
      use qusdef, only : nmom
      use atmcol_com, only: qv, qvmom
#endif
      use geom, only : imaxj
      implicit none
      INTEGER J_0, J_1, I_0, I_1
      real*8 :: now
      integer :: i,j,n
C****
C**** Extract useful local domain parameters from "grid"
C****

      call getDomainBounds(grid, J_STRT=J_0, J_STOP=J_1)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP


#if (defined TRACERS_SPECIAL_Shindell) || (defined TRACERS_AEROSOLS_Koch) ||\
    (defined TRACERS_AMP) || (defined TRACERS_TOMAS) ||\
    (defined TRACERS_GASEXCH_GCC)
c If GCC_ZERO_EMISSION is defined in GCC scheme, don't prepare 3d (aircraft+rocket)
#if (defined TRACERS_GASEXCH_GCC) && (defined GCC_ZERO_EMISSION)
#else
      call emissions_3d_prep
#endif
#endif

#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) || \
    (defined TRACERS_TOMAS)
      if (coupled_chem.le.0) call aerosol_gas_chem_prep ! testing moving it to beginning of tr3dsrc
#endif

#ifdef TRACERS_SPECIAL_Shindell
      call masterchem_prep
#endif

#ifdef TRACERS_SPECIAL_Lerner
      call lernerchem_prep
#endif

C****
C**** Loop over columns
C****

      do j=j_0,j_1
      do i=i_0,imaxj(j)

C****
C**** Initialize tracer source array
C****
#ifndef SKIP_TRACER_SRCS
        tr3Dsource(:,:,:) = 0.d0
#endif

        call load_atmcol(i,j)

        ! copy into column array
        trm_col(:,:) = trm(i,j,:,:)
        trmom_col(:,:,:) = trmom(:,i,j,:,:)

        do n=1,ntm
          call diagtca_1pt(1,n,i,j) ! 1 means reinit
        enddo

C****
C**** Tracer gravitational settling for aerosols
C****
        call TRGRAV(i,j)

C****
C**** Calculate tracer sources
C****

C****
C**** Tracer radioactive decay (and possible source)
C****
        call TDECAY(i,j)

#ifdef TRACERS_SPECIAL_Lerner
c**** Calculate and apply sources for Lerner tracers
        call calculate_and_apply_lerner(i,j)
#endif

#ifdef TRACERS_PASSIVE
c**** Calculate and apply sources for Passive tracers
        call calculate_and_apply_passive(i,j)
#endif

#ifdef TRACERS_COSMO
C**** Calculate and apply cosmogenic sources
        call calculate_and_apply_cosmo(i,j)
#endif

#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS) 
C**** Apply volcanic sources
        call apply_volcanic_emissions(i,j)
#endif


#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_SPECIAL_Shindell) || (defined TRACERS_TOMAS)
C**** Apply biomass burning sources
        call apply_biomass_burning_emissions(i,j)
#endif

#if (defined TRACERS_AEROSOLS_Koch) ||\
    (defined TRACERS_SPECIAL_Shindell) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS)
c Calculation of gas phase reaction rates for sulfur chemistry
        call get_sulf_gas_rates(i,j)
#endif

#if (defined TRACERS_SPECIAL_Shindell) || (defined TRACERS_AEROSOLS_Koch) ||\
    (defined TRACERS_AMP) || (defined TRACERS_TOMAS) ||\
    (defined TRACERS_GASEXCH_GCC)
c If GCC_ZERO_EMISSION is defined in GCC scheme, don't add 3d (aircraft+rocket) sources
#if (defined TRACERS_GASEXCH_GCC) && (defined GCC_ZERO_EMISSION)
#else
C**** Apply 3d sources
        call apply_3d_emissions(i,j)
#endif
#endif

#ifdef TRACERS_SPECIAL_Shindell
c**** Calculate and apply sources from gas-phase chemistry
        call timer(now, mtrace)
        call calculate_and_apply_chemistry(i,j)
        call timer(now, mchem)
#endif

#ifdef TRACERS_HETCHEM
c**** Calculate and apply sources from heterogeneous chemistry on dust
        call calculate_and_apply_hetchem(i,j)
#endif  /* TRACERS_HETCHEM */

#ifdef TRACERS_NITRATE
c**** Calculate and apply nitrate (thermo) sources
        call calculate_and_apply_nitrate(i,j)
#endif


#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) || \
    (defined TRACERS_TOMAS)
c**** Calculate aerosol-gas chemistry tendencies (sources/sinks)
        call aerosol_gas_chem(i,j)
        call apply_aerosol_gas_chem(i,j)
#endif

#ifdef TRACERS_TOMAS
c**** Calculate and add sources from TOMAS aerosol chemistry/physics
        call calculate_and_apply_tomas(i,j)
#endif

#ifdef TRACERS_AEROSOLS_Koch
c**** Calculate and apply sources from Koch/OMA aerosol chemistry/physics
        call calculate_and_apply_oma(i,j)
#endif

#ifdef TRACERS_AMP
c**** Calculate and apply sources from AMP/MATRIX
        call calculate_and_apply_matrix(i,j)
#endif

        ! copy out of column array
        trm(i,j,:,:) = trm_col(:,:)
        trmom(:,i,j,:,:) = trmom_col(:,:,:)
#ifdef TRACERS_SPECIAL_Shindell
        ! also update humidity if it's been changed by chemistry:
        if(clim_interact_chem > 0)then
          q(i,j,:) = qv(:)
          qmom(1:nmom,i,j,1:lm) = qvmom(1:nmom,1:lm)
        end if
#endif

      enddo
      enddo

#ifdef TRACERS_SPECIAL_Shindell
      call masterchem_post
#endif
#ifdef TRACERS_AMP
      call matrix_post
#endif

      ! fixup
      do n=1,ntm
        call diagtca(1,n) ! 1 means reinit
      enddo

c*****
c***** End tracer source calculations 
c*****

#ifdef CACHED_SUBDD
      ! Accumulate the tracer-related subdaily diagnostics
      ! (seems like a reasonable place to put this, as tracer_3Dsource
      ! is called each time step, and here chemistry has been done, etc.,
      ! but we could move it):
      call accumCachedTracerSUBDDs

#ifdef TRACERS_DRYDEP_DIAG_SUBDD
      call inc_subdd('Ox_vd_for',
     &      vd_glob_for(:,:,n_Ox)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='Ox deposition velocity for forests') 

      call inc_subdd('Ox_vd_cro',
     &      vd_glob_cro(:,:,n_Ox)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='Ox deposition velocity for crops')

      call inc_subdd('Ox_vd_gra',
     &      vd_glob_gra(:,:,n_Ox)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='Ox deposition velocity for grasses')

      call inc_subdd('Ox_vd_shr',
     &      vd_glob_shr(:,:,n_Ox)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='Ox deposition velocity for shrubs')

      call inc_subdd('Ox_egs_for',
     &      egs_glob_for(:,:,n_Ox)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='Ox effective stomatal conductance for forests')

      call inc_subdd('Ox_egs_cro',
     &      egs_glob_cro(:,:,n_Ox)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='Ox effective stomatal conductance for crops')

      call inc_subdd('Ox_egs_gra',
     &      egs_glob_gra(:,:,n_Ox)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='Ox effective stomatal conductance for grasses')

      call inc_subdd('Ox_egs_shr',
     &      egs_glob_shr(:,:,n_Ox)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='Ox effective stomatal conductance for shrubs')

      call inc_subdd('Ox_egcut_wet',
     &      egcut_wet_glob(:,:,n_Ox)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='Ox effective wet cuticular conductance')

      call inc_subdd('SO2_egcut_wet',
     &      egcut_wet_glob(:,:,n_SO2)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='SO2 effective wet cuticular conductance')

      call inc_subdd('SO2_vd_for',
     &      vd_glob_for(:,:,n_SO2)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='SO2 deposition velocity for forests')

      call inc_subdd('SO2_vd_cro',
     &      vd_glob_cro(:,:,n_SO2)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='SO2 deposition velocity for crops')

      call inc_subdd('SO2_vd_gra',
     &     vd_glob_gra(:,:,n_SO2)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='SO2 deposition velocity for grasses')

      call inc_subdd('SO2_vd_shr',
     &      vd_glob_shr(:,:,n_SO2)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='SO2 deposition velocity for shrubs')

      call inc_subdd('SO2_egs_for',
     &      egs_glob_for(:,:,n_SO2)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='SO2 effective stomatal conductance for forests')

      call inc_subdd('SO2_egs_cro',
     &      egs_glob_cro(:,:,n_SO2)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='SO2 effective stomatal conductance for crops')

      call inc_subdd('SO2_egs_gra',
     &     egs_glob_gra(:,:,n_SO2)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='SO2 effective stomatal conductance for grasses')

      call inc_subdd('SO2_egs_shr',
     &      egs_glob_shr(:,:,n_SO2)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='SO2 effective stomatal conductance for shrubs')

      call inc_subdd('H2O2_vd_for',
     &      vd_glob_for(:,:,n_H2O2)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='H2O2 deposition velocity for forests')

      call inc_subdd('HNO3_vd_for',
     &      vd_glob_for(:,:,n_HNO3)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='HNO3 deposition velocity for forests')

      call inc_subdd('HNO3_vd_shr',
     &      vd_glob_shr(:,:,n_HNO3)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='HNO3 deposition velocity for shrubs')

      call inc_subdd('HNO3_vd_gra',
     &      vd_glob_gra(:,:,n_HNO3)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='HNO3 deposition velocity for grasses')

      call inc_subdd('NH3_vd_for',
     &      vd_glob_for(:,:,n_NH3)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='NH3 deposition velocity for forests')

      call inc_subdd('NH3_vd_shr',
     &      vd_glob_shr(:,:,n_NH3)*100.d0,
     &      6,.false.,units='cm/s',
     &      long_name='NH3 deposition velocity for shrubs')
#endif /* TRACERS_DRYDEP_DIAG_SUBDD */
#endif /* CACHED_SUBDD */

      return

      END SUBROUTINE tracer_3Dsource
#endif /* TRACERS_ON */

#ifdef TRACERS_WATER
C---SUBROUTINES FOR TRACER WET DEPOSITION-------------------------------

      SUBROUTINE GET_COND_FACTOR(
     &     NTX,WMXTR,TEMP,TEMP0,LHX,FCLOUD,
     &    FQ0,fq,TR_CONV,FQAERFAC,TRWML,TM,THLAW,TR_LEF,pl,ntix,CLDSAVT)
!@sum  GET_COND_FACTOR calculation of condensate fraction for tracers
!@+    within or below convective or large-scale clouds. Gas
!@+    condensation uses Henry's Law if not freezing.
!@auth Dorothy Koch (modelEifications by Greg Faluvegi)
! NOTE: THLAW is only computed for the tracers in hlaw_list!
c
C**** GLOBAL parameters and variables:
      USE CONSTANT, only: BYGASC, MAIR,teeny,LHE,tf,by3

      USE TRACER_COM, only :
     &     aero_count,water_count,hlaw_count,
! NB: these lists are often used for implicit loops
     &     aero_list,water_list,hlaw_list

      use OldTracer_mod, only: tr_RKD, tr_DHD, tr_wd_type
      use OldTracer_mod, only: nWater, ngas,nPART
      use OldTracer_mod, only: trname, t_qlimit, fq_aer, trpdens
      USE TRACER_COM, only:
     *     NTM,n_SO2,n_H2O2,n_H2O2_s
#ifdef TRACERS_SPECIAL_O18
      USE TRACER_COM, only: supsatfac
#endif
#ifdef TRACERS_TOMAS
      USE TRACER_COM, only: NBS,NBINS
     &     ,n_ANUM,n_ASO4,n_ANACL,n_AECIL,n_AECOB
     &     ,n_AOCIL,n_AOCOB,n_ADUST,n_AH2O
#endif
#ifdef TRACERS_HETCHEM
      USE TRACER_COM, only: n_SO4_d1, n_SO4_d2, n_SO4_d3,n_SO4
     *     ,n_N_d1,n_N_d2,n_N_d3,n_NO3p, n_Clay,n_Silt1,n_Silt2
     &     , n_clayilli, n_claykaol, n_claysmec, n_claycalc, n_clayquar
     &     , n_clayfeld, n_clayhema, n_claygyps, n_clayilhe, n_claykahe
     &     , n_claysmhe, n_claycahe, n_clayquhe, n_clayfehe, n_claygyhe
     &     , n_sil1quar, n_sil1feld, n_sil1calc, n_sil1illi, n_sil1kaol
     &     , n_sil1smec, n_sil1hema, n_sil1gyps, n_sil1quhe, n_sil1fehe
     &     , n_sil1cahe, n_sil1gyhe, n_sil1ilhe, n_sil1kahe, n_sil1smhe
     &     , n_sil2quar, n_sil2feld, n_sil2calc, n_sil2hema, n_sil2gyps
     &     , n_sil2illi, n_sil2kaol, n_sil2smec, n_sil2quhe, n_sil2fehe
     &     , n_sil2cahe, n_sil2gyhe, n_sil2ilhe, n_sil2kahe, n_sil2smhe
     &     , ntm_clay
      USE MODEL_COM, only  : dtsrc
      use RunTimeControls_mod, only: tracers_minerals, tracers_nitrate
#endif
      use OldTracer_mod, only: set_fq_aer
      IMPLICIT NONE
C**** Local parameters and variables and arguments:
!@param BY298K unknown meaning for now (assumed= 1./298K)
!@var Ppas pressure at current altitude (in Pascal=kg/s2/m)
!@var TFAC exponential coeffiecient of tracer condensation temperature
!@+   dependence (mole/joule)
!@var FCLOUD fraction of cloud available for tracer condensation
!@var SSFAC dummy variable (assumed units= kg water?)
!@var FQ            fraction of tracer that goes into condensate
!@var FQ0 default fraction of water tracer that goes into condensate
!@var L index for altitude loop
!@var N index for tracer number loop
!@var WMXTR mixing ratio of water available for tracer condensation?
!@var SUPSAT super-saturation ratio for cloud droplets
!@var LHX latent heat flag for whether condensation is to ice or water
!@var RKD dummy variable (= tr_RKD*EXP[ ])
!@var FQAERFAC scales the application of fq_aer (currently < 1 when the
!@+   calls to this routine are for successive states of a rising updraft
!@+   air parcel)
      REAL*8, PARAMETER :: BY298K=3.3557D-3
      REAL*8 Ppas, tfac, RKD,CLDINC,trlef
#ifdef TRACERS_SPECIAL_O18
      real*8 tdegc,alph,fracvs,fracvl,kin_cond_ice
#endif
      REAL*8,  INTENT(IN) :: fq0, FCLOUD, WMXTR, TEMP, TEMP0,LHX
     &     , TR_LEF(NTM), pl,CLDSAVT, FQAERFAC
      REAL*8,  INTENT(IN), DIMENSION(NTM) :: trwml
      REAL*8,  INTENT(IN), DIMENSION(NTM) :: TM
      REAL*8,  INTENT(OUT):: fq(NTM),thlaw(NTM)
      INTEGER, INTENT(IN) :: NTX, ntix(NTM)
      LOGICAL TR_CONV
      REAL*8 :: FQ0FAC,SUPSAT,SSFAC(NTM),SSFAC0
      INTEGER :: N,IHLAW,IAERO,IWAT

!@param list_of_bins  list of dust bin names considered
      character( len=4 ), dimension( 3 ), parameter :: list_of_bins=(/
     &     'Clay', 'Sil1', 'Sil2' /)
!@param vol_ratio_threshold  discrete threshold of ratio between volume of
!@+     coating on dust to volume of dust, above which 100% solubility is
!@+     assumed
      real( kind=8), parameter :: vol_ratio_threshold=0.03d0
!@var vol_dust  dust volume
      real( kind=8 ) :: vol_dust
#ifdef TRACERS_HETCHEM
!@var n_dust_list  array of dust indices
      integer, dimension( ntm_clay ) :: n_dust_list
#endif
      integer :: ind_so4_d, ind_n_d, nn

#ifdef TRACERS_TOMAS
      integer :: k
      real*8,dimension(nbins):: fraction !where to read fraction?
#endif

#if (defined TRACERS_AEROSOLS_Koch) ||\
    (defined TRACERS_SPECIAL_Shindell) ||\
    (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS)
c
c gases with a henry law constant
c
c     cldinc=max(0.,cldsavt-fcloud)
      if(lhx.eq.lhe .and. fcloud.ge.1d-16) then
        Ppas = PL*1.D2          ! pressure to pascals
        tfac = (1.D0/TEMP - BY298K)*BYGASC
        ssfac0 = WMXTR*MAIR*1.D-6*Ppas/(CLDSAVT+teeny)
        ssfac(hlaw_list) = ssfac0*tr_RKD(hlaw_list)
     &    *exp(-tr_DHD(hlaw_list)*tfac)
        if(tr_conv) then ! convective cloud
          fq0fac = 1.
          if (fq0.eq.0.) fq0fac=0.d0
          do ihlaw=1,hlaw_count
            n = hlaw_list(ihlaw)
            fq(n) = fq0fac*ssfac(n) / (1d0 + ssfac(n))
            thlaw(n) = 0.
          enddo
        else             ! stratiform cloud
          do ihlaw=1,hlaw_count
            n = hlaw_list(ihlaw)
            fq(n) = 0.
c limit gas dissolution to incremental cloud change after cloud forms
c   only apply to non-aqueous sulfur species since this is already
c   done in GET_SULFATE
c but H2O2 should be limited if not coupled with sulfate, have not done this
c           if (n.ne.n_h2O2.and.n.ne.n_so2.and.n.ne.n_h2O2_s) then
c           if (FCLOUD.ne.0.) tr_lef(n)=cldinc
c           endif
            trlef=min(tr_lef(n),cldsavt)
            thlaw(n) = min(tm(n),max(0d0,
     &           (ssfac(n)*trlef*tm(n)-TRWML(n))
     &           /(1.D0+ssfac(n)) ))
          enddo
        endif
      else
        fq(hlaw_list) = 0.
        thlaw(hlaw_list) = 0.
      endif
#else
      thlaw(:) = 0. ! set a default to avoid compiler complaints on intent(out)
#endif /* dissolved gases with a henry law constant */

c
c loop over water species
c
      do iwat=1,water_count
        n = water_list(iwat)
        fq(n) = fq0
#ifdef TRACERS_SPECIAL_O18
          if (fq0.gt.0. .and. fq0.lt.1.) then
C**** If process occurs at constant temperature, calculate condensate
C**** in equilibrium with source vapour. Otherwise, use mid-point
C**** temperature and estimate instantaneous fractionation. This gives
C**** a very good estimate to complete integral
C****
            if (abs(temp-temp0).gt.1d-14) then  ! use instantaneous frac
              tdegc=0.5*(temp0 + temp) -tf
C**** Calculate alpha (fractionation coefficient)
                if (LHX.eq.LHE) then ! cond to water
                  alph=1./fracvl(tdegc,ntix(n))
                else            ! cond to ice
                  alph=1./fracvs(tdegc,ntix(n))
C**** kinetic fractionation can occur as a function of supersaturation
C**** this is a parameterisation from Georg Hoffmann
                  supsat=1d0-supsatfac*tdegc
                  if (supsat .gt. 1.) alph=kin_cond_ice(alph,supsat
     *                 ,ntix(n))
                end if
                fq(n) = 1.- (1.-fq0)**alph
            else
C**** assume condensate in equilibrium with vapour at temp
              tdegc=temp -tf
              if (LHX.eq.LHE) then ! cond to water
                alph=1./fracvl(tdegc,ntix(n))
              else              ! cond to ice
                alph=1./fracvs(tdegc,ntix(n))
C**** kinetic fractionation can occur as a function of supersaturation
C**** this is a parameterisation from Georg Hoffmann
                supsat=1d0-supsatfac*tdegc
                if (supsat .gt. 1.) alph=kin_cond_ice(alph,supsat
     *               ,ntix(n))
              end if
              fq(n) = alph * fq0/(1.+(alph-1.)*fq0)
            end if
          else
            fq(n) = fq0
          end if
#endif
      enddo ! end loop over water species

#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_COSMO) ||\
    (defined TRACERS_DUST) || (defined TRACERS_MINERALS) ||\
    (defined TRACERS_AMP) || (defined TRACERS_RADON) ||\
    (defined TRACERS_TOMAS) || (defined TRACERS_AEROSOLS_SEASALT)

c
c aerosols
c

c     if (FCLOUD.lt.1.D-16 .or. fq0.eq.0.) then
      if (CLDSAVT.lt.1.D-16 .or. fq0.eq.0.) then
        fq(aero_list) = 0.      ! defaults to zero
      else

#if (defined TRACERS_AEROSOLS_Koch) && (defined TRACERS_HETCHEM)
#ifdef TRACERS_DUST

      n = n_Clay
      if ( ( TM(ntix(n_SO4_d1)) /trpdens(n_SO4)) >
     *     (( TM(ntix(n))  /trpdens(n)) * 0.03 ) ) then
        call set_fq_aer(NTIX(N), 1.d0)
      else
        call set_fq_aer(NTIX(N), 0.d0)
      endif
#ifdef TRACERS_NITRATE
      if ( ( TM(ntix(n_N_d1)) /trpdens(n_NO3p)) >
     *     (( TM(ntix(n))  /trpdens(n)) * 0.03 ) ) then
        call set_fq_aer(NTIX(N), 1.d0)
      endif
#endif

      n = n_Silt1
      if ( ( TM(ntix(n_SO4_d2)) /trpdens(n_SO4)) >
     *     (( TM(ntix(n))  /trpdens(n)) * 0.03 ) ) then
        call set_fq_aer(NTIX(N), 1.d0)
      else
        call set_fq_aer(NTIX(N), 0.d0)
      endif
#ifdef TRACERS_NITRATE
      if ( ( TM(ntix(n_N_d2)) /trpdens(n_NO3p)) >
     *     (( TM(ntix(n))  /trpdens(n)) * 0.03 ) ) then
        call set_fq_aer(NTIX(N), 1.d0)
      endif
#endif

      n = n_Silt2
      if ( ( TM(ntix(n_SO4_d3)) /trpdens(n_SO4)) >
     *     (( TM(ntix(n))  /trpdens(n)) * 0.03 ) ) then
        call set_fq_aer(NTIX(N), 1.d0)
      else
        call set_fq_aer(NTIX(N), 0.d0)
      endif
#ifdef TRACERS_NITRATE
      if ( ( TM(ntix(n_N_d3)) /trpdens(n_NO3p)) >
     *     (( TM(ntix(n))  /trpdens(n)) * 0.03 ) ) then
        call set_fq_aer(NTIX(N), 1.d0)
      endif
#endif

#endif /* TRACERS_DUST */

#ifdef TRACERS_MINERALS

      if ( tracers_minerals ) then

        do n = 1,size( list_of_bins )

          select case ( list_of_bins( n ) )

          case ('Clay')
            n_dust_list = (/ n_clayilli, n_claykaol, n_claysmec,
     &           n_claycalc,n_clayquar, n_clayfeld, n_clayhema,
     &           n_claygyps,n_clayilhe, n_claykahe, n_claysmhe,
     &           n_claycahe,n_clayquhe, n_clayfehe, n_claygyhe /)
            ind_so4_d = n_SO4_d1
            ind_n_d = n_N_d1
          case ('Sil1')
            n_dust_list = (/ n_sil1illi, n_sil1kaol, n_sil1smec,
     &           n_sil1calc, n_sil1quar, n_sil1feld, n_sil1hema,
     &           n_sil1gyps, n_sil1ilhe, n_sil1kahe, n_sil1smhe,
     &           n_sil1cahe, n_sil1quhe, n_sil1fehe, n_sil1gyhe / )
            ind_so4_d = n_SO4_d2
            ind_n_d = n_N_d2
          case ('Sil2')
            n_dust_list = (/ n_sil2illi, n_sil2kaol, n_sil2smec,
     &           n_sil2calc, n_sil2quar, n_sil2feld, n_sil2hema,
     &           n_sil2gyps, n_sil2ilhe, n_sil2kahe, n_sil2smhe,
     &           n_sil2cahe, n_sil2quhe, n_sil2fehe, n_sil2gyhe /)
            ind_so4_d = n_SO4_d3
            ind_n_d = n_N_d3
          end select

          vol_dust = 0.d0
          do nn = 1,size( n_dust_list )
            vol_dust = vol_dust + tm( ntix( n_dust_list( nn ) ) ) /
     &           trpdens( n_dust_list( nn ) )
          end do

          if ( ( tm( ntix( ind_SO4_d ) ) / trpdens( n_SO4 ) ) > (
     &         vol_dust * vol_ratio_threshold ) ) then
            call set_fq_aer( ntix( ind_SO4_d ), 1.d0 )
          else
            call set_fq_aer( ntix( ind_SO4_d ), 0.d0 )
          end if
          if ( tracers_nitrate ) then
            if ( ( tm( ntix( ind_N_d ) ) / trpdens( n_NO3p ) ) >
     &           ( vol_dust * vol_ratio_threshold ) ) then
              call set_fq_aer( ntix( ind_N_d ), 1.d0 )
            else
              call set_fq_aer( ntix( ind_N_d ), 0.d0 )
            end if
          end if

        end do

      end if

#endif /* TRACERS_MINERALS */

#endif /* TRACERS_AEROSOLS_Koch && TRACERS_HETCHEM */
#ifdef TRACERS_TOMAS

      if(tr_conv)then

         CALL getfraction (.true.,TM,FRACTION) !1% supersaturation assumption

      else                      ! large-scale clouds
         CALL getfraction (.false.,TM,FRACTION) !0.2% supersaturation assumption

      endif

      do k=1,nbins

        call set_fq_aer(ntix(n_ANUM(k)),fraction(k))
        call set_fq_aer(ntix(n_ASO4(k)), fraction(k))
        call set_fq_aer(ntix(n_ANACL(k)),  fraction(k))
        call set_fq_aer(ntix(n_AECIL(k)),fraction(k))
        call set_fq_aer(ntix(n_AECOB(k)),fraction(k))
        call set_fq_aer(ntix(n_AOCIL(k)),fraction(k))
        call set_fq_aer(ntix(n_AOCOB(k)),fraction(k))
        call set_fq_aer(ntix(n_ADUST(k)),fraction(k))
        call set_fq_aer(ntix(n_AH2O(k)), fraction(k))

         if (fraction(k).gt.1.or.fraction(k).lt.0) then
            print*,'fraction>1 or fraction<0'
            call stop_model('wrong fraction',255)
         endif

      enddo
#endif

      cldinc=cldsavt-fcloud
      if(tr_conv) then          ! convective cloud
c complete dissolution in convective clouds
c with double dissolution if partially soluble
        if(lhx.eq.lhe) then
          fq(aero_list) = fq_aer(aero_list)*fqaerfac
        else
          fq(aero_list) = fq_aer(aero_list)*fqaerfac*0.12d0
        endif
c this should not work because cldinc should be fcld for
c    when cloud first forms
      elseif(fq0.gt.0 .and. cldinc.gt.0.) then ! growing stratiform cloud
        if(lhx.eq.lhe) then
          fq(aero_list) = fq_aer(aero_list)*cldinc
        else
          fq(aero_list) = fq_aer(aero_list)*cldinc*0.12d0
        endif
      else
        fq(aero_list) = 0.
      endif
      where(fq(aero_list).ge.1.d0) fq(aero_list)=0.9999
c
c use this code in place of the above if the commented-out formulas
c for (dust?) fq are reinstated
c

c      do iaero=1,aero_count ! loop over aerosols
c        n = aero_list(iaero)
cc complete dissolution in convective clouds
cc with double dissolution if partially soluble
c          if (TR_CONV) then ! convective cloud
c            if (LHX.EQ.LHE) then !liquid cloud
ccdust #if (defined TRACERS_DUST) || (defined TRACERS_MINERALS)
ccdust           IF (fq_aer(ntix(n)) > 0.)
ccdust #endif
c              fq(n)=fq_aer(ntix(n))
ccdust?              fq(n)=(1.d0+fq_aer(ntix(n)))/2.d0
ccdust?              fq(n)=(1.d0+3.d0*fq_aer(ntix(n)))/4.d0
c            else
ccdust #if (defined TRACERS_DUST) || (defined TRACERS_MINERALS)
ccdust           IF (fq_aer(ntix(n)) > 0.)
ccdust #endif
c              fq(n)=fq_aer(ntix(n))*0.12d0
ccdust?              fq(n)=(1.d0+fq_aer(ntix(n)))/2.d0*0.05d0
ccdust?              fq(n)=(1.d0+3.d0*fq_aer(ntix(n)))/4.d0*0.05d0
c
c            endif
c          elseif (fq0.gt.0.and.CLDINC.gt.0.) then ! stratiform cloud.
cc only dissolve if the cloud has grown
c            if(LHX.EQ.LHE) then !liquid cloud
c              fq(n) = fq_aer(ntix(n))*CLDINC
c            else                ! ice cloud - small dissolution
c              fq(n) = fq_aer(ntix(n))*CLDINC*0.12d0
c            endif
c          endif
c          if (fq(n).ge.1.d0) fq(n)=0.9999
c      enddo ! end loop over aerosols

      endif ! fcloud>0 and fq0.ne.0

#endif /* aerosols */

      RETURN
      END SUBROUTINE GET_COND_FACTOR


      SUBROUTINE GET_WASH_FACTOR(NTX,b_beta_DT,PREC,fq
     * ,TEMP,LHX,WMXTR,FCLOUD,TM,TRPR,THLAW,pl,ntix,BELOW_CLOUD
#ifdef TRACERS_TOMAS
     * ,I,J,L
#endif
     *)
!@sum  GET_WASH_FACTOR calculation of the fraction of tracer
!@+    scavanged by precipitation below convective clouds ("washout").
!@auth Dorothy Koch (modelEifications by Greg Faluvegi)
! NOTE: THLAW is only computed for the tracers in hlaw_list!
! NOTE: FQ is only computed for the tracers in aero_list!
c
C**** GLOBAL parameters and variables:
      use OldTracer_mod, only: nWATER, ngas, nPART, tr_wd_type
      use OldTracer_mod, only: tr_RKD, tr_DHD, rc_washt, trname
      USE TRACER_COM, only:
     * NTM

      USE TRACER_COM, only :
     &     aero_count,water_count,hlaw_count,
! NB: these lists are often used for implicit loops
     &     aero_list,water_list,hlaw_list
#ifdef TRACERS_TOMAS
      USE TRACER_COM, only :
     &     NBS,NBINS,n_ANUM,n_ASO4,n_ANACL
     &    ,n_AOCOB,n_AECIL,n_AECOB,n_AOCIL,n_ADUST,n_AH2O
      use OldTracer_mod, only: set_rc_washt
#endif
      USE CONSTANT, only: BYGASC,LHE,MAIR,teeny,pi

      IMPLICIT NONE
c
C**** Local parameters and variables and arguments:
!@var FQ fraction of tracer scavenged by below-cloud precipitation
!@param rc_wash aerosol washout rate constant (mm-1)
!@var PREC precipitation amount from layer above for washout (mm)
!@var b_beta_DT precipitating grid box fraction from lowest
!@+   percipitating layer.
!@+   The name was chosen to correspond to Koch et al. p. 23,802.
!@var N index for tracer number loop
      INTEGER, INTENT(IN) :: NTX,ntix(NTM)
      REAL*8, INTENT(OUT), DIMENSION(NTM) :: THLAW
      REAL*8, INTENT(INOUT), DIMENSION(NTM) :: FQ
      REAL*8, INTENT(IN) :: PREC,b_beta_DT,TEMP,LHX,WMXTR,FCLOUD,
     *  TM(NTM),pl, TRPR(NTM)
      REAL*8, PARAMETER :: BY298K=3.3557D-3
      REAL*8 Ppas, tfac, ssfac0, ssfac(NTM), bb_tmp
      INTEGER :: N,IHLAW,IAERO
      LOGICAL BELOW_CLOUD
C
#ifdef TRACERS_TOMAS
      integer k,i,j,l
!@var scavr/stratsav : below-cloud scavenging coefficient (per mm rain)
      real*8 scavr
      real stratscav
!@var dpaero : aerosol diameter [m]
      real*8 dpaero,mtot
      real*8,dimension(nbins) ::  getdp,density
#endif

c
c gases with a henry law constant
c
c      fq(hlaw_list) = 0.D0
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_SPECIAL_Shindell) ||\
    (defined TRACERS_AMP) || (defined TRACERS_TOMAS)
      if(      LHX.EQ.LHE ! if not frozen
     &   .AND. FCLOUD.GE.1D-16 .AND. WMXTR.GT.0. AND . BELOW_CLOUD) THEN
        bb_tmp = max(b_beta_DT,0.d0) ! necessary check?
        Ppas = PL*1.D2          ! pressure to pascals
        tfac = (1.D0/TEMP - BY298K)*BYGASC
        ssfac0 = WMXTR*MAIR*1.D-6*Ppas/(FCLOUD+teeny)
        ssfac(hlaw_list) = ssfac0*tr_RKD(hlaw_list)
     &    *exp(-tr_DHD(hlaw_list)*tfac)
        do ihlaw=1,hlaw_count
          n = hlaw_list(ihlaw)
          thlaw(n) = min( tm(n),max( 0d0,(FCLOUD*
     &               ssfac(n)*tm(n)-TRPR(n))/(1.D0+ssfac(n)) ))
        enddo
      else
        thlaw(hlaw_list) = 0.
      endif
#else
      thlaw(:) = 0. ! set a default to avoid compiler complaints on intent(out)
#endif
#ifdef TRACERS_TOMAS

      if(FCLOUD.GE.1.D-16 .and. prec.gt.0.) then

         call dep_getdp(i,j,l,getdp,density) !1 for tempk, dummy=vs
         do k=1,nbins
            dpaero=getdp(k)
            scavr=stratscav(dpaero)  
            call set_rc_washt(ntix(n_ASO4(k)), scavr)
            call set_rc_washt(ntix(n_ANACL(k)),  scavr)
            call set_rc_washt(ntix(n_AECOB(k)),scavr)
            call set_rc_washt(ntix(n_AECIL(k)),scavr)
            call set_rc_washt(ntix(n_AOCOB(k)),scavr)
            call set_rc_washt(ntix(n_AOCIL(k)),scavr)
            call set_rc_washt(ntix(n_ADUST(k)),scavr)
            call set_rc_washt(ntix(n_AH2O(k)), scavr)
            call set_rc_washt(ntix(n_ANUM(k)),scavr)
         enddo

      endif

#endif
c
c water species
c
c      fq(water_list) = 0d0

#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_COSMO) ||\
    (defined TRACERS_DUST) || (defined TRACERS_MINERALS) ||\
    (defined TRACERS_AEROSOLS_SEASALT) ||\
    (defined TRACERS_AMP) ||\
    (defined TRACERS_RADON) || (defined TRACERS_TOMAS)
c
c aerosols
c
      if(FCLOUD.GE.1.D-16 .and. prec.gt.0.) then
        bb_tmp = max(b_beta_DT,0.d0) ! necessary check?
        do iaero=1,aero_count
          n = aero_list(iaero)
          fq(n) = bb_tmp*(1d0-exp(-prec*rc_washt(n)))
        enddo
      else
        fq(aero_list) = 0.
      endif
#endif

      RETURN
      END SUBROUTINE GET_WASH_FACTOR

      SUBROUTINE GET_EVAP_FACTOR(
     &     NTX,TEMP,LHX,HEFF,FQ0,fq,ntix)
!@sum  GET_EVAP_FACTOR calculation of the evaporation fraction
!@+    for tracers.
!@auth Dorothy Koch (modelEifications by Greg Faluvegi)
c
C**** GLOBAL parameters and variables:
      USE CONSTANT, only : tf,lhe
      use OldTracer_mod, only: tr_wd_type,nwater,trname
      USE TRACER_COM, only: NTM, tr_evap_fact, water_count,water_list
c      USE CLOUDS, only: NTIX
c
      IMPLICIT NONE
c
C**** Local parameters and variables and arguments:
!@var FQ            fraction of tracer evaporated
!@var FQ0 [default] fraction of tracer evaporated
!@var N index for tracer number loop
      INTEGER, INTENT(IN) :: NTX,ntix(NTM)
      REAL*8,  INTENT(OUT):: FQ(NTM)
      REAL*8,  INTENT(IN) :: FQ0,TEMP,LHX
!@var HEFF effective relative humidity for evap occuring below cloud
      REAL*8, INTENT(IN) :: HEFF
#ifdef TRACERS_SPECIAL_O18
      real*8 tdegc,alph,fracvl,fracvs,kin_evap_prec
      integer :: iwat
#endif
      integer :: n
c

      if(fq0.ge.1d0) then
        fq(1:ntx) = 1d0 ! total evaporation
      else
        do n = 1, ntx
          fq(n) = fq0*tr_evap_fact(tr_wd_type(ntix(n)))
        end do
      endif

#ifdef TRACERS_SPECIAL_O18
c overwrite fq for water isotopes
      tdegc=temp-tf
      do iwat=1,water_count
        n = water_list(iwat)
        if (lhx.eq.lhe) then
          alph=fracvl(tdegc,ntix(n))
C**** kinetic effects with evap into unsaturated air
          if (heff.lt.1.)
     &         alph=kin_evap_prec(alph,heff,ntix(n))
        else
C**** no fractionation for ice evap
          alph=1.
        end if
        if (fq0.ne.1.) then
          fq(n) = 1. - (1.-fq0)**alph
        else
          fq(n) = fq0
        end if
      enddo
#endif

      RETURN
      END SUBROUTINE GET_EVAP_FACTOR

#endif


#if (defined TRACERS_SPECIAL_Shindell) || (defined TRACERS_AEROSOLS_Koch) ||\
    (defined TRACERS_AMP) || (defined TRACERS_TOMAS)
      SUBROUTINE GET_SULF_GAS_RATES(i,j)
!@sum  GET_SULF_GAS_RATES calculation of rate coefficients for
!@+    gas phase sulfur oxidation chemistry
!@vers 2013/03/26
!@auth Bell
      USE RESOLUTION, only : im,jm,lm
      use atmcol_com, only: tl   ! layer temperature (K)
      use atmcol_com, only: pl   ! layer pressure (mb)
      USE TRACER_COM, only: rsulf1,rsulf2,rsulf3,rsulf4
      implicit none
      integer, intent(in) :: i,j
!
      integer :: l
      real*8 ppres,tt,dmm,rk4,ek4,f
C Greg: certain things now done outside the loops for speed:
      real*8, parameter ::  a= 73.41463d20, ! 6.02d20/.082d0
     *     aa=1.d-20,
     *     b= 0.357d-22,         ! 1.7d-22*0.21d0*1.d-20/aa
     *     c= 1.155d-11,         ! 5.5d-20*0.21d0*1.d-11/aa
     *     d= 4.0d-11            ! 4.0d-20*1.d-11/aa

C Reactions
C***1.DMS + OH -> 0.75SO2 + 0.25MSA
C***2.DMS + OH -> SO2
C***3.DMS + NO3 -> HNO3 + SO2
C***4.SO2 + OH -> SO4 + HO2


      do l=1,LM

C Calculate effective temperature

        ppres=pl(l)*9.869d-4 !in atm
        tt = 1.d0/tl(l)

c DMM is number density of air in molecules cm-3

        dmm=ppres*tt*a
        rsulf1(l) =
     & b*dmm*exp(7810.d0*tt)*aa/(1.d0+c*exp(7460.d0*tt)*dmm*aa)

        rsulf2(l) = 9.6d-12*exp(-234.d0*tt)

        rsulf3(l) = 1.9d-13*exp(520.d0*tt)

        rk4 = aa*((tt*300.d0)**(3.3d0))*dmm*d
        f=log10(0.5d12*rk4)
        ek4 = 1.d0/(1.d0 + (f*f))

        rsulf4(l) = (rk4/(1.d0 + 0.5d12*rk4  ))*(0.45d0**ek4)

      enddo

      END SUBROUTINE GET_SULF_GAS_RATES
#endif

#ifdef TRACERS_TOMAS

!    **************************************************
!@sum  initbounds
!    **************************************************
!@+    This subroutine initializes the array, xk, which describes the
!@+    boundaries between the aerosol size bins.  xk is in terms of dry
!@+    single-particle mass (kg).  The aerosol microphysics algorithm
!@+    used here assumes mass doubling such that each xk is twice the
!@+    previous.

!@auth Peter Adams, November 1999 (Modified by Yunha Lee)

      SUBROUTINE initbounds()



C-----INCLUDE FILES-----------------------------------------------------

      USE TRACER_COM,only : nbins
      USE TOMAS_AEROSOL, only: xk,sqrt_xk_xk1

C-----VARIABLE DECLARATIONS---------------------------------------------
      IMPLICIT NONE

      integer k
!@var Mo : lower mass bound for first size bin (kg)
      real*8 Mo

C-----ADJUSTABLE PARAMETERS---------------------------------------------

#if (defined TOMAS_12_3NM) || (defined TOMAS_15_3NM)
      parameter(Mo=1.5625d-23) ! 3nm
#else
      parameter(Mo=1.0d-21)    ! 10nm
#endif


C-----CODE--------------------------------------------------------------

      do k=1,nbins+1
!YUNHA LEE - working on adding more version of TOMAS (Aug, 2012)
#if (defined TOMAS_12_10NM) || (defined TOMAS_12_3NM)
         if(k.lt.nbins)then
            xk(k)=Mo*4.d0**(k-1)
         else
            xk(k)=xk(k-1)*32.d0
         endif
#elif (defined TOMAS_15_10NM) || (defined TOMAS_15_3NM)
           xk(k)=Mo*4.d0**(k-1)
#elif (defined TOMAS_30_10NM) || (defined TOMAS_30_3NM)
           xk(k)=Mo*2.d0**(k-1)
#endif
        if (k>1) sqrt_xk_xk1(k-1)=sqrt(xk(k-1)*xk(k))
      enddo

      RETURN
      END


!    **************************************************
!@sum   momentfix
!    **************************************************
!@+    This routine changes the first and second order moments of a
!@+    given tracer's distribution such that they match the shape of
!@+    another specified tracer.  Since the zeroth order moment is
!@+    unchanged, the routine conserves mass.

!@auth Peter Adams

C-----INPUTS------------------------------------------------------------

!@var     pn - the number of the tracer that will serve as the pattern
!@var     fn - the number of the tracer whose moments will be fixed

C-----OUTPUTS-----------------------------------------------------------

      SUBROUTINE momentfix(pn,fn)

C-----INCLUDE FILES-----------------------------------------------------

      USE QUSDEF, only : nmom
      USE RESOLUTION, ONLY : IM,JM,LM
      USE TRACER_COM, only : trm, trmom
      USE DOMAIN_DECOMP_ATM, only : GRID, getDomainBounds
      USE GEOM, only : imaxj

      integer pn, fn

C-----VARIABLE DECLARATIONS---------------------------------------------

      integer i,j,l,n
      INTEGER I_0, I_1, J_0, J_1
      real*8 ratio

C-----CODE--------------------------------------------------------------
C****
C**** Extract useful local domain parameters from "grid"
C****
      call getDomainBounds(grid, J_STRT=J_0, J_STOP=J_1,
     $     I_STRT=I_0, I_STOP=I_1)

      do l=1,lm; do j=J_0,J_1; do i=I_0,imaxj(j)
         if (TRM(i,j,l,pn) .ge. 1.e-10) then
            ratio=TRM(i,j,l,fn)/TRM(i,j,l,pn)
            do n=1,nmom
               trmom(n,i,j,l,fn)=ratio*trmom(n,i,j,l,pn)
            enddo
         else

            do n=1,nmom
               trmom(n,i,j,l,fn)=0.0
               trmom(n,i,j,l,pn)=0.0
            enddo
         endif
      end do; end do; end do

      RETURN
      END



!    **************************************************
!@sum  stratscav
!@    **************************************************
!@+    This function is basically a lookup table to get the below-cloud
!@+    scavenging rate (per mm of rainfall) as a function of particle
!@+    diameter.  The data are taken from Dana, M. T., and
!@+    J. M. Hales, Statistical Aspects of the Washout of Polydisperse
!@+    Aerosols, Atmos. Environ., 10, 45-50, 1976.  I am using the
!@+    monodisperse aerosol curve from Figure 2 which assumes a
!@+    lognormal distribution of rain drops with Rg=0.02 cm and a
!@+    sigma of 1.86, values typical of a frontal rain spectrum
!@+    (stratiform clouds).

!@auth  Peter Adams, January 2001

      real FUNCTION stratscav(dp)

      IMPLICIT NONE

C-----ARGUMENT DECLARATIONS------------------------------------------
!@var dp : particle diameter [m]
      real*8 dp

C-----VARIABLE DECLARATIONS------------------------------------------
!@param numpts : number of points in lookup table
!@var dpdat : particle diameter in lookup table [m]
!@var scdat : scavenging rate in lookup table [mm-1]
!@var n1/n2 : indices of nearest data points
      integer numpts
      real dpdat
      real scdat
      integer n1, n2

C-----VARIABLE COMMENTS----------------------------------------------

C-----ADJUSTABLE PARAMETERS------------------------------------------

      parameter(numpts=37)
      dimension dpdat(numpts), scdat(numpts)

      data dpdat/ 2.0E-09, 4.0E-09, 6.0E-09, 8.0E-09, 1.0E-08,
     &            1.2E-08, 1.4E-08, 1.6E-08, 1.8E-08, 2.0E-08,
     &            4.0E-08, 6.0E-08, 8.0E-08, 1.0E-07, 1.2E-07,
     &            1.4E-07, 1.6E-07, 1.8E-07, 2.0E-07, 4.0E-07,
     &            6.0E-07, 8.0E-07, 1.0E-06, 1.2E-06, 1.4E-06,
     &            1.6E-06, 1.8E-06, 2.0E-06, 4.0E-06, 6.0E-06,
     &            8.0E-06, 1.0E-05, 1.2E-05, 1.4E-05, 1.6E-05,
     &            1.8E-05, 2.0E-05/

      data scdat/ 6.99E-02, 2.61E-02, 1.46E-02, 9.67E-03, 7.07E-03,
     &            5.52E-03, 4.53E-03, 3.87E-03, 3.42E-03, 3.10E-03,
     &            1.46E-03, 1.08E-03, 9.75E-04, 9.77E-04, 1.03E-03,
     &            1.11E-03, 1.21E-03, 1.33E-03, 1.45E-03, 3.09E-03,
     &            4.86E-03, 7.24E-03, 1.02E-02, 1.36E-02, 1.76E-02,
     &            2.21E-02, 2.70E-02, 3.24E-02, 4.86E-01, 8.36E-01,
     &            1.14E+00, 1.39E+00, 1.59E+00, 1.75E+00, 1.85E+00,
     &            1.91E+00, 1.91E+00/

C-----CODE-----------------------------------------------------------

C If particle diameter is in bounds, interpolate to find value
      if ((dp .gt. dpdat(1)) .and. (dp .lt. dpdat(numpts))) then
         !loop over lookup table points to find nearest values
         n1=1
         do while (dp .gt. dpdat(n1+1))
            n1=n1+1
         enddo
         n2=n1+1
         stratscav=scdat(n1)+(scdat(n2)-scdat(n1))
     &             *(dp-dpdat(n1))/(dpdat(n2)-dpdat(n1))
      endif

C If particle diameter is out of bounds, return reasonable value
!YUNHA - (TOMAS bug) I changed the condition that gt ==> ge.  lt ==> le
!YUNHA - Because stratscav has no value when dp=dpdat(numpts) and dpdat(1).

      if (dp .ge. dpdat(numpts)) stratscav=2.0
      if (dp .le. dpdat(1))      stratscav=7.0e-2

      RETURN
      END FUNCTION stratscav


#endif

!--------------------------------------------------------------------
! sph_mod -- generate spherical harmonics
!--------------------------------------------------------------------
      module sph_mod
      implicit none

      private

      type, public :: sph
        private
        real*8 :: base
        real*8, dimension(:), allocatable :: coeff
        integer :: l, m
      contains
        procedure :: build
        procedure :: disp
        procedure :: value
      end type sph

      contains

      function binomial(n, k)
      implicit none
      integer, intent(in) :: n, k
      integer :: binomial
      integer :: i, kk

      kk=merge(k, n-k, (k<n-k).and.(k<=n))
      binomial=1
      do i=1, kk
        binomial=binomial*(n+1-i)/i
      end do
      return
      end function binomial

      subroutine deriv(a)
      implicit none
      real*8, dimension(:), intent(inout) :: a
      integer :: i

      do i=1, size(a)-1
        a(i)=a(i+1)*i
      end do
      a(size(a))=0
      return
      end subroutine deriv

      subroutine mult(a, b, res)
      implicit none
      real*8, dimension(:), intent(in) :: a, b
      real*8, dimension(:), allocatable, intent(out) :: res
      integer :: i, j, sz, r0, r1
      real*8 :: val

      sz=size(a)+size(b)-1
      allocate(res(sz))
      do i=0, sz-1
        r0=max(0, i+1-size(b))
        r1=min(i, size(a)-1)
        val=0
        do j=r0, r1
          val=val+a(j+1)*b(i-j+1)
        end do
        res(i+1)=val
      end do
      return
      end subroutine mult

      subroutine legendre_poly(n, coeff)
      implicit none
      integer, intent(in) :: n
      real*8, dimension(:), allocatable, intent(out) :: coeff
      integer :: i, j, k
      real*8 :: val

      allocate(coeff(n+1))
      coeff=0
      do i=mod(n, 2), n, 2
        val=1
        k=merge(i, n-i, i<n-i)
        do j=1, k
          val=val*(n+1-j)/j
        end do
        do j=1, n
          val=val*(n+i+1-j*2)/j
        end do
        coeff(i+1)=val
      end do
      return
      end subroutine legendre_poly

!----------------------
! build(l, m) builds the spherical harmonic function l, m
!----------------------
      subroutine build(this, l, m)
      implicit none
      class(sph), intent(inout) :: this
      integer, intent(in) :: l, m
      integer :: i, absm
      real*8, parameter :: pi=4.d0*datan(1.d0)
      real*8, dimension(:), allocatable :: coeff
      real*8 :: term

      this%l=l
      this%m=m
      absm=abs(m)
      term=1.
      do i=l-absm+1, l+absm
        term=term*i
      end do
      this%base=sqrt((.5*l+.25)/term/pi)
      if (m/=0) this%base=this%base*sqrt(2.)
      call legendre_poly(l, coeff)
      do i=1, absm
        call deriv(coeff)
      end do
      allocate(this%coeff(l+1-absm))
      this%coeff(:)=coeff(1:(l+1-absm))
      return
      end subroutine build

      subroutine disp(this)
      implicit none
      class(sph), intent(inout) :: this
      integer :: i

      write(*,*)'Y',this%l,this%m,'= ...'
      write(*,*)this%base,'*'
      if (this%m/=0) then
        write(*,*)'sin'
        write(*,*)'^',abs(this%m)
        write(*,*)'(t)*'
      end if
      write(*,*)'(',this%coeff(1)
      do i=2, size(this%coeff)
        write(*,*)'+',this%coeff(i),'*cos^',i-1,'(t)'
      end do
      write(*,*)')'
      if (this%m<0) write(*,*)'*sin(',abs(this%m),'*p)'
      if (this%m>0) write(*,*)'*cos(',abs(this%m),'*p)'
      return
      end subroutine disp

!----------------------
! value(t, p) value at a given point (in spherical coordinates theta, phi)
!----------------------
      function value(this, t, p)
      implicit none
      class(sph), intent(in) :: this
      real*8, intent(in) :: t, p
      real*8 :: value
      integer :: i

      value=0.
      do i=1, size(this%coeff)
        value=value+this%coeff(i)*cos(t)**(i-1)
      end do
      value=value*this%base
      if (this%m/=0) then
        value=value*sin(t)**abs(this%m)
      end if
      if (this%m<0) value=value*sin(-this%m*p)
      if (this%m>0) value=value*cos(this%m*p)
      return
      end function value

      end module sph_mod

!--------------------------------------------------------------------
!--------------------------------------------------------------------


      subroutine src_dist_config
      use resolution, only : im,jm
      USE DOMAIN_DECOMP_atm, ONLY : GRID, getdomainbounds, am_i_root
      use tracer_com, only: xyztr, ntm_sph, ntm_reg
      use dictionary_mod, only: get_param, is_set_param
      implicit none
      include 'netcdf.inc'
      interface
        subroutine src_dist_config_sph(l, m, n, arr)
        integer, intent(in) :: l, m, n
        real*8, dimension(:, :, :), allocatable, intent(inout) :: arr
        end subroutine src_dist_config_sph
      end interface

      integer :: status,fid,vid,did,srt(3),cnt(3)
c
      character(len=80) :: xyzfile='src_dist_cfg'
      integer, dimension(im,jm) :: regions
      real*8, dimension(:,:,:), allocatable :: xyztr_sph
      integer :: i,j,n,l,m,ntm,n_sph
      INTEGER :: J_0, J_1
C****
C**** Extract useful local domain parameters from "grid"
C****
      CALL getdomainbounds(grid, J_STRT=J_0, J_STOP=J_1)
      ntm_sph = 1  ! always keep trname(1)='Water   '
      ntm_reg = 0
      status = nf_open(trim(xyzfile),nf_nowrite,fid)
      if(status.ne.nf_noerr) then
        if(am_i_root())
     &       write(6,*) 'input file ',trim(xyzfile),' not found'
        call stop_model('missing input file in src_dist_config',255)
      endif
      status = nf_inq_dimid(fid,'tracer',did)
      if(status.eq.nf_noerr) then
        status = nf_inq_dimlen(fid,did,ntm_sph)
      endif
      status = nf_inq_varid(fid,'regions',vid)
      if(status.eq.nf_noerr) then
        status = nf_get_var_int(fid,vid,regions)
        ntm_reg = maxval(regions)
        do n=1,ntm_reg
          if(.not.any(regions.eq.n)) then
            if(am_i_root()) write(6,*) 'missing region ',n
            call stop_model('missing region in src_dist_config',255)
          endif
        enddo
      endif
      n_sph=0
      if (is_set_param('src_dist_sph'))
     &   call get_param('src_dist_sph', n_sph)
      ntm=ntm_sph+ntm_reg+n_sph**2
      allocate(xyztr(ntm,IM,J_0:J_1))

c read spherical harmonic basis functions
      status = nf_inq_varid(fid,'xyztr',vid)
      if(status.eq.nf_noerr .and. ntm_sph.gt.1) then
        srt(1:3) = (/ 1, 1, j_0 /)
        cnt(1:3) = (/ ntm_sph, im, 1+j_1-j_0 /)
        if(ntm_reg.gt.0) then
          allocate(xyztr_sph(ntm_sph,IM,J_0:J_1))
          status = nf_get_vara_double(fid,vid,srt,cnt,xyztr_sph)
        else
          status = nf_get_vara_double(fid,vid,srt,cnt,xyztr)
        endif
      else  ! if no xyztr in file, retain normal water as a tracer
        xyztr(1,:,:) = 1d0
      endif
      status = nf_close(fid)
c copy spherical funcs and region tracers into xyztr
      do j=j_0,j_1
        do i=1,im
          if(ntm_sph.gt.1 .and. ntm_reg.gt.0) then
            xyztr(1:ntm_sph,i,j)=xyztr_sph(1:ntm_sph,i,j)
          endif
          if(ntm_reg.gt.0) then
            xyztr(ntm_sph+1:ntm,i,j) = 0d0
            xyztr(ntm_sph+regions(i,j),i,j) = 1d0
          endif
        enddo
      enddo
      do l=0, n_sph-1
        call src_dist_config_sph(l, 0,
     &                  ntm_sph+ntm_reg+l**2+1, xyztr)
        do m=1, l
          call src_dist_config_sph(l, m,
     &                  ntm_sph+ntm_reg+l**2+m*2, xyztr)
          call src_dist_config_sph(l, -m,
     &                  ntm_sph+ntm_reg+l**2+m*2+1, xyztr)
        end do
      end do
      return
      end subroutine src_dist_config

      subroutine src_dist_config_sph(l, m, n, arr)
      use sph_mod, only: sph
      use geom, only: lat_dg,lon_dg
      use constant, only: pi
      use domain_decomp_atm, only : grid, getdomainbounds
      implicit none
      integer, intent(in) :: l, m, n
      real*8, dimension(:, :, :), allocatable, intent(inout) :: arr
      integer :: i, j, i0, i1, j0, j1
      real*8, parameter :: factor=sqrt(4.*pi)
      type(sph) :: gen

      call getdomainbounds(grid,
     &                 i_strt=i0, i_stop=i1, j_strt=j0, j_stop=j1)
      call gen%build(l, m)
      do i=i0, i1
        do j=j0, j1
          arr(n,i,j)=gen%value((90.-lat_dg(j,1))*pi/180.,
     &                                 lon_dg(i,1)*pi/180.)*factor
        end do
      end do

      return
      end subroutine src_dist_config_sph

      subroutine init_src_dist
      use resolution, only : im
      use tracer_com, only: ntm, xyztr, ntm_sph, ntm_reg
      use domain_decomp_atm, only : grid, getdomainbounds
      use fluxes, only: focean, asflx
      use oldtracer_mod, only: src_dist_index
      implicit none
      integer :: i, j, n, j_0, j_1

      if (allocated(xyztr)) then
        call getdomainbounds(grid, j_strt=j_0, j_stop=j_1)
        do j=j_0, j_1
          do i=1, im
            do n=1, ntm_reg
              xyztr(n+ntm_sph, i, j)=xyztr(n+ntm_sph, i, j)*focean(i, j)
            enddo
          enddo
        enddo
        do i=1, size(asflx)
          do n=1, ntm
            if (src_dist_index(n)>0) asflx(i)%gtracer(n, :, j_0:j_1)=
     &                             xyztr(src_dist_index(n), :, j_0:j_1)
          enddo
        enddo
      endif
      end subroutine init_src_dist
