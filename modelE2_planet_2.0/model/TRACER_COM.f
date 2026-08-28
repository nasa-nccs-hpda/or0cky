#include "rundeck_opts.h"
      

!@sum  TRACER_COM: Exists alone to minimize the number of dependencies
!@+    This version for simple trace gases/chemistry/isotopes
!@auth Jean Lerner/Gavin Schmidt

      MODULE TRACER_COM
!@sum  TRACER_COM tracer variables
!@auth Jean Lerner
C
      USE QUSDEF, only: NMOM
      USE RESOLUTION, only: lm
      use OldTracer_mod, only: itime_tr0, tr_mm
      use OldTracer_mod, only: tr_wd_type
      use OldTracer_mod, only: tr_RKD
      use OldTracer_mod, only: nGAS, nPART, nWATER
      use timestream_mod, only : timestream
#ifdef TRACERS_AEROSOLS_VBS
      use TRACERS_VBS, only: vbs_bins
#endif

      use TracerBundle_mod, only: TracerBundle, newTracerBundle
      use vector_integer_mod, only: vector_integer=>vector
c     
      IMPLICIT NONE
      SAVE

      type (TracerBundle), target :: tracers
      type (TracerBundle) :: shindellTracers
      type (TracerBundle) :: lernerTracers
      type (TracerBundle) :: tomasTracers
      type (TracerBundle) :: ampTracers

!@dbparam COUPLED_CHEM: 0 means uncoupled; read files, 1 means coupled
!@+ use chemistry and aerosol codes, -1 means uncoupled and use zero
!@+ aerosols and don't read from files.
      integer :: COUPLED_CHEM = 0

!@dbparam emiss_over_model_top_at_LM .false. means emissions over the model top
!@+       are not allowed; .true. means emissions over the model top are dumped
!@+       at the topmost model layer
      logical :: emiss_over_model_top_at_LM=.false.

! simple aerosol & gas injections based on rundeck parameters
!@dbparam direct_inject_num Number of injections
!@dbparam ex_volc_num repeat of direct_inject_num for backward compatibility
!@dbparam direct_inject_hr0 UTC hour to start the injection (default 0)
!@dbparam direct_inject_hr1 UTC hour of final day to end the injection (default INT_HOURS_PER_DAY)
!@dbparam direct_inject_jday Julian day that injection begins
!@dbparam direct_inject_ndays Number of days injection lasts (int, def=1, with const rate)
!@dbparam direct_inject_year Year of injections
!@dbparam direct_inject_pointlat Latitude of injection (single-column point injection)
!@dbparam direct_inject_pointlon Longitude of injection (single-column point injection)
!@dbparam direct_inject_rectlat0 Southernmost latitude of injections (rectangular region)
!@dbparam direct_inject_rectlat1 Northernmost latitude of injections (rectangular region)
!@dbparam direct_inject_rectlon0 Westernmost longitude of injections (rectangular region)
!@dbparam direct_inject_rectlon1 Easternmost longitude of injections (rectangular region)
!@dbparam direct_inject_bot Plume bottom (in m) of injections
!@dbparam direct_inject_top Plume top (in m) of injectionss
!@dbparam direct_inject_SO2 SO2 emissions (in Tg day-1) of volcanoes
!@dbparam direct_inject_H2O H2O emissions (in Tg day-1) of volcanoes
!@dbparam direct_inject_SU SO4 direct emissions (in Tg day-1) for simplified volc case
!@dbparam direct_inject_DD1 DU emissions (fine, in Tg day-1) as volc ash
!@dbparam direct_inject_DD2 DU emissions (coarse, in Tg day-1) as volc ash
!@dbparam direct_inject_BC BC emissions (in Tg day-1) of wildfires
!@dbparam direct_inject_OC OC emissions (in Tg day-1) of wildfires
      integer                            :: direct_inject_num
      integer                            :: ex_volc_num
      integer, allocatable, dimension(:) :: direct_inject_hr0
      integer, allocatable, dimension(:) :: direct_inject_hr1
      integer, allocatable, dimension(:) :: direct_inject_jday
      integer, allocatable, dimension(:) :: direct_inject_ndays
      integer, allocatable, dimension(:) :: direct_inject_year
      real*8,  allocatable, dimension(:) :: direct_inject_pointlat
      real*8,  allocatable, dimension(:) :: direct_inject_pointlon
      real*8,  allocatable, dimension(:) :: direct_inject_rectlat0
      real*8,  allocatable, dimension(:) :: direct_inject_rectlat1
      real*8,  allocatable, dimension(:) :: direct_inject_rectlon0
      real*8,  allocatable, dimension(:) :: direct_inject_rectlon1
      real*8,  allocatable, dimension(:) :: direct_inject_bot
      real*8,  allocatable, dimension(:) :: direct_inject_top
      real*8,  allocatable, dimension(:) :: direct_inject_SO2
      real*8,  allocatable, dimension(:) :: direct_inject_H2O
      real*8,  allocatable, dimension(:) :: direct_inject_SU
#ifdef TRACERS_AMP
      real*8,  allocatable, dimension(:) :: direct_inject_DD1
      real*8,  allocatable, dimension(:) :: direct_inject_DD2
#endif
      real*8,  allocatable, dimension(:) :: direct_inject_BC
      real*8,  allocatable, dimension(:) :: direct_inject_OC

#ifdef WATER_MISC_GRND_CH4_SRC
!@dbparam scale_CH4MGOL global scaling available to either Shindell
!@+ or Lerner tracers for water and misc. ground source of CH4
      real*8 :: scale_CH4MGOL=1.d0
#endif

!@dbparam CO50_yield_from_CH4 Yield of CO50 from CH4
      real*8 :: CO50_yield_from_CH4=0.d0

!@dbparam nc_emis_use_ppm_interp: 1 means use ppm (non-linear) interpolation
!@+ in timestream emissions (only) to preserve monthly totals. Else use linear
!@+ month-to-month (linm2m)
      integer :: nc_emis_use_ppm_interp=1

C**** Each tracer has a variable name and a unique index
!@var NTM number of tracers
      integer :: NTM

!@var ntm_O18: Number of TRACERS_SPECIAL_O18 tracers.
#ifdef TRACERS_SPECIAL_O18
#ifdef TRACERS_WISO_O17
      integer, parameter :: ntm_o18=3
#else
      integer, parameter :: ntm_o18=2
#endif
#else
      integer, parameter :: ntm_o18=0
#endif  /* TRACERS_SPECIAL_O18 */
      type(vector_integer) :: gasex_index
!@var ntm_lerner: Number of TRACERS_SPECIAL_Lerner tracers.
#ifdef TRACERS_SPECIAL_Lerner
      integer, parameter :: ntm_lerner=9
#else
      integer, parameter :: ntm_lerner=0
#endif  /* TRACERS_SPECIAL_Lerner */
!@var ntm_water: Number of TRACERS_WATER tracers.
#if (defined TRACERS_WATER) 
      integer, parameter :: ntm_water=1
#else
      integer, parameter :: ntm_water=0
#endif  /* TRACERS_WATER */
!@var ntm_shindell: Number of TRACERS_SPECIAL_Shindell tracers.
#ifdef TRACERS_SPECIAL_Shindell
#ifdef TRACERS_ACETONE
      integer, parameter :: ntm_shindell=26
#else
      integer, parameter :: ntm_shindell=25
#endif  /* TRACERS_ACETONE */
#else
      integer, parameter :: ntm_shindell=0
#endif  /* TRACERS_SPECIAL_Shindell */
!@var ntm_terp: Number of TRACERS_TERP tracers.
#ifdef TRACERS_TERP
      integer, parameter :: ntm_terp=1
#else
      integer, parameter :: ntm_terp=0
#endif  /* TRACERS_TERP */
!@var ntm_koch: Number of TRACERS_AEROSOLS_Koch tracers.
!@var ntm_vbs: Number of TRACERS_AEROSOLS_VBS tracers.
#ifdef TRACERS_AEROSOLS_Koch
#ifdef SULF_ONLY_AEROSOLS
#ifdef TRACERS_SPECIAL_Shindell
      integer, parameter :: ntm_koch=4
#else
      integer, parameter :: ntm_koch=5
#endif  /* TRACERS_SPECIAL_Shindell */
      integer, parameter :: ntm_vbs=0
#elif (defined TRACERS_AEROSOLS_VBS)
#ifdef TRACERS_SPECIAL_Shindell
      integer, parameter :: ntm_koch=7
#else
      integer, parameter :: ntm_koch=8
#endif  /* TRACERS_SPECIAL_Shindell */
      integer, parameter :: ntm_vbs=2*vbs_bins
#else
#ifdef TRACERS_SPECIAL_Shindell
      integer, parameter :: ntm_koch=10
#else
      integer, parameter :: ntm_koch=11
#endif  /* TRACERS_SPECIAL_Shindell */
      integer, parameter :: ntm_vbs=0
#endif  /* SULF_ONLY_AEROSOLS or TRACERS_AEROSOLS_VBS */
#else
      integer, parameter :: ntm_koch=0
      integer, parameter :: ntm_vbs=0
#endif  /* TRACERS_AEROSOLS_Koch */
!@var ntm_ss: Number of TRACERS_AEROSOLS_SEASALT tracers.
#ifdef TRACERS_AEROSOLS_SEASALT
      integer, parameter :: ntm_ss=2
#else
      integer, parameter :: ntm_ss=0
#endif
!@var ntm_ococean: Number of TRACERS_AEROSOLS_OCEAN tracers.
#ifdef TRACERS_AEROSOLS_OCEAN
      integer, parameter :: ntm_ococean=1
#else
      integer, parameter :: ntm_ococean=0
#endif  /* TRACERS_AEROSOLS_OCEAN */
!@var ntm_dCO: Number of TRACERS_dCO tracers.
#ifdef TRACERS_dCO
      integer, parameter :: ntm_dCO=14
#elif defined(TRACERS_dCOlite)
      integer, parameter :: ntm_dCO=3
#else
      integer, parameter :: ntm_dCO=0
#endif  /* TRACERS_dCO || TRACERS_dCOlite */

!@param ntm_dust: Number of dust aerosol tracers.
#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS) ||\
    (defined TRACERS_AMP)|| (defined TRACERS_TOMAS) 
#if (defined TRACERS_DUST) || (defined TRACERS_AMP)||\
    (defined TRACERS_TOMAS) 
      integer, parameter :: ntm_clay = 1
      integer, parameter :: ntm_sil1 = 1
      integer, parameter :: ntm_sil2 = 1
      integer, parameter :: ntm_sil3 = 1
#ifdef TRACERS_DUST_Silt5
      integer, parameter :: ntm_sil4 = 1
      integer, parameter :: ntm_sil5 = 1
#else
#ifdef TRACERS_DUST_Silt4
      integer, parameter :: ntm_sil4 = 1
      integer, parameter :: ntm_sil5 = 0
#else
      integer, parameter :: ntm_sil4 = 0
      integer, parameter :: ntm_sil5 = 0
#endif  /* TRACERS_DUST_Silt4 */
#endif  /* TRACERS_DUST_Silt5 */
#else /* !(TRACERS_DUST || TRACERS_AMP || TRACERS_TOMAS) */
!@var ntm_minerals: Number of TRACERS_MINERALS tracers.
#ifdef TRACERS_MINERALS
      integer, parameter :: ntm_clay = 15
      integer, parameter :: ntm_sil1 = 15
      integer, parameter :: ntm_sil2 = 15
      integer, parameter :: ntm_sil3 = 15
#ifdef TRACERS_DUST_Silt5
      integer, parameter :: ntm_sil4 = 15
      integer, parameter :: ntm_sil5 = 15
#else
#ifdef TRACERS_DUST_Silt4
      integer, parameter :: ntm_sil4 = 15
      integer, parameter :: ntm_sil5 = 0
#else
      integer, parameter :: ntm_sil4 = 0
      integer, parameter :: ntm_sil5 = 0
#endif  /* TRACERS_DUST_Silt4 */
#endif  /* TRACERS_DUST_Silt5 */
#endif  /* TRACERS_MINERALS */
#endif  /* TRACERS_DUST || TRACERS_AMP || TRACERS_TOMAS */
      integer, parameter :: ntm_dust = ntm_clay + ntm_sil1 + ntm_sil2 +
     &     ntm_sil3 + ntm_sil4 + ntm_sil5
#else /* !(TRACERS_DUST || TRACERS_MINERALS || TRACERS_AMP || TRACERS_TOMAS) */
      integer, parameter :: ntm_dust = 0
#endif  /* TRACERS_DUST || TRACERS_MINERALS || TRACERS_AMP || TRACERS_TOMAS */

!@var ntm_het: Number of TRACERS_HETCHEM tracers.
#ifdef TRACERS_HETCHEM
#ifdef TRACERS_NITRATE
      integer, parameter :: ntm_het=6
#else
      integer, parameter :: ntm_het=3
#endif  /* TRACERS_NITRATE */
#else
      integer, parameter :: ntm_het=0
#endif  /* TRACERS_HETCHEM */
!@var ntm_nitrate: Number of TRACERS_NITRATE tracers.
#ifdef TRACERS_NITRATE
      integer, parameter :: ntm_nitrate=3
#else
      integer, parameter :: ntm_nitrate=0
#endif  /* TRACERS_NITRATE */
!@param ntm_soa: Number of SOA tracers.
#ifdef TRACERS_AEROSOLS_SOA
!@+            Index g means gas phase, index p means particulate
!@+            (aerosol) phase. g should ALWAYS be EXACTLY BEFORE
!@+            the corresponding p and all SOA species should be
!@+            one after the other. Do not forget to declare them
!@+            to the chemistry species as well.
#ifdef TRACERS_TERP
      integer, parameter :: ntm_soa=8
#else
      integer, parameter :: ntm_soa=4
#endif  /* TRACERS_TERP */
!@param nsoa the total number of aerosol-phase SOA related species (=ntm_soa/2)
      integer, parameter :: nsoa=ntm_soa/2
#else
      integer, parameter :: ntm_soa=0
#endif  /* TRACERS_AEROSOLS_SOA */
!@var ntm_ocean: Number of TRACERS_OCEAN tracers.
#ifdef TRACERS_OCEAN
      integer, parameter :: ntm_ocean=0
#else
      integer, parameter :: ntm_ocean=0
#endif  /* TRACERS_OCEAN */
!@var ntm_air: Number of TRACERS_AIR tracers.
#if defined TRACERS_AIR
      integer, parameter :: ntm_air=1
#else
      integer, parameter :: ntm_air=0
#endif  /* TRACERS_AIR */

!@param ntm_chem number of drew-only tracers
      integer, parameter :: ntm_chem=ntm_shindell+
     *                               ntm_terp+
     *                               ntm_dCO+
     *                               ntm_soa
      ! Set by Shindell
      integer :: NTM_chem_beg = 0
      integer :: NTM_chem_end = 0
#ifdef TRACERS_AMP
#else
#ifdef TRACERS_TOMAS
       !constants that have to do with the number of tracers
!@param NBS is the number of bulk species (gases and aerosols that don't
!@+      have size resolution such as MSA) but excluding Shindell's gas tracers
!@param NAP is the number of size-resolved "prognostic" aerosol species
!@+      (ones that undergo transport)
!@param NAD is the number of size-resolved "diagnostic" aerosol species
!@+      (ones that don't undergo transport or have a complete budget
!@+      such as aerosol water and nitrate)
!@param NSPECIES is the number of unique chemical species not counting
!@+      size-resolved species more than once (the sum of NBS, NAP,
!@+      and NAD)
!@param NBINS is the number of bins used to resolve the size distribution
!@param NTM is the total number of tracer concentrations that the model
!@+      tracks.  This counts bulk species, and both prognostic and 
!@+      diagnostic aerosols.  Each size-resolved aerosol has a number
!@+      of tracers equal to NBINS to resolve its mass distribution.
!@+      An additional NBINS are required to resolve the aerosol number
!@+      distribution.
!@param NTT is the total number of transported tracers.  This is the same
!@+      as NTM, but excludes the "diagnostic" aerosol species which
!@+      do not undergo transport - ??? will I use this

       !constants that determine the size of diagnostic arrays
!@param NXP is the number of transport processes that are tracked
!@+      separately (NO USE)
!@param NCR is the number of chemical reactions for which data is saved (NO USE)
!@param NOPT is the number of aerosol optical properties tracked
!@param NFOR is the number of different forcings that are calculated
!@param NAERO is the number of aerosol microphysics diagnostics
!@param NCONS is the number of conservation quantity diagnostics
!@param KCDGN is the number of cloud microphysics and optical depth diagnostics

#if (defined TOMAS_12_10NM) 
      integer, parameter :: NBINS=12 
#elif (defined TOMAS_15_10NM) || (defined TOMAS_12_3NM)
      integer, parameter :: NBINS=15 
#endif
      integer, parameter :: NBS=7,NAP=7, NAD=1 !, NXP=7, NCR=2, 
      integer, parameter :: ntm_tomas=NBINS*(NAP+NAD+1)
#endif  /* TRACERS_TOMAS */
#endif

!@var N_XXX: variable names of indices for tracers (init = 0)
#ifdef TRACERS_TOMAS
      integer, dimension(nbins) :: n_ANUM =0
      integer, dimension(nbins) :: n_ASO4 =0
      integer, dimension(nbins) :: n_ANACL=0
      integer, dimension(nbins) :: n_AECIL=0
      integer, dimension(nbins) :: n_AECOB=0
      integer, dimension(nbins) :: n_AOCIL=0
      integer, dimension(nbins) :: n_AOCOB=0
      integer, dimension(nbins) :: n_AH2O=0
      integer, dimension(nbins) :: n_ADUST=0  
      integer :: n_SOAgas=0
#endif
      integer ::
     *     n_SF6=0,    n_SF6_c=0, n_nh5=0,  n_tape_rec=0,  n_st8025=0,                                   
     *     n_nh50=0,   n_e90=0,   n_aoa=0,      n_aoanh=0,   n_nh15=0,
     *     n_CO50,
     *     n_Air=0,    n_Rn222=0, n_CO2=0,      n_N2O=0,
     *     n_CFC11=0,  n_14CO2=0, n_CH4=0,   n_O3=0,       n_water=0,
     *     n_H2O18=0,  n_HDO=0,   n_HTO=0,   n_Ox=0,       n_NOx=0,
     *     n_N2O5=0,   n_HNO3=0,  n_H2O2=0,  n_CH3OOH=0,   n_HCHO=0,
     *     n_HO2NO2=0, n_CO=0,    n_PAN=0,   n_H2O17=0,
     *     n_Isoprene=0, n_AlkylNit=0, n_Alkenes=0, n_Paraffin=0,
     *     n_Terpenes=0, n_Acetone=0,
     *     n_isopp1g=0,n_isopp1a=0,n_isopp2g=0,n_isopp2a=0,
     *     n_apinp1g=0,n_apinp1a=0,n_apinp2g=0,n_apinp2a=0,
     *     n_DMS=0,    n_MSA=0,   n_SO2=0,   n_SO4=0,    n_H2O2_s=0,
     *     n_ClOx=0,   n_BrOx=0,  n_HCl=0,   n_HOCl=0,   n_ClONO2=0,
     *     n_HBr=0,    n_HOBr=0,  n_BrONO2=0,n_CFC=0,    n_GLT=0,
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
#ifdef TRACERS_dCO
     *     n_d13Calke=0, n_d13CPAR=0,
     *     n_d17OPAN=0, n_d18OPAN=0, n_d13CPAN=0,
     *     n_dMe17OOH=0, n_dMe18OOH=0, n_d13MeOOH=0,
     *     n_dHCH17O=0, n_dHCH18O=0, n_dH13CHO=0,
#endif  /* TRACERS_dCO */
     *     n_dC17O=0, n_dC18O=0, n_d13CO=0,
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
     *     n_Pb210 = 0,n_Be7=0,   n_Be10=0,
     .     n_CFCn=0,   n_CO2n=0,  n_Age=0,
     *     n_seasalt1=0,  n_seasalt2=0, n_SO4_d1=0,  n_SO4_d2=0,
     *     n_SO4_d3=0,n_N_d1=0,  n_N_d2=0,  n_N_d3=0,
     &     n_NH3=0,   n_NH4=0,   n_NO3p=0,
     *     n_BCII=0,  n_BCIA=0,  n_BCB=0,
     *     n_OCII=0,  n_OCIA=0,  n_OCB=0,
     *     n_vbsGm2=0, n_vbsGm1=0, n_vbsGz=0,  n_vbsGp1=0, n_vbsGp2=0,
     *     n_vbsGp3=0, n_vbsGp4=0, n_vbsGp5=0, n_vbsGp6=0,
     *     n_vbsAm2=0, n_vbsAm1=0, n_vbsAz=0,  n_vbsAp1=0, n_vbsAp2=0,
     *     n_vbsAp3=0, n_vbsAp4=0, n_vbsAp5=0, n_vbsAp6=0,
     *     n_OCocean=0,
     &     n_clay=0,  n_silt1=0, n_silt2=0, n_silt3=0, n_silt4=0,
     &     n_silt5=0,
     &     n_clayilli=0, n_claykaol=0, n_claysmec=0, n_claycalc=0,
     &     n_clayquar=0, n_clayfeld=0, n_clayhema=0, n_claygyps=0,
     &     n_clayilhe=0, n_claykahe=0, n_claysmhe=0, n_claycahe=0,
     &     n_clayquhe=0, n_clayfehe=0, n_claygyhe=0,
     &     n_sil1illi=0, n_sil1kaol=0, n_sil1smec=0, n_sil1calc=0,
     &     n_sil1quar=0, n_sil1feld=0, n_sil1hema=0, n_sil1gyps=0, 
     &     n_sil1ilhe=0, n_sil1kahe=0, n_sil1smhe=0, n_sil1cahe=0,
     &     n_sil1quhe=0, n_sil1fehe=0, n_sil1gyhe=0,
     &     n_sil2illi=0, n_sil2kaol=0, n_sil2smec=0, n_sil2calc=0,
     &     n_sil2quar=0, n_sil2feld=0, n_sil2hema=0, n_sil2gyps=0,
     &     n_sil2ilhe=0, n_sil2kahe=0, n_sil2smhe=0, n_sil2cahe=0,
     &     n_sil2quhe=0, n_sil2fehe=0, n_sil2gyhe=0,
     &     n_sil3illi=0, n_sil3kaol=0, n_sil3smec=0, n_sil3calc=0,
     &     n_sil3quar=0, n_sil3feld=0, n_sil3hema=0, n_sil3gyps=0,
     &     n_sil3ilhe=0, n_sil3kahe=0, n_sil3smhe=0, n_sil3cahe=0,
     &     n_sil3quhe=0, n_sil3fehe=0, n_sil3gyhe=0,
     &     n_sil4illi=0, n_sil4kaol=0, n_sil4smec=0, n_sil4calc=0,
     &     n_sil4quar=0, n_sil4feld=0, n_sil4hema=0, n_sil4gyps=0,
     &     n_sil4ilhe=0, n_sil4kahe=0, n_sil4smhe=0, n_sil4cahe=0,
     &     n_sil4quhe=0, n_sil4fehe=0, n_sil4gyhe=0,
     &     n_sil5illi=0, n_sil5kaol=0, n_sil5smec=0, n_sil5calc=0,
     &     n_sil5quar=0, n_sil5feld=0, n_sil5hema=0, n_sil5gyps=0,
     &     n_sil5ilhe=0, n_sil5kahe=0, n_sil5smhe=0, n_sil5cahe=0,
     &     n_sil5quhe=0, n_sil5fehe=0, n_sil5gyhe=0,
     *     n_M_NO3=0,   n_M_NH4=0,   n_M_H2O=0,   n_M_AKK_SU=0,
     *     n_N_AKK_1=0, n_M_ACC_SU=0,n_N_ACC_1=0, n_M_DD1_SU=0,
     *     n_M_DD1_DU=0,n_N_DD1_1=0, n_M_DS1_SU=0,n_M_DS1_DU=0,
     *     n_N_DS1_1 =0,n_M_DD2_SU=0,n_M_DD2_DU=0,n_N_DD2_1 =0,
     *     n_M_DS2_SU=0,n_M_DS2_DU=0,n_N_DS2_1 =0,n_M_SSA_SU=0,
     *     n_M_SSA_SS=0,n_M_SSC_SS=0,
     *     n_M_OCC_SU=0,n_M_OCC_OC=0,n_N_OCC_1 =0,
     *     n_M_BC1_SU=0,n_M_BC1_BC=0,n_N_BC1_1 =0,n_M_BC2_SU=0,
     *     n_M_BC2_BC=0,n_N_BC2_1 =0,n_M_BC3_SU=0,n_M_BC3_BC=0,
     *     n_N_BC3_1 =0,n_M_DBC_SU=0,n_M_DBC_BC=0,n_M_DBC_DU=0,
     *     n_N_DBC_1 =0,n_M_BOC_SU=0,n_M_BOC_BC=0,n_M_BOC_OC=0,
     *     n_N_BOC_1=0, n_M_BCS_SU=0,n_M_BCS_BC=0,n_N_BCS_1 =0,
     *     n_M_MXX_SU=0,n_M_MXX_BC=0,n_M_MXX_OC=0,n_M_MXX_DU=0,
     *     n_M_MXX_SS=0,n_N_MXX_1 =0,n_M_OCS_SU=0,n_M_OCS_OC=0,
     *     n_N_OCS_1=0,n_M_SSS_SS=0,n_M_SSS_SU=0,
     *     n_H2SO4=0, n_N_SSA_1=0, n_N_SSC_1=0
      integer :: n_M_ACC_OCM2=0, n_M_ACC_OCM1=0, n_M_ACC_OCM0=0,
     &           n_M_ACC_OCP1=0, n_M_ACC_OCP2=0, n_M_ACC_OCP3=0,
     &           n_M_ACC_OCP4=0, n_M_ACC_OCP5=0, n_M_ACC_OCP6=0,
     &           n_M_DD1_OCM2=0, n_M_DD1_OCM1=0, n_M_DD1_OCM0=0,
     &           n_M_DD1_OCP1=0, n_M_DD1_OCP2=0, n_M_DD1_OCP3=0,
     &           n_M_DD1_OCP4=0, n_M_DD1_OCP5=0, n_M_DD1_OCP6=0,
     &           n_M_DS1_OCM2=0, n_M_DS1_OCM1=0, n_M_DS1_OCM0=0,
     &           n_M_DS1_OCP1=0, n_M_DS1_OCP2=0, n_M_DS1_OCP3=0,
     &           n_M_DS1_OCP4=0, n_M_DS1_OCP5=0, n_M_DS1_OCP6=0,
     &           n_M_DD2_OCM2=0, n_M_DD2_OCM1=0, n_M_DD2_OCM0=0,
     &           n_M_DD2_OCP1=0, n_M_DD2_OCP2=0, n_M_DD2_OCP3=0,
     &           n_M_DD2_OCP4=0, n_M_DD2_OCP5=0, n_M_DD2_OCP6=0,
     &           n_M_DS2_OCM2=0, n_M_DS2_OCM1=0, n_M_DS2_OCM0=0,
     &           n_M_DS2_OCP1=0, n_M_DS2_OCP2=0, n_M_DS2_OCP3=0,
     &           n_M_DS2_OCP4=0, n_M_DS2_OCP5=0, n_M_DS2_OCP6=0,
     &           n_M_SSA_OCM2=0, n_M_SSA_OCM1=0, n_M_SSA_OCM0=0,
     &           n_M_SSA_OCP1=0, n_M_SSA_OCP2=0, n_M_SSA_OCP3=0,
     &           n_M_SSA_OCP4=0, n_M_SSA_OCP5=0, n_M_SSA_OCP6=0,
     &           n_M_SSC_OCM2=0, n_M_SSC_OCM1=0, n_M_SSC_OCM0=0,
     &           n_M_SSC_OCP1=0, n_M_SSC_OCP2=0, n_M_SSC_OCP3=0,
     &           n_M_SSC_OCP4=0, n_M_SSC_OCP5=0, n_M_SSC_OCP6=0,
     &           n_M_OCC_OCM2=0, n_M_OCC_OCM1=0, n_M_OCC_OCM0=0,
     &           n_M_OCC_OCP1=0, n_M_OCC_OCP2=0, n_M_OCC_OCP3=0,
     &           n_M_OCC_OCP4=0, n_M_OCC_OCP5=0, n_M_OCC_OCP6=0,
     &           n_M_BC1_OCM2=0, n_M_BC1_OCM1=0, n_M_BC1_OCM0=0,
     &           n_M_BC1_OCP1=0, n_M_BC1_OCP2=0, n_M_BC1_OCP3=0,
     &           n_M_BC1_OCP4=0, n_M_BC1_OCP5=0, n_M_BC1_OCP6=0,
     &           n_M_BC2_OCM2=0, n_M_BC2_OCM1=0, n_M_BC2_OCM0=0,
     &           n_M_BC2_OCP1=0, n_M_BC2_OCP2=0, n_M_BC2_OCP3=0,
     &           n_M_BC2_OCP4=0, n_M_BC2_OCP5=0, n_M_BC2_OCP6=0,
     &           n_M_OCS_OCM2=0, n_M_OCS_OCM1=0, n_M_OCS_OCM0=0,
     &           n_M_OCS_OCP1=0, n_M_OCS_OCP2=0, n_M_OCS_OCP3=0,
     &           n_M_OCS_OCP4=0, n_M_OCS_OCP5=0, n_M_OCS_OCP6=0,
     &           n_M_BOC_OCM2=0, n_M_BOC_OCM1=0, n_M_BOC_OCM0=0,
     &           n_M_BOC_OCP1=0, n_M_BOC_OCP2=0, n_M_BOC_OCP3=0,
     &           n_M_BOC_OCP4=0, n_M_BOC_OCP5=0, n_M_BOC_OCP6=0,
     &           n_M_BCS_OCM2=0, n_M_BCS_OCM1=0, n_M_BCS_OCM0=0,
     &           n_M_BCS_OCP1=0, n_M_BCS_OCP2=0, n_M_BCS_OCP3=0,
     &           n_M_BCS_OCP4=0, n_M_BCS_OCP5=0, n_M_BCS_OCP6=0,
     &           n_M_MXX_OCM2=0, n_M_MXX_OCM1=0, n_M_MXX_OCM0=0,
     &           n_M_MXX_OCP1=0, n_M_MXX_OCP2=0, n_M_MXX_OCP3=0,
     &           n_M_MXX_OCP4=0, n_M_MXX_OCP5=0, n_M_MXX_OCP6=0

! Shindell tracer indices with offsets:
      integer :: nn_CH4,  nn_N2O, nn_Ox,   nn_NOx,  
     *     nn_N2O5,   nn_HNO3,  nn_H2O2,  nn_CH3OOH,   nn_HCHO,  
     *     nn_HO2NO2, nn_CO,    nn_PAN,   nn_H2O17,
     *     nn_Isoprene, nn_AlkylNit, nn_Alkenes, nn_Paraffin,   
     *     nn_Terpenes, nn_Acetone,
     *     nn_isopp1g,nn_isopp1a,nn_isopp2g,nn_isopp2a,         
     *     nn_apinp1g,nn_apinp1a,nn_apinp2g,nn_apinp2a,         
     *     nn_ClOx,   nn_BrOx,  nn_HCl,   nn_HOCl,   nn_ClONO2,  
     *     nn_HBr,    nn_HOBr,  nn_BrONO2,nn_CFC,    nn_GLT
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
#ifdef TRACERS_dCO
     *    ,nn_d13Calke,nn_d13CPAR
     *    ,nn_d17OPAN,nn_d18OPAN,nn_d13CPAN
     *    ,nn_dMe17OOH,nn_dMe18OOH,nn_d13MeOOH
     *    ,nn_dHCH17O,nn_dHCH18O,nn_dH13CHO
#endif  /* TRACERS_dCO */
     *    ,nn_dC17O, nn_dC18O, nn_d13CO
#endif  /* TRACERS_dCO || TRACERS_dCOlite */

!@var n_soilDust index of first soil dust aerosol tracer
      integer :: n_soilDust = 0
#ifdef TRACERS_AMP
!@var ntmAMPi Index of the first AMP tracer
!@var ntmAMPe Index of the last AMP tracer
!@var ntmAMP Total number of AMP tracers
      integer :: ntmAMPi=0,ntmAMPe=0,ntmAMP=0
#endif

C**** standard tracer and tracer moment arrays

!@var TRM: Tracer array (kg/m2/layer)
      REAL*8, ALLOCATABLE, DIMENSION(:,:,:,:) :: trm

!@var TRMOM: Second order moments for tracers (kg/m2/layer)
      REAL*8, ALLOCATABLE, DIMENSION(:,:,:,:,:) :: trmom

!@var trm_col, trmom_col the contents of trm,trmom for the current i,j
!@+   column within the master loop over columns in tracer_3Dsource
      REAL*8, ALLOCATABLE, DIMENSION(:,:) :: trm_col
      REAL*8, ALLOCATABLE, DIMENSION(:,:,:) :: trmom_col

!@var TRDN1: lowest level downdraft tracer concentration (kg/kg)
       REAL*8, ALLOCATABLE, DIMENSION(:,:,:) :: trdn1

#ifdef TRACERS_WATER
!@var TRWM tracer in cloud liquid water amount (kg/m2/layer)
      REAL*8, ALLOCATABLE, DIMENSION(:,:,:,:) :: trwm
#endif

!@var daily_z: altitude of model levels (m), updated once per day
!@+   and used for vertical distribution of 3D emissions
      REAL*8, ALLOCATABLE, DIMENSION(:,:,:) :: daily_z


#ifdef TRACERS_WATER
!@param nWD_TYPES number of tracer types for wetdep purposes
      integer, parameter :: nWD_TYPES=3 !(gas,particle,water)
!@param tr_evap_fact fraction of re-evaporation of raindrops by tracer type
C note, tr_evap_fact is not dimensioned as NTM:
      REAL*8, parameter, dimension(nWD_TYPES) :: tr_evap_fact=
     *     (/1.d0, 0.5d0,  1.d0/)
#endif

C**** Diagnostic indices and meta-data

!@var sfc_src array holds tracer sources that go into trsource( )
!@+ maybe wasteful of memory, but works for now...
      real*8, allocatable, dimension(:,:,:,:) :: sfc_src

!@var MTRACE: timing index for tracers
!@var MCHEM: timing index for chemistry (if needed)
      integer mtrace, mchem
!@var MTRADV: timing index for tracer advection (if requested)
#ifdef TRAC_ADV_CPU
      integer mtradv
#endif
!@dbparam no_emis_over_ice, switch whether (>0) or not (<=0) to
!@+ set surface tracer emissions to zero over >90% ice boxes
      integer :: no_emis_over_ice=0

C**** EVERYTHING BELOW HERE IS TRACER SPECIFIC. PLEASE THINK 
C**** ABOUT MOVING IT ELSEWHERE

#ifdef TRACERS_SPECIAL_O18
C**** Water isotope specific parameters

!@dbparam supsatfac factor controlling super saturation for isotopes
      real*8 :: supsatfac = 2d-3
#endif

#if (defined TRACERS_SPECIAL_Shindell) || (defined TRACERS_AEROSOLS_Koch) ||\
    (defined TRACERS_AMP) || (defined TRACERS_TOMAS) 
C**** Chemistry specific 3-D arrays

!@var RSULF1, RSULF2, RSULF3, RSULF4: rate coefficients
c for gas phase sulfur chemistry used by aerosol and chemistry models
      REAL*8, DIMENSION(LM) :: rsulf1,rsulf2,rsulf3,rsulf4
#endif

!@dbparam tune_BBsources Factor to multiply biomass burning emissions with
      real*8 :: tune_BBsources = 1.d0

#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS) || (defined TRACERS_AEROSOLS_SEASALT)
C**** Aerosol specific switches and arrays
!@dbparam aer_int_yr indicates year of emission
      integer :: aer_int_yr = 0
!@dbparam SO2_int_yr Year of SO2 emissions to use
      integer :: SO2_int_yr=0
!@dbparam NH3_int_yr Year of NH3 emissions to use
      integer :: NH3_int_yr=0
!@dbparam BC_int_yr Year of BC emissions to use
      integer :: BC_int_yr=0
!@dbparam OC_int_yr Year of OC emissions to use
      integer :: OC_int_yr=0
#endif

C**** tracer specific switches

!@dbparam rnsrc is a switch for Radon sources 
!@+       0=standard, 1=Conen&Robertson, 2=modified Schery&Wasiolek
      integer :: rnsrc = 0

C**** arrays that could be general, but are only used by chemistry

!@dbparam trans_emis_overr_yr year for overriding Shindell tracer
!@+       transient emissions
!@dbparam trans_emis_overr_day day for overriding Shindell tracer
!@+       transient emissions
      integer :: trans_emis_overr_yr=0, trans_emis_overr_day=0
!@param nChemistry index for tracer net chemistry 3D source (+) or loss (-)
!@param nOverwrite index for tracer overwrite 3D source (+) or loss (-)
!@param nOther index for tracer misc. 3D source
!@param nAircraft index for tracer aircraft 3D source
!@param nBiomass index for tracer biomass burning 3D source
!@param nVolcanic index for tracer volcano 3D source
!@param nChemloss index for tracer chemistry 3D loss
!@param nChemprod index for tracer chemistry 3D production
!@param nPartition index for tracer partitioning 3D source (+) or loss (-)
!@param nThermo index for tracer changes due to thermodynamic equilibrium
!@param nRocket index for tracer rocket 3D source
! Must be a better way to do this, but for now, it is better than
! hardcoding indicies with a number like "3":
      INTEGER, PARAMETER :: nChemistry = 1, nOverwrite = 2,
     &     nOther = 3, nAircraft = 4, nBiomass = 5,
     &     nVolcanic = 6, nChemloss = 7, nChemprod = 8,
     &     nMicrophys = 9, nThermo = 10, nRocket = 11

!The list below is for non-standard tracer sources and sinks:
!-----------------------------------------------------------
#ifdef TRACERS_SPECIAL_Lerner
!@param nTropCH4 for CH4 tropospheric chemistry 3D sources/sinks
!@param nStratCH4 for CH4 stratospheric chemistry 3D sources/sinks
!@param nTropO3P for O3 tropospheric chemistry 3D sources (production)
!@param nTropO3L for O3 tropospheric chemistry 3D sinks (loss)
!@param nStratO3 for O3 stratospheric chemistry 3D sources/sinks
      integer, parameter ::  nTropCH4 = 1, nStratCH4 = 2,
     &      nTropO3P = 2, nTropO3L = 3, nStratO3 = 1
#endif
#ifdef TRACERS_TOMAS
!@param nSO4anum for SO4 aerosol number 3D sources/sinks
!@param nECanum for Elemental (black) Carbon aerosol number 3D sources/sinks
!@param nOCanum for Organic Carbon aerosol number 3D sources/sinks
      integer, parameter ::  nSO4anum = 1, nECanum = 2,
     &      nOCanum = 4 
#endif
!------------------------------------------------------------

#if (defined TRACERS_HETCHEM) || (defined TRACERS_NITRATE)
      integer, parameter :: rhet=3
      REAL*8, DIMENSION(lm) :: rxts,rxts1,rxts2,rxts3,rxts4
      REAL*8, DIMENSION(lm,8,rhet) :: krate
#endif
#ifdef TRACERS_VOLCEXP
      type(timestream) :: SO2_volc_stream  ! explosive emissions
      type(timestream) :: SO2_vphe_stream  ! explosive plume height
#endif
#if (defined TRACERS_SPECIAL_Shindell) || (defined TRACERS_AEROSOLS_Koch) ||\
    (defined TRACERS_AMP) || (defined TRACERS_TOMAS) ||\
    (defined TRACERS_GASEXCH_GCC)
!@var AIRCstreams organizes nc-reading of tracer 3D aircraft sources
      type(timestream), allocatable, dimension(:) :: AIRCstreams
      real*8, dimension(:,:,:,:), allocatable :: AIRCsrc
!@var AIRSstreams organizes nc-reading aircraft source scalings
      type(timestream), allocatable, dimension(:) :: AIRSstreams
!@var ROCKETstreams organizes nc-reading of tracer 3D rocket sources
      type(timestream), allocatable, dimension(:) :: ROCKETstreams
      real*8, dimension(:,:,:,:), allocatable :: ROCKETsrc
#endif

#ifdef PRESC_BB_INJ
C **** file containing plume injection height bottom and top
      type(timestream) :: BB_injbot_stream  ! Altitude of plume bottom
      type(timestream) :: BB_injmami_stream ! Mean altitude of maximum injection
      type(timestream) :: BB_injtop_stream  ! Altitude of plume top
      real*8, allocatable, dimension(:,:) :: BB_inj_bot
      real*8, allocatable, dimension(:,:) :: BB_inj_mami
      real*8, allocatable, dimension(:,:) :: BB_inj_top
      real*8, allocatable, dimension(:,:,:) :: fire_src_3d_fact
#endif  /* PRESC_BB_INJ */

!@var xyz_count,xyz_list count/list of tracers in category xyz.
!@+   A tracer can belong to more than one list.
c note: not applying CPP when declaring counts/lists.
      integer ::  ! counts of tracers meeting the following criteria:
     &      active_count  ! itime_tr0 <= itime
     &     ,gases_count   ! tr_wd_type == nGas
     &     ,aero_count    ! tr_wd_type == nPart
     &     ,water_count   ! tr_wd_type == nWater
     &     ,hlaw_count    ! tr_wd_type == nGas and tr_RKD != 0
     &     ,aqchem_count  ! participates in cloud aqueous chemistry
      integer, dimension(:), allocatable ::
     &     active_list,gases_list,aero_list,water_list,
     &     hlaw_list,aqchem_list
      integer, allocatable, dimension(:) :: itrSO4
      integer, allocatable, dimension(:) :: itrBC

      ! temporary support of legacy interface
      interface ntsurfsrc
        module procedure ntsurfsrc_1
        module procedure ntsurfsrc_all
      end interface ntsurfsrc

      real*8, allocatable, dimension(:, :, :) :: xyztr
      integer :: ntm_sph=0, ntm_reg=0

      contains

      subroutine initTracerCom()
      use TracerBundle_mod, only: TracerBundle

      tracers = newTracerBundle()

      call tracers%addDefaultValue('mass2vol', 0.0d0)
      call tracers%addDefaultValue('dodrydep', .false.)
      call tracers%addDefaultValue('t_qlimit',.true.)
      call tracers%addDefaultValue('trpdens', 0.0d0)
      call tracers%addDefaultValue('needtrs', .false.)
      call tracers%addDefaultValue('trdecay', 0.0d0)

      call tracers%addDefaultValue('trsi0', 0.0d0)
      call tracers%addDefaultValue('trw0', 0.0d0)
      call tracers%addDefaultValue('vol2mass', 0.0d0)
      call tracers%addDefaultValue('F0', 0.0d0)
      call tracers%addDefaultValue('KH_298', 0.0d0)
      call tracers%addDefaultValue('deltaH_R',0.0d0)
      call tracers%addDefaultValue('K1_298',0.0d0)
      call tracers%addDefaultValue('deltaH1_R',0.0d0)
      call tracers%addDefaultValue('K2_298',0.0d0)
      call tracers%addDefaultValue('deltaH2_R',0.0d0)
      call tracers%addDefaultValue('do_fire', .false.)
      call tracers%addDefaultValue('do_aircraft', .false.)
      call tracers%addDefaultValue('scale_aircraft', .false.)
      call tracers%addDefaultValue('first_aircraft', .true.)
      call tracers%addDefaultValue('do_rocket', .false.)
      call tracers%addDefaultValue('first_rocket', .true.)
      call tracers%addDefaultValue('nBBsources', 0)

      call tracers%addDefaultValue('trradius', 0.0d0)
      call tracers%addDefaultValue('tr_wd_type', nGas)
      call tracers%addDefaultValue('tr_RKD', 0.0d0)
      call tracers%addDefaultValue('tr_DHD', 0.0d0)
      call tracers%addDefaultValue('fq_aer', 0.0d0)
      call tracers%addDefaultValue('rc_washt', 1.d-1)
      call tracers%addDefaultValue('isDust', 0)
      call tracers%addDefaultValue('H2ObyCH4', 0.0d0)
      call tracers%addDefaultValue('dowetdep', .false.)
      call tracers%addDefaultValue('ntrocn', 0)
      call tracers%addDefaultValue('conc_from_fw', .true.)

      call tracers%addDefaultValue('iso_index', 1)
      call tracers%addDefaultValue('om2oc', 1.d0)
      call tracers%addDefaultValue('pm2p5fact', 0.d0)
      call tracers%addDefaultValue('pm10fact', 0.d0)
      call tracers%addDefaultValue('to_volume_MixRat', 0)
      call tracers%addDefaultValue('to_conc', 0)
      call tracers%addDefaultValue('TRLI0', 0.0d0)

      call initTracerMetadata()

      end subroutine initTracerCom

      subroutine remake_tracer_lists()
      use OldTracer_mod, only: trname
!@sum regenerates the counts and lists of tracers in various categories
      use model_com, only : itime
      implicit none
      integer :: n,nactive
      integer, dimension(1000) ::
     &     tmplist_active,tmplist_gases,tmplist_aero,tmplist_water,
     &     tmplist_hlaw,tmplist_aqchem
      active_count = 0
      gases_count = 0
      aero_count = 0
      water_count = 0
      hlaw_count = 0
      aqchem_count = 0
      do n=1,NTM

        if(itime.lt.itime_tr0(n)) cycle

        active_count = active_count + 1
        tmplist_active(active_count) = n

        select case(tr_wd_type(n))
        case(nGAS)
          gases_count = gases_count + 1
          tmplist_gases(gases_count) = active_count
          if(tr_RKD(n).ne.0.) then
            hlaw_count = hlaw_count + 1
            tmplist_hlaw(hlaw_count) = active_count
          endif
        case(nPART)
          aero_count = aero_count + 1
          tmplist_aero(aero_count) = active_count
        case(nWATER)
          water_count = water_count + 1
          tmplist_water(water_count) = active_count
        end select
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP)||\
    (defined TRACERS_TOMAS)
        select case (trname(n))
        case('SO2','SO4','H2O2_s','H2O2','M_ACC_SU','ASO4__01')
          if ( .not.(
     *         (trname(n).eq."H2O2" .and. coupled_chem.eq.0).or.
     *         (trname(n).eq."H2O2_s" .and. coupled_chem.eq.1)) )
     *         then
            aqchem_count = aqchem_count + 1
            tmplist_aqchem(aqchem_count) = active_count
          end if
        end select
#endif
      enddo
      if(allocated(active_list)) deallocate(active_list)
      allocate(active_list(active_count))
      active_list = tmplist_active(1:active_count)

      if(allocated(gases_list)) deallocate(gases_list)
      allocate(gases_list(gases_count))
      gases_list = tmplist_gases(1:gases_count)

      if(allocated(aero_list )) deallocate(aero_list )
      allocate(aero_list(aero_count))
      aero_list  = tmplist_aero(1:aero_count)

      if(allocated(water_list)) deallocate(water_list)
      allocate(water_list(water_count))
      water_list = tmplist_water(1:water_count)

      if(allocated(hlaw_list)) deallocate(hlaw_list)
      allocate(hlaw_list(hlaw_count))
      hlaw_list = tmplist_hlaw(1:hlaw_count)

      if (aqchem_count>0) then
        if(allocated(aqchem_list)) deallocate(aqchem_list)
        allocate(aqchem_list(aqchem_count))
        aqchem_list = tmplist_aqchem(1:aqchem_count)
      endif

      call get_tracer_subset_indices(tracers, 'SO4', itrSO4)
      call get_tracer_subset_indices(tracers, 'BC', itrBC)

      return
      end subroutine remake_tracer_lists

      integer function ntsurfsrc_1(index) result(n)
      use OldTracer_mod, only: trname
      use Tracer_mod
      integer, intent(in) :: index

      class (Tracer), pointer :: t

      t => tracers%getReference(trname(index))
      n = t%ntsurfsrc
      
      end function ntsurfsrc_1

      function ntsurfsrc_all() result(nSurf)
      use Tracer_mod
      integer, pointer :: nSurf(:)

      integer :: i, n

      n = tracers%size()
      allocate(nSurf(n))

      do i = 1, n
        nSurf(i) = ntsurfsrc(i)
      end do
      
      end function ntsurfsrc_all

      subroutine set_ntsurfsrc(index, value)
      use OldTracer_mod, only: trname
      use Tracer_mod
      use SpecialIO_mod, only: write_parallel
      integer, intent(in) :: index
      integer, intent(in) :: value
      character(len=300) :: out_line
      
      class (Tracer), pointer :: t

      if (value>ntsurfsrcmax) then
        write(out_line,*)'value>ntsurfsrcmax in set_ntsurfsrc'
        call write_parallel(trim(out_line))
        call stop_model(trim(out_line),255)
      endif
      t => tracers%getReference(trname(index))
      t%ntSurfSrc = value

      end subroutine set_ntsurfsrc

      SUBROUTINE ALLOC_TRACER_COM(grid)
!@sum  To allocate arrays whose sizes now need to be determined at
!@+    run time
!@auth NCCS (Goddard) Development Team
      use Tracer_mod, only: ntsurfsrcmax
      USE DOMAIN_DECOMP_ATM, ONLY : DIST_GRID, getDomainBounds
      IMPLICIT NONE
      TYPE (DIST_GRID), INTENT(IN) :: grid

      INTEGER :: I_0,I_1,J_0,J_1
      INTEGER :: J_1H, J_0H, I_1H, I_0H

C****
C**** Extract useful local domain parameters from "grid"
C****
      call getDomainBounds(grid)
      I_0=GRID%I_STRT
      I_1=GRID%I_STOP
      J_0=GRID%J_STRT
      J_1=GRID%J_STOP
      I_0H=GRID%I_STRT_HALO
      I_1H=GRID%I_STOP_HALO
      J_0H=GRID%J_STRT_HALO
      J_1H=GRID%J_STOP_HALO

      ALLOCATE(              trm(I_0H:I_1H,J_0H:J_1H,LM,NTM),
     *                trmom(NMOM,I_0H:I_1H,J_0H:J_1H,LM,NTM),
     *                trdn1(NTM,I_0:I_1,J_0:J_1),
     *              sfc_src(I_0:I_1,J_0:J_1,NTM,ntsurfsrcmax))

      ALLOCATE(  daily_z(I_0:I_1,J_0:J_1,LM) )
      daily_z = 0.

      allocate(trm_col(LM,NTM))
      allocate(trmom_col(NMOM,LM,NTM))

#ifdef TRACERS_WATER
      ALLOCATE(        trwm(I_0H:I_1H,J_0H:J_1H,LM,NTM) )
#endif
#if (defined TRACERS_SPECIAL_Shindell) || (defined TRACERS_AEROSOLS_Koch) ||\
    (defined TRACERS_AMP) || (defined TRACERS_TOMAS) || (defined TRACERS_GASEXCH_GCC)
      ALLOCATE( AIRCstreams(NTM) )
      ALLOCATE( AIRSstreams(NTM) )
      ALLOCATE( AIRCsrc(I_0:I_1,J_0:J_1,LM,NTM) )
      ALLOCATE( ROCKETstreams(NTM) )
      ALLOCATE( ROCKETsrc(I_0:I_1,J_0:J_1,LM,NTM) )
#endif

#ifdef PRESC_BB_INJ
! where deallocate?
      ALLOCATE(  BB_inj_bot(I_0:I_1,J_0:J_1) )
      ALLOCATE(  BB_inj_mami(I_0:I_1,J_0:J_1) )
      ALLOCATE(  BB_inj_top(I_0:I_1,J_0:J_1) )
      ALLOCATE(  fire_src_3d_fact(LM,I_0:I_1,J_0:J_1) )

      BB_inj_bot=0.d0
      BB_inj_mami=0.d0
      BB_inj_top=0.d0
      fire_src_3d_fact=0.d0
#endif  /* PRESC_BB_INJ */

      END SUBROUTINE ALLOC_TRACER_COM

      subroutine syncProperty(tracers, property, setValue, values)
      use Dictionary_mod, only: sync_param
      use TracerBundle_mod
      use oldtracer_mod, only: src_dist_index
      implicit none

      type (TracerBundle), intent(inout) :: tracers
      character(len=*) :: property
      interface
        subroutine setValue(n,value)
        integer, intent(in) :: n
        integer, intent(in) :: value
        end subroutine setValue
      end interface
      integer, intent(in) :: values(:)

      integer :: scratch(size(values))
      integer :: n 
      integer :: i

      n = 0
      do i=1, tracers%size()
!        Count tracers, ignoring duplicates,
!        except when it comes to the "itime_tr0"
!        parameter for water tracers, which is
!        needed for exact restarts:
         if (((tr_wd_type(i).eq.nWater) .and. 
     &        (property.eq."itime_tr0")) .or.
     &        (src_dist_index(i)<=1)) n=n+1
      end do
      scratch = values
      if(property.ne."itime_tr0")call sync_param(property,scratch,n)
      do i = 1, n
         call setValue(i, scratch(i))
      end do
      do i=n+1, tracers%size()
         call setValue(i, scratch(src_dist_index(i)))
      end do

      end subroutine syncProperty


      subroutine get_tracer_subset_indices(bundle, property, indices)

      use TracerBundle_mod
      use TracerHashMap_mod
      use Attributes_mod
      use Tracer_mod

      implicit none

      type (TracerBundle), intent(in) :: bundle
      character(len=*), intent(in) :: property
      integer, allocatable, dimension(:) :: indices

      type (TracerBundle), target :: subset
      type (TracerIterator) :: iter
      integer :: i, j, k, n
      type (Tracer), pointer :: p
      class (AbstractAttribute), pointer :: pattr

      if (allocated(indices)) deallocate(indices)

      subset = tracers%makeSubset(property)
      n = subset%size()
      allocate(indices(n))

      i = 1
      iter = subset%begin()
      do while (iter /= subset%last())
        p => iter%value()
        pattr => p%getReference('index')
        call toType(indices(i), pattr) ! cast to integer
        i = i + 1
        call iter%next()
      end do

      ! sort to preserve ordering
      do j = 1, n - 1
        do i = j + 1, n
          if (indices(i) < indices(j)) then ! swap
            k = indices(i)
            indices(i) = indices(j)
            indices(j) = k
          end if
        end do
      end do

      end subroutine get_tracer_subset_indices

      END MODULE TRACER_COM

      subroutine rescale_tracer_state(dir,n)
! convenience routine to change tracer units between kg and kg/m2
      use resolution, only : lm
      use domain_decomp_atm, only : grid
      use geom, only : axyp,byaxyp
      use tracer_com, only : ntm,trm,trmom
      implicit none
      integer :: dir ! -1,+1 to divide,multiply by area
      integer :: n   ! index of tracer to rescale
!
      integer :: i,j,l
      real*8, dimension(grid%i_strt_halo:grid%i_stop_halo,
     &                  grid%j_strt_halo:grid%j_stop_halo) :: fac

      if(dir == -1) then
        fac = byaxyp
      elseif(dir == +1) then
        fac = axyp
      else
        call stop_model('bad dir in rescale_tracer_state',255)
      endif

      do l=1,lm
        do j=grid%j_strt,grid%j_stop
          do i=grid%i_strt,grid%i_stop
            trm(i,j,l,n) = trm(i,j,l,n)*fac(i,j)
            trmom(:,i,j,l,n) = trmom(:,i,j,l,n)*fac(i,j)
          enddo
        enddo
      enddo

      end subroutine rescale_tracer_state
