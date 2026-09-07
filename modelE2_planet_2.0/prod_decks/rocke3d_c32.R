rocke3d_c32.R GISS Model E coupled version         kelley 02/25/2010

rocke3d_c32: C32 cubed-sphere version of E4M20 coupled to 4x5 13-layer ocean.
            Intended to be used for regression testing and demonstration only.

E4M20.R GISS Model E  2004 modelE                     rar     07/15/2009
E4M20: modelE as frozen in July 2009 without gravity wave drag
atmospheric composition from year 1850
time steps: physics 30 min.; radiation 2.5 hrs

Preprocessor Options
!#define TRACERS_ON                  ! include tracers code
#define CHECK_OCEAN                 ! needed to compile aux/file CMPE002
#define AG2OG_PRECIP_BUNDLE
#define OG2AG_TOC2SST_BUNDLE
#define AG2OG_OCEANS_BUNDLE
#define OG2AG_OCEANS_BUNDLE
#define BUNDLE_INTERP
#define NEW_IO
!#define CALC_GWDRAG                 ! need to make ZVAR for C32 res first
#define SET_SOILCARBON_GLOBAL_TO_ZERO
!#define ROUGHL_HACK ! no longer needed?
End Preprocessor Options

Object modules: (in order of decreasing priority)

     ! resolution-specific source codes
AtmCS32                           ! 32 Cube-Sphere Grid
AtmL20 STRAT_DUM                 ! vertical resolution is 20 layers -> 0.1mb
RES_5x4_L13                         ! ocean horiz res 4x5deg, 13 layers
OFFT72E                             ! Fast Fourier Transform

     ! Codes used by the cubed-atmosphere configuration (FV dynamics)
GNOM_CS                             ! geometry
IO_DRV                              ! I/O driver
ATM_DUM                             ! atmospheric utilities
FV_UTILS FV_CS_Mod FV_INTERFACE     ! FV dynamical core wrapper
QUScubed                            ! cubed-sphere adaptation of QUS
ADVSIcs                             ! cubed-sphere adaptation of sea ice advection
COSZ_2D                             ! solar zenith angle computations
GCDIAGcs DIAG_ZONALcs               ! lat-circle diags
cs2ll_utils                         ! CS utilities for diags/regrids
pario_fbsa                          ! fortran-style I/O
QUICKPRT POUT                       ! diagnostics; not used for cubed sphere
FFTW_COM                            ! Fast Fourier Transform for spectral diagnostics
STRAT_DIAG                          ! dynamics diagnostics

MODEL_COM                           ! calendar, timing variables
MODELE_DRV                          ! ModelE cap
MODELE                              ! initialization and main loop
ATM_COM                             ! main atmospheric variables
ATM_DRV                             ! driver for atmosphere-grid components
ATMDYN_COM                          ! atmospheric dynamics
ATM_UTILS                           ! utilities for some atmospheric quantities
QUS_COM QUSDEF                      ! T/Q moments, 1D QUS
CLOUDS2 CLOUDS2_DRV CLOUDS_COM      ! clouds modules
SURFACE SURFACE_LANDICE FLUXES      ! surface calculation and fluxes
GHY_COM GHY_DRV    ! + giss_LSM     ! land surface and soils + snow model
VEG_DRV                             ! vegetation
! VEG_COM VEGETATION                ! old vegetation
ENT_DRV  ENT_COM   ! + Ent          ! new vegetation
PBL_COM PBL_DRV PBL                 ! atmospheric pbl
IRRIGMOD                            ! irrigation module
ATURB                               ! turbulence in whole atmosphere
LAKES_COM LAKES                     ! lake modules
SEAICE SEAICE_DRV                   ! seaice modules
LANDICE LANDICE_COM LANDICE_DRV     ! land ice modules
ICEDYN_DRV ICEDYN                   ! ice dynamics modules
Zenith                              ! shared zenith calculations
RAD_COM RAD_DRV RADIATION           ! radiation modules
RAD_UTILS GHGMOD ALBEDO READ_AERO ocalbedo ! radiation and albedo
DIAG_COM DIAG DEFACC                ! diagnostics
OCN_DRV                             ! driver for ocean-grid components
OLAYERS                             ! ocean layering options
OCEAN_COM  OGEOM                    ! dynamic ocean modules
OCNDYN  OCNDYN2  OTIDELL            ! dynamic ocean routines
OCN_Interp                          ! dynamic ocean routines
OSTRAITS_COM  OSTRAITS              ! straits parameterization
OCNKPP                              ! ocean vertical mixing
OCNMESO_DRV OCNTDMIX OCNGM          ! ocean mesoscale mixing
OCEANR_DIM  OFLUXES
ODIAG_COM ODIAG_ZONAL               ! ocean diagnostics modules
ODIAG_PRT                           ! ocean diagnostic print out
OCNFUNTAB                           ! ocean function look up table
SparseCommunicator_mod              ! sparse gather/scatter module
OCNQUS                              ! QUS advection scheme
OCNGISS_TURB                        ! GISS Turbulence vertical mixing scheme
OCNGISS_SM                          ! GISS Sub-mesoscale mixing scheme

regrid_com                          ! auxiliary regrid routines

Components:
shared MPI_Support solvers giss_LSM
Ent
dd2d

Component Options:
OPTS_Ent = ONLINE=YES PS_MODEL=FBB
FVCUBED = YES

Data input files:
  ! atmosphere, land surface, and sea ice
AIC=AIC_CS32.nc                  ! initial conditions (atm.)      needs GIC, ISTART=2
GIC=GIC_CS32_1.nc                  ! I.C. for land surface and sea ice
TOPO=Z_CS32_4X5.nc               ! topography
SOIL=SOIL_CS32.nc                ! soil types
VEG=V_CS32_144X90_1percent.nc    ! veg. fractions
CROPS=CROPS_CS32_4X5.nc          ! crops history
CDN=CD_CS32.nc                   ! surf.drag coefficient
REG=REG.txt                      ! special regions-diag
RVR=RDdistocean_CS32.nc             ! river direction file
NAMERVR=RDdistocean_CS32.names.txt  ! named river outlets
TOP_INDEX=top_index_CS32.nc         ! only used if #define DO_TOPMODEL_RUNOFF

  ! ocean
OIC=OIC4X5LD.Z12.gas1.CLEV94.DEC01.nc ! ocean initial conditions
OFTAB=OFTABLE_NEW                    ! ocean function table
KBASIN=KB4X513.OCN.gas1.nc           ! ocean basin designations
TOPO_OC=OZ72X46N_gas.1_nocasp.nc     ! ocean bdy.cond
OSTRAITS=OSTRAITS_72x46.nml          ! parameterized straits info
GLMELT=GLMELT_CS32.nc                   ! glacial melt distribution (on atm grid)
REMAP=remap72-46C32-32.nc            ! weights for atm-ocean coupling

RADN1=sgpgxg.table8                           ! rad.tables and history files
RADN2=LWTables33k_lowH2O_CO2_O3_planck_1-800  ! rad.tables and history files
RADN4=LWCorrTables33k                         ! rad.tables and history files
RADN5=H2Ocont_MT_CKD  ! Mlawer/Tobin_Clough/Kneizys/Davies H2O continuum table
! other available H2O continuum tables:
!    RADN5=H2Ocont_Ma_2000
!    RADN5=H2Ocont_Ma_2004
!    RADN5=H2Ocont_Roberts
!    RADN5=H2Ocont_MT_CKD  ! Mlawer/Tobin_Clough/Kneizys/Davies
RADN3=miescatpar.abcdv2

RH_QG_Mie=oct2003.relhum.nr.Q633G633.table
RADN7=STRATAER.VOL.1850-2014_CMIP6_hdr  ! needs MADVOL=2
RADN8=cloud.epsilon4.72x46
!RADN9=solar.lean2015.ann1610-2014.nc ! need KSOLAR=2
RADN9=solar.CMIP6official.ann1850-2299_with_E3_fastJ.nc ! need KSOLAR=2
RADNE=topcld.trscat8

ISCCP=ISCCP.tautables
GHG=GHG.CMIP6.1-2014.txt  !  GreenHouse Gases for CMIP6 runs up to 2014
CO2profile=CO2profile.Jul16-2017.txt ! scaling of CO2 in stratosphere
dH2O=dH2O_by_CH4_monthly
AMP_MIE_TABLES=AMP_MIE_Q_G_A_S.nc
AMP_CORESHELL_TABLES=AMP_CORESHELL_TABLES.nc
DUSTaer=dust_mass_CakmurMillerJGR06_C32L20x7x12.nc
BC_dep=BC.Dry+Wet.depositions.ann_c32.nc
! updated aerosols need MADAER=3
TAero_SUL=SUL_Koch2008_kg_m2_C32L20_1890-2000h.nc
TAero_SSA=SSA_Koch2008_kg_m2_C32L20h.nc
TAero_NIT=NIT_Bauer2008_kg_m2_C32L20_1890-2000h.nc
TAero_OCA=OCA_Koch2008_kg_m2_C32L20_1890-2000h.nc
TAero_BCA=BCA_Koch2008_kg_m2_C32L20_1890-2000h.nc
TAero_BCB=BCB_Koch2008_kg_m2_C32L20_1890-2000h.nc
O3file=o3_2005_shindelltrop_C32L49_1850-1997_ple.nc

MSU_wts=MSU_SSU_RSS_weights.txt

Label and Namelist:
rocke3d_c32 (cubed-sphere version of E4M20 coupled to 4x5 13-layer ocean)

&&PARAMETERS
! parameters set for coupled ocean runs:
KOCEAN=1            ! ocn is prognostic
OBottom_drag=1      !  Drags at the ocean bottom (NO drags -> OBottom_drag=0)
OCoastal_drag=1     !  Drags at the ocean coasts (NO drags -> OCoastal_drag=0)
OTIDE = 0           !  Ocean tides are not used
variable_lk=1
init_flake=1

wsn_max=2.   ! restrict snow depth to 2 m-h2o (if 0. snow depth is NOT restricted)

! drag params if grav.wave drag is not used and top is at .01mb
X_SDRAG=.002,.0002  ! used above P(P)_sdrag mb (and in top layer)
C_SDRAG=.0002       ! constant SDRAG above PTOP=150mb
P_sdrag=1.          ! linear SDRAG only above 1mb (except near poles)
PP_sdrag=1.         ! linear SDRAG above PP_sdrag mb near poles
P_CSDRAG=1.         ! increase CSDRAG above P_CSDRAG to approach lin. drag
Wc_JDRAG=30.        ! crit.wind speed for J-drag (Judith/Jim)
ANG_SDRAG=1         ! conserve ang. mom.

xCDpbl=1.
! cond_scheme=2    ! more elaborate conduction scheme (GHY, Nancy Kiang)

UOdrag=1 ! for flux-coupler verifications, atmosphere sees nonzero ocean/seaice velocity

U00a=.55    ! above 850mb w/o MC region; tune this first to get 30-35% high clouds
U00b=1.00   ! below 850mb and MC regions; then tune this to get rad.balance
! U00a,U00b replace the U00 parameters below - U00ice/U00wtrX are kept only for the _E1 version
U00ice=.59      ! U00ice up => nethtz0 down (alb down); goals: nethtz0=0,plan.alb=30%
U00wtrX=1.39    ! U00wtrX+.01=>nethtz0+.7                      for global annual mean
!?1979 U00wtrX=1.38
! HRMAX=500.    ! not needed unless do_blU00=1, HRMAX up => nethtz0 down (alb up)

CO2X=1.
H2OstratX=1.
KSIALB=0        ! 6-band albedo (Hansen) (=1 A.Lacis orig. 6-band alb)

PTLISO=15.       ! press(mb) above which rad. assumes isothermal layers
H2ObyCH4=1.      ! activates strat.H2O generated by CH4
KSOLAR=2         ! 2: use long annual mean file ; 1: use short monthly file

! parameters that control the atmospheric/boundary conditions
! if set to 0, the current (day/) year is used: transient run
master_yr=1850
!crops_yr=1850  ! if -1, crops in VEG-file is used
!s0_yr=1850
!s0_day=182
!ghg_yr=1850
!ghg_day=182
!irrig_yr=1850
volc_yr=-1
!volc_day=182
!aero_yr=1850
od_cdncx=0.        ! don't include 1st indirect effect
cc_cdncx=0.        ! don't include 2nd indirect effect (used 0.0036)
!albsn_yr=1850
dalbsnX=1.
!o3_yr=1850
!aer_int_yr=1850    !select desired aerosol emissions year or 0 to use JYEAR
! atmCO2=368.6          !uatm for year 2000 - enable for CO2 tracer runs

!variable_orb_par=0
!orb_par_year_bp=100  !  BP i.e. 1950-orb_par_year_bp AD = 1850 AD
MADVOL=2
madaer=3         ! 3: updated aerosols          ; 1: default sulfates/aerosols

DTsrc=1800.      ! physics timestep
DT=1800.         ! for FV dynamics, set same as DTsrc

! parameters that may have to be changed in emergencies:
NIsurf=1        ! increase as layer 1 gets thinner

! parameters that affect at most diagn. output:  standard if DTsrc=1800. (sec)
TAero_aod_diag=2 ! 0: no output; 1: save optical properties of TAero fields in aij for all bands; 2: save band6 only
aer_rad_forc=0   ! if set =1, radiation is called numerous times - slow !!
cloud_rad_forc=1 ! calls radiation twice; use =0 to save cpu time, 2= calculates crf_toa2
SUBDD=' '        ! no sub-daily frequency diags
NSUBDD=0         ! saving sub-daily diags every NSUBDD-th physics time step (1/2 hr)
KCOPY=1          ! 0: no output; 1: save .acc; 2: unused; 3: include ocean data
KRSF=12          ! 0: no output; X: save rsf at the beginning of every X month
isccp_diags=1    ! use =0 to save cpu time, but you lose some key diagnostics
nda5d=13         ! use =1 to get more accurate energy cons. diag (increases CPU time)
nda5s=13         ! use =1 to get more accurate energy cons. diag (increases CPU time)
ndaa=13
nda5k=13
nda4=48          ! to get daily energy history use nda4=24*3600/DTsrc

Nssw=2           ! until diurnal diags are fixed, Nssw has to be even
Ndisk=480       ! use =48 except on halem

&&END_PARAMETERS

 &INPUTZ
   YEARI=1901,MONTHI=1,DATEI=1,HOURI=0, !  from default: IYEAR1=YEARI
   YEARE=1901,MONTHE=1,DATEE=2,HOURE=0, KDIAG=13*0,
   ISTART=2,IRANDI=0, YEARE=1901,MONTHE=1,DATEE=1,HOURE=1
/
### Information below describes your run. Do not delete! ###
