test_rocke3d.R ROCKE-3D

test_rocke3d: ROCKE-3D, based on ModelE 2.1 with very minor updates and fixes
  P2: planet_2.0 model version
  S: SOCRATES radiation
  A: atmospheric composition for Earth, year 1850
  o: dynamic 4x5 horizontal resolution with 13 layer ocean
  M40: 4x5 horizontal resolution with 40 layer atmosphere
       Model top at 0.1 mb (+3 radiation-only layers above)
Uses turbulence scheme, no gravity wave drag
Time steps: dynamics 7.5 min leap frog
            physics 30 min.
            radiation 2.5 hrs (NRAD=5)
Filters: U,V in E-W and N-S direction (after every physics time step)
         U,V in E-W direction near poles (after every dynamics time step)
         sea level pressure (after every physics time step)

Preprocessor Options
#define USE_PLANET_RAD
#define GISS_RAD_OFF
#define NEW_IO                   ! new I/O (netcdf) on
#define IRRIGATION_ON
#define CHECK_OCEAN                  ! needed to compile aux/file CMPE002
#define CONSTANT_MESO_DIFFUSIVITY
#define OCN_LAYERING L13
#define ODIFF_FIXES_2017
#define EXPEL_COASTAL_ICEXS
#define NEW_BCdalbsn
#define CACHED_SUBDD
End Preprocessor Options

Object modules:
     ! resolution-specific source codes
Atm72x46                            ! horizontal resolution is 72x46 -> 4x5deg
AtmL40p                             ! vertical resolution
FFT72                               ! Fast Fourier Transform
ORES_5x4 OFFT72E                    ! ocean horiz res 4x5deg

IO_DRV                              ! new i/o

     ! GISS dynamics without gravity wave drag
ATMDYN MOMEN2ND                     ! atmospheric dynamics
QUS_DRV QUS3D                       ! advection of Q/tracers
STRAT_DUM

    ! lat-lon grid specific source codes
AtmRes
GEOM_B                              ! model geometry
DIAG_ZONAL GCDIAGb                  ! grid-dependent code for lat-circle diags
DIAG_PRT POUT                       ! diagn/post-processing output
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

OCN_Int_LATLON                      ! atm-ocn regrid routines

planet_rad planet_alb lw_control sw_control ! planet radiation source files

SUBDD

Components:
shared MPI_Support solvers giss_LSM 
dd2d
socrates
Ent

Component Options:
OPTS_Ent = ONLINE=YES PS_MODEL=FBB PFT_MODEL=ENT 
!make> no PNETCDFHOME - removing NC_IO=PNETCDF

Data input files:
    ! start from the restart file of an earlier run ...                 ISTART=8
! AIC=1....rsfE... ! initial conditions, no GIC needed, use

    ! start from observed conditions AIC(,OIC), model ground data GIC   ISTART=2
AIC=NCARIC.72x46.D7712010_ext.nc     ! AIC for automatic relayering to model vertical grid
GIC=GIC.E046D3M20A.1DEC1955.ext_1.nc ! initial ground conditions
OIC=OIC4X5LD.Z12.gas1.CLEV94.DEC01.nc ! ocean initial conditions
OFTAB=OFTABLE_NEW                     ! ocean function table
KBASIN=KB4X513.OCN.gas1.nc            ! ocean basin designations
TOPO_OC=OZ72X46N_gas.1_nocasp.nc      ! ocean bdy.cond
OSTRAITS=OSTRAITS_72x46.nml           ! parameterized straits info
TOPO=Z72X46N_gas.1_nocasp.nc
RVR=RD_modelE_M.nc                ! river direction file
NAMERVR=RD_modelE_M.names.txt     ! named river outlets

CDN=CD4X500S.ext.nc                 ! surf.drag coefficient
VEG=V72x46_EntMM16_lc_max_trimmed_scaled_nocrops.ext.nc   ! veg. fractions
LAIMAX=V72x46_EntMM16_lai_max_trimmed_scaled_ext.nc
HITEent=V72x46_EntMM16_height_trimmed_scaled_ext.nc
LAI=V72x46_EntMM16_lai_trimmed_scaled_ext.nc
CROPS=CROPS_and_pastures_Pongratz_to_Hurtt_72x46N_nocasp.nc ! crops history
IRRIG=Irrig72x46_1848to2100_FixedFuture_v3.nc
SOIL=S72x460098M.ext.nc                ! soil bdy.conds
TOP_INDEX=top_index_72x46_a.ij.ext.nc  ! only used if #define DO_TOPMODEL_RUNOFF
soil_textures=soil_textures_top30cm
SOILCARB_global=soilcarb_top30cm_4x5.nc
GLMELT=GLMELT_4X5.OCN.nc   ! glacial melt distribution
RADN1=sgpgxg.table8                           ! rad.tables and history files
RADN3=miescatpar.abcdv2

RH_QG_Mie=oct2003.relhum.nr.Q633G633.table
RADN7=STRATAER.VOL.1850-2014_CMIP6_hdr  ! needs MADVOL=2
RADN8=cloud.epsilon4.72x46

ISCCP=ISCCP.tautables
GHG=GHG.CMIP6.1-2014.txt  !  GreenHouse Gases for CMIP6 runs up to 2014
dH2O=dH2O_by_CH4_monthly

! Begin NINT E2.1 input files

BCdalbsn=cmip6_nint_inputs_E14TomaOCNf10_4av_decadal/72x46/BCdalbsn
DUSTaer=cmip6_nint_inputs_E14TomaOCNf10_4av_decadal/72x46/DUST
TAero_SUL=cmip6_nint_inputs_E14TomaOCNf10_4av_decadal/72x46/SUL
TAero_SSA=cmip6_nint_inputs_E14TomaOCNf10_4av_decadal/72x46/SSA
TAero_NIT=cmip6_nint_inputs_E14TomaOCNf10_4av_decadal/72x46/NIT
TAero_OCA=cmip6_nint_inputs_E14TomaOCNf10_4av_decadal/72x46/OCA
TAero_BCA=cmip6_nint_inputs_E14TomaOCNf10_4av_decadal/72x46/BCA
TAero_BCB=cmip6_nint_inputs_E14TomaOCNf10_4av_decadal/72x46/BCB

O3file=cmip6_nint_inputs_E14TomaOCNf10_4av_decadal/72x46/O3
Ox_ref=o3_2010_shindell_72x46x49_April1850.nc

! End NINT E2.1 input files

MSU_wts=MSU_SSU_RSS_weights.txt      ! MSU-diag
REG=REG4X5                        ! special regions-diag

Label and Namelist:  (next 2 lines)
test_rocke3d (ROCKE-3D, based on P2SAoM40 template)

&&PARAMETERS
! parameters set for coupled ocean runs:
KOCEAN=1            ! ocn is prognostic
OBottom_drag=1      !  Drags at the ocean bottom (NO drags -> OBottom_drag=0)
OCoastal_drag=1     !  Drags at the ocean coasts (NO drags -> OCoastal_drag=0)
OTIDE = 0           !  Ocean tides are not used
variable_lk=1
init_flake=1
meso_diffusivity_const=800.
ocean_fkph=0.6
! drag params if grav.wave drag is not used and top is at .01mb
X_SDRAG=.002,.0002  ! used above P(P)_sdrag mb (and in top layer)
C_SDRAG=.0002       ! constant SDRAG above PTOP=150mb
P_sdrag=1.          ! linear SDRAG only above 1mb (except near poles)
PP_sdrag=1.         ! linear SDRAG above PP_sdrag mb near poles
P_CSDRAG=1.         ! increase CSDRAG above P_CSDRAG to approach lin. drag
Wc_JDRAG=30.        ! crit.wind speed for J-drag (Judith/Jim)
ANG_sdrag=1     ! if 1: SDRAG conserves ang.momentum by adding loss below PTOP

! Input files for planet radiation (SOCRATES)
solar_spec='sun'
spectral_file_lw='sp_lw_ga7/sp_lw_ga7_dsa'
spectral_file_sw='sp_sw_ga7/sp_sw_ga7_dsa'
aer_opt_prop_lw='sp_lw_ga7/aer_lw_ga7.nc'
aer_opt_prop_sw='sp_sw_ga7/aer_sw_ga7.nc'
aer_opt_prop_diag='sp_diag/aer_diag_std.nc'
! cond_scheme=2   ! newer conductance scheme (N. Kiang) ! not used with Ent

! The following two lines are only used when aerosol/radiation interactions are off
FS8OPX=1.,1.,1.,1.,1.5,1.5,1.,1.
FT8OPX=1.,1.,1.,1.,1.,1.,1.3,1.

! Tuning parameters as of 2/2019
U00a=0.695  ! above 850mb w/o MC region;  tune this first to get 30-35% high clouds
U00b=0.60   ! below 850mb and MC regions; tune this last  to get rad.balance
WMU_multiplier = 1.
WMUI_multiplier = 0.001
radiusl_multiplier=1.01
radiusi_multiplier=1.
use_vmp=1

H2ObyCH4=1.      ! if =1. activates stratospheric H2O generated by CH4 without interactive chemistry

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

DTsrc=1800.      ! cannot be changed after a run has been started
DT=450.
! parameters that control the Shapiro filter
DT_XUfilter=450. ! Shapiro filter on U in E-W direction; usually same as DT
DT_XVfilter=450. ! Shapiro filter on V in E-W direction; usually same as DT
DT_YVfilter=0.   ! Shapiro filter on V in N-S direction
DT_YUfilter=0.   ! Shapiro filter on U in N-S direction

! Equation of time has 3 options:
!   'F' - off
!   'T' - on
!   'N' - uses "naive" EOT formula that neglects obliquity (initial implementation)
! Note: the default value of EOT is 'F' for Earth and 'T' for non-Earth.
! EOT='T'

NIsurf=2         ! surface interaction computed NIsurf times per source time step
NRAD=5           ! radiation computed NRAD times per source time step
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
Ndisk=1440       ! write fort.1.nc or fort.2.nc every NDISK source time step
&&END_PARAMETERS

&INPUTZ
 YEARI=1949,MONTHI=12,DATEI=1,HOURI=0, ! pick IYEAR1=YEARI (default) or < YEARI
 YEARE=1949,MONTHE=12,DATEE=2,HOURE=0,     KDIAG=12*0,9,
 ISTART=2,IRANDI=0, YEARE=1949,MONTHE=12,DATEE=1,HOURE=1,
/
### Information below describes your run. Do not delete! ###
