! Paul Lerner Apr 6, 2018

E6obioF40.R      GISS Model E atm-ocean-otracers

xxtest63: This should be a run with a non-spun-up ocean carbon cycle
EPLobio_AR6: Earobio_AR6 + fixes for: wsdet + pCO2 above 1000 uatm + air-sea CO2 flux 
Earobio_AR6: E6F40 + E200F40oQ40 + new obio :
             new qus tracer moments
             new ABIO calculation
             drop ihra calculations for better avgq
             new pp (rmus,remin nitr/sili)
E200F40oQ40: E199F40oQ40 + 5m maximum for lake ice (LAKE_ICE_MAX=5.) 

Preprocessor Options
#define STDHYB                   ! standard hybrid vertical coordinate
#define ATM_LAYERING L40         ! 40 layers, top at .1 mb
#define NEW_IO                   ! new I/O (netcdf) on
#define NEW_IO_SUBDD
#define CACHED_SUBDD
#define IRRIGATION_ON
#define SWFIX_20151201
#define CHECK_OCEAN                  ! needed to compile aux/file CMPE002
#define SIMPLE_MESODIFF
#define OCN_LAYERING L40_5008m
#define ODIFF_FIXES_2017
#define EXPEL_COASTAL_ICEXS
#define NEW_BCdalbsn
#define CALCULATE_LIGHTNING      ! Calculate lightning flash rates
!  OFF #define AUTOTUNE_LIGHTNING  ! Automatically generate lightning tuning parameters (present-day only)
#define CALCULATE_FLAMMABILITY   ! activated code to determine flammability of surface veg
#define TRACERS_OCEAN               ! RUSSELL's Ocean tracers activated 
#define TRACERS_OCEAN_INDEP         ! independently defined ocn tracers -- ocean tracers indept of atm tracers
#define OBIO_ON_GISSocean           ! obio on Russell ocean
#define TRACERS_ON                  ! include tracers code
#define TRACERS_OceanBiology
#define pCO2_ONLINE
#define TRACERS_GASEXCH_ocean       ! ANY ocean: special tracers to be passed to ocean
#define TRACERS_GASEXCH_ocean_CO2   ! ANY ocean: special tracers to be passed to ocean
!!!#define OCN_CFC
#define constCO2
#define new_NFIXATION
#define newpp_May10_2018c
!!#define newpp_Sep29_2019a
#define increaseNremin5
!!#define increaseNremin6
#define increaseSremin
#define decreaseIremin              
#define exp_wsdiat
#define exp_wsdet
#define obio_rhsdiags
#define TRACERS_Alkalinity
#define Jprod_based_on_pp
#define no_offtermalk
#define alk_adj4
#define OBIO_RUNOFF
#define OBIO_QUIET_MODE
#define detr_estuarysink
End Preprocessor Options

Object modules:
     ! resolution-specific source codes
Atm144x90                           ! horizontal resolution is 144x90 -> 2x2.5deg
AtmLayering                         ! vertical resolution
FFT144                              ! Fast Fourier Transform
ORES_1Qx1 OFFT288E                  ! ocean horiz res 1.25x1deg

IO_DRV                              ! new i/o

     ! GISS dynamics with gravity wave drag
ATMDYN MOMEN2ND                     ! atmospheric dynamics
QUS_DRV QUS3D                       ! advection of Q/tracers
STRATDYN STRAT_DIAG                 ! stratospheric dynamics (incl. gw drag)

#include "latlon_source_files"
#include "modelE4_source_files"
#include "lightning_fire_source_files"
#include "dynamic_ocn_source_files"
SUBDD

OCN_Int_LATLON                      ! atm-ocn regrid routines

#include "tracer_shared_source_files" 
#include "ocarbon_cycle_oR_files" 

Components:
tracers
#include "E4_components_nc"
Ent

Component Options:
OPTS_Ent = ONLINE=YES PS_MODEL=FBB PFT_MODEL=ENT 
!make> no PNETCDFHOME - removing NC_IO=PNETCDF

Data input files:
    ! start from the restart file of an earlier run ...                 ISTART=8
AIC=/discover/nobackup/aromanou/RSFs/1JAN4190.rsfE200F40oQ40.nc

#include "dynamic_ocn_288x180_input_files_CMIP6_istart8or9"
TOPO=Z2HX2fromZ1QX1N.BS1.nc        ! surface fractions and topography (1 cell Bering Strait)
ICEDYN_MASKFAC=iceflowmask_144x90.nc

TDISS=altocnbc288x180_20170717/TIDAL_e_v2_1QX1.HB.nc
TDISS_N=tdiss/Jayne2009_288x180.nc
POROS=altocnbc288x180_20170717/poros.nc

RVR=RD_Fd.nc             ! river direction file
NAMERVR=RD_Fd.names.txt  ! named river outlets
FLAMPOPDEN=gsin/fire/RCP8.5_PopDens_2000-2100.nc ! for fire model

#include "land144x90_input_files"
#include "rad_input_files"
#include "rad_144x90_input_files_CMIP6clim"

MSU_wts=MSU_SSU_RSS_weights.txt      ! MSU-diag
REG=REG2X2.5                      ! special regions-diag

#include "ocarbon_cycle_input_files"

Label and Namelist:  (next 2 lines)
E6obioF40 = (Earobio3_piAR6+rivers+alk_adj4,w/bugfixes and correction, atm source files from E6F40.R)


&&PARAMETERS
ocean_trname = 'OceanAge abioDIC'
#include "dynamic_ocn_params"
ocean_use_qus=1     ! Advection uses the quadratic upstream scheme
DTO=112.5
ocean_use_tdmix=1  ! tdmix scheme for meso mixing
ocean_use_gmscz=1  ! vertically variation of meso diffusivity, option 1
ocean_kvismult=2.  ! mult. factor for meso diffusivity
ocean_enhance_shallow_kmeso=1 ! stronger meso mixing in shallow water
ocean_use_tdiss=1  ! simple tidally induced diapycnal diffusivity
ocean_ntrtrans=1   ! tracer advection speedup

#include "sdragF40_params"
#include "gwdragF40_params"

! cond_scheme=2   ! newer conductance scheme (N. Kiang) ! not used with Ent

! The following two lines are only used when aerosol/radiation interactions are off
FS8OPX=1.,1.,1.,1.,1.5,1.5,1.,1.
FT8OPX=1.,1.,1.,1.,1.,1.,1.3,1.

! Increasing U00a decreases the high cloud cover; increasing U00b decreases net rad at TOA
U00a=0.655  ! above 850mb w/o MC region;  tune this first to get 30-35% high clouds
U00b=1.00   ! below 850mb and MC regions; tune this last  to get rad.balance
WMUI_multiplier = 2.
use_vmp=1
radius_multiplier=1.1

PTLISO=0.        ! pressure(mb) above which radiation assumes isothermal layers
H2ObyCH4=1.      ! if =1. activates stratospheric H2O generated by CH4 without interactive chemistry
KSOLAR=2         ! 2: use long annual mean file ; 1: use short monthly file

#include "atmCompos_1850_params"
#include "lightning_fire_params"

madaer=3         ! 3: updated aerosols          ; 1: default sulfates/aerosols

DTsrc=1800.      ! cannot be changed after a run has been started
DT=225.
! parameters that control the Shapiro filter
DT_XUfilter=225. ! Shapiro filter on U in E-W direction; usually same as DT
DT_XVfilter=225. ! Shapiro filter on V in E-W direction; usually same as DT
DT_YVfilter=0.   ! Shapiro filter on V in N-S direction
DT_YUfilter=0.   ! Shapiro filter on U in N-S direction

NIsurf=2         ! surface interaction computed NIsurf times per source time step
NRAD=5           ! radiation computed NRAD times per source time step
#include "diag_params"

NMONAV=12
Nssw=48          ! until diurnal diags are fixed, Nssw has to be even
Ndisk=960       ! write fort.1.nc or fort.2.nc every NDISK source time step

! parameters that affect CO2 gas exchange
!!! atmCO2=368.6          !uatm for year 2000
!!! atmCO2=0.             !prognostic atmCO2
!!!atmCO2=285.226         !uatm for new preindustrial runs AR5 runs
atmCO2=284.65             !uatm for OCMIP6 preindustrial runs
to_volume_MixRat=1    ! for tracer printout
solFe=0.02            ! default iron solubility
!!!solFe=0.05            ! enhanced iron solubility

&&END_PARAMETERS

&INPUTZ
YEARI=1850,MONTHI=1,DATEI=1,HOURI=0, !IYEAR1=1850,
YEARE=1850,MONTHE=1,DATEE=2,HOURE=0,     KDIAG=12*0,9,
ISTART=8,IRANDI=0, YEARE=1850,MONTHE=1,DATEE=1,HOURE=1,
/

