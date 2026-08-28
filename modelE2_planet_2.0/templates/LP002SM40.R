LP002SM40.R  ROCKE-3D planet branch. M40    N.Y.Kiang 6/1/2018, M. Kelley 7/2013, R.Ruedy 10/2018

LP002SM40: one of about 110 planets with the following common properties:
   planet branch: planet_2.0, 4x5 grid, 40 lyrs (old discontinuous sigma-CP, non-STDHYB)
   Flat Land Planet with Earth's radius, gravity, 1-year orbital period (but eccentricity=0)
   dynamic lakes: 10m sill depth (no ocean)
   atmospheric composition: N2 and CO2 only, no O2 - using planet radiation (SOCRATES)

Non-common properties: LP002.M8.2V.S0X1.1.OBL28.ROT72.Pmb1916.CO2log10f-2.H2Om0.41.Sa
   Star: TRAPPIST-1, eff.temperature 2516 K, mass/SunMass 0.089, S0x=1.051
   Orbit: Obliquity 28, Eccentricity 0, radius 0.0222811164181847d0 AU, rotation 90558.2950819672045
   Atmosphere: Psurf(mb) 1916.7, N2: 990200d0 ppm, CO2: 9800.00000000003d0 ppm
   Ground: SurfAlbedo 0.27, roughness length 1.d0, soil texture sand, wetness 0.224673153704296

Preprocessor Options
#define PLANET_PARAMS LP002 ! parameter set selector in shared/PlanetParams_mod.F90
#define NEW_IO
#define USE_PLANET_RAD
#define GISS_RAD_OFF
!  #define DO_CO2_CONDENSATION
!  #define DO_CO2_CONDENSATION_APPLY
!  #define DO_CO2_CONDENSATION_DEBUG_SITES
#define RH_INIT_FROM_TandQ
#define TOPO_DIRECTED_RIVER_FLOW ! uses added source file RIVERF.F90
End Preprocessor Options

Object modules: (in order of decreasing priority)
     ! resolution-specific source codes
Atm72x46                   ! horizontal resolution is 72x46 -> 4x5deg
AtmL40p                    ! vertical resolution is 40 layers with non-standard PSF
FFT72                              ! Fast Fourier Transform
IO_DRV                             ! new i/o

     ! GISS dynamics w/o gravity wave drag
ATMDYN MOMEN2ND                     ! atmospheric dynamics
QUS_DRV QUS3D                       ! advection of Q/tracers
STRAT_DUM

#include "latlon_source_files"
#include "planet_source_files"
RIVERF                              ! new RIVERF May2018
#include "static_ocn_source_files"
planet_rad planet_alb lw_control sw_control ! planet radiation source files

Components:
#include "E4_components_nc" /* without "Ent" */
Ent
socrates

Component Options:
OPTS_Ent = ONLINE=YES PS_MODEL=FBB

Data input files:

! Initial atmospheric temperature, specific humidity, wind, surface pressure.
! If rundeck parameter initial_psurf_from_topo=1, initial surface pressure is
! reset to be consistent with orography.
! either use files AIC and GIC or SOILIC with ISTART=2
!     e.g. AIC=planet/Mars/AIC_M40_10mb_273K.nc
!          SOILIC=planet/desert_world/soilic_wetsoil.nc
! or use ISTART=1 and the defaults and the following db-parameters
!        in the &&PARAMETERS section to overwrite the defaults:
!   Tinit(1:*), Qinit(1:*), Tg_init, Gwet_init(1:*), snowdp_init

! No TOPO file is needed for flat land planets with (initially) no lakes
! in that case sill_depth (m) may be set in the &&PARAMETERS list

! If file ROUGHL is present, the variable top_dev in TOP_INDEX will only affect
! snow masking and its precise value is probably not crucial.
! TOP_INDEX=planet/desert_world/stdev_72x46_desertworld.nc
! For flat desert planets with uniform roughness, two &&PARAMETERS may be used to set
!   top_dev_mean   and  roughl_bare_soil (m)

! Soil textures file (array q).  Sand (imt=1) is a reasonable texture to start with.
! Arrays sl and qk do not matter for dry conditions.  dz is layer thickness.
SOIL=planet/SOIL/soil_sandSL05_4x5.nc

! In modelE, the percentages of bright and dark soil are read from
! the vegetation file.  The ratio of the two soils is chosen to give a
! surface albedo of roughly 15% for dry conditions.
! VEG=planet/Mars/veg_allbare_alb15.nc
! Alternatively, the fraction of sand may be prescribed in the &&PARAMETERS section:
!    fsand   (fraction of dark soil is set to 1-fsand, the surface albedo = fsand/2

! CFCs are set to zero in this file
! CO2/N2O/CH4 all 1ppm; use CO2x,N2Ox,CH4x to set the correct amount (ppm)
! Note that the absence of ozone and aerosol files here implies they are all zero.
GHG=GHG.AllOnes.txt

! radiation input files
RADN1=sgpgxg.table8               ! rad.tables and history files
RADN3=miescatpar.abcdv2
RH_QG_Mie=oct2003.relhum.nr.Q633G633.table
!!!RADN2=LWTables33k_lowH2O_CO2_O3_planck_1-800              ! rad.tables and history files
!!!RADN5=H2Ocont_MT_CKD  ! Mlawer/Tobin_Clough/Kneizys/Davies H2O continuum table
!!!RADNE=topcld.trscat8

!!!RADN4=LWCorrTables33k ! correction factors for Earth conditions are not applicable

Label and Namelist:
LP002SM40 (LP002.M8.2V.S0X1.1.OBL28.ROT72.Pmb1916.CO2log10f-2.H2Om0.41.Sa)

&&PARAMETERS

! Input files for planet radiation (SOCRATES)
solar_spec='trappist1'
spectral_file_lw='sp_lw_etw_arcc10bar/sp_lw_17_etw_arcc10bar'
spectral_file_sw='sp_sw_etw_arcc10bar/sp_sw_43_etw_arcc10bar'
! aer_opt_prop_lw='sp_lw_ga7/aer_lw_ga7.nc'
! aer_opt_prop_sw='sp_sw_ga7/aer_sw_ga7.nc'
! aer_opt_prop_diag='sp_diag/aer_diag_std.nc'

! calculate initial surface pressure for hydrostatic consistency with topography
initial_psurf_from_topo=1

! linear damping timescales (sec) for winds in the top few layers.
! These probably need to be tuned.  Change the number of values here
! to change the number of layers in which damping is applied.
!!rtau=30000.,20000.,10000. ! recommended by Kostas - however several runs bombed
rtau=60000.,50000.,40000.,30000.,20000.,10000. ! latest attempt by Reto

! minimum and maximum allowed column mass (kg/m2) for error checking
mincolmass=0.
maxcolmass=20000000000. !  default: 500.

! minimum and maximum allowed ground temperature (degC) for error checking
minGroundTemperature=-250.  ! default: -150.
maxGroundTemperature=400.   ! default:  130.

! minimum allowed sea/lake ice temperature (degC) for error checking
minIceTemperature=-150.     ! default: -100.

!--------------------
! The following block is used to control both the orbit _and_ the
! calendar in the case on non-Earth configurations.  If the
! 'planetName' parameter is set to anything other than 'Earth', then a
! calendar is derived from the specified orbital parameters.  Each
! orbital parameter has a default value corresponding to Earth, and
! the planet calendar will be exactly the same as the usual
! Julian_no_leap calendar if one uses those defaults.
!--------------------
! Planetary parameters (optional)
! for all land Earth-like planet with 0.0 obliquity and eccentricity
planetname='LP002'
obliquity=28.                       !Default Earth degrees  23.440
eccentricity=0.0                         !Default Earth  1.670E-002
longitudeatperiapsis=0.0                 !Default Earth degrees 282.90
siderealorbitalperiod=31536000.0         !Default Earth seconds 31536000.0
siderealrotationperiod=90558.2950819672             !Default Earth seconds 86163.9344262295
meandistance=1.                      !Default Earth AU 1.0
hourangleoffset=0.0                      !Default Earth ? 0.0
quantizeyearlength='T'              ! 'T', used to adjust year for non-Earth orbits
!--------------------

! scaling factor for solar brightness is now from mean distance.
planet_s0=1430.411d0   ! total stellar flux for planet rad, s0X unused if GISS_RAD_OFF

! scaling factor for O2 amounts. 0.7% of present-day Earth gives 0.146%.
o2x=0.00   !0.007

! scaling factor for CO2 amounts - no other greenhouse gases
co2x=9800.3d0 ! ppm (input file: all 1)
n2ox=0.
ch4x=0.
cfc11x=0.
cdc12x=0.
xghgx=0.

! bare-soil roughness length (m)
roughl_bare_soil=1.d0 ! desired_length (m)
! top_dev_mean=100. ! default 100. uniform std.dev. of topography (m)

! No vegetation (desert) case; fsand really only determnes the surface albedo
 fsand = 0.54d0 ! bright soil fraction; dark_soil=1-fsand;  surf.alb=fsand/2

! Physics timestep (sec). Cannot be changed after a run has been started.
DTsrc=1800.

! Dynamics timestep.  This is half of the value for 4x5 Earth runs.
! Note that DTsrc/DT should be an even integer.
DT=450. ! or 225.

! parameters that control the Shapiro filter
DT_XUfilter=450. ! Shapiro filter on U in E-W direction; usually same as DT (below)
DT_XVfilter=450. ! Shapiro filter on V in E-W direction; usually same as DT (below)
DT_YVfilter=0.   ! Shapiro filter on V in N-S direction
DT_YUfilter=0.   ! Shapiro filter on U in N-S direction

! Number of surface timesteps per physics timestep.  Probably does not
! need to be changed for M20 Mars.
NIsurf=1        ! increase as layer 1 gets thinner

! Number of physics timesteps per radiation timestep.  Default is 5.
nrad=5

! save alternating checkpoint files every Ndisk physics timesteps
Ndisk=1440

! KCOPY=1: save acc and alternating checkpoint files only.  KCOPY=2: save rsf also
KCOPY=2

nda5d=13        ! use =1 to get more accurate energy cons. diag (increases CPU time)
nda5s=13        ! use =1 to get more accurate energy cons. diag (increases CPU time)
ndaa=13
nda5k=13
nda4=48         ! to get daily energy history use nda4=24*3600/DTsrc

! restart state is saveable every nssw timesteps.
nssw=2          ! until diurnal diagn. are fixed, nssw should be even

! the model requires the following two parameters to be set, though they
! do not affect Mars runs
master_yr=1850
KOCEAN=0

! zero-out some constituents not present in Mars or CO2-only spectral files
O3X=0.
N2OX=0.
CFC11X=0.
CFC12X=0.
CH4X=0.
XGHGX=0.
YGHGX=0.
O2X=0.

! disable some Earth hacks
H2ObyCH4=0.     ! deactivates strat.H2O generated by CH4
l_uniform_ghg=1 ! deactivates prescribed GHG distributions

! Skip expensive diagnostics which need extra call to radiation
cloud_aer_o3_rad_forc=0
cloud_rad_forc=0

! Initial T,Q : Temp. and sp.Hum may be prescribed for layers 1:??
!     the last value sets also the values for all layers above
 Tinit = 370.,  ! 270.  deg K
 Qinit = 1.d-20,  ! 1.d-20 definitely in layers above 50mb (= maxctop )
 maxctop=0.
! Initial ground temperatures and wetness for layers 1:??
!     the last value sets also the values for all layers below
 Tg_init = 10. ! 5 deg C
 Gwet_init = 0.2246731537
 snowdp_init = 0.

! flat topography with potential lakes (initially none)
! A dry planet will have no lakes, but we set this in case someone
! eventually decides to start with an initially wet soil.
variable_lk=1     ! variable lake fractions with time; 0-no, 1-yes, default ?
RIVER_FAC=16384.0 ! tuning factor to multiply lakes runoff by to balance sea level;
                  ! must use with RIVERF module introduced for planet for 4-way lake spillage; default 1.0.
sill_depth=10. ! default 10.
lake_ice_max=1.d30
wsn_max=1.d30

&&END_PARAMETERS

 &INPUTZ
 YEARI=0001,MONTHI=1,DATEI=1,HOURI=0, ! pick IYEAR1=YEARI (default) or < YEARI
 YEARE=0031,MONTHE=1,DATEE=1,HOURE=0,     KDIAG=12*0,9,
 ISTART=1,IRANDI=0, YEARE=0001,MONTHE=1,DATEE=1,HOURE=1,
/

