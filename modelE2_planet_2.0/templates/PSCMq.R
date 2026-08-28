PSCMq.R GISS Model E      M. Kelley 10/2013
                          A. Wolf Planet branch 9/2017

Run SCM using GCM IC's and no advective forcings w/qflux ocean
* * * *   to run using qflux ocean with SCM the extraction file OHT,OCNML
* * * *   is in netcdf format.  For now need to edit OCNML.f file
* * * *   after line 202 - DATA FOR FOR QFLUX MIXED LAYER OCEAN RUNS
* * * *     if(.true.) then ! assuming GISS format    change to .false. to force it read netcdf file

P4qM40SMP: P4qM40 with 65m q-flux ocean    qfluxX=0.
P4qM40SMP: modelE as frozen in July 2009 without gravity wave drag
modelE 4x5 hor. grid with 40 lyrs, top at .1 mb (+ 3 rad.lyrs)
atmospheric composition from year 1850
ocean data: prescribed, 1876-1885 climatology  (see OSST/SICE)
uses turbulence scheme, simple strat.drag (not grav.wave drag)
time steps: dynamics 7.5 min leap frog; physics 30 min.; radiation 2.5 hrs
filters: U,V in E-W and N-S direction (after every physics time step)
         U,V in E-W direction near poles (after every dynamics time step)
         sea level pressure (after every physics time step)
radiation: GISS

Initial framework for truly single-column mode for Model E.

This is a functioning rundeck, not a template (excepting the lines
containing  /path/to/user/directory/extractions - see notes below)

SCM-irrelevant codes and input files are excluded.
Template #includes should be refactored so that this exclusion happens automatically.

SCM case: example SCM_lat & SCM_lon correspond to ARM SGP site 
For other cases, change one or more of the following as necessary:
(1) SCM input variable namelist (SCM_NML) and SCM input files (SCM_PS, SCM_SFLUX, etc.)
(2) SCM parameters SCM_lon, SCM_lat, and files extracted from gridded data at SCM_lon, SCM_lat
    See notes in the input files section on a pre-scripted extraction procedure etc.
    SCM_area and other SCM parameters may also be changed (or added).

Preprocessor Options
#define CACHED_SUBDD
#define SCM
#define USE_ENT
#define NEW_IO
#define PLANET_PARAMS notEarth
End Preprocessor Options

Object modules: (in order of decreasing priority)

SUBDD

AtmL40p 
AtmRes

SCM_COM SCM
ATMDYN_SCM
ATMDYN_SCM_EXT

CLOUDS_COM CLOUDS2 CLOUDS2_DRV

SURFACE
PBL_COM PBL_DRV PBL ATURB

LANDICE LANDICE_COM SURFACE_LANDICE LANDICE_DRV

GHY_COM GHY_DRV

VEG_DRV
! VEG_COM VEGETATION 
ENT_DRV ENT_COM ! + Ent

LAKES_COM LAKES

OCN_DRV OCEAN OCNML

SEAICE SEAICE_DRV ICEDYN_DUM
Zenith

RAD_COM RADIATION RAD_DRV
COSZ_2D RAD_UTILS ALBEDO READ_AERO ocalbedo

DIAG_COM DIAG DEFACC QUICKPRT

ATM_DRV ATMDYN_COM ATM_UTILS ATM_COM

MODEL_COM
IO_DRV
MODELE
MODELE_DRV

QUS_COM QUSDEF

FLUXES

STRAT_DUM

Components:
shared MPI_Support solvers giss_LSM 
dd2d
Ent

Component Options:
OPTS_Ent = ONLINE=YES PS_MODEL=FBB
OPTS_giss_LSM = USE_ENT=YES

Data input files:

! SCM input files
SCM_NML=SCM_ARM.nml            ! input variable namelist with units (dummy if no SCM input data)

! The set of forcings for a particular SCM test case typically does not include
! all of the data required to run Model E.  Each line below of the form
!   SHORTNAME=/path/to/user/directory/extractions/filename.nc
! corresponds to a location-dependent input dataset for which one of the following
! two options must be chosen.  The second option is provided for convenience.
! (1) Replace /path/to/user/directory/extractions/filename.nc with the
!     path to a single-column file that has already been created somehow.
!     Note that the GCM file-reading infrastructure considers single-column
!     files to be on a horizontal grid of size 1, and therefore the
!     two horizontal dimensions must be retained in netcdf variables
!     (which must have the netcdf names that model E expects).
!     For any files previously extracted from global data via option (2),
!     make sure that the path does not contain substring "/extractions/"
!     if option (2) will be used create other files.
! (2) Use exec/extract_scm.sh to sample gridded files at location
!     lon_targ, lat_targ.
!     Firstly,
!       replace /path/to/user/directory with a real path, preferably which
!        (a) contains a string denoting the SCM location/case being run
!        (b) is unlikely to be chosen by any other users on the system
!       Habits (a) and (b) will prevent clutter and accidental overwrites.
!       Note that
!        (a) substring "/extractions/" must be retained in each Users
!        (b) the rundeck paths indicate the resulting single-column files
!            to be read by the model.  GCMSEARCHPATH (from your modelErc)
!            is the location of the gridded file from which to extract the column
!        (c) There is no requirement that all files from which data are extracted are
!            on the same grid - dimension and coordinate information is scanned per-file.
!        (d) As currently programmed (10/2013), extract_scm.sh does require that
!            each file possesses 1D coordinate variables named lon(lon) and lat(lat).
!            Extraction from arbitrary grids (cubed-sphere etc.) will soon be enabled.
!        (e) NCO must be installed on your system and in your $PATH.
!     Secondly, execute
!        extract_scm.sh THISRUNDECK.R


! Topography, area fractions of surface types
TOPO=/path/to/user/directory/extractions/Z72X46N.cor4_nocasp.nc

! Atm. initial conditions (temperature, wind, humidity, surface pressure)
! on the model's vertical grid and consistent with the
! orography from TOPO.  While some SCM modes
! of operation may subsequently overwrite the values from
! this file, the model has not yet been programmed to
! skip reading this file if it is absent.
AIC=/path/to/user/directory/extractions/NCARIC.72x46.D7712010_ext.nc

! Optional: if absent, ozone is set to zero.
! Zero stratospheric ozone is usually a bad idea though.
!O3file=/path/to/user/directory/extractions/o3_2005_shindelltrop_144x90x49_1850-1997_ple.nc

! Optional: if absent, dust is set to zero
!DUSTaer=/path/to/user/directory/extractions/dust_mass_CakmurMillerJGR06_144x90x20x7x12_unlim.nc

! Optional: if absent and MADAER flag not set, aerosols are zero.
! If these files are omitted, rundeck parameters od_cdncx and cc_cdncx
! must not be set (or if set, set to zero.)
! Currently, these files must be used as a group (will change in future).
!TAero_SUL=/path/to/user/directory/extractions/SUL_Koch2008_kg_m2_144x90x20_1890-2000h.nc
!TAero_SSA=/path/to/user/directory/extractions/SSA_Koch2008_kg_m2_144x90x20h.nc
!TAero_NIT=/path/to/user/directory/extractions/NIT_Bauer2008_kg_m2_144x90x20_1890-2000h.nc
!TAero_OCA=/path/to/user/directory/extractions/OCA_Koch2008_kg_m2_144x90x20_1890-2000h.nc
!TAero_BCA=/path/to/user/directory/extractions/BCA_Koch2008_kg_m2_144x90x20_1890-2000h.nc
!TAero_BCB=/path/to/user/directory/extractions/BCB_Koch2008_kg_m2_144x90x20_1890-2000h.nc

! These ocean files are only needed if the TOPO file for the
! SCM case contains a nonzero ocean fraction.  Certain SCM
! modes of operation may also prescribe ocean surface conditions
! via mechanisms other than these files. SICE and ZSIFAC can be
! omitted if the simulation location is free of sea ice.
!OSST=/path/to/user/directory/extractions/OST4X5.B.1876-85avg.Hadl1.1.nc  ! SST
!OSST_eom=/path/to/user/directory/extractions/OST4X5.B.1876-85avg.Hadl1.1.nc  ! SST
! rsi var. in SICE is sea ice fraction, ZSIFAC var. dm is used to get ice thickness
!SICE=/path/to/user/directory/extractions/SICE4X5.B.1876-85avg.Hadl1.1.nc
!SICE_eom=/path/to/user/directory/extractions/SICE4X5.B.1876-85avg.Hadl1.1.nc
ZSIFAC=/path/to/user/directory/extractions/SICE4X5.B.1876-85avg.Hadl1.1.nc
OHT=/path/to/user/directory/extractions/zero_OHT_4x5.nc
!OCNML=XXX ! see comment above regarding the OCNML source code.
OCNML=/path/to/user/directory/extractions/zero_OCNML_4x5.nc  ! mixed layer depth (needed for post processing)

! These land-surface files are only needed if the TOPO file for
! the SCM case contains a nonzero land fraction.  But note
! that the GIC contains initial conditions for surface types other
! than the land surface and that the SOILIC method is an optional
! replacement for the land-surface GIC.
VEG=/path/to/user/directory/extractions/V72X46.1.cor2_no_crops.ext.nc
CROPS=/path/to/user/directory/extractions/CROPS2007_72X46N.cor4_nocasp.nc 
SOIL=/path/to/user/directory/extractions/S4X50093.ext.nc 
SOILCARB_global=/path/to/user/directory/extractions/soilcarb_top30cm_4x5.nc
TOP_INDEX=/path/to/user/directory/extractions/top_index_72x46_a.ij.ext.nc
CDN=/path/to/user/directory/extractions/CD4X500S.ext.nc  


! Optional land-surface file which prescribes soil initial
! conditions in intuitive intensive units (temperature,
! relative wetness, snow depth) rather than the extensive
! units (total heat and water per layer) of the arrays in
! the file GIC.
!SOILIC=/path/to/SOILIC.nc

! Optional file to specify a "background" roughness
! length for the land surface that is not tied to
! the scale-dependent topographic standard deviation in
! TOP_INDEX as per the procedure developed for 8x10 Model II.
! Note that the actual roughness length for the
! land surface is taken as the maximum of the vegetation-derived
! value and the value from either ROUGHL or TOP_INDEX.
!ROUGHL=/path/to/ROUGHL.nc

! Initial conditions for surface components from an arbitrary restart file
!GIC=/path/to/user/directory/extractions/GIC.E046D3M20A.1DEC1955.ext_1.nc
GIC=/path/to/user/directory/extractions/GIC.144X90.DEC01.1.ext_2.nc

! All input files below this line are location-independent.

GHG=GHG.CMIP6.1-2014.txt
RADN1=sgpgxg.table8
RADN2=LWTables33k_lowH2O_CO2_O3_planck_1-800
RADN4=LWCorrTables33k
RADN5=H2Ocont_MT_CKD
RADN3=miescatpar.abcdv2
RH_QG_Mie=oct2003.relhum.nr.Q633G633.table
!RADN7=STRATAER.VOL.1850-2012.May13_hdr
!RADN8=cloud.epsilon4.72x46
RADN9=solar.CMIP6official.ann1850-2299.nc ! need KSOLAR=2 
RADNE=topcld.trscat8

! optional files for optional diagnostics
!ISCCP=ISCCP.tautables
!dH2O=dH2O_by_CH4_monthly
MSU_wts=MSU_SSU_RSS_weights.txt

Label and Namelist:
PSCMq (Run SCM Planetary Master Free-Running GCM ICs w/qflux ocean)


&&PARAMETERS

! SCM parameters
                         ! test location ocean surface 
SCM_lon=73.10            ! Gan Island Ocean Site (deg)
SCM_lat=-0.63            ! Gan Island Ocean Site (deg)
SCM_sfc=0
! nominal gridbox area (m2) from a 72x46 lon-lat grid
SCM_area=247085409323.51

GLMELT_ON=0      !turn off GLMELT in run deck for SCM long runs

!for removing diurnal cycle set COSZ to a constant for run
!COSZ=###
!for removing seasonal cycle set orbital parameters 
!planetName='notEarth'
!eccentricity=0.000
!obliquity=0.000
!quantizeYearLength='T'

DTsrc=1800.     ! Atm. physics timestep.
NIsurf=1        ! Number of surface physics timesteps per atm. physics timestep.
NRAD=1          ! Full radiation calculation every NRAD physics timesteps.

! cloud tuning parameters
U00a=.50
U00b=0.55
radiusl_multiplier=0.8
radiusi_multiplier=0.7
wmu_multiplier=5.0

! Cloud inhomogeneity correction
KCLDEP=1    ! use a constant value for CLDEPS
EPSCON=0.12 ! use CLDEPS=0.12

CO2X=1.
O3X=0. ! turn of O3, to turn on delete this line and add O3file
H2OstratX=1.

H2ObyCH4=0.     ! deactivates strat.H2O generated by CH4
l_uniform_ghg=1 ! deactivates prescribed GHG distributions


! parameters that control temporally varying inputs:
! if set to 0, the current (day/) year is used: transient run
master_yr=1850
!crops_yr=1850  ! if -1, crops in VEG-file is used
!ghg_yr=1850
!ghg_day=182
volc_yr=-1
!volc_day=182
!aero_yr=1850
od_cdncx=0.        ! don't include 1st indirect effect
cc_cdncx=0.        ! don't include 2nd indirect effect (used 0.0036)
!o3_yr=-1850
! atmCO2=368.6          !uatm for year 2000 - enable for CO2 tracer runs

! radiation flags
KSIALB=0        ! 6-band albedo (Hansen) (=1 A.Lacis orig. 6-band alb)
KSOLAR=2
PTLISO=15.      ! press(mb) above which rad. assumes isothermal layers

xCDpbl=1.0

!madaer=3        ! indicates use of TAero_XXX aerosol files by radiation.


! parameters affecting diagn. output
cloud_aer_o3_rad_forc=0 ! Turn off clear-sky + aerosol + Ox forcing diagnostic
aer_rad_forc=0   ! if set =1, radiation is called numerous times - slow !!
cloud_rad_forc=1 ! calls radiation twice; use =0 to save cpu time
isccp_diags=0    ! use =0 to save cpu time, but you lose some key diagnostics
nda5d=13         ! use =1 to get more accurate energy cons. diag (increases CPU time)
nda5s=13         ! use =1 to get more accurate energy cons. diag (increases CPU time)
ndaa=13
nda5k=13
nda4=48          ! to get daily energy history use nda4=24*3600/DTsrc

! KCOPY=1: save acc and alternating checkpoint files only.
! KCOPY=2: save end-of-month rsf also (probably not useful for SCM).
KCOPY=1

! save alternating checkpoint files every Ndisk timesteps
! (checkpoint also saved when model reaches end time).
! Useful for perusing instantaneous states or checking
! checking whether model states are identical after coding
! rearrangements.
Ndisk=1440

! restart state is saveable every nssw timesteps.
Nssw=2

! SCM-useful GCM-native subdaily diagnostics system not yet imported to master branch
!SUBDD=' '        ! no sub-daily frequency diags
!NSUBDD=0         ! saving sub-daily diags every NSUBDD-th physics time step (1/2 hr)

SUBDD='u v t q rh z p_3d p_surf prec mcp ssp snowfall snowdp qcl qci'
SUBDD1='cldss cldmc cldss_2d totcld totcld_diag'
SUBDD2='gtempr shflx lhflx ustar pblht pwv lwp iwp tau_ss tau_mc'
!SUBDD3='olrrad olrcs lwds lwdscs lwus swds swus swdf'
!SUBDD4='dq_turb dth_turb dq_mc dth_mc dq_ss dth_ss dth_sw dth_lw dth_rad'
!SUBDD5='dq_ls dth_ls dq_nudge dth_nudge'
!SUBDD6='isccp_sunlit isccp_ctp isccp_tau isccp_lcld isccp_hcld'
NSUBDD=1         ! saving sub-daily diags every NSUBDD-th physics time step (1/2 hr)
SCM_PlumeDiag=0  !to save Plume diagnostics set SCM_PlumeDiag=1
WRITE_ONE_FILE=1 ! all outputs to a single file

! KOCEAN=0 means prescribed surface ocean conditions.  This parameter is currently
! mandatory even if ocean is absent at the SCM location.
KOCEAN=1    !ocean is computed
qfluxX=0.
Kvflxo=0
ocn_cycl=1

! variable lakes.  Probably unimportant for typical SCM simulation lengths.
! note that lakes can only evolve in response to local precip, evap, runoff.
variable_lk=1

wsn_max=2.   ! restrict snow depth to 2 m-h2o (if 0. snow depth is NOT restricted)

&&END_PARAMETERS

 &INPUTZ
 YEARI=1949,MONTHI=12,DATEI=1,HOURI=0, ! pick IYEAR1=YEARI (default) or < YEARI
 YEARE=1949,MONTHE=12,DATEE=30,HOURE=0,     KDIAG=12*0,9,
 ISTART=2,IRANDI=0, YEARE=1949,MONTHE=12,DATEE=3,HOURE=0,
/
