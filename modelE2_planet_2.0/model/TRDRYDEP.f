#include "rundeck_opts.h"

#ifdef TRACERS_DRYDEP
      SUBROUTINE get_dep_vel(I,J,ITYPE,CH,WS,USTAR,TEMPK
     & ,RHOSRF,RH,VISC
     & ,TcanopyC,FWETCUT
     & ,Soiltemp,Soilmoist
#ifdef DRYDEP_AEROSOLS_OLD
     & ,OBK,ZHH
#endif
#ifndef TRACERS_TOMAS
     & ,TRNRADIUS,TRNDENS
#endif
     & ,TRNMM
     & ,VD1
#ifdef TRACERS_DRYDEP_DIAG
     & ,EGSTOM1,EGCUT_DRY1,EGGRO_DRY1,EGCUT_WET1,EGGRO_WET1
     & ,EGCUT_SNOW1,EGGRO_SNOW1,VD_FOREST,VD_CROP
     & ,VD_GRASS,VD_SHRUB,VD_BARE
     & ,EGSTOM_FOREST,EGSTOM_CROP
     & ,EGSTOM_GRASS,EGSTOM_SHRUB
     & ,frac_forest,frac_crop,frac_grass
     & ,frac_shrub,frac_bare
#endif
     & )


! GLOBAL parameters and variables:  
      USE CONSTANT,only: tf,pi,grav,mwat,lhe,rt3,byrt3,
     &    visc_air0,visc_air_kin0,rhow 
      USE ITYPE_ENUM,only: ITYPE_OCEAN,ITYPE_LANDICE, 
     &    ITYPE_OCEANICE,ITYPE_LAND
      USE SEAICE_COM, only: si_atm
      USE OldTracer_mod,only: tr_wd_TYPE,nPART,trname, 
     &    dodrydep,F0_glob=>F0, hygro=>hygro_oma,
     &    KH_298_glob=>KH_298,deltaH_R_glob=>deltaH_R,
     &    K1_298_glob=>K1_298,deltaH1_R_glob=>deltaH1_R,
     &    K2_298_glob=>K2_298,deltaH2_R_glob=>deltaH2_R
      USE TRACER_COM,only: NTM
#ifdef TRACERS_SPECIAL_Shindell
      USE TRCHEM_Shindell_COM,only: pNOx
#endif
      USE ent_com,only: entcells
      USE ent_mod,only: ent_get_exports
      USE ent_const,only: N_COVERTYPES, 
     &    PTRACE,NPOOLS,N_CASA_LAYERS,NLIVE,CARBON,
     &    NEEDLELEAF
      USE ent_pfts,only: pfpar
      USE ghy_com,only: snowbv, fr_snow_ij
#ifdef TRACERS_TOMAS 
      USE TRACER_COM,only: NBS, NBINS, n_ASO4
#endif
      USE Dictionary_mod,only: sync_param
 
      IMPLICIT NONE


! Local parameters and variables and arguments

! BELOW ARE VARIABLES READ IN FROM PBL 
!@var I,J GCM horizontal position
!@var ITYPE GCM surface type  
      INTEGER,INTENT(IN) :: I,J,ITYPE
!@var USTAR Friction velocity (m s-1)
!@var TEMPK Surface air temperature (K)
!@var RHOSRF surface air density (kg m-3; assuming psurf going into calc is hPa)
!@var RH surface relative humidity (fraction) 
!@var VISC dynamic viscosity of air for TEMPK  (kg/(m s))
!@var CH surface transfer coefficient for heat (unitless)
!@var WS surface wind speed (m s-1)
!@var TcanopyC canopy temperature (C)
!@var FWETCUT fraction of leaf that is wet
!@var Soiltemp soil temperature (*I think in degC*)
!@var Soilmoist soil moisture (m/m)
      REAL*8,INTENT(IN) :: CH,WS,USTAR,TEMPK,
     &        RHOSRF,RH,VISC,TcanopyC,FWETCUT,
     &        Soiltemp,Soilmoist
!@var OBK Obukhov length (m)  
!@var ZHH Boundary layer height (m)
#ifdef DRYDEP_AEROSOLS_OLD
      REAL*8,INTENT(IN) :: OBK,ZHH
#endif
!@var TRNMM gas molecular weights (g mol-1)
      REAL*8,INTENT(IN),DIMENSION(NTM) :: TRNMM
!@var TRNRADIUS aerosol radius (m) 
!@var TRNDENS aerosol density (kg m-3) 
#ifndef TRACERS_TOMAS
      REAL*8,INTENT(IN),DIMENSION(NTM) :: TRNRADIUS,TRNDENS
#endif

!! BELOW ARE PARAMETERS DEFINED HERE
!@var KAPPA thermal diffusivity of air (m2/s)
!@var VONK  von Karman constant (unitless)
!@var DIFF_H2O diffusivity of water vapor in air (m2/s)
!@var GASC1 gas constant R = 0.08205 in atm/M/K (S&P2006)
!@var kB Boltzmann constant (J/K) (S&P2011)
!@var twothirds 2/3
!@var onethird 1/3
      REAL*8 :: DIFF_H2O,KAPPA,VONK,GASC1,twothirds,onethird,kB
!! The below choices are a bit arbitrary and should be updated according to new literature/constraints
!@var pH_OCEAN pH of ocean
!@var pH_LAND pH of land - very uncertain - taken from https://doi.org/10.5194/gmd-13-2879-2020
!@var pH_SNOW pH of snow - very uncertain - taken from https://doi.org/10.5194/gmd-13-2879-2020
      REAL*8 :: pH_OCEAN,pH_LAND,pH_SNOW
!! The below need to be defined for the code to work; hopefully changes in them don't matter much
!@var LOW_LAI threshold for barely any vegetation (important for avoiding dividing by zero and incorporating in-canopy turbulent transport)  
!@var BIG_RESIST used to set a resistance equivalent of zero conductance
      REAL*8 :: LOW_LAI,BIG_RESIST

!! BELOW ARE AEROSOL DRY DEP TUNING PARAMETERS
!@var Cb coefficient for brownian diffusion to vegetation (unitless)
!@var Cim coefficient for impaction to vegetation (unitless)
!@var Cin coefficient for interception by vegetation (unitless)
!@var beta1 parameter used in impaction to vegetation scheme (unitless)
!@var Vphor velocity represeting phoretic effects for ice and water (m/s)
      REAL*8 :: Cb, Cim, Cin, Vphor, beta1

!! BELOW ARE VARIABLES RELATED TO AEROSOL DRY DEP 
!@var Sc particle Schmidt number (unitless)
!@var St particle Stokes number (unitless)
!@var Dk particle diffusivity (m2/s)
!@var fstick fraction of particle that sticks to surface after deposition (i.e., does not bounce off) (unitless) 
!@var GGRAV conductance for gravitational settling (m/s)
!@var Eb efficiency for brownian diffusion to vegetation or ground (unitless)
!@var Eim efficiency for impaction to vegetation (unitless)
!@var Ein efficiency for interception by vegetation (unitless)
!@var trrwet wet aerosol radius (m; need to hydrate OMA aerosols; MATRIX aerosols already hydrated)
!@var VGS function for gravitational settling velocity in TRACERS.f
!@var SLIPC function for Cunningham slip correction in TRACERS.f
      REAL*8 :: Sc,St,Dk,fstick,GGRAV,Eb,Eim,Ein,trrwet
      REAL*8 :: VGS,SLIPC
#ifdef DRYDEP_AEROSOLS_OLD
!@var VDS
!@var IVSMAX
      REAL*8 :: VDS,IVSMAX
#endif
#ifdef TRACERS_TOMAS
      integer KBI,binnum,Ni !size bin # that corresponds to current tracer
      real*8 RB_TOMAS(NBINS)  !quasilaminar sublayer resistance (s m-1)
      real*8 vs(NBINS)  !gravitational settling velocity (m s-1)
      real*8 Dp(NBINS),density(NBINS) !particle diameter (m), density
#endif

!! BELOW ARE THE GAS DRY DEP TUNING PARAMETERS
!@var IR_GRO_S initial resistance for SO2 for the ground (s/m)
!@var IR_GRO_O_WET initial resistance for O3 for wet ground (s/m)
!@var IR_GRO_O_DRY initial resistance for O3 for dry ground (s/m)
!@var IR_CUT_S initial resistance for SO2 for leaf cuticles (s/m)
!@var IR_CUT_O_WET initial resistance for O3 for wet leaf cuticles (s/m)
!@var IR_CUT_O_DRY initial resistance for O3 for dry leaf cuticles (s/m)
!@var IR_WATER_S initial resistance for SO2 for water (s/m)
!@var IR_WATER_O initial resistance for O3 for water (s/m)
!@var IR_SNOW_S initial resistance for SO2 for snow (s/m)
!@var IR_SNOW_O initial resistance for O3 for snow (s/m)
      REAL*8 :: IR_GRO_S,IR_GRO_O_WET,IR_GRO_O_DRY,
     &       IR_WATER_S,IR_WATER_O,
     &       IR_MESO_S,IR_MESO_O,IR_CUT_S,
     &       IR_CUT_O_WET,IR_CUT_O_DRY,
     &       IR_SNOW_S,IR_SNOW_O

!! BELOW ARE VARIABLES RELATED TO GAS DRY DEP FOR ITYPE_LAND 
!@var EGSTOM Effective stomatal conductance for species K & Ent type LDT (m/s) 
!@var EGCUT_WET Effective wet cuticular conductance for species K & Ent type LDT (m/s)
!@var EGCUT_DRY Effective dry cuticular conductance for species K & Ent type LDT (m/s)
!@var EGCUT_SNOW Effective snow cuticular conductance for species K & Ent type LDT (m/s)
!@var EGGRO_WET Effective wet ground conductance for species K & Ent type LDT (m/s)
!@var EGGRO_DRY Effective dry ground conductance for species K & Ent type LDT (m/s)
!@var EGGRO_SNOW Effective snow ground conductance for species K & Ent type LDT (m/s)
      REAL*8,DIMENSION(NTM,N_COVERTYPES) ::
     &       EGSTOM,EGCUT_WET,
     &       EGCUT_DRY,EGCUT_SNOW,
     &       EGGRO_WET,EGGRO_DRY,EGGRO_SNOW

!! BELOW ARE VARIABLES RELATED TO GAS DRY DEP FOR ITYPE_LAND 
!@var FRCUTWET fraction of wet vs. dry cuticular gas deposition that is wet
!@var FRCUTSNOW fraction of total cuticular gas deposition that is snow
!@var FRLEAF fraction of leaf vs. ground gas deposition that is leaf
!@var FRSTOM fraction of leaf gas deposition that is stomatal
!@var FRGROSNOW fraction of total ground gas deposition that is snow
!@var FRGROWET fraction of wet vs. dry ground gas deposition that is wet
      REAL*8 :: FRCUTWET,FRCUTSNOW,FRLEAF,FRSTOM,FRGROSNOW,FRGROWET

!! BELOW ARE VARIABLES RELATED TO GAS DRY DEP FOR ITYPE ice
!@var FMP is fraction of melt pond
      REAL*8 :: FMP

!! BELOW ARE VARIABLES RELATED TO GAS DRY DEP INDIVIDUAL PROCESSES 
!@var RBGRO resistance for quasi-laminar transport thru boundary layer between ground and air (s/m), fixed at Massman 2004 values 
!@var RB_WATER_ICE resistance for transport thru boundary layer between ice/water and air (s/m)
!@var RCUT resistance to cuticular uptake (s/m)
!@var RCUT_WET resistance to wet cuticular uptake (s/m)
!@var RCUT_DRY resistance to dry cuticular uptake (s/m)
!@var RGRO resistance to ground uptake (s/m)
!@var RGRO_WET resistance to wet ground uptake (s/m)
!@var RGRO_DRY resistance to dry ground uptake (s/m)
!@var RSTOM resistance to stomatal uptake (s/m)
!@var RMESO resistance to reactions with fluids and tissues (e.g. mesophyll) inside the leaf (s/m)
!@var RSNOW resistance to snow (s/m)
      REAL*8 :: RBGRO,RB_WATER_ICE,RCUT,RCUT_WET,RCUT_DRY,
     &       RGRO,RGRO_WET,RGRO_DRY,RSTOM,RMESO,RSNOW
!@var GAC conductance for in-canopy turbulent transport (m/s)
!@var GBLEAF conductance for quasi-laminar transport through leaf boundary layer (m/s)
!@var GCUT conductance for cuticular uptake (m/s)
!@var GLEAF conductance for leaf uptake (m/s)
!@var GGRO conductance for ground uptake (m/s)
!@var GSTOM conductance for stomatal uptake (m/s)
      REAL*8 :: GAC,GBLEAF,GCUT,GLEAF,GGRO,GSTOM

!! BELOW ARE VARIABLES RELATED TO SPECIES SPECIFIC COMPONENT OF GAS DRY DEP
!@var F0 reactivity of gas
!@var diff molecular diffusivity of gas
!@var RBSCAL scaling variable for diffusivity of gs for Rb term 
!@var HEFF_VEG effective Henry law constant for vegetation
!@var HEFF_VEG_SNOW effective Henry law constant for snow-covered vegetation
!@var HEFF_GRO effective Henry law constant for the ground
!@var HEFF_GRO_SNOW effective Henry law constant for snow-covered ground
!@var HEFF effective Henry law constant for ice or ocean
      REAL*8 :: F0,diff,RBSCAL
      REAL*8 :: HEFF_VEG,HEFF_GRO,HEFF_VEG_SNOW,HEFF_GRO_SNOW,HEFF
#ifdef TRACERS_SPECIAL_Shindell
      REAL*8 :: HEFF_VEG_NO,HEFF_VEG_SNOW_NO,HEFF_NO,
     &          HEFF_GRO_NO,HEFF_GRO_SNOW_NO
      REAL*8 :: KH_298_NO,deltaH_R_NO,F0_NO,TRNMM_NO,TRNMM_NO2
#endif

!! BELOW ARE VARIABLES USED FOR BOTH AEROSOL AND GAS DRY DEP
!@var VD Deposition velocity for species K & Ent type LDT (used for ITYPE == land)
!@var GA turbulent transport from 10m above surface to surface (surface=top of vegetation canopy if there is vegetation) (m/s)
      REAL*8, DIMENSION(NTM,N_COVERTYPES) :: VD
      REAL*8 :: GA
!@var G_DUMMY dummy conductance 
!@var DUMMY dummy variable 
!@var DUMMY_O variable for creating different dep to ice vs. water due to reactivity
!@var DUMMY_S variable for creating different dep to ice vs. water due to solubility
!@var DUMMY_PH variable for distinguishing ocean vs. land ice pH 
      REAL*8 :: DUMMY,G_DUMMY,DUMMY_O,DUMMY_S,DUMMY_PH

!! Below are variables that are pulled over from land model
      REAL*8 :: fv, fb
! pft specific variables
!@var lai_pft leaf area index per pft such that sum(lai_pft) is total lai of cell, so divide by frac_pft to get absolute lai
!@var frac_pft is the fraction of pft per gridcell
!@var gs_pft is the stomatal conductance for h2o per pft (m s-1)
      REAL*8, DIMENSION(N_COVERTYPES) :: frac_pft,lai_pft,gs_pft
! grid cell average variables
!@var soilcpools soil pools of different nutrients (g m-2)
!@var canopyheight height of the canopy (m) 
      REAL*8, DIMENSION(PTRACE,NPOOLS,N_CASA_LAYERS) :: soilcpools
      REAL*8 :: canopyheight

! variables calculated here related to land model
!@var leafdim characteristic leaf dimension (m) (defined values using  Petroff and Zhang 2010)  
! based on three options from Ent (broadleaf, needleleaf, monocot)
      REAL*8, DIMENSION(3) :: leafdim
!@var leafindex index for leaf dimension (from Ent; three options; broadleaf, needleleaf, monocot)
      INTEGER :: leafindex

!@var asoilCpoolsum sum of soil carbon pools
!@var fm fraction of canopy covered by snow (not completely sure... Igor sent these calculations)
!@var lai absolute leaf area index per pft
!@var FWETGRO fraction of wet ground 
!@var FSNOWCUT fraction of leaves that are snow covered
!@var FSNOWGRO fraction of ground that is snow covered
      REAL*8 :: asoilCpoolsum,fm
      REAL*8, DIMENSION(N_COVERTYPES) :: lai
      REAL*8 :: FWETGRO,FSNOWCUT,FSNOWGRO

!@var LDT loop variable for Ent land type
!@var K loop variable for tracer species
      INTEGER :: K,LDT

!! BELOW ARE OUTPUT FOR EACH SPECIES (GASES & AEROSOLS) 
! PATHWAY SPECIFIC VARIABLES NONZERO FOR GASES ONLY
!@var VD1 Average deposition velocity for species K for ITYPE (m/s)
      REAL*8, INTENT(OUT), DIMENSION(NTM) :: VD1
!@var EGSTOM1 Effective stomatal conductance averaged across Ent types (m/s) 
!@var EGCUT_WET1 Effective wet cuticular conductance averaged across Ent types (m/s)
!@var EGCUT_DRY1 Effective dry cuticular conductance averaged across Ent types (m/s)
!@var EGCUT_SNOW1 Effective snow cuticular conductance averaged across Ent types (m/s)
!@var EGGRO_WET1 Effective wet ground conductance averaged across Ent types (m/s)
!@var EGGRO_DRY1 Effective dry ground conductance averaged across Ent types (m/s)
!@var EGGRO_SNOW1 Effective snow ground conductance averaged across Ent types (m/s)
!@var VD_FOREST Average deposition velocity for forests for species K (m/s)
!@var VD_CROP Average deposition velocity for crops for species K (m/s)
!@var VD_BARE Average deposition velocity for bare surfaces for species K (m/s)
!@var VD_GRASS Average deposition velocity for grasslands for species K (m/s)
!@var VD_SHRUB Average deposition velocity for shrubs for species K (m/s)
!@var EGSTOM_FOREST Average effective stomatal conductance for forests for species K (m/s)
!@var EGSTOM_CROP Average effective stomatal conductance for crops for species K (m/s)
!@var EGSTOM_GRASS Average effective stomatal conductance for grasslands for species K (m/s)
!@var EGSTOM_SHRUB Average effective stomatal conductance for shrubs for species K (m/s)
#ifdef TRACERS_DRYDEP_DIAG
      REAL*8, INTENT(OUT), DIMENSION(NTM) ::
     &       VD_FOREST,VD_CROP,VD_GRASS,VD_SHRUB,VD_BARE,
     &       EGSTOM_FOREST,EGSTOM_CROP,EGSTOM_GRASS,EGSTOM_SHRUB,
     &       EGSTOM1,EGCUT_DRY1,EGGRO_DRY1,EGCUT_WET1,
     &       EGCUT_SNOW1,EGGRO_WET1,EGGRO_SNOW1
#endif


!! BELOW ARE FRACTIONS OF LAND COVERED BY AGGREGATE PFT FOR ITYPE_LAND 
!@var frac_forest fraction of land covered by forests 
!@var frac_crop fraction of land covered by crops
!@var frac_grass fraction of land covered by grass
!@var frac_shrub fraction of land covered by shrubs
!@var frac_bare fraction of land covered by bare ground
#ifdef TRACERS_DRYDEP_DIAG
      REAL*8, INTENT(OUT) :: frac_forest,frac_crop,frac_grass,
     &      frac_shrub,frac_bare
#endif

! SET PARAMETER VALUES 
! Paramters whose values should not change
      twothirds = 2.d0/3.d0
      onethird  = 1.d0/3.d0
      KAPPA = 2.d-5
      kB = 1.381d-23
      VONK = 0.4d0
      GASC1 = 0.08205d0
      BIG_RESIST = 999999.d0
      LOW_LAI = 1.d-3

! Tuning parameters for gas dry dep  
      RBGRO = 40.d0
      IR_GRO_S = 1.2d10 
      IR_GRO_O_WET = (2.5d3)/6.d-2
      IR_GRO_O_DRY = ((2.5d2)/6.d-2)*1.25d0
      IR_WATER_S = 1.d9 
      IR_WATER_O = 5.d3
      IR_MESO_S = 1.d6 
      IR_MESO_O = 1.d1 
      IR_CUT_S = 1.75d9
      IR_CUT_O_DRY = ((1.5d3)/6.d-2) 
      IR_CUT_O_WET = (2.d2)/6.d-2
      IR_SNOW_S = 5.d0*IR_WATER_S
      IR_SNOW_O = 3.5d0*IR_WATER_O

! Tuning parameters for aerosol dry dep 
      beta1 = 1.7d0
      Cb = 4.d-1
      Cim = 4.d-1
      Cin = 8.d0

! Define parameters related to surface characteristics
! these parameters should change w/ better constraints
      pH_OCEAN = 8.1d0
      pH_SNOW = 5.4d0
      pH_LAND = 7.d0
      leafdim = (/3.d-2,0.15d-2,1.d-2/) 

#ifdef TRACERS_SPECIAL_Shindell
! Define parameters for NO  
! Values in ShindellTracersMetadata related to HSTAR and F0 are for NO2
      KH_298_NO = 1.92d-03 ! M/atm (Schwantes et al. 2020)
      deltaH_R_NO = 1762.d0 ! (Schwantes et al. 2020)
      F0_NO = 0.d0 
! Value in ShindellTracersMetadata for molecular weight is for N
      TRNMM_NO = 30.01d0  ! g/mol
      TRNMM_NO2 = 46.01d0 ! g/mol
#endif

! Initialize output as zero
      DO K = 1,NTM
#ifdef TRACERS_DRYDEP_DIAG
          VD_FOREST(K)       = 0.d0
          VD_CROP(K)         = 0.d0
          VD_GRASS(K)        = 0.d0
          VD_SHRUB(K)        = 0.d0
          VD_BARE(K)         = 0.d0
          EGSTOM_FOREST(K)   = 0.d0
          EGSTOM_CROP(K)     = 0.d0
          EGSTOM_GRASS(K)    = 0.d0
          EGSTOM_SHRUB(K)    = 0.d0
          EGSTOM1(K)         = 0.d0
          EGCUT_WET1(K)      = 0.d0
          EGCUT_DRY1(K)      = 0.d0
          EGCUT_SNOW1(K)     = 0.d0
          EGGRO_WET1(K)      = 0.d0
          EGGRO_DRY1(K)      = 0.d0
          EGGRO_SNOW1(K)     = 0.d0
#endif
          VD1(K)             = 0.d0
      ENDDO 
#ifdef TRACERS_DRYDEP_DIAG   
      frac_forest = 0.d0
      frac_crop = 0.d0
      frac_grass = 0.d0
      frac_shrub = 0.d0
      frac_bare = 0.d0
#endif
! leave the subroutine if wind speed is zero 
! all deposition velocities and effective conductances will be zero 
#ifndef DRYDEP_AEROSOLS_OLD
      IF (WS==0.d0) return 
#endif    
          
#ifdef TRACERS_TOMAS
      call dep_getdp(i,j,1,Dp,density)
      do KBI=1,NBINS         
         Dk=kB*TEMPK/(3.0*pi*visc_air0*Dp(KBI)) 
         Sc=visc_air_kin0/Dk
! oec 1/2022 tomas aerosols may need to be hydrated, not sure
! not hydrating currently
         vs(KBI)=VGS(RHOSRF,Dp(KBI)/2.d0,density(KBI),VISC)
         St=vs(KBI)*USTAR**2/grav/visc_air_kin0
         RB_TOMAS(KBI)=1.d0/(USTAR
     &        *(Sc**(-twothirds)+10.d0**(-3.d0/St)))
      enddo
#endif

! Compute deposition velocity for gases + aerosols

!  Calculate diffusivity of water vapor in air
!  OEC ATTN got this from GFDL model, need reference
!  should change to S&P 8.10 prob
      DIFF_H2O = 21.2d-6*(1.d0+0.0071d0*(TEMPK-tf))

!  Calculate above-canopy/surface turbulent transport
      GA = CH*WS

!  Calculate deposition for different ITYPES differently
      SELECT CASE(ITYPE)
      CASE(ITYPE_LAND)

        call get_fb_fv(fb,fv,i,j)

!  Read entcell variables
!  Read in pft-level variables
        call ent_get_exports(entcells(i,j),
     &           leaf_area_index_pft=lai_pft)
        call ent_get_exports(entcells(i,j),
     &           vegetation_fractions=frac_pft)
        call ent_get_exports(entcells(i,j),
     &           canopy_conductance_pft=gs_pft)
!  Read in cell-level variables
        call ent_get_exports(entcells(i,j),
     &           canopy_height=canopyheight)
        call ent_get_exports(entcells(i,j),
     &           soilcpools=soilcpools)

!  Calculate cell-level quantities needed for dry dep
!  Calculate soil carbon pools (g m-2) 
        asoilCpoolsum =
     &       sum( soilcpools(CARBON,NLIVE+1:NPOOLS,1:N_CASA_LAYERS) )
!  Convert to kg m-2
        asoilCpoolsum = asoilCpoolsum*1.d-3

!  Calculate fraction of snow on canopy & ground below canopy 
!  Note that I don't think these exact quantities are calculated for the hydrology code, 
!  but these calculations should follow hydrology code 
!  Note fraction of snow on ground below canopy is not 
!  the same as snow on bare ground (which is fr_snow_ij(1,i,j))
        IF (canopyheight == 0.d0) THEN
            FSNOWCUT = 0.d0
            FSNOWGRO = fr_snow_ij(1,i,j)
        ELSE
            fm=1.d0-EXP(-snowbv(2,i,j)/((canopyheight*0.1d0)+1.d-12))
            FSNOWCUT = fr_snow_ij(2,i,j)*fm
            FSNOWGRO = fr_snow_ij(2,i,j)*(1.d0-fm)
        ENDIF

!  Calculate fraction of ground that is wet  
!  Assume its the same as the amount of soil moisture at the top level of soil (uncertain)
         FWETGRO = Soilmoist
!  This was leading to some negative very very small dry soil uptake so have the following
         FWETGRO = MIN(FWETGRO,1.d0)
         FWETGRO = MAX(FWETGRO,0.d0)

!  LAI needs to be defined outside of tracer loop in case first tracer is particulate 
         DO LDT = 1,N_COVERTYPES
            IF (frac_pft(LDT).eq.0.d0) CYCLE
            lai(LDT) = lai_pft(LDT)/frac_pft(LDT)
         ENDDO 

! Calculate deposition velocities
         DO K = 1,NTM
           IF (.not. dodrydep(K)) CYCLE
            IF (tr_wd_TYPE(K)/=nPART) THEN ! GASES 
! Calculate gas-specific dry dep parameters
                  F0=F0_glob(K)
                  DIFF=DIFF_H2O*SQRT(mwat/TRNMM(K)) ! Graham's Law
#ifdef TRACERS_SPECIAL_Shindell
! Need to weigh NOx dep parameters by NO/NO2 
! pNOx is the fraction of NOx that is NO2
                  IF (trname(K) == 'NOx') THEN
                     DUMMY = pNOx(1,i,j)*TRNMM_NO2
     +                + (1.d0 - pNOx(1,i,j))*TRNMM_NO
                     DIFF = DIFF_H2O * SQRT(mwat/DUMMY)
                  ENDIF
#endif
! Calculate species scaling for quasi-laminar transport 
                  RBSCAL=(KAPPA/DIFF)**twothirds

!! GROUND DRY DEP
! Get effective Henry's Law constant for ground & snow on ground
                  CALL GET_HSTAR(trname(K),KH_298_glob(K),
     +             deltaH_R_glob(K),K1_298_glob(K),
     +             deltaH1_R_glob(K),K2_298_glob(K),deltaH2_R_glob(K),
     +             pH_LAND,Soiltemp+tf,HEFF_GRO)
                  CALL GET_HSTAR(trname(K),KH_298_glob(K),
     +             deltaH_R_glob(K),K1_298_glob(K),
     +             deltaH1_R_glob(K),K2_298_glob(K),deltaH2_R_glob(K),
     +             pH_SNOW,Soiltemp+tf,HEFF_GRO_SNOW)
#ifdef TRACERS_SPECIAL_Shindell
! Need to weigh NOx dep parameters by NO/NO2 
! pNOx is the fraction of NOx that is NO2
                  IF (trname(K) == 'NOx') THEN
                     CALL GET_HSTAR('NO',KH_298_NO,deltaH_R_NO,0.d0,
     +             0.d0,0.d0,0.d0,
     +             pH_LAND,Soiltemp+tf,HEFF_GRO_NO)
                     CALL GET_HSTAR('NO',KH_298_NO,deltaH_R_NO,0.d0,
     +             0.d0,0.d0,0.d0,
     +             pH_SNOW,Soiltemp+tf,HEFF_GRO_SNOW_NO)
                     HEFF_GRO = pNOx(1,i,j)*HEFF_GRO
     +                + (1.d0 - pNOx(1,i,j))*HEFF_GRO_NO
                     HEFF_GRO_SNOW = pNOx(1,i,j)*HEFF_GRO_SNOW
     +                + (1.d0 - pNOx(1,i,j))*HEFF_GRO_SNOW_NO
                  ENDIF
#endif
! Calculate dimensionless Henry's Law constant 
                  HEFF_GRO=HEFF_GRO*GASC1*(Soiltemp+tf)
                  HEFF_GRO_SNOW=HEFF_GRO_SNOW*GASC1*(Soiltemp+tf)
! Calculate dry ground resistance 
                  IF (F0==0.d0) THEN
                     RGRO_DRY=BIG_RESIST
                  ELSE
                     RGRO_DRY=IR_GRO_O_DRY/(F0*SQRT(Soiltemp+tf))
! Increase dry ground uptake for oxidants with soil carbon
                     IF ((trname(K)=='H2O2') .or. (trname(K)=='Ox')
     +                  .or. (trname(K)=='NOx')) THEN
                            IF (asoilCpoolsum>3.d0) THEN
                                RGRO_DRY=RGRO_DRY/LOG(asoilCpoolsum)
                            ELSE
                                RGRO_DRY=RGRO_DRY*2.d0
                            ENDIF
                     ENDIF
                  ENDIF
! Calculate wet ground resistance
                  RGRO_WET=1.d0/(HEFF_GRO/IR_GRO_S)
! Tune values for H2O2 & Ox
                  IF ((trname(K)=='H2O2')      
     +                           .or. (trname(K)=='Ox')) THEN
                     RGRO_WET=IR_GRO_O_WET/(HEFF_GRO*
     +                  F0*SQRT(Soiltemp+tf))
                  ENDIF
! Add wet and dry ground resistances in parallel
                  RGRO=1.d0/(FWETGRO/RGRO_WET+(1.d0-FWETGRO)/RGRO_DRY)
! Calculate wet fraction of wet+dry ground deposition
                  FRGROWET = (FWETGRO/RGRO_WET)*RGRO
! Calculate snow resistance for ground
                  RSNOW=1.d0/(HEFF_GRO_SNOW/IR_SNOW_S+F0/IR_SNOW_O)
! Add snow pathways for ground in parallel with wet & dry 
                  RGRO=1.d0/(FSNOWGRO/RSNOW+(1.d0-FSNOWGRO)/RGRO)
! Calculate snow fraction of ground deposition 
                  FRGROSNOW=(FSNOWGRO/RSNOW)*RGRO

!! CANOPY STUFF
                  IF (fv.gt.0.d0) THEN 
! Get effective Henry's Law constant for canopy & snow on canopy
                   CALL GET_HSTAR(trname(K),KH_298_glob(K),
     +             deltaH_R_glob(K),K1_298_glob(K),
     +             deltaH1_R_glob(K),K2_298_glob(K),deltaH2_R_glob(K),
     +             pH_LAND,TcanopyC+tf,HEFF_VEG)
                   CALL GET_HSTAR(trname(K),KH_298_glob(K),
     +             deltaH_R_glob(K),K1_298_glob(K),
     +             deltaH1_R_glob(K),K2_298_glob(K),deltaH2_R_glob(K),
     +             pH_SNOW,TcanopyC+tf,HEFF_VEG_SNOW)
#ifdef TRACERS_SPECIAL_Shindell
! Need to weigh NOx dep parameters by NO/NO2 
! pNOx is the fraction of NOx that is NO2
                   IF (trname(K) == 'NOx') THEN
                     CALL GET_HSTAR('NO',KH_298_NO,deltaH_R_NO,0.d0,
     +              0.d0,0.d0,0.d0,
     +              pH_LAND,TcanopyC+tf,HEFF_VEG_NO)
                     CALL GET_HSTAR('NO',KH_298_NO,deltaH_R_NO,0.d0,
     +              0.d0,0.d0,0.d0,
     +              pH_SNOW,TcanopyC+tf,HEFF_VEG_SNOW_NO)
                     HEFF_VEG = pNOx(1,i,j)*HEFF_VEG
     +                + (1.d0 - pNOx(1,i,j))*HEFF_VEG_NO
                     HEFF_VEG_SNOW = pNOx(1,i,j)*HEFF_VEG_SNOW
     +                + (1.d0 - pNOx(1,i,j))*HEFF_VEG_SNOW_NO
                   ENDIF
#endif
! Calculate dimensionless Henry's Law constant 
                   HEFF_VEG=HEFF_VEG*GASC1*(TcanopyC+tf)
                   HEFF_VEG_SNOW=HEFF_VEG_SNOW*GASC1*(TcanopyC+tf)
! Calculate mesophyll resistance
                   RMESO=1.d0/(HEFF_VEG/IR_MESO_S+F0/IR_MESO_O)
! Calculate wet leaf cuticular resistance 
                   RCUT_WET=1.d0/(HEFF_VEG/IR_CUT_S)
! Tune values for H2O2 & Ox
                   IF ((trname(K)=='H2O2') .or. (trname(K)=='Ox')) THEN
                      RCUT_WET=IR_CUT_O_WET/(HEFF_VEG*
     +                     F0*SQRT(TcanopyC+tf))
                   ENDIF
! Calculate dry leaf cuticular resistance
                   IF (F0==0.d0) THEN
                      RCUT_DRY=BIG_RESIST
                   ELSE
                      RCUT_DRY=IR_CUT_O_DRY/(F0
     +                     *SQRT(TcanopyC+tf))
                   ENDIF
! Add wet and dry cuticular resistances in parallel
                   RCUT=1.d0/(FWETCUT/RCUT_WET
     +                     +(1.d0-FWETCUT)/RCUT_DRY)
! Calculate wet frac of wet+dry cuticular deposition
                   FRCUTWET=(FWETCUT/RCUT_WET)*RCUT
! Calculate snow resistance for leaf cuticles 
                   RSNOW=1.d0/(HEFF_VEG_SNOW/IR_SNOW_S
     +                     +F0/IR_SNOW_O)
! Add snow pathways for cuticles in parallel with wet & dry
                   RCUT=1.d0/(FSNOWCUT/RSNOW
     +                     +(1.d0-FSNOWCUT)/RCUT)
! Calculate snow fraction of cuticular deposition 
                   FRCUTSNOW = (FSNOWCUT/RSNOW)*RCUT
                  ENDIF 
! Loop over Ent COVERTYPES
                  DO LDT = 1,N_COVERTYPES
                     IF (frac_pft(LDT).eq.0.d0) CYCLE
                        IF ((lai(LDT).ge.LOW_LAI) 
     +                       .and. (fv.gt.0.d0)) THEN
                           GBLEAF=GA*lai(LDT)
                           GAC=GA*(EXP(-lai(LDT))+
     +                       0.01d0*(1.d0-EXP(-lai(LDT))))
                           GGRO=1.d0/(1.d0/GAC+RBSCAL*RBGRO+RGRO)
                           IF (gs_pft(LDT).ne.0.d0) THEN
                               RSTOM=(1.d0/gs_pft(LDT))*DIFF_H2O/DIFF
                               GSTOM=1.d0/(RSTOM+RMESO)
                           ELSE
                               GSTOM=0.d0 
                           ENDIF
                           GCUT=(1.d0/RCUT)*lai(LDT)
                           GLEAF=GSTOM+GCUT
                           FRSTOM=GSTOM/GLEAF
                           GLEAF=1.d0/(RBSCAL/GBLEAF+1.d0/GLEAF)
#ifdef TRACERS_DRYDEP_DIAG
                           FRLEAF=GLEAF/(GLEAF+GGRO)
                           EGSTOM(K,LDT)=FRSTOM*FRLEAF
                           EGCUT_WET(K,LDT)=FRCUTWET*(1.d0-FRCUTSNOW)
     +                          *(1.d0-FRSTOM)*FRLEAF
                           EGCUT_DRY(K,LDT)=(1.d0-FRCUTWET)
     +                          *(1.d0-FRCUTSNOW)
     +                          *(1.d0-FRSTOM)*FRLEAF
                           EGCUT_SNOW(K,LDT)=FRCUTSNOW
     +                          *(1.d0-FRSTOM)*FRLEAF
#endif
                        ELSE
                           GGRO=1.d0/(RBSCAL*RBGRO+RGRO)
                           GLEAF=0.d0
#ifdef TRACERS_DRYDEP_DIAG
                           FRLEAF=0.d0
                           EGSTOM(K,LDT)=0.d0
                           EGCUT_WET(K,LDT)=0.d0
                           EGCUT_DRY(K,LDT)=0.d0
                           EGCUT_SNOW(K,LDT)=0.d0
#endif
                        ENDIF
! Add leaf and ground deposition in parallel, 
! then add in series with above-canopy/surface turbulent transport 
                        VD(K,LDT)=1.d0/(1.d0/GA+1.d0/(GLEAF+GGRO))
#ifdef TRACERS_DRYDEP_DIAG
                        EGSTOM(K,LDT)=EGSTOM(K,LDT)*VD(K,LDT)
                        EGCUT_WET(K,LDT)=EGCUT_WET(K,LDT)*VD(K,LDT) 
                        EGCUT_DRY(K,LDT)=EGCUT_DRY(K,LDT)*VD(K,LDT)
                        EGCUT_SNOW(K,LDT)= EGCUT_SNOW(K,LDT)*VD(K,LDT)
                        EGGRO_WET(K,LDT)=FRGROWET*(1.d0-FRGROSNOW)
     +                          *(1.d0-FRLEAF)*VD(K,LDT)
                        EGGRO_DRY(K,LDT)=(1.d0-FRGROWET)
     +                           *(1.d0-FRGROSNOW)
     +                           *(1.d0-FRLEAF)*VD(K,LDT)
                        EGGRO_SNOW(K,LDT)=FRGROSNOW
     +                           *(1.d0-FRLEAF)*VD(K,LDT)
#endif 
                  ENDDO  
            ELSEIF (tr_wd_TYPE(K) == nPART) THEN ! aerosols 
#ifdef TRACERS_TOMAS
                   IF (K .GE. n_ASO4(1)) THEN
                   binnum = mod(K-n_ASO4(1)+1,NBINS)
                   IF (binnum .EQ. 0) binnum = NBINS
                    GGRAV = vs(binnum)
                    G_DUMMY = 1.d0/RB_TOMAS(binnum)
                   ENDIF
#else
! Calculate conductance due to gravitational settling
#ifndef TRACERS_AMP
! Hydroscopic growth following Ghan and Zaveri, JGR (2007)
              call modal_aero_kohler(TRNRADIUS(K)*1e6,hygro(K),RH,
     &                trrwet,1)
              trrwet=trrwet*1e-6
#else
              trrwet=TRNRADIUS(K)
#endif
              if (trrwet<1.d-20) then ! practically zero size
                VD(K,:)=0.d0
                cycle
              endif

              GGRAV=VGS(RHOSRF,trrwet,TRNDENS(K),VISC)
! Calculate Schmidt number
              DUMMY = SLIPC(RHOSRF,trrwet)
              Dk = kB*TEMPK*DUMMY
     &            /(3.d0*pi*visc_air0*trrwet*2.d0)
              Sc = visc_air_kin0/Dk


              DO LDT = 1,N_COVERTYPES
               IF (frac_pft(LDT).gt.0.d0) THEN
! Calculate aerosol deposition differently for vegetation vs. bare surfaces
                   IF (lai(LDT).GE.LOW_LAI) THEN
                      leafindex = INT(pfpar(LDT)%leaftype)
! Calculate Stokes number
                      St = USTAR*GGRAV/grav/leafdim(leafindex)
! Calculate how much of the particle sticks to the surface
! Where the surface is wet or snow covered the particle sticks 100% (i.e. fstick=1)
! Where the surface is dry the particle sticks according to the Stokes number
                      fstick = EXP(-SQRT(St))*(1.d0-FWETCUT) + FWETCUT
                      fstick = fstick*(1.d0-FSNOWCUT) + FSNOWCUT
! Calculate the efficiency of Brownian diffusion         
                      Eb = Cb*Sc**(-twothirds)
! Calculate the efficiency of impaction 
                      Eim = Cim*(St/(1.d0+St))**beta1
! Calculate the efficiency of interception
                      Ein = Cin*trrwet*2.d0/leafdim(leafindex)
                      IF (leafindex .NE. NEEDLELEAF) THEN
                       Ein = Ein*(2.d0+
     +                     LOG(2.d0*leafdim(leafindex)/trrwet))
                      ENDIF
! Calculate the conductance for dry deposition to vegetation 
                      G_DUMMY = USTAR*
     +                        (Eb+Eim+Ein)*fstick*lai(LDT)
                   ELSE
! Calculate the conductance for dry deposition to bare surfaces
! Only due to brownian motion at this time
                      DUMMY = Sc**onethird/2.9d0
                      Eb = (Sc**(-twothirds))/14.5d0
     +                  *(
     +                  LOG((1.d0+DUMMY)**2/(1.d0-DUMMY+DUMMY**2))/6.d0
     +                  +ATAN((2.d0*DUMMY-1.d0)*byrt3)*byrt3
     +                  +pi/(6.d0*rt3)
     +                  )**(-1)


! Unsure if there should be sticking to ground
! Use Stokes number that emphasizes flow not leaf dimension 
                     St = GGRAV*USTAR**2/grav/visc_air0
                     fstick = EXP(-SQRT(St))*(1.d0-FWETGRO) + FWETGRO
                     fstick = fstick*(1.d0-FSNOWGRO) + FSNOWGRO
                     G_DUMMY = USTAR*Eb*fstick
                   ENDIF
#endif

! Combine the conductance for dry deposition with the conductance for gravitational settling 
! to obtain deposition velocity
                 VD(K,LDT) = GGRAV/(1.d0-EXP(-GGRAV*
     +                             (1.d0/GA+1.d0/G_DUMMY)))


#ifdef DRYDEP_AEROSOLS_OLD
                 VDS = 2.d-3*USTAR
                 IF(OBK < 0.)VDS = VDS*(1.d0+(-3.d2/OBK)**twothirds)
                 IF((ZHH/OBK)<-30.)VDS=9.d-4*USTAR*(-ZHH/OBK)**twothirds
                 IVSMAX=100.d0
                 IF (LDT == 17) IVSMAX=10.d0
                 DUMMY=1.d0/MIN(VDS,1.d-4*REAL(IVSMAX))      
                 VD(K,LDT)=1.d0/MAX(1.d0,MIN(DUMMY,9999.d0))
                 VD(K,LDT) = VD(K,LDT)+GGRAV
#endif
               ENDIF     
              ENDDO      
            ENDIF        
         ENDDO          

! Loop through the different pft present in the cell.
! frac_pft is the fraction of the cell occupied by each aggregate PFT.  
! Add the contribution of the aggregate PFTs 
        DO LDT = 1, N_COVERTYPES
          IF (frac_pft(LDT).gt.0.d0) THEN
#ifdef TRACERS_DRYDEP_DIAG
            IF (LDT.le.8) THEN
               frac_forest=frac_forest+frac_pft(LDT)
            ELSEIF ((LDT.ge.9).and.(LDT.le.10)) THEN
               frac_shrub=frac_shrub+frac_pft(LDT)
            ELSEIF ((LDT.ge.11).and.(LDT.le.14)) THEN
               frac_grass=frac_grass+frac_pft(LDT)
            ELSEIF ((LDT.ge.15).and.(LDT.le.16)) THEN
               frac_crop=frac_crop+frac_pft(LDT)
            ELSEIF (LDT.ge.17) THEN
               frac_bare=frac_bare+frac_pft(LDT)
            ENDIF
#endif
            DO K = 1, NTM
              IF (.not. dodrydep(K)) CYCLE
                VD1(K) = VD1(K) +
     &             frac_pft(LDT)*VD(K,LDT)
#ifdef TRACERS_DRYDEP_DIAG
! Calculate contribution of different PFTs to deposition velocity                
                IF (LDT.le.8) THEN
                 VD_FOREST(K) = VD_FOREST(K) +
     &             frac_pft(LDT)*VD(K,LDT)
                ELSEIF ((LDT.ge.9).and.(LDT.le.10)) THEN
                 VD_SHRUB(K) = VD_SHRUB(K) +
     &             frac_pft(LDT)*VD(K,LDT)
                ELSEIF ((LDT.ge.11).and.(LDT.le.14)) THEN
                 VD_GRASS(K) = VD_GRASS(K) +
     &             frac_pft(LDT)*VD(K,LDT)
                ELSEIF ((LDT.ge.15).and.(LDT.le.16)) THEN
                 VD_CROP(K) = VD_CROP(K) +
     &             frac_pft(LDT)*VD(K,LDT)
                ELSEIF (LDT.ge.17) THEN
                 VD_BARE(K) = VD_BARE(K) +
     &             frac_pft(LDT)*VD(K,LDT)
                ENDIF
                IF(tr_wd_TYPE(K) /= nPART) THEN
                  EGSTOM1(K) = EGSTOM1(K) +
     &               frac_pft(LDT)*EGSTOM(K,LDT)
                  EGCUT_WET1(K) = EGCUT_WET1(K) +
     &               frac_pft(LDT)*EGCUT_WET(K,LDT)
                  EGCUT_DRY1(K) = EGCUT_DRY1(K) +
     &               frac_pft(LDT)*EGCUT_DRY(K,LDT)
                  EGCUT_SNOW1(K) = EGCUT_SNOW1(K) +
     &               frac_pft(LDT)*EGCUT_SNOW(K,LDT)
                  EGGRO_WET1(K) = EGGRO_WET1(K) +
     &               frac_pft(LDT)*EGGRO_WET(K,LDT)
                  EGGRO_DRY1(K) = EGGRO_DRY1(K) +
     &               frac_pft(LDT)*EGGRO_DRY(K,LDT)
                  EGGRO_SNOW1(K) = EGGRO_SNOW1(K) +
     &               frac_pft(LDT)*EGGRO_SNOW(K,LDT)
                  IF (LDT.le.8) THEN
                   EGSTOM_FOREST(K) = EGSTOM_FOREST(K) +
     &              frac_pft(LDT)*EGSTOM(K,LDT)
                  ELSEIF ((LDT.ge.9).and.(LDT.le.10)) THEN
                   EGSTOM_SHRUB(K) = EGSTOM_SHRUB(K) +
     &              frac_pft(LDT)*EGSTOM(K,LDT)
                  ELSEIF ((LDT.ge.11).and.(LDT.le.14)) THEN
                   EGSTOM_GRASS(K) = EGSTOM_GRASS(K) +
     &              frac_pft(LDT)*EGSTOM(K,LDT)
                  ELSEIF ((LDT.ge.15).and.(LDT.le.16)) THEN
                   EGSTOM_CROP(K) = EGSTOM_CROP(K) +
     &              frac_pft(LDT)*EGSTOM(K,LDT)
                  ENDIF ! LDT type 
                ENDIF   ! gas or aerosol 
#endif
            ENDDO       ! tracer loop
          ENDIF         ! frac_pft   
        ENDDO           ! LDT loop   
#ifdef TRACERS_DRYDEP_DIAG
! Calculate average for forests
        IF (frac_forest .gt. 0.d0) THEN
           VD_FOREST = VD_FOREST/frac_forest
           EGSTOM_FOREST = EGSTOM_FOREST/frac_forest
        ENDIF 
! Calculate average for crops
        IF (frac_crop .gt. 0.d0) THEN
           VD_CROP = VD_CROP/frac_crop
           EGSTOM_CROP = EGSTOM_CROP/frac_crop
        ENDIF 
! Calculate average for grass
        IF (frac_grass .gt. 0.d0) THEN
           VD_GRASS = VD_GRASS/frac_grass
           EGSTOM_GRASS = EGSTOM_GRASS/frac_grass
        ENDIF
! Calculate average for shrub
        IF (frac_shrub .gt. 0.d0) THEN
           VD_SHRUB = VD_SHRUB/frac_shrub
           EGSTOM_SHRUB = EGSTOM_SHRUB/frac_shrub
        ENDIF
! Calculate average for bare surfaces (LAND)
        IF (frac_bare .gt. 0.d0) THEN
           VD_BARE = VD_BARE/frac_bare
        ENDIF
#endif
      CASE(ITYPE_OCEAN,ITYPE_LANDICE,ITYPE_OCEANICE) 
        DO K = 1,ntm
          IF (.not. dodrydep(K)) CYCLE
           IF (tr_wd_TYPE(K) /= nPART) THEN 
! Load species-dependent drydep parameters      
              F0=F0_glob(K)
              DIFF= DIFF_H2O * SQRT(mwat/TRNMM(K)) ! Graham's Law
! Distinguish pH for land ice vs. ocean water/ice
              IF (ITYPE == ITYPE_LANDICE) THEN 
                DUMMY_PH = pH_SNOW
              ELSE 
                DUMMY_PH = pH_OCEAN
              ENDIF
! Get effective Henry's Law constant for each gas for water
              CALL GET_HSTAR(trname(K),KH_298_glob(K),
     +             deltaH_R_glob(K),K1_298_glob(K),
     +             deltaH1_R_glob(K),K2_298_glob(K),deltaH2_R_glob(K),
     +             DUMMY_PH,TEMPK,HEFF)
#ifdef TRACERS_SPECIAL_Shindell
! Need to weigh NOx deposition by NO/NO2 because NO2 deposits + NO does not
! The KH_298, F0 and TRNMM read in from ShindellTracersMetadata.F90 is for NO2
! pNOx is the fraction of NOx that is NO2
              IF (trname(K) == 'NOx') THEN
! First need to get effective Henry's Law constant for NO 
                 CALL GET_HSTAR('NO',KH_298_NO,deltaH_R_NO,0.d0,
     +             0.d0,0.d0,0.d0,
     +             DUMMY_PH,TEMPK,HEFF_NO)
                 HEFF=pNOx(1,i,j)*HEFF
     &                + (1.d0 - pNOx(1,i,j))*HEFF_NO
                 F0=pNOx(1,i,j)*F0_glob(K)
     &                + (1.d0 - pNOx(1,i,j))*F0_NO
                 DUMMY=pNOx(1,i,j)*TRNMM_NO2
     &                + (1.d0 - pNOx(1,i,j))*TRNMM_NO
                 DIFF=DIFF_H2O * SQRT(mwat/DUMMY) ! Graham's Law
              ENDIF
#endif
! Calc dimensionless Henry's Law constant by scaling by RT
              HEFF=HEFF*GASC1*TEMPK
              RBSCAL = (KAPPA/DIFF)**twothirds 
              RB_WATER_ICE = 2.d0/(VONK*USTAR)*RBSCAL

             IF (ITYPE == ITYPE_LANDICE) THEN 
                DUMMY_O=IR_SNOW_O
                DUMMY_S=IR_SNOW_S
             ELSEIF (ITYPE == ITYPE_OCEAN) THEN 
                DUMMY_O=IR_WATER_O
                DUMMY_S=IR_WATER_S
             END IF 

! for sea ice, consider how much is melted at the surface
! in future, incorporate this for land ice, but no tracking of water now
             IF (ITYPE == ITYPE_OCEANICE) THEN
                FMP=min(1.6d0*sqrt(si_atm%pond_melt(i,j)/rhow),1.d0)
                G_DUMMY=FMP*(F0/IR_WATER_O+HEFF/IR_WATER_S)
     +               +(1.d0-FMP)*(F0/IR_SNOW_O+HEFF/IR_SNOW_S)
             ELSE
                G_DUMMY=HEFF/DUMMY_S+F0/DUMMY_O
             ENDIF

             VD1(K) = 1.d0/(1.d0/GA 
     &           + RB_WATER_ICE + 1.d0/G_DUMMY)

          ELSE IF (tr_wd_TYPE(K) == nPART) THEN
             Vphor = 1.d-4
! Increase Vphor for ice surfaces
             IF ((ITYPE == ITYPE_LANDICE) .OR. 
     &         (ITYPE == ITYPE_OCEANICE)) Vphor=4.d-4
#ifdef TRACERS_TOMAS
              IF (K .GE. n_ASO4(1)) THEN
                 binnum = mod(K-n_ASO4(1)+1,NBINS)
                 IF (binnum .EQ. 0) binnum = NBINS
                 GGRAV = vs(binnum)
                 G_DUMMY = 1.d0/RB_TOMAS(binnum)
              END IF
#else
! Calculate conductance due to gravitational settling
#ifndef TRACERS_AMP
! Hydroscopic growth following Ghan and Zaveri, JGR (2007)
             call modal_aero_kohler(TRNRADIUS(K)*1e6,hygro(K),RH,
     &                trrwet,1)
             trrwet=trrwet*1e-6 
#else
             trrwet=TRNRADIUS(K)
#endif
             GGRAV=VGS(RHOSRF,trrwet,TRNDENS(K),VISC)
             DUMMY = SLIPC(RHOSRF,trrwet)
             Dk = kB*TEMPK*DUMMY
     &            /(3.d0*pi*visc_air0*trrwet*2.d0)
             Sc = visc_air_kin0/Dk
! Add phoretic effects
             GGRAV = GGRAV+Vphor
! Calculate the conductance for dry deposition to bare surfaces
! Only due to brownian motion at this time
             DUMMY = Sc**onethird/2.9d0
             Eb = (Sc**(-twothirds))/14.5d0
     +                  *(
     +                  LOG((1.d0+DUMMY)**2/(1.d0-DUMMY+DUMMY**2))/6.d0
     +                  +ATAN((2.d0*DUMMY-1.d0)*byrt3)*byrt3
     +                  +pi/(6.d0*rt3)
     +                  )**(-1)

              G_DUMMY = USTAR*Eb

#endif
              VD1(K) = GGRAV/(1.d0-EXP(-GGRAV*
     &                             (1.d0/GA+1.d0/G_DUMMY)))

#ifdef DRYDEP_AEROSOLS_OLD
              VDS = 2.d-3*USTAR
              IF(OBK < 0.)VDS = VDS*(1.d0+(-3.d2/OBK)**twothirds)
              IF((ZHH/OBK)<-30.)VDS=9.d-4*USTAR*(-ZHH/OBK)**twothirds
              IVSMAX=100.d0
              IF (ITYPE == ITYPE_OCEAN) IVSMAX=10.d0
              DUMMY=1.d0/MIN(VDS,1.d-4*REAL(IVSMAX))
              VD1(K)=1.d0/MAX(1.d0,MIN(DUMMY,9999.d0))
              VD1(K) = VD1(K)+GGRAV
#endif
          END IF
        END DO                


      CASE DEFAULT
        call stop_model('ITYPE error in TRDRYDEP',255)
      END SELECT

      RETURN
      END SUBROUTINE get_dep_vel 
#endif
  
!====================================================================================
      subroutine GET_HSTAR(species,KH_298,deltaH_R,K1_298,deltaH1_R,
     +                   K2_298,deltaH2_R,pH, sfc_temp, HEFF )
!========================================================================
! REVISION HISTORY:
! 2008-Nov-12 - F. Vitt - first version
! 2021-Jul-23 - O. Clifton - adapted from CESM for ModelE
! input is effective henry's law constant at 298 K that accounts for hydrolysis
! output is effective henry's law constant at given T and pH (latter accounts for dissociation)
!========================================================================

      implicit none
      character(len=*) :: species
      real*8, intent(in)  :: KH_298,! effectve henry's law coefficient (M/atm) at 298 K
     +                       deltaH_R,K1_298,deltaH1_R,
     +                       K2_298,deltaH2_R
      real*8, intent(in)  :: pH ! pH of the surface
      real*8, intent(in)  :: sfc_temp ! Surface temperature (K)
      real*8, intent(out) :: HEFF  ! effective Henry's law coefficient (M/atm) for a given T and pH

      real*8 :: t0
      real*8 :: Hplus 
      real*8 :: Hplus_inv 
      real*8 :: dk1s           
      real*8 :: dk2s           
      real*8 :: temp_diff
      real*8 :: dummy 


      t0 = 298.d0    ! Standard Temperature (K)
      Hplus = (10.d0)**(-pH)
      Hplus_inv = 1.d0/Hplus
      temp_diff = (t0 - sfc_temp)/(t0*sfc_temp) ! (1/sfc_T - 1/t0)
      dummy = KH_298*exp(deltaH_R*temp_diff)
      dk1s  = K1_298*exp(deltaH1_R*temp_diff)
      if (deltaH2_R == 0.d0 ) then
          HEFF = dummy*(1.d0 + dk1s*Hplus_inv)
      else
          dk2s = K2_298*exp(deltaH2_R*temp_diff)
          !--- For ACIDS ---
          ! (SO2 is only acid with nonzero deltaH2_R)
          if ( species == 'SO2' ) then
               HEFF = dummy*(1.d0 + dk1s*Hplus_inv*
     +              (1.d0 + dk2s*Hplus_inv))
          !--- For BASES ---
          ! (NH3 is only base with nonzero deltaH2_R)
          elseif ( species == 'NH3' ) then
              HEFF = dummy*(1.d0 + dk1s*Hplus/dk2s)
          endif
      endif

      RETURN
      END SUBROUTINE
