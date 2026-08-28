!**** LAKES3.f     2022/03/01
!#!   Replacement code that allows non-zero heat content of atmospheric water
!****
!**** Prognostic lake variables
!**** MWL (kg)  = mass of liquid water in lake
!**** GML (J)   = entrhalpy (heat content) of liquid water in lake
!**** MLDLK (m) = vertical depth of liquid mixed (upper) layer of lake
!**** TLAKE (C) = temperature of mixed layer
!**** T2Lbot(C) = tempreature at bottom of liquid lower layer
!**** RSI (1)   = lake ice cover divided by area of entire lake
!**** MSI(1)  (kg/m^2) = ACE1I + SNOW of lake ice area
!**** MSI(2)  (kg/m^2) = ice mass of second (lower) layer of lake ice
!**** HSI(1:4) (J/m^2) = heat content of 4 thermal layers
!**** FLAKE (1) = areal fraction of grid cell covered by lake
!****
!**** Conversion to variables used in lake subroutines
!**** MLAKE(1) (kg/m^2 of lake fraction) = MLDLK*RHOW
!**** MLAKE(2) (kg/m^2 of lake fraction) = MWL/FLAKE*AXYP - MLAKE(1)
!**** ELAKE(1)  (J/m^2 of lake fraction) = TLAKE*SHW*MLAKE(1) = heat content
!**** ELAKE(2)  (J/m^2 of lake fraction) = GML/FLAKE*AXYP - ELAKE(1)
!**** ES (J/kg) = MLAKE/ELAKE = specific enthalpy
!****
!**** Liquid lakes have either 1 or 2 layers
!**** Mixed layer 1 has a uniform temerature in the vertical
!**** Lower layer 2 temperature fits quadratic polynomial measured from bottom
!**** DLAKE0(m) = mean depth of liquid lake at sill level (from TOPO file)
!**** DLAKE (m) = variable total depth of liquid lake
!**** Z2 (m)    = vetical depth of lower layer 2 = DLAKE - MLDLK
!**** T2mean*Z2 = GWL/FLAKE*AXYP*SHW - TLAKE*MLDLK
!**** T2mean    = mean temperature of lower layer = (GML/FLAKE*AXYP*SHW - TLAKE*MLDLK) / Z2
!**** T2(z)     = A*z^2 + B*z + C = A*z^2 + T2Lbot
!**** T2(0)     = C = T2Lbot = temperature at bottom of layer 2
!**** dT2/dZ(0) = B = 0 = vertical temperature gradient at bottom of layer 2
!**** Int[T2(z)]= A*Z2^3/3 + T2Lbot*Z2 = T2mean*Z2  =>  A = 3*(T2mean - T2Lbot) / Z2^2
!**** T2Ltop    = A*Z2^2 + T2Lbot = temperature at top of layer 2 = 3*T2mean - 2*T2Lbot
!****

#include "rundeck_opts.h"
#ifdef TRACERS_ATM_ONLY
#undef TRACERS_ON
#undef TRACERS_WATER
#endif

      MODULE LAKES
!@sum  LAKES subroutines for Lakes and Rivers
!@auth Gavin Schmidt/Gary Russell
!@ver  2010/08/04 (based on LB265); enhanced in June 2021 as follows:
!@+    River speed is variable, emergency direction outflows are replaced by
!@+    a more stable scheme, the River Direction file is now optional,
!@+    but if one is used, select for the 144x90 grid RVR=RD2HX2E.nc, not RD_Fd.nc.
!@+
!@+   An alternative to lakes being cone shaped or of some other specific
!@+   shape, a statistical relation between area and volume of Earth lakes
!@+   is applied to the lake fraction in each grid box, if it is not too small.
!@+   The area-volume relation is taken from 2016 Cael et al, "The volume
!@+   and mean depths of Earth's lakes": Vol = C * Area^(1+H/2).
!@+   For Earth: C=0.235, Hurst exponent H=0.408. For Mars: H=0.7 .
!@+   That alternative is activated by setting       "Power_Law_Lakes=2".
!@+
!@+   A hybrid option uses the cone scheme where we have observations,
!@+   the Cael scheme where we don't: To get it, use "Power_Law_Lakes=1"

      USE CONSTANT, only : grav,bygrav,shw,rhow,lhm,shi,teeny,undef
#ifdef TRACERS_WATER
      use OldTracer_mod, only: trname
      USE TRACER_COM, only : NTM
#endif
      IMPLICIT NONE
      SAVE
C****
C**** Changes from Model III: MO -> MWL (kg), G0M -> GML (J),
C****                         GZM -> TLAKE (deg C)
!@var KDIREC directions for river flow
C**** 0 no flow, 1-8 anti-clockwise from top RH corner
      INTEGER, ALLOCATABLE, DIMENSION(:,:) :: KDIREC
!@var DHORZ horizontal distance to downstream box (m)
      REAL*8, ALLOCATABLE, DIMENSION(:,:) :: DHORZ
!@var XYZC (x,y,z) unit vectors on sphere of primary cell centers
      REAL*8, ALLOCATABLE, DIMENSION(:,:,:) :: XYZC
!@var IFLOW,JFLOW grid box indexes for downstream direction
      INTEGER, ALLOCATABLE, DIMENSION (:,:) :: IFLOW,JFLOW
!@param MINMLD minimum mixed layer depth in lake (m)
      REAL*8, PARAMETER :: MINMLD = 1.
!@param DLAKE0_MIN minimum sill depth for lake (m)
      REAL*8, PARAMETER :: DLAKE0_MIN = 1.
!@param TMAXRHO temperature of maximum density (pure water) (C)
      REAL*8, PARAMETER :: TMAXRHO = 4.
!@param KVLAKE lake diffusion constant at mixed layer depth (m^2/s)
      REAL*8, PARAMETER :: KVLAKE = 1d-5
!@param TFL freezing temperature for lakes (=0 C)
      REAL*8, PARAMETER :: TFL = 0.
!@param AC1LMIN, AC2LMIN minimum ice thickness for lake ice (kg/m^2)
      REAL*8, PARAMETER :: AC1LMIN = 0.1, AC2LMIN=0.1  ! (not used)
!@param FLEADLK lead fraction for lakes
      REAL*8, PARAMETER :: FLEADLK = 0.
!@param BYZETA reciprocal of solar rad. extinction depth for lake (1/m)
      REAL*8, PARAMETER :: BYZETA = 1./0.35d0
!@dbparam RIVER_FAC (1/s) = river SPEED (m/s) / dZ (m)
      REAL*8 :: river_fac=3.
!@dbparam xEKTlake = fraction of turb. kin. energy EKT available for mixing
      REAL*8 :: xEKTlake = .25d0
!@dbparam yMEKTlake = fraction of turb. kin. energy EKT available for mixing
      REAL*8 :: yMEKTlake = 1/2000d0  !  decay rate (m^2/kg) of EKT effectiveness: EKTeff = EKT * Exp(M/MEKT)
!@dbparam init_flake used to make sure FLAKE is properly initialized
!@+       when using older restart files
!@+       =0 for no change to lakes, =1 for a complete reset to initial values
!@+       =2 removal of any excess water that may have accumulated
      INTEGER :: init_flake=1   ! default = 1
!@dbparam Flake_cutoff - if Flake < Flake_cutoff we set Flake=0
      REAL*8 :: Flake_cutoff = 1d-7
!@dbparam small_lake_evap - if =0, shallow lakes are prevented from evaporating by setting flake=0
      INTEGER :: small_lake_evap = 1
!@dbparam variable_lk 1 if lakes are to be variable
!@+       (temporary variable for development purposes)
      INTEGER :: variable_lk=0    ! default = 0
!@dbparam lake_rise_max amount of lake rise (m) over sill level before
!@+       spillover into next box (only if lake covers >95% of box)
      REAL*8 :: lake_rise_max = 1d2 ! default 100m
!@dbparam  Lake ice exceeding  lake_ice_max (m)  of water equivalent
!@+        is dumped into ice berg arrays
      REAL*8 :: lake_ice_max = 5.
!@dbparam Power_law_lakes: if > 0 Volume=const*Area^1.2... is used
      integer :: Power_law_lakes = 0 ! use 1 or 2 to activate this option
!               if 2, Power Law is applied to all lakes
!               if 1, that law is only applied where flake0=0., i.e.
!                     for new or unknown lakes
! Cael Power Law: Lake_Volume = C_lake * Lake_Area**E_lake
!                 E_lake = 1 + H/2 where H=Hurst coefficient/exponent
      real*8 :: C_lake = 0.235d0 ! Earth values for Power_law_lakes > 0
      real*8 :: E_lake = 1.204d0 ! Earth values for Power_law_lakes > 0
      logical, allocatable :: conical(:,:)

      CONTAINS

      real*8 function TANLK_Cael(cell_m2, DLAKE0)
!      use MathematicalConstants_mod, only : pi
! @auth N.Kiang - ModelE single lake per grid cell scheme.
! Return TANLK given grid cell area.
! For conical lake same area and depth for given size class of lakes in Cael et al. (2017) GRL 44(1)..
! Scales TANLK to have FLAKE0 and DLAKE0 same for all grid cells.
!   Default size class 5 is selected to capture the lake size closest to density of 1 lake per equator grid cell on a 2x2.5 grid.
! TANLK then scales by grid cell area relative to the equator lake.
!        Cael et al. (2017)
!    Cael: V = k * A^(1.2 +- 0.05)
!          A = (V/k)^(1/1.2)
!    cone: V = A * h/3 = A * DLAKE
!          h = 3V/A
!
!    TANLK = R/h = sqrt(A/pi) / ( 3V/A )
!       = (V^0.25) / { 3 * sqrt(pi) * k^1.25 }
!       where k = 0.341 +- 0.005

      real*8, intent(in) :: cell_m2  !Area of grid cell
      real*8 :: FLAKE0 !FLAKE for a SINGLE lake for the grid cell.
      real*8, intent(out) :: DLAKE0 !DLAKE corresponding to mean depth of the lake size class.
      !--- Local -----
      real*8,parameter :: PI = 3.1415926535897932d0 !@param pi    pi
      integer, parameter :: sizeclass=5 !Default lake size class in Cael et al (2017) lookup table.
      real*8, parameter :: A_equator_2x2h_m2=61809004544d0  !Reference value for scaling. Area of equator 2x2.5 grid cell.
      integer, parameter :: AREA = 1
      integer, parameter :: H = 2
      integer, parameter :: DENS = 3
      real*8 :: A_m2   !Area of lake in the grid cell.

      real*8, parameter :: cael_lut(3,7) =  RESHAPE ( (/
      !By Area size class
      !  A.mean.m2, H.mean.m, density.m2 (where density is number per ice-free land area 1.30577E+14 m2;
      !  included here for future versions that incorporate density of lakes by size class)
     1 !*1 - 10^4-10^5 m2
     &  2.88E+04, 2.61, 1.82E-07,
     2 !*2 - 10^5-10^6 m2
     &  2.61E+05, 4.07, 2.92E-08,
     3 !*3 - 10^6-10^7 m2
     &  2.39E+06, 6.41, 2.54E-09,
     4 !*4 - 10^7-10^8 m2
     &  2.51E+07, 10.38, 1.86E-10,
     5 !*5 - 10^8-10^9m2
     &  2.51E+08, 16.58, 1.49E-11,
     6 !6 - 10^9-10^10 m2
     &  2.55E+09, 26.44, 1.62E-12,
     7 !7 - > 10^10 m2
     &  5.10E+10, 156.86, 1.53E-13
     *  /), (/ 3, 7/ ) )

      FLAKE0 = cael_lut(AREA,sizeclass)/A_equator_2x2h_m2
      !FLAKE0 = signif(cael_lut["A.mean.m2",sizeclass]/A.equator.2x2h.m2, 3)

      !Approach 1: Cael size class lookup table.
      DLAKE0 = cael_lut(H,sizeclass)

      !Approach 2: Cael equation with awkward exponent.
      !k = 0.341d0
      !V0 = k * A_mean_m2_class5^1.2d0
      !DLAKE0 = V0/A_mean_m2_class5 = k * A_mean_m2_class5^0.2d0

      A_m2 = FLAKE0 * cell_m2
      !V_m3 = DLAKE0 * A_m2

      !Returns FLAKE0, constant same for all grid cells.
      TANLK_Cael = sqrt(A_m2/pi)/(3*DLAKE0)

      end function TANLK_Cael

      EndModule LAKES



      SUBROUTINE LKSOURC (I0,J0,ROICE,MLAKE,ELAKE,RUN0,FODT,FIDT,SROX
     *                   ,FSR2,T2Lbot,EKT,LAKEFR,
#ifdef TRACERS_WATER
     *     TRLAKEL,TRUN0,TREVAP,TRO,TRI,
#endif
     *     EVAPO,ENRGFO,ACEFO,ACEFI,ENRGFI)
!@sum  LKSOURC applies fluxes to lake in ice-covered and ice-free areas
!@auth Gary Russell/Gavin Schmidt
      Use LAKES,     Only: RHOW, SHW,SHI, LHM, TFL, minMLD, xEKTlake
      Use MODEL_COM, Only: ModelEclock, dTSrc,ITime, IWrite_Sv,JWrite_Sv
      USE DOMAIN_DECOMP_ATM, only : WRITE_PARALLEL
      USE DIAG_COM, Only: aij=>aij_loc, ij_tlake1, ij_tlake2bot,
     &  ij_tlake2mean, ij_tlake2top
#ifdef TRACERS_WATER
      Use TRACER_COM, Only: NTM
#endif
      IMPLICIT NONE

      INTEGER, INTENT(IN) :: I0,J0
      Real*8,Intent(InOut) ::
     *   MLAKE(2),  !  mass per unit area (kg/m^2 of lake)
     *   ELAKE(2),  !  enthalpy per unit area (kg/m^2 of lake)
     *   T2Lbot     !  temperature at bottom of lake layer 2 (C)
      Real*8,Intent(In) ::
     *   ROICE,  !  areal fraction of lake covered by ice
     *   EVAPO,  !  evaporation (kg/m^2 of open lake)
     *   RUN0,   !  runoff from lake ice (kg/m^2 of lake ice)
     *   FODT,   !  surface heating (J/m^2 of open lake)
     *   FIDT,   !  heat from lake ice (J/m^2 of lake ice)
     *   EKT,    !  turbulent kinetic energy from surface (J/m^2)
     *   FSR2,   !  fraction of sun light penetrates to layer 2
     *   SROX(2),!  sun light (J/m^2 of open lake, of lake ice)
     *   LAKEFR  !  lake fraction (used for diagnostic)      
      Real*8,Intent(Out) ::
     *   ENRGFO,  !  energy of new ice < 0 (J/m^2 of open lake)
     *   ENRGFI,  !  energy of new ice < 0 (J/m^2 of lake ice)
     *   ACEFO,   !  mass of new ice > 0 (kg/m^2 of open lake)
     *   ACEFI    !  mass of new ice > 0 (kg/m^2 of lake ice)

#ifdef TRACERS_WATER
      REAL*8, INTENT(INOUT), DIMENSION(NTM,2) :: TRLAKEL
      REAL*8, INTENT(IN), DIMENSION(NTM) :: TRUN0,TREVAP
      REAL*8, INTENT(OUT), DIMENSION(NTM) :: TRO,TRI
      REAL*8, DIMENSION(NTM) :: dTR1O,dTR1I, TRUNO,TRUNI, FRAC
#ifdef TRACERS_SPECIAL_O18
      REAL*8 fracls
      INTEGER N
#endif
#endif

!**** Local variables
      Real*8, PARAMETER ::
     *   Emin = -1d-10, !  minimum energy deficit required before ice forms (J/m^2)
     *   ESmin = 4*SHW, !  specific enthalpy (J/kg) at minimum specific volume occurs at 4 (C)
     *   KDIFF  = 1d-6  !  thermal diffusivity (m^2/s) of lake water
      Real*8 ::
     *   ES2mean,    !  specific enthalpy of mean layer 2 = ELAKE(2) / MLAKE(2) (J/kg)
     *   ES2bot,     !  specific enthalpy at bottom of layer 2 = T2Lbot*SHW (J/kg)
     *   dM1O,dM1I,  !  lake water moved from layer 1 into layer 2 (kg/m^2), may be + or -
     *   dE1O,dE1I   !  liquid heat content moved from layer 1 into layer 2 (J/m^2), may be + or -
      Real*8 :: ENRGO,ENRGO2,RUNO, MLAKEO(2),ELAKEO(2),ES2botO,
     *          ENRGI,ENRGI2,RUNI, MLAKEI(2),ELAKEI(2),ES2botI,
     *          ES2top,EShalf,M,Q,dH,dES
!@var out_line local variable to hold mixed-type output for parallel I/O
      character(len=300) :: out_line
      Integer :: YEAR,MONTH,DATE,HOUR
      Real*8  :: OUT(0:9), T1L, T2Lmean,T2Ltop,T2LbotX ! T2LbotX = duplicate for diag
      Character*6   :: NAME
      Character*90  :: TEXT
      Character*256 :: FILEOUT

      Call ModelEclock%get (year=YEAR, month=MONTH, date=DATE,hour=HOUR)
!****
!**** Write LinePlot file of lake parameters
!****
      NAME = '    '
!     If (I0== 40 .and. J0==66)  NAME = 'S.Erie'
!     If (I0== 40 .and. J0==67)  NAME = 'N.Erie'
!     If (I0== 38 .and. J0==68)  NAME = 'N.Mich'
!     If (I0== 38 .and. J0==69)  NAME = 'W.Supr'
!     If (I0==103 .and. J0==73)  NAME = 'ObBasn'
!     If (I0== 93 .and. J0==68)  NAME = 'CaspME'
!     If (I0== 94 .and. J0==68)  NAME = 'KazakW'
!     If (I0==106 .and. J0==69)  NAME = 'KazakE'
      If (I0==IWrite_Sv .and. J0==JWrite_Sv)  NAME = 'WriteD'
      If (NAME .ne. '    ') Then
         OUT(0) = DATE-1 + HOUR/24d0 +
     *            Modulo (ITIME*dTSrc+1d-5, 3600d0) / 86400d0
         OUT(1) = ROICE * 100
         OUT(2) = xEKTlake * EKT / dTSrc
         OUT(3) = FODT / dTSrc
         OUT(4) = FIDT / dTSrc
         OUT(5) = MLAKE(1) * .001
         OUT(6) = ELAKE(1) / (SHW * MLAKE(1))
         If (MLAKE(2) > 0)
     *      Then  ;  T2Lmean = ELAKE(2) / (SHW * MLAKE(2))
                     T2Ltop  = 3*T2Lmean - 2*T2Lbot
                     OUT(7)  = T2Ltop
                     OUT(8)  = T2Lmean
                     OUT(9)  = T2Lbot
            Else  ;  OUT(7:9) = OUT(6)  ;  EndIf
         Write (FILEOUT,900) NAME,YEAR,MONTH
         Open  (I0*1000+J0*10+7, File=FILEOUT, Position='Append')
         If (OUT(0) < 1d-4) Then
            TEXT = '  Day.Hr  ROICE(%) EKT(W/m2)  Eopen      Eice' //
     *             '   MLD(m)    T1(C)   T2Ltop   T2mean   T2Lbot'
            Write (I0*1000+J0*10+7,901) TEXT  ;  EndIf
         Write (I0*1000+J0*10+7,902) OUT  ;  EndIf
  900 Format (A6,I4,I2.2,'.TXT')
  901 Format (A)
  902 Format (3F9.3,F9.2,6F9.3)

      ! out(6) = tlake1, out(7:9) = tlake2 top/mean/bot
      t1l = ELAKE(1) / (SHW * MLAKE(1))
      IF (MLAKE(2) > 0) then
        T2Lmean = ELAKE(2) / (SHW * MLAKE(2))
        T2Ltop  = 3*T2Lmean - 2*T2Lbot
        T2LbotX = T2Lbot
      ELSE
        T2Lmean = T1L
        T2Ltop = T1L
        T2LbotX = T1L
      ENDIF

      aij(i0,j0,ij_tlake1) = aij(i0,j0,ij_tlake1) +
     &  t1l * LAKEFR
      aij(i0,j0,ij_tlake2top) = aij(i0,j0,ij_tlake2top) +
     &  T2Ltop * LAKEFR
      aij(i0,j0,ij_tlake2mean) = aij(i0,j0,ij_tlake2mean) +
     &  T2Lmean * LAKEFR
      aij(i0,j0,ij_tlake2bot) = aij(i0,j0,ij_tlake2bot) +
     &  T2LbotX * LAKEFR

C**** initialize output
      ENRGFO=0. ; ACEFO=0. ; ACEFI=0. ; ENRGFI=0.

      ES2bot = T2Lbot * SHW
C**** Calculate heat and mass fluxes to lake
      ENRGO = FODT-SROX(1)*FSR2 ! in open water
      ENRGO2=     +SROX(1)*FSR2 ! in open water, second layer
      ENRGI = FIDT-SROX(2)*FSR2 ! under ice
      ENRGI2=     +SROX(2)*FSR2 ! under ice, second layer
      RUNO  =-EVAPO
      RUNI  = RUN0

#ifdef TRACERS_WATER
      TRUNO(:)=-TREVAP(:)
      TRUNI(:)= TRUN0(:)
      FRAC(:)=1.
#ifdef TRACERS_SPECIAL_O18
      do n=1,ntm
        FRAC(n)=fracls(n) ! fractionation when freezing
      end do
#endif
#endif

!****
!**** Compute water exchange between liquid lake layers under open lake
!****
      dM1O = 0  ;  dE1O = 0  ;  ES2botO = 0
      If (ROICE < 1) Then
         Call LAKMIX (MLAKE,ELAKE,ES2bot,RUNO,ENRGO,ENRGO2,xEKTlake*EKT,
     *                dM1O,dE1O)
         MLAKEO(1) = MLAKE(1) - dM1O + RUNO
         ELAKEO(1) = ELAKE(1) - dE1O + ENRGO
         MLAKEO(2) = MLAKE(2) + dM1O
         ELAKEO(2) = ELAKE(2) + dE1O + ENRGO2
         If (MLAKEO(2) <= 0)
     *      Then  ;  ELAKEO(2) = 0
                     ES2botO = ELAKEO(1) / MLAKEO(1)
            Else  ;  ES2botO = ES2bot  ;  EndIf
         If (ELAKEO(1) < Emin) Then
            ACEFO  = ELAKEO(1) / (TFL*(SHI-SHW)-LHM)
            ACEFO  = Min (ACEFO, Max(MLAKEO(1)-.75*minMLD*RHOW, 0d0))
            ENRGFO = ACEFO*(TFL*SHI-LHM)  ;  EndIf

#ifdef TRACERS_WATER
         If (dM1O < 0)
     *      Then  ;  dTR1O(:) = dM1O * TRLAKEL(:,2) / MLAKE(2)
            Else  ;  dTR1O(:) = dM1O * TRLAKEL(:,1) / MLAKE(1)  ; EndIf
         TRO(:) = ACEFO * FRAC(:) * TRLAKEL(:,1) / MLAKEO(1)
#endif
         EndIf  !  If (ROICE < 1) Then

!****
!**** Compute water exchange between liquid lake layers under beneath lake ice
!****
      dM1I = 0  ;  dE1I = 0  ;  ES2botI = 0
      If (ROICE > 0) Then
         Call LAKMIX (MLAKE,ELAKE,ES2bot,RUNI,ENRGI,ENRGI2,0d0,
     *                dM1I,dE1I)
         MLAKEI(1) = MLAKE(1) - dM1I + RUNI
         ELAKEI(1) = ELAKE(1) - dE1I + ENRGI
         MLAKEI(2) = MLAKE(2) + dM1I
         ELAKEI(2) = ELAKE(2) + dE1I + ENRGI2
         If (MLAKEI(2) <= 0)
     *      Then  ;  ELAKEI(2) = 0
                     ES2botI = ELAKEI(1) / MLAKEI(1)
            Else  ;  ES2botI = ES2bot  ;  EndIf
         If (ELAKEI(1) < Emin) Then
            ACEFI  = ELAKEI(1) / (TFL*(SHI-SHW)-LHM)
            ACEFI  = Min (ACEFI, Max(MLAKEI(1)-.75*minMLD*RHOW, 0d0))
            ENRGFI = ACEFI*(TFL*SHI-LHM)  ;  EndIf

#ifdef TRACERS_WATER
         If (dM1I < 0)
     *      Then  ;  dTR1I(:) = dM1I * TRLAKEL(:,2) / MLAKE(2)
            Else  ;  dTR1I(:) = dM1I * TRLAKEL(:,1) / MLAKE(1)  ; EndIf
         TRI(:) = ACEFI * FRAC(:) * TRLAKEL(:,1) / MLAKEI(1)
#endif
         EndIf  !  If (ROICE > 0) Then

!****
!**** Update prognostic liquid lake variables
!****
      MLAKE(1) = MLAKE(1) + (1-ROICE) * (RUNO   - dM1O - ACEFO)
     +                    +    ROICE  * (RUNI   - dM1I - ACEFI)
      ELAKE(1) = ELAKE(1) + (1-ROICE) * (ENRGO  - dE1O - ENRGFO)
     +                    +    ROICE  * (ENRGI  - dE1I - ENRGFI)
      MLAKE(2) = MLAKE(2) + (1-ROICE) *           dM1O
     +                    +    ROICE  *           dM1I
      ELAKE(2) = ELAKE(2) + (1-ROICE) * (ENRGO2 + dE1O)
     +                    +    ROICE  * (ENRGI2 + dE1I)
      ES2bot = (1-ROICE)*ES2botO + ROICE*ES2botI

!**** Update TRACER prognostic liquid lake variables
#ifdef TRACERS_WATER
      TRLAKEL(:,1) = TRLAKEL(:,1)
     +             + (1-ROICE) * (TRUNO(:) - dTR1O(:) - TRO(:))
     +             +    ROICE  * (TRUNI(:) - dTR1I(:) - TRI(:))
      TRLAKEL(:,2) = TRLAKEL(:,2)
     +             + (1-ROICE) *             dTR1O(:)
     +             +    ROICE  *             dTR1I(:)
      If (MLAKE(2) <= 0)  TRLAKEL(:,2) = 0
#endif

!****
!**** Diffusion of heat occurs between the the new mixed layer and the top of new layer 2,
!**** and diffusion of heat between the middle of new layer 2 and ES2bot; both use an implicit computation
!**** EScen (J/kg) = (ES1old + ES2old) / 2 = (ES1new + ES2new) / 2 = constant specific enthalpy at center
!**** M (kg/m^2) = same mass per unit area thicknesses for ES1 and ES2
!**** dZ (m) = M / RHOW
!**** Q  (1) = dT (s) * KDIFF (m^2/s) / dZ^2 (m^2) = fixed factors for computation
!**** dH (J/m^2) = (ES1new - ES2new) * M * Q = 2*(ES1new - EScen) * M * Q = implicit heat transported from 1 to 2
!**** M*ES1new = M*ES1old - dH = M*ES1old - 2*(ES1new - EScen)*M*Q  =>  ES1new = ES1old - 2*(ES1new - EScen)*Q  =>
!**** =>  ES1new + ES1new*2*Q = ES1old + EScen*2*Q + EScen - EScen  =>  (ES1new - EScen) * (1 + 2*Q) = ES1old - EScen  =>
!**** =>  ES1new - EScen = (ES1old - EScen) / (1 + 2Q) = [(ES1old - ES2old)/2] / (1 + 2*Q)
!**** dH = 2*(ES1new - EScen)*M*Q = (ES1old - ES2old)*M*Q / (1 + 2*Q)
!****
      If (MLAKE(2) > 0) Then
         ES2top = 3*ELAKE(2)/MLAKE(2) - 2*ES2bot
         M = Min (.5*MLAKE(1), .5*MLAKE(2))
         Q = dTSrc * KDIFF * (RHOW/M)**2
         dH = (ELAKE(1)/MLAKE(1) - ES2top) * M*Q / (1 + 2*Q)
         ELAKE(1) = ELAKE(1) - dH
         ELAKE(2) = ELAKE(2) + dH
!****
         EShalf = .75*ELAKE(2)/MLAKE(2) + .25*ES2bot  !  EShalf = A * (.5*MLAKE2)^2 + ES2bot
         M = .5*MLAKE(2)
         Q = dTSrc * KDIFF * (RHOW/M)**2
         dES = (EShalf - ES2bot) * Q / (1 + 2*Q)  !  dES = dH/M = [(EShalf - ES2bot) * M*Q / (1 + 2*Q)] / M
         ES2bot = ES2bot + dES

!**** If specific volume of mean lake layer 2 is less (more dense) than that of lake bottom, mix layer 2 temperature to be uniform
!**** Specific volume, VS (m^3/kg) = VSmin + Hd2VSdES2*(ES-ESmin)^2, is a quadratic polynomial of ES
!**** VSmin + Hd2VSdES2*(ES2mean-ESmin)^2 - [VSmin + Hd2VSdES2*(ES2bot-ESmin)^2]  =>  (ES2mean-ESmin)^2 - (ES2bot-ESmin)^2
          ES2mean = ELAKE(2) / MLAKE(2)
          If ((ES2mean-ESmin)**2 < (ES2bot-ESmin)**2)  ES2bot = ES2mean  !  specific enthalpy of layer 2 everywhere = ES2mean
      EndIf

      T2Lbot = ES2bot / SHW
      END SUBROUTINE LKSOURC



      Subroutine LAKMIX (MLAKE,ELAKE,ES2bot,dM0,dE0,dE1in,EKT, dM1,dE1)
!****
!**** LAKMIX modifies the lake vertical structure; it is called under open lake with EKT, and under lake ice with EKT = 0
!**** Surface water and heat fluxes from SURFACE assisted by EKTeff compare geopotential energy (EG) of the water column.
!**** The mixed layer increases if new EG exceeds old EG, otherwise the mixed layer decreases.
!**** EG variations are 4+ orders of magnitude smaller than enthalpy.  EKT is not added to enthalpy in lakes coding.
!****
!**** Input: MLAKE (kg/m^2) = lake mass of layers 1 and 2
!****        ELAKE  (J/m^2) = heat content of layers 1 and 2
!****        ES2bot (J/kg)  = specific energy at lake bottom
!****        dM0   (kg/m^2) = dew falling onto open lake, negative for evaporation
!****        dE0    (J/m^2) = downward surface fluxes onto open lake excluding sun light that penetrates into layer 2
!****        dE1in  (J/m^2) = sun light that penetrates into layer 2
!****        EKT    (J/m^2) = turbulent kinetic energy absorbed by lake; effective EKT*Exp(-M/MEKT)
!**** Output: dM1  (kg/m^2) = lake mass trasported from layer 1 to layer 2, may be + or -
!****         dE1   (J/m^2) = lake heat content transported from layer 1 to layer 2
!****
!**** Specific enthalpy, ES (J/kg) = TC*SHCW, of the second layer is a quadratic polynomial in the vertical,
!**** measured from the lake bottom.
!**** ES(m)      = A*m^2 + B*m + C
!**** ES(0)      = C = ES2bot = specific enthalpy at lake botton
!**** dES/dm(0)  = B = 0     = vertical gradient of ES at lake botton
!**** ES(MLAKE2) = A*MLAKE2^2 + ES2bot = ES2top = ES at top of Layer 2
!**** Int[ES(m)] = A*MLAKE2^3/3 + ES2bot*MLAKE2 = ELAKE2 =>
!****              A = 3*(ELAKE2 - MLAKE2*ES2bot) / MLAKE2^3
!**** ES2top     = A*MLAKE2^2 + ES2bot = 3*ELAKE2/MLAKE2 - 2*ES2bot
!****
!**** Specific volume, VS (m^3/kg) = VSmin + Hd2VSdES2*(ES-ESmin)^2, is a quadratic polynomial of ES
!**** dVS/dES = Hd2VSdES2 * 2 (ES-ESmin)  =>  .5 d2VS/dES2 = Hd2VSdES2
!**** Minimum value, VSmin, occurs when ES = ESmin, approximately 4 C
!**** VSof0 = VSmin + Hd2VSdES2 * (0 - ESmin)^2  =>  Hd2VSdES2 = (VSof0 - VSmin) / ESmin^2
!****
      Use CONSTANT, Only: GRAV,RHOW,SHW
      Use LAKES,    Only: minMLD, yMEKTlake != decay rate (m^2/kg) of EKT effectiveness
      Implicit  None
      Real*8,Parameter ::
     *   dM1inc = 1000,  !  maximum positive increase in MLD
     *   dM1dec = .75,   !  dM1dec*M1/RHOW is minimum value of new MLD when MLD is decreasing
     *   ESmin = 4*SHW,  !  specific enthalpy (J/kg) at minimum specific volume occurs at 4 (C)
     *   VSmin = 1d-3,   !  minimum specific volume (m^3/kg) occurs at 4 (C)
     *   VSof0 = 1 / 999.8425494d0, !  specific volume (m^3/kg) of lake water at 0 (C)
     *   M1min = minMLD*RHOW,       !  minimum thickness for liquid lake layer 1 (kg/m^2)
     *   Hd2VSdES2 = (VSof0 - VSmin) / ESmin**2  !  .5 times second derivative of VS with respect to ES (kg m^3/J^2)
      Real*8,Intent(In)  :: MLAKE(2),ELAKE(2),ES2bot,dM0,dE0,dE1in,EKT
      Real*8,Intent(Out) :: dM1,dE1
!**** Local variables
      Real*8 :: Ay3, M1,MU,MD, EGXmEGUD,ESD,ESUxMU,
     *          ES1old,ES2topold, ES1new,ES2topnew,
     *          EKTeff  !  effective EKT uses expotential decay to raise heavy water up

      dM1 = 0  ;  dE1 = 0
      M1 = dM0 + MLAKE(1)
      If (M1 + MLAKE(2) > M1min)  GoTo 200

!****
!**** Lake has insufficient water: lake will have a single layer
!****
  100 dM1 = - MLAKE(2)
      dE1 = - ELAKE(2) - dE1in
      Return

!****
!**** Total lake mass exceeds M1min
!****
  200 If (M1 <= M1min)  GoTo 310
      If (MLAKE(2) > 0) Then
         ES1old = (dE0 + ELAKE(1)) / M1
         ES2topold = 3*(dE1in + ELAKE(2)) / MLAKE(2) - 2*ES2bot
!**** If present layer 1 with dE0 is heavier than top of layer 2 with dE1in, then new mixed layer will be deeper and jump to 310
         If (ESmin <= ES1old .and. ES1old <= ES2topold .or.
     *       ESmin >= ES1old .and. ES1old >= ES2topold)  GoTo 310
             EndIf

!**** Compute sign of change in specific enthalpy (temperature) by adding mass dM0 and heat dE0 to lake layer 1
!**** (dE0+ELAKE1)/(dM0+MLAKE1) - ELAKE1/MLAKE1  =>  (dE0+ELAKE1)*MLAKE1 - ELAKE1*(dM0+MLAKE1) = dE0*MLAKE1 - ELAKE1*dM0
!**** If sign is - and ELAKE1/MLAKE1 > ESmin or sign is + and ELAKE1/MLAKE1 < ESmin, then dM0,dE0 is heavier than present layer 1
      If ((dE0*MLAKE(1)-ELAKE(1)*dM0) * (ELAKE(1)-MLAKE(1)*ESmin) <= 0)
     *   GoTo 300  !  let new heavy water sink

!**** dM0 and dE0 added to surface lake water is lighter, compute new mixed layer that may be shallower
!**** Compute geopotential energy EGX of newly mixed layer of mass M1, measured from bottom of layer M1
!****     and geopotential energy EGU+EGD of upper layer MU receiving dE0 and lower layer MD of ES = ELAKE1/MLAKE1
!**** Compute EGX - EGU-EGD > 0.  If EGXmEGUD > EKTeff, then new mixed layer is inside old layer 1
!****
!**** MU  = Max (dM1dec*M1, M1min)
!**** ESU = [dE0 + (MU-dM0)*ELAKE1/MLAKE1] / MU = [dE0 + heat content of MU-dM0 of original layer 1] / MU
!**** VSU = VSmin + Hd2VSdES2 * (ESU - ESmin)^2 = VSmin + Hd2VSdES2 * dESU^2
!**** ZU  = VSU * MU = (VSmin + Hd2VSdES2 * dESU^2) * MU
!**** EGU = GRAV * (ZD + .5*ZU) * MU = GRAV * [(VSmin + Hd2VSdES2*dESD^2)*MD + .5*(VSmin + Hd2VSdES2*dESU^2)*MU] * MU =
!****     = GRAV * VSmin * (MD + .5*MU) * MU + GRAV * Hd2VSdES2 * (dESD^2 * MD + .5*dESU^2 * MU) * MU
!****
!**** MD  = dM0+MLAKE1 - MU = M1 - MU
!**** ESD = ELAKE1/MLAKE1
!**** VSD = VSmin + Hd2VSdES2 * (ESD - ESmin)^2 = VSmin + Hd2VSdES2 * dESD^2
!**** ZD  = VSD * MD = (VSmin + Hd2VSdES2 * dESD^2) * MD
!**** EGD = GRAV * .5*ZD * MD = .5*GRAV * (VSmin + Hd2VSdES2 * dESD^2) * MD^2 =
!****     = .5*GRAV * VSmin * MD^2 + .5*GRAV * Hd2VSdES2 * dESD^2 * MD^2
!****
!**** Note: ESD*MD + ESU*MU = (dM0+MLAKE1-MU)*ELAKE1/MLAKE1 + dE0 + (MU-dM0)*ELAKE1/MLAKE1 = ELAKE1 + dH0
!****
!**** MX  = dM0+MLAKE1 = M1 = MD + MU
!**** ESX = (dE0 + ELAKE1) / MX = (ESD*MD + ESU*MU) / (MD + MU)
!**** VSX = VSmin + Hd2VSdES2 * (ESX - ESmin)^2 = VSmin + Hd2VSdES2 * [(dESD*MD + dESU*MU) / (MD + MU)]^2
!**** ZX  = VSX * MX = [VSmin + Hd2VSdES2 * (dESD*MD + dESU*MU)^2 / (MD + MU)^2] * (MD + MU)
!**** EGX = GRAV * .5*ZX * MX = .5*GRAV * VSX * MX^2 =
!****     = .5*GRAV * {VSmin + Hd2VSdES2 * [(dESD*MD + dESU*MU)^2 / (MD + MU)]^2} * (MD + MU)^2 =
!****     = .5*GRAV * VSmin * (MD+MU)^2 + .5*GRAV * Hd2VSdES2 * (dESD*MD + dESU*MU)^2
!****
!**** EGXmEGUD = EGX - EGD-EGU > 0
!****     = .5*GRAV * VSmin * (MD+MU)^2 + .5*GRAV * Hd2VSdES2 * (dESD*MD + dESU*MU)^2 -
!****       - .5*GRAV * VSmin * MD^2 + .5*GRAV * Hd2VSdES2 * dESD^2 * MD^2 -
!****       - GRAV * VSmin * (MD + .5*MU) * MU + GRAV * Hd2VSdES2 * (dESD^2 * MD + .5*dESU^2 * MU) * MU =
!****     = .5*GRAV * VSmin * [(MD+MU)^2 - MD^2 - (2*MD + MU)*MU] +
!****       + .5*GRAV * HdVSdES2 * [(dESD*MD)^2 + 2*dESD*MD*dESU*MU + (dESU*MU)^2 -
!****                               - dESD^2 * MD^2 - 2*dESD^2 * MD*MU - dESU^2 * MU^2] =
!****     = .5*GRAV * HdVSdES2 * [2*dESD*MD*dESU*MU - 2*dESD^2 * MD*MU] =
!****     = GRAV * HdVSdES2 * dESD * (dESU - dESD) * MD*MU =
!****     = GRAV * HdVSdES2 * (ESD-ESmin) * (ESU*MU - ESD*MU) * MD
      MU       = Max (dM1dec*M1, M1min)
      MD       = M1 - MU
      ESD      = ELAKE(1) / MLAKE(1)
      ESUxMU   = dE0 + (MU-dM0) * ESD  !  = {[dE0 + (MU-dM0)*ELAKE1/MLAKE1] / MU} * MU
      EGXmEGUD = GRAV * Hd2VSdES2 * (ESD-ESmin) * (ESUxMU - ESD*MU) * MD

      EKTeff = EKT * Exp(- (MU + .25*MD) * yMEKTlake)
      If (EKTeff > EGXmEGUD)  GoTo 300
      If (EGXmEGUD == 0)  Return

!**** Current EKTeff is insufficient to mix through the former layer 1 with current heating: estimate new shallower layer 1
!**** 0 < MDnew < MD: new layer 1 will be inside old layer 1: MDnew = MD * EKTeff/EGXmEGUD
      dM1 = MD - MD * EKTeff / EGXmEGUD  !  > 0
      dE1 = dM1 * ESD
      If (MLAKE(2) <= 0)  Return

!**** Check whether new mixed mixed layer is heavier than new ES2top; if so, reduce dM1,dE1
      ES1new = (dE0+ELAKE(1) - dE1) / (M1 - dM1)
      ES2topnew = 3*(dE1in+ELAKE(2) + dE1) / (MLAKE(2)+dM1) - 2*ES2bot
      If ((ESmin < ES1new .and. ES1new < ES2topnew .or.
     *     ESmin > ES1new .and. ES1new > ES2topnew) .and.
     *    (ES1old - ES2topold) * (ES1new - ES2topnew) < 0) Then
!**** ES1old is lighter than ES2topold with 0 change to mixed layer depth
!**** ES2new is heavier than ES2topnew with dM1 decrease of mixed layer depth
!**** Choose intermediate value of dM1: dM1new = dM1 * (ES1old - ES2topold) / (ES1old - ES2topold - ES1new + ES2topnew)
         dM1 = dM1 * (ES1old - ES2topold) /
     /               (ES1old - ES2topold - ES1new + ES2topnew)
         dE1 = dM1 * ESD  ;  EndIf
      Return

!**** Current EKTeff is more than sufficient to mix through the former layer 1 with current heating
  300 If (MLAKE(2) <= 0)  Return

!**** Mix layer 2 water into layer 1: dM1 < 0
!**** Either surface water is heavier, EKT is enough to mix through former layer 1, or M1 = dM0+MLAKE1 < M1min
!**** Compute geopotential energy measured from M1-dM1 twice:
!**** EGX  = geopotential energy when new mixed layer is M1-dM1
!**** EGUD = geopotential energy of upper layer of M1 from -dM1 to -dM1+M1 with heat dE0 and lower from 0 to -dM1 from quadratic fit
!**** Compute EGX - EGU-EGD.  If EGXmEGUD > EKTeff, then new mixed layer is between M1 and M1-dM1
!**** ES(m) = A*m^2 + ES2bot
!**** E2tot = dE1in + ELAKE2 = Int (A*m^2 + ES2bot) dm : from 0 to MLAKE2  =  A*MLAKE2^3/3 + ES2bot*MLAKE2
!**** A   = 3 * (E2tot - ES2bot*MLAKE2) / MLAKE2^3 = 3 * (dE1in+ELAKE2 - ES2bot*MLAKE2) / MLAKE2^3
!**** dM1 = - Min [dM1inc, MLAKE(2)] < 0
!**** dE1 = - Int (A*m^2 + ES2bot) dm : from MLAKE2+dM1 to MLAKE2 =
!****     = - {A*[MLAKE2^3 - (MLAKE2+dM1)^3] / 3 + ES2bot*[MLAKE2 - (MLAKE2+dM1)]} =
!****     = - {A*(- 3*MLAKE2^2*dM1 - 3*MLAKE2*dM1^2 - dM1^3) / 3 - ES2bot*dM1} =
!****     = [(A/3) * (3*MLAKE2^2 + 3*MLAKE2*dM1 + dM1^2) + ES2bot] * dM1 = ESD * dM1 < 0
!**** MU  = M1
!**** ESU = (dE0 + ELAKE1) / MU
!**** MD  = - dM1 > 0
!**** ESD = dE1 / dM1 = (Ay3) * (3*MLAKE2^2 + 3*MLAKE2*dM1 + dM1^2) + ES2bot > 0
!**** MX  = M1 - dM1
!**** ESX = (dE0 + ELAKE1 - dE1) / MX
!**** EGXMEGUD = EGX - EGU-EGD = GRAV * HdVSdES2 * (ESD-ESmin) * (ESU*MU - ESD*MU) * MD
  310 Ay3 = (dE1in+ELAKE(2) - ES2bot*MLAKE(2)) / MLAKE(2)**3
      dM1 = - Min (dM1inc, MLAKE(2))
      ESD = Ay3 * (3*MLAKE(2)**2 + 3*MLAKE(2)*dM1 + dM1**2) + ES2bot
      dE1 = ESD * dM1
      ESUxMU = dE0 + ELAKE(1)
      EGXmEGUD = GRAV*Hd2VSdES2*(ESD-ESmin)*(ESUxMU-ESD*M1) * (- dM1)

      EKTeff = EKT * Exp(- (M1 - .25*dM1) * yMEKTlake)
      If (EKTeff >= EGXmEGUD)  GoTo 400

!**** Current EKTeff is insufficient to mix through the former layer 1 and -dM1 of layer 2: estimate new layer 1
!**** M1 < M1-dM1new < M1-dM1  =>  0 < - dM1new <  - dM1  =>  dM1new = dM1 * EKTeff/EGXmEKUD
      dM1 = - Max (- dM1 * EKTeff / EGXmEGUD, M1min-M1)  !  -dM1new+M1 >= M1min
      ESD = Ay3 * (3*MLAKE(2)**2 + 3*MLAKE(2)*dM1 + dM1**2) + ES2bot
      dE1 = ESD * dM1
      Return

!**** EKT is sufficient to mix through M1 - dM1
  400 Return
      EndSubroutine LAKMIX



      SUBROUTINE ALLOC_LAKES (GRID)
C23456789012345678901234567890123456789012345678901234567890123456789012
!@SUM  To alllocate arrays whose sizes now need to be determined
!@+    at run-time
!@auth Raul Garza-Robles
      USE DOMAIN_DECOMP_ATM, only: DIST_GRID, getDomainBounds
      USE LAKES, ONLY: DHORZ,KDIREC,IFLOW,JFLOW, XYZC, conical
      IMPLICIT NONE
      TYPE (DIST_GRID), INTENT(IN) :: grid
      integer :: i_0h,i_1h,j_0h,j_1h

      I_0H = grid%I_STRT_HALO
      I_1H = grid%I_STOP_HALO
      J_0H = grid%J_STRT_HALO
      J_1H = grid%J_STOP_HALO

      ALLOCATE ( KDIREC (I_0H:I_1H,J_0H:J_1H),
     *            IFLOW (I_0H:I_1H,J_0H:J_1H),
     *            JFLOW (I_0H:I_1H,J_0H:J_1H),
     *            XYZC(3,I_0H:I_1H,J_0H:J_1H),
     *            DHORZ (I_0H:I_1H,J_0H:J_1H),
     *           conical(I_0H:I_1H,J_0H:J_1H)
     *            )
      RETURN
      END SUBROUTINE ALLOC_LAKES


      SUBROUTINE init_LAKES(inilake,istart)
!@sum  init_LAKES initializes lake variables
!@auth Gary Russell/Gavin Schmidt
      USE FILEMANAGER
      USE CONSTANT, only : rhow,shw,tf,pi,grav
      USE RESOLUTION, only : im,jm
      Use GEOM,       Only : lon2d,sinlat2d,coslat2d
      USE MODEL_COM, only : dtsrc, IWrite_Sv,JWrite_Sv
      USE ATM_COM, only : zatmo
      USE ATM_COM, only : traditional_coldstart_aic
#ifdef SCM
      USE SCM_COM, only : SCMopt,SCMin
#endif
      USE DOMAIN_DECOMP_ATM, only : GRID,WRITE_PARALLEL
      USE DOMAIN_DECOMP_ATM, only : getDomainBounds,HALO_UPDATE
      USE DOMAIN_DECOMP_ATM, only : am_i_root
      USE GEOM, only : axyp,imaxj,lonlat_to_ij,lon2d_dg,lat2d_dg
#ifdef TRACERS_WATER
      USE OldTracer_mod, only : trw0
#endif
      USE FLUXES, only : atmocn,atmsrf
     &     ,flice,focean,fearth0,flake0,fland
      USE GHY_COM, only : fearth
      USE LAKES
      USE LAKES_COM
      USE DIAG_COM, only : npts,conpt0,icon_LKM,icon_LKE
      USE Dictionary_mod
      use pario, only : par_open,par_close,read_dist_data

      IMPLICIT NONE
      INTEGER :: J_0,J_1,J_0H,J_1H,J_0S,J_1S,I_0,I_1,I_0H,I_1H
      LOGICAL :: HAVE_NORTH_POLE, HAVE_SOUTH_POLE
      INTEGER :: JMIN_FILL,JMAX_FILL

      LOGICAL, INTENT(InOut) :: inilake
      INTEGER, INTENT(IN) :: ISTART
!@sum sill level: max. level above which lake outflow occurs
!@dbparam sill_depth = mean lake depth (m) at sill level (=DLAKE0)
      real*8 :: sill_depth = 10d0 ! only used if TOPO file not present
!@dbparam flake_ic uniform initial lake fraction
      real*8 :: flake_ic = -1. ! default: flake0 (TOPO) - at sill level
!@var I,J,IU,JU,ID,JD loop variables
      INTEGER I,J,IU,JU,ID,JD,INM
      integer :: fid,ios
      character(len=80) :: fmtstr
      INTEGER iu_RVR  !@var iu_RVR unit number for river direction file
      CHARACTER TITLEI*80, CONPT(NPTS)*10
      REAL*8 SPMIN,SPMAX,SPEED0,SPEED,DZDH,DZDH1,MLK1,fac,fac1
      LOGICAL :: QCON(NPTS), T=.TRUE. , F=.FALSE.
!@var out_line local variable to hold mixed-type output for parallel I/O
      character(len=300) :: out_line
      Logical :: QCS,QLL, QHALO
      integer :: jloop_min,jloop_max
      real*8 :: horzdist_2pts ! external function for now
      REAL*8, allocatable, dimension(:,:) :: down_lat_loc,down_lon_loc
      INTEGER, dimension(2) :: ij
      REAL*8, dimension(2) :: ll
      REAL*4, dimension(nrvrmx) :: lat_rvr,lon_rvr
      INTEGER get_dir
      REAL*8, DIMENSION(:,:), POINTER :: GTEMP,GTEMP2,GTEMPR
#ifdef TRACERS_WATER
      REAL*8, DIMENSION(:,:,:), POINTER :: GTRACER
#endif
!@dbparam Cael_lakes switch to select default shape of new lakes
      Integer :: Cael_lakes = 0

      GTEMP => ATMOCN%GTEMP
      GTEMP2 => ATMOCN%GTEMP2
      GTEMPR => ATMOCN%GTEMPR
#ifdef TRACERS_WATER
      GTRACER => ATMOCN%GTRACER
#endif

      call getDomainBounds(GRID, J_STRT = J_0, J_STOP = J_1,
     &               J_STRT_SKP = J_0S, J_STOP_SKP = J_1S,
     &               J_STRT_HALO= J_0H, J_STOP_HALO= J_1H,
     &               HAVE_SOUTH_POLE = HAVE_SOUTH_POLE,
     &               HAVE_NORTH_POLE = HAVE_NORTH_POLE)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP
      I_0H = grid%I_STRT_HALO
      I_1H = grid%I_STOP_HALO

C****
C**** Progn.  MWL      Mass of water in lake (kg)
C****         GML      Liquid lake enthalpy (J)
C****         TLAKE    Temperature of lake surface (C)
!****         T2Lbot   Temperature at bottom of lake layer 2 (C)
C****         DLAKE    Lake mean depth (m)
C****         FLAKE    Lake fraction (1)
C****         TANLK    Lake slope (tan(alpha)) (1)
C****
C**** Fixed   FLAKE0   Lake fraction at sill level (1)
C****         DLAKE0   Lake sill depth (m)

C**** Get parameters from rundeck
      call sync_param("river_fac",river_fac)
      call sync_param("xEKTlake", xEKTlake)
      call sync_param("yMEKTlake", yMEKTlake)
      call sync_param("init_flake",init_flake)
      if (init_flake == 1 .and. istart <  9) inilake = .true.
      call sync_param("variable_lk",variable_lk)
      call sync_param("lake_rise_max",lake_rise_max)
      call sync_param("lake_ice_max" ,lake_ice_max)
      call sync_param("Cael_lakes",Cael_lakes)
      call sync_param("Power_law_lakes",Power_law_lakes)
      call sync_param("C_lake",C_lake) ! used if Power_law_lakes>0
      call sync_param("E_lake",E_lake) ! used if Power_law_lakes>0
      call sync_param("Flake_cutoff",Flake_cutoff)
      call sync_param("small_lake_evap",small_lake_evap)
C****
C**** Set fixed geometric variables
C****
C**** Set Sill Depth DLAKE0 (mean lake depth above which outflow starts)
C**** The corresponding lake fraction FLAKE0 was set in ALLOC_FLUXES
      if(file_exists('TOPO')) then
        fid = par_open(grid,'TOPO','read')
        call read_dist_data(grid,fid,'hlake',dlake0)
        call par_close(grid,fid)
      else
        call sync_param("sill_depth", sill_depth)
        dlake0 = sill_depth
      end if
!**   Ensure that DLAKE0 is a minimum of 1m
      dlake0 = max(dlake0,dlake0_min)

C**** Set the logical array "conical" according to "Power_law_lakes"
C**** In some cases, DLAKE0 is reset 
      if (Power_law_lakes==2) then ! ignore dlake0 in TOPO where flake>0
        where (flake0 > 0) dlake0 = C_lake * (flake0*AXYP)**(E_lake-1)
!***    if flake0 == 0, & Cael_lakes == 1, DLAKE0 is reset below
        conical = .false.
      else if (Power_law_lakes==1) then
        conical = flake0 > 0
      else ! Power_law_lakes == 0
        conical = .true.
      end if
 
      if (Power_law_lakes < 2) then
        !! to relate volume and area of a lake, we use a conical shape
        !!  of equivalent volume with TANLK=TAN(ALPHA) = R/H (1/slope)
        where (flake0 > 0)
          TANLK = SQRT(FLAKE0*AXYP/PI)/(3d0*DLAKE0)
        elsewhere
          TANLK = 2d3   ! reasonable average value
        end where
      end if

      if (Power_law_lakes == 0 .and. Cael_lakes == 1) then
      !! get new default values for DLAKE0 and TANLK for new lakes
        do j=j_0,j_1
          do i=i_0,i_1
            if (flake0(i,j)==0)
     *         TANLK(I,J) = TANLK_Cael(axyp(I,J),DLAKE0(i,j))
          end do
        end do
        dlake0 = max(dlake0,dlake0_min)
      endif
      Call HALO_UPDATE (GRID, DLAKE0)

      if (INILAKE) THEN
C**** initialize FLAKE if requested (i.e. from older restart files)
        call sync_param("flake_ic",flake_ic)
        if (flake_ic < 0 .and. .not.file_exists('LAKEIC')) then
          If (Am_I_Root()) print*,"Initializing FLAKE from TOPO file..."
          FLAKE = FLAKE0
        else
          if (file_exists('LAKEIC')) then
            if(Am_I_Root()) print*,"Initial FLAKE from LAKEIC file..."
            fid = par_open(grid,'LAKEIC','read')
            call read_dist_data(grid,fid,'flake',flake)
            call par_close(grid,fid)
          else
            If (Am_I_Root()) print*,"Initial flake uniform",flake_ic
            flake = flake_ic
          end if
!****   prevent flooding at initial start
          flake = min(flake,flake0)
        end if

C**** Initialize lake depth DLAKE
        Dlake = Dlake0
        where (conical .and. flake < flake0)
          dlake = dlake0*sqrt(flake/flake0)
        else where (flake < flake0)
          dlake = C_lake*(flake*AXYP)**(E_lake-1)
        end where

        Dlake = max(Dlake, Dlake0_min)

C**** Set lake variables from surface temperature
C**** This is just an estimate for the initialization
        if(istart<=2) then ! pbl has not been initialized yet
          if(traditional_coldstart_aic)
          ! todo: get this temperature via other means
     &         call read_pbl_tsurf_from_nmcfile
        endif

        DO J=J_0, J_1
          DO I=I_0, I_1
            IF (FLAKE(I,J).gt.0) THEN
              TLAKE(I,J) = MAX(0d0,atmsrf%TSAVG(I,J)-TF)
              MWL(I,J)   = RHOW*DLAKE(I,J)*FLAKE(I,J)*AXYP(I,J)
              MLK1       = MINMLD*RHOW*FLAKE(I,J)*AXYP(I,J)
              GML(I,J)   = SHW*(MLK1*TLAKE(I,J)
     *             +(MWL(I,J)-MLK1)*MAX(TLAKE(I,J),4d0))
              MLDLK(I,J) = MINMLD
              T2Lbot(I,J) = Max (TLAKE(I,J), 4d0)
#ifdef TRACERS_WATER
              TRLAKE(:,1,I,J)=MLK1*TRW0()
              TRLAKE(:,2,I,J)=(MWL(I,J)-MLK1)*TRW0()
#endif
            ELSE
              TLAKE(I,J) = 0.
              MWL(I,J)   = 0.
              GML(I,J)   = 0.
              MLDLK(I,J) = MINMLD
              T2Lbot(I,J) = 0
#ifdef TRACERS_WATER
              TRLAKE(:,:,I,J)=0.
#endif
            END IF
            atmocn%MLHC(I,J)= SHW*MLDLK(I,J)*RHOW

            If (I==IWrite_Sv .and. J==JWrite_Sv)  Write (0,*)
     *        'INIT_LAKE: FLAKE=',I,J,FLAKE(I,J),DLAKE0(I,J),MLDLK(I,J),
     *        TLAKE(I,J),T2Lbot(I,J),DLAKE0(I,J),MWL(I,J),GML(I,J)

          END DO
        END DO
      end if

      IF (init_flake.eq.2 .and. istart.le.9) THEN
        print*,"Checking for excess water..."
        DO J=J_0, J_1
          DO I=I_0, I_1
            if (FLAKE(I,J) > .949d0*(1-FOCEAN(I,J)-FLICE(I,J)) .and.
     *           MWL(I,J) > (DLAKE0(I,J)+LAKE_RISE_MAX)*FLAKE(I,J)*RHOW
     *           *AXYP(I,J) ) then
              print*,"Adjusting lake level:",i,j," from ",MWL(I,J)
     *             /(FLAKE(I,J)*RHOW*AXYP(I,J))," to ",DLAKE0(I,J)
     *             +LAKE_RISE_MAX,MLDLK(I,J)
              fac=(MWL(I,J)-(DLAKE0(I,J)+LAKE_RISE_MAX)*FLAKE(I,J)*RHOW
     *             *AXYP(I,J))  ! excess mass
                                ! fractional loss of layer 1 mass
              fac1=fac/(MLDLK(I,J)*FLAKE(I,J)*RHOW*AXYP(I,J))
              if (fac1.lt.1) then
#ifdef TRACERS_WATER
                TRLAKE(:,1,I,J)=TRLAKE(:,1,I,J)*(1d0-fac1)
#endif
                MLDLK(I,J)=MLDLK(I,J)*(1-fac1)
                GML(I,J)=GML(I,J)-fac*SHW*TLAKE(I,J)
                MWL(I,J)=MWL(I,J)-fac
                atmocn%MLHC(I,J)= SHW*MLDLK(I,J)*RHOW
              else
                call stop_model(
     *               'INIT_LAKE: Inconsistent ml depth in lakes',255)
              end if
            end if
          END DO
        END DO
      END IF

      CALL PRINTLK("IN")
C**** Set GTEMP arrays for lakes
      DO J=J_0, J_1
        DO I=I_0, I_1
          IF (FLAKE(I,J).gt.0) THEN
            GTEMP(I,J)=TLAKE(I,J)
            GTEMPR(I,J) =TLAKE(I,J)+TF
#ifdef SCM
            if (SCMopt%Tskin) then
              GTEMP(I,J) = SCMin%Tskin - TF
              GTEMPR(I,J) = SCMin%Tskin
            endif
#endif
            IF (MWL(I,J).gt.(1d-10+MLDLK(I,J))*RHOW*FLAKE(I,J)*
     &           AXYP(I,J)) THEN
             GTEMP2(I,J)=(GML(I,J)-TLAKE(I,J)*SHW*MLDLK(I,J)*RHOW
     *             *FLAKE(I,J)*AXYP(I,J))/(SHW*(MWL(I,J)-MLDLK(I,J)
     *             *RHOW*FLAKE(I,J)*AXYP(I,J)))
C**** If starting from a possibly corrupted rsf file, check Tlk2
              IF(GTEMP2(I,J)>TLAKE(I,J)+1.and.GTEMP2(I,J)>10
     *           .and. istart<9) THEN
                WRITE(6,*) "Warning: Unphysical Tlk2 fixed",I,J,
     &               GTEMP(I,J),GTEMP2(I,J)
                GTEMP2(I,J)=GTEMP(I,J)  ! set to Tlk1
                GML(I,J)=TLAKE(I,J)*SHW*MWL(I,J)
                T2Lbot(I,J) = GTEMP2(I,J)
              END IF
            ELSE
              GTEMP2(I,J)=TLAKE(I,J)
            END IF
#ifdef SCM
            if (SCMopt%Tskin) then
              GTEMP2(I,J) = GTEMP(I,J)
            endif
#endif
#ifdef TRACERS_WATER
            GTRACER(:,I,J)=TRLAKE(:,1,I,J)/(MLDLK(I,J)*RHOW*FLAKE(I,J)
     *           *AXYP(I,J))
#endif
          END IF
        END DO
      END DO

!****
!**** Set river flow parameters and directions
!****
      allocate(down_lat_loc(I_0H:I_1H,J_0H:J_1H),
     *     down_lon_loc(I_0H:I_1H,J_0H:J_1H))

C**** Read named river mouth positions
      nrvr = 0
      if(file_exists('NAMERVR')) then
        call openunit("NAMERVR",iu_RVR,.false.,.true.)
        read(iu_RVR,*)
        read(iu_RVR,'(a)') fmtstr
        do
          read(iu_RVR,trim(fmtstr),iostat=ios)
     &         namervr(nrvr+1),lat_rvr(nrvr+1),lon_rvr(nrvr+1)
          if(ios.ne.0) exit
          nrvr = nrvr + 1
        enddo
        call closeunit(iu_RVR)

        WRITE (out_line,*) 'Named river file read '
        CALL WRITE_PARALLEL(trim(out_line), UNIT=6)
      endif
C**** Read in down stream lat/lon positions
      if(file_exists('RVR')) then
        fid = par_open(grid,'RVR','read')
        call read_dist_data(grid,fid,'down_lat',down_lat_loc)
        call read_dist_data(grid,fid,'down_lon',down_lon_loc)
        call par_close(grid,fid)
      else
        do j=j_0,j_1
        do i=i_0,i_1
          down_lon_loc(i,j) = lon2d_dg(i,j)
          down_lat_loc(i,j) = lat2d_dg(i,j)
        enddo
        enddo
      endif
      CALL HALO_UPDATE(GRID, down_lat_loc)
      CALL HALO_UPDATE(GRID, down_lon_loc)

C**** From each box calculate the downstream river box
!**** Compute IFLOW, JFLOW, KDIREC, and DHORZ
      if(have_south_pole) then
        jloop_min=1
      else
        jloop_min=j_0h
      endif
      if(have_north_pole) then
        jloop_max=jm
      else
        jloop_max=j_1h
      endif
      QCS = I_0H < I_0 ; QLL = .not. QCS
      do j=jloop_min,jloop_max
        Do I=I_0H,I_1H
          QHALO = I < I_0 .or. I_1 < I .or. J < J_0 .or. J_1 < J
!**** Invalidate upstream cells that will not be used by RIVERF
          If (I==0    .and. J==0 .or.     !  Invalidate upstream cells
     *        I==IM+1 .and. J==0 .or.     !  that are halo corners
     *        I==0    .and. J==IM+1 .or.  !  of cube faces
     *        I==IM+1 .and. J==IM+1 .or.
     *        QLL .and. J== 1 .and. I>=2 .or.
     *        QLL .and. J==JM .and. I>=2 .or.
     *        FOCEAN(I,J) == 1 .or.               !  entirely ocean cell
     *        FOCEAN(I,J) >  0 .and. QHALO) Then  !  partial ocean cell
               IFLOW(I,J) = -99  ;  JFLOW(I,J) = -99
              KDIREC(I,J) = 0    ;  DHORZ(I,J) = 0  ;  Cycle  ;  EndIf
!**** In RIVERF, lake water in partial ocean cells is dumped into ocean
          If (FOCEAN(I,J) > 0 .or. DOWN_LON_LOC(I,J) == -9999) Then
               IFLOW(I,J) = I  ;  JFLOW(I,J) = J
              KDIREC(I,J) = 0  ;  DHORZ(I,J) = 0  ;  Cycle  ;  EndIf
!**** All subsequent cells are entirely continental: FOCEAN(I,J) = 0
          If (down_lon_loc(i,j) <= -1000) Then
             Write (6,*) 'LAKES_INIT: Invalid Down_Lon_Loc(I,J):',
     *          I,J,Down_Lon_Loc(I,J)
             Call STOP_MODEL ('LAKES_INIT: Bad Down_Lon_Loc',255)
          EndIf
!**** Compute downstream cells and river directions
          ll(1)=down_lon_loc(i,j)
          ll(2)=down_lat_loc(i,j)
          call lonlat_to_ij(ll,ij)
          IFLOW(I,J)=ij(1) ; JFLOW(I,J)=ij(2)
!**** If I == IFLOW(I,J) and J == JFLOW(I,J), then KDIREC(I,J) = 0
          If (I == IFLOW(I,J) .and. J == JFLOW(I,J)) Then
             KDIREC(I,J) = 0  ;  DHORZ(I,J) = 0  ;  Cycle  ;  EndIf
!**** All subsequent cells have directional flow
!**** Invalidate cells where both up and down cells are in halo ring
          If (QHALO .and.
     *        (IFLOW(I,J) < I_0 .or. I_1 < IFLOW(I,J) .or.
     *         JFLOW(I,J) < J_0 .or. J_1 < JFLOW(I,J))) Then
               IFLOW(I,J) = -99  ;  JFLOW(I,J) = -99
              KDIREC(I,J) = 9    ;  DHORZ(I,J) = 0  ;  Cycle  ;  EndIf
!**** Valid directional flow
          DHORZ(I,J) = horzdist_2pts(i,j,iflow(i,j),jflow(i,j))
          KDIREC(I,J) = get_dir (I,J,IFLOW(I,J),JFLOW(I,J),IM,JM)
          If (KDIREC(I,J) < 1 .or. 8 < KDIREC(I,J)) Then
             Write (6,*) 'LAKES_INIT: I,J,KDIREC,ID,JD,DHORZ =',
     *           I,J,KDIREC(I,J),IFLOW(I,J),JFLOW(I,J),DHORZ(I,J)
             Call STOP_MODEL ('LAKES_INIT: Bad KDIREC',255)  ;  EndIf
        END DO
      END DO

!**** XYZC = (x,y,z) unit vectors on sphere of primary cell centers
      Do J=jloop_min,jloop_max  ;  Do I=I_0H,I_1H
         XYZC(1,I,J) = CosLat2d(I,J) * Cos(Lon2d(I,J))
         XYZC(2,I,J) = CosLat2d(I,J) * Sin(Lon2d(I,J))
         XYZC(3,I,J) = SinLat2d(I,J)  ;  EndDo  ;  EndDo

C**** define river mouths
      do inm=1,nrvr
        ll(1)=lon_rvr(inm)
        ll(2)=lat_rvr(inm)
        call lonlat_to_ij(ll,ij)

        IRVRMTH(INM)=ij(1) ; JRVRMTH(INM)=ij(2)
c        write(*,*) "rvr->",namervr(inm),ij(1),ij(2)

        if (IRVRMTH(INM).ge.I_0H .and. IRVRMTH(INM).le.I_1H .and.
     *       JRVRMTH(INM).ge.J_0H .and. JRVRMTH(INM).le.J_1H) THEN
          IF (FOCEAN(IRVRMTH(INM),JRVRMTH(INM)).le.0) WRITE(6,*)
     *         "Warning: Named river outlet must be in ocean"
     *         ,INM,IRVRMTH(INM),JRVRMTH(INM),NAMERVR(INM)
     *         ,FOCEAN(IRVRMTH(INM),JRVRMTH(INM)),FLICE(IRVRMTH(INM)
     *         ,JRVRMTH(INM)),FLAKE0(IRVRMTH(INM),JRVRMTH(INM))
     *         ,FEARTH0(IRVRMTH(INM),JRVRMTH(INM))
        end if
      end do

      do j=j_0,j_1
      do i=i_0,imaxj(j)
        if(flake(i,j).gt.0.) then
          DLAKE(I,J)=MWL(I,J)/(RHOW*FLAKE(I,J)*AXYP(I,J))
          GLAKE(I,J)=GML(I,J)/(FLAKE(I,J)*AXYP(I,J))
        else
          DLAKE(I,J)=0.
          GLAKE(I,J)=0.
        endif
      enddo
      enddo

C**** assume that at the start GHY is in balance with LAKES
      SVFLAKE = FLAKE

C**** Make sure that constraints are satisfied by defining FLAND/FEARTH
C**** as residual terms.
      DO J=J_0,J_1
      DO I=I_0,IMAXJ(J)
!!      FLAND(I,J)=1-FOCEAN(I,J)  !! already set if FOCEAN>0
        IF (FOCEAN(I,J).le.0) THEN
          FLAND(I,J)=1
          IF (FLAKE(I,J).gt.0) FLAND(I,J)=1-FLAKE(I,J)
        END IF
        FEARTH(I,J)=1-FOCEAN(I,J)-FLICE(I,J)-FLAKE(I,J) ! Earth fraction
      END DO
      END DO
      If (HAVE_SOUTH_POLE) Then
         FLAND(2:IM,1)=FLAND(1,1)
         FEARTH(2:IM,1)=FEARTH(1,1)
      End If
      If (HAVE_NORTH_POLE) Then
         FLAND(2:IM,JM)=FLAND(1,JM)
         FEARTH(2:IM,JM)=FEARTH(1,JM)
      End If

      Call HALO_UPDATE (GRID, FLAKE)
      Call HALO_UPDATE (GRID, FLAND)
      Call HALO_UPDATE (GRID, FEARTH)

C**** Set conservation diagnostics for Lake mass and energy
      CONPT=CONPT0
      CONPT(4)="PREC+LAT M"
      CONPT(5)="SURFACE"   ; CONPT(8)="RIVERS"
      QCON=(/ F, F, F, T, T, F, F, T, T, F, F/)
      CALL SET_CON(QCON,CONPT,"LAK MASS","(KG/M^2)        ",
     *     "(10**-9 KG/SM^2)",1d0,1d9,icon_LKM)
      QCON=(/ F, F, F, T, T, F, F, T, T, F, F/)
      CALL SET_CON(QCON,CONPT,"LAK ENRG","(10**3 J/M^2)   ",
     *     "(10**-3 W/M^2)  ",1d-3,1d3,icon_LKE)

      RETURN
C****
 910  FORMAT (A72)
 911  FORMAT (72A1)

      END SUBROUTINE init_LAKES

      function horzdist_2pts(i1,j1,i2,j2)
      use constant, only : radius
      use geom, only : lon2d,sinlat2d,coslat2d,axyp
      implicit none
      real*8 :: horzdist_2pts
      integer :: i1,j1,i2,j2
      real*8 :: x1,y1,z1, x2,y2,z2
      if(i1.eq.i2 .and. j1.eq.j2) then ! within same box
        horzdist_2pts = SQRT(AXYP(I1,J1))
      else                      ! use great circle distance
        x1 = coslat2d(i1,j1)*cos(lon2d(i1,j1))
        y1 = coslat2d(i1,j1)*sin(lon2d(i1,j1))
        z1 = sinlat2d(i1,j1)
        x2 = coslat2d(i2,j2)*cos(lon2d(i2,j2))
        y2 = coslat2d(i2,j2)*sin(lon2d(i2,j2))
        z2 = sinlat2d(i2,j2)
        horzdist_2pts = radius*acos(x1*x2+y1*y2+z1*z2)
      endif
      end function horzdist_2pts

      integer function get_dir(I,J,ID,JD,IM,JM)
      use domain_decomp_atm, only : grid
      use domain_decomp_1d, only: hasSouthPole, hasNorthPole

!@sum get_dir derives the locally orientated river direction
      integer, intent(in) :: I,J,ID,JD,IM,JM
      integer ::  DI,DJ

      DI=I-ID
      IF (DI.eq.IM-1) DI=-1
      IF (DI.eq.1-IM) DI=1
      DJ=J-JD
      get_dir=-99
      if (DI.eq.-1 .and. DJ.eq.-1) then
        get_dir=1
      elseif (DI.eq.-1 .and. DJ.eq.0) then
        get_dir=8
      elseif (DI.eq.-1 .and. DJ.eq.1) then
        get_dir=7
      elseif (DI.eq.0 .and. DJ.eq.1) then
        get_dir=6
      elseif (DI.eq.0 .and. DJ.eq.0) then
        get_dir=0
      elseif (DI.eq.0 .and. DJ.eq.-1) then
        get_dir=2
      elseif (DI.eq.1 .and. DJ.eq.-1) then
        get_dir=3
      elseif (DI.eq.1 .and. DJ.eq.0) then
        get_dir=4
      elseif (DI.eq.1 .and. DJ.eq.1) then
        get_dir=5
      end if
      if (hasNorthPole(grid) .and. J.eq.JM) then         ! north pole
        if (DI.eq.0) then
          get_dir=6
        else
          get_dir=8
        end if
      elseif (hasSouthPole(grid) .and. J.eq.1) then      ! south pole
        if (DI.eq.0) then
          get_dir=2
        else
          get_dir=8
        end if
! these next two cases need two latitudes to be on same processor
      elseif (hasNorthPole(grid) .and. J.eq.JM-1) then
        if (JD.eq.JM) get_dir=2
      elseif (hasSouthPole(grid) .and. J.eq.2) then
        if (JD.eq.1) get_dir=6
      end if
      if (get_dir.eq.-99) then
        print*,"get_dir error",i,j,id,jd
        get_dir=0
      end if

      return
      end function get_dir



      SUBROUTINE RIVERF
!@sum  RIVERF transports lake water from each grid cell downstream
!@auth Gary L. Russell
!@ver  2022/03/01   Emergency river directions are eliminated

      USE CONSTANT, only : shw,rhow,teeny,byGRAV,RADIUS,tf
      USE RESOLUTION, only : im,jm
      USE MODEL_COM, only : dtsrc,itime
      USE ATM_COM, only : GZATMO=>ZATMO  !  (m^2/s^2)
      USE DOMAIN_DECOMP_ATM, only : HALO_UPDATE, GRID,getDomainBounds
      USE GEOM, only : axyp,byaxyp,imaxj,byIM
      USE DIAG_COM, only : aij=>aij_loc,ij_ervr,ij_mrvr,ij_f0oc,
     *     jreg,j_rvrd,j_ervr,ij_fwoc,ij_ervro,ij_mrvro, ij_rvrflo
     *     ,IJ_RiverSpeed,IJ_ZLakeTop, itlake,itlkice,itocean,itoice
      USE FLUXES, only : atmocn,focean,FLICE
      USE LAKES, only : kdirec,iflow,jflow,river_fac, DHORZ,XYZC,
     *                  lake_rise_max
      USE LAKES_COM, only : tlake,gml,mwl,mldlk,flake,dlake,glake,
     *                      DLAKE0  !  fixed lake sill depth
      USE SEAICE_COM, only : lakeice=>si_atm
      Use TimerPackage_Mod, only: StartTimer=>Start,StopTimer=>Stop

#ifdef SCM
      USE SCM_COM, only : SCMopt,SCMin
#endif
#ifdef TRACERS_WATER
      USE TRDIAG_COM, only : taijn =>taijn_loc , tij_rvr, tij_rvro
      Use LAKES_COM, Only: NTM,TRLAKE
#endif

      IMPLICIT NONE
      Real*8,Parameter ::
     *   URATE  = 1d-6,  !  e-folding time (1/s) lake mass to ocean
     *   SPEEDMAX = 20   !  maximum river speed (m/s)
      Integer :: I,J, IU,JU, ID,JD, I0,I1,IN,IP, J0,J1,JN,JP, dI,dJ, JR
      Logical :: QCS,QLL, QHALO
      Real*8 MWLSILL, !  lake mass (kg) below sill depth
     *      MWLSILLD, !  downstream lake mass (kg) below sill depth
     *      FLAKEmin, !  for river flow, small lakes are spread horizontally
     *       FLAKEup, !  = Max (FLAKE(IU,JU), FLAKEmin)
     *       FLAKEdn, !  = Max (FLAKE(ID,JD), FLAKEmin)
     *        ZLtopU, !  upstream lake top altitude above sea level (m)
     *        ZLtopD, !  downstream lake top altitude or upstream sill
     *            dZ, !  ZLtopU minus (ZATMO sill or ZLtopD) (m), + or -
     *         SPEED, !  river SPEED (m/s) = dZ * RIVER_FAC
     *      MFLiqMax, !  maximum MFLiq, from .75 * Min(MLDLK,minMLD)
     *         MFLiq, !  river flow mass (kg)
     *         EFLiq, !  static energy of river flow (J)
!!   *         MFIce, !  river flow ice mass (kg)
!!   *         EFIce, !  static energy of river ice flow (J)
     *         DHORT, !  distance (m) between centers of 2 primary cells
     *         HLDLK, !  enthalpy per unit area of mixed layer (J/m^2)
     *          dMWL, !  = MFLiqIn - MFLiqOut = d(Mass) (kg)
     *          dEWL, !  = EFLiqIn - EFLiqOut = d(StaticEnergy) (J)
     *          dGML, !  = dEWL - dMWL*GZATMO = d(Enthalpy) (J)
     *        dMLDLK, !  change in mixed layer depth (m)
     *             A  !  A coefficient in quadratic polynomial
      REAL*8, DIMENSION(GRID%I_STRT_HALO:GRID%I_STOP_HALO,
     &                  GRID%J_STRT_HALO:GRID%J_STOP_HALO) ::
     *     MFLiqOut,  !  river mass (kg) leaving grid cell
     *     EFLiqOut   !  static energy (J) including surface geopotental
!!   *     MFIceOut,  !  river ice mass (kg) leaving grid cell
!!   *     EFIceOut   !  static energy (J) including surface geopotental
      REAL*8,DIMENSION(:,:),POINTER :: RSI,GTEMP,GTEMPR,
     *     MFLiqIn,   !  river mass (kg) entering grid cell
     *     EFLiqIn    !  static energy (J) including surface geopotental
!!   *     MFIceIn,   !  river ice mass (kg) entering grid cell
!!   *     EFIceIn    !  static energy (J) including surface geopotental

#ifdef TRACERS_WATER
      Real*8 :: TRMFLiq(NTM)
      REAL*8, DIMENSION(NTM,GRID%I_STRT_HALO:GRID%I_STOP_HALO,
     &                      GRID%J_STRT_HALO:GRID%J_STOP_HALO) ::
     *     TRMFLiqOut  !  tracer mass (kg) leaving grid cell via river
      REAL*8, DIMENSION(:,:,:), POINTER :: GTRACER,
     *     TRMFLiqIn   !  tracer mass (kg) entering grid cell via river
#endif

!**** MWL (kg) = Lake water in cell, defined even when FLAKE = 0
!****            such as ice sheets, deserts, and partial ocean cells
!**** GML (J)  = Liquid lake enthalpy excludes surface geopotential
!**** TLAKE(C) = Lake surface temperature
!****
!**** Enthalpy (heat content) (J) uses variables starting with G or H.
!**** Static Energy (J) starts with E = Enthalpy + Mass*GZATMO.
!**** Diagnostics accumulate E?? which may later be added among cells.
!**** Prognostic G?? is used within a column and ignores its ZATMO.
!**** Loss of surface geopotential energy by downhill river flow is
!**** eventually added to the downhill cell's enthalpy.
!****
!**** From entirely continental cells to entirely continental or ocean cell
!****   MFLiqOut,MRVRO leaves upstream entirely continental cell
!****   MFLiqIn,MRVR enters downstream entirely continental or ocean cell
!**** From entirely continental cells to partial continental cell
!****   MFLiqOut,MRVRO leaves upstream entirely continental cell
!****   -MFLiqOut,-MRVRO enters downstream partial continental cell
!****   above oddity necessary because 3 reservoirs, but 2 transfer arrays
!**** From partial continental to partial ocean within the same cell
!****   MFLiqOut,MRVRO leaves upstream partial continental cell
!****   MFLiqIn,MRVR enters downstream partial ocean cell

      call startTimer('RIVERF()')

!     QSP = HasSouthPole (GRID)
!     QNP = HasNorthPole (GRID)
      I1  = GRID%I_STRT       ;  IN = GRID%I_STOP  !  computational I
      J1  = GRID%J_STRT       ;  JN = GRID%J_STOP  !  computational J
      I0  = GRID%I_STRT_HALO  ;  IP = GRID%I_STOP_HALO  !  halo I
      J0  = GRID%J_STRT_HALO  ;  JP = GRID%J_STOP_HALO  !  halo J
      QCS = I0 .ne. I1
      QLL = I0 .eq. I1
      If (QLL) Then  ;  J0 = Max (J0,1)  ;  JP = Min (JP,JM)  ;  Endif

      MFLiqIn => ATMOCN%FLOWO
      EFLiqIn => ATMOCN%EFLOWO
      GTEMP   => ATMOCN%GTEMP
      GTEMPR  => ATMOCN%GTEMPR
      RSI     => LAKEICE%RSI
!!    MSI     => LakeIce%MSI
!!    HSI     => LakeIce%HSI
!!    SNOWI   => LakeIce%SNOWI
      MFLiqIn(:,:) = 0  ;  MFLiqOut(:,:) = 0
      EFLiqIn(:,:) = 0  ;  EFLiqOut(:,:) = 0
!!    MFIceIn(:,:) = 0  ;  MFIceOut(:,:) = 0
!!    EFIceIn(:,:) = 0  ;  EFIceOut(:,:) = 0
      CALL HALO_UPDATE(grid,   MWL)
      CALL HALO_UPDATE(grid, MLDLK)
      CALL HALO_UPDATE(grid, TLAKE)

#ifdef TRACERS_WATER
      TRMFLiqIn => ATMOCN%TRFLOWO
      GTRACER   => ATMOCN%GTRACER
      TRMFLiqIn(:,:,:) = 0  ;  TRMFLiqOut(:,:,:) = 0
      CALL HALO_UPDATE(grid,  GTRACER, jdim=3)
      CALL HALO_UPDATE(grid,  TRLAKE(:,1,:,:), jdim=3)
#endif

! Note on MPI fixes: Upstream and downstream cells may reside in the
! computational (non halo) cells of different PEs.  Upstream Do-loops
! include halo cells becuse the downstream cell may be among the PE's
! computational cells.  All input and output arrays are computed, but
! only a PE's prognostic computational cells are modified.  Halo edge
! prognostic variables will no longer be correct and HALO_UPDATE will
! be necessary.

      Do JU=J0,JP  ;  Do IU=I0,IP
!**** Skip upstream cells in the following situations:
!**** Cell is a non-existant halo corner cell of a cube-sphere face
!**** Cell is a lat-lon cell at poles with I=2:IM
!**** Cell is entirely ocean: FOCEAN(IU,JU) = 1
!**** Partial ocean cell resides in halo ring: 0 < FOCEAN(IU,JU) < 1
!**** Both up and down cells reside in halo ring and KDIREC(IU,JU) >< 0
         If (IFLOW(IU,JU) == -99)  Cycle

!        If (FOCEAN(IU,JU) == 1)  Cycle
!**** If FOCEAN = 1: Cycle to next non-full ocean upstream cell

         AIJ(IU,JU,IJ_ZLakeTop) = AIJ(IU,JU,IJ_ZLakeTop) +
     +                            GZATMO(IU,JU)*byGRAV

         If (FOCEAN(IU,JU) == 0)  GoTo 200
!**** 0 < FOCEAN < 1:  Then FLAKE = 0 and (ID,JD) = (IU,JU)
!****    MWL exits to ocean with e-folding time URATE = 10^-6 (1/s)
         If (MWL(IU,JU) < 1d-3)  Cycle
         MFLiq = MWL(IU,JU) * URATE * DTSRC
!#!      EFLiq = MFLiq * (SHW*TLAKE(IU,JU) + GZATMO(IU,JU))
         EFLiq = MFLiq *  SHW*TLAKE(IU,JU)
         MFLiqOut(IU,JU) = MFLiqOut(IU,JU) + MFLiq
         EFLiqOut(IU,JU) = EFLiqOut(IU,JU) + EFLiq
         MFLiqIn (IU,JU) = MFLiqIn (IU,JU) + MFLiq
         EFLiqIn (IU,JU) = EFLiqIn (IU,JU) + EFLiq
!!       SPEEDxDT = MFLiq / (RHOW * CrossArea)
!!       MFIce = (SNOWI(IU,JU) + ACE1I + MSI(IU,JU))    * (RSI(IU,JU)*FLAKE(IU,JU)*AXYP(IU,JU) * RFICE * SPEEDxDT / dX
!!       EFIce = MFIce*ZATMO(IU,JU) + Sum(HSI(:,UI,JU)) * (RSI(IU,JU)*FLAKE(IU,JU)*AXYP(IU,JU) * RFICE * SPEEDxDT / dX
!!       MFIceOut(IU,JU) = MFIceOut(IU,JU) + MFIce
!!       EFIceOut(IU,JU) = EFIceOut(IU,JU) + EFIce
!!       MFIceIn (IU,JU) = MFIceIn (IU,JU) + MFIce
!!       EFIceIn (IU,JU) = EFIceIn (IU,JU) + EFIce
#ifdef TRACERS_WATER
         TRMFLiq(:) = MFLiq*TRLAKE(:,1,IU,JU) / (MWL(IU,JU)+TEENY)
         TRMFLiqOut(:,IU,JU) = TRMFLiqOut(:,IU,JU) + TRMFLiq(:)
         TRMFLiqIn (:,IU,JU) = TRMFLiqIn (:,IU,JU) + TRMFLiq(:)
#endif
         Cycle

!****
!**** FOCEAN(IU,JU) = 0: Upstream cell is entirely continental
!****
  200    If (FLICE (IU,JU) <  1)  GoTo 300
!**** FLICE(IU,JU) = 1: Upstream cell is entirely land ice
!        If (KDIREC(IU,JU) == 0) Then
!           Write (0,*) 'Entire land ice cell has no river outlet',IU,JU
!           Write (6,*) 'Entire land ice cell has no river outlet',IU,JU
!           Call STOP_MODEL ('Land ice cell has no river outlet', 255)
!           EndIf
         If (KDIREC(IU,JU) == 0)  GoTo 300  !  compute ZLtopU needed for undirected flow at 500
!**** KDIREC = 1:8 and FLICE(IU,JU) = 1:
!**** Liquid lake water beneath land ice flows downstream
         ID = IFLOW(IU,JU)  ;  JD = JFLOW(IU,JU)
         SPEED = .25
         MFLiq = MWL(IU,JU) * dTSRC * SPEED / DHORZ(IU,JU)
         GoTo 400  !  branch to code section that stores single direction river flow

!**** FOCEAN(IU,JU) = 0 and FLICE(IU,JU) < 1:
!**** Allow river flow even when FLAKE(IU,JU) == 0
!**** Increase MWLSILL by setting FLAKEup = Max[FLAKE(IU,JU),FLAKEmin]
!**** MWLSILL (kg) = DLAKE0 * RHOW * FLAKEup * AXYP = water below ZATMO sill
!**** dZ (m) = (MWL-MWLSILL) / RHOW*FLAKEup*AXYP = lake altitude above ZATMO sill
!**** MFLiqMax (kg) = (MWL-MWLSILL)/4 = maximum river flow this step
!**** If FLAKE(IU,JU) > 0, limit MLDLKnew to MLDLK/4 via MFLiqMax
  300    FLAKEmin = 110000d0**2 / (1024*AXYP(IU,JU))  !  = 1/1024 for 1x1 degree cell
         FLAKEup = Max (FLAKE(IU,JU), FLAKEmin)
         MWLSILL = DLAKE0(IU,JU) * RHOW * FLAKEup * AXYP(IU,JU)
         dZ = (MWL(IU,JU)-MWLSILL) / (RHOW * FLAKEup * AXYP(IU,JU))
         AIJ(IU,JU,IJ_ZLakeTop) = AIJ(IU,JU,IJ_ZLakeTop) + dZ
         ZLtopU = dZ + GZATMO(IU,JU)*byGRAV
         MFLiqMax = .25 * (MWL(IU,JU) - MWLSILL)
         If (FLAKE(IU,JU) > 0)  MFLiqMax =
     *      Min(MFLiqMax,.25*RHOW*MLDLK(IU,JU)*FLAKE(IU,JU)*AXYP(IU,JU))
         If (MFLiqMax < 1d-3)  Cycle  !  ignore flow less than 1 (g)

         If (KDIREC(IU,JU) == 0)  GoTo 500

!**** KDIREC = 1:8 and FOCEAN(IU,JU) = 0 and FLICE(IU,JU) < 1:
         ID = IFLOW(IU,JU)  ;  JD = JFLOW(IU,JU)

         If (KDIREC(ID,JD) == 0 .and. FOCEAN(ID,JD) == 0) Then
!**** Downstream cell (ID,JD) has KDIREC(ID,JD) = 0 and is entirely continental
!**** Check whether backwash flow will occur later; if so do not allow flow here
!**** Check whether ZLtopU <= ZLtopD(ID,JD) and ZLtopD is above (ID,JD) sill
            FLAKEmin = 110000d0**2 / (1024*AXYP(ID,JD))  !  = 1/1024 for 1x1 degree cell
            FLAKEdn  = Max (FLAKE(ID,JD), FLAKEmin)
            MWLSILLD = DLAKE0(ID,JD) * RHOW * FLAKEdn * AXYP(ID,JD)
            If (MWL(ID,JD) > MWLSILLD) Then
               ZLtopD = (MWL(ID,JD)-MWLSILLD)/(RHOW*FLAKEdn*AXYP(ID,JD))
     +                  + GZATMO(ID,JD)*byGRAV
               If (ZLtopU <= ZLtopD)  Cycle
               dZ = Min (dZ, ZLtopU-ZLtopD)  !  reduce flow based on lake tops
               EndIf  ;  EndIf

!**** Normal single direction river flow
!**** DHORZ(IU,JU) (m) = distance from upstream cell to downstream cell
!**** River SPEED (m/s) = dZ (m) * RIVER_FAC (1/s)
!**** MFLiq (kg) = dZ*RHOW*FLAKEup*AXYP * dT*SPEED/DHORZ(IU,JU)
         SPEED = Min (SPEEDmax, RIVER_FAC * dZ)
         MFLiq = dZ*RHOW*FLAKEup*AXYP(IU,JU)* DTSRC*SPEED/DHORZ(IU,JU)
         MFLiq = Min (MFLiq, MFLiqMax)

!**** Store river flow mass and static energy when 1 <= KDIREC <= 8
  400    AIJ(IU,JU,IJ_RiverSpeed) = AIJ(IU,JU,IJ_RiverSpeed) + SPEED
!#!      EFLiq = MFLiq * (SHW*TLAKE(IU,JU) + GZATMO(IU,JU))
         EFLiq = MFLiq *  SHW*TLAKE(IU,JU)
         MFLiqOut(IU,JU) = MFLiqOut(IU,JU) + MFLiq
         EFLiqOut(IU,JU) = EFLiqOut(IU,JU) + EFLiq
         If (0 < FOCEAN(ID,JD) .and. FOCEAN(ID,JD) < 1)
     *      Then  ;  MFLiqOut(ID,JD) = MFLiqOut(ID,JD) - MFLiq  !  Partial continental and ocean cell
                     EFLiqOut(ID,JD) = EFLiqOut(ID,JD) - EFLiq
            Else  ;  MFLiqIn (ID,JD) = MFLiqIn (ID,JD) + MFLiq  !  Entirely continental or ocean cell
                     EFLiqIn (ID,JD) = EFLiqIn (ID,JD) + EFLiq  ;  EndIf
!!       SPEEDxDT = MFLiq / (RHOW * CrossArea)
!!       MFIce = (SNOWI(IU,JU) + ACE1I + MSI(IU,JU))    * (RSI(IU,JU)*FLAKE(IU,JU)*AXYP(IU,JU) * RFICE * SPEEDxDT / dX
!!       EFIce = MFIce*ZATMO(IU,JU) + Sum(HSI(:,UI,JU)) * (RSI(IU,JU)*FLAKE(IU,JU)*AXYP(IU,JU) * RFICE * SPEEDxDT / dX
!!       MFIceOut(IU,JU) = MFIceOut(IU,JU) + MFIce
!!       EFIceOut(IU,JU) = EFIceOut(IU,JU) + EFIce
!!       MFIceIn (ID,JD) = MFIceIn (ID,JD) + MFIce
!!       EFIceIn (ID,JD) = EFIceIn (ID,JD) + EFIce
#ifdef TRACERS_WATER
         If (FLAKE(IU,JU) > 0)
     *      Then  ;  TRMFLiq(:) = MFLiq*GTRACER(:,IU,JU)
            Else  ;  TRMFLiq(:) = MFLiq*TRLAKE(:,1,IU,JU) /
     *                            (MWL(IU,JU)+TEENY)  ;  EndIf
         TRMFLiqOut(:,IU,JU) = TRMFLiqOut(:,IU,JU) + TRMFLiq(:)
         If (0 < FOCEAN(ID,JD) .and. FOCEAN(ID,JD) < 1) Then
             TRMFLiqOut(:,ID,JD) = TRMFLiqOut(:,ID,JD) - TRMFLiq(:)
           Else  !  downstream cell is entirely continent or ocean
             TRMFLiqIn(:,ID,JD) = TRMFLiqIn(:,ID,JD) + TRMFLiq(:)
           EndIf
#endif
         Cycle  !  end normal single directional river flow

!**** KDIREC(IU,JU) = 0 and FOCEAN(IU,JU) = 0 and FLICE(IU,JU) < 1:
!**** Check flow to equilibrate lake top altitude of multi-cell lake
!**** MWL(IU,JU) > MWLSILL or ZLtopU > ZATMO(IU,JU), check each of 8 touching cells
!**** If KDIREC(ID,JD) = 0 (Caspian Sea)
!**** or KDIREC(ID,JD) directs flow back into (IU,JU) cell
!**** Then: ZLtopU = lake top altitude of (IU,JU) upstream cell
!****       ZLtopD = lake top altitude of (ID,JD) or upstream sill
!****       SPEED  = (ZLtopU-ZLtopD) * RIVER_FAC
!****       MFLiq  = .125 * (ZLtopU-ZLtopD)*RHOW*FLAKE*AXYP*dT*SPEED / DHORT (8 cells)
  500    If (QLL .and. (JU==1.or.JU==JM))  Cycle  ! not allowed at poles
         QHALO = IU < I1 .or. IN < IU .or. JU < J1 .or. JN < JU
         Do dJ=-1,1  ;  Do dI=-1,1
!**** Check that either upstream or downstream is a computational cell
            If (dI == 0 .and. dJ == 0)  Cycle
            If (JU < J1 .and. dJ <= 0)  Cycle
            If (JU > JN .and. dJ >= 0)  Cycle
            If (QCS) Then
               If (IU < I1 .and. dI <= 0)  Cycle
               If (IU > IN .and. dI >= 0)  Cycle
               ID = IU+dI  ;  JD = JU+dJ
               If (ID == 0    .and. JD == 0)  Cycle  !  skip halo cells
               If (ID == IM+1 .and. JD == 0)  Cycle  !  at cube-sphere
               If (ID == 0    .and. JD == IM+1)  Cycle  !  corners
               If (ID == IM+1 .and. JD == IM+1)  Cycle  ;  EndIf
            If (QLL) Then
               ID = IU+dI  ;  If (ID <  1) ID = ID+IM
               JD = JU+dJ  ;  If (ID > IM) ID = ID-IM
               If (JD==1 .or. JD==JM)  Cycle  !  not allowed at poles
               EndIf
!**** Check that downstream cell has KDIREC = 0, or
!**** downstream cell normally directs flow into present upstream cell
            If (.not.(KDIREC(ID,JD) == 0 .or.
     *                 IFLOW(ID,JD)==IU .and. JFLOW(ID,JD)==JU))  Cycle
!**** Compute lake top altitude ZLtopD (m) = Max (downstream lake top, upstream sill)
            FLAKEmin = 110000d0**2 / (1024*AXYP(ID,JD))  !  = 1/1024 for 1x1 degree cell
            FLAKEdn  = Max (FLAKE(ID,JD), FLAKEmin)
            MWLSILLD = DLAKE0(ID,JD) * RHOW * FLAKEdn * AXYP(ID,JD)
            ZLtopD = (MWL(ID,JD)-MWLSILLD)/(RHOW*FLAKEdn*AXYP(ID,JD))
     +               + GZATMO(ID,JD)*byGRAV
            If (ZLtopU <= ZLtopD)  Cycle
            ZLtopD = Max (ZLtopD, GZATMO(IU,JU)*byGRAV)

!**** Perform implicit computation for dZnew and MFLiq  ;  DHORT (m) = horizontal distance from (IU,JU) to (ID,JD)
!**** MF (m^3) = dZnew * FLAKEup*AXYPup * DTSRC*SPEED/DHORT = dZnew*2 * FLAKEup*AXYPup * DTSRC*RIVER_FAC/DHORT > 0
!**** ZLtopUnew * FLAKEup*AXYPup = ZLtopUold * FLAKEup*AXYPup - MF  =>  ZLtopUnew = ZLtopUold - MF / FLAKEup*AXYPup
!**** ZLtopDnew * FLAKEdn*AXYPdn = ZLtopDold * FLAKEdn*AXYPdn + MF  =>  ZLtopDnew = ZLtopDold + MF / FLAKEdn*AXYPdn
!**** dZnew = ZLtopUnew - ZLtopDnew = ZLtopUold - ZLtopDold - MF * (1/FLAKEup*AXYPup + 1/FLAKEdn*AXYPdn) =
!****       = ZLtopUold - ZLtopDold - dZnew*2 * FLAKEup*AXYPup * DTSRC*RIVER_FAC * (1/FLAKEup*AXYPup + 1/FLAKEdn*AXYPdn) / DHORT =
!****       = ZLtopUold - ZLtopDold - dZnew*2 * DTSRC*RIVER_FAC * (1 + FLAKEup*AXYPup/FLAKEdn*AXYPdn) / DHORT =
!****       = ZLtopUold - ZLtopDold - dZnew*2 * A  =>
!**** A * dZnew^2 + dZnew + (ZLtopDold - ZLtopUold = 0)     note: A (1/m) > 0 and C (m) < 0
!**** dZnew = [- B + Sqrt(B^2 - 4AC)] / 2A = {- 1 + Sqrt[1 - 4*A * (ZLtopDold - ZLtopUold)]} / 2*A > 0
            DHORT = RADIUS * ACos(Sum(XYZC(:,IU,JU)*XYZC(:,ID,JD)))
            A  = DTSRC * RIVER_FAC *
     *           (1 + FLAKEup*AXYP(IU,JU)/(FLAKEdn*AXYP(ID,JD))) / DHORT
            dZ = (- 1 + Sqrt(1 - 4*A * (ZLtopD - ZLtopU))) / (2*A)
            dZ = Min(dZ,(MWL(IU,JU)-MWLSILL)/(RHOW*FLAKEup*AXYP(IU,JU)))
            SPEED = Min (SPEEDmax, RIVER_FAC * dZ)
            MFLiq = dZ*RHOW*FLAKEup*AXYP(IU,JU) * DTSRC*SPEED/DHORT
            MFLiq = .125 * Min(MFLiq, MFLiqMax)  !  8 cells
!#!         EFLiq = MFLiq * (SHW*TLAKE(IU,JU) + GZATMO(IU,JU))
            EFLiq = MFLiq *  SHW*TLAKE(IU,JU)


    3 FORMAT ('RIVERF:',2I5,F15.10,5F15.8)
!      IF(ID==109.AND.JD==67) THEN
!        WRITE (0,3)
!        WRITE (0,3) ID,JD,FLAKE(ID,JD),ZLTOPD,A*1000,DZ,SPEED
!        WRITE (0,3) IU,JU,FLAKE(IU,JU),ZLTOPU,TLAKE(IU,JU),
!     *    MFLiq/(RHOW*FLAKE(IU,JU)*AXYP(IU,JU)),
!     *    MWL(IU,JU)/(RHOW*FLAKE(IU,JU)*AXYP(IU,JU))  ;  ENDIF

            MFLiqOut(IU,JU) = MFLiqOut(IU,JU) + MFLiq
            EFLiqOut(IU,JU) = EFLiqOut(IU,JU) + EFliq
            If (0 < FOCEAN(ID,JD) .and. FOCEAN(ID,JD) < 1)
     *         Then  ;  MFLiqOut(ID,JD) = MFLiqOut(ID,JD) - MFLiq
                        EFLiqOut(ID,JD) = EFLiqOut(ID,JD) - EFLiq
               Else  ;  MFLiqIn (ID,JD) = MFLiqIn (ID,JD) + MFLiq
                        EFLiqIn (ID,JD) = EFLiqIn (ID,JD) + EFLiq
               EndIf
#ifdef TRACERS_WATER
            TRMFLiq(:) = MFLiq*GTRACER(:,IU,JU)
            TRMFLiqOut(:,IU,JU) = TRMFLiqOut(:,IU,JU) + TRMFLiq(:)
            If (0 < FOCEAN(ID,JD) .and. FOCEAN(ID,JD) < 1) Then
                 TRMFLiqOut(:,ID,JD) = TRMFLiqOut(:,ID,JD) - TRMFLiq(:)
              Else  !  downstream cell is entirely continent or ocean
                 TRMFLiqIn(:,ID,JD) = TRMFLiqIn(:,ID,JD) + TRMFLiq(:)
              EndIf
#endif
            EndDo  ;  EndDo  ;  EndDo  ;  EndDo  !  loops dI, dJ, IU, JU

!****
!**** Apply river flow to prognostic variables and to diagnostics
!****
      Do J=J1,JN
!**** Copy Lat-Lon triangular polar wedge to all slices of circle later
      If (QLL .and. (J==1.or.J==JM)) Then
         MFLiqOut(1,J) = MFLiqOut(1,J) * byIM
         EFLiqOut(1,J) = EFLiqOut(1,J) * byIM
         MFLiqIn (1,J) = MFLiqIn (1,J) * byIM
         EFLiqIn (1,J) = EFLiqIn (1,J) * byIM
#ifdef TRACERS_WATER
         TRMFLiqOut(:,1,J) = TRMFLiqOut(:,1,J) * byIM
         TRMFLiqIn (:,1,J) = TRMFLiqIn (:,1,J) * byIM
#endif
         EndIf

      Do I=I1,IMAXJ(J)
         dMWL = MFLiqIn(I,J) - MFLiqOut(I,J)
         dEWL = EFLiqIn(I,J) - EFLiqOut(I,J)
!#!      dGML = dEWL - dMWL * GZATMO(I,J)
         dGML = dEWL
         AIJ(I,J,IJ_MRVRO) = AIJ(I,J,IJ_MRVRO) + MFLiqOut(I,J)
         AIJ(I,J,IJ_ERVRO) = AIJ(I,J,IJ_ERVRO) + EFLiqOut(I,J)
         AIJ(I,J,IJ_MRVR)  = AIJ(I,J,IJ_MRVR)  + MFLiqIn (I,J)
         AIJ(I,J,IJ_ERVR)  = AIJ(I,J,IJ_ERVR)  + EFLiqIn (I,J)
         JR = JREG(I,J)
         Call INC_AREG (I,J,JR,J_RVRD, dMWL*byAXYP(I,J))
         Call INC_AREG (I,J,JR,J_ERVR, dEWL*byAXYP(I,J))
#ifdef TRACERS_WATER
         TAIJN(I,J,TIJ_RVRO,:) = TAIJN(I,J,TIJ_RVRO,:) +
     +                           TRMFLiqOut(:,I,J) * byAXYP(I,J)
         TAIJN(I,J,TIJ_RVR ,:) = TAIJN(I,J,TIJ_RVR ,:) +
     +                           TRMFLiqIn(:,I,J) * byAXYP(I,J)
#endif
#ifdef TRACERS_OBIO_RIVERS
         AIJ(I,J,IJ_RVRFLO) = AIJ(I,J,IJ_RVRFLO) + MFLiqIn(I,J)
#endif

         If (FOCEAN(I,J) == 1)  GoTo 700
!****
!**** Apply river flow to continental cells
!****
         If (FOCEAN(I,J) > 0) Then  !  partial continental cell
            dMWL = - MFLiqOut(I,J)
            dEWL = - EFLiqOut(I,J)
!#!         dGML = dEWL - dMWL * GZATMO(I,J)  ;  EndIf
            dGML = dEWL                       ;  EndIf
         MWL(I,J) = MWL(I,J) + dMWL
         GML(I,J) = GML(I,J) + dGML
#ifdef TRACERS_WATER
         If (FOCEAN(I,J) > 0) Then
             TRLAKE(:,1,I,J) = TRLAKE(:,1,I,J) - TRMFLiqOut(:,I,J)
           Else
             TRLAKE(:,1,I,J) = TRLAKE(:,1,I,J) +
     +                       (TRMFLiqIn(:,I,J) - TRMFLiqOut(:,I,J))
           EndIf
#endif
         Call INC_AJ (I,J,ITLAKE ,J_RVRD, dMWL*byAXYP(I,J)*(1-RSI(I,J)))
         Call INC_AJ (I,J,ITLAKE ,J_ERVR, dEWL*byAXYP(I,J)*(1-RSI(I,J)))
         Call INC_AJ (I,J,ITLKICE,J_RVRD, dMWL*byAXYP(I,J)*   RSI(I,J) )
         Call INC_AJ (I,J,ITLKICE,J_ERVR, dEWL*byAXYP(I,J)*   RSI(I,J) )

         If (FLAKE(I,J) > 0) Then
            dMLDLK = dMWL / (RHOW*FLAKE(I,J)*AXYP(I,J))
            If (dMLDLK + .5*MLDLK(I,J) < 0) Then
               Write (6,*) 'See .OU file'
               Write (0,902) I,J,FLAKE(I,J),
     *            dMLDLK,MLDLK(I,J),DLAKE0(I,J),DLAKE(I,J),
     *            MFLiqOut(I,J)/(RHOW*FLAKE(I,J)*AXYP(I,J)),
     *            MWL     (I,J)/(RHOW*FLAKE(I,J)*AXYP(I,J))
               Call STOP_MODEL ('RIVERF: dMLDLK+MLDLK < 0.', 255)
               EndIf
            HLDLK      = TLAKE(I,J) *
     *         ((MLDLK(I,J) * RHOW * FLAKE(I,J) * AXYP(I,J)) * SHW)
            MLDLK(I,J) = MLDLK(I,J) + dMLDLK
            TLAKE(I,J) = (HLDLK + dGML) /
     /         ((MLDLK(I,J) * RHOW * FLAKE(I,J) * AXYP(I,J)) * SHW)
            GTEMP(I,J) = TLAKE(I,J)
           GTEMPR(I,J) = TLAKE(I,J) + TF
            DLAKE(I,J) = MWL(I,J) / (RHOW*FLAKE(I,J)*AXYP(I,J))
            GLAKE(I,J) = GML(I,J) / (FLAKE(I,J)*AXYP(I,J))
            AtmOcn%MLHC(I,J) = SHW*MLDLK(I,J)*RHOW
#ifdef TRACERS_WATER
            GTRACER(:,I,J) = TRLAKE(:,1,I,J) /
     /          (MLDLK(I,J) * RHOW * FLAKE(I,J) * AXYP(I,J))
#endif
#ifdef SCM
            If (SCMopt%Tskin) Then
                GTEMP(I,J) = SCMin%Tskin - TF
               GTEMPR(I,J) = SCMin%Tskin  ;  EndIf
#endif
         Else  !  FLAKE(I,J) == 0
            TLAKE(I,J) = GML(I,J) / (SHW*MWL(I,J) + TEENY)
            DLAKE(I,J) = 0
            GLAKE(I,J) = 0  ;  EndIf  !  If FLAKE > 0

         If (FOCEAN(I,J) == 0)  Cycle
!****
!**** Apply river flow to ocean cells, FOCEAN(I,J) > 0
!****
  700    If (MFLiqIn(I,J) == 0)  Cycle
         AIJ(I,J,IJ_FWOC) = AIJ(I,J,IJ_FWOC)+MFLiqIn(I,J)*byAXYP(I,J)
         AIJ(I,J,IJ_F0OC) = AIJ(I,J,IJ_F0OC)+EFLiqIn(I,J)*byAXYP(I,J)
         Call INC_AJ (I,J,ITOCEAN,J_RVRD,
     *                MFLiqIn(I,J)*byAXYP(I,J)*(1-RSI(I,J)))
         Call INC_AJ (I,J,ITOCEAN,J_ERVR,
     *                EFLiqIn(I,J)*byAXYP(I,J)*(1-RSI(I,J)))
         Call INC_AJ (I,J,ITOICE ,J_RVRD,
     *                MFLiqIn(I,J)*byAXYP(I,J)*   RSI(I,J) )
         Call INC_AJ (I,J,ITOICE ,J_ERVR,
     *                EFLiqIn(I,J)*byAXYP(I,J)*   RSI(I,J) )
!**** Convert mass (kg) and static energy (J) to mass per unit area
!**** and enthalpy per unit area (?/m^2 over ocean fraction)
         MFLiqIn(I,J) =  MFLiqIn(I,J) / (FOCEAN(I,J)*AXYP(I,J))
         EFLiqIn(I,J) =  EFLiqIn(I,J) / (FOCEAN(I,J)*AXYP(I,J))
#ifdef TRACERS_WATER
         TRMFLiqIn(:,I,J) = TRMFLiqIn(:,I,J) /(FOCEAN(I,J)*AXYP(I,J))
#endif
         EndDo  ;  EndDo  !  Do I  !  Do J

      Call PRINTLK ('RV')
      Call StopTimer ('RIVERF()')
  901 Format ('RIVERF: FLAKE,dZ,SPEED,MLDLK,MFLiq(m)',I8,2I5,5F11.6)
  902 Format ('RIVERF: dMLDLK=MLDmax*MLDLK < 0: I,J,FLAKE =',2I5,F10.6 /
     *        '  dMLDLK,MLDLK,DLAKE0,DLAKE,MFLiqOut,MLW(m) =',6F10.5)
      EndSubroutine RIVERF


      SUBROUTINE diag_RIVER
!@sum  diag_RIVER prints out the river outflow for various rivers
!@sum  (now parallel)
!@auth Gavin Schmidt

      USE CONSTANT, only : rhow,teeny,undef
      USE RESOLUTION, only : im,jm
      USE MODEL_COM, only : modelEclock
      USE MODEL_COM, only : jyear0,amon0,jdate0,jhour0,amon
     *     ,itime,dtsrc,idacc,itime0,nday, calendar
      use TimeConstants_mod, only: INT_MONTHS_PER_YEAR
      USE DOMAIN_DECOMP_ATM, only : GRID,WRITE_PARALLEL,
     $     AM_I_ROOT, getDomainBounds, sumxpe
      USE GEOM, only : byaxyp
      USE DIAG_COM, only : aij=>aij_loc,ij_mrvr
#ifdef TRACERS_WATER
      use OldTracer_mod, only: trname, trw0, itime_tr0,tr_wd_type,nWATER
      USE TRACER_COM, only : NTM,n_water
      USE TRDIAG_COM, only : taijn=>taijn_loc
      USE TRDIAG_COM, only : tij_rvr,to_per_mil,units_tij,scale_tij
#endif
      USE LAKES_COM, only : irvrmth,jrvrmth,namervr,nrvr
      use TimeInterval_mod
      use Rational_mod

      IMPLICIT NONE
      REAL*8 RVROUT(NRVR), RVROUT_root(NRVR), scalervr, days
      INTEGER INM,I,N,J
      LOGICAL increment
#ifdef TRACERS_WATER
      REAL*8 TRVROUT(NRVR,NTM)
#endif
!@var out_line local variable to hold mixed-type output for parallel I/O
      character(len=300) :: out_line
      integer :: I_0, I_1, J_0, J_1
      integer :: year, hour, date
      type (Rational) :: secondsPerYear

      call getDomainBounds(grid, J_STRT=J_0, J_STOP=J_1)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

      DAYS=(Itime-Itime0)/REAL(nday,kind=8)
      call modelEclock%get(year=year, hour=hour, date=date)
      WRITE(out_line,900) JYEAR0,AMON0,JDATE0,JHOUR0,YEAR,AMON,DATE,
     *      HOUR,ITIME,DAYS
      IF (AM_I_ROOT()) CALL WRITE_PARALLEL(trim(out_line), UNIT=6)
C**** convert kg/(source time step) to km^3/mon
      secondsPerYear =
     &     calendar%getMaxDaysInYear() * calendar%getSecondsPerDay()
      SCALERVR = 1d-9*real(secondsPerYear)/
     &          (INT_MONTHS_PER_YEAR*RHOW*DTSRC)

      RVROUT(:)=0
#ifdef TRACERS_WATER
      TRVROUT(:,:)=0.
#endif
C**** loop over whole grid
      DO J=J_0,J_1
        DO I=I_0,I_1
          DO INM=1,NRVR
            if (I.eq.IRVRMTH(INM).and. J.eq.JRVRMTH(INM)) THEN
              RVROUT(INM) = SCALERVR*AIJ(I,J,IJ_MRVR)/IDACC(1)
#ifdef TRACERS_WATER
              IF (RVROUT(INM).gt.0)  THEN
                DO N=1,NTM
                  if (to_per_mil(n).gt.0) then
                    if (TAIJN(I,J,TIJ_RVR,N_water).gt.0) then
                      TRVROUT(INM,N)=1d3*(TAIJN(I,J,TIJ_RVR,N)/(trw0(n)
     $                      *TAIJN(I,J,TIJ_RVR,N_water))-1.)
                    else
                      TRVROUT(INM,N)=undef
                    endif
                  else
                    TRVROUT(INM,N)=scale_tij(TIJ_RVR,n)*TAIJN(I,J
     $                    ,TIJ_RVR,N)/(AIJ(I,J,IJ_MRVR)*BYAXYP(I,J)
     $                    +teeny)
                  end if
                END DO
              ELSE
                TRVROUT(INM,:)=undef
              END IF
#endif
            end if
          END DO
        END DO
      END DO

C**** gather diags + print out on root processor
      rvrout_root=0.
      call sumxpe(rvrout, rvrout_root, increment=.true.)

      IF (AM_I_ROOT()) THEN
        DO INM=1,NRVR,6
          WRITE(out_line,901) (NAMERVR(I-1+INM),RVROUT_root(I-1+INM),I
     $          =1,MIN(6,NRVR+1-INM))
          CALL WRITE_PARALLEL(trim(out_line), UNIT=6)
        END DO
      END IF

#ifdef TRACERS_WATER
      DO N=1,NTM
        if (itime.ge.itime_tr0(n) .and. tr_wd_TYPE(n).eq.nWater) then
          rvrout_root=0.
          call sumxpe(trvrout(:,N), rvrout_root, increment=.true.)

          IF (AM_I_ROOT()) THEN
            WRITE(out_line,*) "River outflow tracer concentration "
     *            ,trim(units_tij(tij_rvr,n)),":",TRNAME(N)
            CALL WRITE_PARALLEL(trim(out_line), UNIT=6)
            DO INM=1,NRVR,6
              WRITE(out_line,901) (NAMERVR(I-1+INM)
     $              ,RVROUT_root(I-1+INM),I=1,MIN(6,NRVR+1-INM))
              CALL WRITE_PARALLEL(trim(out_line), UNIT=6)
            END DO
          END IF
        end if
      END DO
#endif

      RETURN
C****
 900  FORMAT ('1* River Outflow (km^3/mon) **  From:',I6,A6,I2,',  Hr'
     *     ,I3,6X,'To:',I6,A6,I2,', Hr',I3,'  Model-Time:',I9,5X
     *     ,'Dif:',F7.2,' Days')
 901  FORMAT (' ',A8,':',F8.3,5X,A8,':',F8.3,5X,A8,':',F8.3,5X,
     *            A8,':',F8.3,5X,A8,':',F8.3,5X,A8,':',F8.3)
      END SUBROUTINE diag_RIVER

      SUBROUTINE CHECKL (SUBR)
!@sum  CHECKL checks whether the lake variables are reasonable.
!@auth Gavin Schmidt/Gary Russell
      USE CONSTANT, only : rhow
      USE RESOLUTION, only : im,jm
      USE MODEL_COM, only : qcheck
      USE FLUXES, only : focean
      USE DOMAIN_DECOMP_ATM, only : getDomainBounds, GRID
      USE GEOM, only : axyp,imaxj
#ifdef TRACERS_WATER
      use OldTracer_mod, only: trname, t_qlimit
      USE TRACER_COM, only : NTM
#endif
      USE LAKES
      USE LAKES_COM
      IMPLICIT NONE
      INTEGER :: J_0,J_1,J_0H,J_1H,J_0S,J_1S,I_0,I_1,I_0H,I_1H,njpol
      INTEGER I,J,N !@var I,J loop variables
      CHARACTER*6, INTENT(IN) :: SUBR
      LOGICAL QCHECKL
#ifdef TRACERS_WATER
      integer :: imax,jmax
      real*8 relerr,errmax
#endif
      call getDomainBounds(grid, J_STRT=J_0,      J_STOP=J_1,
     *               J_STRT_HALO=J_0H,J_STOP_HALO=J_1H,
     &               J_STRT_SKP=J_0S, J_STOP_SKP=J_1S)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP
      I_0H = grid%I_STRT_HALO
      I_1H = grid%I_STOP_HALO
      njpol = grid%J_STRT_SKP-grid%J_STRT

C**** Check for NaN/INF in lake data
      CALL CHECK3B(MWL(I_0:I_1,J_0:J_1)  ,I_0,I_1,J_0,J_1,NJPOL,1,
     &     SUBR,'mwl')
      CALL CHECK3B(GML(I_0:I_1,J_0:J_1)  ,I_0,I_1,J_0,J_1,NJPOL,1,
     &     SUBR,'gml')
      CALL CHECK3B(MLDLK(I_0:I_1,J_0:J_1),I_0,I_1,J_0,J_1,NJPOL,1,
     &     SUBR,'mld')
      CALL CHECK3B(TLAKE(I_0:I_1,J_0:J_1),I_0,I_1,J_0,J_1,NJPOL,1,
     &     SUBR,'tlk')

      QCHECKL = .FALSE.
      DO J=J_0S, J_1S
      DO I=I_0, I_1
        IF(FOCEAN(I,J).eq.0.) THEN
C**** check for negative mass
          IF (MWL(I,J).lt.0 .or. MLDLK(I,J).lt.0) THEN
            WRITE(6,*) 'After ',SUBR,': I,J,TSL,MWL,GML,MLD=',
     *           I,J,TLAKE(I,J),MWL(I,J),GML(I,J),MLDLK(I,J)
            QCHECKL = .TRUE.
          END IF
C**** check for reasonable lake surface temps
          IF (TLAKE(I,J).ge.50 .or. TLAKE(I,J).lt.-0.5) THEN
            WRITE(6,*) 'After ',SUBR,': I,J,TSL=',I,J,TLAKE(I,J)
            if (TLAKE(I,J).lt.-5.and.FLAKE(I,J).gt.0) QCHECKL = .TRUE.
          END IF
        END IF
C**** Check total lake mass ( <0.4 m, >20x orig depth)
        IF(FLAKE(I,J).gt.0.) THEN
!!         IF(MWL(I,J).lt.0.4d0*RHOW*AXYP(I,J)*FLAKE(I,J)) THEN
!!           WRITE (6,*) 'After ',SUBR,
!!    *           ': I,J,FLAKE,DLAKE0,lake level low=',I,J,FLAKE(I,J),
!!    *           DLAKE0(I,J),MWL(I,J)/(RHOW*AXYP(I,J)*FLAKE(I,J))
!!         END IF
!          IF(MWL(I,J).gt.RHOW*MAX(20.*DLAKE0(I,J),3d1)*AXYP(I,J)*FLAKE(I,J)
           IF(MWL(I,J).gt.RHOW*(DLAKE0(I,J)+lake_rise_max)*AXYP(I,J)*
     *        FLAKE(I,J))THEN
            WRITE (6,*) 'After ',SUBR,
     *           ': I,J,FLAKE,DLAKE0,lake level high=',I,J,FLAKE(I,J),
     *           DLAKE0(I,J),MWL(I,J)/(RHOW*AXYP(I,J)*FLAKE(I,J))
          END IF
        END IF
      END DO
      END DO

#ifdef TRACERS_WATER
      do n=1,ntm
C**** Check for neg tracers in lake
        if (t_qlimit(n)) then
         do j=J_0, J_1
          do i=I_0,imaxj(j)
            if (focean(i,j).eq.0) then
              if (trlake(n,1,i,j).lt.0 .or. trlake(n,2,i,j).lt.0) then
                print*,"Neg tracer in lake after ",SUBR,i,j,trname(n)
     *               ,trlake(n,:,i,j)
                QCHECKL=.TRUE.
              end if
            end if
          end do
          end do
        end if
C**** Check conservation of water tracers in lake
        if (trname(n).eq.'Water') then
          errmax = 0. ; imax=I_0 ; jmax=J_0
          do j=J_0, J_1
          do i=I_0,imaxj(j)
            if (focean(i,j).eq.0) then
              if (flake(i,j).gt.0) then
                relerr=max(
     *               abs(trlake(n,1,i,j)-mldlk(i,j)*rhow*flake(i,j)
     *               *axyp(i,j))/trlake(n,1,i,j),abs(trlake(n,1,i,j)
     *               +trlake(n,2,i,j)-mwl(i,j))/(trlake(n,1,i,j)
     *               +trlake(n,2,i,j)))
              else
                if ((mwl(i,j).eq.0 .and. trlake(n,1,i,j)+trlake(n,2,i,j)
     *               .gt.0) .or. (mwl(i,j).gt.0 .and. trlake(n,1,i,j)
     *               +trlake(n,2,i,j).eq.0))  then
                  print*,"CHECKL ",SUBR,i,j,mwl(i,j),trlake(n,1:2,i,j)
                  relerr=0.
                else
                  if (mwl(i,j).gt.1d-20) then
                    relerr=abs(trlake(n,1,i,j)
     *                 +trlake(n,2,i,j)-mwl(i,j))/(trlake(n,1,i,j)
     *                 +trlake(n,2,i,j))
                  else
                    if (mwl(i,j).gt.0) print*,"CHECKL2 ",SUBR,i,j,mwl(i
     *                   ,j),trlake(n,1:2,i,j)
                    relerr=0.
                  end if
                end if
              end if
              if (relerr.gt.errmax) then
                imax=i ; jmax=j ; errmax=relerr
              end if
            end if
          end do
          end do
          print*,"Relative error in lake mass after ",trim(subr),":"
     *         ,imax,jmax,errmax,trlake(n,:,imax,jmax),mldlk(imax,jmax)
     *         *rhow*flake(imax,jmax)*axyp(imax,jmax),mwl(imax,jmax)
     *         -mldlk(imax,jmax)*rhow*flake(imax,jmax)*axyp(imax,jmax)
        end if
      end do
#endif

      IF (QCHECKL)
     &     call stop_model('CHECKL: Lake variables out of bounds',255)
      RETURN
C****
      END SUBROUTINE CHECKL



      SUBROUTINE daily_LAKE
!@sum  daily_LAKE does lake things at the beginning of every day
!@auth Gavin A. Schmidt, Gary L. Russell
!@ver  2022/03/31
      USE CONSTANT, only : rhow,by3,pi,lhm,shi,shw,teeny,tf
      USE RESOLUTION, only : im
      Use MODEL_COM,  Only: IWrite_Sv,JWrite_Sv
      USE LAKES, only : minmld,variable_lk,lake_ice_max,flake_cutoff
      USE LAKES, only : conical, C_lake, E_lake, small_lake_evap
      USE LAKES_COM, only : mwl,flake,tanlk,mldlk,tlake,T2Lbot,gml
     &     ,svflake,dlake0,dlake,glake
      USE SEAICE_COM, only : lakeice=>si_atm
      USE SEAICE, only : ace1i,xsi,ac2oim
      USE GEOM, only : axyp,imaxj,byaxyp
      USE GHY_COM, only : fearth
      USE FLUXES, only : atmocn,dmwldf,dgml
     &     ,fland,flice,focean
      USE LANDICE_COM, only : mdwnimp,edwnimp
      USE DIAG_COM, only : j_run,j_erun,jreg,j_implm
     *                    ,J_IMPLH, AIJ=>AIJ_LOC,itlkice,itlake,
     *                     IJ_MLKtoGR,IJ_HLKtoGR,IJ_IMPMKI,IJ_IMPHKI
      USE DOMAIN_DECOMP_ATM, only : getDomainBounds, GRID, HALO_UPDATE
      use CubicEquation_mod, only : cubicroot
      use model_com, only : calendar
      use Rational_mod

#ifdef TRACERS_WATER
      Use FLUXES,      Only: dtrl
      Use LAKES_COM,   Only: trlake,ntm
      Use LANDICE_COM, Only: trdwnimp
#endif
#ifdef IRRIGATION_ON
      USE IRRIGMOD, only : read_irrig
#endif  /* IRRIGATION_ON   */
#ifdef SCM
      USE SCM_COM, only : SCMopt,SCMin
#endif

      IMPLICIT NONE

      Integer :: i,j,N, J_0,J_1,I_0,I_1,jr,itm, N_ROOTS, n_iter
      Real*8  :: FLAKEnew,msinew,snownew,frac,fmsi2,fmsi3,
     *           fmsi4,fhsi2,fhsi3,fhsi4,imlt,hmlt,plake,plkic,
     *           frsat,FLAKEold,FEARTHold,
     *           mwsat, FRACI, FEpFLK, R, DLAKEnew,RHOWxVnew
      Real*8  :: A,B,C,D,Y,X(3)
      REAL*8, DIMENSION(:,:), POINTER :: RSI,MSI,SNOWI,GTEMP,GTEMPR
      REAL*8, DIMENSION(:,:,:), POINTER :: HSI
      real*8 :: CalendarSecondsPerDay

#ifdef TRACERS_WATER
      REAL*8, DIMENSION(:,:,:,:), POINTER :: TRSI
      REAL*8, DIMENSION(:,:,:), POINTER :: GTRACER
      Real*8 :: ftsi2(ntm),ftsi3(ntm),ftsi4(ntm),dtr(ntm),tottr(ntm)
#endif

      RSI => LAKEICE%RSI
      MSI => LAKEICE%MSI
      HSI => LAKEICE%HSI
      SNOWI => LAKEICE%SNOWI
      GTEMP => ATMOCN%GTEMP
      GTEMPR => ATMOCN%GTEMPR
#ifdef TRACERS_WATER
      TRSI => LAKEICE%TRSI
      GTRACER => ATMOCN%GTRACER
#endif

      call getDomainBounds(grid, J_STRT=J_0, J_STOP=J_1)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

      CalendarSecondsPerDay = real(calendar%getSecondsPerDay())

#ifdef IRRIGATION_ON
C**** Read potential irrigation daily
      call read_irrig(.true.)
#endif  /* IRRIGATION_ON   */

!****
C**** Update lake fraction as a function of lake mass at end of day
C****
      SVFLAKE=FLAKE  ! save for ghy purposes
      if (variable_lk == 0)  GoTo 700

      DO J=J_0,J_1  ;  DO I=I_0,IMAXJ(J)
          FEpFLK = 1 - FOCEAN(I,J) - FLICE(I,J)
          IF (FEpFLK == 0 .or. FOCEAN(I,J) > 0)  Cycle
          JR=JREG(I,J)
C**** Save original fractions
            FLAKEold=FLAKE(I,J); FEARTHold=FEARTH(I,J)
            PLAKE=FLAKE(I,J)*(1-RSI(I,J))
            PLKIC=FLAKE(I,J)*    RSI(I,J)
!****
!**** Compute FLAKEnew, assumes conical shape of lake, converted to cylinder later
!**** Change 1 from prior code: lake shape is based on liquid water only, not liquid plus ice
!**** Change 2 from prior code: FLAKEnew is not so small that lake ice must be compressed horizontally
!****
!**** MWL   = liquid lake mass (kg) = RHOW * A * DLAKE = RHOW * FLAKE*AXYP * DLAKE = RHOW * pi*R^2 * DLAKE
!**** TANLK = R0 / 3*DLAKE0 (specified from initial start) = R / 3*DLAKE
!**** DLAKE = mean liquid  depth of cone = depth of cylinder  ;  3*DLAKE = height of cone
!**** V = volume (m^3) = MWL / RHOW = pi * R^2 * DLAKE = pi * R^3 / 3*TANLK = pi * 9 * TANLK^2 * DLAKE^3
!**** R = radius of cone (m) = (3*TANLK*V/pi)^1/3 = (3*TANLK*MWL/pi*RHOW)^1/3 = (V/pi*DLAKE)^1/2 = (MWL/RHOW*pi*DLAKE)^1/2
!**** A = area of cone or cylinder (m^2) = pi*R^2 = pi*(3*TANLK*MWL / pi*RHOW)^2/3
!**** FLAKE = A / AXYP = pi*R^2 / AXYP(I,J) = lake area divided by cell area = MWL / (RHOW * AXYP * DLAKE)
         if (conical(i,j)) then
           R        = (3*TANLK(I,J)*MWL(I,J) / (pi*RHOW))**by3
           DLAKEnew =  R/(3*TANLK(I,J))
         else ! Cael's power law lake
           DLAKEnew = ( C_lake * (MWL(I,J)/RHOw)**(E_lake-1) )**1/E_lake
         end if
         if(small_lake_evap == 0 .and. DLAKEnew<minMLD) then
           FLAKEnew = 0
         else
           DLAKEnew = max (minMLD,DLAKEnew) ! if DLAKE == minMLD, Lake is viewed as cylinder of depth minMLD
           FLAKEnew = MWL(I,J) / (RHOW * AXYP(I,J) * DLAKEnew)
         end if

      If (I==IWrite_Sv .and. J==JWrite_Sv)  Write (0,9) 'DAILY_LAKE 1:',
     *   I,J,FLAKEold,FLAKEnew, DLAKEnew, TLAKE(I,J)

         If (FLAKEnew <= FLAKEold .or. DMWLDF(I,J) <= 0)  GoTo 50
!**** If FLAKEnew > FLAKEold, reduce FLAKEnew because lake water needs to saturate ground beneath new lake
!**** DMWLDF = water deficit (kg/m^2) over FEARTH fraction = saturated ground water - present ground water
!****
!**** Suppose DLAKEnew == minMLD ; compute FLAKEnew and Vnew , and compare RHOW*Vnew to MWL - (FLAKEnew-FLAKEold)*AXYP*DMWLDF
!**** FLAKEnew * RHOW*AXYP*minMLD = MWL - (FLAKEnew - FLAKEold) * AXYP*DMWLDF  =>
!**** =>  FLAKEnew * (RHOW*AXYP*minMLD + AXYP*DMWLDF) = MWL + FLAKEold*AXYP*DMWLDF
!**** Vnew = pi * 9 * TANLK^2 * minMLD^3
         FLAKEnew = (MWL(I,J) + FLAKEold*AXYP(I,J)*DMWLDF(I,J)) /
     /              (RHOW*AXYP(I,J)*minMLD + AXYP(I,J)*DMWLDF(I,J))
         if (conical(i,j)) then
           RHOWxVnew = RHOW * pi * 9 * TANLK(I,J)**2 * minMLD**3
         else ! Cael's power law lake
           RHOWxVnew = RHOW * (minMLD**E_lake / C_lake)**(1/(E_lake-1))
         end if
!**** If RHOW*Vnew is greater, then MWL is insufficient for cone shaped lake and shape is cylinder with depth minMLD
         If (RHOWxVnew >=
     *       MWL(I,J)-(FLAKEnew-FLAKEold)*AXYP(I,J)*DMWLDF(I,J)) GoTo 50
         if (conical(i,j)) then
!**** If MWL - (FLAKEnew-FLAKEold)*AXYP*DMWLDF exceeds RHOW*Vnew, then solve cubic polynomial:
!**** FLAKEnew = (pi*{3*TANLK*[MWL - (FLAKEnew-FLAKEold)*AXYP*DMWLDF] / pi*RHOW}^2/3) / AXYP  =>
!**** =>  (FLAKEnew*AXYP)^3/2 = pi^3/2 * 3*TANLK*[MWL+FLAKEold*AXYP*DMWLDF - FLAKEnew*AXYP*DMWLDF] / pi*RHOW  =>
!**** =>  FLAKEnew^3/2 * RHOW*AXYP^3/2 + FLAKEnew * 3*TANLK*AXYP*DMWLDF*pi^1/2 - 3*TANLK*(MWL+FLAKEold*AXYP*DMWLDF)*pi^1/2 = 0
!**** Y^3 * RHOW*AXYP^3/2 + Y^2 * 3*TANLK*AXYP*DMWLDF*pi^1/2 - 3*TANLK*(MWL+FLAKEold*AXYP*DMWLDF)*pi^1/2 = 0  where  Y^2 = FLAKEnew
            A = RHOW * AXYP(I,J)**1.5
            B = 3 * TANLK(I,J) * AXYP(I,J) * DMWLDF(I,J) * Sqrt(pi)
            C = 0
            D = - 3*TANLK(I,J) * Sqrt(pi) *
     *          (MWL(I,J) + FLAKEold * AXYP(I,J) * DMWLDF(I,J))
            call cubicroot(a,b,c,d,x,n_roots)
            if (n_roots<1) call stop_model("lakes: no solution",255)
            y = maxval( x(1:n_roots) )

            If (I==IWrite_Sv .and. J==JWrite_Sv)  Then
              C = (MWL(I,J) - (Y**2 - FLAKEold)*AXYP(I,J)*DMWLDF(I,J)) /
     /           (RHOW * AXYP(I,J))
              Write (0,9) 'DAILY_LAKE 2:', I,J,FLAKEold, Y**2,
     *                    (A*Y**3+B*Y**2+D)/B, C, C/Y**2  ;  EndIf
            FLAKEnew = Y**2
         else ! Power law lakes
!****       solve for the new lake area A=Anew the equation:
!****       MWLnew               + DMWLDF*A =  MWL + DMWLDF*Aold  i.e.
!***        c_lake*A^E_lake*RHOw + DMWLDF*A - (MWL + DMWLDF*Aold) = 0
              b = DMWLDF(I,J)/(RHOW*C_lake)
              c = (mwl(i,j)+FLAKEold*AXYP(I,J)*DMWLDF(I,J))/
     /             (RHOW*C_lake)
              call newton(A,b,c,E_lake,1.d-12,n_iter,i,j,axyp(i,j))
              FLAKEnew = A/AXYP(I,J)
cddd          write(678,*) "b,c,c/b", b,c,c/b
cddd          write(678,*) "A,err,n_iter",A, A**E_lake + b*A-c
cddd &                                         ,n_iter
         end if
         FLAKEnew = Min (FLAKEnew, MWL(I,J) / (RHOW*minMLD*AXYP(I,J))) ! Shallow Lakes are cylindrical with depth minMLD
C**** prevent confusion due to round-off errors
             If (FLAKEnew < FLAKEold + 1d-8)  Cycle

   50     FLAKEnew = Min (FLAKEnew, FEpFLK*.95d0)                  !  FLAKEnew limited to 95% of non-ice continent
          FLAKEnew = Min (FLAKEnew, FLAKEold + FEARTHold*.049d0)  !  do not flood more than 4.9% of land per day
          FLAKEnew = Max (FLAKEnew, PLKIC)                        !  FLAKEnew must be large enough to not compress ice
          If (FLAKEnew < FLAKE_cutoff)  FLAKEnew = 0

    9 Format (A,2I5,6F16.10)
      If (I==IWrite_Sv .and. J==JWrite_Sv)  Write (0,9) 'DAILY_LAKE 3:',
     *   I,J,FLAKEold,FLAKEnew, MWL(I,J)/(RHOW*AXYP(I,J))

!****
!**** FLAKEnew == FLAKEold
!****
         If (FLAKEnew == FLAKEold)  Cycle  !  end Do I, Do J

!****
!**** 0 == FLAKEnew < FLAKEold
!****
         If (0 == FLAKEnew .and. 0 < FLAKEold) Then
!**** remove/do not create lakes that are too small
!**** transfer lake ice mass/energy for accounting purposes; do not add ice mass to river - instead Use implicit array
            IMLT = ACE1I + MSI(I,J) + SNOWI(I,J)
            HMLT = Sum(HSI(:,I,J))
            MDWNIMP(I,J) = MDWNIMP(I,J) + PLKIC*IMLT*AXYP(I,J)
            EDWNIMP(I,J) = EDWNIMP(I,J) + PLKIC*HMLT*AXYP(I,J)
!           MWL(I,J)     = MWL(I,J) + PLKIC*IMLT*AXYP(I,J)
!           GML(I,J)     = GML(I,J) + PLKIC*HMLT*AXYP(I,J)
            RSI(I,J)     = 0
            SNOWI(I,J)   = 0
            HSI(1:2,I,J) = - LHM*XSI(1:2)*ACE1I
            HSI(3:4,I,J) = - LHM*XSI(3:4)*AC2OIM
            MSI(I,J)     = AC2OIM
            TLAKE(I,J)   = GML(I,J) / (SHW*MWL(I,J)+teeny)
            GTEMPR(I,J)  = TF
            MLDLK(I,J)   = minMLD
!**** Accumulate diagnostics
            AIJ(I,J,IJ_IMPMKI) = AIJ(I,J,IJ_IMPMKI) + PLKIC*IMLT
            AIJ(I,J,IJ_IMPHKI) = AIJ(I,J,IJ_IMPHKI) + PLKIC*HMLT
            Call INC_AJ (I,J,ITLKICE,J_IMPLM,PLKIC*IMLT)
            Call INC_AJ (I,J,ITLKICE,J_IMPLH,PLKIC*HMLT)
!           Call INC_AJ (I,J,ITLKICE,J_IMELT,PLKIC*IMLT)
!           Call INC_AJ (I,J,ITLKICE,J_HMELT,PLKIC*HMLT)
            Call INC_AREG (I,J,JR,J_IMPLM,PLKIC*IMLT)
            Call INC_AREG (I,J,JR,J_IMPLH,PLKIC*HMLT)
!           Call INC_AREG (I,J,JR,J_IMELT,PLKIC*IMLT)
!           Call INC_AREG (I,J,JR,J_HMELT,PLKIC*HMLT)

#ifdef TRACERS_WATER
            DO ITM=1,NTM
               TRLAKE(ITM,1,I,J) = SUM(TRLAKE(ITM,:,I,J))
               TRLAKE(ITM,2,I,J) = 0
               TRDWNIMP(ITM,I,J) = TRDWNIMP(ITM,I,J) +
     +                             Sum(TRSI(ITM,:,I,J))*PLKIC*AXYP(I,J)
               TRSI(ITM,:,I,J) = 0
            END DO
#endif
#ifdef SCM
            if (SCMopt%Tskin)  GTEMPR(I,J) = SCMin%Tskin
#endif

            GoTo 500
         EndIf  !  If (0 == FLAKEnew < FLAKEold)

!****
!**** 0 < FLAKEnew < FLAKEold
!****
!**** Mass of liquid lake water in upper and lower layers does not change ; lake layer temperatures do not change
!**** MLDnew*FLAKEnew = MLDold*FLAKEold  =>  MLDnew = MLDold*FLAKEold / FLAKEnew
         If (FLAKEnew < FLAKEold) Then
            MLDLK(I,J) = MLDLK(I,J) * FLAKEold / FLAKEnew
!**** Conserve lake ice
            If (PLKIC >= FLAKEnew)  Then  !  crunch ice up ?
!****          FLAKEnew = Max (FLAKEnew, PLKIC)  !  FLAKEnew must be large enough to not compress ice
!****          Crunch ice up, RSInew = 1
               RSI(I,J) = 1
!!             FRAC = PLKIC / FLAKEnew
!!             SNOWnew = SNOWI(I,J)*FRAC
!!             MSInew  = (MSI(I,J)+ACE1I)*FRAC - ACE1I
!!             FMSI3 = ACE1I*(FRAC-1)  !  kg/m^2 flux over new fraction
!!             FMSI2 = FMSI3*XSI(1)
!!             FMSI4 = FMSI3*XSI(4)
!!             FHSI2 = FMSI2*HSI(1,I,J) / (XSI(1)*(ACE1I+SNOWI(I,J)))
!!             If (FMSI3 < FRAC*XSI(2)*(ACE1I+SNOWI(I,J))) Then
!!                 FHSI3 = FMSI3*HSI(2,I,J) /(XSI(2)*(ACE1I+SNOWI(I,J)))
!!             Else
!!                 FHSI3 = HSI(2,I,J)*FRAC +
!!   +                (FMSI3-FRAC*XSI(2)*(ACE1I+SNOWI(I,J)))*HSI(1,I,J)
!!   /                / (XSI(1)*(ACE1I+SNOWI(I,J)))
!!             EndIf
!!             If (FMSI4 < FRAC*XSI(3)*MSI(I,J)) Then
!!                 FHSI4 = FMSI4*HSI(3,I,J) / (XSI(3)*MSI(I,J))
!!             Else
!!                 FHSI4 = HSI(3,I,J)*FRAC +
!!   +                     (FMSI4-FRAC*XSI(3)*MSI(I,J))*FHSI3 / FMSI3
!!             EndIf
!!             HSI(1,I,J) =HSI(1,I,J)*(ACE1I+SNOWNEW)/(ACE1I+SNOWI(I,J))
!!             HSI(2,I,J) = HSI(2,I,J)*FRAC + FHSI2 - FHSI3
!!             HSI(3,I,J) = HSI(3,I,J)*FRAC + FHSI3 - FHSI4
!!             HSI(4,I,J) = HSI(4,I,J)*FRAC + FHSI4

!**** all tracers --> tracer*FRAC, then adjust layering
!!#ifdef TRACERS_WATER
!!             FTSI2(:) =FMSI2*TRSI(:,1,I,J)/(XSI(1)*(ACE1I+SNOWI(I,J)))
!!             IF (FMSI3.LT.FRAC*XSI(2)*(ACE1I+SNOWI(I,J))) THEN
!!                 FTSI3(:) = FMSI3*TRSI(:,2,I,J) /
!!   /                        (XSI(2)*(ACE1I+SNOWI(I,J)))
!!             Else
!!                 FTSI3(:) = TRSI(:,2,I,J)*FRAC +
!!   +               (FMSI3-FRAC*XSI(2)*(ACE1I+SNOWI(I,J)))*TRSI(:,1,I,J)
!!   /               / (XSI(1)*(ACE1I+SNOWI(I,J)))
!!             EndIf
!!             IF (FMSI4.LT.FRAC*XSI(3)*MSI(I,J)) THEN
!!                 FTSI4(:) = FMSI4*TRSI(:,3,I,J) / (XSI(3)*MSI(I,J))
!!             Else
!!                 FTSI4(:) = TRSI(:,3,I,J)*FRAC +
!!   +                (FMSI4-FRAC*XSI(3)*MSI(I,J))*FTSI3(:) / FMSI3
!!             EndIf
!!             TRSI(:,1,I,J) = TRSI(:,1,I,J)*(ACE1I+SNOWNEW) /
!!   /                                       (ACE1I+SNOWI(I,J))
!!             TRSI(:,2,I,J) = TRSI(:,2,I,J)*FRAC + FTSI2(:) - FTSI3(:)
!!             TRSI(:,3,I,J) = TRSI(:,3,I,J)*FRAC + FTSI3(:) - FTSI4(:)
!!             TRSI(:,4,I,J) = TRSI(:,4,I,J)*FRAC + FTSI4(:)
!!#endif

!!             MSI(I,J)   = MSInew
!!             SNOWI(I,J) = SNOWnew
            Else
!****          Do not crunch ice up, instead increase RSI
               RSI(I,J) = PLKIC / FLAKEnew  !  = RSIold*FLAKEold / FLAKEnew
            EndIf  !  crunch ice up ?

            GoTo 500
         EndIf  !  If (0 < FLAKEnew < FLAKE)

!****
!**** FLAKEold < FLAKEnew
!****
!!!      If (FLAKEold < FLAKEnew) Then  !  If Then not needed, only case left
!**** If FLAKEold < FLAKEnew, then lake expands and ground beneath new lake must be saturated
!**** DMWLDF = water deficit over FEARTH fraction (kg/m^2) = saturated ground water - actual ground water
!**** MWSAT  = (FLAKEnew - FLAKEold) * AXYP * DMWLDF (kg) = water used to saturate ground beneath incresed FLAKE
!**** Mass of liquid in upper and lower lake layers are proportionally reduced ; lake temperatures do not change
!**** MLDnew*FLAKEnew / MLDold*FLAKEold = (MWL-MWSAT) / MWL = 1 - FRSAT  =>  MLDnew = (1 - FRSAT)*MLDold*FLAKEold/FLAKEnew
!**** Total lake ice and thickness is maintained:  RSInew = RSIold * FLAKEold / FLAKEnew = PLKIC / FLAKEnew
!**** MWLnew = MWLold - MWSAT
!**** DGML   = GML * MWSAT / MWL = energy (J) used to saturate ground beneath increased FLAKE
         MWSAT      = (FLAKEnew-FLAKEold)*AXYP(I,J)*DMWLDF(I,J)
         FRSAT      = MWSAT / MWL(I,J)
         MLDLK(I,J) = MLDLK(I,J)*(1-FRSAT)*FLAKEold / FLAKEnew
         RSI(I,J)   = PLKIC / FLAKEnew  !  = RSIold*FLAKEold / FLAKEnew
         If (DMWLDF(I,J) > 0)  Then
            MWL(I,J)   = MWL(I,J) - MWSAT
            DGML(I,J)  = GML(I,J) * FRSAT
            GML(I,J)   = GML(I,J) - DGML(I,J)
            Call INC_AJ (I,J,ITLAKE, J_RUN,
     *                   PLAKE*DMWLDF(I,J)*(FLAKEnew-FLAKEold))
            Call INC_AJ (I,J,ITLKICE,J_RUN,
     *                   PLKIC*DMWLDF(I,J)*(FLAKEnew-FLAKEold))
            Call INC_AJ (I,J,ITLAKE, J_ERUN,PLAKE*DGML(I,J)*byAXYP(I,J))
            Call INC_AJ (I,J,ITLKICE,J_ERUN,PLKIC*DGML(I,J)*byAXYP(I,J))
            AIJ(I,J,IJ_MLKtoGR) = AIJ(I,J,IJ_MLKtoGR) +
     +                            DMWLDF(I,J)*(FLAKEnew-FLAKEold)
     &                            / CalendarSecondsPerDay
            AIJ(I,J,IJ_HLKtoGR) = AIJ(I,J,IJ_HLKtoGR) +
     +                            DGML(I,J)*byAXYP(I,J)
     &                            / CalendarSecondsPerDay
#ifdef TRACERS_WATER
            DTRL(:,I,J)    = (TRLAKE(:,1,I,J)+TRLAKE(:,2,I,J)) * FRSAT
            TRLAKE(:,:,I,J) = TRLAKE(:,:,I,J) * (1-FRSAT)
#endif
         EndIf  !  If (DMWLDF > 0)

!**** Compute TLAKE and MLDLK for newly formed lake
         If (FLAKEold == 0) Then
            TLAKE(I,J)  = GML(I,J) / (MWL(I,J)*SHW + teeny)
            MLDLK(I,J)  = MWL(I,J) / (RHOW * FLAKEnew * AXYP(I,J))
            T2Lbot(I,J) = TLAKE(I,J)
         EndIf
!        GoTo 500

!**** Adjust land surface fractions
  500    FLAKE(I,J)  = FLAKEnew
         FLAND(I,J)  = 1 - FLAKEnew
         FEARTH(I,J) = FEpFLK - FLAKEnew

!**** Adjust radiative fluxes for conservation and restartability
C**** Complications due to ice or water going to earth if lake shrinks
            if (FLAKE(I,J).gt.FLAKEold) ! new lake from Earth frac
     *           call RESET_SURF_FLUXES(I,J,4,1,FLAKEold,FLAKE(I,J))
            if (FLAKEold.gt.FLAKE(I,J)) then ! lake shrinks
! originally some open water
              if (PLAKE.gt.0) call RESET_SURF_FLUXES(I,J,1,4,FEARTHold,
     *             FEARTHold+PLAKE-FLAKE(I,J)*(1-RSI(I,J)))
! originally some ice, now different
              if (PLKIC.gt.0 .and. PLKIC.ne.FLAKE(I,J)*RSI(I,J))
     *             call RESET_SURF_FLUXES(I,J,2,4,
     *             FEARTHold+PLAKE-FLAKE(I,J)*(1-RSI(I,J)),FEARTH(I,J))
            end if

C**** Set GTEMP array for lakes
         If (FLAKE(I,J) > 0) Then
            DLAKE(I,J)  = MWL(I,J)/(RHOW*FLAKE(I,J)*AXYP(I,J))
            GLAKE(I,J)  = GML(I,J)/(FLAKE(I,J)*AXYP(I,J))
            GTEMP(I,J)  = TLAKE(I,J)
            GTEMPR(I,J) = TLAKE(I,J)+TF
            atmocn%MLHC(I,J) = SHW*MLDLK(I,J)*RHOW
         Else
            DLAKE(I,J) = 0
            GLAKE(I,J) = 0
         EndIf

#ifdef SCM
         if (SCMopt%Tskin) then
            GTEMP(I,J) = SCMin%Tskin - TF
            GTEMPR(I,J) = SCMin%Tskin
         endif
#endif
#ifdef TRACERS_WATER
         GTRACER(:,I,J) = TRLAKE(:,1,I,J) /
     /                    (MLDLK(I,J)*RHOW*FLAKE(I,J)*AXYP(I,J))
#endif

!    3 FORMAT ('DAILY:',2I5,2F12.8,5F12.4)
!      IF(I==109.AND.J==67) WRITE (0,3) I,J,FLAKEold,FLAKEnew,
!     *  MLDLK(I,J),DLAKE(I,J),
!     *  MWL(I,J)/(RHOW*FLAKE(I,J)*AXYP(I,J)),TLAKE(I,J),T2LBOT(I,J)

      END DO  ;  END DO  !  end Do J= ; Do I=

!**** Update halo edges for next hour
      Call HALO_UPDATE (GRID, FLAKE)
      Call HALO_UPDATE (GRID, FLAND)
      Call HALO_UPDATE (GRID, FEARTH)

      CALL PRINTLK("DY")

!****
!**** Dump lake ice exceeding LAKE_ICE_MAX (m) into ice berg arrays
!****
  700 DO J=J_0,J_1  ;  DO I=I_0,IMAXJ(J)
         If (FLAKE(I,J) > 0 .and. RSI(I,J) > 0 .and.
     *       MSI(I,J) > LAKE_ICE_MAX * RHOW) Then
            IMLT  = MSI(I,J) - LAKE_ICE_MAX * RHOW
            FRACI = IMLT / MSI(I,J)
            HMLT  = Sum(HSI(3:4,I,J)) * FRACI
            PLKIC = FLAKE(I,J) * RSI(I,J)
            MDWNIMP(I,J) = MDWNIMP(I,J) + PLKIC*IMLT*AXYP(I,J)
            EDWNIMP(I,J) = EDWNIMP(I,J) + PLKIC*HMLT*AXYP(I,J)
            MSI(I,J)     = (1-FRACI)*MSI(I,J)  ! = LAKE_ICE_MAX * RHOW
            HSI(3:4,I,J) = (1-FRACI)*HSI(3:4,I,J)
!**** save some diags
            AIJ(I,J,IJ_IMPMKI) = AIJ(I,J,IJ_IMPMKI) + PLKIC*IMLT
            AIJ(I,J,IJ_IMPHKI) = AIJ(I,J,IJ_IMPHKI) + PLKIC*HMLT
            CALL INC_AJ(I,J,ITLKICE,J_IMPLM,PLKIC*IMLT)
            CALL INC_AJ(I,J,ITLKICE,J_IMPLH,PLKIC*HMLT)
!           CALL INC_AJ(I,J,ITLKICE,J_IMELT,PLKIC*IMLT)
!           CALL INC_AJ(I,J,ITLKICE,J_HMELT,PLKIC*HMLT)
!**** Accumulate regional diagnostics
            JR = JREG(I,J)
!           CALL INC_AREG(I,J,JR,J_IMELT,PLKIC*IMLT)
!           CALL INC_AREG(I,J,JR,J_HMELT,PLKIC*HMLT)
            CALL INC_AREG(I,J,JR,J_IMPLM,PLKIC*IMLT)
            CALL INC_AREG(I,J,JR,J_IMPLH,PLKIC*HMLT)

#ifdef TRACERS_WATER
            DO ITM=1,NTM
               TRDWNIMP(ITM,I,J) = TRDWNIMP(ITM,I,J)
     +            + Sum(TRSI(ITM,3:4,I,J))*PLKIC*AXYP(I,J)*FRACI
               TRSI(ITM,3:4,I,J) = TRSI(ITM,3:4,I,J) * (1-FRACI)
            END DO
#endif

         EndIf  !  end If (FLAKE > 0 & RSI > 0 & MSI > LAKE_RISE_MAX
      END DO  ;  END DO  !  end Do J= ; Do I=
      END SUBROUTINE daily_LAKE



      SUBROUTINE PRECIP_LK
!@sum  PRECIP_LK driver for applying precipitation/melt to lake fraction
!@auth Gavin Schmidt
      USE CONSTANT, only : rhow,shw,teeny,tf
      USE RESOLUTION, only : im,jm
#ifdef SCM
      USE SCM_COM, only : SCMopt,SCMin
#endif
      USE DOMAIN_DECOMP_ATM, only : GRID,getDomainBounds
      USE GEOM, only : imaxj,axyp,byaxyp
      USE SEAICE_COM, only : lakeice=>si_atm
      USE LAKES_COM, only : mwl,gml,tlake,mldlk,flake,dlake,glake
     *     ,icelak
#ifdef TRACERS_WATER
     *     ,trlake,ntm
#endif
      USE FLUXES, only : atmocn,atmgla,prec,eprec,flice
#ifdef TRACERS_WATER
     *     ,trprec
#endif
      USE DIAG_COM, only : aj=>aj_loc,j_run,aij=>aij_loc,ij_lk
     &     ,itlake,itlkice
      IMPLICIT NONE

      REAL*8 PRCP,ENRGP,PLICE,PLKICE,RUN0,ERUN0,POLAKE,HLK1
      INTEGER :: J_0,J_1,J_0H,J_1H,J_0S,J_1S,I_0H,I_1H,I_0,I_1
      INTEGER I,J,ITYPE
#ifdef TRACERS_WATER
      REAL*8, DIMENSION(NTM) :: TRUN0
#endif

      REAL*8, DIMENSION(:,:), POINTER :: RSI,GTEMP,GTEMP2,GTEMPR,
     &     RUNPSI,MELTI,EMELTI
#ifdef TRACERS_WATER
      REAL*8, DIMENSION(:,:,:), POINTER :: GTRACER,TRUNPSI,TRMELTI
#endif

      RSI => LAKEICE%RSI
      GTEMP => ATMOCN%GTEMP
      GTEMP2 => ATMOCN%GTEMP2
      GTEMPR => ATMOCN%GTEMPR
      RUNPSI => ICELAK%RUNPSI
      MELTI => ICELAK%MELTI
      EMELTI => ICELAK%EMELTI
#ifdef TRACERS_WATER
      GTRACER => ATMOCN%GTRACER
      TRUNPSI => ICELAK%TRUNPSI
      TRMELTI => ICELAK%TRMELTI
#endif

      call getDomainBounds(grid, J_STRT=J_0,      J_STOP=J_1,
     &               J_STRT_SKP=J_0S, J_STOP_SKP=J_1S)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

      CALL PRINTLK("PR")

      DO J=J_0, J_1
      DO I=I_0,IMAXJ(J)
      IF (FLAKE(I,J)+FLICE(I,J).gt.0) THEN
        POLAKE=(1-RSI(I,J))*FLAKE(I,J)
        PLKICE=RSI(I,J)*FLAKE(I,J)
        PLICE=FLICE(I,J)
        PRCP=PREC(I,J)
        ENRGP=EPREC(I,J)        ! energy of precipitation

C**** calculate fluxes over whole box
        RUN0 =POLAKE*PRCP  + PLKICE* RUNPSI(I,J) +
     &       PLICE*atmgla%RUNO(I,J)
        ERUN0=POLAKE*ENRGP ! PLKICE*ERUNPSI(I,J) + PLICE*ERUNOLI(I,J) =0

C**** simelt is given as kg/area
        IF (FLAKE(I,J).gt.0) THEN
          RUN0  =RUN0+ MELTI(I,J)
          ERUN0=ERUN0+EMELTI(I,J)
        END IF

        MWL(I,J) = MWL(I,J) +  RUN0*AXYP(I,J)
        GML(I,J) = GML(I,J) + ERUN0*AXYP(I,J)
#ifdef TRACERS_WATER
        TRUN0(:) = POLAKE*TRPREC(:,I,J)
     *       + PLKICE*TRUNPSI(:,I,J) + PLICE *atmgla%TRUNO(:,I,J)
        IF(FLAKE(I,J).gt.0) TRUN0(:)=TRUN0(:)+TRMELTI(:,I,J)
        TRLAKE(:,1,I,J)=TRLAKE(:,1,I,J) + TRUN0(:)*AXYP(I,J)
#endif

        IF (FLAKE(I,J).gt.0) THEN
          HLK1=TLAKE(I,J)*MLDLK(I,J)*RHOW*SHW
          MLDLK(I,J)=MLDLK(I,J) + RUN0/(FLAKE(I,J)*RHOW)
          TLAKE(I,J)=(HLK1*FLAKE(I,J)+ERUN0)/(MLDLK(I,J)*FLAKE(I,J)
     *         *RHOW*SHW)
          DLAKE(I,J)=MWL(I,J)/(RHOW*FLAKE(I,J)*AXYP(I,J))
          GLAKE(I,J)=GML(I,J)/(FLAKE(I,J)*AXYP(I,J))
          GTEMP(I,J)=TLAKE(I,J)
          GTEMPR(I,J) =TLAKE(I,J)+TF
#ifdef SCM
          if (SCMopt%Tskin) then
            GTEMP(I,J) = SCMin%Tskin - TF
            GTEMPR(I,J) = SCMin%Tskin
          endif
#endif
          IF (MWL(I,J).gt.(1d-10+MLDLK(I,J))*RHOW*FLAKE(I,J)*AXYP(I,J))
     *         THEN
            GTEMP2(I,J)=(GML(I,J)-TLAKE(I,J)*SHW*MLDLK(I,J)*RHOW
     *           *FLAKE(I,J)*AXYP(I,J))/(SHW*(MWL(I,J)-MLDLK(I,J)
     *           *RHOW*FLAKE(I,J)*AXYP(I,J)))
          ELSE
            GTEMP2(I,J)=TLAKE(I,J)
          END IF
#ifdef SCM
          if (SCMopt%Tskin) then
            GTEMP2(I,J) = GTEMP(I,J)
          endif
#endif
#ifdef TRACERS_WATER
          GTRACER(:,I,J)=TRLAKE(:,1,I,J)/(MLDLK(I,J)*RHOW*FLAKE(I,J)
     *         *AXYP(I,J))
#endif
          CALL INC_AJ(I,J,ITLAKE,J_RUN,
     &         -PLICE*atmgla%RUNO(I,J)*(1-RSI(I,J)))
          CALL INC_AJ(I,J,ITLKICE,J_RUN,
     &         -PLICE*atmgla%RUNO(I,J)   *RSI(I,J))
        ELSE
          TLAKE(I,J)=GML(I,J)/(MWL(I,J)*SHW+teeny)
          DLAKE(I,J)=0.
          GLAKE(I,J)=0.
C**** accounting fix to ensure runoff with no lakes is counted
C**** no regional diagnostics required
          CALL INC_AJ(I,J,ITLAKE,J_RUN,-PLICE*atmgla%RUNO(I,J))
        END IF

C**** save area diag
        AIJ(I,J,IJ_LK) = AIJ(I,J,IJ_LK) + FLAKE(I,J)
      END IF
      END DO
      END DO
      RETURN
C****
      END SUBROUTINE PRECIP_LK

#ifdef IRRIGATION_ON
      SUBROUTINE IRRIG_LK
!@sum  IRRIG_LK driver for calculating irrigation fluxes from lakes/rivers
!@auth Gavin Schmidt
      USE CONSTANT, only : rhow,shw,teeny
      USE RESOLUTION, only : im,jm
      USE DOMAIN_DECOMP_ATM, only : GRID, getDomainBounds
      USE GEOM, only : imaxj,axyp,byaxyp
      USE DIAG_COM, only : itearth,jreg,aij=>aij_loc,ij_mwlir
     *     ,ij_gmlir,ij_irrgw,ij_irrgwE,j_irgw,j_irgwE
      USE LAKES_COM, only : mwl,gml,tlake,mldlk,flake
#ifdef TRACERS_WATER
     *     ,trlake,ntm
#endif
      USE LAKES, only : minmld
      USE IRRIGMOD, only : irrigate_extract
      USE FLUXES,only : fland,irrig_water_act, irrig_energy_act
#ifdef TRACERS_WATER
     *     ,irrig_tracer_act
#endif
      USE TimerPackage_mod, only: startTimer => start
      USE TimerPackage_mod, only: stopTimer => stop
      IMPLICIT NONE
C**** grid box variables
      REAL*8 M1,M2,E1,E2,DM,DE
      REAL*8 :: MWL_to_irrig,GML_to_irrig,irrig_gw,irrig_gw_energy
     *     ,irrig_water_actij,irrig_energy_actij
#ifdef TRACERS_WATER
     *     ,TRML_to_irrig(NTM,2),TRML_temp(NTM,2)
     *     ,irrig_tracer_actij(ntm),irrig_gw_tracer(ntm)
#endif
      INTEGER I,J,JR
      INTEGER :: J_0,J_1,J_0S,J_1S,I_0,I_1

      call startTimer('PRECIP_LK()')
      call getDomainBounds(grid, J_STRT=J_0,      J_STOP=J_1,
     &               J_STRT_SKP=J_0S, J_STOP_SKP=J_1S)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

      CALL PRINTLK("IR")

      DO J=J_0, J_1
      DO I=I_0,IMAXJ(J)
      JR=JREG(I,J)

C**** Remove mass/energy associated with irrigation
      IF (FLAND(I,J).gt.0) THEN

#ifdef TRACERS_WATER
        TRML_temp(:,:) = TRLAKE(:,:,I,J)
#endif
C****   Compute actual irrigation every timestep
        call irrigate_extract(I,J,MWL(I,J),GML(I,J),MLDLK(I,J),TLAKE(I
     *       ,J),FLAKE(I,J),minmld,MWL_to_irrig,GML_to_irrig,irrig_gw
     *       ,irrig_gw_energy,irrig_water_actij,irrig_energy_actij
#ifdef TRACERS_WATER
     *       ,TRML_temp,TRML_to_irrig,irrig_tracer_actij,irrig_gw_tracer
#endif
     *       )
C**** save fluxes for GHY (m/s), (J/s), (kg/s)
        irrig_water_act(i,j) =irrig_water_actij
        irrig_energy_act(i,j)=irrig_energy_actij
#ifdef TRACERS_WATER
        irrig_tracer_act(:,i,j)=irrig_tracer_actij(:)
#endif

        IF (MWL_to_irrig .gt. 0) THEN
C**** update lake mass/energy
        MWL(I,J) = MWL(I,J) - MWL_to_irrig
        GML(I,J) = GML(I,J) - GML_to_irrig
#ifdef TRACERS_WATER
        TRLAKE(:,:,I,J) = TRLAKE(:,:,I,J) - TRML_to_irrig(:,:)
        IF (MWL(I,J).eq.0) TRLAKE(:,:,I,J)=0.  ! round off issues
#endif

!!! mixed layer depth and surface temperature adjustments for lakes
!!        if (FLAKE(I,J).gt.0) THEN  !  ***** UNECESSARY IF *****
!!          if (MWL_to_irrig.lt.MLDLK(I,J)*FLAKE(I,J)*AXYP(I,J)*RHOW) then ! layer 1 only
!!            MLDLK(I,J)=MLDLK(I,J)-MWL_to_irrig/(FLAKE(I,j)*AXYP(I,J)
!!     *           *RHOW)
!!            M1=MLDLK(I,J)*RHOW*FLAKE(I,J)*AXYP(I,J) ! kg
!!            M2=max(MWL(I,J)-M1,0d0)
!!            if (MLDLK(I,J).LT.MINMLD .and. M2.gt.0) THEN ! bring up from layer 2
!!              E1=TLAKE(I,J)*SHW*M1
!!              E2=GML(I,J)-E1
!!              DM=max(MINMLD*RHOW*FLAKE(I,J)*AXYP(I,J)-M1,0d0) ! kg
!!              DE=DM*E2/(M2+teeny)
!!              TLAKE(I,J)=(E1+DE)/((M1+DM)*SHW) ! deg C
!!#ifdef TRACERS_WATER
!!              TRLAKE(:,1,I,J)=TRLAKE(:,1,I,J)+DM*TRLAKE(:,2,I,J)/
!!     *             (M2+teeny)
!!              TRLAKE(:,2,I,J)=TRLAKE(:,2,I,J)-DM*TRLAKE(:,2,I,J)/
!!     *             (M2+teeny)
!!#endif
!!              MLDLK(I,J) = MLDLK(I,J) + DM/(FLAKE(I,j)*AXYP(I,J)*RHOW)
!!            end if
!!          else ! all layer 1 and some layer 2 gone, relayer
!!            MLDLK(I,J)=MWL(I,J)/(FLAKE(I,J)*AXYP(I,J)*RHOW)
!!            TLAKE(I,J)=GML(I,J)/(MWL(I,J)*SHW+teeny)
!!#ifdef TRACERS_WATER
!!            TRLAKE(:,1,I,J)=TRLAKE(:,1,I,J)+TRLAKE(:,2,I,J)
!!            TRLAKE(:,2,I,J)=0.
!!#endif
!!          end if  !  if (MWL_to_irrig.lt.MLDLK(I,J)*FLAKE(I,J)*AXYP(I,J)*RHOW) then ! layer 1 only
!!        end if  !  if (FLAKE(I,J).gt.0) THEN  !  ***** UNECESSARY IF *****

C****   Compute lake- and irrigation-related diagnostics
        AIJ(I,J,IJ_MWLir)=AIJ(I,J,IJ_MWLir)+MWL_to_irrig*byaxyp(i,j)
        AIJ(I,J,IJ_GMLir)=AIJ(I,J,IJ_GMLir)+GML_to_irrig*byaxyp(i,j)

        END IF ! MWL_to_irrig .gt. 0

        AIJ(I,J,IJ_irrgw) =AIJ(I,J,IJ_irrgw) +irrig_gw
        AIJ(I,J,IJ_irrgwE)=AIJ(I,J,IJ_irrgwE)+irrig_gw_energy

        CALL INC_AJ(I,J,itearth, j_irgw , irrig_gw)
        CALL INC_AJ(I,J,itearth, j_irgwE, irrig_gw_energy)

      END IF ! FLAND(I,J).gt.0

      END DO  ! i loop
      END DO  ! j loop

      CALL PRINTLK("I2")

      call stopTimer('PRECIP_LK()')
      RETURN
C****
      END SUBROUTINE IRRIG_LK
#endif


      SUBROUTINE GROUND_LK
!@sum  GROUND_LK driver for applying surface fluxes to lake fraction
!@auth Gavin Schmidt
!@calls
      USE CONSTANT, only : rhow,shw,teeny,tf
      USE RESOLUTION, only : im,jm
      USE MODEL_COM, only : dtsrc, IWrite_Sv,JWrite_Sv
#ifdef SCM
      USE SCM_COM, only : SCMopt,SCMin
#endif
      USE DOMAIN_DECOMP_ATM, only : GRID, getDomainBounds

      USE GEOM, only : imaxj,axyp,byaxyp
      USE FLUXES, only : atmocn,atmgla,atmlnd,flice,fland
      USE SEAICE_COM, only : lakeice=>si_atm
      USE DIAG_COM, only : jreg,j_wtr1,j_wtr2,j_run,j_erun,ij_geotherm
     *     ,aij=>aij_loc,ij_mwl,ij_gml,itlake,itlkice,itearth
      USE LAKES_COM, only : icelak,mwl,gml,tlake,mldlk,flake,dlake0
     *                     ,T2Lbot,EKT, DLAKE
#ifdef TRACERS_WATER
     *     ,trlake,ntm
      USE TRDIAG_COM,only: taijn=>taijn_loc , tij_lk1,tij_lk2
#endif
      USE LAKES, only : byzeta,minmld
      USE GHY_COM, only : fearth, fgeotherm
      USE TimerPackage_mod, only: startTimer => start
      USE TimerPackage_mod, only: stopTimer => stop
      IMPLICIT NONE
C**** grid box variables
      REAL*8 ROICE, POLAKE, PLKICE, PEARTH, PLICE
!@var MLAKE,ELAKE mass and energy /m^2 for lake model layers
      REAL*8, DIMENSION(2) :: MLAKE,ELAKE
C**** fluxes
      REAL*8 EVAPO, FIDT, FODT, RUN0, ERUN0, RUNLI, RUNE, ERUNE,
     *     HLK1,TLK1,TLK2,TKE,SROX(2),FSR2,Egeoth  ! , U2RHO
C**** output from LKSOURC
      REAL*8 ENRGFO, ACEFO, ACEFI, ENRGFI
#ifdef TRACERS_WATER
      REAL*8, DIMENSION(NTM) :: TRUN0,TRO,TRI,TREVAP,TOTTRL
      REAL*8, DIMENSION(NTM,2) :: TRLAKEL
#endif
      INTEGER I,J,JR
      INTEGER :: J_0,J_1,J_0S,J_1S,I_0,I_1

      REAL*8, DIMENSION(:,:), POINTER :: RSI,GTEMP,GTEMP2,GTEMPR,
     &     RUNOSI,ERUNOSI,EVAPOR,E0
      REAL*8, DIMENSION(:,:,:), POINTER :: DMSI,DHSI,DSSI
#ifdef TRACERS_WATER
      REAL*8, DIMENSION(:,:,:,:), POINTER :: DTRSI
      REAL*8, DIMENSION(:,:,:), POINTER :: GTRACER,TREVAPOR,TRUNOSI
#ifdef TRACERS_DRYDEP
      REAL*8, DIMENSION(:,:,:), POINTER :: TRDRYDEP
#endif
#endif

      RSI => LAKEICE%RSI
      E0 => ATMOCN%E0
      EVAPOR => ATMOCN%EVAPOR
      GTEMP => ATMOCN%GTEMP
      GTEMP2 => ATMOCN%GTEMP2
      GTEMPR => ATMOCN%GTEMPR
#ifdef TRACERS_WATER
      TREVAPOR => ATMOCN%TREVAPOR
#ifdef TRACERS_DRYDEP
      TRDRYDEP => ATMOCN%TRDRYDEP
#endif
      GTRACER => ATMOCN%GTRACER
#endif
      RUNOSI => ICELAK%RUNOSI
      ERUNOSI => ICELAK%ERUNOSI
      DMSI => ICELAK%DMSI
      DHSI => ICELAK%DHSI
      DSSI => ICELAK%DSSI
#ifdef TRACERS_WATER
      TRUNOSI => ICELAK%TRUNOSI
      DTRSI => ICELAK%DTRSI
#endif

      call startTimer('GROUND_LK()')
      call getDomainBounds(grid, J_STRT=J_0,      J_STOP=J_1,
     &               J_STRT_SKP=J_0S, J_STOP_SKP=J_1S)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

      CALL PRINTLK("GR")

      DO J=J_0, J_1
      DO I=I_0,IMAXJ(J)
      JR=JREG(I,J)
      ROICE=RSI(I,J)
      PLKICE=FLAKE(I,J)*ROICE
      POLAKE=FLAKE(I,J)*(1-ROICE)

   9  Format (A,2I5,5F16.10)
      If (I==IWrite_Sv .and. J==JWrite_Sv)  Write (0,9)
      If (I==IWrite_Sv .and. J==JWrite_Sv)  Write (0,9) 'GROUND_LK 1:',
     *    I,J,FLAKE(I,J),MLDLK(I,J),DLAKE(I,J),TLAKE(I,J),T2Lbot(I,J)

C**** Add land ice and surface runoff to lake variables
      IF (FLAND(I,J).gt.0) THEN
        PLICE =FLICE(I,J)
        PEARTH=FEARTH(I,J)
        RUNLI=atmgla%RUNO(I,J)
        RUNE =atmlnd%RUNO(I,J)
        ERUNE=atmlnd%ERUNO(I,J)
C**** calculate flux over whole box
        RUN0 =RUNLI*PLICE + RUNE*PEARTH
        ERUN0=             ERUNE*PEARTH
        Egeoth = fgeotherm(i,j)*FLAKE(I,J)*dtsrc
        MWL(I,J) = MWL(I,J) + RUN0*AXYP(I,J)
        GML(I,J) = GML(I,J) +ERUN0*AXYP(I,J)
!**** Also add the geothermal heat flux
        GML(I,J) = GML(I,J) + Egeoth*AXYP(I,J)
#ifdef TRACERS_WATER
        TRLAKE(:,1,I,J)=TRLAKE(:,1,I,J)+
     *      (atmgla%TRUNO(:,I,J)*PLICE
     *      +atmlnd%TRUNO(:,I,J)*PEARTH)*AXYP(I,J)
#endif

        AIJ(I,J,IJ_MWL)=AIJ(I,J,IJ_MWL)+MWL(I,J)*byaxyp(i,j)
        AIJ(I,J,IJ_GML)=AIJ(I,J,IJ_GML)+GML(I,J)*byaxyp(i,j)

        IF (FLAKE(I,J).gt.0) THEN
          HLK1=TLAKE(I,J)*MLDLK(I,J)*RHOW*SHW
          MLDLK(I,J)=MLDLK(I,J) + RUN0/(FLAKE(I,J)*RHOW)
          TLAKE(I,J)=(HLK1*FLAKE(I,J)+ERUN0+Egeoth)/
     *         (MLDLK(I,J)*FLAKE(I,J)*RHOW*SHW)
#ifdef TRACERS_WATER
          GTRACER(:,I,J)=TRLAKE(:,1,I,J)/(MLDLK(I,J)*RHOW*FLAKE(I,J)
     *         *AXYP(I,J))
#endif
          CALL INC_AJ(I,J,ITLAKE ,J_RUN ,-(RUNE*PEARTH+RUNLI*PLICE)
     *         *(1-RSI(I,J)))
          CALL INC_AJ(I,J,ITLKICE,J_RUN ,-(RUNE*PEARTH+RUNLI*PLICE)
     *         *    RSI(I,J))
          CALL INC_AJ(I,J,ITLAKE ,J_ERUN,-ERUNE*PEARTH*(1-RSI(I,J)))
          CALL INC_AJ(I,J,ITLKICE,J_ERUN,-ERUNE*PEARTH*    RSI(I,J))
          AIJ(I,J,IJ_Geotherm)=AIJ(I,J,IJ_Geotherm)+
     *                                     fgeotherm(i,j)*FLAKE(I,J)
        ELSE
          TLAKE(I,J)=GML(I,J)/(MWL(I,J)*SHW+teeny)
C**** accounting fix to ensure runoff with no lakes is counted
C**** no regional diagnostics required
          CALL INC_AJ(I,J,ITLAKE,J_RUN, -(RUNE*PEARTH+RUNLI*PLICE))
          CALL INC_AJ(I,J,ITLAKE,J_ERUN,-ERUNE*PEARTH)
        END IF
      END IF

      IF (FLAKE(I,J).gt.0) THEN
        TLK1 =TLAKE(I,J)
        EVAPO=EVAPOR(I,J)     ! evap/dew over open lake (kg/m^2)
        FODT =E0(I,J)         ! net heat over open lake (J/m^2)
        SROX(1)=atmocn%SOLAR(I,J)      ! solar radiation open lake (J/m^2)
        SROX(2)=icelak%SOLAR(I,J)      ! solar radiation through ice (J/m^2)
        FSR2 =EXP(-MLDLK(I,J)*BYZETA)
C**** get ice-lake fluxes from sea ice routine (over ice fraction)
        RUN0 =RUNOSI(I,J) ! includes ACE2M + basal term
        FIDT =ERUNOSI(I,J)
C**** calculate kg/m^2, J/m^2 from saved variables
        MLDLK(I,J) = Min (MLDLK(I,J),
     *                    MWL(I,J)/(FLAKE(I,J)*AXYP(I,J)*RHOW))
        MLAKE(1)=MLDLK(I,J)*RHOW
        MLAKE(2)=MAX(MWL(I,J)/(FLAKE(I,J)*AXYP(I,J))-MLAKE(1),0d0)
        ELAKE(1)=TLK1*SHW*MLAKE(1)
        ELAKE(2)=GML(I,J)/(FLAKE(I,J)*AXYP(I,J))-ELAKE(1)
#ifdef TRACERS_WATER
        TRLAKEL(:,:)=TRLAKE(:,:,I,J)/(FLAKE(I,J)*AXYP(I,J))
        TRUN0(:)=TRUNOSI(:,I,J)
        TREVAP(:)=TREVAPOR(:,I,J)
#ifdef TRACERS_DRYDEP
     *       -trdrydep(:,i,j)
#endif
#endif
        IF (MLAKE(2).lt.1d-10) THEN
          MLAKE(1)=MLAKE(1)+MLAKE(2)
          MLAKE(2)=0.
          ELAKE(1)=ELAKE(1)+ELAKE(2)
          ELAKE(2)=0.
#ifdef TRACERS_WATER
          TRLAKEL(:,1)=TRLAKEL(:,1)+TRLAKEL(:,2)
          TRLAKEL(:,2)=0.
#endif
        END IF

C**** Limit FSR2 in the case of thin second layer
        FSR2=MIN(FSR2,MLAKE(2)/(MLAKE(1)+MLAKE(2)))

      If (I==IWrite_Sv .and. J==JWrite_Sv)  Write (0,9) 'GROUND_LK 2:',
     *    I,J,FLAKE(I,J),MLAKE(:)/RHOW,TLAKE(I,J),T2Lbot(I,J)

C**** Apply fluxes and calculate the amount of frazil ice formation
        CALL LKSOURC (I,J,ROICE,MLAKE,ELAKE,RUN0,FODT,FIDT,SROX,FSR2,
     *                T2Lbot(I,J),EKT(I,J),FLAKE(I,J),
#ifdef TRACERS_WATER
     *       TRLAKEL,TRUN0,TREVAP,TRO,TRI,
#endif
     *       EVAPO,ENRGFO,ACEFO,ACEFI,ENRGFI)

C**** Resave prognostic variables
        MWL(I,J)  =(MLAKE(1)+MLAKE(2))*(FLAKE(I,J)*AXYP(I,J))
        GML(I,J)  =(ELAKE(1)+ELAKE(2))*(FLAKE(I,J)*AXYP(I,J))
        MLDLK(I,J)= MLAKE(1)/RHOW
        TLAKE(I,J)= ELAKE(1)/(SHW*MLAKE(1))
        IF (MLAKE(2).gt.0) THEN
          TLK2    = ELAKE(2)/(SHW*MLAKE(2))
        ELSE
          TLK2    = TLAKE(I,J)
        END IF

      If (I==IWrite_Sv .and. J==JWrite_Sv)  Write (0,9) 'GROUND_LK 3:',
     *    I,J,FLAKE(I,J),MLAKE(:)/RHOW,TLAKE(I,J),T2Lbot(I,J)

#ifdef TRACERS_WATER
        TRLAKE(:,:,I,J)=TRLAKEL(:,:)*(FLAKE(I,J)*AXYP(I,J))
        GTRACER(:,I,J)=TRLAKEL(:,1)/(MLDLK(I,J)*RHOW)
#endif
        GTEMP(I,J)=TLAKE(I,J)
        GTEMP2(I,J)=TLK2       ! diagnostic only
        GTEMPR(I,J) =TLAKE(I,J)+TF
#ifdef SCM
        if (SCMopt%Tskin) then
          GTEMP(I,J) = SCMin%Tskin - TF
          GTEMP2(I,J) = SCMin%Tskin - TF
          GTEMPR(I,J) = SCMin%Tskin
        endif
#endif
C**** Open lake diagnostics
        CALL INC_AJ(I,J, ITLAKE,J_WTR1,MLAKE(1)*POLAKE)
        CALL INC_AJ(I,J, ITLAKE,J_WTR2,MLAKE(2)*POLAKE)
C**** Ice-covered ocean diagnostics
        CALL INC_AJ(I,J, ITLKICE,J_WTR1,MLAKE(1)*PLKICE)
        CALL INC_AJ(I,J, ITLKICE,J_WTR2,MLAKE(2)*PLKICE)
C**** regional diags
        CALL INC_AREG(I,J,JR,J_WTR1,MLAKE(1)*FLAKE(I,J))
        CALL INC_AREG(I,J,JR,J_WTR2,MLAKE(2)*FLAKE(I,J))
#ifdef TRACERS_WATER
C**** tracer diagnostics
        TAIJN(I,J,tij_lk1,:)=TAIJN(I,J,tij_lk1,:)+TRLAKEL(:,1) !*PLKICE?
        TAIJN(I,J,tij_lk2,:)=TAIJN(I,J,tij_lk2,:)+TRLAKEL(:,2) !*PLKICE?
#endif

C**** Store mass and energy fluxes for formation of sea ice
        DMSI(1,I,J)=ACEFO
        DMSI(2,I,J)=ACEFI
        DHSI(1,I,J)=ENRGFO
        DHSI(2,I,J)=ENRGFI
        DSSI(:,I,J)=0.     ! always zero salinity
#ifdef TRACERS_WATER
        DTRSI(:,1,I,J)=TRO(:)
        DTRSI(:,2,I,J)=TRI(:)
#endif
      END IF
      END DO  ! i loop
      END DO  ! j loop

      CALL PRINTLK("G2")

      call stopTimer('GROUND_LK()')
      RETURN
C****
      END SUBROUTINE GROUND_LK


      SUBROUTINE PRINTLK(STR)
!@sum  PRINTLK print out selected diagnostics from specified lakes
!@auth Gavin Schmidt
      USE CONSTANT, only : lhm,byshi,rhow,shw
      USE MODEL_COM, only : qcheck
      USE GEOM, only : axyp
      USE LAKES_COM, only : tlake,mwl,mldlk,gml,flake
#ifdef TRACERS_WATER
     *         ,trlake
#endif
      USE SEAICE, only : xsi,ace1i,rhoi
      USE SEAICE_COM, only : lakeice=>si_atm
      USE DOMAIN_DECOMP_ATM, only : GRID, getDomainBounds
      IMPLICIT NONE
      CHARACTER*2, INTENT(IN) :: STR
      INTEGER, PARAMETER :: NDIAG=4
      INTEGER I,J,N, J_0, J_1
      INTEGER, DIMENSION(NDIAG) :: IDIAG = (/112, 103, 131, 79/),
     *                             JDIAG = (/66, 59, 33, 34/)
      REAL*8 HLK2,TLK2, TSIL(4)

      REAL*8, DIMENSION(:,:), POINTER :: RSI,MSI,SNOWI
      REAL*8, DIMENSION(:,:,:), POINTER :: HSI
#ifdef TRACERS_WATER
      REAL*8, DIMENSION(:,:,:,:), POINTER :: TRSI
#endif

      RSI => LAKEICE%RSI
      MSI => LAKEICE%MSI
      HSI => LAKEICE%HSI
      SNOWI => LAKEICE%SNOWI
#ifdef TRACERS_WATER
      TRSI => LAKEICE%TRSI
#endif

      IF (.NOT.QCHECK) RETURN

      call getDomainBounds(grid, J_STRT=J_0,      J_STOP=J_1)

      DO N=1,NDIAG
        I=IDIAG(N)
        J=JDIAG(N)
        if (J.lt. J_0 .or. J.gt. J_1) CYCLE
        IF (FLAKE(I,J).gt.0) THEN
          HLK2 = MWL(I,J)/(RHOW*FLAKE(I,J)*AXYP(I,J)) - MLDLK(I,J)
          IF (HLK2.gt.0) THEN
            TLK2 = (GML(I,J)/(SHW*RHOW*FLAKE(I,J)*AXYP(I,J)) -
     *           TLAKE(I,J)*MLDLK(I,J))/HLK2
          ELSE
            TLK2=0.
          END IF
          TSIL(:)=0.
          IF (RSI(I,J).gt.0) THEN
            TSIL(1:2) = (HSI(1:2,I,J)/(XSI(1:2)*(ACE1I+SNOWI(I,J)))+LHM)
     *           *BYSHI
            TSIL(3:4) = (HSI(3:4,I,J)/(XSI(3:4)*MSI(I,J))+LHM)*BYSHI
          END IF
          WRITE(99,*) STR,I,J,FLAKE(I,J),TLAKE(I,J),TLK2,MLDLK(I,J),HLK2
     *         ,RSI(I,J),MSI(I,J)/RHOI,SNOWI(I,J)/RHOW,TSIL(1:4)
#ifdef TRACERS_WATER
     *         ,TRLAKE(1,1:2,I,J),MWL(I,J)
#endif
        ELSE
          WRITE(99,*) STR,I,J,TLAKE(I,J),MWL(I,J)
#ifdef TRACERS_WATER
     *         ,TRLAKE(1,1:2,I,J)
#endif
        END IF
      END DO

      RETURN
      END  SUBROUTINE PRINTLK

      SUBROUTINE conserv_LKM(LKM)
!@sum  conserv_LKM calculates lake mass
!@auth Gary Russell/Gavin Schmidt
      USE RESOLUTION, only : im,jm
      USE FLUXES, only : fland
      USE DOMAIN_DECOMP_ATM, only : GRID, getDomainBounds
      USE GEOM, only : imaxj,byaxyp
      USE LAKES_COM, only : mwl,flake
      IMPLICIT NONE
      REAL*8, DIMENSION(GRID%I_STRT_HALO:GRID%I_STOP_HALO,
     &                  GRID%J_STRT_HALO:GRID%J_STOP_HALO) :: LKM
      INTEGER :: I,J
      INTEGER :: J_0,J_1,J_0S,J_1S,I_0,I_1
      LOGICAL :: HAVE_SOUTH_POLE, HAVE_NORTH_POLE

      call getDomainBounds(grid, J_STRT=J_0,      J_STOP=J_1,
     &               J_STRT_SKP=J_0S, J_STOP_SKP=J_1S,
     &               HAVE_SOUTH_POLE = HAVE_SOUTH_POLE,
     &               HAVE_NORTH_POLE = HAVE_NORTH_POLE )
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

C****
C**** LAKE MASS (kg/m^2)
C****
      DO J=J_0, J_1
      DO I=I_0,IMAXJ(J)
        IF (FLAND(I,J)+FLAKE(I,J).gt.0) THEN
          LKM(I,J)=MWL(I,J)*BYAXYP(I,J)
        ELSE
          LKM(I,J)=0.
        ENDIF
      ENDDO
      ENDDO
      IF (HAVE_SOUTH_POLE) LKM(2:im,1) =LKM(1,1)
      IF (HAVE_NORTH_POLE) LKM(2:im,JM)=LKM(1,JM)
      RETURN
      END SUBROUTINE conserv_LKM



      SUBROUTINE conserv_LKE(LKE)
!@sum  conserv_LKE calculates lake energy
!@auth Gary Russell/Gavin Schmidt
      USE RESOLUTION, only : im,jm
      USE ATM_COM, only : zatmo
      USE FLUXES, only : fland
      USE DOMAIN_DECOMP_ATM, only : GRID, getDomainBounds
      USE GEOM, only : imaxj,byaxyp
      USE LAKES_COM, only : gml,mwl,flake
      IMPLICIT NONE
      REAL*8, DIMENSION(GRID%I_STRT_HALO:GRID%I_STOP_HALO,
     &                  GRID%J_STRT_HALO:GRID%J_STOP_HALO) :: LKE
      INTEGER :: I,J
      INTEGER :: J_0,J_1,J_0S,J_1S,I_0,I_1
      LOGICAL :: HAVE_SOUTH_POLE, HAVE_NORTH_POLE

      call getDomainBounds(grid, J_STRT=J_0,      J_STOP=J_1,
     &               J_STRT_SKP=J_0S, J_STOP_SKP=J_1S,
     &     HAVE_SOUTH_POLE=HAVE_SOUTH_POLE,
     &     HAVE_NORTH_POLE=HAVE_NORTH_POLE)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

C****
C**** LAKE ENERGY (J/m^2) (includes potential energy (DISABLED))
C****
        DO J=J_0, J_1
        DO I=I_0,IMAXJ(J)
          IF (FLAND(I,J)+FLAKE(I,J).gt.0) THEN
!#!         LKE(I,J) = GML(I,J)*BYAXYP(I,J) + ZATMO(I,J)*MWL(I,J)
            LKE(I,J)=GML(I,J)*BYAXYP(I,J)
          ELSE
            LKE(I,J)=0.
          ENDIF
        END DO
      END DO
      IF (HAVE_SOUTH_POLE) LKE(2:im,1) =LKE(1,1)
      IF (HAVE_NORTH_POLE) LKE(2:im,JM)=LKE(1,JM)
      RETURN
      END SUBROUTINE conserv_LKE



      subroutine diag_river_prep
      use constant, only : rhow
      use domain_decomp_atm, only : grid,getDomainBounds,sumxpe
      use constant, only : rhow
      use model_com, only : dtsrc, calendar
      use TimeConstants_mod, only: INT_MONTHS_PER_YEAR
      use diag_com, only : aij=>aij_loc,ij_mrvr
      use lakes_com, only : irvrmth,jrvrmth,nrvrmx,nrvr,rvrout
      use Rational_mod
      implicit none
      real*8 rvrout_loc(nrvrmx), scalervr
      integer inm,i,j
      integer :: i_0, i_1, j_0, j_1
      type (Rational) :: secondsPerYear

      if(nrvr.lt.1) return
      call getDomainBounds(grid, j_strt=j_0, j_stop=j_1)
      i_0 = grid%i_strt
      i_1 = grid%i_stop
c**** convert kg/(source time step) to km^3/mon
      secondsPerYear =
     &     calendar%getMaxDaysInYear() * calendar%getSecondsPerDay()
      SCALERVR = 1d-9*real(secondsPerYear)/
     &          (INT_MONTHS_PER_YEAR*RHOW*DTSRC)

c**** fill in the river discharges in the local domain
      rvrout_loc(:)=0
      do j=j_0,j_1
      do i=i_0,i_1
        do inm=1,nrvr
          if (i.eq.irvrmth(inm).and. j.eq.jrvrmth(inm)) then
            rvrout_loc(inm) = scalervr*aij(i,j,ij_mrvr)
          end if
        end do
      end do
      end do
c**** sum over processors to compose the global table
      call sumxpe(rvrout_loc, rvrout)
      return
      end subroutine diag_river_prep

      SUBROUTINE init_lakeice(iniLAKE,do_IC_fixups)
!@sum  init_ice initializes ice arrays
!@auth Original Development Team
      USE CONSTANT, only : rhows,omega
      USE MODEL_COM, only : kocean
      USE SEAICE_COM, only : lakeice=>si_atm
      USE LAKES_COM, only : icelak
      USE FLUXES, only : flake0,atmice
      USE Dictionary_mod
      USE DOMAIN_DECOMP_ATM, only : GRID, getDomainBounds
      USE GEOM, only : sinlat2d
      USE DIAG_COM, only : npts,conpt0,icon_LMSI,icon_LHSI
      IMPLICIT NONE
      LOGICAL :: QCON(NPTS), T=.TRUE. , F=.FALSE. , iniLAKE
      CHARACTER CONPT(NPTS)*10
      INTEGER I,J,do_IC_fixups
      integer :: I_0, I_1, J_0, J_1
C****
C**** Extract useful local domain parameters from "grid"
C****
      call getDomainBounds(grid, J_STRT = J_0, J_STOP = J_1)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

C**** clean up ice fraction/sea ice salinity possibly incorrect in I.C.
      if (do_IC_fixups == 1) then
        DO J=J_0, J_1
        DO I=I_0, I_1
          IF (FLAKE0(I,J).eq.0 .and. lakeice%RSI(i,j).gt.0)
     &         lakeice%RSI(I,J)=0
          IF (lakeice%RSI(I,J).gt.0 .and. FLAKE0(I,J).gt.0)
     &         lakeice%SSI(:,I,J)=0.
        END DO
        END DO
      end if

      DO J=J_0,J_1
      DO I=I_0,I_1
        icelak%coriol(i,j) = ABS(2.*OMEGA*SINLAT2D(I,J))
      ENDDO
      ENDDO

      IF (KOCEAN.EQ.0.and.iniLAKE) THEN
        ! why should lake ice init depend on kocean,iniocean?
        call set_noice_defaults(lakeice,icelak)
      END IF

      !call seaice_to_atmgrid(atmice) ! set gtemp etc.

C**** Set conservation diagnostics for Lake ice mass and energy
      CONPT=CONPT0
      CONPT(3)="LAT. MELT" ; CONPT(4)="PRECIP"
      CONPT(5)="THERMO"
      CONPT(8)="LK FORM"
      QCON=(/ F, F, T, T, T, F, F, T, T, F, F/)
      CALL SET_CON(QCON,CONPT,"LKICE MS","(KG/M^2)        ",
     *     "(10**-9 KG/SM^2)",1d0,1d9,icon_LMSI)
      QCON=(/ F, F, T, T, T, F, F, T, T, F, F/)
      CALL SET_CON(QCON,CONPT,"LKICE EN","(10**6 J/M^2)   ",
     *     "(10**-3 W/M^2)  ",1d-6,1d3,icon_LHSI)

      END SUBROUTINE init_lakeice

      SUBROUTINE conserv_LMSI(ICE)
!@sum  conserv_LMSI calculates total amount of snow and ice over lakes
!@auth Gavin Schmidt
      USE RESOLUTION, only : im,jm
      USE GEOM, only : imaxj
      USE SEAICE, only : ace1i
      USE SEAICE_COM, only : lakeice=>si_atm
      USE LAKES_COM, only : flake
      USE DOMAIN_DECOMP_ATM, only : GRID,getDomainBounds
      IMPLICIT NONE
!@var ICE total lake snow and ice mass (kg/m^2)
      REAL*8, DIMENSION(GRID%I_STRT_HALO:GRID%I_STOP_HALO,
     &                  GRID%J_STRT_HALO:GRID%J_STOP_HALO) :: ICE
      INTEGER I,J

c**** Extract useful domain information from grid
      INTEGER J_0, J_1, I_0,I_1
      LOGICAL HAVE_SOUTH_POLE, HAVE_NORTH_POLE
      call getDomainBounds(GRID, J_STRT     =J_0,    J_STOP     =J_1,
     &               HAVE_SOUTH_POLE=HAVE_SOUTH_POLE    ,
     &               HAVE_NORTH_POLE=HAVE_NORTH_POLE    )
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

      DO J=J_0,J_1
      DO I=I_0,IMAXJ(J)
        ICE(I,J)=lakeice%RSI(I,J)*
     &       (lakeice%MSI(I,J)+ACE1I+lakeice%SNOWI(I,J))*FLAKE(I,J)
      END DO
      END DO
      IF (HAVE_SOUTH_POLE) ICE(2:im,1) =ICE(1,1)
      IF (HAVE_NORTH_POLE) ICE(2:im,JM)=ICE(1,JM)
      RETURN
C****
      END SUBROUTINE conserv_LMSI

      SUBROUTINE conserv_LHSI(EICE)
!@sum  conserv_LHSI calculates total ice energy over lakes
!@auth Gavin Schmidt
      USE RESOLUTION, only : im,jm
      USE GEOM, only : imaxj
      USE SEAICE_COM, only : lakeice=>si_atm
      USE LAKES_COM, only : flake
      USE DOMAIN_DECOMP_ATM, only : GRID,getDomainBounds
      IMPLICIT NONE
!@var EICE total lake snow and ice energy (J/m^2)
      REAL*8, DIMENSION(GRID%I_STRT_HALO:GRID%I_STOP_HALO,
     &                  GRID%J_STRT_HALO:GRID%J_STOP_HALO) :: EICE
      INTEGER I,J

c**** Extract useful domain information from grid
      INTEGER J_0, J_1, I_0,I_1
      LOGICAL HAVE_SOUTH_POLE, HAVE_NORTH_POLE
      call getDomainBounds(GRID, J_STRT     =J_0,    J_STOP     =J_1,
     &               HAVE_SOUTH_POLE=HAVE_SOUTH_POLE    ,
     &               HAVE_NORTH_POLE=HAVE_NORTH_POLE    )
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

      DO J=J_0,J_1
      DO I=I_0,IMAXJ(J)
        EICE(I,J)=lakeice%RSI(I,J)*FLAKE(I,J)*SUM(lakeice%HSI(:,I,J))
      END DO
      END DO
      IF (HAVE_SOUTH_POLE) EICE(2:im,1) =EICE(1,1)
      IF (HAVE_NORTH_POLE) EICE(2:im,JM)=EICE(1,JM)
      RETURN
C****
      END SUBROUTINE conserv_LHSI

      SUBROUTINE CHECKI(SUBR)
!@sum  CHECKI Checks whether Ice values are reasonable
!@auth Original Development Team
      USE MODEL_COM
      USE GEOM, only : imaxj
#ifdef TRACERS_WATER
      use OldTracer_mod, only: trname, t_qlimit
      USE TRACER_COM, only : NTM
#endif
      USE SEAICE, only : lmi,xsi,ace1i,Ti,Ti2b
      USE SEAICE_COM, only : x=>si_atm
      USE LAKES_COM, only : flake
      USE FLUXES
      USE DOMAIN_DECOMP_ATM, only : GRID
      USE DOMAIN_DECOMP_ATM, only : getDomainBounds
      IMPLICIT NONE

!@var SUBR identifies where CHECK was called from
      CHARACTER*6, INTENT(IN) :: SUBR
!@var QCHECKI true if errors found in seaice
      LOGICAL QCHECKI
      INTEGER I,J,L
      REAL*8 TICE
#ifdef TRACERS_WATER
      integer :: imax,jmax, n
      real*8 relerr,errmax
#endif

      integer :: J_0, J_1, J_0H, J_1H, I_0, I_1, I_0H, I_1H, njpol
      REAL*8 MSI1,SNOWL(2),MICE(2)
C****
C**** Extract useful local domain parameters from "grid"
C****
      call getDomainBounds(grid, J_STRT = J_0, J_STOP = J_1,
     *     J_STRT_HALO=J_0H, J_STOP_HALO=J_1H)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP
      I_0H = grid%I_STRT_HALO
      I_1H = grid%I_STOP_HALO
      njpol = grid%J_STRT_SKP-grid%J_STRT

C**** Check for NaN/INF in ice data
      CALL CHECK3B(x%RSI(I_0:I_1,J_0:J_1),I_0,I_1,J_0,J_1,NJPOL,1,
     &     SUBR,'rsi   ')
      CALL CHECK3B(x%MSI(I_0:I_1,J_0:J_1),I_0,I_1,J_0,J_1,NJPOL,1,
     &     SUBR,'msi   ')
      CALL CHECK3C(x%HSI(:,I_0:I_1,J_0:J_1),LMI,I_0,I_1,J_0,J_1,NJPOL,
     &     SUBR,'hsi   ')
      CALL CHECK3C(x%SSI(:,I_0:I_1,J_0:J_1),LMI,I_0,I_1,J_0,J_1,NJPOL,
     &     SUBR,'ssi   ')
      CALL CHECK3B(x%SNOWI(I_0:I_1,J_0:J_1),I_0,I_1,J_0,J_1,NJPOL,1,
     &     SUBR,'sni   ')

      QCHECKI = .FALSE.
C**** Check for reasonable values for ice variables
      DO J=J_0, J_1
        DO I=I_0,IMAXJ(J)
          IF (x%RSI(I,J).lt.0 .or. x%RSI(I,j).gt.1
     *         .or. x%MSI(I,J).lt.0) THEN
            WRITE(6,*) 'After ',SUBR,': I,J,RSI,MSI=',I,J,x%RSI(I,J)
     *           ,x%MSI(I,J)
            QCHECKI = .TRUE.
          END IF
          IF ( (FOCEAN(I,J)+FLAKE(I,J))*x%RSI(I,J).gt.0) THEN
          MSI1 = ACE1I + x%SNOWI(I,J)
          IF (ACE1I.gt.XSI(2)*MSI1) THEN ! some ice in first layer
            MICE(1) = ACE1I-XSI(2)*MSI1
            MICE(2) = XSI(2)*MSI1
c           SNOWL(1)= SNOW
            SNOWL(1)= x%SNOWI(I,J)
            SNOWL(2)= 0.
          ELSE  ! some snow in second layer
            MICE(1) = 0.
            MICE(2) = ACE1I
            SNOWL(1)= XSI(1)*MSI1
            SNOWL(2)= XSI(2)*MSI1-ACE1I
          ENDIF
          DO L=1,LMI
            IF (L.EQ.1) THEN
              IF(MICE(1).NE.0.) THEN
                TICE = Ti2b(x%HSI(1,I,J)/(XSI(1)*MSI1),
     *                      1d3*x%SSI(L,I,J)/MICE(1),SNOWL(1),MICE(1))
              ELSE
                TICE = Ti(x%HSI(1,I,J)/(XSI(1)*MSI1),0d0)
              ENDIF
            ENDIF
            IF (L.EQ.2)
     *          TICE = Ti2b(x%HSI(2,I,J)/(XSI(2)*MSI1),
     *                      1d3*x%SSI(L,I,J)/MICE(2),SNOWL(2),MICE(2))

            IF (L.gt.2) TICE = Ti(x%HSI(L,I,J)/(XSI(L)*x%MSI(I,J))
     *           ,1d3*x%SSI(L,I,J)/(XSI(L)*x%MSI(I,J)))
            IF (x%HSI(L,I,J).gt.0.or.TICE.gt.1d-10.or.TICE.lt.-80.) THEN
              WRITE(6,'(3a,3i3,6e12.4/1X,6e12.4)')
     *             'After ',SUBR,': I,J,L,TSI=',I,J,L,TICE,x%RSI(I,J)
            WRITE(6,*) x%HSI(:,I,J),x%MSI(I,J),x%SNOWI(I,J),x%SSI(:,I,J)
              IF (TICE.gt.1d-3.or.TICE.lt.-100.) QCHECKI = .TRUE.
            END IF
            IF (x%SSI(L,I,J).lt.0) THEN
              WRITE(6,*) 'After ',SUBR,': I,J,L,SSI=',I,J,L,x%SSI(:,I
     *             ,J),x%MSI(I,J),x%SNOWI(I,J),x%RSI(I,J)
              QCHECKI = .TRUE.
            END IF
           IF (L.gt.2 .and. x%SSI(L,I,J).gt.0.04*XSI(L)*x%MSI(I,J)) THEN
              WRITE(6,*) 'After ',SUBR,': I,J,L,SSI/MSI=',I,J,L,1d3
     *         *x%SSI(:,I,J)/(XSI(L)*x%MSI(I,J)),x%SSI(:,I,J),x%MSI(I,J)
     *             ,x%SNOWI(I,J),x%RSI(I,J)
              QCHECKI = .TRUE.
            END IF
          END DO
          IF (x%SNOWI(I,J).lt.0) THEN
            WRITE(6,*) 'After ',SUBR,': I,J,SNOWI=',I,J,x%SNOWI(I,J)
            QCHECKI = .TRUE.
          END IF
          IF (x%MSI(I,J).gt.10000) THEN
         WRITE(6,*) 'After ',SUBR,': I,J,MSI=',I,J,x%MSI(I,J),x%RSI(I,J)
c            QCHECKI = .TRUE.
          END IF
          END IF
        END DO
      END DO

#ifdef TRACERS_WATER
      do n=1,ntm
C**** check negative tracer mass
        if (t_qlimit(n)) then
        do j=J_0, J_1
          do i=I_0,imaxj(j)
            if ((focean(i,j)+flake(i,j))*x%rsi(i,j).gt.0) then
              do l=1,lmi
                if (x%trsi(n,l,i,j).lt.0.) then
                  print*,"Neg Tracer in sea ice after ",subr,i,j,l,
     *                 trname(n),x%trsi(n,l,i,j),x%rsi(i,j),
     *                 x%msi(i,j),x%ssi(l,i,j),x%snowi(i,j)
                  QCHECKI=.true.
                end if
              end do
            end if
          end do
        end do
        end if
C**** Check conservation of water tracers in sea ice
        if (trname(n).eq.'Water') then
          errmax = 0. ; imax=I_0 ; jmax=J_0
          do j=J_0, J_1
          do i=I_0,imaxj(j)
            if ((focean(i,j)+flake(i,j))*x%rsi(i,j).gt.0) then
              relerr=max(
     *        abs(x%trsi(n,1,i,j)-(x%snowi(i,j)+ace1i)*xsi(1)+
     &             x%ssi(1,i,j
     *           ))/x%trsi(n,1,i,j),abs(x%trsi(n,2,i,j)-
     &             (x%snowi(i,j)+ace1i)
     *             *xsi(2)+x%ssi(2,i,j))/x%trsi(n,2,i,j),
     &             abs(x%trsi(n,3,i,j)
     *        -x%msi(i,j)*xsi(3)+x%ssi(3,i,j))/x%trsi(n,3,i,j),
     &             abs(x%trsi(n
     *       ,4,i,j)-x%msi(i,j)*xsi(4)+x%ssi(4,i,j))/x%trsi(n,4,i,j))
              if (relerr.gt.errmax) then
                imax=i ; jmax=j ; errmax=relerr
              end if
            end if
          end do
          end do
          write(*,'(A36,A7,A,2I3,11E24.16)')
     $         "Relative error in sea ice mass after",trim(subr),":"
     $         ,imax,jmax,errmax,x%trsi(n,:,imax,jmax),
     $         (x%snowi(imax,jmax)
     $         +ace1i)*xsi(1)-x%ssi(1,imax,jmax),
     $         (x%snowi(imax,jmax)+ace1i)
     $         *xsi(2)-x%ssi(2,imax,jmax),
     $         x%msi(imax,jmax)*xsi(3:4)-x%ssi(3:4
     $         ,imax,jmax),x%rsi(imax,jmax),x%msi(imax,jmax)
        end if
      end do
#endif

      IF (QCHECKI)
     &     call stop_model("CHECKI: Ice variables out of bounds",255)

      END SUBROUTINE CHECKI

      subroutine newton (a,b,c,d,errmax,n,i,j,axyp)
!@sum Solves the equation F(a) = 0 using Newton's method
!@sum Here, F(a) = a**d + a*b -c , b,c > 0, d > 1 (a>0)
!@sum F'(a) = d*a**(d-1) + b and F"(a) are > 0 for a>0 => 1 solution
!@sum between a=0 where F=-c < 0 and a=c/b where F=(c/b)**d > 0
!@auth Reto Ruedy
      IMPLICIT NONE
      real*8, intent(out) :: a
      real*8, intent(in) :: b,c,d,errmax,axyp
      integer, intent(out) :: n
      integer, intent(in) :: i,j
      real*8 Foff,alow
      a = .5d0*c/b  ; n=0
      Foff = a**d + a*b -c
      do while(n<10)
        a = a - Foff/(d*a**(d-1) + b)
        alow = (1-errmax)*a
        Foff = alow**d + alow*b -c ; n = n+1
        if(n>7) write(6,'(a,3i4,a,d11.4,a,d22.15)') 'Newton: i,j,niter',
     *  i,j,n,' flake:',a/axyp,' zero:',Foff/axyp
        if(Foff < errmax) return
        a=alow
      end do
      return
      end subroutine newton
