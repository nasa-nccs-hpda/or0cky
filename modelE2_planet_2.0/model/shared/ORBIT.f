      SUBROUTINE ORBIT (DOBLIQ,ECCEN,DOMEGVP,VEDAY,EDPY, DAY,
     *                  SDIST,SIND,COSD,SUNLON,SUNLAT,EQTIME)
C****
C**** ORBIT receives orbital parameters and time of year, and returns
C**** distance from Sun, declination angle, and Sun's overhead position.
C**** Reference for following caculations is:  V.M.Blanco and
C**** S.W.McCuskey, 1961, "Basic Physics of the Solar System", pages
C**** 135 - 151.  Existence of Moon and heavenly bodies other than
C**** Earth and Sun are ignored.  Earth is assumed to be spherical.
C****
C**** Program author: Gary L. Russell 2004/11/16
C**** Angles, longitude and latitude are measured in radians.
C****
C**** Input: ECCEN  = eccentricity of the orbital ellipse
C****        OBLIQ  = latitude of Tropic of Cancer
C****        OMEGVP = longitude of perihelion (sometimes Pi is added) =
C****               = spatial angle from vernal equinox to perihelion
C****                 with Sun as angle vertex
C****        DAY    = days measured since 2000 January 1, hour 0
C****
C****        EDPY  = Earth days per year
C****                tropical year = 365.2425 (Gregorgian Calendar)
C****                tropical year = 365      (Generic Year)
C****        VEDAY = Vernal equinox
C****                79.0 (Generic year Mar 21 hour 0)
C****                79.5 (Generic year Mar 21 hour 12 - PMIP standard)
C****                79.3125d0 for days from 2000 January 1, hour 0 till vernal
C****                     equinox of year 2000 = 31 + 29 + 19 + 7.5/24
C****
C**** Intermediate quantities:
C****    BSEMI = semi minor axis in units of semi major axis
C****   PERIHE = perihelion in days since 2000 January 1, hour 0
C****            in its annual revolution about Sun
C****       TA = true anomaly = spatial angle from perihelion to
C****            current location with Sun as angle vertex
C****       EA = eccentric anomaly = spatial angle measured along
C****            eccentric circle (that circumscribes Earth's orbit)
C****            from perihelion to point above (or below) Earth's
C****            absisca (where absisca is directed from center of
C****            eccentric circle to perihelion)
C****       MA = mean anomaly = temporal angle from perihelion to
C****            current time in units of 2*Pi per tropical year
C****   TAofVE = TA(VE) = true anomaly of vernal equinox = - OMEGVP
C****   EAofVE = EA(VE) = eccentric anomaly of vernal equinox
C****   MAofVE = MA(VE) = mean anomaly of vernal equinox
C****   SLNORO = longitude of Sun in Earth's nonrotating reference frame
C****   VEQLON = longitude of Greenwich Meridion in Earth's nonrotating
C****            reference frame at vernal equinox
C****   ROTATE = change in longitude in Earth's nonrotating reference
C****            frame from point's location on vernal equinox to its
C****            current location where point is fixed on rotating Earth
C****   SLMEAN = longitude of fictitious mean Sun in Earth's rotating
C****            reference frame (normal longitude and latitude)
C****
C**** Output: SIND = sine of declination angle = sin(SUNLAT)
C****         COSD = cosine of the declination angle = cos(SUNLAT)
C****       SUNDIS = distance to Sun in units of semi major axis
C****       SUNLON = longitude of point on Earth directly beneath Sun
C****       SUNLAT = latitude of point on Earth directly beneath Sun
C****       EQTIME = Equation of Time =
C****              = longitude of fictitious mean Sun minus SUNLON
C****
C**** From the above reference:
C**** (4-54): [1 - ECCEN*cos(EA)]*[1 + ECCEN*cos(TA)] = (1 - ECCEN^2)
C**** (4-55): tan(TA/2) = sqrt[(1+ECCEN)/(1-ECCEN)]*tan(EA/2)
C**** Yield:  tan(EA) = sin(TA)*sqrt(1-ECCEN^2) / [cos(TA) + ECCEN]
C****    or:  tan(TA) = sin(EA)*sqrt(1-ECCEN^2) / [cos(EA) - ECCEN]
C****
      USE CONSTANT, only : twopi,pi,radian
      IMPLICIT NONE
      REAL*8, INTENT(IN) :: DOBLIQ,ECCEN,DOMEGVP,DAY,VEDAY,EDPY
      REAL*8, INTENT(OUT) :: SIND,COSD,SDIST,SUNLON,SUNLAT,EQTIME

      REAL*8 MA,OMEGVP,OBLIQ,EA,DEA,BSEMI
     *     ,TAofVE,EAofVE,MAofVE,SUNDIS,TA,SUNX,SUNY,SLNORO
     *     ,VEQLON,ROTATE,SLMEAN
c      REAL*8, PARAMETER :: EDAYzY=365.2425d0, VE2000=79.3125d0
c      REAL*8, PARAMETER :: EDAYzY=365d0, VE2000=79d0  ! original parameters
      REAL*8  EDAYzY,VE2000
C****
      VE2000=VEDAY
      EDAYzY=EDPY
      OMEGVP=DOMEGVP*radian
      OBLIQ=DOBLIQ*radian
C**** Determine EAofVE from geometry: tan(EA) = b*sin(TA) / [e+cos(TA)]
C**** Determine MAofVE from Kepler's equation: MA = EA - e*sin(EA)
C**** Determine MA knowing time from vernal equinox to current day
C****
      BSEMI  = SQRT (1 - ECCEN*ECCEN)
      TAofVE = - OMEGVP
      EAofVE = ATAN2 (BSEMI*SIN(TAofVE), ECCEN+COS(TAofVE))
      MAofVE = EAofVE - ECCEN*SIN(EAofVE)
C     PERIHE = VE2000 - MAofVE*EDAYzY/TWOPI
      MA     = MODULO (TWOPI*(DAY-VE2000)/EDAYzY + MAofVE, TWOPI)
C****
C**** Numerically invert Kepler's equation: MA = EA - e*sin(EA)
C****
      EA  = MA + ECCEN*(SIN(MA) + ECCEN*SIN(2*MA)/2)
   10 dEA = (MA - EA + ECCEN*SIN(EA)) / (1 - ECCEN*COS(EA))
      EA  = EA + dEA
      IF(ABS(dEA).gt.1d-10)  GO TO 10
C****
C**** Calculate distance to Sun and true anomaly
C****
      SUNDIS = 1 - ECCEN*COS(EA)
      TA     = ATAN2 (BSEMI*SIN(EA), COS(EA)-ECCEN)
      SDIST  = SUNDIS*SUNDIS   ! added for compatiblity
C****
C**** Change reference frame to be nonrotating reference frame, angles
C**** fixed according to stars, with Earth at center and positive x
C**** axis be ray from Earth to Sun were Earth at vernal equinox, and
C**** x-y plane be Earth's equatorial plane.  Distance from current Sun
C**** to this x axis is SUNDIS sin(TA-TAofVE).  At vernal equinox, Sun
C**** is located at (SUNDIS,0,0).  At other times, Sun is located at:
C****
C**** SUN = (SUNDIS cos(TA-TAofVE),
C****        SUNDIS sin(TA-TAofVE) cos(OBLIQ),
C****        SUNDIS sin(TA-TAofVE) sin(OBLIQ))
C****
      SIND   = SIN(TA-TAofVE) * SIN(OBLIQ)
      COSD   = SQRT (1 - SIND*SIND)
      SUNX   = COS(TA-TAofVE)
      SUNY   = SIN(TA-TAofVE) * COS(OBLIQ)
      SLNORO = ATAN2 (SUNY,SUNX)
C****
C**** Determine Sun location in Earth's rotating reference frame
C**** (normal longitude and latitude)
C****
      VEQLON = TWOPI*VE2000 - PI + MAofVE - TAofVE  !  modulo 2*Pi
      ROTATE = TWOPI*(DAY-VE2000)*(EDAYzY+1)/EDAYzY
      SUNLON = MODULO (SLNORO-ROTATE-VEQLON, TWOPI)
      IF(SUNLON.gt.PI)  SUNLON = SUNLON - TWOPI
      SUNLAT = ASIN (SIN(TA-TAofVE)*SIN(OBLIQ))
C****
C**** Determine longitude of fictitious mean Sun
C**** Calculate Equation of Time
C****
      SLMEAN = PI - TWOPI*(DAY-FLOOR(DAY))
      EQTIME = MODULO (SLMEAN-SUNLON, TWOPI)
      IF(EQTIME.gt.PI)  EQTIME = EQTIME - TWOPI
C****
      RETURN
      END SUBROUTINE ORBIT
