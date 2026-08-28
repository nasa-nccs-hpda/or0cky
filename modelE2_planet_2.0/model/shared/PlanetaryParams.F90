module PlanetaryParams_mod
!@type PlanetaryParams a derived type that manages physical parameters
!@+    that are directly dependent on the planet being simulated.
!@+    Default values are those for modern Earth as specified in AR5.
!@+    
!@+    Note that data components here are encapsulated to ensure that 
!@+    they are not accidentally modified after being set. 
!@+    It is expected that a higher container (e.g. Constants_mod) will
!@+    provide these parameters and various related quantities using
!@+    PROTECTED to prevent accidental change during a simulation.

   use KindParameters_mod, only: WP => DP
   use TimeConstants_mod, only: SECONDS_PER_DAY
   use TimeConstants_mod, only: SECONDS_PER_YEAR
   use TimeConstants_mod, only: DAYS_PER_YEAR
   implicit none
   private
   
   public :: PlanetaryParams

   ! Earth defaults

   public :: DEFAULT_ECCENTRICITY
   public :: DEFAULT_OBLIQUITY
   public :: DEFAULT_LONGITUDE_AT_PERIAPSIS
   public :: DEFAULT_SIDEREAL_ORBITAL_PERIOD
   public :: DEFAULT_SIDEREAL_ROTATION_PERIOD
   public :: DEFAULT_HOUR_ANGLE_OFFSET

!**** DEFAULT ORBITAL PARAMETERS FOR EARTH
!**** Note PMIP runs had specified values that do not necesarily
!**** coincide with those used as the default, or the output of ORBPAR.
!****                    OMEGT          OBLIQ        ECCEN
!**** DEFAULT (2000 AD): 282.9          23.44        0.0167
!**** PMIP CONTROL:      282.04         23.446       0.016724
!**** PMIP 6kyr BP:      180.87         24.105       0.018682
!**** PMIP LGM (21k):    294.42         22.949       0.018994

!@par DEFAULT_ECCENTRICITY
   real (kind=WP), parameter :: DEFAULT_ECCENTRICITY = 0.0167D0
!@par DEFAULT_OBLIQUITY - degrees
   real (kind=WP), parameter :: DEFAULT_OBLIQUITY = 23.44D0
!@par DEFAULT_LONGITUDE_AT_PERIAPSIS - degrees
   real (kind=WP), parameter :: DEFAULT_LONGITUDE_AT_PERIAPSIS = 282.9D0
!@par DEFAULT_SIDEREAL_ORBITAL_PERIOD - seconds
   real (kind=WP), parameter :: DEFAULT_SIDEREAL_ORBITAL_PERIOD = &
        & SECONDS_PER_YEAR ! seconds
!@par DEFAULT_SIDEREAL_ROTATION_PERIOD - sidereal rotation (seconds)
   real (kind=WP), parameter :: DEFAULT_SIDEREAL_ROTATION_PERIOD = &
        & SECONDS_PER_DAY * DAYS_PER_YEAR/(DAYS_PER_YEAR+1) ! seconds
!@par DEFAULT_HOUR_ANGLE_OFFSET - adjust the location midnight relative to prime meridian
!@+   This allows the topographic prime meridian to be different than that used for
!@+   determining zenith (hour) angle.
   real (kind=WP), parameter :: DEFAULT_HOUR_ANGLE_OFFSET = 0 ! degrees East


   type PlanetaryParams

      ! Orbital parameters
      character(len=:), allocatable :: planetName
      real (kind=WP) :: obliquity = DEFAULT_OBLIQUITY
      real (kind=WP) :: eccentricity = DEFAULT_ECCENTRICITY
      real (kind=WP) :: longitudeAtPeriapsis = DEFAULT_LONGITUDE_AT_PERIAPSIS
      real (kind=WP) :: siderealOrbitalPeriod = DEFAULT_SIDEREAL_ORBITAL_PERIOD
      real (kind=WP) :: meanDistance  = 1.0 ! A.U.  

!@par hourAngleOffset - (degrees) adjusts the location of 'midnight'.  A value of 0
!@par+ corresponds to midnight over Greenwich (noon over international
!@par+ dateline).  
      real (kind=WP) :: hourAngleOffset = DEFAULT_HOUR_ANGLE_OFFSET ! degrees East

!@ If quantizeYearLength is true (default), then orbital period is adjusted such that
!@ it is an integral multiple of the rotation period.
      logical :: quantizeYearLength = .true. ! 

!@ The original orbit implementation neglected the effect of obliquity
!@ on the equation of time.  The following flag defaults to false, but
!@ when set to true, the corrected EOT is used.  The difference is
!@ generally small.
!@ Note: For Earth, EOT is set to False for consistency.
!@par EOT - 'F', 'N', or 'T
      character :: EOT = '-' ! undefined

      ! Intrinsic parameters
      real (kind=WP) :: siderealRotationPeriod = SECONDS_PER_DAY * &
           &  (DAYS_PER_YEAR/(1+DAYS_PER_YEAR))  ! seconds
!!$      real (kind=WP) :: radius          ! meters
!!$      real (kind=WP) :: gravity         ! m s^-2

   contains

      ! Orbital parameters
      procedure :: getObliquity
      procedure :: getEccentricity
      procedure :: getLongitudeAtPeriapsis
      procedure :: getSiderealOrbitalPeriod
      procedure :: getMeanDistance
      procedure :: getHourAngleOffset

      ! Intrinsic parameters
      procedure :: getSiderealRotationPeriod
!!$      procedure :: getRadius
!!$      procedure :: getGravity

   end type PlanetaryParams


   interface PlanetaryParams
      module procedure newPlanetaryParams_fromRundeck
   end interface PlanetaryParams

contains

   function newPlanetaryParams_fromRundeck()  result(params)
      use Dictionary_mod
      type (PlanetaryParams) :: params
      character(len=80) :: string
      character(len=80) :: planetName
      character :: default_EOT

      planetName = 'Earth'

      call sync_param('planetName', planetName)
      params%planetName = planetName

      ! Defaults that differ for Earth vs not Earth
      select case (params%planetName)
      case ('Earth', 'earth', 'EARTH')
         default_EOT = 'F'
         if (is_set_param('quantizeYearLength')) then
            call stop_model('Must NOT specify "quantizeYearLength" for Earth.', 255)
            return
         endif
      case default ! anything else
         default_EOT = 'T'
         ! Due to numerous accidental omissions, we now require 'quantizeYearLength'
         ! to be set for non-Earth orbits.
         if (.not. is_set_param('quantizeYearLength')) then
            call stop_model('Must specify "quantizeYearLength" flag for planetary orbits.', 255)
            return
         else
            ! Dictionary does not support logicals
            call get_param('quantizeYearLength', string)
            select case (trim(string))
            case ('T','t','TRUE','True','true','.TRUE.','.True.','.true.')
               params%quantizeYearLength = .true.
            case default
               params%quantizeYearLength = .false.
            end select
         end if
      end select

      call sync_param('obliquity', params%obliquity)
      call sync_param('eccentricity', params%eccentricity)
      call sync_param('longitudeAtPeriapsis', params%longitudeAtPeriapsis)
      call sync_param('siderealOrbitalPeriod', params%siderealOrbitalPeriod)
      call sync_param('siderealRotationPeriod', params%siderealRotationPeriod)
      call sync_param('meanDistance', params%meanDistance)
      call sync_param('hourAngleOffset', params%hourAngleOffset)

      string = default_EOT
      call sync_param('EOT', string)
      select case (trim(string))
      case ('N','n','NAIVE','Naive','naive')
         params%EOT = 'N'
      case ('F','f','FALSE','False','false','.FALSE.','.False.','.false.','OFF','Off','off')
         params%EOT = 'F'
      case ('T','t','TRUE','True','true','.TRUE.','.True.','.true.','ON','On','on')
         params%EOT = 'T'
      case default
         call stop_model('Unsupported value for EOT specifier: '//trim(string), 255)
         params%EOT = default_EOT
      end select


   end function newPlanetaryParams_fromRundeck


   real (kind=WP) function getObliquity(this) result(obliquity)
      class (PlanetaryParams), intent(in) :: this
      obliquity = this%obliquity
   end function getObliquity


   real (kind=WP) function getEccentricity(this) result(eccentricity)
      class (PlanetaryParams), intent(in) :: this
      eccentricity = this%eccentricity
   end function getEccentricity


   real (kind=WP) function getLongitudeAtPeriapsis(this) result(longitudeAtPeriapsis)
      class (PlanetaryParams), intent(in) :: this
      longitudeAtPeriapsis = this%longitudeAtPeriapsis
   end function getLongitudeAtPeriapsis


   real (kind=WP) function getSiderealOrbitalPeriod(this) result(siderealOrbitalPeriod)
      class (PlanetaryParams), intent(in) :: this
      siderealOrbitalPeriod = this%siderealOrbitalPeriod
   end function getSiderealOrbitalPeriod
   

   real (kind=WP) function getMeanDistance(this) result(meanDistance)
      class (PlanetaryParams), intent(in) :: this
      meanDistance = this%meanDistance
   end function getMeanDistance


   real (kind=WP) function getHourAngleOffset(this) result(hourAngleOffset)
      class (PlanetaryParams), intent(in) :: this
      hourAngleOffset = this%hourAngleOffset
   end function getHourAngleOffset


   real (kind=WP) function getSiderealRotationPeriod(this) result(siderealRotationPeriod)
      class (PlanetaryParams), intent(in) :: this
      siderealRotationPeriod = this%siderealRotationPeriod
   end function getSiderealRotationPeriod


!!$   real (kind=WP) function getRadius(this) result(radius)
!!$      class (PlanetaryParams), intent(in) :: this
!!$      radius = this%radius
!!$   end function radius
!!$
!!$
!!$   real (kind=WP) function getGravity(this) result(gravity)
!!$      class (PlanetaryParams), intent(in) :: this
!!$      gravity = this%gravity
!!$   end function gravity
!!$

end module PlanetaryParams_mod
