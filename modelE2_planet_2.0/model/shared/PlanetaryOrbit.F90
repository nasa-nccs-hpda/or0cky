!-----------------------------------------------------------------------
! PlanetaryOrbit extends FixedOrbit and is intended to be used for
! planets other than the Earth.  The only functionality added here is
! a constructor for specifying orbital parameters and a trivial
! makeCalendar() method which delegates to the PlanetaryCalendar class.
!-----------------------------------------------------------------------

module PlanetaryOrbit_mod
  use AbstractOrbit_mod
  use FixedOrbit_mod
  use KindParameters_Mod, only: WP => DP, DP
  implicit none
  private

  public :: PlanetaryOrbit

  type, extends(FixedOrbit) :: PlanetaryOrbit
    private
  contains
    procedure :: makeCalendar
    procedure :: print_unit
  end type PlanetaryOrbit


  interface PlanetaryOrbit
    module procedure newPlanetaryOrbit
    module procedure newPlanetaryOrbit_fromParams
    module procedure newPlanetaryOrbit_exact_rational
  end interface PlanetaryOrbit


  real(kind=WP), parameter :: EARTH_LON_AT_PERIHELION = 282.9
  real (kind=DP), parameter :: PI = 2*asin(1.d0)

contains

  function newPlanetaryOrbit_fromParams(planetParams) result(orbit)
     use PlanetaryParams_mod
     type (PlanetaryOrbit) :: orbit
     type (PlanetaryParams), intent(in) :: planetParams

     associate(p => planetParams)
       orbit = PlanetaryOrbit( &
          & p%getObliquity(), &
          & p%getEccentricity(), &
          & p%getLongitudeAtPeriapsis(), &
          & p%getSiderealOrbitalPeriod(), &
          & p%getSiderealRotationPeriod(), &
          & p%getMeanDistance(), &
          & p%getHourAngleOffset())
     end associate
     
  end function newPlanetaryOrbit_fromParams

  function newPlanetaryOrbit(obliquity, eccentricity, longitudeAtPeriapsis, &
       & siderealPeriod, siderealRotationPeriod, meanDistance, hourAngleOffset) result(orbit)
    use Rational_mod
    use BaseTime_mod
    use TimeInterval_mod
    use OrbitUtilities_mod, only: computeMeanAnomaly
    type (PlanetaryOrbit) :: orbit
    real (kind=WP), intent(in) :: obliquity
    real (kind=WP), intent(in) :: eccentricity
    real (kind=WP), intent(in) :: longitudeAtPeriapsis
    real (kind=WP), intent(in) :: siderealPeriod
    real (kind=WP), intent(in) :: siderealRotationPeriod
    real (kind=WP), intent(in) :: meanDistance
    real (kind=WP), intent(in) :: hourAngleOffset

    type (TimeInterval) :: adjustedSiderealPeriod
    type (TimeInterval) :: adjustedRotationPeriod
    type (Rational) :: period_ratio

    real (kind=WP), parameter :: TOLERANCE = 1.d-1
    ! To avoid issues with rounding times at beginning of days, years,
    ! etc, we use a Rational representation of times and associated
    ! anglesfor orbit internals.  User-specified FP values must
    ! therefore be rounded to a rational.  The denominators of these
    ! rationals grow under some arithmetic operations so it is
    ! important that the denominators for the base periods (orbital
    ! and rotational) be modest.

    ! Keep the denominators in rational approximations from being too large
    period_ratio = Rational(siderealRotationPeriod / siderealPeriod, 1.d-6)

    if (abs(siderealRotationPeriod) > abs(siderealPeriod)) then ! long days
       adjustedRotationPeriod = newTimeInterval(Rational(siderealRotationPeriod, tolerance=TOLERANCE))
       adjustedSiderealPeriod =  newTimeInterval(adjustedRotationPeriod / period_ratio)
    else
       adjustedSiderealPeriod = newTimeInterval(Rational(siderealPeriod, tolerance=TOLERANCE))
       adjustedRotationPeriod =  newTimeInterval(adjustedSiderealPeriod * period_ratio)
    end if

    orbit = PlanetaryOrbit(obliquity, eccentricity, longitudeAtPeriapsis, &
         & adjustedSiderealPeriod, adjustedRotationPeriod, meanDistance, hourAngleOffset)

   end function newPlanetaryOrbit

   !---------------------------------------------
   ! In this interface the orbital and rotational periods are expressed as exact Rationals to enforce
   ! certain periodicities.  E.g., length of day exactly divides length of year.
   !---------------------------------------------
   function newPlanetaryOrbit_exact_rational(obliquity, eccentricity, longitudeAtPeriapsis, &
     & adjustedSiderealPeriod, adjustedRotationPeriod, meanDistance, hourAngleOffset) result(orbit)
     use Rational_mod
     use BaseTime_mod
     use TimeInterval_mod
     use OrbitUtilities_mod, only: computeMeanAnomaly
     use dictionary_mod, ONLY: sync_param
     integer, parameter :: LONG = selected_int_kind(16)
     type (PlanetaryOrbit) :: orbit
     real (kind=WP), intent(in) :: obliquity
     real (kind=WP), intent(in) :: eccentricity
     real (kind=WP), intent(in) :: longitudeAtPeriapsis
     type (TimeInterval), intent(in) :: adjustedSiderealPeriod
     type (TimeInterval), intent(in) :: adjustedRotationPeriod
     real (kind=WP), intent(in) :: meanDistance
     real (kind=WP), intent(in) :: hourAngleOffset
     
     type (TimeInterval) :: meanDay
     integer :: numerator, denominator
     type (Rational) :: ratio
     type (BaseTime) :: timeAtVernalEquinox
     character(len=16) :: date_of_vernal_equinox
     
     call orbit%setLongitudeAtPeriapsis(longitudeAtPeriapsis)
     call orbit%setObliquity(obliquity)
     call orbit%setEccentricity(eccentricity)
     
     call orbit%setMeanDistance(meanDistance)
     call orbit%setHourAngleOffset(hourAngleOffset)
     
     ! Need to support tidally locked scenario:
     if (adjustedSiderealPeriod == adjustedRotationPeriod) then
        meanDay = newTimeInterval(Rational(2_LONG**48)) ! large but not Inf
     else
        meanDay = newTimeInterval(1/(1/adjustedRotationPeriod - 1/adjustedSiderealPeriod))
     end if

     ! Mean day is just for calendar purposes - it is not negative even
     ! for retrograde orbits.
     meanDay = newTimeInterval(abs(meanDay))

     call orbit%setMeanDay(meanDay)
     call orbit%setSiderealOrbitalPeriod(adjustedSiderealPeriod)
     call orbit%setSiderealRotationPeriod(adjustedRotationPeriod)

     ! Time at periapsis needs to be set after other parameters are established.
     ! Keep vernal equinox at the fraction of the year as for modern Earth.
     ! March 21 @ 12:00 day-of-year=80
     numerator = 79*24 + 12
     denominator = 365*24
     ratio = Rational(numerator,denominator)
     date_of_vernal_equinox = 'earth_default'
     call sync_param('date_of_vernal_equinox', date_of_vernal_equinox)
     select case (date_of_vernal_equinox)
     case ('earth_default')
        timeAtVernalEquinox = newBaseTime(ratio*orbit%getSiderealOrbitalPeriod())
     case ('jan01')
        timeAtVernalEquinox = newBaseTime(0)
     case default
        call stop_model('PlanetaryCalendar unsupported date_of_vernal_equinox.')
     end select

     call orbit%setTimeAtVernalEquinox(timeAtVernalEquinox)
     call orbit%setTimeAtPeriapsis()
     
   end function newPlanetaryOrbit_exact_rational



  ! Pass instance of self to constructor for PlanetaryCalendar.
  function makeCalendar(this) result(calendar)
    use PlanetaryCalendar_mod
    use AbstractCalendar_mod
    class (AbstractCalendar), allocatable :: calendar
    class (PlanetaryOrbit), intent(in) :: this

    allocate(calendar, source=PlanetaryCalendar(this))
  end function makeCalendar


  subroutine print_unit(this, unit)
    class (PlanetaryOrbit), intent(in) :: this
    integer, intent(in) :: unit

    write(unit,*) 'Fixed orbital parameters for planet.'
    write(unit,*) '  Eccentricity:', this%getEccentricity()
    write(unit,*) '  Obliquity (degs):',this%getObliquity()
    write(unit,*) '  Longitude at periapsis (degs from ve):', &
         & this%getLongitudeAtPeriapsis()

  end subroutine print_unit

end module PlanetaryOrbit_mod
