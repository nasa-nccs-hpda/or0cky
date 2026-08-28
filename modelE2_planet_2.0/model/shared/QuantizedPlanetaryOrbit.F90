!-----------------------------------------------------------------------
! QuantizedPlanetaryOrbit extends PlanetaryOrbit and is intended to be used for
! planets other than the Earth.  
! In this subclass, the orbital period is adjusted from the requested value
! to ensure that the number of solar days per year is an exact integer.
!
! The implementation is simple, and most work is delegated back to the
! usual PlanetaryOrbit class. 
!-----------------------------------------------------------------------

module QuantizedPlanetaryOrbit_mod
  use AbstractOrbit_mod
  use FixedOrbit_mod
  use PlanetaryOrbit_mod
  use KindParameters_Mod, only: WP => DP, DP
  implicit none
  private

  public :: QuantizedPlanetaryOrbit

  type, extends(PlanetaryOrbit) :: QuantizedPlanetaryOrbit
    private
  contains
    procedure :: makeCalendar
    procedure :: print_unit
  end type QuantizedPlanetaryOrbit


  interface QuantizedPlanetaryOrbit
    module procedure newQuantizedPlanetaryOrbit
    module procedure newQuantizedPlanetaryOrbit_fromParams
  end interface QuantizedPlanetaryOrbit


  real(kind=WP), parameter :: EARTH_LON_AT_PERIHELION = 282.9
  real (kind=DP), parameter :: PI = 2*asin(1.d0)

contains

  function newQuantizedPlanetaryOrbit_fromParams(planetParams) result(orbit)
     use PlanetaryParams_mod
     type (QuantizedPlanetaryOrbit) :: orbit
     type (PlanetaryParams), intent(in) :: planetParams

     associate(p => planetParams)
       orbit = QuantizedPlanetaryOrbit( &
          & p%getObliquity(), &
          & p%getEccentricity(), &
          & p%getLongitudeAtPeriapsis(), &
          & p%getSiderealOrbitalPeriod(), &
          & p%getSiderealRotationPeriod(), &
          & p%getMeanDistance(), &
          & p%getHourAngleOffset())
     end associate
     
  end function newQuantizedPlanetaryOrbit_fromParams

  function newQuantizedPlanetaryOrbit(obliquity, eccentricity, longitudeAtPeriapsis, &
       & siderealPeriod, siderealRotationPeriod, meanDistance, hourAngleOffset) result(orbit)
    use Rational_mod
    use BaseTime_mod
    use TimeInterval_mod
    use OrbitUtilities_mod, only: computeMeanAnomaly
    type (QuantizedPlanetaryOrbit) :: orbit
    real (kind=WP), intent(in) :: obliquity
    real (kind=WP), intent(in) :: eccentricity
    real (kind=WP), intent(in) :: longitudeAtPeriapsis
    real (kind=WP), intent(in) :: siderealPeriod
    real (kind=WP), intent(in) :: siderealRotationPeriod
    real (kind=WP), intent(in) :: meanDistance
    real (kind=WP), intent(in) :: hourAngleOffset

    integer :: siderealDaysPerYear

    type (TimeInterval) :: adjustedRotationPeriod
    type (TimeInterval) :: adjustedSiderealPeriod

    real (kind=WP) :: TOLERANCE = 1.d-4 ! seconds

    !--------------------------------------------------------------------------------------
    ! Rotation period is rounded to a rational with a small denominator.
    ! Then the year length is rounded to a rational that is an integral multiple of the sidereal
    ! day.
    ! Length of sidereal day is unimpacted.
    !--------------------------------------------------------------------------------------
    
    adjustedRotationPeriod = newTimeInterval(Rational(siderealRotationPeriod, TOLERANCE))
    siderealDaysPerYear = abs(nint(siderealPeriod / siderealRotationPeriod))
    if (siderealDaysPerYear == 0) then
       print*,'Warning:  sidereal day is longer than the sidereal year.'
       print*,'A quantized orbit is probably nonsensical for these settings, and'
       print*,'Automated adjustment of the orbital _period_ will be quite large.'
       print*,"You should consider using quantizeYearLength='false' in the rundeck."
       print*,'Proceeding with larger orbital period = sidereal rotation period.'
       siderealDaysPerYear = 1
    end if
    adjustedSiderealPeriod =  newTimeInterval(siderealDaysPerYear * adjustedRotationPeriod)

    ! Set parent component using adjusted values
    orbit%PlanetaryOrbit = PlanetaryOrbit(obliquity, eccentricity, longitudeAtPeriapsis, &
         & adjustedSiderealPeriod, adjustedRotationPeriod, meanDistance, hourAngleOffset)

  end function newQuantizedPlanetaryOrbit


  ! Pass instance of self to constructor for PlanetaryCalendar.
  function makeCalendar(this) result(calendar)
    use PlanetaryCalendar_mod
    use AbstractCalendar_mod
    class (AbstractCalendar), allocatable :: calendar
    class (QuantizedPlanetaryOrbit), intent(in) :: this

    allocate(calendar, source=PlanetaryCalendar(this))
  end function makeCalendar


  subroutine print_unit(this, unit)
    class (QuantizedPlanetaryOrbit), intent(in) :: this
    integer, intent(in) :: unit

    write(unit,*) 'Fixed orbital parameters for planet.'
    write(unit,*) '  Eccentricity:', this%getEccentricity()
    write(unit,*) '  Obliquity (degs):',this%getObliquity()
    write(unit,*) '  Longitude at periapsis (degs from ve):', &
         & this%getLongitudeAtPeriapsis()

  end subroutine print_unit

end module QuantizedPlanetaryOrbit_mod
