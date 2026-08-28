!----------------------------------------------------------------------
!
! This representation of orbital mechanics is a bit biased by the
! requirements of supporting climate modeling.  In particular these
! interfaces support computation of zenith angle of the primary which
! in turn depends upon obliquity and rotation period in addition to
! traditional Keplerian orbital parameters.
!
! Note that the interfaces provided here do not assume an ideal
! Keplerian orbit.  (See FixedOrbit for that.)  Keplerian parameters
! can evolve slowly over time, which unfortunately means that a Time
! object must be passed to many of the methods that would seemingly be
! returning constants.
!
! The implementation does provide an interface for returning the
! "osculating" orbit - a fancy term for the orbit in the absence of
! perturbations).  Due to the requirements of Fortran, the FixedOrbit
! class must be implemented in the same module.
!
! One interesting feature of this design is that concrete
! implementations must implement the makeCalendar() method.  This
! allows the architecture to enforce that the Calendar and Orbit used
! by the model are consistent with one another.
!
! One awkward aspect of implementing orbits is that the constructors
! will generally have many parameters.  In many intances it may be
! better to use an AttributeDictionary than a long list of parameters.
!
! The implementation of getDeclinationAngle() and getHourAngle() are 
! based upon the original implementation by Gary Russel.  His implementation
! was based upon   V.M.Blanco and  S.W.McCuskey, 1961, 
! "Basic Physics of the Solar System", pages 135 - 151. 
! Note - Existence of Moon and heavenly bodies other than
! Earth and Sun are ignored.  Earth is assumed to be spherical.
!
!----------------------------------------------------------------------

module AbstractOrbit_mod
  use AbstractCalendar_mod
  use KindParameters_mod, only: WP => DP
  use Rational_mod
  use TimeInterval_mod
  use MathematicalConstants_mod, only: PI, RADIANS_PER_DEGREE
  use BaseTime_mod
  implicit none
  private

  public :: AbstractOrbit

  type, abstract :: AbstractOrbit
    private

    type (BaseTime) :: timeAtPeriapsis    ! seconds
    type (BaseTime) :: timeAtvernalEquinox    ! seconds

    type (TimeInterval) :: siderealOrbitalPeriod ! seconds
    type (TimeInterval) :: siderealRotationPeriod ! seconds
    type (TimeInterval) :: meanDay        ! seconds
    real(kind=WP) :: meanDistance  ! in astronomical units (AU's)
    real(kind=WP) :: hourAngleOffset ! degrees East from international dateline
    logical :: verbose = .false.

  contains

    ! Primary methods provide information necessary for 
    ! determining incoming radiation from primary.
    procedure :: getSinDeclinationAngle
    procedure :: getDeclinationAngle
    procedure :: getHourAngle
    procedure :: getMeanHourAngle
    procedure :: getDistance
    
    ! Secondary methods provide other orbital parameters
    procedure(setYear), deferred :: setYear
    procedure(getSlow), deferred :: getEccentricity
    procedure(getSlow), deferred :: getObliquity
    procedure(getSlow), deferred :: getLongitudeAtPeriapsis
    
    procedure :: getSiderealOrbitalPeriod
    procedure :: getSiderealRotationPeriod
    procedure :: getMeanDay
    procedure :: getMeanDistance
    procedure :: getHourAngleOffset
    procedure :: get_EOT ! equation of time

    procedure :: getTimeAtVernalEquinox
    procedure :: setTimeAtVernalEquinox

    procedure :: getTimeAtPeriapsis
    procedure :: setTimeAtPeriapsis
    
    procedure :: setSiderealRotationPeriod
    procedure :: setSiderealOrbitalPeriod
    procedure :: setMeanDay
    procedure :: setMeanDistance
    procedure :: setHourAngleOffset
    
    procedure :: getMeanAnomaly
    procedure :: getTrueAnomaly
    procedure :: isRetrograde
    
    procedure(makeCalendar), deferred :: makeCalendar

    procedure(print_unit), deferred :: print_unit
    procedure :: print_stdout
    generic :: print => print_unit, print_stdout

    procedure :: rotate

    procedure :: setVerbose
    procedure :: getVerbose

  end type AbstractOrbit


  abstract interface
     

    ! Constant - or nearly constant parameters 
    function getSlow(this) result(q)
      use KindParameters_mod, only: WP => DP
      import AbstractOrbit
      real (kind=WP) :: q
      class (AbstractOrbit), intent(in) :: this
    end function getSlow

    
    ! Modify slowly evolving values
    subroutine setYear(this, year)
      use KindParameters_mod, only: WP => DP
      import AbstractOrbit
      class (AbstractOrbit), intent(inout) :: this
      real(kind=WP), intent(in) :: year
    end subroutine setYear


    function get(this, t) result(q)
      use KindParameters_mod, only: WP => DP
      use BaseTime_mod
      import AbstractOrbit
      real (kind=WP) :: q
      class (AbstractOrbit), intent(in) :: this
      class (BaseTime), intent(in) :: t
    end function get


    function getInterval(this) result(q)
      use KindParameters_mod, only: WP => DP
      use BaseTime_mod
      use TimeInterval_mod, only: TimeInterval
      import AbstractOrbit
      type (TimeInterval) :: q
      class (AbstractOrbit), intent(in) :: this
    end function getInterval


    function makeCalendar(this) result(calendar)
      use AbstractCalendar_mod, only: AbstractCalendar
      import AbstractOrbit
      class (AbstractCalendar), allocatable :: calendar
      class (AbstractOrbit), intent(in) :: this
    end function makeCalendar


    subroutine print_unit(this, unit) 
      import AbstractOrbit
      class (AbstractOrbit), intent(in) :: this
      integer, intent(in) :: unit
    end subroutine print_unit


  end interface

contains


  function getSinDeclinationAngle(this, t) result(sinDeclinationAngle)
    use BaseTime_mod, only: BaseTime
    real (kind=WP) :: sinDeclinationAngle
    class (AbstractOrbit), intent(in) :: this
    class (BaseTime), intent(in) :: t

    real (kind=WP) :: trueAnomaly
    real (kind=WP) :: deltaAnomaly
    real (kind=WP) :: sinObliquity
    real (kind=WP), parameter :: pi = 2*asin(1.0d0)

    trueAnomaly = this%getTrueAnomaly(t)
    deltaAnomaly = trueAnomaly + this%getLongitudeAtPeriapsis()*RADIANS_PER_DEGREE

    sinObliquity = sin(this%getObliquity() * RADIANS_PER_DEGREE)
    sinDeclinationAngle = sinObliquity * sin(deltaAnomaly)

  end function getSinDeclinationAngle

  function getDeclinationAngle(this, t) result(declinationAngle)
    use BaseTime_mod, only: BaseTime
    real (kind=WP) :: declinationAngle
    class (AbstractOrbit), intent(in) :: this
    class (BaseTime), intent(in) :: t

    declinationAngle = asin(this%getSinDeclinationAngle(t))

  end function getDeclinationAngle


  function getMeanHourAngle(this, t) result(meanHourAngle)
    use Rational_mod
    use BaseTime_mod
    use TimeInterval_mod
    real (kind=WP) :: meanHourAngle
    class (AbstractOrbit), intent(in) :: this
    class (BaseTime), intent(in) :: t

    type (Rational) :: f
    real (kind=WP) :: offset

    type (Rational) :: f1, f2

    f1 = t/this%getSiderealRotationPeriod()
    f2 = t/this%getSiderealOrbitalPeriod()
    f = fraction(f1 - f2)
    offset = this%getHourAngleOffset() * RADIANS_PER_DEGREE

    meanHourAngle = modulo(2*PI * real(f) - offset, 2*PI)

 end function getMeanHourAngle


  function getHourAngle(this, t) result(hourAngle)
    use Rational_mod
    use BaseTime_mod
    use TimeInterval_mod
    real(kind=WP) :: hourAngle
    class (AbstractOrbit), intent(in) :: this
    class (BaseTime), intent(in) :: t

    real (kind=WP) :: meanHourAngle

    meanHourAngle = this%getMeanHourAngle(t)
    hourAngle = meanHourAngle + this%get_EOT(t)

  end function getHourAngle

  ! Equation of time
  function get_EOT(this, t) result(EOT)
    use BaseTime_mod
    use Dictionary_mod
    real(kind=WP) :: EOT
    class(AbstractOrbit), intent(in) :: this
    class(BaseTime), intent(in) :: t

    real(kind=WP) :: TA
    real(kind=WP) :: MA
    character(len=80) :: optEOT

    TA = this%getTrueAnomaly(t)
    MA = this%getMeanAnomaly(t)

    ! "Naive" legacy formula
    call get_param('EOT', optEOT)

    select case (trim(optEOT))
    case ('F','f','FALSE','False','false','.FALSE.','.False.','.false.','OFF','Off','off')
       EOT = 0
    case ('N','n','NAIVE','Naive','naive')
       ! This incorrect formula was introduced in the early planet work.
       ! It neglects the effect of obliquity on the EOT.
       EOT = modulo(MA - TA + pi, 2*pi) - pi
    case default
       associate (w => this%getObliquity() *RADIANS_PER_DEGREE, phi0 => this%getLongitudeAtPeriapsis() * RADIANS_PER_DEGREE)
         EOT =  modulo((phi0 + MA) - atan2(cos(w)*sin(phi0 + TA), cos(phi0 + TA)) + pi, 2*pi) - pi
       end associate
    end select

  end function get_EOT


  function getDistance(this, t) result(distance)
    use BaseTime_mod, only: BaseTime
    use OrbitUtilities_mod, only: computeDistance
    real(kind=WP) :: distance
    class (AbstractOrbit), intent(in) :: this
    class (BaseTime), intent(in) :: t

    associate(a => this%meanDistance, e => this%getEccentricity(), &
         & ma => this%getMeanAnomaly(t))
      distance = computeDistance(a, e, ma)
    end associate

  end function getDistance


  subroutine setMeanDistance(this, meanDistance)
    class (AbstractOrbit), intent(inout) :: this
    real(kind=WP), intent(in) :: meanDistance
    this%meanDistance = meanDistance
  end subroutine setMeanDistance


  subroutine setHourAngleOffset(this, hourAngleOffset)
    class (AbstractOrbit), intent(inout) :: this
    real(kind=WP), intent(in) :: hourAngleOffset ! degrees
    this%hourAngleOffset = hourAngleOffset
  end subroutine setHourAngleOffset


  function getMeanDistance(this) result(meanDistance)
    real(kind=WP) :: meanDistance
    class (AbstractOrbit), intent(in) :: this
    meanDistance = this%meanDistance
  end function getMeanDistance


  function getHourAngleOffset(this) result(hourAngleOffset)
    real(kind=WP) :: hourAngleOffset
    class (AbstractOrbit), intent(in) :: this
    hourAngleOffset = this%hourAngleOffset
  end function getHourAngleOffset


  subroutine setSiderealRotationPeriod(this, siderealRotationPeriod)
    use TimeInterval_mod, only: TimeInterval
    class (AbstractOrbit), intent(inout) :: this
    type (TimeInterval) :: siderealRotationPeriod
    this%siderealRotationPeriod = siderealRotationPeriod
  end subroutine setSiderealRotationPeriod


  function getSiderealRotationPeriod(this) result(siderealRotationPeriod)
    use TimeInterval_mod, only: TimeInterval
    type (TimeInterval) :: siderealRotationPeriod
    class (AbstractOrbit), intent(in) :: this
    siderealRotationPeriod = this%siderealRotationPeriod
  end function getSiderealRotationPeriod


  subroutine setMeanDay(this, meanDay)
    use TimeInterval_mod, only: TimeInterval
    class (AbstractOrbit), intent(inout) :: this
    type (TimeInterval) :: meanDay
    this%meanDay = meanDay
  end subroutine setMeanDay


  function getMeanDay(this) result(meanDay)
    use TimeInterval_mod, only: TimeInterval
    type (TimeInterval) :: meanDay
    class (AbstractOrbit), intent(in) :: this
    meanDay = this%meanDay
  end function getMeanDay


  subroutine setSiderealOrbitalPeriod(this, siderealOrbitalPeriod)
    use TimeInterval_mod, only: TimeInterval
    class (AbstractOrbit), intent(inout) :: this
    type (TimeInterval), intent(in) :: siderealOrbitalPeriod
    this%siderealOrbitalPeriod = siderealOrbitalPeriod
 end subroutine setSiderealOrbitalPeriod


  function getSiderealOrbitalPeriod(this) result(siderealOrbitalPeriod)
    use TimeInterval_mod, only: TimeInterval
    type (TimeInterval) :: siderealOrbitalPeriod
    class (AbstractOrbit), intent(in) :: this
    siderealOrbitalPeriod = this%siderealOrbitalPeriod
  end function getSiderealOrbitalPeriod


  ! For diagnsotic purposes - default to stdout
  subroutine print_stdout(this)
    use iso_fortran_env, only: OUTPUT_UNIT
    class (AbstractOrbit), intent(in) :: this

    call this%print(OUTPUT_UNIT)

  end subroutine print_stdout


  ! Useful method for determining times of orbit events such as
  ! solstices and equinoctes.
  ! Returns time after orbit has subtended angle (in radians) from
  ! time t.  
  function rotate(this, t, angle) result(newT)
    use OrbitUtilities_mod, only: computeMeanAnomaly
    use TimeInterval_mod
    use BaseTime_mod
    type (BaseTime) :: newT

    class (AbstractOrbit), intent(in) :: this
    type (BaseTime), intent(in) :: t
    real (kind=WP), intent(in) :: angle
    
    real (kind=WP) :: trueAnomaly
    real (kind=WP) :: M0, M1
    type (TimeInterval) :: tOrbit
    
    trueAnomaly =  this%getTrueAnomaly(t)
    M0 = this%getMeanAnomaly(t)
    M1 = computeMeanAnomaly(trueAnomaly + angle, this%getEccentricity())
    
    tOrbit = this%getSiderealOrbitalPeriod()
    newT = newBaseTime(t + Rational((M1-M0)/(2*PI) * real(tOrbit), 1.d-3))
  end function rotate


  logical function isRetrograde(this)
     use Rational_mod
     class (AbstractOrbit), intent(in) :: this
     type (Rational) :: zero
     zero = Rational(0)
     isRetrograde = (this%getSiderealRotationPeriod() < zero)
  end function isRetrograde

  subroutine setVerbose(this, verbose)
     class (AbstractOrbit), intent(inout) :: this
     logical, intent(in) :: verbose

     this%verbose = verbose

  end subroutine setVerbose

  logical function getVerbose(this)
     class (AbstractOrbit), intent(in) :: this
     
     getVerbose = this%verbose

  end function getVerbose


  subroutine setTimeAtPeriapsis(this)
     use MathematicalConstants_mod, only: PI, RADIAN
     use OrbitUtilities_mod, only: computeMeanAnomaly, computeTrueAnomaly
     use BaseTime_mod
     class (AbstractOrbit), intent(inout) :: this

     real (kind=WP) :: meanAnomaly
     real (kind=WP) :: trueAnomaly

     trueAnomaly = -this%getLongitudeAtPeriapsis() * RADIAN
     meanAnomaly = computeMeanAnomaly(trueAnomaly, this%getEccentricity())

     this%timeAtPeriapsis = newBaseTime(this%getTimeAtVernalEquinox() - &
          & Rational(meanAnomaly/(2*PI) * this%siderealOrbitalPeriod%toReal(), tolerance=1.d-3))

  end subroutine setTimeAtPeriapsis


  function getTimeAtPeriapsis(this) result(timeAtPeriapsis)
    use BaseTime_mod, only: BaseTime
    class (AbstractOrbit), intent(in) :: this
    type (BaseTime) :: timeAtPeriapsis
    timeAtPeriapsis = this%timeAtPeriapsis
  end function getTimeAtPeriapsis


  subroutine setTimeAtVernalEquinox(this, timeAtVernalEquinox)
    use BaseTime_mod, only: BaseTime
    class (AbstractOrbit), intent(inout) :: this
    type (BaseTime), intent(in) :: timeAtVernalEquinox
    this%timeAtVernalEquinox = timeAtVernalEquinox
  end subroutine setTimeAtVernalEquinox


  function getTimeAtVernalEquinox(this) result(timeAtVernalEquinox)
    use BaseTime_mod, only: BaseTime
    class (AbstractOrbit), intent(in) :: this
    type (BaseTime) :: timeAtVernalEquinox

    timeAtVernalEquinox = this%timeAtVernalEquinox

  end function getTimeAtVernalEquinox

  function getMeanAnomaly(this, t) result(meanAnomaly)
    use BaseTime_mod, only: BaseTime
    use Rational_mod
    real (kind=WP) :: meanAnomaly
    class (AbstractOrbit), intent(in) :: this
    class (BaseTime), intent(in) :: t

    type (Rational) :: f

    type (TimeInterval) :: P

    P = this%getSiderealOrbitalPeriod()
    f = modulo(t,P) - modulo(this%timeAtPeriapsis,P)
    f = f / P
    meanAnomaly = real(f) * (2*PI)

  end function getMeanAnomaly


  function getTrueAnomaly(this, t) result(trueAnomaly)
    use BaseTime_mod
    use OrbitUtilities_mod, only: computeTrueAnomaly
    real (kind=WP) :: trueAnomaly
    class (AbstractOrbit), intent(in) :: this
    class (BaseTime), intent(in) :: t

    real(kind=WP) :: meanAnomaly

    meanAnomaly = this%getMeanAnomaly(t)
    trueAnomaly = computeTrueAnomaly(meanAnomaly, this%getEccentricity())

  end function getTrueAnomaly


end module AbstractOrbit_mod

