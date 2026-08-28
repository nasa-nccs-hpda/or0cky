!---------------------------------------------------------------------
! FixedOrbit extends AbstractOrbit for the common case of an ideal
! Keplerian orbit.  I.e. eccentricity, obliquity, and longitude at
! Periapsis can all be treated as constants.
!
! FixedOrbit is still abstract, as different subclasses will likely
! have alternative constructors, but more importantly will almost
! certainly have different methods for making a corresponding
! Calendar.  E.g. Earth has calendars that while based upon
! astronomical angles, are also imbedded with significant history,
! while one will want Planetary Calendars to be guided by the Earthly
! (Julian) counterparts
!
!---------------------------------------------------------------------


module FixedOrbit_mod
  use AbstractOrbit_mod, only: AbstractOrbit
  use KindParameters_mod, only: WP => DP, DP
  use BaseTime_mod, only: BaseTime
  use TimeInterval_mod, only: TimeInterval
  use Rational_mod
  implicit none
  private

  public :: FixedOrbit

  type, abstract, extends(AbstractOrbit) :: FixedOrbit
    private
    
    real(kind=WP) :: eccentricity
    real(kind=WP) :: obliquity
    real(kind=WP) :: longitudeAtPeriapsis

    logical :: eccentricity_set = .false.
    logical :: obliquity_set = .false.
    logical :: longitudeAtPeriapsis_set = .false.

  contains

    procedure :: setYear

    procedure :: setEccentricity
    procedure :: getEccentricity

    procedure :: setObliquity
    procedure :: getObliquity

    procedure :: setLongitudeAtPeriapsis
    procedure :: getLongitudeAtPeriapsis

  end type FixedOrbit

  abstract interface

    function get(this, t) result (q)
      use KindParameters_mod, only: WP => DP
      import FixedOrbit
      class (FixedOrbit), intent(in) :: this

      real(kind=WP) :: q

    end function get

  end interface

  real (kind=DP), parameter :: PI = 2*asin(1.d0)

contains


  ! No-op
  subroutine setYear(this, year)
    class (FixedOrbit), intent(inout) :: this
    real(kind=WP), intent(in) :: year
  end subroutine setYear
  

  function getEccentricity(this) result(eccentricity)
    real(kind=WP) :: eccentricity
    class (FixedOrbit), intent(in) :: this
    eccentricity = this%eccentricity
  end function getEccentricity


  function getObliquity(this) result(obliquity)
    real(kind=WP) :: obliquity
    class (FixedOrbit), intent(in) :: this
    obliquity = this%obliquity
  end function getObliquity


  function getLongitudeAtPeriapsis(this) result(longitudeAtPeriapsis)
    real(kind=WP) :: longitudeAtPeriapsis
    class (FixedOrbit), intent(in) :: this
    longitudeAtPeriapsis = this%longitudeAtPeriapsis
  end function getLongitudeAtPeriapsis


  subroutine setEccentricity(this, eccentricity)
    class (FixedOrbit), intent(inout) :: this
    real(kind=WP), intent(in) :: eccentricity

    this%eccentricity = eccentricity
    this%eccentricity_set = .true.

    if (all([this%longitudeAtPeriapsis_set, this%eccentricity_set, this%obliquity_set])) then
       call this%setTimeAtPeriapsis()
    end if

  end subroutine setEccentricity


  subroutine setObliquity(this, obliquity)
    class (FixedOrbit), intent(inout) :: this
    real(kind=WP), intent(in) :: obliquity

    this%obliquity = obliquity
    this%obliquity_set = .true.

    if (all([this%longitudeAtPeriapsis_set, this%eccentricity_set, this%obliquity_set])) then
       call this%setTimeAtPeriapsis()
    end if

  end subroutine setObliquity


  subroutine setLongitudeAtPeriapsis(this, longitudeAtPeriapsis)
    class (FixedOrbit), intent(inout) :: this
    real(kind=WP), intent(in) :: longitudeAtPeriapsis

    this%longitudeAtPeriapsis = longitudeAtPeriapsis
    this%longitudeAtPeriapsis_set = .true.

    if (all([this%longitudeAtPeriapsis_set, this%eccentricity_set, this%obliquity_set])) then
       call this%setTimeAtPeriapsis()
    end if

  end subroutine setLongitudeAtPeriapsis


end module FixedOrbit_mod
