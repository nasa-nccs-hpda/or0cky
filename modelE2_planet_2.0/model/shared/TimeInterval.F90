module TimeInterval_mod
  use BaseTime_mod, only: BaseTime, real
  use KindParameters_mod, only: DP
  implicit none
  private

  public :: TimeInterval
  public :: newTimeInterval ! ifort 15.0 workaround
  public :: real ! from BaseTime and Rational

  type, extends(BaseTime) :: TimeInterval
     integer :: placeholder
  end type TimeInterval

!!$  interface TimeInterval
!!$     module procedure newTimeInterval_DP
!!$     module procedure newTimeInterval_rational
!!$     module procedure newTimeInterval_integer
!!$  end interface TimeInterval
!!$
  interface newTimeInterval
     module procedure newTimeInterval_DP
     module procedure newTimeInterval_rational
     module procedure newTimeInterval_integer
  end interface NewTimeInterval


  real (kind=DP), parameter :: DEFAULT_TOLERANCE = 0.001 ! milliseconds

contains

  function newTimeInterval_DP(dt, tolerance) result(interval)
    use Rational_mod
    type (TimeInterval) :: interval
    real (kind=DP), intent(in) :: dt
    real (kind=DP), optional, intent(in) :: tolerance
    real (kind=DP) :: tolerance_

    tolerance_ = DEFAULT_TOLERANCE
    if (present(tolerance)) tolerance_ = tolerance

    interval = newTimeInterval(Rational(dt, tolerance_))

  end function newTimeInterval_DP

  function newTimeInterval_rational(r) result(interval)
    use Rational_mod
    type (TimeInterval) :: interval
    type (Rational), intent(in) :: r
    interval%Rational = r
  end function newTimeInterval_rational

  function newTimeInterval_integer(i) result(interval)
    use Rational_mod
    type (TimeInterval) :: interval
    integer, intent(in) :: i
    interval%Rational = Rational(i)
  end function newTimeInterval_integer

end module TimeInterval_mod
