
module PlanetaryCalendar_mod
  use AbstractCalendar_mod
  use FixedCalendar_mod
  use FixedOrbit_mod
  use AbstractOrbit_mod
  use CalendarMonth_mod, only: CalendarMonth
  use KindParameters_mod, only: WP => DP
  use TimeInterval_mod
  use Rational_mod
  implicit none
  private

  public :: PlanetaryCalendar

  type, extends(FixedCalendar) :: PlanetaryCalendar
     private
     integer :: foo
   contains
  end type PlanetaryCalendar

  interface PlanetaryCalendar
     module procedure newPlanetaryCalendar_likeJulian
     module procedure newPlanetaryCalendar_keplerian
     module procedure newPlanetaryCalendar_longitudes
  end interface PlanetaryCalendar

  real (kind=WP), parameter :: PI = 2*asin(1.d0)

  integer :: minCalendarDaysPerYear = 120
  integer :: monthPeriod = 1

contains


  ! Create a calendar that preserves the longitudes of the month
  ! starts relative to the vernal equinox.  This choice results in a
  ! correspondence between months and seasons that will be familiar to
  ! climate scientists.  Of course if the eccentricity is large and
  ! the periapsis is in the Northern summer, the results might be a
  ! bit counterintuitive

  ! Note that if the rotational period of a planet is sufficiently
  ! slow this approach may result in months that have zero days.  The
  ! code will issue a message and terminate under such conditions.
  ! The most obvious case for this would be for a tidally locked
  ! planet.

  function newPlanetaryCalendar_likeJulian(orbit) result(calendar)
    use AbstractOrbit_mod
    use FixedOrbit_mod
    use Earth365DayOrbit_mod, only: Earth365DayOrbit
    type (PlanetaryCalendar) :: calendar
    class (FixedOrbit), intent(in) :: orbit

    calendar = PlanetaryCalendar(orbit, Earth365DayOrbit())
    
  end function newPlanetaryCalendar_likeJulian


  function newPlanetaryCalendar_keplerian(orbit, referenceOrbit) result(calendar)
    use AbstractOrbit_mod
    use FixedOrbit_mod
    use AbstractCalendar_mod, only: AbstractCalendar
    use Rational_mod
    USE dictionary_mod, ONLY: sync_param
    type (PlanetaryCalendar) :: calendar
    class (FixedOrbit), intent(in) :: orbit
    class (FixedOrbit), intent(in) :: referenceOrbit

    class (AbstractCalendar), allocatable :: referenceCalendar
    integer :: i
    character(len=16) :: date_of_vernal_equinox
    real (kind=WP) :: longitudes(MONTHS_PER_YEAR),Lp

    allocate(referenceCalendar, source=referenceOrbit%makeCalendar())
    date_of_vernal_equinox = 'earth_default'
    call sync_param('date_of_vernal_equinox', date_of_vernal_equinox)

    do i = 1, MONTHS_PER_YEAR
       select case (date_of_vernal_equinox)
       case ('earth_default')
          longitudes(i) = referenceOrbit%getTrueAnomaly(referenceCalendar%convertToTime(1,i,1,0)) + orbit%getLongitudeAtPeriapsis()*PI/180.
       case ('jan01')
          longitudes(i) = 2. * PI * (i-1) / MONTHS_PER_YEAR
       case default
          call stop_model('PlanetaryCalendar unsupported date_of_vernal_equinox.')
       end select
      if (calendar%getVerbose()) then
         print*,'month=',i,'solar longitude = ',longitudes(i)*180/PI
      end if
    end do

    calendar = PlanetaryCalendar(orbit, longitudes)

    deallocate(referenceCalendar)
      
  end function newPlanetaryCalendar_keplerian


  ! This intermediate constructor uses the longitudes of month starts as 
  ! determiners of month lengths for the specified orbit.
  function newPlanetaryCalendar_longitudes(orbit, monthLongitudes) result(calendar)
    use OrbitUtilities_mod, only: computeMeanAnomaly
    use AbstractOrbit_mod
    use FixedOrbit_mod
    use JulianCalendar_mod, only: JULIAN_MONTHS
    use OrbitUtilities_mod, only: computeMeanAnomaly
    use Rational_mod
    use TimeInterval_mod
    use BaseTime_mod
    use Dictionary_mod
    type (PlanetaryCalendar) :: calendar
    class (FixedOrbit), intent(in) :: orbit
    real(kind=WP), intent(in) :: monthLongitudes(:) ! in radians

    real(kind=WP) :: M     ! mean anomaly
    real(kind=WP) :: M0    ! mean anomaly for Jan 01
    real(kind=WP) :: trueAnomaly
    integer :: i, j
    real(kind=WP) :: Lp
    type (CalendarMonth) :: months(0:MONTHS_PER_YEAR+1)
    type (BaseTime) :: tVE
    integer :: daysPerYear
    
    call calendar%setSecondsPerDay(orbit%getMeanDay())

    daysPerYear = nint(orbit%getSiderealOrbitalPeriod()/orbit%getMeanDay())
    ! Calendar does not care whether an orbit is retrograde.
    daysPerYear = abs(daysPerYear)

    ! Require a minimum number of days per year. (Suggested by G. Schmidt.)
    call sync_param('minCalendarDaysPerYear', minCalendarDaysPerYear)
    if (daysPerYear < minCalendarDaysPerYear) then
       if (calendar%getVerbose()) then
          write(*,*) '***********************************************************'
          write(*,*) '* Warning calendar days do not correspond to solar days.  *'
          write(*,*) '* Hourly diagnostics should not be used.                  *'
          write(*,*) '***********************************************************'
       end if
       daysPerYear = minCalendarDaysPerYear
       call calendar%setSecondsPerDay( &
           & newTimeInterval( orbit%getSiderealOrbitalPeriod() / daysPerYear ))
    end if
    
    call calendar%setDaysPerYear(daysPerYear)

    if (size(monthLongitudes) /= MONTHS_PER_YEAR) then
       call stop_model('PlanetaryCalendar assumes 12 month years.')
       return
    end if

    trueAnomaly = monthLongitudes(1) - orbit%getLongitudeAtPeriapsis()*PI/180
    M0 = computeMeanAnomaly(trueAnomaly, orbit%getEccentricity())

    months%fullName = JULIAN_MONTHS%fullName
    months%abbreviation = JULIAN_MONTHS%abbreviation

    ! First pass - just determine first day in month
    months(1)%firstDayInMonth = 1
    months(MONTHS_PER_YEAR+1)%firstDayInMonth = 1 + calendar%getDaysInYear()
    call sync_param('monthPeriod', monthPeriod)

    do i = 2, MONTHS_PER_YEAR

       j = 1 + (1 + (i-2)/monthPeriod) * monthPeriod

       if (j <= MONTHS_PER_YEAR) then
          trueAnomaly = monthLongitudes(j) - orbit%getLongitudeAtPeriapsis()*PI/180
          M     = modulo(computeMeanAnomaly(trueAnomaly, orbit%getEccentricity()) - M0, 2*PI)
          
          months(i)%firstDayInMonth = 1 + nint(M * calendar%getDaysInYear() / (2*PI))
          ! force the 1st month to have at least one day
          months(i)%firstDayInMonth = max(2, months(i)%firstDayInMonth)
       else
          months(i)%firstDayInMonth = 1 + calendar%getDaysInYear()
       end if

    end do
    months(0)%firstDayInMonth = months(MONTHS_PER_YEAR)%firstDayInMonth - calendar%getDaysInYear()


    ! second pass - remaining data
    months(0)%lastDayInMonth = 0
    months(MONTHS_PER_YEAR)%lastDayInMonth = calendar%getDaysInYear()

    do i = 1, MONTHS_PER_YEAR - 1
       months(i)%lastDayInMonth = months(i+1)%firstDayInMonth - 1
    end do
    months(MONTHS_PER_YEAR+1)%lastDayInMonth = months(1)%lastDayInMonth + calendar%getDaysInYear()

    ! And finally mid day, days per,  ...
    months(:)%midDayInMonth = (months(:)%firstDayInMonth + months(:)%lastDayInMonth)/2
    months(:)%daysInMonth = 1 + months(:)%lastDayInMonth - months(:)%firstDayInMonth

    do i = 0, MONTHS_PER_YEAR+1
      call calendar%setNthCalendarMonth(i, months(i))
    end do


    ! Special dates

    call calendar%initTransitionDates()

    tVE = orbit%getTimeAtVernalEquinox()

    Lp = orbit%getLongitudeAtPeriapsis()*(PI/180)
    call calendar%addTransitionDate('periapsis', calendar%getAnniversaryDate(orbit%rotate(tVE, Lp)))
    call calendar%addTransitionDate('apsis', calendar%getAnniversaryDate(orbit%rotate(tVE, Lp+PI)))

    call calendar%addTransitionDate('vernal equinox', calendar%getAnniversaryDate(orbit%rotate(tVE, 0.d0)))
    call calendar%addTransitionDate('summer solstice', calendar%getAnniversaryDate(orbit%rotate(tVE, PI/2)))
    call calendar%addTransitionDate('autumnal equinox', calendar%getAnniversaryDate(orbit%rotate(tVE, PI)))
    call calendar%addTransitionDate('winter solstice', calendar%getAnniversaryDate(orbit%rotate(tVE, 3*PI/2)))

 end function newPlanetaryCalendar_longitudes

end module PlanetaryCalendar_mod
