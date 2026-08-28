module TracerSource_mod
  use TimeConstants_mod, only: INT_HOURS_PER_DAY
  use constant, only : undef
  implicit none
  private

  public :: TracerSource
  public :: TracerSource3D

!@var applyDiurnalCycle whether or not to apply the factors in the diurnalCycle array
!@var diurnalCycle an optional diurnal cycle to be read from a file
  type TracerSource
    logical :: applyDiurnalCycle = .false.
    real*8 :: diurnalCycle(INT_HOURS_PER_DAY) = undef
  end type TracerSource

  type, extends(TracerSource) :: TracerSource3D
  end type TracerSource3D

end module TracerSource_mod
