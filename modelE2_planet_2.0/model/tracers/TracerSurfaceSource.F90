#include "rundeck_opts.h"

module TracerSurfaceSource_mod
  use TracerSource_mod
  use timestream_mod, only : timestream
  implicit none
  private

!@var EMstream interface for reading and time-interpolating emissions file. See timestream_mod.
!@var sourceName holds source name from metadata, used to set up diagnostics.
!@var sourceLname holds long source name, used to set up diagnostics.
!@var tracerName tracer name associated with emissions file (often matches trname)
!@var skipReason when gt 0 tags sources to be skipped in trname_## file reading
!@var firstTrip indicates source initialization is needed
!@param itsMegan skip reason is online MEGAN vegetation source
!@param itsCH4MGOL skip reason is CH4 ocean, lake, misc ground source
!@param itsOcean skip reason is online ocean source
!@var SCstream interface for reading and time-interpolating optional scaling factors
!@var scaleIt logical whether or not to apply gridded scalings to the source

  public :: TracerSurfaceSource
  public :: initSurfaceSource
  public :: readSurfaceSource
  public :: itsMegan
  public :: itsOcean
  public :: itsCH4MGOL

  type, extends(TracerSource) :: TracerSurfaceSource
    character(len=30) :: sourceName
    character(len=30) :: sourceLname
    type (timestream) :: EMstream, SCstream
    character(len=10) :: tracerName
    integer :: skipReason = 0
    logical :: firstTrip = .true., scaleIt = .false.
  end type TracerSurfaceSource

  integer, parameter :: itsMegan=1
  integer, parameter :: itsCH4MGOL=2
  integer, parameter :: itsOcean=3

contains

  subroutine initSurfaceSource(this, tracerName, fileName)
    USE FILEMANAGER, only: openunit,closeunit
    use pario, only : par_open,par_close,read_attr
    USE DOMAIN_DECOMP_ATM, only: GRID
    use TimeConstants_mod, only: HOURS_PER_DAY
    use timestream_mod, only: getFirstFileByYear
    use SpecialIO_mod, only: write_parallel,read_parallel
    type (TracerSurfaceSource), intent(inout) :: this
    character(len=*), intent(in) :: tracerName
    character(len=*), intent(in) :: fileName
    character(len=300) :: out_line
    logical :: diurnalFileExists = .false.
    logical :: scalingFileExists = .false.

    integer :: nn, i, j, iu, fid
    character*32 :: pname,fileToRead
    character*35 :: fname
    character(len=80) :: name
    real*8 :: sumDiurnal
    real*8, parameter :: diurnalSumTolerance=1.d-4
    character*80 :: targetVariable
    integer :: nfileyrs
    integer, dimension(:), allocatable :: fileyrs

    ! -- Obtain metadata on how to label this source in diagnostics:

    call getFirstFileByYear &
     & (grid,trim(fileName),fileToRead,nfileyrs,fileyrs)

    ! continue reading netCDF file:
    this%tracerName = tracerName
    this%sourceName = 'notfound'
    fid = par_open(grid,trim(fileToRead),'read')
    ! First try to read the variable attribute to get source name:
    call read_attr(grid,fid,this%tracerName,'source',i,this%sourceName)
    ! If that fails, look for the source attribute of the variable
    ! that varname attribute points to (like init_stream would):
    if(trim(this%sourceName).eq.'notfound') then
      targetVariable=this%tracerName
      call read_attr(grid,fid,'global',trim(this%tracerName)//'name',&
      & i,targetVariable)
      call read_attr(grid,fid,trim(targetVariable),'source',&
      & i,this%sourceName)
    endif
    ! If that fails, look for a global source attribute:
    if(trim(this%sourceName).eq.'notfound') then
      call read_attr(grid,fid,'global','source',i,this%sourceName)
    endif
    ! If even that fails, stop the model:
    call par_close(grid,fid)
    if(trim(this%sourceName).eq.'notfound') then
      call stop_model('source name not found in file '//trim(fileName),255)
    endif

    ! append ' source' to the long name, and '_src' to the short name
    this%sourceLname = trim(this%sourceName)//' source'
    this%sourceName = trim(this%sourceName)//'_src'

    ! -- Determine if a user-defined diurnal cycle should be applied:
    !    (governed by file existance)
    fname=trim('diurnal_'//trim(fileName))
    inquire(file=trim(fname), exist=diurnalFileExists)
    if(diurnalFileExists)then
       this%applyDiurnalCycle=.true.
       write(out_line,*)'Applying diurnal cycle to file '//trim(fileName)
       call write_parallel(trim(out_line))
       call openunit(fname,iu,.false.,.true.)
       call read_parallel(this%diurnalCycle,iu)
       ! check that the diurnal cycle's sum is close to the number of hours
       ! in a day (meaning it's hourly average would be a factor of 1.):
       sumDiurnal=SUM(this%diurnalCycle)
       if(  sumDiurnal > HOURS_PER_DAY + diurnalSumTolerance  &
     & .or. sumDiurnal < HOURS_PER_DAY - diurnalSumTolerance) then
         write(out_line,*) &
     &   trim(fname),' sum is ',sumDiurnal,' not',HOURS_PER_DAY
         call write_parallel(trim(out_line))
         call stop_model('Problem with emissions diurnal cycle.',255)
       end if
       call closeunit(iu)
    end if

    ! -- Determine if a grid-based linear scaling should be applied:
    !    (governed by file existance)
    fname=trim(trim(fileName)//'_scale')
    inquire(file=trim(fname), exist=scalingFileExists)
    if(scalingFileExists)then
      this%scaleIt=.true.
    end if

  end subroutine initSurfaceSource


  subroutine readSurfaceSource(tracerName, this, fname, sfc_src, xyear, xday, isChemTracer)
    USE DOMAIN_DECOMP_ATM, only: GRID,  readt_parallel, write_parallel
    use Domain_decomp_atm, only: getDomainBounds
    use TimeConstants_mod, only: EARTH_DAYS_PER_YEAR
    use timestream_mod, only : init_stream,read_stream
    use dictionary_mod, only : get_param
    character(len=*), intent(in) :: tracerName
    type (TracerSurfaceSource), intent(inout) :: this
    character(*), intent(in) :: fname
    real*8, intent(inout) :: sfc_src(grid%i_strt:,grid%j_strt:)
    real*8 :: scaling( grid%i_strt:grid%i_stop, &
    &                  grid%j_strt:grid%j_stop )
    integer, intent(in) :: xyear, xday
    logical, intent(in) :: isChemTracer
    integer :: cyclic_yr,master_yr,nc_emis_use_ppm_interp

    ! first time after restarts; set up streams:
    if(this%firstTrip) then
      this%firstTrip = .false.
      call get_param('master_yr',master_yr)
      if (isChemTracer) then
        call get_param('o3_yr',cyclic_yr,default=master_yr)
        select case (tracerName)
        case ('NOx')
          call get_param('NOx_yr',cyclic_yr,default=cyclic_yr)
        case ('CO')
          call get_param('CO_yr',cyclic_yr,default=cyclic_yr)
        case ('Alkenes', 'Paraffin')
          call get_param('VOC_yr',cyclic_yr,default=cyclic_yr)
        end select
      else
        call get_param('aer_int_yr',cyclic_yr,default=master_yr)
        select case (tracerName)
        case ('SO2', 'SO4', 'M_ACC_SU', 'M_AKK_SU', 'ASO4__01')
          call get_param('SO2_int_yr',cyclic_yr,default=cyclic_yr)
        case ('NH3')
          call get_param('NH3_int_yr',cyclic_yr,default=cyclic_yr)
        case ('BCII', 'BCB', 'M_BC1_BC', 'M_BOC_BC', 'AECOB_01')
          call get_param('BC_int_yr',cyclic_yr,default=cyclic_yr)
        case ('OCII', 'OCB', 'M_OCC_OC', 'M_BOC_OC', 'AOCOB_01',&
              'vbsAm2', 'vbsAm1', 'vbsAz', 'vbsAp1', 'vbsAp2',&
              'vbsAp3', 'vbsAp4', 'vbsAp5', 'vbsAp6')
          call get_param('OC_int_yr',cyclic_yr,default=cyclic_yr)
        end select
      end if
      call get_param('nc_emis_use_ppm_interp',nc_emis_use_ppm_interp,&
        & default=1)
      if (nc_emis_use_ppm_interp==1) then
        call init_stream(grid,this%EMstream,trim(fname), &
           trim(this%tracername),0d0,1d30,'ppm',xyear,xday, &
           cyclic = (cyclic_yr > 0) )
      else
        call init_stream(grid,this%EMstream,trim(fname), &
           trim(this%tracername),0d0,1d30,'linm2m',xyear,xday, &
           cyclic = (cyclic_yr > 0) )
      endif
      if(this%scaleIt)then
        call init_stream(grid,this%SCstream,trim(fname)//'_scale', &
           'scale',-1d10,1d10,'linm2m',xyear,xday, &
           cyclic = (cyclic_yr > 0) )
      endif
    endif

    ! update source stream:
    call read_stream(grid,this%EMstream,xyear,xday,sfc_src)

    ! If a scaling file was listed for this source, update that
    ! scaling timstream as well. Then, scale the source immediately:
    if(this%scaleIt) then
      call read_stream(grid,this%SCstream,xyear,xday,scaling)
      sfc_src=sfc_src*scaling
    endif

  end subroutine readSurfaceSource

end module TracerSurfaceSource_mod
