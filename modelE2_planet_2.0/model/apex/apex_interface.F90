! The modules and routines in this file have the intention of
! creating an "interface" between modelE and the apex.F90 code.
! Essentially, mimicing/replacing that code's reference to other
! NCAR modules/routines, just for our purposes and with the intention
! of making few changes in that code itself. This is why some of the
! code below looks ridiculous.
!@auth Greg Faluvegi, but again, just in the authorship sense noted above.

module shr_kind_mod
  implicit none
  integer, parameter :: SHR_KIND_R8 = selected_real_kind(12)
end module shr_kind_mod


module cam_logfile
  implicit none
  integer, parameter :: iulog = 6 ! suits us, b/c --> PRT
end module cam_logfile


module cam_abortutils
   implicit none
   private
   public :: endrun
contains
   subroutine endrun(msg)
     character(len=*), intent(in) :: msg
     call stop_model(msg, 255)
   end subroutine endrun
end module cam_abortutils


module spmd_utils
   use MpiSupport_mod, only: masterproc=>am_i_root
end module spmd_utils


module ioFileMod
implicit none
private
public :: getfil
contains
  subroutine getfil(fulpath, locfn, iflag)
    character(len=*), intent(in)  :: fulpath
    character(len=*), intent(out) :: locfn
    integer, optional, intent(in) :: iflag
    locfn=fulpath
  end subroutine getfil
end module ioFileMod


module pio
! TODO: I guess since all processors need to know these coefficients,
! for now simply have each read the cdf input, rather than using
! pario module and then broadcasting. But there is likely a better
! way I should do this, once it is working.
implicit none
private
include 'netcdf.inc'
public :: pio_inq_dimid, pio_inq_dimlen, pio_inq_varid, pio_get_var
public :: pio_closefile, pio_stop, PIO_NOWRITE
character(len=80) :: act
type, public :: File_desc_t ! i.e. this is not the full NCAR type
  integer :: fh
end type File_desc_t

integer, parameter :: PIO_NOWRITE=nf_nowrite

interface pio_get_var
  module procedure pio_get_var_2Dr8
  module procedure pio_get_var_int
end interface pio_get_var

contains

  integer function pio_inq_dimid(f, theName, theId)
#include "getIdByName.inc"
    act='getting dimension ID'
    pio_inq_dimid = nf_inq_dimid(f%fh, trim(theName), theId)
    if(pio_inq_dimid /= nf_noerr) call pio_stop(act, pio_inq_dimid)
  end function pio_inq_dimid

  integer function pio_inq_dimlen(f, theId, len)
#include "getById.inc"
    integer, intent(out) :: len
    act='getting dimension length'
    pio_inq_dimlen = nf_inq_dimlen( f%fh, theId, len)
    if(pio_inq_dimlen /= nf_noerr) call pio_stop(act, pio_inq_dimlen)
  end function pio_inq_dimlen

  integer function pio_inq_varid(f, theName, theId)
#include "getIdByName.inc"
    act='getting variable ID'
    pio_inq_varid = nf_inq_varid( f%fh, trim(theName), theId)
    if(pio_inq_varid /= nf_noerr) call pio_stop(act, pio_inq_varid)
  end function pio_inq_varid

  integer function pio_get_var_2Dr8( f, theId, vout )
    use shr_kind_mod,  only : r8 => shr_kind_r8
    real(r8) :: vout(:,:)
#include "getById.inc"
    act='reading 2D double variable'
    pio_get_var_2Dr8 = nf_get_var_double( f%fh, theId, vout)
    if(pio_get_var_2Dr8 /= nf_noerr) call pio_stop(act, pio_get_var_2Dr8)
  end function pio_get_var_2Dr8

  integer function pio_get_var_int( f, theId, vout )
    integer :: vout
#include "getById.inc"
    act='reading scalar integer variable'
    pio_get_var_int = nf_get_var_int( f%fh, theId, vout)
    if(pio_get_var_int /= nf_noerr) call pio_stop(act, pio_get_var_int)
  end function pio_get_var_int

  subroutine pio_closefile(f)
    type(file_desc_t), intent(in) :: f
    integer :: ierr
    act='closing the file'
    ierr = nf_close( f%fh )
    if(ierr /= nf_noerr) call pio_stop(act, ierr)
  end subroutine pio_closefile

  subroutine pio_stop(activity, ierror)
    use MpiSupport_mod, only: am_i_root
    integer, intent(in) :: ierror
    character(len=80), intent(in) :: activity
    if (am_i_root()) &
      & print *, 'apex_mod: while model was '//trim(activity) &
      & //', encountered the NF error: '//trim(nf_strerror(ierror))
      call stop_model('apex_mod: see PRT for detailed message.',255)
  end subroutine pio_stop
end module pio


module cam_pio_utils
  implicit none
  private
  public :: cam_pio_openfile
  include 'netcdf.inc'
contains
  subroutine cam_pio_openfile(f, fname, mode)
    use pio, only : file_desc_t, pio_stop
    type(file_desc_t), intent(inout), target :: f
    character(len=*), intent(in) :: fname
    integer, intent(in) :: mode
    integer :: ierr
    character(len=80) :: act='opening the file'
    ierr = nf_open( trim(fname), mode, f%fh)
    if(ierr /= nf_noerr) call pio_stop(act, ierr)
  end subroutine cam_pio_openfile
end module cam_pio_utils
