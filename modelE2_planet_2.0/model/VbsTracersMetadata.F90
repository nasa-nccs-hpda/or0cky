#include "rundeck_opts.h"
!------------------------------------------------------------------------------
module VbsTracersMetadata_mod
!------------------------------------------------------------------------------
!@sum  VbsTracersMetadata_mod encapsulates the VBS tracers metadata
!@auth NCCS ASTG
  use sharedTracersMetadata_mod, only: convert_HSTAR
  use OldTracer_mod, only: oldAddTracer
  use OldTracer_mod, only: nPart, nGAS
  use OldTracer_mod, only: set_tr_mm
  use OldTracer_mod, only: set_ntm_power
  use OldTracer_mod, only: set_trpdens
  use OldTracer_mod, only: set_trradius
  use OldTracer_mod, only: set_fq_aer
  use OldTracer_mod, only: set_tr_wd_type
  use OldTracer_mod, only: set_KH_298
  use OldTracer_mod, only: set_tr_RKD
  use OldTracer_mod, only: set_pm2p5fact
  use OldTracer_mod, only: set_pm10fact
  use OldTracer_mod, only: set_has_chemistry
  use OldTracer_mod, only: tr_RKD 
  use TRACER_COM, only: n_vbsGm2, n_vbsGm1, n_vbsGz,  n_vbsGp1, n_vbsGp2, &
                        n_vbsGp3, n_vbsGp4, n_vbsGp5, n_vbsGp6, &
                        n_vbsAm2, n_vbsAm1, n_vbsAz,  n_vbsAp1, n_vbsAp2, &
                        n_vbsAp3, n_vbsAp4, n_vbsAp5, n_vbsAp6
  use TRACER_COM, only: tracers
  use Dictionary_mod, only: sync_param
  use RunTimeControls_mod, only: tracers_drydep
  use RunTimeControls_mod, only: sulf_only_aerosols
  use RunTimeControls_mod, only: tracers_amp
  use Tracer_mod, only: Tracer
  use TRACERS_VBS, only: ivbs_m2,ivbs_m1,ivbs_m0,ivbs_p1,ivbs_p2,ivbs_p3,&
                         ivbs_p4,ivbs_p5,ivbs_p6

  implicit none
  private

  public VBS_initMetadata

  integer :: n ! class scoped temporary tracer index

!------------------------------------------------------------------------------
  contains
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
  subroutine VBS_InitMetadata(pTracer)
    use TRACER_COM, only: coupled_chem
!------------------------------------------------------------------------------
    class (Tracer), pointer :: pTracer

    if (.not. sulf_only_aerosols) then
      call  VBS_setSpec('vbsGm2', ivbs_m2,'igas')
      call  VBS_setSpec('vbsGm1', ivbs_m1,'igas')
      call  VBS_setSpec('vbsGz', ivbs_m0,'igas')
      call  VBS_setSpec('vbsGp1', ivbs_p1,'igas')
      call  VBS_setSpec('vbsGp2', ivbs_p2,'igas')
      call  VBS_setSpec('vbsGp3', ivbs_p3,'igas')
      call  VBS_setSpec('vbsGp4', ivbs_p4,'igas')
      call  VBS_setSpec('vbsGp5', ivbs_p5,'igas')
      call  VBS_setSpec('vbsGp6', ivbs_p6,'igas')

      if (.not. tracers_amp) then
        call  VBS_setSpec('vbsAm2', ivbs_m2,'iaer')
        call  VBS_setSpec('vbsAm1', ivbs_m1,'iaer')
        call  VBS_setSpec('vbsAz', ivbs_m0,'iaer')
        call  VBS_setSpec('vbsAp1', ivbs_p1,'iaer')
        call  VBS_setSpec('vbsAp2', ivbs_p2,'iaer')
        call  VBS_setSpec('vbsAp3', ivbs_p3,'iaer')
        call  VBS_setSpec('vbsAp4', ivbs_p4,'iaer')
        call  VBS_setSpec('vbsAp5', ivbs_p5,'iaer')
        call  VBS_setSpec('vbsAp6', ivbs_p6,'iaer')
      endif
    end if

!------------------------------------------------------------------------------
  contains
!------------------------------------------------------------------------------

    subroutine VBS_setSpec(name, index, type)
      use OldTracer_mod, only: om2oc, set_om2oc, set_is_VBS_tracer
#ifdef TRACERS_AEROSOLS_Koch
      use aerosol_sources, only: vbs_conc
#endif  /* TRACERS_AEROSOLS_Koch */
      implicit none
      character(len=*), intent(in) :: name
      real*8 :: tmp
      integer, intent(in) :: index
      character(len=4), intent(in) :: type

      n = oldAddTracer(name)
      call set_is_VBS_tracer(n, .true.)

      select case(name)
        case("vbsGm2"); n_vbsGm2 = n
        case("vbsGm1"); n_vbsGm1 = n
        case("vbsGz"); n_vbsGz = n
        case("vbsGp1"); n_vbsGp1 = n
        case("vbsGp2"); n_vbsGp2 = n
        case("vbsGp3"); n_vbsGp3 = n
        case("vbsGp4"); n_vbsGp4 = n
        case("vbsGp5"); n_vbsGp5 = n
        case("vbsGp6"); n_vbsGp6 = n
        case("vbsAm2"); n_vbsAm2 = n
        case("vbsAm1"); n_vbsAm1 = n
        case("vbsAz"); n_vbsAz = n
        case("vbsAp1"); n_vbsAp1 = n
        case("vbsAp2"); n_vbsAp2 = n
        case("vbsAp3"); n_vbsAp3 = n
        case("vbsAp4"); n_vbsAp4 = n
        case("vbsAp5"); n_vbsAp5 = n
        case("vbsAp6"); n_vbsAp6 = n
      end select
#ifdef TRACERS_AEROSOLS_Koch
      select case (type)
      case ('igas')
        vbs_conc(1)%igas(index) = n
      case ('iaer')
        vbs_conc(1)%iaer(index) = n
      end select
#endif  /* TRACERS_AEROSOLS_Koch */

      call set_om2oc(n, 1.4d0)
      tmp = om2oc(n)
      call sync_param(trim(name)//"_om2oc",tmp)
      call set_om2oc(n, tmp)
      call set_ntm_power(n, -11)
      tmp = 12.d0 * om2oc(n)
      call set_tr_mm(n, tmp)
      select case(name)
      case ('vbsGm2', 'vbsGm1', 'vbsGz',  'vbsGp1', 'vbsGp2', &! VBS gas-phase
        'vbsGp3', 'vbsGp4', 'vbsGp5', 'vbsGp6')
        call set_tr_wd_type(n, ngas)
        call set_tr_RKD(n, 1.d4 / convert_HSTAR ) !Henry; from mole/(L atm) to mole/J
      if (tracers_drydep) call set_KH_298(n, tr_RKD(n)*convert_HSTAR)
      case ('vbsAm2', 'vbsAm1', 'vbsAz',  'vbsAp1', 'vbsAp2', &! VBS aerosol-phase
            'vbsAp3', 'vbsAp4', 'vbsAp5', 'vbsAp6')
        call set_tr_wd_type(n, npart)
        call set_trpdens(n, 1.5d3) !kg/m3
        call set_trradius(n, 3.d-7 ) !m
        call set_fq_aer(n, 1.0d0   ) !fraction of aerosol that dissolves
        call set_pm2p5fact(n, 1.d0) ! fraction that's PM2.5
        call set_pm10fact(n, 1.d0) ! fraction that's PM10
        call set_has_chemistry(n, .true.)
      end select
    end subroutine VBS_setSpec

  end subroutine VBS_InitMetadata

end module VbsTracersMetadata_mod



