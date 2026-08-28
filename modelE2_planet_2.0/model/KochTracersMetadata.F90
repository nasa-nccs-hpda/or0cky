#include "rundeck_opts.h"
!------------------------------------------------------------------------------
module KochTracersMetadata_mod
!------------------------------------------------------------------------------
!@sum  KochTracersMetadata_mod encapsulates the KOCH tracers metadata
!@auth NCCS ASTG
  use sharedTracersMetadata_mod, only: DMS_setspec, &
    SO2_setspec, H2O2_s_setspec
  use OldTracer_mod, only: oldAddTracer
  use OldTracer_mod, only: nPart, nGAS
  use OldTracer_mod, only: set_tr_mm
  use OldTracer_mod, only: set_ntm_power
  use OldTracer_mod, only: set_trpdens
  use OldTracer_mod, only: set_trradius
  use OldTracer_mod, only: set_fq_aer
  use OldTracer_mod, only: set_tr_wd_type
  use OldTracer_mod, only: set_tr_RKD
  use OldTracer_mod, only: set_pm2p5fact
  use OldTracer_mod, only: set_pm10fact
  use OldTracer_mod, only: set_has_chemistry
  use OldTracer_mod, only: tr_RKD 
  use OldTracer_mod, only: set_hygro_oma
  use TRACER_COM, only:  n_MSA, n_SO4, n_DMS, &
    n_BCII,  n_BCIA,  n_BCB, n_OCII,  n_OCIA,  n_OCB, n_H2O2_s
  use TRACER_COM, only: tracers
  use Dictionary_mod, only: sync_param
  use RunTimeControls_mod, only: tracers_drydep
  use RunTimeControls_mod, only: sulf_only_aerosols
  use RunTimeControls_mod, only: tracers_special_shindell
  use Tracer_mod, only: Tracer

  implicit none
  private

  public Koch_initMetadata

  integer :: n ! class scoped temporary tracer index

!------------------------------------------------------------------------------
  contains
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
  subroutine KOCH_InitMetadata(pTracer)
    use TRACER_COM, only: coupled_chem
!------------------------------------------------------------------------------
    class (Tracer), pointer :: pTracer

    call  DMS_setSpec('DMS')
    call  MSA_setSpec('MSA')
    call  SO2_setSpec('SO2')
    call  SO4_setSpec('SO4')
    if (.not. tracers_special_shindell .or. coupled_chem.le.0) then
      call  H2O2_s_setSpec('H2O2_s')
    end if
    if (.not. sulf_only_aerosols) then
      call  BCII_setSpec('BCII')
      call  BCIA_setSpec('BCIA')
      call  BCB_setSpec('BCB')
#ifndef TRACERS_AEROSOLS_VBS
      call  OCII_setSpec('OCII')   !Insoluble industrial organic mass
      call  OCIA_setSpec('OCIA')   !Aged industrial organic mass
      call  OCB_setSpec('OCB')     !Biomass organic mass
#endif /* not TRACERS_AEROSOLS_VBS */
    end if

!------------------------------------------------------------------------------
  contains
!------------------------------------------------------------------------------

    subroutine MSA_setSpec(name)
      character(len=*), intent(in) :: name
      n = oldAddTracer(name)
      n_MSA = n
      call set_ntm_power(n, -13)
      call set_tr_mm(n, 96.d+0) !(H2O2 34;SO2 64)
      call set_trpdens(n, 1.7d3) !kg/m3 this is sulfate value
      call set_trradius(n, 5.d-7 ) !m (SO4 3;BC 1;OC 3)
      call set_fq_aer(n, 1.0d0   ) !fraction of aerosol that dissolves
      call set_tr_wd_type(n, npart)
      call set_pm2p5fact(n, 1.d0) ! fraction that's PM2.5
      call set_pm10fact(n, 1.d0) ! fraction that's PM10
      call set_has_chemistry(n, .true.)
      call set_hygro_oma(n,0.8d0)
    end subroutine MSA_setSpec

    subroutine SO4_setSpec(name)
      character(len=*), intent(in) :: name
      type (Tracer), pointer :: t
      n = oldAddTracer(name)
      n_SO4 = n 
      call set_ntm_power(n, -11)
      call set_tr_mm(n, 96.d+0)
      call set_trpdens(n, 1.7d3) !kg/m3 this is sulfate value
      call set_trradius(n, 3.d-7 ) !m
      call set_fq_aer(n, 1.d0   ) !fraction of aerosol that dissolves
      call set_tr_wd_type(n, npart)
      call set_pm2p5fact(n, 1.d0) ! fraction that's PM2.5
      call set_pm10fact(n, 1.d0) ! fraction that's PM10
      call set_has_chemistry(n, .true.)
      call set_hygro_oma(n,0.53d0) 
      t => tracers%getReference(trim(name))
      call t%insert('SO4',.true.)

    end subroutine SO4_setSpec

    subroutine BCII_setSpec(name)
      character(len=*), intent(in) :: name
      type (Tracer), pointer :: t
      n = oldAddTracer(name)
      n_BCII = n
      call set_ntm_power(n, -12)
      call set_tr_mm(n, 12.d0)
      call set_trpdens(n, 1.3d3) !kg/m3
      call set_trradius(n, 1.d-7 ) !m
      call set_fq_aer(n, 0.0d0   ) !fraction of aerosol that dissolves
      call set_tr_wd_type(n, npart)
      call set_pm2p5fact(n, 1.d0) ! fraction that's PM2.5
      call set_pm10fact(n, 1.d0) ! fraction that's PM10
      call set_has_chemistry(n, .true.)
      call set_hygro_oma(n,0.01d0)

      t => tracers%getReference(trim(name))
      call t%insert('BC',.true.)
      
    end subroutine BCII_setSpec

    subroutine BCIA_setSpec(name)
      character(len=*), intent(in) :: name
      type (Tracer), pointer :: t
      n = oldAddTracer(name)
      n_BCIA = n
      call set_ntm_power(n, -12)
      call set_tr_mm(n, 12.d0)
      call set_trpdens(n, 1.3d3) !kg/m3
      call set_trradius(n, 1.d-7 ) !m
      call set_fq_aer(n, 1.d0   ) !fraction of aerosol that dissolves
      call set_tr_wd_type(n, npart)
      call set_pm2p5fact(n, 1.d0) ! fraction that's PM2.5
      call set_pm10fact(n, 1.d0) ! fraction that's PM10
      call set_has_chemistry(n, .true.)
      call set_hygro_oma(n,0.01d0)

      t => tracers%getReference(trim(name))
      call t%insert('BC',.true.)

    end subroutine BCIA_setSpec

    subroutine BCB_setSpec(name)
      character(len=*), intent(in) :: name
      type (Tracer), pointer :: t
      real*8 :: tmp
      n = oldAddTracer(name)
      n_BCB = n
      call set_ntm_power(n, -12)
      call set_tr_mm(n, 12.d0)
      call set_trpdens(n, 1.3d3) !kg/m3
      call set_trradius(n, 1.d-7 ) !m
      tmp = 0.8d0
      call sync_param("BCB_fq_aer",tmp)
      call set_fq_aer(n, tmp ) !fraction of aerosol that dissolves
      call set_tr_wd_type(n, npart)
      call set_pm2p5fact(n, 1.d0) ! fraction that's PM2.5
      call set_pm10fact(n, 1.d0) ! fraction that's PM10
      call set_hygro_oma(n,0.01d0)
      t => tracers%getReference(trim(name))
      call t%insert('BC',.true.)
    end subroutine BCB_setSpec

    subroutine OCII_setSpec(name)
      use OldTracer_mod, only: om2oc, set_om2oc
      character(len=*), intent(in) :: name
      real*8 :: tmp
      n = oldAddTracer(name)
      n_OCII = n
      call set_om2oc(n, 1.4d0)
      tmp = om2oc(n)
      call sync_param("OCII_om2oc",tmp)
      call set_om2oc(n, tmp)
      call set_ntm_power(n, -11)
      tmp = 12.d0 * om2oc(n)
      call set_tr_mm(n, tmp)
      call set_trpdens(n, 1.5d3) !kg/m3
      call set_trradius(n, 3.d-7 ) !m
      call set_fq_aer(n, 0.0d0   ) !fraction of aerosol that dissolves
      call set_tr_wd_type(n, npart)
      call set_pm2p5fact(n, 1.d0) ! fraction that's PM2.5
      call set_pm10fact(n, 1.d0) ! fraction that's PM10
      call set_has_chemistry(n, .true.)
      call set_hygro_oma(n,0.15d0)
    end subroutine OCII_setSpec

    subroutine OCIA_setSpec(name)
      use OldTracer_mod, only: om2oc, set_om2oc
      character(len=*), intent(in) :: name
      real*8 :: tmp
      n = oldAddTracer(name)
      n_OCIA = n
      call set_om2oc(n, 1.4d0)
      tmp = om2oc(n)
      call sync_param("OCIA_om2oc",tmp)
      call set_om2oc(n, tmp)
      call set_ntm_power(n, -11)
      tmp = 12.d0 * om2oc(n)
      call set_tr_mm(n, tmp)
      call set_trpdens(n, 1.5d3) !kg/m3
      call set_trradius(n, 3.d-7 ) !m
      call set_fq_aer(n, 1.d0   ) !fraction of aerosol that dissolves
      call set_tr_wd_type(n, npart)
      call set_pm2p5fact(n, 1.d0) ! fraction that's PM2.5
      call set_pm10fact(n, 1.d0) ! fraction that's PM10
      call set_has_chemistry(n, .true.)
      call set_hygro_oma(n,0.15d0)
    end subroutine OCIA_setSpec

    subroutine OCB_setSpec(name)
      use OldTracer_mod, only: om2oc, set_om2oc
      character(len=*), intent(in) :: name
      real*8 :: tmp
      n = oldAddTracer(name)
      n_OCB = n
      call set_om2oc(n, 1.4d0)
      tmp = om2oc(n)
      call sync_param("OCB_om2oc",tmp)
      call set_om2oc(n, tmp)
      call set_ntm_power(n, -11)
      tmp = 12.d0 * om2oc(n)
      call set_tr_mm(n, tmp)
      call set_trpdens(n, 1.5d3) !kg/m3
      call set_trradius(n, 3.d-7 ) !m
      tmp = 0.8d0
      call sync_param("OCB_fq_aer",tmp)
      call set_fq_aer(n, tmp ) !fraction of aerosol that dissolves
      call set_tr_wd_type(n, npart)
      call set_pm2p5fact(n, 1.d0) ! fraction that's PM2.5
      call set_pm10fact(n, 1.d0) ! fraction that's PM10
      call set_hygro_oma(n,0.15d0)
    end subroutine OCB_setSpec

  end subroutine KOCH_InitMetadata

end module KochTracersMetadata_mod



