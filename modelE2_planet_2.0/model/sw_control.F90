!-------------------------------------------------------------------------------
! Control options for planet radiation (SOCRATES)
!-------------------------------------------------------------------------------

MODULE sw_control

  USE realtype_rd,      ONLY: RealK
  USE missing_data_mod, ONLY: rmdi, imdi
  USE rad_pcf

  IMPLICIT NONE

! Physical processes
  LOGICAL :: l_microphysics_sw    = .TRUE.
!   Flag for microphysics
  LOGICAL :: l_gas_sw             = .TRUE.
!   Flag for gaseous absorption
  LOGICAL :: l_rayleigh_sw        = .TRUE.
!   Flag for Rayleigh scattering
  LOGICAL :: l_continuum_sw       = .TRUE.
!   Flag for the continuum
  LOGICAL :: l_cont_gen_sw        = .TRUE.
!   Flag for the generalised continua
  LOGICAL :: l_drop_sw            = .TRUE.
!   Flag for droplets
  LOGICAL :: l_ice_sw             = .TRUE.
!   Flag for ice crystals
  LOGICAL :: l_aerosol_sw         = .TRUE.
!   Flag for aerosols
  LOGICAL :: l_orog_sw            = .FALSE.
!   Correct the direct solar flux at the surface for sloping terrain
  LOGICAL :: l_solvar_sw          = .FALSE.
!   Time variation of the solar spectrum

! Gaseous absorption:
  INTEGER :: i_gas_overlap_sw             = ip_overlap_k_eqv_scl
!   Treatment of overlapping gaseous absorption

! Default gases included in short-wave radiation in ModelE.
  LOGICAL :: l_o2_sw                      = .TRUE.
!   Flag for absorption by oxygen
  LOGICAL :: l_no2_sw                     = .FALSE.
!   Flag for absorption by nitrogen dioxide
  LOGICAL :: l_n2o_sw                     = .TRUE.
!   Flag for absorption by nitrous oxide
  LOGICAL :: l_ch4_sw                     = .TRUE.
!   Flag for absorption by methane
  LOGICAL :: l_cfc11_sw                   = .FALSE.
!   Flag for absorption by CFC11
  LOGICAL :: l_cfc12_sw                   = .FALSE.
!   Flag for absorption by CFC12
  LOGICAL :: l_co2_sw                     = .TRUE.
!   Flag for absorption by carbon dioxide
  LOGICAL :: l_o3_sw                      = .TRUE.
!   Flag for absorption by ozone
  LOGICAL :: l_so2_sw                     = .TRUE.
!   Flag for absorption by sulfur dioxide

! Cloud representation (use this to turn clouds off)
  LOGICAL :: l_cloud_sw                = .TRUE.
  INTEGER :: i_cloud_representation_sw = ip_cloud_csiw
  INTEGER :: i_st_water_sw             = 5
  INTEGER :: i_cnv_water_sw            = 5
  INTEGER :: i_st_ice_sw               = 8
  INTEGER :: i_cnv_ice_sw              = 8
  LOGICAL :: l_global_cloud_top_sw     = .FALSE.
  INTEGER :: i_inhom_sw                = ip_cairns
  INTEGER :: i_overlap_sw              = ip_max_rand

! Angular integration and algorithmic options
  INTEGER :: i_angular_integration_sw = ip_two_stream
  INTEGER :: i_2stream_sw             = ip_pifm80
  INTEGER :: i_scatter_method_sw      = ip_scatter_full
  LOGICAL :: l_rescale_sw             = .TRUE.

! Miscallaneous options
  LOGICAL :: l_tile_sw         = .TRUE.
!   Allow tiling of the surface

! Diagnostics (set automatically, please leave as is)
  LOGICAL :: l_clear_sw = .FALSE.
!   Calculate clear-sky fluxes

END MODULE sw_control
