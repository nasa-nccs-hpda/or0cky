!-------------------------------------------------------------------------------
! Control options for planet radiation (SOCRATES)
!-------------------------------------------------------------------------------

MODULE lw_control

  USE realtype_rd,      ONLY: RealK
  USE missing_data_mod, ONLY: rmdi, imdi
  USE rad_pcf
  USE planet_alb,       ONLY: n_veg, ip_sea_ice_alb_Hansen

  IMPLICIT NONE

! Sea ice albedo scheme (KSIALB)
  INTEGER :: i_sea_ice_alb = ip_sea_ice_alb_Hansen

! Physical processes
  LOGICAL :: l_microphysics_lw    = .TRUE.
!   Flag for microphysics
  LOGICAL :: l_gas_lw             = .TRUE.
!   Flag for gaseous absorption
  LOGICAL :: l_continuum_lw       = .TRUE.
!   Flag for the continuum
  LOGICAL :: l_cont_gen_lw        = .TRUE.
!   Flag for the generalised continua
  LOGICAL :: l_drop_lw            = .TRUE.
!   Flag for droplets
  LOGICAL :: l_ice_lw             = .TRUE.
!   Flag for ice crystals
  LOGICAL :: l_aerosol_lw         = .TRUE.
!   Flag for aerosols
  LOGICAL :: l_solar_tail_flux_lw = .FALSE.
!   Flag for adding solar tail flux to LW ragion

! Gaseous absorption:
  INTEGER :: i_gas_overlap_lw             = ip_overlap_k_eqv_scl
!   Treatment of overlapping gaseous absorption

! Default gases included in long-wave radiation in ModelE.
  LOGICAL :: l_o2_lw                      = .FALSE.
!   Flag for absorption by oxygen
  LOGICAL :: l_no2_lw                     = .FALSE.
!   Flag for absorption by nitrogen dioxide
  LOGICAL :: l_n2o_lw                     = .TRUE.
!   Flag for absorption by nitrous oxide
  LOGICAL :: l_ch4_lw                     = .TRUE.
!   Flag for absorption by methane
  LOGICAL :: l_cfc11_lw                   = .TRUE.
!   Flag for absorption by CFC11
  LOGICAL :: l_cfc12_lw                   = .TRUE.
!   Flag for absorption by CFC12
  LOGICAL :: l_co2_lw                     = .TRUE.
!   Flag for absorption by carbon dioxide
  LOGICAL :: l_o3_lw                      = .TRUE.
!   Flag for absorption by ozone
  LOGICAL :: l_so2_lw                     = .TRUE.
!   Flag for absorption by sulfur dioxide

! Cloud representation (use this to turn clouds off)
  LOGICAL :: l_cloud_lw                = .TRUE.
  INTEGER :: i_cloud_representation_lw = ip_cloud_csiw
  INTEGER :: i_st_water_lw             = 5
  INTEGER :: i_cnv_water_lw            = 5
  INTEGER :: i_st_ice_lw               = 8
  INTEGER :: i_cnv_ice_lw              = 8
  LOGICAL :: l_global_cloud_top_lw     = .FALSE.
  INTEGER :: i_inhom_lw                = ip_cairns
  INTEGER :: i_overlap_lw              = ip_max_rand

! Angular integration and algorithmic options
  INTEGER :: i_angular_integration_lw = ip_two_stream
  INTEGER :: i_2stream_lw             = ip_elsasser
  LOGICAL :: l_ir_source_quad_lw      = .TRUE.
  INTEGER :: i_scatter_method_lw      = ip_scatter_full
  LOGICAL :: l_rescale_lw             = .TRUE.

! Miscallaneous options
  LOGICAL :: l_tile_lw         = .TRUE.
!   Allow tiling of the surface

! Diagnostics (set automatically, please leave as is)
  LOGICAL :: l_clear_lw = .FALSE.
!   Calculate clear-sky fluxes

END MODULE lw_control
