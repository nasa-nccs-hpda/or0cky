! Minimal I/O-driven driver around SOCRATES's gauss_angle (real compiled library code, not
! reimplemented). Reads a fixed-size binary request from stdin, calls gauss_angle, writes the
! result to stdout. Proof-of-concept for calling libsocrates.a from Python: since the precompiled
! .o files in libsocrates.a are not position-independent (-fPIC), a shared object cannot be linked
! from them without a full source recompile (confirmed: `ld: relocation ... can not be used when
! making a shared object`); this subprocess/stdin-stdout architecture avoids that constraint and
! still calls the real, unmodified SOCRATES object code.
program gauss_angle_driver
  use realtype_rd, only: RealK
  implicit none
  integer, parameter :: nd_profile=1, nd_layer_max=100
  integer :: n_layer, n_order_gauss, i
  real(RealK) :: tau(nd_profile,nd_layer_max), flux_inc_down(nd_profile)
  real(RealK) :: diff_planck(nd_profile,nd_layer_max), source_ground(nd_profile)
  real(RealK) :: albedo_surface_diff(nd_profile), diff_planck_2(nd_profile,nd_layer_max)
  real(RealK) :: flux_diffuse(nd_profile, 2*nd_layer_max+2)
  logical :: l_ir_source_quad

  read(*,*) n_layer, n_order_gauss
  read(*,*) tau(1,1:n_layer)
  read(*,*) diff_planck(1,1:n_layer)
  read(*,*) flux_inc_down(1), source_ground(1), albedo_surface_diff(1)
  diff_planck_2 = 0.0_RealK
  l_ir_source_quad = .false.

  call gauss_angle(1, n_layer, n_order_gauss, tau, flux_inc_down, &
       diff_planck, source_ground, albedo_surface_diff, flux_diffuse, &
       l_ir_source_quad, diff_planck_2, nd_profile, nd_layer_max)

  do i=1,2*n_layer+2
    write(*,'(ES24.16E3)') flux_diffuse(1,i)
  end do
end program gauss_angle_driver
