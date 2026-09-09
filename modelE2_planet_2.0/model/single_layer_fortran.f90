program single_layer_fortran
    ! ============================================
    ! Minimal Fortran Program for Single-Layer Global Pixel Map
    ! ============================================
    ! This program replicates the JAX-based single-layer simulation
    ! for benchmarking purposes.
    !
    ! Compile with:
    !   gfortran -O3 -o single_layer_fortran single_layer_fortran.f90
    !
    implicit none
    
    ! Constants
    integer, parameter :: N_LAT = 180
    integer, parameter :: N_LON = 360
    real(8), parameter :: PI = 3.141592653589793d0
    
    ! Arrays for latitude, longitude, and outputs
    real(8), dimension(N_LAT) :: lat
    real(8), dimension(N_LON) :: lon
    real(8), dimension(N_LAT, N_LON) :: temperature, pressure, heat_flux, solar_flux
    
    ! Loop variables
    integer :: i, j
    
    ! Time measurement
    real(8) :: start_time, end_time
    
    ! Initialize latitude and longitude
    call cpu_time(start_time)
    
    ! Create latitude array: -90 to 90
    do i = 1, N_LAT
        lat(i) = -90.0d0 + (i - 1) * (180.0d0 / (N_LAT - 1))
    end do
    
    ! Create longitude array: -180 to 180
    do j = 1, N_LON
        lon(j) = -180.0d0 + (j - 1) * (360.0d0 / (N_LON - 1))
    end do
    
    ! Compute outputs (same as JAX version)
    do i = 1, N_LAT
        do j = 1, N_LON
            ! Temperature (K) as a function of latitude
            temperature(i, j) = 288.0d0 + 20.0d0 * sin(lat(i) * PI / 180.0d0)
            
            ! Pressure (hPa) as a function of latitude
            pressure(i, j) = 1013.0d0 - 10.0d0 * sin(lat(i) * PI / 180.0d0)
            
            ! Heat flux (W/m²) as a function of longitude
            heat_flux(i, j) = 100.0d0 + 50.0d0 * cos(lon(j) * PI / 180.0d0)
            
            ! Solar flux (W/m²) as a function of latitude
            solar_flux(i, j) = 1365.0d0 * (1.0d0 + 0.1d0 * sin(lat(i) * PI / 180.0d0))
        end do
    end do
    
    call cpu_time(end_time)
    
    ! Print execution time
    print *, "Fortran Execution Time: ", end_time - start_time, " seconds"
    
    ! Write outputs to files (unformatted binary)
    open(unit=10, file='fortran_temperature.bin', form='unformatted')
    write(10) temperature
    close(10)
    
    open(unit=11, file='fortran_pressure.bin', form='unformatted')
    write(11) pressure
    close(11)
    
    open(unit=12, file='fortran_heat_flux.bin', form='unformatted')
    write(12) heat_flux
    close(12)
    
    open(unit=13, file='fortran_solar_flux.bin', form='unformatted')
    write(13) solar_flux
    close(13)
    
    ! Print summary
    print *, "Fortran Single-Layer Simulation Complete"
    print *, "Grid: ", N_LAT, "x", N_LON
    print *, "Outputs: Temperature, Pressure, Heat Flux, Solar Flux"
    
end program single_layer_fortran
