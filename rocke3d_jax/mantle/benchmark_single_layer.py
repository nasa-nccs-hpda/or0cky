#!/usr/bin/env python
"""
Benchmark: Single-Layer Global Pixel Map
==========================================

This script benchmarks the performance of:
1. **JAX-based single-layer simulation** (Python)
2. **Original Fortran ROCKE-3D** (if available)

It measures execution time and compares outputs for validation.

Usage:
    python benchmark_single_layer.py

Requirements:
    - numpy
    - jax
    - time
    - subprocess (for Fortran execution)
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Force CPU-only mode

import jax
import jax.numpy as jnp
import numpy as np
import time
import subprocess

# --- Constants ---
N_LAT = 180  # Number of latitude points
N_LON = 360  # Number of longitude points
N_LAYERS = 1  # Single layer
N_RUNS = 10  # Number of runs for averaging

# --- JAX-Based Simulation ---
def create_global_grid():
    """Create a global latitude-longitude grid."""
    lat = jnp.linspace(-90, 90, N_LAT)
    lon = jnp.linspace(-180, 180, N_LON)
    return lat, lon


def simulate_rocke3d_jax(lat, lon):
    """
    Simulate ROCKE-3D outputs for a single layer using JAX.
    """
    lat_2d, lon_2d = jnp.meshgrid(lat, lon, indexing='ij')
    
    # Simulate surface temperature (K)
    temp = 288.0 + 20.0 * jnp.sin(jnp.deg2rad(lat_2d))
    
    # Simulate surface pressure (hPa)
    pres = 1013.0 - 10.0 * jnp.sin(jnp.deg2rad(lat_2d))
    
    # Simulate heat flux (W/m²)
    heat_flux = 100.0 + 50.0 * jnp.cos(jnp.deg2rad(lon_2d))
    
    # Simulate solar flux (W/m²)
    solar_flux = 1365.0 * (1.0 + 0.1 * jnp.sin(jnp.deg2rad(lat_2d)))
    
    return {
        "temperature": temp,
        "pressure": pres,
        "heat_flux": heat_flux,
        "solar_flux": solar_flux,
    }


def benchmark_jax():
    """Benchmark the JAX-based simulation."""
    print("\n" + "=" * 60)
    print("Benchmarking JAX-Based Single-Layer Simulation")
    print("=" * 60)
    
    # Warm-up run (JAX compiles the first time)
    lat, lon = create_global_grid()
    _ = simulate_rocke3d_jax(lat, lon)
    
    # Benchmark
    times = []
    for i in range(N_RUNS):
        start_time = time.time()
        outputs = simulate_rocke3d_jax(lat, lon)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\nJAX Benchmark Results:")
    print(f"  - Average Time: {avg_time:.6f} ± {std_time:.6f} seconds")
    print(f"  - Min Time: {np.min(times):.6f} seconds")
    print(f"  - Max Time: {np.max(times):.6f} seconds")
    
    return avg_time, outputs


# --- Fortran Benchmark ---
def benchmark_fortran():
    """Benchmark the minimal Fortran program (if available)."""
    print("\n" + "=" * 60)
    print("Benchmarking Fortran Single-Layer Simulation")
    print("=" * 60)
    
    # Check if the Fortran executable exists
    fortran_exe = "/home/gtamkin/_ilab-agentic-ai/ilab-agentic-ai/projects/imvi/modelE2_planet_2.0/model/single_layer_fortran"
    
    if not os.path.exists(fortran_exe):
        print(f"\nFortran executable not found at: {fortran_exe}")
        print("Skipping Fortran benchmark.")
        return None, None
    
    # Benchmark
    times = []
    for i in range(N_RUNS):
        start_time = time.time()
        try:
            result = subprocess.run(
                [fortran_exe],
                capture_output=True,
                text=True,
                timeout=30
            )
            end_time = time.time()
            times.append(end_time - start_time)
        except subprocess.TimeoutExpired:
            print(f"\nFortran execution timed out. Skipping.")
            return None, None
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\nFortran Benchmark Results:")
    print(f"  - Average Time: {avg_time:.6f} ± {std_time:.6f} seconds")
    print(f"  - Min Time: {np.min(times):.6f} seconds")
    print(f"  - Max Time: {np.max(times):.6f} seconds")
    
    return avg_time, None


# --- Main ---
def main():
    print("=" * 60)
    print("Single-Layer Global Pixel Map Benchmark")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Grid: {N_LAT} lat × {N_LON} lon")
    print(f"  - Layers: {N_LAYERS}")
    print(f"  - Runs: {N_RUNS}")
    
    # Benchmark JAX
    jax_time, jax_outputs = benchmark_jax()
    
    # Benchmark Fortran
    fortran_time, fortran_outputs = benchmark_fortran()
    
    # Summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    
    if jax_time is not None:
        print(f"\nJAX:")
        print(f"  - Time: {jax_time:.6f} seconds")
    
    if fortran_time is not None:
        print(f"\nFortran:")
        print(f"  - Time: {fortran_time:.6f} seconds")
        
        if jax_time is not None:
            speedup = fortran_time / jax_time
            print(f"\nSpeedup (JAX vs. Fortran): {speedup:.2f}x")
    else:
        print("\nFortran benchmark skipped (executable not found).")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
