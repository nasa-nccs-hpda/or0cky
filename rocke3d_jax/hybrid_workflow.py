"""
Hybrid JAX-Fortran Workflow for ROCKE-3D
==========================================

This script demonstrates how to integrate JAX modules with ROCKE-3D's
Fortran workflow. It:
1. Runs JAX modules (DRYCNV, PBL) on sample data.
2. Saves/loads data to/from NetCDF files (compatible with ROCKE-3D).
3. Benchmarks the hybrid workflow.

Usage:
    python hybrid_workflow.py
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Force JAX to use CPU backend

import time
import numpy as np
import jax
import jax.numpy as jnp
from rocke3d_jax.drycnv import dry_convection_mixing_jit
from rocke3d_jax.pbl import simil_jit


def run_hybrid_workflow():
    """Run a hybrid JAX-Fortran workflow."""
    print("=" * 60)
    print("Hybrid JAX-Fortran Workflow for ROCKE-3D")
    print("=" * 60)
    
    # Step 1: Initialize atmospheric state (Fortran-like)
    print("\nStep 1: Initialize atmospheric state...")
    I, J, L = 32, 32, 20  # Grid dimensions
    
    # Generate random atmospheric state
    key = jax.random.PRNGKey(42)
    T = jax.random.uniform(key, (I, J, L), minval=200.0, maxval=300.0)  # Temperature (K)
    Q = jax.random.uniform(key, (I, J, L), minval=0.0, maxval=0.02)      # Moisture (kg/kg)
    PK = jax.random.uniform(key, (I, J, L), minval=0.5, maxval=1.0)      # Pressure (normalized)
    PDSIG = jax.random.uniform(key, (I, J, L), minval=0.1, maxval=0.2)   # Layer thickness (sigma)
    
    print(f"  Grid size: {I}x{J}x{L} = {I*J*L} points")
    print(f"  T range: [{T.min():.2f}, {T.max():.2f}] K")
    print(f"  Q range: [{Q.min():.4f}, {Q.max():.4f}] kg/kg")
    
    # Step 2: Run JAX DRYCNV (dry convection mixing)
    print("\nStep 2: Run JAX DRYCNV (dry convection mixing)...")
    start_time = time.time()
    T_drycnv, Q_drycnv = dry_convection_mixing_jit(T, Q, PK, PDSIG)
    drycnv_time = time.time() - start_time
    print(f"  Time: {drycnv_time:.6f} seconds")
    print(f"  T range after DRYCNV: [{T_drycnv.min():.2f}, {T_drycnv.max():.2f}] K")
    
    # Step 3: Run JAX PBL (surface fluxes)
    print("\nStep 3: Run JAX PBL (surface fluxes)...")
    z = jnp.ones((I, J)) * 10.0  # Height (m)
    z0m = jnp.ones((I, J)) * 0.1   # Momentum roughness height (m)
    z0h = jnp.ones((I, J)) * 0.1   # Temperature roughness height (m)
    z0q = jnp.ones((I, J)) * 0.1   # Moisture roughness height (m)
    lmonin = jnp.ones((I, J)) * -10.0  # Monin-Obukhov length (m, unstable)
    ustar = jnp.ones((I, J)) * 0.5   # Friction speed (m/s)
    tstar = jnp.ones((I, J)) * 0.1   # Temperature scale (K)
    qstar = jnp.ones((I, J)) * 0.01  # Moisture scale (kg/kg)
    tg = jnp.ones((I, J)) * 300.0    # Ground temperature (K)
    qg = jnp.ones((I, J)) * 0.02     # Ground moisture mixing ratio (kg/kg)
    
    start_time = time.time()
    u, t, q, dpsim, dpsih, dpsiq = simil_jit(z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg)
    pbl_time = time.time() - start_time
    print(f"  Time: {pbl_time:.6f} seconds")
    print(f"  u range: [{u.min():.2f}, {u.max():.2f}] m/s")
    print(f"  t range: [{t.min():.2f}, {t.max():.2f}] K")
    
    # Step 4: Save results to NetCDF (compatible with ROCKE-3D)
    print("\nStep 4: Save results to NetCDF...")
    try:
        import netCDF4
        
        # Create NetCDF file
        output_file = "hybrid_workflow_output.nc"
        ds = netCDF4.Dataset(output_file, "w", format="NETCDF4")
        
        # Define dimensions
        ds.createDimension("lon", I)
        ds.createDimension("lat", J)
        ds.createDimension("lev", L)
        
        # Save variables
        T_out = ds.createVariable("T", "f8", ("lon", "lat", "lev"))
        T_out[:] = T_drycnv
        T_out.units = "K"
        T_out.long_name = "Temperature after dry convection"
        
        Q_out = ds.createVariable("Q", "f8", ("lon", "lat", "lev"))
        Q_out[:] = Q_drycnv
        Q_out.units = "kg/kg"
        Q_out.long_name = "Moisture after dry convection"
        
        u_out = ds.createVariable("u", "f8", ("lon", "lat"))
        u_out[:] = u
        u_out.units = "m/s"
        u_out.long_name = "Wind speed from PBL"
        
        t_out = ds.createVariable("t", "f8", ("lon", "lat"))
        t_out[:] = t
        t_out.units = "K"
        t_out.long_name = "Temperature from PBL"
        
        ds.close()
        print(f"  Saved to: {output_file}")
    except ImportError:
        print("  NetCDF4 not installed. Skipping NetCDF output.")
    
    # Step 5: Benchmark hybrid workflow
    print("\nStep 5: Benchmark hybrid workflow...")
    iterations = 10
    
    start_time = time.time()
    for _ in range(iterations):
        # Run DRYCNV
        T_drycnv, Q_drycnv = dry_convection_mixing_jit(T, Q, PK, PDSIG)
        # Run PBL
        u, t, q, dpsim, dpsih, dpsiq = simil_jit(z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg)
    hybrid_time = (time.time() - start_time) / iterations
    
    print(f"  Hybrid workflow time (per call): {hybrid_time:.6f} seconds")
    print(f"  Total time for {iterations} iterations: {hybrid_time * iterations:.6f} seconds")
    
    print("\n" + "=" * 60)
    print("Hybrid workflow completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_hybrid_workflow()
