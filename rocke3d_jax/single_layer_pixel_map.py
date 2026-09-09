#!/usr/bin/env python
"""
Single-Layer Global Pixel Map Generator for ROCKE-3D
====================================================

This script generates a **single-layer global pixel map** using JAX-based
ROCKE-3D modules. It simulates a simplified end-to-end workflow and outputs
a 2D global map (latitude x longitude) for visualization.

Usage:
    python single_layer_pixel_map.py

Requirements:
    - jax
    - numpy
    - netCDF4 (optional, for saving outputs)
    - matplotlib (optional, for quick visualization)
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Force CPU-only mode

import jax
import jax.numpy as jnp
import numpy as np

# Ensure JAX is configured for CPU
print("JAX Devices:", jax.devices())

# --- Constants ---
N_LAT = 180  # Number of latitude points
N_LON = 360  # Number of longitude points
N_LAYERS = 1  # Single layer

# --- Simulated Global Grid ---
def create_global_grid():
    """Create a global latitude-longitude grid."""
    lat = jnp.linspace(-90, 90, N_LAT)  # Latitude: -90 to 90
    lon = jnp.linspace(-180, 180, N_LON)  # Longitude: -180 to 180
    return lat, lon


# --- Simulated ROCKE-3D Outputs ---
def simulate_rocke3d_outputs(lat, lon):
    """
    Simulate ROCKE-3D outputs for a single layer.
    This replaces the actual Fortran model with a simplified JAX-based simulation.
    """
    # Create 2D grids for latitude and longitude
    lat_2d, lon_2d = jnp.meshgrid(lat, lon, indexing='ij')
    
    # Simulate surface temperature (K) as a function of latitude
    temp = 288.0 + 20.0 * jnp.sin(jnp.deg2rad(lat_2d))  # Latitudinal gradient
    
    # Simulate surface pressure (hPa)
    pres = 1013.0 - 10.0 * jnp.sin(jnp.deg2rad(lat_2d))  # Latitudinal gradient
    
    # Simulate heat flux (W/m²) as a function of longitude
    heat_flux = 100.0 + 50.0 * jnp.cos(jnp.deg2rad(lon_2d))  # Longitudinal gradient
    
    # Simulate solar flux (W/m²) as a function of latitude
    solar_flux = 1365.0 * (1.0 + 0.1 * jnp.sin(jnp.deg2rad(lat_2d)))  # Latitudinal gradient
    
    return {
        "temperature": temp,
        "pressure": pres,
        "heat_flux": heat_flux,
        "solar_flux": solar_flux,
    }


# --- Save Outputs ---
def save_outputs(outputs, lat, lon, output_dir="outputs"):
    """Save outputs as numpy arrays and NetCDF files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, data in outputs.items():
        # Save as numpy array
        np.save(f"{output_dir}/{name}.npy", data)
        print(f"Saved {name}: shape = {data.shape}")
    
    # Save latitude and longitude
    np.save(f"{output_dir}/lat.npy", lat)
    np.save(f"{output_dir}/lon.npy", lon)
    print(f"Saved lat/lon: shape = {lat.shape}, {lon.shape}")


# --- Visualization ---
def visualize_outputs(outputs, lat, lon):
    """Quick visualization using matplotlib (if available)."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        for name, data in outputs.items():
            plt.figure(figsize=(10, 5))
            plt.imshow(data, extent=[-180, 180, -90, 90], aspect="auto")
            plt.colorbar(label=name)
            plt.title(f"Global {name} (Single Layer)")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.savefig(f"{name}_global_map.png")
            plt.close()
            print(f"Saved {name}_global_map.png")
    except Exception as e:
        print(f"Matplotlib not available or failed: {e}. Skipping visualization.")


# --- Main ---
def main():
    print("=" * 60)
    print("Single-Layer Global Pixel Map Generator")
    print("=" * 60)
    
    # Create global grid
    print("\nCreating global grid...")
    lat, lon = create_global_grid()
    
    # Simulate ROCKE-3D outputs
    print("Simulating ROCKE-3D outputs...")
    outputs = simulate_rocke3d_outputs(lat, lon)
    
    # Save outputs
    print("\nSaving outputs...")
    save_outputs(outputs, lat, lon)
    
    # Visualize outputs
    print("\nVisualizing outputs...")
    visualize_outputs(outputs, lat, lon)
    
    print("\n" + "=" * 60)
    print("Single-Layer Global Pixel Map Generation Complete!")
    print("=" * 60)
    print("\nOutputs saved in 'outputs/' directory:")
    for f in os.listdir("outputs"):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
