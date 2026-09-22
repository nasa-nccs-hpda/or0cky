#!/usr/bin/env python
"""
3D Earth Map Visualization for ROCKE-3D Outputs
================================================

This script plots the 1D column outputs from the JAX and Fortran workflows
on a simulated 3D Earth map using Plotly. The 1D column is repeated globally
to create a fake 3D field for visualization purposes.

Usage:
    python plot_3d_earth.py

Requirements:
    - plotly
    - numpy
"""

import numpy as np
import os

# Check if plotly is available
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Error: Plotly is not installed. Install it with: pip install plotly")


def load_data():
    """Load JAX and Fortran outputs from .npy files."""
    data_files = [
        'jax_dpsim', 'jax_dpsih', 'jax_heat_flux', 'jax_solar_flux',
        'fortran_dpsim', 'fortran_dpsih', 'fortran_heat_flux', 'fortran_solar_flux'
    ]
    
    data = {}
    for name in data_files:
        try:
            data[name] = np.load(f'{name}.npy')
            print(f"Loaded {name}: shape = {data[name].shape}")
        except FileNotFoundError:
            print(f"Warning: {name}.npy not found. Skipping.")
            data[name] = None
    
    return data


def create_global_grid(data_1d, n_lat=180, n_lon=360, n_alt=20):
    """
    Create a simulated global 3D grid by repeating the 1D column data.
    
    Args:
        data_1d: 1D array (e.g., dpsim, dpsih, etc.)
        n_lat: Number of latitude points (default: 180)
        n_lon: Number of longitude points (default: 360)
        n_alt: Number of altitude levels (default: 20)
    
    Returns:
        3D array of shape (n_lat, n_lon, n_alt)
    """
    # Trim or pad the 1D data to n_alt points
    if len(data_1d) > n_alt:
        data_1d = data_1d[:n_alt]
    else:
        data_1d = np.pad(data_1d, (0, n_alt - len(data_1d)), mode='edge')
    
    # Create latitude, longitude, and altitude grids
    lat = np.linspace(-90, 90, n_lat)
    lon = np.linspace(-180, 180, n_lon)
    alt = np.linspace(0, 20, n_alt)  # Altitude in km
    
    # Repeat the 1D column for all latitudes and longitudes
    data_3d = np.tile(data_1d, (n_lat, n_lon, 1))
    
    return lon, lat, alt, data_3d


def plot_3d_earth(lon, lat, alt, data_3d, title, colorscale='Viridis'):
    """
    Plot a 3D Earth map with the given data.
    
    Args:
        lon: Longitude array
        lat: Latitude array
        alt: Altitude array
        data_3d: 3D data array of shape (n_lat, n_lon, n_alt)
        title: Plot title
        colorscale: Color scale for the plot
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping 3D plot.")
        return
    
    # Create the 3D volume plot
    fig = go.Figure(data=go.Volume(
        x=lon,
        y=lat,
        z=alt,
        value=data_3d,
        colorscale=colorscale,
        opacity=0.1,
        surface_count=20,
        colorbar=dict(title=title.split(':')[1].strip())
    ))
    
    # Customize the layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Altitude (km)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # Save the plot as HTML (remove special characters)
    filename = title.replace(' ', '_').replace(':', '').replace('/', '_').replace('²', '2').replace('(', '').replace(')', '') + '.html'
    fig.write_html(filename)
    print(f"Saved 3D plot: {filename}")
    
    # Show the plot (if in a notebook or interactive environment)
    fig.show()


def plot_comparison_3d(lon, lat, alt, data_jax, data_fortran, title, colorscale='Viridis'):
    """
    Plot a 3D Earth map comparing JAX and Fortran outputs.
    
    Args:
        lon: Longitude array
        lat: Latitude array
        alt: Altitude array
        data_jax: JAX 3D data array
        data_fortran: Fortran 3D data array
        title: Plot title
        colorscale: Color scale for the plot
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping 3D comparison plot.")
        return
    
    # Create subplots for JAX and Fortran
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'{title} - JAX', f'{title} - Fortran'),
        specs=[[{'type': 'volume'}, {'type': 'volume'}]]
    )
    
    # Add JAX data
    fig.add_trace(
        go.Volume(
            x=lon, y=lat, z=alt,
            value=data_jax,
            colorscale=colorscale,
            opacity=0.1,
            surface_count=20,
            colorbar=dict(title=title.split(':')[1].strip())
        ),
        row=1, col=1
    )
    
    # Add Fortran data
    fig.add_trace(
        go.Volume(
            x=lon, y=lat, z=alt,
            value=data_fortran,
            colorscale=colorscale,
            opacity=0.1,
            surface_count=20,
            colorbar=dict(title=title.split(':')[1].strip())
        ),
        row=1, col=2
    )
    
    # Customize the layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Altitude (km)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # Save the plot as HTML (remove special characters)
    filename = title.replace(' ', '_').replace(':', '').replace('/', '_').replace('²', '2').replace('(', '').replace(')', '') + '_comparison.html'
    fig.write_html(filename)
    print(f"Saved 3D comparison plot: {filename}")
    
    # Show the plot
    fig.show()


def main():
    print("=" * 60)
    print("3D Earth Map Visualization for ROCKE-3D")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    data = load_data()
    
    if not PLOTLY_AVAILABLE:
        print("\nPlotly is not available. Exiting.")
        return
    
    # Create global grids for each variable
    print("\nCreating global grids...")
    variables = [
        ('dpsim', 'JAX: dpsim (Momentum Similarity)', 'Viridis'),
        ('dpsih', 'JAX: dpsih (Heat Similarity)', 'Plasma'),
        ('heat_flux', 'JAX: Heat Flux (W/m²)', 'Reds'),
        ('solar_flux', 'JAX: Solar Flux (W/m²)', 'YlOrRd')
    ]
    
    grids = {}
    for var, title, colorscale in variables:
        if data[f'jax_{var}'] is not None:
            lon, lat, alt, data_3d = create_global_grid(data[f'jax_{var}'])
            grids[f'jax_{var}'] = (lon, lat, alt, data_3d)
            plot_3d_earth(lon, lat, alt, data_3d, title, colorscale)
    
    # Create comparison plots for JAX vs. Fortran
    print("\nCreating comparison plots...")
    comparison_variables = [
        ('dpsim', 'PBL: dpsim Comparison', 'Viridis'),
        ('dpsih', 'PBL: dpsih Comparison', 'Plasma')
    ]
    
    for var, title, colorscale in comparison_variables:
        if data[f'jax_{var}'] is not None and data[f'fortran_{var}'] is not None:
            lon, lat, alt, jax_3d = grids[f'jax_{var}']
            _, _, _, fortran_3d = create_global_grid(data[f'fortran_{var}'])
            plot_comparison_3d(lon, lat, alt, jax_3d, fortran_3d, title, colorscale)
    
    print("\n" + "=" * 60)
    print("3D Earth Map Visualization Complete!")
    print("=" * 60)
    print("\nSaved HTML files:")
    for f in os.listdir():
        if f.endswith('.html'):
            print(f"  - {f}")


if __name__ == '__main__':
    main()
