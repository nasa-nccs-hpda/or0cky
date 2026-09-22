#!/usr/bin/env python
"""
Visualization Script for ROCKE-3D Fortran vs. JAX Comparison
==============================================================
Generates:
1. Scientific result maps (spatial distributions of key variables)
2. Performance bar plots (execution time in minutes)
"""

import numpy as np
import os

# Try to import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    PLOTTING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Matplotlib not available ({e}). Cannot generate plots.")
    PLOTTING_AVAILABLE = False


def load_data():
    """Load Fortran and JAX outputs from .npy files."""
    data = {}
    
    # JAX data
    jax_files = {
        'dpsim': 'jax_dpsim.npy',
        'dpsih': 'jax_dpsih.npy',
        'momentum_flux_u': 'jax_momentum_flux_u.npy',
        'momentum_flux_v': 'jax_momentum_flux_v.npy',
        'heat_flux': 'jax_heat_flux.npy',
        'solar_flux': 'jax_solar_flux.npy',
        'lw_flux': 'jax_lw_flux.npy',
        'T_first_layer': 'jax_T_first_layer.npy',
        'Q_first_layer': 'jax_Q_first_layer.npy'
    }
    
    # Fortran data
    fortran_files = {
        'dpsim': 'fortran_dpsim.npy',
        'dpsih': 'fortran_dpsih.npy',
        'momentum_flux_u': 'fortran_momentum_flux_u.npy',
        'momentum_flux_v': 'fortran_momentum_flux_v.npy',
        'heat_flux': 'fortran_heat_flux.npy',
        'solar_flux': 'fortran_solar_flux.npy',
        'lw_flux': 'fortran_lw_flux.npy'
    }
    
    # Load JAX data
    for key, filename in jax_files.items():
        if os.path.exists(filename):
            data[f'jax_{key}'] = np.load(filename)
        else:
            print(f"Warning: {filename} not found. Skipping.")
    
    # Load Fortran data
    for key, filename in fortran_files.items():
        if os.path.exists(filename):
            data[f'fortran_{key}'] = np.load(filename)
        else:
            print(f"Warning: {filename} not found. Skipping.")
    
    return data


def generate_scientific_maps(data, output_dir='.'):
    """Generate spatial maps of key scientific variables."""
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available. Skipping scientific maps.")
        return
    
    print("\nGenerating scientific result maps...")
    
    # Create a grid for visualization (assuming 1D data is repeated for 2D mapping)
    grid_size = 1000
    I = int(np.sqrt(grid_size))
    J = I
    
    # Reshape 1D data to 2D for mapping
    def reshape_to_2d(arr, I, J):
        """Reshape 1D array to 2D for mapping."""
        if len(arr) >= I * J:
            return arr[:I*J].reshape(I, J)
        else:
            # Pad with zeros if needed
            padded = np.zeros(I * J)
            padded[:len(arr)] = arr
            return padded.reshape(I, J)
    
    # Generate maps for JAX outputs
    variables = [
        ('T_first_layer', 'Temperature (K)', 'viridis'),
        ('Q_first_layer', 'Moisture (kg/kg)', 'Blues'),
        ('dpsim', 'Momentum Similarity (dpsim)', 'RdYlBu_r'),
        ('dpsih', 'Heat Similarity (dpsih)', 'RdYlBu_r'),
        ('heat_flux', 'Heat Flux (W/m²)', 'Reds'),
        ('solar_flux', 'Solar Flux (W/m²)', 'Oranges'),
        ('lw_flux', 'Longwave Flux (W/m²)', 'Purples')
    ]
    
    for var, title, cmap in variables:
        jax_key = f'jax_{var}'
        if jax_key in data:
            jax_data = data[jax_key]
            jax_2d = reshape_to_2d(jax_data, I, J)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(jax_2d, cmap=cmap, origin='lower')
            plt.colorbar(label=title)
            plt.title(f'JAX: {title}')
            plt.xlabel('Grid Point (i)')
            plt.ylabel('Grid Point (j)')
            plt.savefig(f'{output_dir}/map_jax_{var}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: map_jax_{var}.png")
    
    # Generate difference maps (Fortran - JAX)
    diff_variables = [
        ('dpsim', 'Momentum Similarity Difference', 'RdYlBu_r'),
        ('dpsih', 'Heat Similarity Difference', 'RdYlBu_r'),
        ('heat_flux', 'Heat Flux Difference (W/m²)', 'RdYlBu_r'),
        ('solar_flux', 'Solar Flux Difference (W/m²)', 'RdYlBu_r'),
        ('lw_flux', 'Longwave Flux Difference (W/m²)', 'RdYlBu_r')
    ]
    
    for var, title, cmap in diff_variables:
        jax_key = f'jax_{var}'
        fortran_key = f'fortran_{var}'
        if jax_key in data and fortran_key in data:
            jax_data = data[jax_key]
            fortran_data = data[fortran_key]
            
            # Ensure same length
            min_len = min(len(jax_data), len(fortran_data))
            jax_data = jax_data[:min_len]
            fortran_data = fortran_data[:min_len]
            
            # Compute absolute difference
            diff = np.abs(fortran_data - jax_data)
            diff_2d = reshape_to_2d(diff, I, J)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(diff_2d, cmap=cmap, origin='lower', norm=LogNorm(vmin=1e-6, vmax=1.0))
            plt.colorbar(label=title)
            plt.title(f'Absolute Difference (Fortran - JAX): {title}')
            plt.xlabel('Grid Point (i)')
            plt.ylabel('Grid Point (j)')
            plt.savefig(f'{output_dir}/map_diff_{var}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: map_diff_{var}.png")


def generate_performance_plots(output_dir='.'):
    """Generate bar plots for performance comparison (in minutes)."""
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available. Skipping performance plots.")
        return
    
    print("\nGenerating performance bar plots...")
    
    # Performance data (from SUMMARY.md and run_end_to_end.py)
    # Times are in seconds, convert to minutes
    performance_data = {
        'PBL': {
            'JAX (CPU)': 0.000190 / 60,  # seconds to minutes
            'NumPy (CPU)': 0.000318 / 60,
            'Fortran (CPU)': 0.000245 / 60
        },
        'DRYCNV': {
            'JAX (CPU)': 0.000873 / 60,
            'NumPy (CPU)': 0.002455 / 60,
            'Fortran (CPU)': 0.000148 / 60
        },
        'FLUXES': {
            'JAX (CPU)': 0.001018 / 60,
            'Fortran (CPU)': 0.0000189 / 60
        },
        'SURFACE': {
            'JAX (CPU)': 0.000572 / 60,
            'Fortran (CPU)': 0.000001 / 60
        },
        'RADIATION': {
            'JAX (CPU)': 0.000595 / 60,
            'Fortran (CPU)': 0.000001 / 60
        }
    }
    
    # Plot 1: Performance comparison for all modules (JAX vs. NumPy vs. Fortran)
    modules = list(performance_data.keys())
    jax_times = [performance_data[mod]['JAX (CPU)'] for mod in modules]
    numpy_times = [performance_data[mod].get('NumPy (CPU)', 0) for mod in modules]
    fortran_times = [performance_data[mod].get('Fortran (CPU)', 0) for mod in modules]
    
    x = np.arange(len(modules))
    width = 0.25
    
    plt.figure(figsize=(14, 8))
    plt.bar(x - width, numpy_times, width, label='NumPy (CPU)', color='lightblue')
    plt.bar(x, jax_times, width, label='JAX (CPU)', color='blue')
    plt.bar(x + width, fortran_times, width, label='Fortran (CPU)', color='darkblue')
    
    plt.xlabel('Module')
    plt.ylabel('Execution Time (minutes)')
    plt.title('Performance Comparison: JAX vs. NumPy vs. Fortran (CPU)')
    plt.xticks(x, modules)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.yscale('log')  # Use log scale for better visibility
    plt.savefig(f'{output_dir}/performance_comparison_cpu.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: performance_comparison_cpu.png")
    
    # Plot 2: Speedup factors (JAX vs. NumPy, JAX vs. Fortran)
    jax_vs_numpy_speedup = [numpy_times[i] / jax_times[i] if jax_times[i] > 0 else 0 for i in range(len(modules))]
    jax_vs_fortran_speedup = [fortran_times[i] / jax_times[i] if jax_times[i] > 0 else 0 for i in range(len(modules))]
    
    plt.figure(figsize=(14, 8))
    plt.bar(x - width/2, jax_vs_numpy_speedup, width, label='JAX vs. NumPy Speedup', color='green')
    plt.bar(x + width/2, jax_vs_fortran_speedup, width, label='JAX vs. Fortran Speedup', color='red')
    
    plt.xlabel('Module')
    plt.ylabel('Speedup Factor')
    plt.title('Speedup Factors: JAX vs. NumPy and JAX vs. Fortran')
    plt.xticks(x, modules)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'{output_dir}/performance_speedup.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: performance_speedup.png")
    
    # Plot 3: Performance breakdown for JAX (percentage of total time)
    jax_total = sum(jax_times)
    jax_percentages = [t / jax_total * 100 for t in jax_times]
    
    plt.figure(figsize=(14, 8))
    plt.bar(modules, jax_percentages, color='purple')
    
    plt.xlabel('Module')
    plt.ylabel('Percentage of Total Time (%)')
    plt.title('JAX Performance Breakdown (CPU)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'{output_dir}/performance_breakdown_jax.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: performance_breakdown_jax.png")


def generate_accuracy_plots(data, output_dir='.'):
    """Generate bar plots for accuracy comparison (absolute differences)."""
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available. Skipping accuracy plots.")
        return
    
    print("\nGenerating accuracy bar plots...")
    
    # Variables to compare
    variables = ['dpsim', 'dpsih', 'heat_flux', 'solar_flux', 'lw_flux']
    max_diffs = []
    mean_diffs = []
    median_diffs = []
    
    for var in variables:
        jax_key = f'jax_{var}'
        fortran_key = f'fortran_{var}'
        if jax_key in data and fortran_key in data:
            jax_data = data[jax_key]
            fortran_data = data[fortran_key]
            
            min_len = min(len(jax_data), len(fortran_data))
            jax_data = jax_data[:min_len]
            fortran_data = fortran_data[:min_len]
            
            diff = np.abs(fortran_data - jax_data)
            max_diffs.append(np.max(diff))
            mean_diffs.append(np.mean(diff))
            median_diffs.append(np.median(diff))
        else:
            max_diffs.append(0)
            mean_diffs.append(0)
            median_diffs.append(0)
    
    # Plot max differences
    plt.figure(figsize=(14, 8))
    plt.bar(variables, max_diffs, color='red', alpha=0.7)
    plt.xlabel('Variable')
    plt.ylabel('Max Absolute Difference')
    plt.title('Max Absolute Differences: Fortran vs. JAX')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'{output_dir}/accuracy_max_diff.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: accuracy_max_diff.png")
    
    # Plot mean differences
    plt.figure(figsize=(14, 8))
    plt.bar(variables, mean_diffs, color='blue', alpha=0.7)
    plt.xlabel('Variable')
    plt.ylabel('Mean Absolute Difference')
    plt.title('Mean Absolute Differences: Fortran vs. JAX')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'{output_dir}/accuracy_mean_diff.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: accuracy_mean_diff.png")


if __name__ == '__main__':
    print('=' * 60)
    print('ROCKE-3D Fortran vs. JAX Visualization')
    print('=' * 60)
    
    # Load data
    print('\nLoading data...')
    data = load_data()
    
    # Generate visualizations
    generate_scientific_maps(data)
    generate_performance_plots()
    generate_accuracy_plots(data)
    
    print('\n' + '=' * 60)
    print('Visualization completed!')
    print('=' * 60)
