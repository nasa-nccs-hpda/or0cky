#!/usr/bin/env python
"""
Visual Comparison Script for Fortran and JAX End-to-End Workflows
==================================================================
This script reads the outputs from the Fortran and JAX workflows and generates
visual comparisons (plots) to assess their accuracy and consistency.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt


def read_outputs(file_path):
    """Read outputs from a workflow output file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    outputs = {}
    current_key = None
    for line in lines:
        line = line.strip()
        if line.endswith(':'):
            current_key = line[:-1]
            outputs[current_key] = []
        elif current_key and line:
            try:
                outputs[current_key].extend(map(float, line.split()))
            except ValueError:
                pass
    
    return outputs


def plot_comparison(fortran_outputs, jax_outputs, output_dir='.'):
    """Generate visual comparisons between Fortran and JAX outputs."""
    
    # Keys to compare (1D arrays)
    keys_to_compare = ['dpsim', 'dpsih', 'momentum_flux_u', 'momentum_flux_v', 'heat_flux', 'solar_flux', 'lw_flux']
    
    # Create a figure for each output
    for key in keys_to_compare:
        if key in fortran_outputs and key in jax_outputs:
            fortran_data = np.array(fortran_outputs[key])
            jax_data = np.array(jax_outputs[key])
            
            # Ensure both arrays have the same length
            min_len = min(len(fortran_data), len(jax_data))
            fortran_data = fortran_data[:min_len]
            jax_data = jax_data[:min_len]
            
            # Create a figure
            plt.figure(figsize=(12, 6))
            
            # Plot Fortran and JAX outputs
            plt.plot(fortran_data, label='Fortran', color='blue', alpha=0.7, linewidth=2)
            plt.plot(jax_data, label='JAX', color='red', alpha=0.7, linewidth=2, linestyle='--')
            
            # Plot the difference
            diff = np.abs(fortran_data - jax_data)
            plt.plot(diff, label='Absolute Difference', color='green', alpha=0.5, linewidth=1)
            
            plt.title(f'Comparison: {key}')
            plt.xlabel('Grid Point')
            plt.ylabel(key)
            plt.legend()
            plt.grid(True)
            
            # Save the plot
            plot_filename = f'{output_dir}/comparison_{key}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f'Saved comparison plot for {key}: {plot_filename}')
            
            # Print statistics
            print(f'  {key}:')
            print(f'    Max difference: {np.max(diff):.6f}')
            print(f'    Mean difference: {np.mean(diff):.6f}')
            print(f'    Median difference: {np.median(diff):.6f}')
            print(f'    Std difference: {np.std(diff):.6f}')
    
    # Create a summary plot with all comparisons
    plt.figure(figsize=(18, 12))
    for idx, key in enumerate(keys_to_compare[:6], 1):  # Limit to 6 subplots
        if key in fortran_outputs and key in jax_outputs:
            fortran_data = np.array(fortran_outputs[key])
            jax_data = np.array(jax_outputs[key])
            min_len = min(len(fortran_data), len(jax_data))
            fortran_data = fortran_data[:min_len]
            jax_data = jax_data[:min_len]
            
            plt.subplot(2, 3, idx)
            plt.plot(fortran_data, label='Fortran', color='blue', alpha=0.7)
            plt.plot(jax_data, label='JAX', color='red', alpha=0.7, linestyle='--')
            plt.title(f'{key}')
            plt.xlabel('Grid Point')
            plt.ylabel(key)
            plt.legend()
            plt.grid(True)
    
    plt.tight_layout()
    summary_plot_filename = f'{output_dir}/comparison_summary.png'
    plt.savefig(summary_plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\nSaved summary comparison plot: {summary_plot_filename}')


if __name__ == '__main__':
    print('=' * 60)
    print('Fortran vs. JAX Visual Comparison')
    print('=' * 60)
    
    # Read outputs
    print('\nReading Fortran outputs...')
    fortran_outputs = read_outputs('fortran_end_to_end_output.txt')
    
    print('Reading JAX outputs...')
    jax_outputs = read_outputs('jax_end_to_end_output.txt')
    
    # Generate visual comparisons
    print('\nGenerating visual comparisons...')
    plot_comparison(fortran_outputs, jax_outputs)
    
    print('\n' + '=' * 60)
    print('Visual comparison completed!')
    print('=' * 60)
