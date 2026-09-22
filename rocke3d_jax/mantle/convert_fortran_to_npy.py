#!/usr/bin/env python
"""
Convert Fortran Binary Outputs to NumPy .npy Format
===================================================
This script reads the binary output files from the Fortran workflow and
converts them to NumPy .npy files for easier plotting and comparison.
"""

import numpy as np
import os


def convert_binary_to_npy(binary_file, npy_file, dtype=np.float64):
    """Convert a Fortran binary file to NumPy .npy format."""
    try:
        # Read binary data
        with open(binary_file, 'rb') as f:
            data = np.fromfile(f, dtype=dtype)
        
        # Save as .npy
        np.save(npy_file, data)
        print(f'Converted {binary_file} -> {npy_file}')
        return True
    except Exception as e:
        print(f'Error converting {binary_file}: {e}')
        return False


if __name__ == '__main__':
    print('=' * 60)
    print('Converting Fortran Binary Files to NumPy .npy Format')
    print('=' * 60)
    
    # List of binary files to convert
    binary_files = [
        'fortran_dpsim.bin',
        'fortran_dpsih.bin',
        'fortran_momentum_flux_u.bin',
        'fortran_momentum_flux_v.bin',
        'fortran_heat_flux.bin',
        'fortran_solar_flux.bin',
        'fortran_lw_flux.bin'
    ]
    
    # Convert each file
    for binary_file in binary_files:
        if os.path.exists(binary_file):
            npy_file = binary_file.replace('.bin', '.npy')
            convert_binary_to_npy(binary_file, npy_file)
        else:
            print(f'Warning: {binary_file} not found. Skipping.')
    
    print('\n' + '=' * 60)
    print('Conversion completed!')
    print('=' * 60)
