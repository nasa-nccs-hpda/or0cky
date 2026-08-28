"""
Debug script to validate JAX dry convection against a minimal Fortran-like case.
"""
import numpy as np
import jax
import jax.numpy as jnp
from drycnv_jax import dry_convection_mixing_jit


def test_minimal_case():
    """Test a minimal case with a single unstable layer."""
    print("=" * 60)
    print("Debug: Minimal Test Case (Single Unstable Layer)")
    print("=" * 60)
    
    # Define a minimal unstable profile (1x1x2 grid)
    I, J, L = 1, 1, 2
    T = jnp.array([[[290.0, 280.0]]])  # T[0] > T[1] (unstable)
    Q = jnp.array([[[0.01, 0.01]]])
    PK = jnp.array([[[1.0, 0.8]]])
    PDSIG = jnp.array([[[0.2, 0.2]]])
    deltx = 0.608
    
    print("Input:")
    print(f"  T: {T[0, 0, :]}")
    print(f"  Q: {Q[0, 0, :]}")
    print(f"  PK: {PK[0, 0, :]}")
    print(f"  PDSIG: {PDSIG[0, 0, :]}")
    
    # Run JAX implementation
    T_jax, Q_jax = dry_convection_mixing_jit(T, Q, PK, PDSIG)
    print("\nJAX Output:")
    print(f"  T: {T_jax[0, 0, :]}")
    print(f"  Q: {Q_jax[0, 0, :]}")
    
    # Run Fortran-like implementation
    T_fortran = np.array(T)
    Q_fortran = np.array(Q)
    PK_fortran = np.array(PK)
    PDSIG_fortran = np.array(PDSIG)
    
    # Check instability
    TV_0 = T_fortran[0, 0, 0] * (1 + Q_fortran[0, 0, 0] * deltx)
    TV_1 = T_fortran[0, 0, 1] * (1 + Q_fortran[0, 0, 1] * deltx)
    print(f"\nFortran Debug:")
    print(f"  TV[0]: {TV_0}")
    print(f"  TV[1]: {TV_1}")
    print(f"  Unstable: {TV_0 > TV_1}")
    
    if TV_0 > TV_1:
        PKMS = PK_fortran[0, 0, 0] * PDSIG_fortran[0, 0, 0] + PK_fortran[0, 0, 1] * PDSIG_fortran[0, 0, 1]
        TVMS = TV_0 * PK_fortran[0, 0, 0] * PDSIG_fortran[0, 0, 0] + TV_1 * PK_fortran[0, 0, 1] * PDSIG_fortran[0, 0, 1]
        QMS = Q_fortran[0, 0, 0] * PDSIG_fortran[0, 0, 0] + Q_fortran[0, 0, 1] * PDSIG_fortran[0, 0, 1]
        RDP = 1.0 / (PDSIG_fortran[0, 0, 0] + PDSIG_fortran[0, 0, 1])
        THM = TVMS / (PKMS * (1 + QMS * RDP * deltx))
        QMS = QMS * RDP
        T_fortran[0, 0, 0] = THM
        T_fortran[0, 0, 1] = THM
        Q_fortran[0, 0, 0] = QMS
        Q_fortran[0, 0, 1] = QMS
        
        print(f"  PKMS: {PKMS}")
        print(f"  TVMS: {TVMS}")
        print(f"  QMS: {QMS}")
        print(f"  RDP: {RDP}")
        print(f"  THM: {THM}")
        print(f"  QM: {QMS}")
    
    print("\nFortran Output:")
    print(f"  T: {T_fortran[0, 0, :]}")
    print(f"  Q: {Q_fortran[0, 0, :]}")
    
    # Compare
    T_diff = np.abs(T_jax[0, 0, :] - T_fortran[0, 0, :])
    Q_diff = np.abs(Q_jax[0, 0, :] - Q_fortran[0, 0, :])
    print("\nDifferences:")
    print(f"  T_diff: {T_diff}")
    print(f"  Q_diff: {Q_diff}")
    
    if np.all(T_diff < 1e-6) and np.all(Q_diff < 1e-6):
        print("\n✓ Debug test PASSED: JAX and Fortran match for minimal case")
    else:
        print("\n✗ Debug test FAILED: JAX and Fortran differ for minimal case")


if __name__ == "__main__":
    test_minimal_case()
