import jax
import jax.numpy as jnp
from jax import jit


@jit
def dry_convection_mixing(
    T: jnp.ndarray,
    Q: jnp.ndarray,
    PK: jnp.ndarray,
    PDSIG: jnp.ndarray,
    deltx: float = 0.608,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optimized JAX implementation of dry convection mixing.
    Uses precomputed TV and in-place updates for better performance.
    """
    # Initialize outputs
    T_out = jnp.copy(T)
    Q_out = jnp.copy(Q)
    TV = T_out * (1 + Q_out * deltx)

    # Precompute RDP for all layers to avoid division in the loop
    RDP = 1.0 / (PDSIG[..., :-1] + PDSIG[..., 1:])

    # Loop over layers (Python loop is unrolled by JIT for small L)
    for L in range(T.shape[-1] - 1):
        unstable = TV[..., L] > TV[..., L + 1]

        # Vectorized mixing for layers L and L+1
        PKMS = PK[..., L] * PDSIG[..., L] + PK[..., L + 1] * PDSIG[..., L + 1]
        TVMS = TV[..., L] * PK[..., L] * PDSIG[..., L] + TV[..., L + 1] * PK[..., L + 1] * PDSIG[..., L + 1]
        QMS = Q_out[..., L] * PDSIG[..., L] + Q_out[..., L + 1] * PDSIG[..., L + 1]
        THM = TVMS / (PKMS * (1 + QMS * RDP[..., L] * deltx))
        QM = QMS * RDP[..., L]

        # Apply mixing
        T_out = T_out.at[..., L].set(jnp.where(unstable, THM, T_out[..., L]))
        T_out = T_out.at[..., L + 1].set(jnp.where(unstable, THM, T_out[..., L + 1]))
        Q_out = Q_out.at[..., L].set(jnp.where(unstable, QM, Q_out[..., L]))
        Q_out = Q_out.at[..., L + 1].set(jnp.where(unstable, QM, Q_out[..., L + 1]))

        # Update TV for next iteration
        TV = T_out * (1 + Q_out * deltx)

    return T_out, Q_out


# Alias for backward compatibility
dry_convection_mixing_jit = dry_convection_mixing


def validate_drycnv():
    """
    Test script to validate the JAX implementation of dry convection.
    Compares JAX output with a simplified Fortran-like implementation.
    """
    import numpy as np
    
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Define test dimensions (small for testing)
    I, J, L = 4, 4, 10
    
    # Generate random inputs (Fortran-like ranges)
    T = jax.random.uniform(key, (I, J, L), minval=200.0, maxval=300.0)  # Temperature (K)
    Q = jax.random.uniform(key, (I, J, L), minval=0.0, maxval=0.02)      # Moisture (kg/kg)
    PK = jax.random.uniform(key, (I, J, L), minval=0.5, maxval=1.0)      # Pressure (normalized)
    PDSIG = jax.random.uniform(key, (I, J, L), minval=0.1, maxval=0.2)   # Layer thickness (sigma)
    
    # Run JAX implementation
    T_jax, Q_jax = dry_convection_mixing_jit(T, Q, PK, PDSIG)
    
    # Run a simplified Fortran-like implementation (for comparison)
    T_fortran = np.array(T)
    Q_fortran = np.array(Q)
    PK_fortran = np.array(PK)
    PDSIG_fortran = np.array(PDSIG)
    deltx = 0.608
    
    for i in range(I):
        for j in range(J):
            for l in range(L - 1):
                # Check if layer l is unstable (TV[l] > TV[l+1] for dry convection)
                TV_l = T_fortran[i, j, l] * (1 + Q_fortran[i, j, l] * deltx)
                TV_lp1 = T_fortran[i, j, l + 1] * (1 + Q_fortran[i, j, l + 1] * deltx)
                if TV_l > TV_lp1:
                    # Mix layers l and l+1
                    PKMS = (
                        PK_fortran[i, j, l] * PDSIG_fortran[i, j, l] +
                        PK_fortran[i, j, l + 1] * PDSIG_fortran[i, j, l + 1]
                    )
                    TVMS = (
                        TV_l * PK_fortran[i, j, l] * PDSIG_fortran[i, j, l] +
                        TV_lp1 * PK_fortran[i, j, l + 1] * PDSIG_fortran[i, j, l + 1]
                    )
                    QMS = (
                        Q_fortran[i, j, l] * PDSIG_fortran[i, j, l] +
                        Q_fortran[i, j, l + 1] * PDSIG_fortran[i, j, l + 1]
                    )
                    RDP = 1.0 / (PDSIG_fortran[i, j, l] + PDSIG_fortran[i, j, l + 1])
                    THM = TVMS / (PKMS * (1 + QMS * RDP * deltx))
                    QMS = QMS * RDP
                    
                    # Update T and Q
                    T_fortran[i, j, l] = THM
                    T_fortran[i, j, l + 1] = THM
                    Q_fortran[i, j, l] = QMS
                    Q_fortran[i, j, l + 1] = QMS
    
    # Compare results
    T_diff = np.abs(np.array(T_jax) - T_fortran)
    Q_diff = np.abs(np.array(Q_jax) - Q_fortran)
    
    print("Validation Results:")
    print(f"- Max T difference: {np.max(T_diff):.6e}")
    print(f"- Max Q difference: {np.max(Q_diff):.6e}")
    print(f"- Mean T difference: {np.mean(T_diff):.6e}")
    print(f"- Mean Q difference: {np.mean(Q_diff):.6e}")
    
    # Check if differences are within tolerance
    tolerance = 1e-6
    T_pass = np.all(T_diff < tolerance)
    Q_pass = np.all(Q_diff < tolerance)
    
    print(f"\nValidation {'PASSED' if T_pass and Q_pass else 'FAILED'}")
    print(f"- T within tolerance: {'YES' if T_pass else 'NO'}")
    print(f"- Q within tolerance: {'YES' if Q_pass else 'NO'}")
    
    return T_jax, Q_jax, T_fortran, Q_fortran


if __name__ == "__main__":
    validate_drycnv()
