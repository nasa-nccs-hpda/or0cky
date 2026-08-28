"""
Benchmark script to compare JAX vs. NumPy (Fortran-like) performance for dry convection.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from drycnv_jax import dry_convection_mixing_jit


def dry_convection_numpy(T, Q, PK, PDSIG, deltx=0.608):
    """
    NumPy implementation of dry convection (Fortran-like, CPU baseline).
    """
    T_out = np.copy(T)
    Q_out = np.copy(Q)
    TV = T_out * (1 + Q_out * deltx)

    for l in range(T.shape[-1] - 1):
        unstable = TV[..., l] > TV[..., l + 1]
        PKMS = PK[..., l] * PDSIG[..., l] + PK[..., l + 1] * PDSIG[..., l + 1]
        TVMS = TV[..., l] * PK[..., l] * PDSIG[..., l] + TV[..., l + 1] * PK[..., l + 1] * PDSIG[..., l + 1]
        QMS = Q_out[..., l] * PDSIG[..., l] + Q_out[..., l + 1] * PDSIG[..., l + 1]
        RDP = 1.0 / (PDSIG[..., l] + PDSIG[..., l + 1])
        THM = TVMS / (PKMS * (1 + QMS * RDP * deltx))
        QM = QMS * RDP

        T_out[..., l] = np.where(unstable, THM, T_out[..., l])
        T_out[..., l + 1] = np.where(unstable, THM, T_out[..., l + 1])
        Q_out[..., l] = np.where(unstable, QM, Q_out[..., l])
        Q_out[..., l + 1] = np.where(unstable, QM, Q_out[..., l + 1])

        TV = T_out * (1 + Q_out * deltx)

    return T_out, Q_out


def benchmark():
    """Benchmark JAX vs. NumPy for dry convection."""
    print("=" * 60)
    print("Benchmark: JAX vs. NumPy (Fortran-like) for Dry Convection")
    print("=" * 60)

    # Define grid sizes to test
    grid_sizes = [
        (10, 10, 10),
        (32, 32, 20),
        (64, 64, 20),
        (128, 128, 20),
    ]

    # Number of iterations for benchmarking
    iterations = 100

    for I, J, L in grid_sizes:
        print(f"\nGrid Size: {I}x{J}x{L} = {I*J*L} points")
        print("-" * 60)

        # Generate random inputs
        key = jax.random.PRNGKey(42)
        T_np = np.random.uniform(200.0, 300.0, (I, J, L))
        Q_np = np.random.uniform(0.0, 0.02, (I, J, L))
        PK_np = np.random.uniform(0.5, 1.0, (I, J, L))
        PDSIG_np = np.random.uniform(0.1, 0.2, (I, J, L))

        # Convert to JAX arrays
        T_jax = jnp.array(T_np)
        Q_jax = jnp.array(Q_np)
        PK_jax = jnp.array(PK_np)
        PDSIG_jax = jnp.array(PDSIG_np)

        # Warm-up runs
        _ = dry_convection_numpy(T_np, Q_np, PK_np, PDSIG_np)
        T_out_jax, _ = dry_convection_mixing_jit(T_jax, Q_jax, PK_jax, PDSIG_jax)
        T_out_jax.block_until_ready()

        # Benchmark NumPy (CPU)
        start_time = time.time()
        for _ in range(iterations):
            T_out_np, Q_out_np = dry_convection_numpy(T_np, Q_np, PK_np, PDSIG_np)
        numpy_time = (time.time() - start_time) / iterations

        # Benchmark JAX (CPU/GPU)
        start_time = time.time()
        for _ in range(iterations):
            T_out_jax, Q_out_jax = dry_convection_mixing_jit(T_jax, Q_jax, PK_jax, PDSIG_jax)
            T_out_jax.block_until_ready()
        jax_time = (time.time() - start_time) / iterations

        # Compute speedup
        speedup = numpy_time / jax_time

        print(f"NumPy (CPU) time:  {numpy_time:.6f} seconds")
        print(f"JAX (CPU/GPU) time: {jax_time:.6f} seconds")
        print(f"Speedup (JAX/NumPy): {speedup:.2f}x")


if __name__ == "__main__":
    benchmark()
