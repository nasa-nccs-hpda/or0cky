"""
P2SAoM40-Grounded Benchmark for ROCKE-3D JAX Implementation
=============================================================

Same methodology as benchmark_all.py / benchmark_all_cpu.py (JAX vs. NumPy,
DRYCNV + PBL modules), but the grid sizes are the *real* dimensions of this
project's own P2SAoM40 configuration instead of arbitrary round numbers.

P2SAoM40 = ROCKE-3D 2.0, SOCRATES radiation, Earth-like ("A") atmosphere,
dynamic ocean, medium resolution. Built and run locally from
dev/modelE2_planet_2.0 (decks/P2SAoM40.R); output lives under
ModelE_Support/huge_space/P2SAoM40 (symlinked from ModelE_Support/prod_runs/
P2SAoM40, where the run directory's fort.1.nc/fort.2.nc restart files
actually sit).

Grid dimensions are read directly from that run's own restart-file NetCDF
dimensions (im/jm/lm/lmo), with a hardcoded fallback to the values confirmed
there at the time this script was written:
    im=72 (longitude), jm=46 (latitude), lm=40 (atmosphere layers),
    lmo=13 (ocean layers)
Note: config_rocke3d2.yaml in this directory lists "levels: 20" for its
atmospheric_resolution -- that does not match this run (it documents a
different/stale template variant) and is NOT used here.

IMPORTANT: only the *shapes* are grounded in the real run. Field VALUES are
still synthetic random draws over physically plausible ranges, exactly as in
benchmark_all.py/benchmark_all_cpu.py -- this script does not extract actual
prognostic field values out of the restart file's internal layout.

Usage:
    python benchmark_all_p2saom40.py
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"  # This node has no GPU (nvidia-smi absent)

import sys
sys.path.insert(0, '.')

import time
import numpy as np
import jax
from drycnv import dry_convection_mixing_jit
from pbl import simil_jit

P2SAOM40_RESTART = (
    "/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/"
    "ModelE_Support/prod_runs/P2SAoM40/fort.2.nc"
)
# Confirmed directly from that file's dimensions at the time of writing.
FALLBACK_DIMS = {"im": 72, "jm": 46, "lm": 40, "lmo": 13}


def get_p2saom40_dims():
    """Read real (im, jm, lm) from the live P2SAoM40 restart file if
    reachable, else fall back to the last-confirmed values above."""
    try:
        import netCDF4 as nc
        ds = nc.Dataset(P2SAOM40_RESTART)
        dims = {k: len(ds.dimensions[k]) for k in ("im", "jm", "lm", "lmo") if k in ds.dimensions}
        ds.close()
        if {"im", "jm", "lm"} <= dims.keys():
            return dims["im"], dims["jm"], dims["lm"]
        print(f"WARNING: expected dims not found in {P2SAOM40_RESTART}, using fallback")
    except Exception as e:
        print(f"WARNING: could not read {P2SAOM40_RESTART} ({e!r}), using fallback")
    return FALLBACK_DIMS["im"], FALLBACK_DIMS["jm"], FALLBACK_DIMS["lm"]


def dry_convection_numpy(T, Q, PK, PDSIG, deltx=0.608):
    """NumPy implementation of dry convection (Fortran-like, CPU baseline)."""
    T_out = np.copy(T)
    Q_out = np.copy(Q)
    TV = T_out * (1 + Q_out * deltx)

    for L in range(T.shape[-1] - 1):
        unstable = TV[..., L] > TV[..., L + 1]
        PKMS = PK[..., L] * PDSIG[..., L] + PK[..., L + 1] * PDSIG[..., L + 1]
        TVMS = TV[..., L] * PK[..., L] * PDSIG[..., L] + TV[..., L + 1] * PK[..., L + 1] * PDSIG[..., L + 1]
        QMS = Q_out[..., L] * PDSIG[..., L] + Q_out[..., L + 1] * PDSIG[..., L + 1]
        RDP = 1.0 / (PDSIG[..., L] + PDSIG[..., L + 1])
        THM = TVMS / (PKMS * (1 + QMS * RDP * deltx))
        QM = QMS * RDP

        T_out[..., L] = np.where(unstable, THM, T_out[..., L])
        T_out[..., L + 1] = np.where(unstable, THM, T_out[..., L + 1])
        Q_out[..., L] = np.where(unstable, QM, Q_out[..., L])
        Q_out[..., L + 1] = np.where(unstable, QM, Q_out[..., L + 1])

        TV = T_out * (1 + Q_out * deltx)

    return T_out, Q_out


def simil_numpy(z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg):
    """NumPy implementation of simil (Fortran-like, CPU baseline)."""
    kappa = 0.4
    gamamu = 16.0
    by3 = 1.0 / 3.0
    zetm = -1.0

    zet = z / lmonin
    zet0_m = z0m / lmonin
    zet0_h = z0h / lmonin
    zet0_q = z0q / lmonin

    x = np.where(1.0 - gamamu * zet >= 0, (1.0 - gamamu * zet) ** 0.25, 0.0)
    x0_m = np.where(1.0 - gamamu * zet0_m >= 0, (1.0 - gamamu * zet0_m) ** 0.25, 0.0)

    mask_unstable_gt = (zet < 0) & (zet > zetm)
    mask_unstable_le = (zet < 0) & (zet <= zetm)

    dpsim = np.where(
        mask_unstable_gt,
        np.log(((1 + x) * (1 + x) * (1 + x * x)) /
               ((1 + x0_m) * (1 + x0_m) * (1 + x0_m * x0_m))) -
        2 * (np.arctan(x) - np.arctan(x0_m)),
        np.where(
            mask_unstable_le,
            np.log(((1 + (1 - gamamu * zetm) ** 0.25) *
                    (1 + (1 - gamamu * zetm) ** 0.25) *
                    (1 + (1 - gamamu * zetm) ** 0.5)) /
                   ((1 + x0_m) * (1 + x0_m) * (1 + x0_m * x0_m))) -
            2 * (np.arctan((1 - gamamu * zetm) ** 0.25) - np.arctan(x0_m)) +
            np.log(zet / zetm) -
            1.140125 * ((-zet) ** by3 - (-zetm) ** by3),
            np.where(
                zet <= 1.0,
                -4.7 * (zet - zet0_m),
                -4.7 * (1.0 - zet0_m) +
                1.0 * (5.0 - 4.7) * np.log(zet / 1.0) -
                5.0 * (zet - 1.0)
            )
        )
    )

    dm = np.maximum(np.log(z / z0m) - dpsim, 1e-3)

    dpsih = np.log(z / z0h) - gamamu * (zet - zet0_h)
    dh = np.maximum(np.log(z / z0h) - dpsih, 1e-3)

    dpsiq = dpsih

    u = (ustar / kappa) * (np.log(z / z0m) - dpsim)
    t = tg + (tstar / kappa) * (np.log(z / z0h) - dpsih)
    q = qg + (qstar / kappa) * (np.log(z / z0q) - dpsiq)

    return u, t, q, dpsim, dpsih, dpsiq


def benchmark_module(name, jax_func, numpy_func, input_generator, configs, iterations=100):
    """Benchmark a single module across named (label, *shape) configs."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {name}")
    print(f"{'=' * 60}")

    for label, *shape in configs:
        n_points = 1
        for s in shape[:2]:
            n_points *= s
        print(f"\nConfig: {label}  (shape={tuple(shape)}, horizontal points={n_points})")
        print("-" * 60)

        inputs = input_generator(*shape)
        inputs_np = tuple(np.array(x) for x in inputs)

        _ = numpy_func(*inputs_np)
        _ = jax_func(*inputs)

        start_time = time.time()
        for _ in range(iterations):
            _ = numpy_func(*inputs_np)
        numpy_time = (time.time() - start_time) / iterations

        start_time = time.time()
        for _ in range(iterations):
            _ = jax_func(*inputs)
        jax_time = (time.time() - start_time) / iterations

        speedup = numpy_time / jax_time

        print(f"NumPy (CPU) time:  {numpy_time:.6f} seconds")
        print(f"JAX (CPU) time:    {jax_time:.6f} seconds")
        print(f"Speedup (JAX/NumPy): {speedup:.2f}x")


def main():
    im, jm, lm = get_p2saom40_dims()
    print("=" * 60)
    print("ROCKE-3D JAX Benchmark Suite -- P2SAoM40-grounded grid sizes")
    print("=" * 60)
    print(f"P2SAoM40 native grid: im={im} (lon), jm={jm} (lat), lm={lm} (atm layers)")
    print(f"Source: {P2SAOM40_RESTART}")

    iterations = 100

    # Native P2SAoM40 grid, plus a couple of tiled-up stress sizes for scaling
    # context (2x2 and 4x4 tiling of the native lon/lat grid; layer count
    # held fixed since that's a physical property of this config, not a
    # decomposition-domain size).
    drycnv_configs = [
        ("P2SAoM40 native (im x jm x lm)", im, jm, lm),
        ("2x2 tiles", im * 2, jm * 2, lm),
        ("4x4 tiles", im * 4, jm * 4, lm),
    ]

    def drycnv_input_generator(I, J, L):
        key = jax.random.PRNGKey(42)
        T = jax.random.uniform(key, (I, J, L), minval=200.0, maxval=300.0)
        Q = jax.random.uniform(key, (I, J, L), minval=0.0, maxval=0.02)
        PK = jax.random.uniform(key, (I, J, L), minval=0.5, maxval=1.0)
        PDSIG = jax.random.uniform(key, (I, J, L), minval=0.1, maxval=0.2)
        return T, Q, PK, PDSIG

    benchmark_module(
        "DRYCNV (Dry Convection) -- P2SAoM40 grid",
        dry_convection_mixing_jit,
        dry_convection_numpy,
        drycnv_input_generator,
        drycnv_configs,
        iterations,
    )

    # PBL similarity functions operate per horizontal gridcell (no vertical
    # layer dependence), so the relevant size is the horizontal point count.
    pbl_configs = [
        ("P2SAoM40 native (im x jm)", im, jm),
        ("2x2 tiles", im * 2, jm * 2),
        ("4x4 tiles", im * 4, jm * 4),
    ]

    def pbl_input_generator(I, J):
        size = I * J
        key = jax.random.PRNGKey(42)
        z = jax.random.uniform(key, (size,), minval=1.0, maxval=100.0)
        z0m = jax.random.uniform(key, (size,), minval=0.01, maxval=0.1)
        z0h = jax.random.uniform(key, (size,), minval=0.01, maxval=0.1)
        z0q = jax.random.uniform(key, (size,), minval=0.01, maxval=0.1)
        lmonin = jax.random.uniform(key, (size,), minval=-100.0, maxval=100.0)
        ustar = jax.random.uniform(key, (size,), minval=0.1, maxval=1.0)
        tstar = jax.random.uniform(key, (size,), minval=0.1, maxval=1.0)
        qstar = jax.random.uniform(key, (size,), minval=0.01, maxval=0.1)
        tg = jax.random.uniform(key, (size,), minval=280.0, maxval=320.0)
        qg = jax.random.uniform(key, (size,), minval=0.01, maxval=0.05)
        return z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg

    benchmark_module(
        "PBL (Planetary Boundary Layer) -- P2SAoM40 grid",
        simil_jit,
        simil_numpy,
        pbl_input_generator,
        pbl_configs,
        iterations,
    )

    print("\n" + "=" * 60)
    print("All P2SAoM40-grounded benchmarks completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
