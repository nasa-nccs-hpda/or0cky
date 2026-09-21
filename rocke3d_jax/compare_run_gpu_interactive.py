"""
GPU leg of the P2SAoM40 Fortran-vs-JAX comparison, for running directly on
an interactive GPU node (no SLURM/sbatch) -- e.g. after `salloc`/`ssh` onto
a GPU node yourself.

Does the whole GPU-side job in one shot:
  1. Confirms JAX actually sees a GPU (fails loudly rather than silently
     falling back to CPU and mislabeling the results).
  2. Ensures compare_data/*.npy shared inputs exist -- generates them ONLY
     if missing (first run), otherwise reuses whatever's already there so
     the GPU leg sees byte-identical inputs to the CPU/Fortran legs that
     may have already run elsewhere.
  3. Runs compare_jax.py's comparison (writes compare_data/timing_jax_gpu.json
     and compare_data/{drycnv,pbl}_jax_gpu_*_out.npy).
  4. Re-runs compare_report.py so compare_data/summary.json is immediately
     updated with the GPU numbers alongside whatever CPU/Fortran results
     are already in compare_data/ -- no separate step needed.

Usage (on a GPU node, in this directory):
    python compare_run_gpu_interactive.py

Requires: JAX built with CUDA support already installed in this Python
environment (e.g. `pip install "jax[cuda12]"` -- I can't install this for
you since I don't have GPU access to verify against; if `import jax` or
the GPU check below fails, that's the first thing to check).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("JAX_PLATFORMS", "cuda")

DATA_DIR = os.path.join(HERE, "compare_data")


def check_gpu():
    import jax
    try:
        devices = jax.devices()
    except Exception as e:
        raise RuntimeError(
            f"JAX could not initialize a GPU backend here ({e!r}). Likely "
            "causes: this Python environment doesn't have a CUDA-enabled "
            "JAX installed (try `pip install \"jax[cuda12]\"`), or this "
            "node has no visible GPU (`nvidia-smi` should show a device)."
        ) from e
    platforms = {d.platform for d in devices}
    print(f"JAX devices: {devices}")
    if "gpu" not in platforms:
        raise RuntimeError(
            f"No GPU visible to JAX (platforms seen: {platforms}). "
            "Check that you're on a GPU node (`nvidia-smi` should show a "
            "device) and that this Python environment has a CUDA-enabled "
            "JAX installed (`python -c \"import jax; print(jax.devices())\"` "
            "should list a GpuDevice)."
        )
    print(f"Confirmed GPU device(s): {[d for d in devices if d.platform == 'gpu']}")


def ensure_inputs():
    have_inputs = os.path.isdir(DATA_DIR) and any(
        f.endswith(".npy") and not f.startswith(("drycnv_jax", "pbl_jax"))
        for f in os.listdir(DATA_DIR)
    )
    if have_inputs:
        print(f"Reusing existing shared inputs in {DATA_DIR}")
        return
    print(f"No shared inputs found in {DATA_DIR} -- generating them now.")
    import compare_generate_inputs
    compare_generate_inputs.main()


def main():
    print("=" * 60)
    print("P2SAoM40 GPU leg (interactive, no SLURM)")
    print("=" * 60)

    check_gpu()
    ensure_inputs()

    print("\n--- Running JAX on GPU ---")
    import compare_jax
    compare_jax.main()

    print("\n--- Updating comparison report with GPU results ---")
    import compare_report
    compare_report.main()

    print("\nDone. compare_data/summary.json now includes the GPU leg.")
    print("Copy compare_data/ back (or confirm shared filesystem) and let ")
    print("Claude know so the results can go into the comparison slide.")


if __name__ == "__main__":
    main()
