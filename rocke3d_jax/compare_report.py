"""
Assemble the P2SAoM40 Fortran-vs-JAX comparison results (accuracy + timing,
CPU and -- once compare_submit_gpu.sbatch has been run and its output
copied/shared back -- GPU) into a single compare_data/summary.json, and
print a text report.

Accuracy is computed against the Fortran output as reference (it's the
direct translation of this project's real ROCKE-3D physics), as max
absolute difference and RMSE, for every JAX leg (cpu, gpu) that has
written output files.

Usage:
    python compare_report.py
"""

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "compare_data")

IM, JM, LM = 72, 46, 40
N = IM * JM

DRYCNV_FIELDS = ["T", "Q"]
PBL_FIELDS = ["u", "t", "q", "dpsim", "dpsih", "dpsiq"]


def load_fortran_3d(field):
    path = os.path.join(DATA_DIR, f"drycnv_fortran_{field}_out.dat")
    return np.fromfile(path, dtype=np.float64).reshape((IM, JM, LM), order="F")


def load_fortran_1d(field):
    path = os.path.join(DATA_DIR, f"pbl_fortran_{field}_out.dat")
    return np.fromfile(path, dtype=np.float64)


def load_jax(module, device, field):
    path = os.path.join(DATA_DIR, f"{module}_jax_{device}_{field}_out.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def diff_stats(ref, other):
    d = np.abs(ref - other)
    return {
        "max_abs_diff": float(np.max(d)),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "ref_scale": float(np.mean(np.abs(ref)) + 1e-30),
    }


def available_jax_devices():
    devices = []
    for path in glob.glob(os.path.join(DATA_DIR, "timing_jax_*.json")):
        dev = os.path.basename(path).replace("timing_jax_", "").replace(".json", "")
        devices.append(dev)
    return sorted(devices)


def main():
    summary = {"grid": {"im": IM, "jm": JM, "lm": LM}, "timing": {}, "accuracy": {}}

    # --- Timing ---
    fortran_timing_path = os.path.join(DATA_DIR, "timing_fortran.json")
    if os.path.exists(fortran_timing_path):
        with open(fortran_timing_path) as f:
            summary["timing"]["fortran_cpu"] = json.load(f)
    else:
        print("WARNING: no timing_fortran.json -- run compare_run_fortran.py first")

    devices = available_jax_devices()
    for dev in devices:
        with open(os.path.join(DATA_DIR, f"timing_jax_{dev}.json")) as f:
            summary["timing"][f"jax_{dev}"] = json.load(f)
    if not devices:
        print("WARNING: no timing_jax_*.json found -- run compare_jax.py first")

    # --- Accuracy: DRYCNV ---
    drycnv_acc = {}
    for field in DRYCNV_FIELDS:
        ref = load_fortran_3d(field)
        for dev in devices:
            other = load_jax("drycnv", dev, field)
            if other is None:
                continue
            drycnv_acc.setdefault(dev, {})[field] = diff_stats(ref, other)
    summary["accuracy"]["drycnv"] = drycnv_acc

    # --- Accuracy: PBL ---
    pbl_acc = {}
    for field in PBL_FIELDS:
        ref = load_fortran_1d(field)
        for dev in devices:
            other = load_jax("pbl", dev, field)
            if other is None:
                continue
            pbl_acc.setdefault(dev, {})[field] = diff_stats(ref, other)
    summary["accuracy"]["pbl"] = pbl_acc

    out_path = os.path.join(DATA_DIR, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --- Text report ---
    print("=" * 70)
    print(f"P2SAoM40 Fortran vs JAX -- grid im={IM}, jm={JM}, lm={LM}")
    print("=" * 70)

    print("\nTiming (mean seconds/call, 100 calls):")
    for leg, t in summary["timing"].items():
        print(f"  {leg:12s}  DRYCNV={t['drycnv_mean_seconds']:.6e}  PBL={t['pbl_mean_seconds']:.6e}")

    print("\nAccuracy vs. Fortran reference (max_abs_diff / rmse):")
    for module, acc in summary["accuracy"].items():
        for dev, fields in acc.items():
            print(f"  [{module}] jax_{dev}:")
            for field, stats in fields.items():
                print(f"    {field:8s}  max_abs_diff={stats['max_abs_diff']:.3e}  rmse={stats['rmse']:.3e}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
