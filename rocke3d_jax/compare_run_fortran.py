"""
Compile (if needed) and run compare_fortran.f90, parse its timing output,
and write compare_data/timing_fortran.json in the same format compare_jax.py
uses for its timing files, so compare_report.py can treat all legs uniformly.

Usage:
    python compare_run_fortran.py
"""

import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "compare_data")
EXE = os.path.join(HERE, "compare_fortran")
SRC = os.path.join(HERE, "compare_fortran.f90")


def ensure_binary():
    if os.path.exists(EXE):
        return
    print("Compiling compare_fortran.f90 with ifort -O2 ...")
    result = subprocess.run(["ifort", "-O2", "-o", EXE, SRC], cwd=HERE, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(EXE):
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(
            "Failed to compile compare_fortran.f90 -- run "
            "`module load intel/2020Update4` in this shell first."
        )


def main():
    ensure_binary()
    result = subprocess.run([EXE], cwd=HERE, capture_output=True, text=True, timeout=300)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"compare_fortran exited with code {result.returncode}")

    timing = {}
    for line in result.stdout.splitlines():
        if line.strip().startswith("DRYCNV_MEAN_SECONDS"):
            timing["drycnv_mean_seconds"] = float(line.split()[1])
        elif line.strip().startswith("PBL_MEAN_SECONDS"):
            timing["pbl_mean_seconds"] = float(line.split()[1])

    if "drycnv_mean_seconds" not in timing or "pbl_mean_seconds" not in timing:
        raise RuntimeError(f"Could not parse timing from compare_fortran output:\n{result.stdout}")

    timing["device"] = "fortran_cpu"
    timing["iterations"] = 100

    out_path = os.path.join(DATA_DIR, "timing_fortran.json")
    with open(out_path, "w") as f:
        json.dump(timing, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
