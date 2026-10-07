#!/usr/bin/env python
"""Environment smoke test for the ROCKE-3D JAX port. Does NO real model work and needs NO repository data.

It mimics what the real run needs (imports, JAX in float64, the XLA flags our bitwise validation sets, a jit + lax.scan step on
model-sized arrays, host callbacks like the radiation server and the libimf bridge, big binary reads, NetCDF, a JAX compile cache check,
GPU memory) and prints PASS / WARN / FAIL lines plus a summary. Copy this single file anywhere and run it in your container:

    python -u smoke_test_env.py                  # all checks
    python -u smoke_test_env.py --repo /path/to/rocke3d_jax/fullfidelity   # also try importing our modules (no data read)

Exit code is 1 only if a check marked CRITICAL failed (JAX missing, no float64, jit failing); otherwise 0.
Written 2026-10-07 for the GPU run on Discover; tested on a CPU-only node (the GPU parts report the device they find).
"""
import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

RESULTS = []


def rec(status, name, detail="", critical=False):
    RESULTS.append((status, name, detail, critical))
    print(f"[{status:4s}] {name}" + (f": {detail}" if detail else ""), flush=True)


def try_import(mod, critical=False, note=""):
    try:
        m = importlib.import_module(mod)
        rec("PASS", f"import {mod}", getattr(m, "__version__", "") + (f" {note}" if note else ""))
        return m
    except Exception as e:  # noqa: BLE001
        rec("FAIL" if critical else "WARN", f"import {mod}", f"{type(e).__name__}: {e}", critical)
        return None


def child(code, env_extra=None, timeout=300):
    """Run a snippet in a fresh interpreter (XLA flags must be set before jax is imported)."""
    env = dict(os.environ)
    env.update(env_extra or {})
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip().splitlines()[-3:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="path to rocke3d_jax/fullfidelity: also try importing our modules (nothing is run)")
    args = ap.parse_args()
    t_all = time.time()

    # 1. platform and CPU affinity (our bitwise results depend on the core count: the job's core set must be known)
    print("== 1. platform")
    rec("INFO", "python", f"{sys.version.split()[0]} {platform.platform()}")
    aff = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    rec("INFO", "cpus visible / usable by this process", f"{os.cpu_count()} / {aff}   (record this: results differ between core counts)")
    for k in ("OMP_NUM_THREADS", "XLA_FLAGS", "JAX_COMPILATION_CACHE_DIR", "CLOUDS_JAX_CACHE", "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "CUDA_VISIBLE_DEVICES"):
        rec("INFO", f"env {k}", os.environ.get(k, "(unset)"))
    if os.environ.get("JAX_COMPILATION_CACHE_DIR") or os.environ.get("CLOUDS_JAX_CACHE"):
        rec("WARN", "compile cache is set", "a warm JAX compile cache changed results in our tests (D175): unset it for validated runs")

    # 2. python packages the real run imports
    print("== 2. packages")
    np = try_import("numpy", critical=True)
    jax = try_import("jax", critical=True)
    try_import("jaxlib")
    try_import("scipy")
    try_import("netCDF4", note="(restart and accumulation files)")
    try_import("mpmath", note="(binary128 emulation of REAL*16 Ti/Ti2b, D173)")
    try_import("omegaconf", note="(needed by some project modules)")
    try_import("pytest")
    if np is None or jax is None:
        return finish(t_all)

    # 3. JAX backend, float64 and device memory
    print("== 3. JAX backend")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    devs = jax.devices()
    rec("INFO", "backend", jax.default_backend())
    rec("INFO", "devices", ", ".join(f"{d.platform}:{d.device_kind}" for d in devs))
    rec("PASS" if jax.default_backend() == "gpu" else "WARN", "GPU backend active",
        "yes" if jax.default_backend() == "gpu" else "NO: JAX runs on CPU here (container without CUDA JAX, or no --nv / no GPU allocation?)")
    rec("PASS" if jnp.ones(2, dtype=jnp.float64).dtype == jnp.float64 else "FAIL", "float64 arrays", critical=True)
    try:
        ms = devs[0].memory_stats() or {}
        if ms:
            rec("INFO", "device memory", f"{ms.get('bytes_limit', 0) / 2**30:.1f} GiB limit, {ms.get('bytes_in_use', 0) / 2**20:.0f} MiB in use")
    except Exception:  # noqa: BLE001
        pass

    # 4. a fake 'model step': jit + lax.scan on model-sized arrays (IM=72, JM=46, LM=40), 6 steps, float64, no physics
    print("== 4. fake model step (jit + scan, model-sized float64 state)")
    IM, JM, LM, NSTEP = 72, 46, 40, 6
    rng = np.random.default_rng(0)
    state = {"T": jnp.asarray(250 + 30 * rng.random((LM, JM, IM))), "U": jnp.asarray(rng.standard_normal((LM, JM, IM))),
             "Q": jnp.asarray(1e-3 * rng.random((LM, JM, IM)))}

    def one_step(s, _):
        lap = lambda a: (jnp.roll(a, 1, 2) + jnp.roll(a, -1, 2) + jnp.roll(a, 1, 1) + jnp.roll(a, -1, 1) - 4 * a)
        T = s["T"] + 1e-3 * lap(s["T"]) - 1e-4 * s["U"]
        U = s["U"] * 0.999 + 1e-3 * lap(s["U"])
        Q = jnp.maximum(s["Q"] + 1e-6 * lap(s["Q"]), 0.0) * jnp.exp(-1e-6 * jnp.abs(U)) ** 0.25
        return {"T": T, "U": U, "Q": Q}, jnp.mean(T)

    run = jax.jit(lambda s: jax.lax.scan(one_step, s, None, length=NSTEP))
    t0 = time.time()
    out, means = run(state)
    jax.block_until_ready(out)
    t1 = time.time()
    out2, _ = run(state)
    jax.block_until_ready(out2)
    t2 = time.time()
    same = all(bool(jnp.array_equal(out[k], out2[k])) for k in out)
    rec("PASS", "jit+scan 6 steps", f"first call (compile) {t1 - t0:.2f} s, second {t2 - t1:.3f} s", critical=True)
    rec("PASS" if same else "FAIL", "repeat call bitwise identical", "", critical=False)
    rec("PASS" if all(bool(jnp.all(jnp.isfinite(out[k]))) for k in out) else "FAIL", "finite results")

    # 5. host callbacks: the real run calls the radiation server and a libimf bridge from inside the JAX program
    print("== 5. host callbacks (stand-ins for the radiation server and the libimf bridge)")
    calls = {"n": 0, "bytes": 0}

    def host_fn(x):  # runs on the host in Python; a real one would write a packet and wait for the Fortran server
        calls["n"] += 1
        calls["bytes"] += x.nbytes
        return np.asarray(x) * 1.0

    try:
        def with_cb(s):
            y = jax.pure_callback(host_fn, jax.ShapeDtypeStruct(s["T"].shape, s["T"].dtype), s["T"])
            return s["T"] + 0.0 * y
        f = jax.jit(with_cb)
        f(state).block_until_ready()
        t0 = time.time()
        for _ in range(3):
            f(state).block_until_ready()
        dt = (time.time() - t0) / 3
        rec("PASS", "pure_callback inside jit", f"{calls['n']} calls, {calls['bytes'] / 2**20:.0f} MiB moved, {dt * 1e3:.1f} ms per call round trip (device<->host of {state['T'].nbytes / 2**20:.1f} MiB)")
    except Exception as e:  # noqa: BLE001
        rec("FAIL", "pure_callback inside jit", f"{type(e).__name__}: {e}")
    import ctypes
    found = None
    for name in ("libimf.so", "libimf.so.6"):
        try:
            ctypes.CDLL(name)
            found = name
            break
        except OSError:
            pass
    rec("PASS" if found else "WARN", "Intel libimf runtime (ctypes)",
        found or "not found: bitwise-with-Fortran (category A) results need it; without it only libm-mode (rounding-level) results are possible")

    # 6. XLA flags the bitwise validation sets BEFORE importing jax (CPU flags); check they do not break a GPU build
    print("== 6. XLA flags used by the bitwise validation (fresh interpreter)")
    probe = "import jax, jax.numpy as jnp; jax.config.update('jax_enable_x64', True); print(jax.default_backend(), float(jnp.sum(jnp.arange(10.0))))"
    rc, tail = child(probe)
    rec("PASS" if rc == 0 else "FAIL", "jax starts with no XLA_FLAGS", " | ".join(tail), critical=True)
    rc, tail = child(probe, {"XLA_FLAGS": "--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp"})
    rec("PASS" if rc == 0 else "WARN", "jax starts with --xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp",
        " | ".join(tail) if rc else "ok (a build that rejects unknown flags would abort here)")

    # 7. file and NetCDF I/O patterns (the real run reads multi-hundred-MB binary dumps and NetCDF restarts)
    print("== 7. I/O")
    with tempfile.TemporaryDirectory() as td:
        a = rng.random(32 * 1024 * 1024 // 8)
        p = os.path.join(td, "dump.bin")
        t0 = time.time()
        a.astype(">f8").tofile(p)
        tw = time.time() - t0
        t0 = time.time()
        b = np.fromfile(p, dtype=">f8")
        tr = time.time() - t0
        rec("PASS" if np.array_equal(a, b) else "FAIL", "32 MiB big-endian float64 file round trip", f"write {32 / tw:.0f} MiB/s, read {32 / tr:.0f} MiB/s in {td}")
        try:
            import netCDF4
            q = os.path.join(td, "t.nc")
            with netCDF4.Dataset(q, "w") as d:
                d.createDimension("lat", JM)
                d.createDimension("lon", IM)
                v = d.createVariable("x", "f8", ("lat", "lon"))
                v[:] = a[: JM * IM].reshape(JM, IM)
            with netCDF4.Dataset(q) as d:
                ok = np.array_equal(d["x"][:], a[: JM * IM].reshape(JM, IM))
            rec("PASS" if ok else "FAIL", "NetCDF4 write/read round trip")
        except Exception as e:  # noqa: BLE001
            rec("WARN", "NetCDF4 write/read", f"{type(e).__name__}: {e}")
    rec("PASS" if os.access(os.getcwd(), os.W_OK) else "WARN", "current directory writable", os.getcwd())

    # 8. optional: can our own modules be imported (nothing is run, no data is read)
    if args.repo:
        print("== 8. project modules (import only)")
        sys.path.insert(0, args.repo)
        for m in ("dyn_jax_env", "clouds_jax_env", "intel_libm_ff", "dyn_step_jax2", "clouds_condse_jax", "ghy_jax", "ocean_step", "jax_atm_step", "model_driver"):
            try_import(m)

    return finish(t_all)


def finish(t_all):
    n = {s: sum(1 for r in RESULTS if r[0] == s) for s in ("PASS", "WARN", "FAIL")}
    crit = [r for r in RESULTS if r[0] == "FAIL" and r[3]]
    print(f"\nSUMMARY: {n['PASS']} pass, {n['WARN']} warn, {n['FAIL']} fail ({len(crit)} critical) in {time.time() - t_all:.1f} s")
    print(json.dumps({"pass": n["PASS"], "warn": n["WARN"], "fail": n["FAIL"], "critical_fail": [r[1] for r in crit]}))
    return 1 if crit else 0


if __name__ == "__main__":
    sys.exit(main())
