#!/usr/bin/env python
"""
P2SAoM40 JAX vs. Fortran physics-only comparison
====================================================

Runs the JAX physics orchestrator (p2saom40_driver.py) from the real
P2SAoM40 restart state (p2saom40_io.py) at the real 72x46x40 grid, and
produces:
  1. Functionality: real-grid global maps of the driver's output
     fields, saved as HTML to outputs/.
  2. Accuracy: a qualitative/pattern-level comparison against the real
     run's accumulated (period-mean) diagnostics -- NOT a per-timestep
     validation, since SUBDD (instantaneous output) was disabled in
     this rundeck. See FINDINGS.md.
  3. Performance (CPU): JAX driver wall-clock vs. the real per-routine
     Fortran costs measured in P2SAoM40.PRT.
  4. Performance (GPU): documented but not executed here -- no GPU is
     available in this environment.
  5. Functional coverage: which real call-order steps are faithfully
     ported vs. simplified vs. out of scope.

Usage: python p2saom40_compare.py
       JAX_PLATFORMS=cpu python p2saom40_compare.py   # force CPU explicitly
"""

import os
# No JAX_PLATFORMS default here: let JAX auto-detect (GPU if present, else
# CPU). This used to hardcode "cpu" as a default from when this script was
# only ever run on a CPU-only node -- that silently pinned every run to CPU
# even on a real GPU node, since os.environ.setdefault() only fills in a
# value that isn't already set, and nothing else was setting it. Pass
# JAX_PLATFORMS=cpu on the command line if you actually want to force CPU.

import time
import numpy as np
import jax

import p2saom40_io as io
import p2saom40_driver as drv

try:
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except ImportError:
    HAVE_PLOTLY = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_map(field, lat, lon, title, fname, colorscale="RdBu_r"):
    if not HAVE_PLOTLY:
        return
    fig = go.Figure(data=go.Heatmap(z=field, x=lon, y=lat, colorscale=colorscale))
    fig.update_layout(title=title, xaxis_title="Longitude", yaxis_title="Latitude",
                       width=900, height=550)
    path = os.path.join(OUT_DIR, fname)
    fig.write_html(path)
    print(f"  saved {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    section("1. Load real P2SAoM40 state (72x46x40)")
    state = io.load_restart_state(f"{io.RUN_DIR}/fort.1.nc")
    frac = io.load_surface_fractions()
    itype = io.build_itype(frac, state["rsi_atm"])
    static_fields = drv.build_static_fields(itype)
    start_dt = io.itime_to_datetime(state["itime"])
    counts = {n: int((itype == k).sum()) for k, n in [(1, "ocean"), (2, "seaice"), (3, "landice"), (4, "land")]}
    print(f"  restart: itime={state['itime']} ({start_dt} UTC)")
    print(f"  itype counts: {counts} (total {sum(counts.values())} / {itype.size})")
    print(f"  JAX devices: {jax.devices()}")

    section("2. Run one real DTsrc=1800s physics step")
    new_state, diag = drv.run_dtsrc_step(
        state, itype, static_fields, frac["lat"], frac["lon"], start_dt, step_index=0,
    )
    for k in ["tg", "cosz", "fsf", "flong", "sensible_heat_flux", "latent_heat_flux", "net_energy_flux"]:
        v = diag[k]
        print(f"  {k:24s} min={np.nanmin(v):10.3f}  max={np.nanmax(v):10.3f}  mean={np.nanmean(v):10.3f}")

    section("3. Functionality: real-grid global maps -> outputs/")
    lat, lon = frac["lat"], frac["lon"]
    save_map(itype.astype(float), lat, lon, "P2SAoM40 surface type (1=ocean,2=seaice,3=landice,4=land)",
              "p2saom40_itype_map.html", colorscale="Viridis")
    save_map(diag["tg"] - 273.15, lat, lon, "JAX: ground/skin temperature (C), one DTsrc step from real restart",
              "p2saom40_jax_tg_map.html")
    save_map(diag["sensible_heat_flux"], lat, lon, "JAX: sensible heat flux (W/m^2)",
              "p2saom40_jax_sensible_heat_map.html")
    save_map(diag["latent_heat_flux"], lat, lon, "JAX: latent heat flux (W/m^2)",
              "p2saom40_jax_latent_heat_map.html")
    save_map(diag["net_energy_flux"], lat, lon, "JAX: net surface energy flux (W/m^2)",
              "p2saom40_jax_net_energy_map.html")
    if not HAVE_PLOTLY:
        print("  plotly not available -- skipped HTML map generation")

    section("4. Accuracy: qualitative comparison vs. real period-mean diagnostics")
    print("  NOTE: PARTIAL.accP2SAoM40.nc holds PERIOD-MEAN accumulated fields,")
    print("  not per-timestep truth (SUBDD was disabled in this rundeck). This")
    print("  is a pattern/order-of-magnitude sanity check, not exact validation.")
    pmid, pedn = drv.build_pressure_profile(state["p"] + drv.PTOP, state["ma"])
    pk, _ = drv.compute_pk_pek(pmid, pedn)
    t1_actual_c = state["t"][..., 0] * pk[..., 0] - 273.15
    try:
        tsurf_real = io.decode_aij("tsurf")
        corr = np.corrcoef(t1_actual_c.ravel(), tsurf_real.ravel())[0, 1]
        bias = float(np.mean(t1_actual_c - tsurf_real))
        print(f"  layer-1 air temp (C) vs. real period-mean 'tsurf' (C):")
        print(f"    spatial pattern correlation = {corr:.3f}   mean bias = {bias:+.2f} C")
    except KeyError as e:
        print(f"  skipped tsurf comparison: {e}")
    try:
        sensht_real = io.decode_aij("sensht")
        corr_s = np.corrcoef(diag["sensible_heat_flux"].ravel(), sensht_real.ravel())[0, 1]
        print(f"  JAX sensible heat flux vs. real period-mean 'sensht':")
        print(f"    spatial pattern correlation = {corr_s:.3f}")
        print(f"    JAX range [{diag['sensible_heat_flux'].min():.0f}, {diag['sensible_heat_flux'].max():.0f}] W/m^2"
              f"  vs. real period-mean range [{sensht_real.min():.0f}, {sensht_real.max():.0f}] W/m^2")
    except KeyError as e:
        print(f"  skipped sensht comparison: {e}")

    section("5. Performance (CPU): JAX driver vs. real Fortran per-routine cost")
    _ = drv.run_dtsrc_step(state, itype, static_fields, lat, lon, start_dt, step_index=1)  # JIT warm-up
    n_rep = 20
    t0 = time.perf_counter()
    for i in range(n_rep):
        do_rad = (i % drv.NRAD == 0)
        drv.run_dtsrc_step(state, itype, static_fields, lat, lon, start_dt, step_index=i, do_radiation=do_rad)
    jax_ms = (time.perf_counter() - t0) / n_rep * 1000.0
    print(f"  JAX (this CPU, jit-compiled, {n_rep}-step average): {jax_ms:.2f} ms/DTsrc-step")
    print()
    print("  Real Fortran per-routine cost, measured in P2SAoM40.PRT (194 real")
    print("  DTsrc steps, single MPI process, same CPU-class hardware):")
    print("    RADIA()            65.57% of runtime, avg 2070 ms/call")
    print("                        (bimodal: ~0.7ms on 4/5 steps, ~10833ms on the")
    print("                         NRAD=5-gated 1/5 steps -- real spectral")
    print("                         radiative transfer; JAX's radiation_jax.py is")
    print("                         a simplified graybody formula, so this huge")
    print("                         JAX/Fortran gap reflects a fidelity difference,")
    print("                         not an apples-to-apples speed comparison)")
    print("    SURFACE()           8.87% of runtime, avg  263 ms/call")
    print("                        (includes NIsurf=2 PBL+EARTH+layer-1 substeps)")
    print("    GROUND_SI/LI/LK    ~0.16% of runtime, avg    ~1 ms/call combined")
    print("    ---- JAX-covered subset: ~74.6% of real per-step Fortran cost ----")
    print("    CONDSE() (moist convection)  9.91% -- OUT OF SCOPE, not ported")
    print("    Atm. Dynamics                8.50% -- OUT OF SCOPE, not ported")

    section("6. Performance (GPU)")
    if jax.devices()[0].platform == "gpu":
        print(f"  GPU detected: {jax.devices()}")
        _ = drv.run_dtsrc_step(state, itype, static_fields, lat, lon, start_dt, step_index=1)  # JIT warm-up (GPU)
        jax.block_until_ready(_)
        t0 = time.perf_counter()
        for i in range(n_rep):
            do_rad = (i % drv.NRAD == 0)
            out = drv.run_dtsrc_step(state, itype, static_fields, lat, lon, start_dt, step_index=i, do_radiation=do_rad)
        jax.block_until_ready(out)
        jax_gpu_ms = (time.perf_counter() - t0) / n_rep * 1000.0
        print(f"  JAX (this GPU, jit-compiled, {n_rep}-step average): {jax_gpu_ms:.2f} ms/DTsrc-step")
        print(f"  vs. JAX (CPU, above): {jax_ms:.2f} ms/DTsrc-step -> {jax_ms / jax_gpu_ms:.1f}x faster on GPU")
        print(f"  vs. real Fortran SURFACE+GROUND (~264 ms/DTsrc-step): {264.0 / jax_gpu_ms:.1f}x faster")
        print("  (radiation excluded from the Fortran comparison, as in section 5 --")
        print("   JAX's radiation is a simplified graybody stand-in, not equivalent physics)")
    else:
        print("  No GPU is visible in this environment (JAX reports:", jax.devices(), ")")
        print("  To benchmark on GPU: run this script on a GPU-enabled node/container")
        print("  without forcing JAX_PLATFORMS=cpu -- see rocke3d_jax/Dockerfile.gpu,")
        print("  which already installs jax[cuda12_pip]. No code changes needed; the")
        print("  driver is plain jax.numpy and will place arrays on whatever backend")
        print("  is active.")

    section("7. Functional coverage (orchestration fidelity)")
    rows = [
        ("Zenith angle",        "Simplified",  "declination+hour-angle formula, not the ORBIT module"),
        ("RADIATION",           "Simplified",  "graybody Stefan-Boltzmann, not spectral radiative transfer"),
        ("PBL (similarity fns)","Faithful",    "pbl.py: find_dpsim/find_dpsih/getcm/getchq, unit-validated"),
        ("Monin-Obukhov solve", "Simplified",  "fixed-point iteration, not PBL.f's Newton solve"),
        ("FLUXES formulas",     "Bug-fixed",   "ws was computed from surface wind only (=0 for land/ice) -- fixed locally in p2saom40_driver.py, see module docstring"),
        ("SURFACE dispatch",    "Simplified",  "single dominant itype per cell, no area-weighted sub-tiling"),
        ("GHY (land)",          "Not wired",   "ghy_jax.py's soil-moisture-table/flux-limit fns not yet used"),
        ("Layer-1 turbulence",  "Simplified",  "direct flux-tendency coupling, not aturb_jax's tridiagonal solve (placeholder PBL-top-finding)"),
        ("DRYCNV (layers 2..LM)","Faithful",   "drycnv.py, unit-validated, used as-is"),
        ("GROUND_SI/LI/LK",     "Not implemented", "seaice_jax/lakes_jax core thermodynamics are documented placeholders/no-ops in the existing port; surface temps held fixed at restart values"),
        ("Atm. Dynamics",       "Out of scope","no dynamical core ported"),
        ("CONDSE (moist conv.)","Out of scope","no moist convection ported"),
        ("Ocean GCM",           "Out of scope","SST approximated from layer-1 air temp; no ocean model ported"),
    ]
    width = max(len(r[0]) for r in rows)
    for name, status, note in rows:
        print(f"  {name:<{width}}  [{status:<16}]  {note}")

    print("\nDone.")


if __name__ == "__main__":
    main()
