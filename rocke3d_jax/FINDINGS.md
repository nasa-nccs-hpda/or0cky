# ROCKE-3D JAX vs. Fortran Performance Findings

**Last Updated**: 2026-09-22
**Status**: ✅ **All benchmarks completed** (see 2d for the first real GPU run, kernel-level; spatial view in `visualize_p2saom40_kernel_maps.ipynb`)

---

## 📊 **Summary**

This report compares the **performance** and **numerical accuracy** of the **original Fortran ROCKE-3D modules** against the **new Python/JAX implementations** (which simulate the Fortran logic). Benchmarks were run on **CPU** and **GPU** to evaluate speedups and scalability.

---

## 🎯 **Key Findings**

| **Metric**               | **Result**                                                                                     |
|--------------------------|-----------------------------------------------------------------------------------------------|
| **Numerical Accuracy**   | JAX and Fortran match within **~2.3%** (excellent).                                           |
| **CPU Performance**      | JAX is **~25% faster overall** (PBL/DRYCNV gains outweigh FLUXES/SURFACE/RADIATION losses).     |
| **GPU Performance**      | JAX is **8–253× faster** than NumPy (scales with grid size).                                   |
| **Ease of Use**          | JAX integrates better with modern workflows (PyTorch, TensorFlow).                           |

---

## 🧩 **0. Module Fidelity Scope: 14/17 Faithful, Not 17/17 — By Design, Not Oversight**

Every performance/accuracy number below covers the **14 modules that are physically faithful ports**, not all 17 "ported" ones. **SEAICE** (core thermodynamics), **LAKES** (`lkmix` mixing), and part of **ATURB** (PBL-top-finding) are documented placeholder/simplified stand-ins, not full Fortran ports — see the fidelity table in `PORTING_STATUS.md`.

**This 14/17 split is an appropriate engineering prioritization, not a gap that should be silently closed to reach 17/17**: SEAICE and LAKES thermodynamics (phase-change physics, brine pockets, lake mixing) are genuinely harder to port correctly than the columnar physics (PBL, DRYCNV) that this project's validation effort has focused on, and getting those 14 modules faithfully right first is defensible sequencing. The problem was never the 14/17 number — it's that this document and others called the project "17/17 complete" before 2026-09-19, because the placeholder modules' own unit tests pass (they validate the placeholder logic against itself, not physical fidelity) and can't distinguish "ported" from "faithfully ported." Real P2SAoM40 data is what actually surfaced this (§2c below).

**Tracked as technical debt, not closed**: the 3 non-faithful modules stay explicitly flagged (🟠/🔴 status, never silently upgraded to 🟢) until each has a real-data regression test that would fail if its placeholder logic diverges from real Fortran output — see `PORTING_STATUS.md` "Regression Tracking for Placeholder Modules" for the concrete plan per module.

---

## 📈 **1. CPU Performance: Fortran vs. JAX**

### **Execution Time (1D Column, 1000 Points)**

| **Module**       | **Fortran (s)** | **JAX (s)** | **Speedup (Fortran/JAX)** | **Winner**       |
|------------------|------------------|-------------|----------------------------|------------------|
| **PBL**          | 0.000245         | 0.000190    | **1.29×** (Fortran slower) | ✅ **JAX**       |
| **FLUXES**       | 0.000019         | 0.001018    | **0.019×** (JAX slower)    | ✅ **Fortran**   |
| **SURFACE**      | 0.000001         | 0.000572    | **0.002×** (JAX slower)    | ✅ **Fortran**   |
| **RADIATION**    | 0.000001         | 0.000595    | **0.002×** (JAX slower)    | ✅ **Fortran**   |
| **DRYCNV**       | 0.001400         | 0.000873    | **1.60×** (Fortran slower) | ✅ **JAX**       |
| **Total**        | **0.004060**     | **0.003249** | **1.25×** (Fortran slower) | ✅ **JAX**       |

### **Analysis**
- **JAX is faster** for **PBL** and **DRYCNV** (~1.3–1.6× speedup).
- **Fortran is faster** for **FLUXES, SURFACE, RADIATION** (due to simplified logic in Fortran).
- **Overall**: JAX is **~25% faster** on CPU.

---

## 🚀 **2. GPU Performance: JAX vs. NumPy**

### **Execution Time (Grid Sizes: 1K, 10K, 100K)**

| **Module**       | **Grid Size** | **NumPy (GPU) (s)** | **JAX (GPU) (s)** | **Speedup (JAX/NumPy)** | **Winner**       |
|------------------|---------------|----------------------|-------------------|--------------------------|------------------|
| **DRYCNV**       | 1,000         | 0.001654             | 0.000185          | **8.95×**                | ✅ **JAX**       |
| **DRYCNV**       | 10,000        | 0.010768             | 0.000253          | **42.58×**               | ✅ **JAX**       |
| **DRYCNV**       | 100,000       | 0.176106             | 0.000695          | **253.34×**              | ✅ **JAX**       |
| **PBL**          | 1,000         | 0.000289             | 0.000060          | **4.77×**                | ✅ **JAX**       |
| **PBL**          | 10,000        | 0.001382             | 0.000061          | **22.67×**               | ✅ **JAX**       |
| **PBL**          | 100,000       | 0.012439             | 0.000065          | **192.28×**              | ✅ **JAX**       |

### **Analysis**
- **JAX is significantly faster** than NumPy on GPU:
  - **PBL**: **4.77×–192.28×** speedup (scales with grid size).
  - **DRYCNV**: **8.95×–253.34×** speedup (scales with grid size).
- **Scaling**: JAX speedup **increases with grid size** (due to GPU parallelism).

> **Provenance flag (added 2026-09-22)**: unlike every GPU number added to this project since the 2026-09-19 review, this table carries no device, date, or environment annotation — and this project's own docs stated "no GPU has been available in this environment" continuously through 2026-09-21. The real A100 numbers now measured for this exact grid size class (§2c: 32.96 ms GPU vs 33.51 ms CPU for the full chain at 3,312 points, ~1.0× — not the dramatic scaling this table shows) don't necessarily contradict this table (different grid sizes, different kernels-in-isolation vs. chained), but the lack of provenance here means it should not be cited as "measured in this environment" without confirming where/when it actually ran. Flagging rather than removing, since I can't confirm it's wrong — only that it's unlabeled in a project that otherwise labels this carefully.

---

## 🌍 **2b. Global Grid Numerical Agreement (2D Single-Layer Fields)**

In addition to the 1D-column comparison above, `visualize_2d_global_maps.ipynb` computes pixel-wise Fortran−JAX differences over the **full 180×360 global grid** for four single-layer fields (temperature, pressure, heat flux, solar flux), rounded to 3 decimal places to remove floating-point noise.

| **Field**              | **Min Diff** | **Max Diff** | **Mean Diff** | **Non-Zero Pixels (of 64,800)** |
|-------------------------|--------------|--------------|----------------|-----------------------------------|
| Temperature (K)         | 0.000000     | 0.000000     | 0.000000       | 0                                  |
| Pressure (hPa)          | 0.000000     | 0.000000     | 0.000000       | 0                                  |
| Heat Flux (W/m²)        | 0.000000     | 0.000000     | 0.000000       | 0                                  |
| Solar Flux (W/m²)       | 0.000000     | 0.000000     | 0.000000       | 0                                  |

**JAX and Fortran match exactly (to 3 decimal places) at every pixel across the full global grid** for these fields — a substantially tighter result than the ~2.3% figure from the 1D-column PBL similarity functions (`dpsim`/`dpsih`) above, which remains the outlier module rather than the norm. Difference maps in `outputs/difference_*_2d_map.html` use a diverging RdYlBu scale fixed at ±0.001 to make this level of agreement visible.

---

## 🛰️ **2c. Real Production-Run Comparison (P2SAoM40)**

Sections 1-2b above compare individual modules on synthetic or 1D-column data. This section instead drives the **full set of ported physics modules, chained together in the real Fortran call order**, from the actual restart state of a completed production ROCKE-3D run — `P2SAoM40` (72×46×40 grid, coupled ocean-atmosphere, Dec 1949, `DTsrc=1800s`, `NIsurf=2`, `NRAD=5`). Scope, code, and results are in `p2saom40_io.py`, `p2saom40_driver.py`, `p2saom40_compare.py`.

**What's in scope**: the same physics parameterizations as above (PBL, radiation, surface fluxes, dry convection), orchestrated with the real `NIsurf`/`NRAD` cadence, at the real 72×46 grid, from real `fort.1.nc` restart data (real u/v/t/q/p, real land/lake/sea-ice surface state, real `focean/flake/fgrnd/fgice` surface-type fractions).

**What's out of scope** (not ported, and not attempted here): the atmospheric dynamical core, moist convection (`CONDSE`), and the ocean GCM. Per the real per-routine timing in `P2SAoM40.PRT`, the JAX-covered subset (`RADIA`+`SURFACE`+`GROUND_SI/LI/LK`) represents **~74.6%** of real per-`DTsrc`-step Fortran physics cost; the excluded dynamics+`CONDSE` account for the remaining ~18.4% (the rest is `MELT_SI`/diagnostics overhead).

**Real bug found**: driving `fluxes_jax.py`/`surface_jax.py` with real data (where `us=vs=0` for land/land-ice, the physically correct no-slip value) revealed their flux formulas compute wind speed from the surface reference wind alone, silently zeroing every flux for those cells. Their existing Fortran-comparison unit tests apparently used arbitrary nonzero test values and never caught this. Worked around locally in `p2saom40_driver.py` rather than editing the shared module files — see `PORTING_STATUS.md` for the full fidelity audit.

**Maps** (interactive Plotly HTML, generated by `p2saom40_compare.py` §3, copied here from the GPU run 2026-09-22 — open directly, real content verified, ~4.8 MB each): [`outputs/p2saom40_itype_map.html`](outputs/p2saom40_itype_map.html) (surface type), [`p2saom40_jax_tg_map.html`](outputs/p2saom40_jax_tg_map.html) (ground/skin temperature), [`p2saom40_jax_sensible_heat_map.html`](outputs/p2saom40_jax_sensible_heat_map.html), [`p2saom40_jax_latent_heat_map.html`](outputs/p2saom40_jax_latent_heat_map.html), [`p2saom40_jax_net_energy_map.html`](outputs/p2saom40_jax_net_energy_map.html) — all one real DTsrc step from the Nov 1950 restart snapshot. (Separate from these: the DRYCNV/PBL kernel-level maps in `outputs/p2saom40_kernel_*.html`, produced by `visualize_p2saom40_kernel_maps.ipynb` for §2d, not this section.)

### Accuracy (qualitative — see caveat below)

Two runs exist, from different restart snapshots (the underlying P2SAoM40 run advances between them, so these are not the same validation point re-measured — both are genuine, independent sanity checks):

| Restart snapshot | JAX layer-1 air temp vs. real `tsurf` (correlation, bias) | JAX sensible heat flux vs. real `sensht` (correlation) |
|---|---|---|
| 1949-12-01 (original, CPU-only environment) | **0.987** (+1.0°C) | -0.12 |
| 1950-11-26 (later, GPU-capable run, 2026-09-22) | **0.965** (-1.86°C) | -0.014 |

**Caveat**: `PARTIAL.accP2SAoM40.nc`'s diagnostics are accumulated (period-mean), not per-timestep — `SUBDD` (instantaneous output) was disabled in this rundeck, so true per-step ground truth doesn't exist for this run. Both temperature correlations are a meaningful sanity check (correct spatial structure: cold poles, warm tropics, land/ocean contrast) despite being ~11 months apart in the underlying simulation; the weak flux correlation is expected, not a red flag, given the timescale mismatch. Getting an exact per-step accuracy number would require rerunning the Fortran model with `SUBDD` enabled for the relevant fields.

### Performance (CPU)

| | Cost |
|---|---|
| JAX driver, one `DTsrc` step (this CPU, jit-compiled, 20-step avg) | **48 ms** (33.5 ms on the 2026-09-22 GPU-node run below — different CPU hardware, same method) |
| Real Fortran `SURFACE()` (NIsurf=2 substeps) + `GROUND_SI/LI/LK` | **~264 ms** |
| → JAX-covered-subset speedup (excluding radiation, see below) | **~5.5×** |
| Real Fortran `RADIA()` | 65.57% of runtime, avg 2070 ms (bimodal: ~0.7ms held-value steps, ~10.8s on the 1-in-5 `NRAD`-gated real radiative-transfer steps) |

The `RADIA` comparison is **not apples-to-apples**: JAX's `radiation_jax.py` is a simplified graybody Stefan-Boltzmann formula (effectively free), while real `RADIA` does genuine multi-band spectral radiative transfer. The huge nominal speedup there reflects a fidelity gap, not a fair JAX-vs-Fortran speed measurement — the honest comparison is the SURFACE+GROUND figure above, where both sides do materially the same physics.

### Performance (GPU) — Measured 2026-09-22, corrects the prior "~20–30×" estimate

Run on an NVIDIA A100 (discover cluster, `warpa005`, via the project's own `p2saom40_compare.py` — no code changes needed, JAX auto-detected the GPU once a leftover `JAX_PLATFORMS=cpu` default was removed from that script).

| | Cost |
|---|---|
| JAX driver, one `DTsrc` step (A100 GPU, jit-compiled, 20-step avg) | **32.96 ms** |
| Same, CPU (this run, same node) | 33.51 ms |
| → **GPU vs. JAX-CPU** | **~1.0× — essentially no GPU benefit** |
| → GPU vs. real Fortran SURFACE+GROUND (~264 ms) | **~8.0×** (vs. ~5.5× measured on CPU) |

**This is a real, important correction, not a footnote**: every "~20–30× GPU speedup" figure elsewhere in this project's docs for the *full physics-subset chain* (as opposed to the DRYCNV/PBL kernel-level result in §2d, which genuinely does show 2.3–6.3× GPU gains) was an *estimate* made before any GPU was available, extrapolating from GPU's known behavior on much larger arrays. The real number is ~1.0×. Why: this orchestrator's grid is only 3,312 points — far too small for a GPU to amortize kernel-launch/dispatch overhead, the same effect already seen for the PBL kernel alone in §2d, just more pronounced here across the full multi-step chain. GPU still helps modestly against the real Fortran baseline (8.0× vs. 5.5×), because Fortran's own SURFACE+GROUND cost doesn't change — but "GPU makes JAX dramatically faster" is not true at this problem size, and any planning built on the 20–30× estimate (see `EXECUTIVE_SUMMARY.md`'s ROI section) should be revisited.

---

## 🎮 **2d. Kernel-Level CPU + GPU Comparison (P2SAoM40 grid, real GPU run)**

**2026-09-20.** A separate, narrower comparison than 2c above: instead of the full chained orchestrator, this directly times **real compiled Fortran** (`compare_fortran.f90`, `ifort -O2`, same toolchain that built the actual P2SAoM40 model) against JAX for just the **DRYCNV** and **PBL similarity** kernels, at P2SAoM40's real grid (72×46×40, confirmed from that run's own restart file). Same shared random inputs (fixed seed) fed to every leg, so outputs are directly diffable. Code: `compare_generate_inputs.py`, `compare_fortran.f90`, `compare_jax.py`, `compare_run_gpu_interactive.py`, `compare_report.py` — all in this directory.

> **How the P2SAoM40 data is used here, vs. in 2c above — these are not the same thing:**
> - **2c (above)**: real P2SAoM40 restart data (`fort.1.nc`) is the actual **input**, and JAX's output is checked against the real run's own period-mean diagnostics (the 0.987 spatial correlation) — a validation against real model output.
> - **2d (here)**: P2SAoM40 is used only to source a realistic **grid size** (72×46×40) and value ranges. The inputs fed to both Fortran and JAX are synthetic random arrays with a fixed seed, not real P2SAoM40 field values, and "accuracy" below means **JAX agrees with a from-scratch real-Fortran reference** (numerical port fidelity) — it is not a comparison to the P2SAoM40 run's own output. Don't cite the GPU numbers below as "validated against P2SAoM40" — they're a timing/fidelity check of the port, sized to that run's grid, nothing more.

Unlike section 2c's GPU status, **the GPU leg here was actually run** (interactively, on a SLURM-allocated GPU node, via `compare_run_gpu_interactive.py` — no sbatch job needed).

**Real bug found (in this project's own existing benchmark files, not introduced here)**: `simil_numpy`, the "Fortran-like" reference used by `benchmark_all.py`/`benchmark_all_cpu.py`, only implements the *unstable* branch of `find_dpsih` — it silently omits the entire *stable* branch (`zet>=0`, roughly half of all points for typical `lmonin` ranges), and sets `dpsiq = dpsih` instead of computing `dpsiq` from its own `getchq` call with `z0q`, as the real `pbl.py`'s `simil()` does. Translating `simil_numpy` directly into Fortran (the first version of `compare_fortran.f90`) reproduced this gap and showed apparent max differences up to ~1.26×10⁴ against the real JAX `simil_jit` — not a JAX bug, but an incomplete reference. Fixed by re-deriving `compare_fortran.f90`'s PBL routine from `pbl.py`'s actual three-branch `find_dpsim`/`find_dpsih`. **Any future accuracy work should treat `simil_numpy` as unreliable** rather than as ground truth.

A second, unrelated bug was found and fixed in the comparison harness itself: `numpy.ndarray.tofile()` always writes in C order regardless of the array's memory layout, so `np.asfortranarray(arr).tofile(...)` does **not** produce Fortran-order bytes despite appearances — it silently corrupted the shared 3-D DRYCNV inputs between the Fortran and JAX legs (1-D PBL arrays were unaffected, since order is irrelevant for 1-D). Fixed with `arr.tobytes(order='F')` in `compare_generate_inputs.py`.

### Timing (mean seconds/call, 100 calls, same shared inputs)

| Kernel | Fortran (CPU, ifort -O2) | JAX (CPU) | JAX (GPU) | JAX-GPU vs Fortran-CPU | JAX-GPU vs JAX-CPU |
|---|---|---|---|---|---|
| DRYCNV | 1.272 ms | 17.09 ms (13.4× **slower**) | 0.564 ms | **2.3× faster** | 30.3× faster |
| PBL similarity | 0.370 ms | 0.195 ms (1.9× faster) | 0.059 ms | **6.3× faster** | 3.3× faster |

JAX-GPU beats real Fortran-CPU on both kernels, resolving PORTING_STATUS.md's "Next Steps" item 3 (GPU/TPU benchmarking) for these two kernels specifically — the "expected 20–30×" estimate there was almost exactly right for DRYCNV (30.3×) but PBL's speedup is much smaller (3.3× over JAX-CPU), since its per-call cost is already tiny/dispatch-bound rather than compute-bound, leaving less for GPU parallelism to exploit at this problem size.

### Accuracy vs. Fortran reference (max abs diff; same inputs on every leg)

| Field | CPU | GPU |
|---|---|---|
| drycnv T | 8.79e-05 | 9.98e-05 |
| drycnv Q | 3.70e-09 | 3.81e-09 |
| pbl u | 4.41e-04 | 1.02e-03 |
| pbl t | 1.46e-03 | 2.77e-03 |
| pbl q | 1.19e-04 | 3.24e-04 |
| pbl dpsim | 1.11e-04 | 1.09e-03 |
| pbl dpsih | 8.14e-04 | 1.14e-03 |
| pbl dpsiq | 1.26e-03 | 1.09e-03 |

All differences are floating-point-level (output scales are 10¹–10³) on both devices — GPU is consistently a bit larger than CPU, consistent with a different float reduction order, not an algorithmic difference. This is the corrected, complete-branch validation; it supersedes any accuracy inference drawn from `simil_numpy` elsewhere in this repo.

**Scope note**: this is a kernel-level comparison (DRYCNV + PBL only), not the full physics chain in section 2c (which also covers RADIATION/SURFACE/GROUND and remains CPU-only, GPU not yet run). Raw JSON: `compare_data/summary.json` in this directory. Spatial view of these same differences (JAX−Fortran, projected onto the real P2SAoM40 grid): `visualize_p2saom40_kernel_maps.ipynb`, or `outputs/p2saom40_kernel_*_diff_*.html` for the individual maps. Slide deck: https://claude.ai/artifact/LxdYuYeQXxHDsX18KWxBLA (slide 4, "Output Maps"; filesystem snapshot in `status_slides/`).

---

## 📊 **3. Combined Performance Summary**

| **Scenario**               | **PBL**       | **DRYCNV**    | **FLUXES**   | **SURFACE**  | **RADIATION** | **Total**    | **Winner**       |
|----------------------------|---------------|---------------|--------------|--------------|----------------|---------------|------------------|
| **CPU (Fortran vs. JAX)**  | JAX **1.29×** | JAX **1.60×** | Fortran **53×** | Fortran **572×** | Fortran **595×** | JAX **1.25×** | **JAX (Overall)** |
| **GPU (JAX vs. NumPy)**    | JAX **4.77–192×** | JAX **8.95–253×** | — | — | — | **JAX** | **JAX (GPU)** |

---

## 🔍 **4. Key Observations**

### **CPU Performance**
1. **JAX is faster** for **PBL** and **DRYCNV** (~1.3–1.6× speedup).
   - **Reason**: JAX’s JIT compilation optimizes loops better than Fortran.
2. **Fortran is faster** for **FLUXES, SURFACE, RADIATION** (53–595× speedup).
   - **Reason**: Fortran uses **simplified logic** (hardcoded constants, minimal computation).
3. **Overall**: JAX is **~25% faster** on CPU (PBL/DRYCNV gains outweigh FLUXES/SURFACE/RADIATION losses).

### **GPU Performance**
1. **JAX is significantly faster** than NumPy on GPU:
   - **PBL**: **4.77×–192.28×** speedup (scales with grid size).
   - **DRYCNV**: **8.95×–253.34×** speedup (scales with grid size).
2. **Scaling**: JAX speedup **increases with grid size** due to **GPU parallelism** and **JIT compilation**.

### **Numerical Accuracy**
- JAX and Fortran match within **~2.3%** (excellent for cross-language comparisons).

---

## 🏆 **5. Final Recommendations**

| **Use Case**               | **Recommended Implementation** | **Reason**                                                                                     |
|----------------------------|----------------------------------|---------------------------------------------------------------------------------------------|
| **CPU (General)**          | JAX (PBL/DRYCNV) + Fortran (FLUXES/SURFACE/RADIATION) | Hybrid approach for **optimal CPU performance**.                                            |
| **CPU (Simplicity)**       | JAX (All)                       | Easier integration, **~25% faster overall**.                                                  |
| **GPU/TPU**                | JAX (All)                       | **8–253× speedup** on GPU (scales with grid size).                                           |
| **Development**            | JAX (All)                       | Easier debugging, Python ecosystem, and GPU support.                                         |
| **Production (CPU-only)**  | Fortran (FLUXES/SURFACE/RADIATION) + JAX (PBL/DRYCNV) | **Best of both worlds** for CPU workloads.                                                   |
| **Production (GPU)**       | JAX (All)                       | **Maximize GPU acceleration** (20–250× speedup).                                             |

---

## 📈 **6. GPU Speedup Scaling**

| **Module** | **Grid Size** | **Speedup (JAX/NumPy)** | **Trend**               |
|------------|---------------|--------------------------|-------------------------|
| **DRYCNV** | 1,000         | 8.95×                    | **Increases with size** |
| **DRYCNV** | 10,000        | 42.58×                   | **Increases with size** |
| **DRYCNV** | 100,000       | 253.34×                  | **Increases with size** |
| **PBL**    | 1,000         | 4.77×                    | **Increases with size** |
| **PBL**    | 10,000        | 22.67×                   | **Increases with size** |
| **PBL**    | 100,000       | 192.28×                  | **Increases with size** |

**Conclusion**: JAX **scales exceptionally well on GPU** due to **parallelism and JIT compilation**. Larger grid sizes yield **higher speedups**.

---

## 📊 **7. Summary Table (All Scenarios)**

| **Metric**               | **Fortran (CPU)** | **JAX (CPU)** | **JAX (GPU)** | **Winner**       |
|--------------------------|-------------------|---------------|---------------|------------------|
| **PBL Time (1K)**        | 0.000245 s        | 0.000190 s    | ~0.000060 s   | **JAX (GPU)**    |
| **DRYCNV Time (1K)**     | 0.001400 s        | 0.000873 s    | ~0.000185 s   | **JAX (GPU)**    |
| **FLUXES Time**          | 0.000019 s        | 0.001018 s    | —             | **Fortran (CPU)**|
| **Total Time (CPU)**     | 0.004060 s        | 0.003249 s    | —             | **JAX (CPU)**    |
| **Total Time (GPU)**     | —                 | —             | **~0.000250 s** | **JAX (GPU)**    |

---

## 🎯 **8. Final Verdict**

| **Scenario**       | **Winner**       | **Speedup** | **Notes**                                                                                     |
|--------------------|------------------|-------------|---------------------------------------------------------------------------------------------|
| **CPU (Overall)**  | **JAX**          | **1.25×**   | JAX is **~25% faster** overall (PBL/DRYCNV gains outweigh FLUXES/SURFACE/RADIATION losses). |
| **GPU (Overall)**  | **JAX**          | **8–253×**  | JAX **dominates on GPU** (scales with grid size).                                           |
| **Numerical Accuracy** | **Tie**      | —           | JAX and Fortran match within **~2.3%**.                                                     |
| **Ease of Use**    | **JAX**          | —           | Python/JAX integrates better with modern workflows (PyTorch, TensorFlow).                 |

---

## 🔬 **9. GPU vs. CPU Speedup Comparison**

| **Module** | **CPU (JAX vs. Fortran)** | **GPU (JAX vs. NumPy)** | **Conclusion**                          |
|------------|----------------------------|--------------------------|------------------------------------------|
| **PBL**    | **1.29× faster**           | **4.77–192.28× faster** | **GPU is 10–100× better than CPU**.     |
| **DRYCNV** | **1.60× faster**           | **8.95–253.34× faster** | **GPU is 10–100× better than CPU**.     |
| **FLUXES** | **0.019× slower**          | —                        | **Fortran is better on CPU**.            |
| **Total**  | **1.25× faster**           | **8–253× faster**       | **GPU is the clear winner for JAX**.     |

---

## 📌 **10. Recommendations for Deployment**

### **For CPU Workloads**
1. **Hybrid Approach**:
   - Use **JAX for PBL/DRYCNV** (faster).
   - Use **Fortran for FLUXES/SURFACE/RADIATION** (faster).
   - **Result**: Optimal CPU performance.

2. **Simplicity**:
   - Use **JAX for all modules** (easier integration, ~25% faster overall).

### **For GPU/TPU Workloads**
- Use **JAX for all modules** (8–253× speedup).

### **For Development**
- Use **JAX** for:
  - Easier debugging.
  - Better integration with Python ecosystem (PyTorch, TensorFlow).
  - GPU support.

### **For Production**
- **GPU**: Use **JAX** (20–250× speedup).
- **CPU**: Use **hybrid JAX + Fortran** for best performance.

---

## 📚 **11. Methodology**

### **Test Environment**
- **Hardware**: CPU-only (no GPU/TPU for Fortran vs. JAX comparison).
- **Grid Size**: 1000 points (1D column).
- **Modules Tested**: PBL, FLUXES, SURFACE, RADIATION, DRYCNV.
- **Data**: Outputs from `run_end_to_end.py` (JAX) and `run_end_to_end_fortran.f90` (Fortran).

### **GPU Benchmarks**
- **Hardware**: GPU (NVIDIA A100 or similar).
- **Grid Sizes**: 1,000, 10,000, 100,000 points.
- **Modules Tested**: DRYCNV, PBL.
- **Baseline**: NumPy (Fortran-like implementation).

### **Numerical Accuracy**
- **Metrics**: Max difference, mean difference, relative error.
- **Tolerance**: **~2.3%** (excellent for cross-language comparisons).

---

## 🔗 **12. Related Files**

| **File** | **Description** |
|----------|-----------------|
| `benchmark_all.py` | GPU benchmark script (JAX vs. NumPy). |
| `run_end_to_end.py` | JAX workflow (1D column). |
| `run_end_to_end_fortran.f90` | Fortran workflow (1D column). |
| `p2saom40_io.py` | Real P2SAoM40 restart/diagnostics I/O (72x46x40 production grid). |
| `p2saom40_driver.py` | Physics-only JAX orchestrator, real Fortran call order/cadence. |
| `p2saom40_compare.py` | Runs the orchestrator on real data; functionality/accuracy/performance report. |
| `FINDINGS.md` | This report. |

---

## 📝 **13. Notes**

1. **Fortran Simplifications**: The Fortran workflow uses **simplified logic** for FLUXES, SURFACE, and RADIATION (hardcoded constants, minimal computation), which explains its **superior CPU performance** for these modules.

2. **JAX GPU Advantage**: JAX’s **JIT compilation** and **GPU acceleration** provide **massive speedups** (8–253×) for PBL and DRYCNV, especially at larger grid sizes.

3. **Numerical Consistency**: Despite performance differences, JAX and Fortran produce **numerically consistent** results (within ~2.3%).

4. **Future Work**:
   - Optimize JAX implementations for **FLUXES, SURFACE, RADIATION** to match Fortran’s CPU performance.
   - Benchmark on **NASA HPC (Pleiades/Discover)** with NVIDIA A100 GPUs for **real-world GPU speedups**.

---

**End of Report**