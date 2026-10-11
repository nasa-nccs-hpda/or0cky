# V100 (ADAPT, explore) against A100 (Discover): comparison table, kept up to date

Policy adopted 2026-10-10: iterate on the V100 (jobs submitted from the dev host, `gpu/v100_run.sh`), confirm milestones on the A100 through Discover (the owner submits), record both here. Same code, replay mode, libm mode, nov26, cloud-base inputs captured at step 1. One run each unless stated; shared nodes. A cell is empty when it was not measured.

| quantity | A100 (Discover, jax 0.6.1, container) | V100 (ADAPT, jax 0.5.3, graphcast-env) | V100 / A100 |
|---|---|---|---|
| whole step, cold step 0 (s) | 657.9 (job 58815361; compile 588.5) | 615.7 (job 37941449; compile 541.6) | 0.94 |
| whole step, steady steps 2-5 (s) | 8.21-8.43, mean 8.3 (58815361) | 8.91-9.29, mean 9.0 (37941449) | 1.09 |
| whole step finite, 6 steps | yes | yes | |
| cloud-base stage alone (condse_mstcnv, D189 loop), compile (s) | 143 (58815669) | 153 (37941450) | 1.07 |
| cloud-base stage alone, steady (s) | 5.63 (58815669) | 6.34 (37941450) | 1.13 |
| cloud-base stage, digest of the result | 2c3f73e875e993ed801f7cdb | 2c3f73e875e993ed801f7cdb | equal (bitwise equal across the two cards and jax versions) |
| cloud-base stage, round-based loop (D214 experiment), steady (s) | | 8.84 (37941451), digest equal to the sequential one | slower than sequential 6.34 |
| one event block at LMIN 3, 32 / 128 / 512 / 1024 columns (s) | 0.529 / 0.555 / 0.607 / 0.659 (58815692) | | |
| XLA command-buffer / graph flags on the cloud-base stage (s) | 5.63 / 5.63 / 5.57 (58815669) | | |

## Progress log of the optimization work (V100, replay mode, steady steps 2-5)
| date | change | whole step (s) | note |
|---|---|---|---|
| 2026-10-10 | baseline (ocean split in, barrier in) | 9.0 | job 37941449; A100 8.3 (58815361) |
| 2026-10-11 | QUS subsidence ADV1D (with limiter) on the device (`clouds_adv1d_dev.py`, default on a non-CPU backend) | **3.98** | job 37941464; cloud-base stage alone 6.34 -> 0.89 s; finite at 6 steps; scorer ratio at step 5 in the same range as before (T 0.87, U 0.59, V 0.79, Q 1.07, P 0.63, QCL 0.24, QCI 0.27); bitwise equal to the NumPy callback on the CPU (80 of 80 stage output leaves), 78 of 80 on the V100 |
| 2026-10-11 | OADVT2 X pre-pass on the device (`oadvt_xpre_dev.py`, default on a non-CPU backend) | **2.14** | job 37941485; ocean dynamics stage alone 1.5 -> 0.039 s; results identical to the previous run in every printed digit |
| 2026-10-11 | pole columns as host steps between jits (`ROCKE_CONDSE_POLE_HOST`, default on a non-CPU backend) | **1.70** | job 37941486; bitwise equal to the in-jit callback version (561/561 and 572/572 arrays category A at steps 0 and 5, V100; also on the CPU for 2 steps) |
Stage shares now (job 37941486, V100, 1.70 s): condse_mstcnv 0.94 (56%), dyn 0.20, template_build 0.20, condse_post 0.075, condse_setup 0.067, post_a 0.060, ocean_b 0.045, surface_tiles_land 0.037, radia 0.033. Round-based cloud-base loop re-tested without callbacks: 1.78 s vs 0.90 s sequential (identical digest): not used. Finding behind the three fixes: a host callback inside a jit program fires only after the whole device program has finished (first callback at 0.03 s in a short program, 1.0 s in a longer one); a callback alone costs 2 ms; cause not identified.
Remaining stage shares (job 37941464, V100): ocean_b 1.56 s, condse_mstcnv 0.94, condse_post 0.51, template_build 0.50, dyn 0.19, condse_setup 0.11, other about 0.2. A100 confirmation of this milestone is due (rule 2 and 3 below): the step changed by more than 10%.

| 2026-10-11 | cycle with the REAL Fortran radiation server (`gpu_server_step_run.py`), no prefetch | **3.40-3.57 per step** (cycle average, NRAD = 5) | jobs 37941496 (16 steps: held 1.90 s, radiation 10.25 s of which 8.2 s server -> 3.57), 37941497 (held 1.71, radiation 10.34 -> 3.44), 37941499 (held 1.71, radiation 10.18 -> 3.40): run-to-run noise about 0.1-0.2 s; hybrid (radiation computed, other inputs recorded) |
| 2026-10-11 | template prefetch (replay scaffolding hidden behind the previous step's device work; opt-in) | **3.33 per step** (cycle average) | job 37941500; end-state digest identical to the run without prefetch (f3ca5c8f083315b874cd); held steps 1.60 s |
Eager dispatches per step by stage (job 37941498): 770 in total, 294 of them in template_build, the rest 1-2 per stage.

## When to confirm a milestone on the A100
Confirm on the A100 (owner submits through Discover) when any of these holds:
1. **Before any number is reported against the 3.5 s target.** The challenge names the A100; V100 numbers are for iteration only.
2. **After a change that alters the compiled program's structure** (jit boundaries, loop restructuring, barriers, flags), because the XLA in jax 0.6.1 behaved differently from 0.5.3 before (the ocean NaN, the 2.6 s single-jit overhead): a win on the V100 may not appear on the A100, and a problem may appear only there.
3. **When a V100 result improves a stage by 20% or more** or the whole step by 10% or more (below that, the 9% card-to-card gap is of the same size).
4. **At the end of each work block** that changes the default code path on a GPU, together with the fidelity check (`gpu_state_diff.py` against the CPU references at steps 0 and 5).
Not needed for: diagnostics, flag sweeps that show no effect, CPU-side changes.
