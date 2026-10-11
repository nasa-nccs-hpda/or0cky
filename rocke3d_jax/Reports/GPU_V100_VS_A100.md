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

## When to confirm a milestone on the A100
Confirm on the A100 (owner submits through Discover) when any of these holds:
1. **Before any number is reported against the 3.5 s target.** The challenge names the A100; V100 numbers are for iteration only.
2. **After a change that alters the compiled program's structure** (jit boundaries, loop restructuring, barriers, flags), because the XLA in jax 0.6.1 behaved differently from 0.5.3 before (the ocean NaN, the 2.6 s single-jit overhead): a win on the V100 may not appear on the A100, and a problem may appear only there.
3. **When a V100 result improves a stage by 20% or more** or the whole step by 10% or more (below that, the 9% card-to-card gap is of the same size).
4. **At the end of each work block** that changes the default code path on a GPU, together with the fidelity check (`gpu_state_diff.py` against the CPU references at steps 0 and 5).
Not needed for: diagnostics, flag sweeps that show no effect, CPU-side changes.
