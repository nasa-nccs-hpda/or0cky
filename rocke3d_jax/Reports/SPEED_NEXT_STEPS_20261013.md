# Speed track: next steps for smaller gains (state locked 2026-10-13)

Owner: G. Tamkin. Written by the speed session. Project-local. Numbers are from `SPEED_STEP_TIME_LOG_20261012.md` (sections 8-13); items marked "estimate" are labelled arithmetic or judgement, not measurements.

## Locked state
Branch `speed/step-time`, tag `speed-step-time-20261013` (local, not pushed). Held step 0.385 s, mean 0.434 s per step (job 37941870); matched benchmark 0.524 s per step against Fortran 2.285 s (job 37941875). Owner decision 2026-10-13: the band shift (day-1 worst QCI step 3, 2.37 -> 2.47, same step and field) is acceptable for now. Run recipe: `SPEED_RUN_RECIPE_20261013.md`. Nothing is pushed to `full-fidelity-port`; the merge is the owner's / port-driver's decision.

## Where the held step still goes (timed, median s; job 37941867, 37941866)
surface_tiles_land 0.086 (GHY 0.030 per substep x2, PBL 0.0075), condse_mstcnv 0.071, dyn 0.065, ocean_b 0.048, condse_post 0.028, post_a 0.019, condse_setup 0.016, ocean_a 0.013, melt_si_dev 0.009; radiation step adds about 0.16.

## Candidates, largest first (each: what, why, expected gain, risk)
1. **GHY loop tail** (about 0.06 s per step in total for both substeps). 10 sub-iterations of the whole 753-lane body for a few slow cells (counts 1..10: 226/301/125/42/31/15/7/2/1/3); about 3 ms per iteration after `ROCKE_ENT_BATCH`. Options: merge the `_ci_cubic` rtsafe into the batched Ent call (1 loop fewer per iteration); run iterations > 4 only for the compacted active lanes (does not cut kernel count, so only useful together with a cheaper tail body); estimate 0.01-0.02 s. Risk: rounding-level (as ENT_BATCH).
2. **MSTCNV host stage** (0.071 s = d2h 0.018 + worker 0.036 + h2d 0.014). Derive byam / um / vm / sm / qm on the host from the arrays that are transferred anyway (exact mul/div only; not tvl, which has a mul-add); pinned buffers; start the worker's input copy per array as it arrives. Estimate 0.01-0.02 s. Risk low.
3. **dyn** (0.065 s): aflux 0.0115 (three 40-level scans), trop 0.009 (nested while loops per column, vmapped; can be rewritten as masked 40x40 candidate arrays with an explicit sequential running sum), aadvt 0.009, pgf 0.0067, advecm 0.006, iso 0.0045. Estimate 0.015-0.025 s for aflux + trop + advecm. Risk: moderate, each needs its recorded-call test.
4. **ocean_b `dynamics`** (0.032 s): 6 ODHORZ calls x (13-layer prefilter + 13-layer step loops, about 20 scatter-style `.at[].set` per layer), OFLUXV, two OADVT2 with the host X pre-pass callback. Estimate 0.01-0.015 s from vectorising the per-layer scatters; the callback could move to the numba worker. Risk moderate. `ROCKE_SCAN_UNROLL=16` made it slower (do not retry).
5. **Radiation step** (+0.16 s every 5th step = 0.03 s mean): 22,403 kernels, device busy 0.167 s; not profiled by source line. First step: map the top kernels to source (HLO dump as in section 2) and look for static scans / while loops as in dyn.
6. **Overlap**: south pole column already runs in a thread (`ROCKE_POLE_ASYNC`). The ocean half of step k (post_a .. post_b, about 0.08 s) is independent of `dyn` of step k+1 (melt_si_dev precedes condse, not dyn), so a two-thread pipeline could hide up to about 0.065 s; needs a lazily resolved surface state and a driver that dumps state one step late. Largest remaining item and the largest engineering risk (estimate 0.04-0.06 s).
7. **Cold start** (925 s): out of scope of this track; XLA compile, rule D175 (no persistent cache by design).

## Do not retry (measured, section 11-13)
`ROCKE_FUSED_DYN` (14 min compile per DIAGA phase), `ROCKE_SCAN_UNROLL` 16 / 80 (slower / out of memory), XLA command buffers incl. WHILE (full step slower), `ROCKE_ENT_SKIP`, `ROCKE_NEWTON_UNROLL`, `ROCKE_WHILE_UNROLL` (no gain).

## Before any merge
1. Run `fullfidelity/run_all_tests.sh` with the new flags on where the tests allow (the suite was run with flags off: 3329 passed, 1 known order-dependent failure that passes alone).
2. Decide the numba interpreter dependency: either an env with numba + NumPy 2.2 for graphcast-env, or keep the worker under DEV-Kernel (`ROCKE_MSTCNV_PY`); record the path in the run recipe of the target branch.
3. Time on exclusive nodes only (a shared node inflated steps 25-107 of job 37941849).
4. Make `ROCKE_SEQSUM_BLK`, `ROCKE_GLUE_JIT`, `ROCKE_QDYN_VMAP` defaults only if the owner wants them as the benchmark; they are bitwise-equal paths (seqsum, vmap) or selections only (glue).
