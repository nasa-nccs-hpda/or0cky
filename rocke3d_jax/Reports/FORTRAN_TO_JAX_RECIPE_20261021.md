# Recipe and plan: porting a Fortran application to JAX with verified fidelity (2026-10-21)

Owner: G. Tamkin. Written by Claude Code session rocke3d-jax-8c for a new session that repeats this exercise on a different Fortran application. Status: DRAFT, not reviewed by the owner. Project-local (see `projects/imvi/AGENTS.md`); once the owner has used it on a second application and it holds, propose promotion to `skills/` or `knowledge/`.
Sources: `Reports/LESSONS_LEARNED.md`, `PROJECT_CHRONOLOGY.md`, `ACCEPTANCE_CRITERIA.md`, `RECONCILIATION_PLAN.md`, `REPEATABILITY_AND_FLAGS_REGRESSION_20261014.md`, `SPEED_RUN_RECIPE_20261013.md`, `QA_HANDOFF_20261016.md`, `NEW_BRANCH_BRIEF_20261016.md`, and this session's own experience. Where a claim is mine and not in those files it is marked (opinion).
Limits: the lessons come from ONE application (ROCKE-3D: a chaotic climate model, large, with a third-party radiation library). Anything specific to chaos, libimf or XLA flags may not apply to the new application; section 1 says how to find out in the first days. Review by: after the new session's first milestone.

## 0. How this project spent its time (so you know what to avoid)

From `PROJECT_CHRONOLOGY.md` (2026-08-28 to 2026-10-08, 273 commits, then speed and release work to 2026-10-16):

| Period | What happened | Cost |
|---|---|---|
| 08-28 to 09-23 | "Track A": reduced-scope ports and GPU kernels. Track A itself recommended against a full port; the owner reversed that, so the work was superseded. | About four weeks, little reused |
| 09-24 | Fortran oracle recovered; noise floor measured. | One day. This was the most valuable day. |
| 09-24 to 10-06 | Component ports validated against dumps from the real model, one at a time (ledger D4 to D145). | About 12 days; worked |
| 10-07 | Independent review found the fidelity work was ahead of the JAX deliverable: coupled results were Python/NumPy orchestration, not JAX-driven. Owner then fixed the direction and approved acceptance criteria before any coupled step was compared. | Rework |
| 10-12 | Radiation ("never port") was ported in a spike after all (SOCRATES). | The deferral held the project's central question open for a week |
| 10-13 | Speed work (1.71 to 0.40 s/step) started only after correctness. | Fine, but the end-to-end shape came late |
| 10-16 | QA found the package depended on the author's paths, personal conda environments, and an unrepeatable (not bitwise) run. | Cleanup at the end |

## 1. Recipe (in order; the numbers are the lessons)

### Step 1. Fix goal, scope, and acceptance in writing BEFORE porting (day 1)
* Write `GOAL.md`: what "done" means (end-to-end JAX-driven, not a Python driver calling JAX kernels), the first target configuration (one rundeck/case), and what is explicitly out of scope. Get the owner's signature on it.
* Write `ACCEPTANCE_CRITERIA.md` first, as this project did on 10-07, and apply it unchanged. Reuse the structure: (a) a "JAX-driven" definition (state lives on device; non-JAX stages listed by name with their time share; host-device transfers counted); (b) two comparisons reported separately, C1 port consistency (JAX vs your own NumPy/reference port) and C2 fidelity (JAX vs the real Fortran); (c) per-field categories A bitwise, B at most 1e-12 of scale, C at most 1e-6, D worse; (d) a statistical rule for multi-step runs.
* Decide the reduced-scope question now: do not start with a "reduced" port as a stepping stone unless the owner wants it as a deliverable. Track A was this project's largest sunk cost. (opinion: if the owner wants early GPU numbers, do a profile of the Fortran first and port the top hotspot with its real inputs, as a component of the full port, not a separate model.)
* List forbidden/untouchable code (this project: a third-party radiation library must never be modified) and write what is allowed instead (call as-is through a server/callback, noted in every result). Also decide, with a date to revisit, whether such a component will be ported; deferring it silently cost a week here.

### Step 2. Recover a trustworthy Fortran oracle (days 1-2)
* Rebuild the Fortran from source, record compiler, flags, math library and MPI/thread settings. Re-run a short case and reproduce the stored restart BITWISE. If it does not reproduce, stop and fix this first.
* Keep a read-only, checksummed copy of every starting state and reference output. Tests must write elsewhere. Lesson: shared workspaces got their start files overwritten more than once, costing up to 10 minutes of regeneration each time and, worse, silent wrong tests.
* Measure the noise floor: run five or more members perturbed by one ulp and see how fast they diverge (here: 1-3% of field variability pointwise in 5 days). This decides whether acceptance beyond one step can be pointwise or must be statistical, and it gives the "reference spread" your scorer needs. If the application is not chaotic, skip the statistical rule and demand tighter agreement.
* Find which math library the Fortran uses (here Intel libimf). Bitwise agreement needed libimf `pow`/`exp`; with it, 18 steps were bitwise; without it, 3-5% of cloud columns flipped at thresholds. Decide early whether bitwise is a goal or ratios against the member spread are; on GPU, bitwise to the CPU build is not achievable (device exp/pow differ by one ulp in 6-12% of values here).

### Step 3. Instrument the Fortran: dump hooks (days 2-4)
* Add dump hooks at module/step boundaries that write inputs and outputs of each routine for real calls. This is the method that found 3 real porting bugs in the first group of ports "by comparison, not by reading" and every later bug. Check that dumps are really from the run you think (see traps below).
* Check the call counts and the actual output. "Ran without error" proved nothing several times.
* Build the call graph and find DEAD CODE before estimating: here about half of one subsystem's legacy driver never executed, and the size estimate was wrong until the code was read fully. Rule: read code fully before giving a time estimate; count only code that executes in the target configuration (coverage run with gcov or equivalent).

### Step 4. Choose the port architecture once (day 4-5)
* Decide: one `jit` per stage or fewer; arrays on device across stages; batched columns (vmap) rather than per-cell loops (the cloud code went from about 111 s to about 5 s per step by batching all columns, a 22x gain on day one of that work); where unavoidable host callbacks live (e.g. libimf). Write down the count of jit boundaries and transfers as a measured number from the start.
* Pin the environment now: exact Python, JAX, jaxlib, NumPy, CUDA plugin versions, XLA flags, no persistent compile cache for validated runs, a lockfile or exported env, and NO hard-coded paths (home directory, personal conda env, scratch). Three of this project's late QA items were exactly this (see section 4). This session also lost a launch to it: the CPU XLA flag `--xla_cpu_max_isa` is unknown to JAX 0.4.30 but accepted by 0.5.3, and the docs said the site python was enough.
* Make the "which file actually runs" check a script (see traps).

### Step 5. Port bottom-up in validated units, in the order the Fortran executes (weeks 1-3)
For each routine or group:
1. Port to NumPy-faithful form first if the arithmetic must match bit for bit; otherwise straight to JAX x64.
2. Validate against the dumps on all real calls (not a sample): this project reports e.g. "bitwise on all 152,022 real calls" for a vertical-mixing kernel. Record category A/B/C/D per field.
3. Add the test to a regression runner that isolates flag-sensitive tests in separate processes.
4. Write a ledger entry (`D<n>`): what was ported, evidence, source dump, owner, date, limits, open question.
* Order by data dependence and by risk: small well-understood routines first; big complex ones (here an ocean mixing scheme) scheduled with explicit hours rather than rushed or approximated. The owner twice rejected a "faster approximation" of the ocean model; ask the same question of the owner up front for your application.
* Look for small transcription errors with loose tolerances: here a wrong rotation constant and two land-model errors (precipitation conditioning, a dropped irrigation term) hid behind loose tolerances and were found only by the stage-level verdict. Use the strict tolerance from the beginning; widen only with a ledger entry.
* One writer per working tree. Commit/test/push in large batches; do not re-run the full suite per change (owner rule). Approvals for commit, push, delete, and tar are per action.

### Step 6. Chain early, open loop first (week 2 onward, not week 5)
* As soon as 3 or 4 units exist, chain them in the real order for one step from the real starting state, with remaining stages taking RECORDED inputs, and score against the real next-step dump. Keep a list of every recorded input (file, size) and shrink it deliberately; the acceptance file requires it.
* Then run a window (here one model day, 54 steps) replayed open loop against the real run and the members; then free-running with the real hybrid component; then compute the surface/ocean/etc. yourself, one recorded input at a time.
* Do this earlier than this project did. Fidelity of parts was ahead of the end-to-end JAX deliverable by the time of the independent review (10-07).

### Step 7. Hybrid components and the third-party library
* Wrap the untouchable component in a persistent server (here: the real RADIA, bitwise oracle, but 30 to 40 s per call, mostly start-up) so that free-running tests are possible while the port is incomplete. Count calls, bytes, time, and sync points in every result.
* Decide on the port of that component after the first free-running day, with the owner. A spike on a separate branch and worktree with a written brief kept that decision cheap here.

### Step 8. Speed, after correctness, as flags that default off
* Freeze one benchmark first (same exclusive node, Fortran then JAX, strictly sequential, timed with the same settings, no compile cache, several runs: node-to-node variation was 2.13 to 2.21 s for the same code). Only then optimize.
* Each optimization is a default-off flag with a test: bitwise equal, or an explicit tolerance. Record tried-and-rejected ones (here five were rejected, one needed 14 minutes of compile per phase). A fast configuration is allowed to be not bitwise with the release configuration, but the difference must be measured (here the day-1 worst ratio moved from 2.37 to 2.47 and the owner accepted that).
* Cold-start compile (about 15 minutes here) is a cost to report in the benchmark.

### Step 9. Package for review as you go, not at the end
* One status document (dated, with owners and the open replication gaps), one README, a manifest with checksums, a scoring tool that reads only the package, a single command that reproduces the headline table from an empty environment.
* Hand-built deliverable archives must be rebuilt by a script and compared to the directory. The review tar here went stale after a README edit.

## 2. Traps that cost this project time (checklist before every test)

1. **Stale counters / stale dumps.** An internal call counter, restored from a restart, made old work look recent. Always compare actual output fields and timestamps.
2. **Near-identical file names.** The running binary and a freshly rebuilt one were different files; tests used old software for a while. Make a script print the path and hash of the artifact under test and of the reference.
3. **A clean exit that did nothing.** Compare outputs every time.
4. **Overwritten start states.** Read-only copies; tests write to their own directory.
5. **Compiler-flag and process sensitivity.** Some JAX tests needed particular XLA flags and could not share a process with others; the compile cache must be off for validated runs; the CPU flags must be exported before importing JAX when scoring. Put flags in one module imported first, and test that it is set.
6. **The math library.** Know it before you start (section 1 step 2).
7. **Chaos amplifies one-ulp differences.** Same-code repeats of the native-radiation JAX day were NOT bitwise; report ratios against the member spread, never equality.
8. **Mixing a Python driver with JAX kernels and calling it a JAX port.** Use the acceptance definition from the start.
9. **Dead code.** Coverage-check before porting.
10. **Hard-coded personal paths and environments.** Keep a lint (a grep for `/home/`, user names, `/discover`, `/explore`, `/panfs`) in the regression runner from week 1.
11. **Estimating before reading.** Read it, then estimate. Early estimates here were "30 to 65 hours" for near-term items while the full one-month target was 100 to 170 more hours.
12. **Interpreting a ratio you do not understand.** The Fortran leave-one-out RMSE to spread ratio is an identity (sqrt(1.2) for six runs) and says nothing; the JAX distance from the six-run mean divided by the spread is informative. Define each statistic in the scorer's docstring and in the report.
13. **Wrong wording in results.** Claims were corrected several times in QA (e.g. "QCI returns inside the band" was too strong). Draft result sentences from the scorer's own output, quote the numbers.
14. **Shared compute.** Do not time on a shared node; request exclusive nodes. Note which cluster login nodes have no scheduler (here forest204 has no SLURM client; use `ssh gpulogin1 sbatch`) and which file systems are not reachable.

## 3. Detailed plan for the new session (FORTRAN to JAX), with exit criteria

Adjust the time boxes after Phase 0; those below are placeholders (opinion), not estimates from the old project.

| Phase | Output | Exit criterion (owner signs) |
|---|---|---|
| P0. Charter | `GOAL.md`, `ACCEPTANCE_CRITERIA.md`, scope list, untouchable list, environment spec, cluster notes | Owner signature on criteria BEFORE any comparison |
| P1. Oracle | Rebuilt Fortran, bitwise restart reproduction, protected start states, five or more one-ulp members, math-library note, noise-floor table | Restart bitwise; spread table; chaos yes/no decided |
| P2. Map | Call graph, coverage run, dead-code list, dump-hook plan, size table counted from executed code | Owner agrees the port list and order |
| P3. Hooks and dumps | Dump hooks; a replay harness for any routine; dump manifest with checksums | Dumps reproduce under the oracle |
| P4. Units | Units ported bottom-up with ledger entries; regression runner; lint for paths | Per unit: A/B and no unexplained C/D; all real calls |
| P5. Chain | One chained step from the real start state, recorded inputs listed, verdict per acceptance categories | Verdict (met / partly / not) and the list of recorded inputs |
| P6. Window | Open-loop day against the members, then free-running day with the hybrid component | Day rule from the criteria |
| P7. Compute the rest | Replace recorded inputs one by one | Recorded-input list empty or justified |
| P8. Hybrid component | Decision and spike on the third-party piece | Written decision with owner |
| P9. Speed | Frozen benchmark, flags default off, tried/rejected list | Matched benchmark job IDs recorded |
| P10. Package | Status doc, README, manifest, scoring tool, parallel runner, environment from empty, QA handoff note | Reproduces headline table from an empty environment |

Standing rules for every phase:
* Ledger entry per result (what, evidence, owner, source, date, limits, review by).
* Write down what went wrong the first time it happens, with the avoidance step; this paid back repeatedly here.
* Report to the owner at each exit criterion with the verdict wording taken from the scorer, not paraphrased.
* Never add credentials, protected data, or production endpoints; keep personal paths out of kept files.

## 4. Reuse from this repository (copy and adapt, do not re-derive)

* Scorer and definitions: `fullfidelity/d207_score.py`, `multiday_score.py`; categories in `ACCEPTANCE_CRITERIA.md` section 3.
* QA tools: `fullfidelity/qa_20261016/` (spread vs RMSE, localization of worst cells, follow columns, rebuild tar); `ENVIRONMENT.md` there (note: it says the site python is enough for the scoring/plot scripts; the CPU model runs need JAX 0.5.3, i.e. graphcast-env).
* GPU recipes: `gpu/v100_run.sh`, `gpu/v100_step6_graphcast.sbatch`, `gpu/benchmark_native_v100.sh`; planned parallel runner in `NEW_BRANCH_BRIEF_20261016.md` section 2 (SLURM job per variant, exclusive for timing, dependent scoring job).
* Method documents: `RECONCILIATION_PLAN.md`, `STANDALONE_INPUT_AUDIT.md` (the recorded-input audit approach), `SOCRATES_SPIKE_BRIEF.md` (how to brief a spike).

## 5. Working with the previous session (this one)

* This session is `rocke3d-jax-8c` (other sessions can message it with that name; the harness lists it as `rocke3d-jax-8c [dcb20e]`). It is currently running a CPU libimf day (scoring pending) and may be busy; it replies when it next gets a turn. If messaging is not available in the new session, the owner relays.
* Good questions to send: why a decision in `RECONCILIATION_PLAN.md` or the ledger was taken; what a flag or script does; the rationale of an acceptance rule; the status of an open item in the QA note. Send the file and line you are looking at.
* This session cannot answer questions about the new application's code, cannot decide for the owner, and has not verified claims about the Discover copy or the SME's thread. Keep the SME and the in-house Climateer separate (QA note section 1).
* Do not ask it to commit or push. Those need the owner.

## 6. First actions for the new session

1. Read this file, `LESSONS_LEARNED.md`, `ACCEPTANCE_CRITERIA.md`, `QA_HANDOFF_20261016.md` section 1 and 6, and the repository `AGENTS.md`.
2. Ask the owner: the target application and case; what "done" is; any untouchable components; the compute (CPU, GPU, scheduler); whether bitwise is a goal; whether reduced scope is a deliverable.
3. Create `projects/<new-project>/` with `AGENTS.md`, `GOAL.md`, `ACCEPTANCE_CRITERIA.md` drafts; do P0 and P1 before any porting.
