# rocke3d_jax (minimal branch)

JAX port of ROCKE-3D (ModelE) used to run and score the nov26 model day against recorded Fortran runs. This branch keeps only the source, tests and recipes needed for the release and fast configurations; everything else is in the tag `archive/full-fidelity-port-20261016`.

* Status, gates and open gaps: `STATUS_20261016.md`. Read it before citing any result.
* Environment and the variables the code requires: `fullfidelity/qa_20261016/ENVIRONMENT.md`.
* Run a day: `fullfidelity/gpu_day_run.py` (GPU) or `fullfidelity/cpu_day_run_imf.py` (CPU, libimf); recipes in `gpu/` and `Reports/SPEED_RUN_RECIPE_20261013.md`, `Reports/RELEASE_NATIVE_RADIATION_20261012.md`.
* Score a day: `fullfidelity/d207_score.py`. Tests: `fullfidelity/run_all_tests_sharded.sh` (and `run_all_tests_flags_on.sh`).
* Criteria: `Reports/ACCEPTANCE_CRITERIA.md` (the 2026-10-16 amended wording is not owner-reviewed).

Owner: G. Tamkin. Data (`ff_data/`, the ModelE tree) is not in the repository.
