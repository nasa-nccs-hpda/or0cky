Zonal-mean per-level check of JAX minus Fortran against the five one-ulp members (2026-10-20). DAY-1 ANALYSIS ONLY (steps 0-53, nov26 day 1).
It is not a resolution of the strict fidelity failure and says nothing about longer runs.

Why: an SME review comment (2026-10-20): "wave like" pattern above layer ~25 in the zonal-mean JAX-Fortran U and T; hard to say after one day, watch how it evolves.

Script:  fullfidelity/zonal_vertical_check.py
Run:     gpu_runs/variants_20261018_e (JAX outputs fid_<kind>_jax/ours_d193; Fortran control ffa_step_<it>_e.bin and members ffpt_p1..p5_<it>.bin under ff_data/nov26_day2)
Command: PYTHONNOUSERSITE=1 FF_DATA=<ff_data> python fullfidelity/zonal_vertical_check.py --ours <run>/fid_<kind>_jax/ours_d193 --out <outdir>   (rocke3d-jax environment)
Steps:   1, 3, 11, 23, 35, 53; fields U and T; levels 1-40 (1-based, model ordering)

Definition: zonal mean over longitude (U and V skip the polar row j=0), D_jax(l,j) = zm(JAX) - zm(Fortran control), D_m(l,j) = zm(member m) - zm(Fortran control).
Per level: rms over latitude (not area-weighted); ratio = rms(D_jax) / max over the five members of rms(D_m). This is an inspection tool, not the acceptance score: no threshold from the acceptance rule is applied (2 is only a reference line).

Results (the numbers are in zonal_vertical_<kind>.txt, one line per field and step):
  Fast, upper levels 26-40, median ratio: T 3.31 (step 1), 1.90 (3), 0.86 (11), 0.85 (23), 0.86 (35), 0.80 (53); U 1.64, 2.23, 1.03, 0.89, 0.98, 0.83.
  Fast, maximum ratio in levels 26-40: T 6.43 at level 38 (step 1); U 3.46 at level 26 (step 3). From step 11 on the maximum is 0.97-1.30 and no level is beyond 2.
  Release: the same pattern (T step 1 max 6.21 at level 40; step 53 medians U 0.73, T 0.93).
  Vertical second difference above level 25, JAX / member p1: step 1 about 60-80x, step 3 about 2-5x (release U 1.9), steps 23-53 0.83-1.21.
Reading: from step 11 the upper-level differences are within the members' own differences, and one member (p1) shows the same structure in the step-53 U map.
        At steps 1-3 upper-level T (and some U) is beyond the members; that is the same early-step excess behind the failed steps 0-2 of the approved rule. Its cause is not identified.
Not covered: day 2 and later (the member data allow steps 54-107; not run), later-day evolution, other fields, area weighting. Only the fast step-53 U map was viewed as an image.

DAY 2 ADDENDUM (2026-10-20, 108-step states of the matched timing pairs, run variants_20261018_e/pair_<kind>_jax/ours_d193; steps 53, 54, 59, 71, 89, 107; members p1-p5 continue the day-1 curves, they are not re-perturbed at step 54)
Command: as above with --ours <run>/pair_<kind>_jax/ours_d193 --steps 53,54,59,71,89,107   Results: zonal_vertical_day2_<kind>.txt, zonal_profile_{U,T}_day2_<kind>.png
  Upper levels 26-40, median ratio, fast: U 0.81, 0.94, 0.92, 0.84, 0.72, 0.69; T 0.86, 1.06, 0.90, 0.86, 0.76, 0.83 (steps 53, 54, 59, 71, 89, 107). Maximum in 26-40: 1.02-1.26 (fast), 0.78-1.26 (release); no level beyond 2 at any of these steps in either configuration.
  Vertical second difference above level 25, JAX / member p1: fast U 1.22 at step 53 falling to 0.77 at step 107, T 1.39 falling to 0.92; release 0.56-0.90 throughout.
  Reading: no growing upper-level offset or vertical-structure difference over day 2 at these six steps; JAX stays inside the members. Only six steps were sampled, only U and T, rms over latitude of the zonal mean; the 30-day shadow run is invalid after step 1288 and nothing past step 107 was checked. Plots of the day-2 maps were not viewed as images.
  (The step-53 values here come from the pair run, not the fidelity run of the day-1 analysis; the runs are not bitwise repeatable, so they differ slightly.)
