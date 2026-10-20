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
