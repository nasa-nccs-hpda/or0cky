Where does the early-step (steps 0-3) JAX-minus-Fortran excess arise? (2026-10-20; day 1 of nov26_day2; release configuration; one run)

Scripts: fullfidelity/stage_localise.py (stage-by-stage), fullfidelity/clouds_condse_compare.py (existing; stand-alone CONDSE against the recorded entry/exit).
Run:     gpu_runs/stage4 (not tracked): job 37944227, V100, release flags, --rad native --surf-fields computed, GPU_DAY_DRIVER=d212_day, GPU_DAYDIR=nov26_day2,
         with --c1dir (full per-stage arrays of every step); the run did 54 steps although --nsteps 4 was given; only steps 0-3 are used here.
Command: FF_DATA=<ff_data> python fullfidelity/stage_localise.py --c1dir <run>/rel_c1 --out stage_localise_release.txt
Stage map (found by matching step 0): dyn = state entering CONDSE (ffd pre_condse), condse = post_condse, radia = post_radia, surface = post_surface, dissip = ffa_step_d, filter = ffa_step_e.

Findings
1. At step 0 the JAX state entering CONDSE equals the recorded Fortran state to rounding (T rms 5.7e-14, QCL exactly 0 in the cell below). After CONDSE the T rms difference is 1.2e-4,
   the same size as the one-ulp members' spread at the end of step 0 (1.0e-4); RADIA, SURFACE, dissipation and the filter change it little. At steps 1-3 the rms difference grows
   inside CONDSE (T 1.1e-4 -> 4.3e-4 at step 1, QCL 2.5e-7 -> 3.2e-6) and little after it (stage_localise_release.txt).
2. The large step 1-2 excess sits in a few cells. Example, step 0, cell (i=3, j=19, level 15, 0-based): QCL in = 0 on both sides; out Fortran 9.7e-7, JAX 1.7e-5 (members agree with Fortran to 6e-9).
   Step 1, same column: input QCL differs (level 15: Fortran 9.7e-7, JAX 1.7e-5), output has the cloud water on different levels (level 14 JAX 6.7e-4 vs 8.7e-5, level 15 JAX 4.6e-5 vs 9.2e-4), T at level 14 differs by -0.105.
   This is a discrete (threshold) difference, not a smooth drift.
3. Stand-alone CONDSE on the recorded step-0 entry (clouds_condse_compare.py --dates nov26 --steps 0, CPU): with NumPy/glibc libm 1171 of 3170 columns inexact and 151 flipped beyond the bound (42 failing fields);
   with --imf (Intel libimf, the reference build's library) 6 of 3170 columns inexact and 6 flipped (1 failing field). (condse_standalone_libm_step0.txt, condse_standalone_libimf_step0.txt)

Reading: the excess is created in CONDSE (the cloud scheme), through threshold decisions that depend on the last bits of exp/pow. The Fortran reference used libimf; the GPU run uses the GPU's own libm and the CPU test without libimf
is a different libm again. The evidence is consistent with the libm/libimf explanation already noted in the QA handoff (section 3 item 8), and with that item still being the open test. It does not prove it for the GPU day.

Not shown: that a CPU libimf day removes the steps 1-3 excess (cpu_day_run_imf.py is still dry-run only: that is the end-to-end test). Fast configuration not run (release only). P at step 0 (24x) was not localised by stage (the stage dumps hold no P);
it is tiny in absolute terms (rms 7e-7 against members 3e-8, a few cells near i=18-20, j=31). Single run; runs are not bitwise repeatable.
