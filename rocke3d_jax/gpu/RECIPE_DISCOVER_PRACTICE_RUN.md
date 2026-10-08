# Recipe: GPU practice run of the assembled step on Discover

Reconstructed from the owner's shell history of 2026-10-08 (job 58786705, one nov26 step, A100). Paths are the owner's.
Result of that run: 13.09 s steady step, 725.8 s cold (654 s compile), all flags 0; judged statistically against the CPU libm reference (see `DISCOVER_RUN.md`).
Labels to keep with any result: libm mode, radiation replayed, hybrid (not end-to-end JAX), GPU not bitwise with CPU/Fortran.

## 0. One-time setup (data)
```
export P=/discover/nobackup/gtamkin/dev/or0cky_port/rocke3d_jax      # the Discover checkout; set in every new shell
cd $P
# The staging area on panfs is visible on Discover as /explore/nobackup/people/gtamkin/dev/ilab-agentic-ai/gpu_data_stage/
cp /explore/nobackup/people/gtamkin/dev/ilab-agentic-ai/gpu_data_stage/rocke3d_coupled6b_data.tar .
md5sum rocke3d_coupled6b_data.tar     # compare with rocke3d_coupled6b_data.tar.md5 in the same directory (f607c3a5be059ad0e3b6a8f7b0076ae0)
tar -xf rocke3d_coupled6b_data.tar    # creates ff_data/ and modelE2_planet_2.0/ inside $P
```
(History used `/panfs/...` for this copy; `/explore/...` is the same directory as seen from Discover. `diff -r` of the two showed no differences.)

## 1. Get the code
```
cd $P && git pull origin full-fidelity-port      # redo this whenever the port changes; the first run failed because the
                                                 # FF_DATA fix (56f4fe9) was not yet pulled
```

## 2. Choose what to run
```
vi gpu/step_command.txt      # the command the job runs; shipped: gpu_step_run.py nov26 $OUT/step --steps 6 --save first-last
                             # practice run used --steps 1 (one step, plus one steady repeat of step 0)
```

## 3. Submit and watch
```
MODE=step sbatch --time=3:00:00 gpu/run_gpu_job.sbatch     # prints "Submitted batch job NNNN"
squeue -u $USER                                            # is it running?
tail -f gpu_runs/NNNN/*                                    # metadata.txt, stdout.txt, stderr.txt, time.txt (Ctrl-c to leave)
ls -alt gpu_runs/NNNN/step/                                # result.json and nov26_libm_step0.npz appear at the end
more gpu_runs/NNNN/step/result.json                        # timings, flags, labels
```
stdout is quiet for ~11 minutes during the cold compile; that is normal. "bind mounts" warnings in stderr are harmless.
First-run failure (job 58786408, FileNotFoundError ffo_geom.bin): modules ignored `FF_DATA`; fixed in 56f4fe9.

## 4. Compare with the CPU reference
```
mkdir -p $P/ref_npz
cp /explore/nobackup/people/gtamkin/dev/ilab-agentic-ai/gpu_data_stage/ref_* $P/ref_npz/      # ref_cpu_libm_nov26_step0.npz, _step5.npz, libimf step0, ref_npz.md5
cd $P/ref_npz && grep libm_nov26_step0 ref_npz.md5 && md5sum ref_cpu_libm_nov26_step0.npz     # the two checksums must match
module load anaconda && conda activate graphcast-env       # the login-node default python lacks numpy; this env has it
cd $P/fullfidelity
python gpu_step_run.py --compare $P/gpu_runs/NNNN/step/nov26_libm_step0.npz $P/ref_npz/ref_cpu_libm_nov26_step0.npz
```
Reading the output: categories A bitwise, B <=1e-12 of the field scale, C <=1e-6, D worse. The 1-ulp exp/pow difference between GPU and CPU
puts many cloud/threshold fields in D. Baseline for comparison (CPU libimf vs CPU libm, same tool): A 126 / B 170 / C 47 / D 215;
the GPU run gave 126 / 165 / 45 / 222, dynamics max_rel 2.4e-11.

## 5. Next practice run (multi-step)
Edit `gpu/step_command.txt` to `--steps 6`, resubmit as in step 3, then compare step 5 with `ref_cpu_libm_nov26_step5.npz`
(the saved file for step 5 is the last of `--save first-last`).

## 6. What to send back to the project session
`stdout.txt` (tail), `step/result.json`, `metadata.txt`, last 30 lines of `stderr.txt`, and the `--compare` output.
