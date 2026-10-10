# D215: matched wall-clock timing of the real Fortran ModelE for the nov26 54-step day

Date 2026-10-10. Owner: Glenn Tamkin (agent D215). Not committed. Review: when COST_TABLE_JAX_VS_FORTRAN.md is next updated.
Fortran tree read-only, nothing rebuilt, SOCRATES untouched. Results: `scoping/d215_results/`.

## 1. How the ensemble runs were launched (found)
D165 section 1 in `FULL_FIDELITY_DELTAS.md`; the script `ens_jan/one.sh` (session scratchpad, copy in `d215_results/one.sh` adapted) did:
run dir with `I P2SAoM40ln P2SAoM40uln runtime_opts` from `ModelE_Support/huge_space/P2SAoM40`, binary copied to `P2SAoM40.bin` and `P2SAoM40`, one edit to `I` line 109 (end date), restart copied to BOTH `fort.1.nc` and `fort.2.nc`, `sh P2SAoM40ln`, then
`OMP_NUM_THREADS=1 ./P2SAoM40 -i I` with all `FFD_*`/`FFPT_*` unset, wrapped in `time`. 5 concurrent single-thread runs, no taskset.
Executable of the ensemble (ctrl and members): the D149-D151 build (pristine physics + dump hooks + `ATM_DRV_pert.f.patch`; hooks write nothing with `FFD_*` unset), md5 866cfbe49e0728d688cd0edc78b72091, 15,221,920 bytes (the copy in `ens_jan/run_ctrl/`; the original `mE_day1/.../model/P2SAoM40.bin` has the same size). So the ensemble binary is "hooks compiled in, inactive", not a build without hooks. The binary is not MPI/OpenMP linked (ldd: no libgomp/libmpi; log says 1 MPI process), so no multi-core variant exists and none was run.
A truly hook-free binary exists as the production one `huge_space/P2SAoM40/P2SAoM40.bin` (md5 e51072f3f1948dbde516f7a70106b5ac, 15,057,624 bytes, dated Sep 18); I ran it as well.

## 2. This measurement
Rundeck: `huge_space/P2SAoM40/I` (as the ensemble) with line 109 changed to `YEARE=1950,MONTHE=11,DATEE=27,HOURE=3,     KDIAG=12*0,9,` (KDIAG kept as in the ensemble). Restart `ff_data/_pristine_restarts/fort1_nov26_itime33312.nc` (itime 33312, 26 Nov 1950 00:00). Run ends at itime 33366 = 54 steps (log: "Terminated normally (reached maximum time)", restart written at Year 1950 NOV 27 Hr 3).
Command per run (`d215_results/one.sh <tag> <binary> <core>`): `OMP_NUM_THREADS=1 taskset -c 7 ./P2SAoM40 -i I` inside bash `time`; the 6 runs were executed strictly one after another (`seq.sh`), all on core 7 (forest204, 12 vCPU Xeon E5-2697 v3 2.6 GHz).
Load: `mpstat` before the first run showed 94-99 percent idle on every core and no heavy process; `uptime` load average nevertheless read 3.7-9.4 during the runs (pre_*.txt), source not identified (no busy process visible in `ps`; may be I/O wait or other VM tenants). Treat the timings as measured on a node that was not provably quiet.

| run | binary | real | user | sys | main loop (model log) | real / 54 | main loop / 54 |
|---|---|---|---|---|---|---|---|
| ens1 | ensemble (hooks inactive) | 3m38.89s | 199.74 s | 14.98 s | 198.20 s | 4.05 s | 3.67 s |
| ens2 | ensemble | 3m21.84s | 184.79 s | 15.14 s | 182.93 s | 3.74 s | 3.39 s |
| ens3 | ensemble | 3m28.19s | 191.78 s | 14.23 s | 190.15 s | 3.86 s | 3.52 s |
| pris1 | production hook-free | 3m22.60s | 185.95 s | 14.75 s | 184.13 s | 3.75 s | 3.41 s |
| pris2 | production hook-free | 3m36.54s | 200.35 s | 13.93 s | 199.16 s | 4.01 s | 3.69 s |
| pris3 | production hook-free | 3m42.05s | 204.12 s | 15.27 s | 202.59 s | 4.11 s | 3.75 s |

Ensemble binary: real mean 209.6 s (201.8-218.9), 3.88 s per step of the whole process. Production binary: real mean 213.7 s (202.6-222.1), 3.96 s per step. Run-to-run spread about 8 percent, larger than the difference between the two binaries (not distinguishable). "Main loop" is the model's own `Time spent in the main loop` line; real minus main loop is about 17-20 s of start-up (restart read, initialisation), and the day includes the one daily update (about 1.1 s in the earlier real54 log).
Reference points: the earlier unpinned 54-step run in this session (`scratchpad/real54`, same restart, another binary, md5 a5179bbd...) reported a main loop of 170.05 s (3.15 s/step); the ensemble average was 3.07 s/step over 1,674 steps (85m42s real, 84m41s user). The matched day measured here is 15-25 percent slower per step than those: not explained (candidates, not tested: node load, start-up and the single daily update amortised over 54 instead of 1,674 steps, January vs November state).

## 3. End-state check
All six end restarts (`fort.2.nc`, itime = itimee = 33366) and the earlier real54 `fort.2.nc` were compared variable by variable with `cmp.py` (257 numeric variables, no non-finite value): all equal bit for bit to ens1 except the wall-clock scalar `cputime`. So the two binaries give the same end state, the runs are deterministic, and the state is finite and at the right day. Not done: comparison with the step-53 instrumented dump (`ffa_step_*.bin` binary records, not easy to line up with the restart's step-54 state) and no global-mean scalar was extracted.

## 4. Not measured
- Multi-core Fortran (binary is serial; no MPI/OpenMP build exists in the files I found).
- A provably quiet node (load average not explained, see above); no repeat at other times of day.
- Core 12-15 (node has only cores 0-11); only core 7 was used (a single core, the 6 runs were sequential).
- Other days, or a longer run to amortise start-up; instrumented (FFD) run speed; any hook-free rebuild.
Files: `D215_FORTRAN_TIMING_ENTRY.md`, `d215_results/{one.sh,seq.sh,cmp.py,state_compare.json,pre_*.txt,run_*.time,run_*.PRT}`. Scratch run dirs: session scratchpad `d215/run_*` (scratch, purge risk).

## Parent-session check (2026-10-10)
Read all six `run_*.time` and one PRT myself: the times in the report match (real 3m22s-3m42s, mean 212 s; the run ends at itime 33366 "Terminated normally"). The agent's bit-identity claim over 257 variables is read from `state_compare.json`, not re-run. Cost table updated (Reports/COST_TABLE_JAX_VS_FORTRAN.md). Not a provably quiet node.
