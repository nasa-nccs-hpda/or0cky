#!/bin/bash
# Sharded full regression: same coverage as run_all_tests.sh, run as JOBS (default 4) concurrent pytest processes.
#   - one job per test file (largest files first, a dynamic queue, so the shards balance themselves);
#   - the three XLA-flag-sensitive JAX files get RUN_XLA_FLAG_TESTS=1 and their own fresh process (see tests/conftest.py);
#   - the radiation-server files (they start model processes and write run directories) run together in ONE serial job.
# Every job is limited to one thread (OMP/OPENBLAS/MKL) so JOBS jobs use about JOBS cores. Optionally pin with CPUS=8-11 (taskset).
# Run from fullfidelity/. Usage: [JOBS=4] [CPUS=8-11] [LOGDIR=dir] [ONLY=regex] ./run_all_tests_sharded.sh
# Exit code 0 only if every job passed. Totals are printed at the end; compare with the serial run (expect the same passed/skipped counts).
PY=${PY:-/home/gtamkin/.conda/envs/graphcast-env/bin/python}
JOBS=${JOBS:-4}
LOGDIR=${LOGDIR:-$(mktemp -d)}
mkdir -p "$LOGDIR"
export PY LOGDIR
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PIN=""
[ -n "$CPUS" ] && PIN="taskset -c $CPUS"
export PIN

XLA="test_dyn_jax test_dyn_jax2 test_clouds_jax"
SERIAL_GROUP=$(ls tests/test_radiation_server*.py tests/test_atm_day_free_rad*.py 2>/dev/null | tr '\n' ' ' | sed 's/ *$//')

one() {   # one <name> <env-assign|-> <files...>
  name=$1; envs=$2; shift 2
  if [ "$envs" = "-" ]; then
    $PIN $PY -m pytest "$@" -q -p no:cacheprovider > "$LOGDIR/$name.log" 2>&1
  else
    env $envs $PIN $PY -m pytest "$@" -q -p no:cacheprovider > "$LOGDIR/$name.log" 2>&1
  fi
  echo "$name rc=$?" >> "$LOGDIR/rc.txt"
}
export -f one

{
  for f in $XLA; do echo "xla_$f RUN_XLA_FLAG_TESTS=1 tests/$f.py"; done
  echo "radiation_serial_group - $SERIAL_GROUP"
  # remaining files, biggest first (file size as a cost proxy)
  for f in $(ls -S tests/test_*.py); do
    b=$(basename "$f" .py)
    case " $XLA " in *" $b "*) continue;; esac
    case " $SERIAL_GROUP " in *" $f "*) continue;; esac
    echo "$b - $f"
  done
} > "$LOGDIR/jobs.txt"

# optional subset for checking the runner itself: ONLY="regex" keeps the job lines that match
[ -n "$ONLY" ] && { grep -E "$ONLY" "$LOGDIR/jobs.txt" > "$LOGDIR/jobs.sel"; mv "$LOGDIR/jobs.sel" "$LOGDIR/jobs.txt"; }
rm -f "$LOGDIR/rc.txt"
t0=$(date +%s)
# shellcheck disable=SC2016
xargs -P "$JOBS" -L 1 bash -c 'one $0 $1 "${@:2}"' < "$LOGDIR/jobs.txt"
t1=$(date +%s)

# aggregate pytest summary lines
$PY - "$LOGDIR" <<'E'
import re, sys, glob, os
d = sys.argv[1]
tot = dict(passed=0, failed=0, skipped=0, xfailed=0, xpassed=0, error=0, errors=0)
bad = []
for f in sorted(glob.glob(d + "/*.log")):
    txt = open(f, errors="replace").read().strip().splitlines()
    last = [l for l in txt if re.search(r"\b(passed|failed|error|errors|skipped|xfailed|xpassed|no tests ran)\b", l) and re.search(r"in [\d.]+s", l)]
    if not last:
        bad.append((os.path.basename(f), "no summary line")); continue
    for n, k in re.findall(r"(\d+) (passed|failed|skipped|xfailed|xpassed|errors?)", last[-1]):
        tot[k] += int(n)
    if re.search(r"\d+ (failed|errors?)", last[-1]):
        bad.append((os.path.basename(f), last[-1].strip()))
print("TOTALS:", {k: v for k, v in tot.items() if v})
print("JOBS WITH FAILURES/NO SUMMARY:", bad if bad else "none")
sys.exit(1 if bad else 0)
E
rc=$?
echo "wall time $((t1 - t0)) s with JOBS=$JOBS; logs in $LOGDIR"
exit $rc
