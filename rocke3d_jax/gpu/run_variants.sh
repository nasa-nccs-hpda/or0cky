#!/bin/bash
# Submit every variant of the evidence set as SLURM jobs, in parallel across nodes, and one dependent job that scores and writes one summary table.
#   pair_release / pair_fast : EXCLUSIVE V100 node each; Fortran x3 (one core) then JAX on the SAME node, strictly sequential (gpu/benchmark_native_v100.sh), 108 steps (the two-day window fixed in benchmark_native_v100.sh). Timing only.
#   fid_release / fid_fast   : V100, not exclusive; JAX 54 steps (one model day) for the fidelity score. Not timed.
#   tests_off / tests_on     : CPU (JAX_PLATFORMS=cpu: a job without --gres may still land on a GPU node, where the CUDA plugin fails to start); sharded regression with the speed flags off / the nine flags on.
#   summary                  : runs after ALL of the above end (afterany, so a failed variant is reported, not silently dropped); scores both fidelity runs against the five one-ulp
#                              members and writes SUMMARY.md (gpu/summarize_variants.py).
# Needs at least three GPU nodes at once (two exclusive) plus CPU slots. Repeats are not bitwise (native radiation): read ratios against the member spread, not equality.
# usage:  gpu/run_variants.sh [NAME]          DRY_RUN=1 writes the batch files and prints them, submits nothing.
# Required environment (see fullfidelity/qa_20261016/ENVIRONMENT.md):
#   FF_DATA, MODELE_PROD_INPUT, ROCKE_SOCRATES_DIR, FORTRAN_HUGE_SPACE, FORTRAN_RUNS, ROCKE_MSTCNV_PY (numba python for the fast configuration), CONDA_BIN (path of the conda executable)
# Optional: ROCKE_PROJ (default: this checkout), ROCKE_ENV (conda env, default graphcast-env), SBATCH_HOST (e.g. gpulogin1: submit over ssh when the local host has no SLURM client),
#   PARTITION (compute), GPU_CONSTRAINT (v100), NSTEPS_FID (54), ONLY="pair_release fid_fast ..." (submit a subset; the summary still runs), TIME_PAIR/TIME_FID/TIME_TESTS (HH:MM:SS).
set -u
P=${ROCKE_PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
for v in FF_DATA MODELE_PROD_INPUT ROCKE_SOCRATES_DIR FORTRAN_HUGE_SPACE FORTRAN_RUNS ROCKE_MSTCNV_PY CONDA_BIN; do
  [ -n "${!v:-}" ] || { echo "run_variants.sh: set $v (see fullfidelity/qa_20261016/ENVIRONMENT.md)" >&2; exit 2; }
done
# pre-flight: fail here, not 30 s into a GPU job (MODELE_PROD_INPUT must be the COMPLETE ModelE_Support/prod_input_files; a partial copy lacks S72x460098M.ext.nc)
for f in "$MODELE_PROD_INPUT/S72x460098M.ext.nc" "$FORTRAN_HUGE_SPACE/P2SAoM40.bin" "$FF_DATA/_pristine_restarts/fort1_nov26_itime33312.nc" "$ROCKE_SOCRATES_DIR" "$ROCKE_MSTCNV_PY" "$CONDA_BIN"; do
  [ -e "$f" ] || { echo "run_variants.sh: missing $f" >&2; exit 2; }
done
ENVN=${ROCKE_ENV:-graphcast-env}
PART=${PARTITION:-compute}
CONS=${GPU_CONSTRAINT:-v100}
NF=${NSTEPS_FID:-54}
NAME=${1:-variants_$(date +%Y%m%d_%H%M)}
REL=$NAME                                   # under gpu_runs/ (benchmark_native_v100.sh writes to $ROCKE_PROJ/gpu_runs/$JAXOUT)
RUN=$P/gpu_runs/$REL
JOBS=$RUN/jobs
mkdir -p "$JOBS"
ONLY=${ONLY:-pair_release pair_fast fid_release fid_fast tests_off tests_on}

# the release flags of the matched benchmark and of the review package (gpu_day_run.py is driven by these) and the nine speed flags of the fast configuration
RELEASE_FLAGS="ROCKE_TPL_OWN=1 ROCKE_OWN_GHY=1 ROCKE_IRRIG_OWN=1 ROCKE_PBL_CARRY=1 ROCKE_REC_LAZY=1 ROCKE_CSE_IN_OWN=1 ROCKE_CARRY_CSE_OUT=1 ROCKE_ENT_LIVE=1 ROCKE_RADCOLS=1 ROCKE_STAGE_ARGS=1"
SPEED="ROCKE_MSTCNV_HOST=1 ROCKE_MSTCNV_IMF=1 ROCKE_MSTCNV_HOST_IO=1 ROCKE_QDYN_VMAP=1 ROCKE_POLE_FAST=2 ROCKE_POLE_ASYNC=1 ROCKE_SEQSUM_BLK=32 ROCKE_GLUE_JIT=1 ROCKE_ENT_BATCH=1 ROCKE_MSTCNV_PY=$ROCKE_MSTCNV_PY"

submit() {   # submit <batchfile> [dependency-spec]  -> prints the job id
  local f=$1 dep=${2:-}
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY_$(basename "$f" .sbatch)"; return; fi
  local cmd="sbatch --parsable ${dep:+--dependency=$dep }$f" out
  # the login banner printed by ssh is not part of the answer: keep stderr apart and take the last all-digit line (sbatch --parsable may print "id" or "id;cluster")
  if [ -n "${SBATCH_HOST:-}" ]; then out=$(timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=5 "$SBATCH_HOST" "$cmd" 2>/dev/null); else out=$($cmd 2>/dev/null); fi
  out=$(printf '%s\n' "$out" | sed 's/;.*//' | grep -E '^[0-9]+$' | tail -1)
  [ -n "$out" ] || { echo "run_variants.sh: sbatch failed for $f (no job id returned)" >&2; return 1; }
  echo "$out"
}

header() {   # header <jobname> <time> <extra #SBATCH lines...>
  local nm=$1 tm=$2; shift 2
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=$nm
#SBATCH --partition=$PART
#SBATCH --time=$tm
#SBATCH --output=$RUN/$nm.out
$(for l in "$@"; do echo "#SBATCH $l"; done)
export FF_DATA="$FF_DATA" MODELE_PROD_INPUT="$MODELE_PROD_INPUT" ROCKE_SOCRATES_DIR="$ROCKE_SOCRATES_DIR"
export FORTRAN_HUGE_SPACE="$FORTRAN_HUGE_SPACE" FORTRAN_RUNS="$FORTRAN_RUNS" ROCKE_PROJ="$P" OMP_NUM_THREADS=1
unset JAX_COMPILATION_CACHE_DIR CLOUDS_JAX_CACHE XLA_FLAGS
echo "host: \$(hostname)  git: \$(cd $P && git log --oneline -1 | cut -c1-60)  dirty: \$(cd $P && git status --short | grep -v '^??' | wc -l)  start: \$(date -Is)"
EOF
}
gpu_lines=("--constraint=$CONS" "--gres=gpu:1" "--cpus-per-task=8" "--mem=64G")
RUNIN="$CONDA_BIN run --no-capture-output -n $ENVN"

declare -A ID
want() { case " $ONLY " in *" $1 "*) return 0;; esac; return 1; }

for kind in release fast; do
  sf=""; [ $kind = fast ] && sf="$SPEED"
  if want pair_$kind; then
    f=$JOBS/pair_$kind.sbatch
    { header pair_$kind "${TIME_PAIR:-03:00:00}" "${gpu_lines[@]}" "--exclusive"
      echo "cd $P/fullfidelity"
      echo "NFORT=3 FTAG=${NAME}_$kind FCORE=0 JAXOUT=$REL/pair_${kind}_jax SPEED_FLAGS='$sf' $RUNIN bash $P/gpu/benchmark_native_v100.sh"
      echo "echo \"exit: \$?  end: \$(date -Is)\""
    } > "$f"
    ID[pair_$kind]=$(submit "$f") || exit 1
  fi
  if want fid_$kind; then
    f=$JOBS/fid_$kind.sbatch
    { header fid_$kind "${TIME_FID:-01:00:00}" "${gpu_lines[@]}"
      echo "cd $P/fullfidelity"
      echo "env PYTHONUNBUFFERED=1 $RELEASE_FLAGS ${sf:+$sf }GPU_DAYDIR=nov26_day2 GPU_DAY_DRIVER=d212_day $RUNIN python -u gpu_day_run.py $RUN/fid_${kind}_jax --rad native --surf-fields computed --seed computed --cosz computed --nsteps $NF --timed 0"
      echo "echo \"exit: \$?  end: \$(date -Is)\""
    } > "$f"
    ID[fid_$kind]=$(submit "$f") || exit 1
  fi
done

if want tests_off; then
  f=$JOBS/tests_off.sbatch
  { header tests_off "${TIME_TESTS:-04:00:00}" "--cpus-per-task=6" "--mem=48G"
    echo "cd $P/fullfidelity"
    echo "env JAX_PLATFORMS=cpu JOBS=4 LOGDIR=$RUN/tests_off $RUNIN bash run_all_tests_sharded.sh"
    echo "echo \"exit: \$?  end: \$(date -Is)\""
  } > "$f"
  ID[tests_off]=$(submit "$f") || exit 1
fi
if want tests_on; then
  f=$JOBS/tests_on.sbatch
  { header tests_on "${TIME_TESTS:-04:00:00}" "--cpus-per-task=6" "--mem=48G"
    echo "cd $P/fullfidelity"
    echo "env JAX_PLATFORMS=cpu ROCKE_MSTCNV_HOST=1 ROCKE_MSTCNV_IMF=1 ROCKE_MSTCNV_HOST_IO=1 ROCKE_QDYN_VMAP=1 ROCKE_POLE_FAST=2 ROCKE_POLE_ASYNC=1 ROCKE_SEQSUM_BLK=32 ROCKE_GLUE_JIT=1 ROCKE_ENT_BATCH=1 ROCKE_MSTCNV_PY=$ROCKE_MSTCNV_PY JOBS=4 LOGDIR=$RUN/tests_on $RUNIN bash run_all_tests_sharded.sh"
    echo "echo \"exit: \$?  end: \$(date -Is)\""
  } > "$f"
  ID[tests_on]=$(submit "$f") || exit 1
fi

dep=""
for k in "${!ID[@]}"; do dep="${dep:+$dep:}${ID[$k]}"; done
f=$JOBS/summary.sbatch
{ header summary "01:00:00" "--cpus-per-task=2" "--mem=16G"
  echo "cd $P/fullfidelity"
  echo "$RUNIN python $P/gpu/summarize_variants.py $RUN"
  echo "echo \"exit: \$?  end: \$(date -Is)\""
} > "$f"
SID=$(submit "$f" "${dep:+afterany:$dep}") || exit 1

{ echo "run: $NAME  project: $P  commit: $(git -C "$P" rev-parse --short HEAD 2>/dev/null)  submitted: $(date -Is)"
  for k in "${!ID[@]}"; do echo "$k ${ID[$k]}"; done | sort
  echo "summary $SID"
} | tee -a "$RUN/JOBS.txt"
echo "results: $RUN/SUMMARY.md (written by the summary job when every variant has ended)"
