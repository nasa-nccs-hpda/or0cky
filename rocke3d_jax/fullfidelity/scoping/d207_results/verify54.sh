#!/bin/bash
# compare steps 0-53 of the new records (dir $1) with ff_data/nov26_day: every per-step file of the old set that exists in the new set
OLD=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26_day
NEW=$1
n=0; same=0; diff=0; miss=0
: > $2.diff; : > $2.miss
for f in $(cd $OLD && ls | grep -E '_(333(1[2-9]|[2-5][0-9]|6[0-5]))(_|\.bin)|(333(1[2-9]|[2-5][0-9]|6[0-5]))\.bin'); do
  case $f in ours_*|rsv_*|d172*|real_acc*|overlay*) continue;; esac
  n=$((n+1))
  if [ ! -f $NEW/$f ]; then miss=$((miss+1)); echo $f >> $2.miss; continue; fi
  if cmp -s $OLD/$f $NEW/$f; then same=$((same+1)); else diff=$((diff+1)); echo $f >> $2.diff; fi
done
echo "files_compared=$n identical=$same different=$diff missing_in_new=$miss"
