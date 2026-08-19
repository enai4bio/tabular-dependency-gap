#!/usr/bin/env bash
# 051-TabDiff: train TabDiff (2nd generator) on adult + default, 5 seeds each,
# using ef-vfm-fix's EXACT split (data copied into TabDiff/data/, md5-verified).
# Default TabDiff recipe (steps 8000/batch 4096/lr 1e-3/dim_t 1024/50 diff steps).
# --seed added to TabDiff (minimal change). Training auto-samples at val steps ->
# result/{ds}/{exp}/8000/samples.csv (train-size, original-column format).
# skip-if-done. adult first (diag headline + larger), then default.
set -uo pipefail
TD=/media/jie/expand_5t/7exp/next/TabDiff
cd "$TD"
PY=/home/jie/anaconda3/envs/tabdiff/bin/python
LOG=/media/jie/expand_5t/7exp/next/ef-vfm-fix/log_tabdiff.txt
: > "$LOG"

run () {  # dataname seed
    local ds="$1" s="$2" exp="${1}_tabdiff_s${2}"
    if [ -f "$TD/tabdiff/result/$ds/$exp/8000/samples.csv" ]; then
        echo "=== [$(date '+%F %T')] SKIP $exp ===" | tee -a "$LOG"; return
    fi
    echo "=== [$(date '+%F %T')] TRAIN $exp (ds=$ds seed=$s) ===" | tee -a "$LOG"
    $PY -u main.py --dataname "$ds" --mode train --exp_name "$exp" \
        --deterministic --seed "$s" --no_wandb --gpu 0 \
        >> "$LOG" 2>&1 \
        || echo "=== WARN: $exp crashed ===" | tee -a "$LOG"
    echo "=== [$(date '+%F %T')] DONE $exp ===" | tee -a "$LOG"
}

for S in 0 1 2 3 4; do run adult   "$S"; done
for S in 0 1 2 3 4; do run default "$S"; done
echo "[$(date '+%F %T')] QUEUE TabDiff COMPLETE" | tee -a "$LOG"
