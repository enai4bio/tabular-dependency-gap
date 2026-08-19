#!/usr/bin/env bash
# Queue A — adult BASELINE, true 5 seeds (serial).
# Each seed is an independent model (--seed N fixes the bug where all seeds were seed=0).
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/jie/anaconda3/envs/tabbyflow/bin/python
LOG=log_adult_base.txt
: > "$LOG"

for S in 0 1 2 3 4; do
    EXP="adult_base_s${S}"
    # skip if already finished (8000-step samples exist)
    if [ -f "ef_vfm/result/adult/${EXP}/8000/samples.csv" ]; then
        echo "=== SKIP ${EXP}: 8000 samples already exist ===" | tee -a "$LOG"
        continue
    fi
    echo "=== [$(date '+%F %T')] Training ${EXP} (baseline, seed=${S}) ===" | tee -a "$LOG"
    $PY -u main.py --dataname adult --mode train \
        --exp_name "$EXP" --deterministic --seed "$S" --no_wandb --gpu 0 \
        >> "$LOG" 2>&1 \
        || echo "=== WARN: ${EXP} crashed, continuing ===" | tee -a "$LOG"
    echo "=== [$(date '+%F %T')] Done ${EXP} ===" | tee -a "$LOG"
done
echo "[$(date '+%F %T')] QUEUE A (adult baseline) COMPLETE" | tee -a "$LOG"
