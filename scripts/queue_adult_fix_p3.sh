#!/usr/bin/env bash
# Queue B — adult FIX, P3-only (cross_head ON, P1/L_head OFF), true 5 seeds (serial).
# --mechanism p3 isolates P3 so a positive result is cleanly attributable to it
# (P1's L is dropped during sampling = dead lever; including it would confound mu).
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/jie/anaconda3/envs/tabbyflow/bin/python
LOG=log_adult_fix_p3.txt
: > "$LOG"

for S in 0 1 2 3 4; do
    EXP="adult_fix_s${S}"
    if [ -f "ef_vfm/result/adult/${EXP}/8000/samples.csv" ]; then
        echo "=== SKIP ${EXP}: 8000 samples already exist ===" | tee -a "$LOG"
        continue
    fi
    echo "=== [$(date '+%F %T')] Training ${EXP} (fix P3-only, seed=${S}) ===" | tee -a "$LOG"
    $PY -u main.py --dataname adult --mode train \
        --exp_name "$EXP" --deterministic --seed "$S" --no_wandb --gpu 0 \
        --fix --mechanism p3 \
        >> "$LOG" 2>&1 \
        || echo "=== WARN: ${EXP} crashed, continuing ===" | tee -a "$LOG"
    echo "=== [$(date '+%F %T')] Done ${EXP} ===" | tee -a "$LOG"
done
echo "[$(date '+%F %T')] QUEUE B (adult fix P3-only) COMPLETE" | tee -a "$LOG"
