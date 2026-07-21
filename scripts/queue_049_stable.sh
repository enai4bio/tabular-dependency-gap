#!/usr/bin/env bash
# 049: retrain default 4x (168M, dim_t=4096) with STABLE optimizer (4x_stable config:
# lr 3e-4, anneal scheduler, warmup 500) so all seeds converge instead of freezing
# at loss~19 (root cause: reduce_lr_on_plateau collapsed LR to ~1e-7).
# ONLY optimizer changed; architecture/steps/batch/dim_t identical to 4x.
# 1x baseline NOT retrained. 3 seeds. skip-if-done.
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/jie/anaconda3/envs/tabbyflow/bin/python
CFG=ef_vfm/configs/ef_vfm_configs_4x_stable.toml
LOG=log_049_stable.txt
SEEDS="${SEEDS:-0 1 2}"
: > "$LOG"

for S in $SEEDS; do
    EXP="default_cap4xstab_s${S}"
    if [ -f "ef_vfm/result/default/${EXP}/8000/samples.csv" ]; then
        echo "=== [$(date '+%F %T')] SKIP ${EXP}: 8000 samples exist ===" | tee -a "$LOG"; continue
    fi
    echo "=== [$(date '+%F %T')] TRAIN ${EXP} (default seed=${S} 4x_stable) ===" | tee -a "$LOG"
    $PY -u main.py --dataname default --mode train --exp_name "$EXP" \
        --deterministic --seed "$S" --no_wandb --gpu 0 --config_path "$CFG" \
        >> "$LOG" 2>&1 \
        || echo "=== WARN: ${EXP} crashed (continuing) ===" | tee -a "$LOG"
    echo "=== [$(date '+%F %T')] DONE ${EXP} ===" | tee -a "$LOG"
done
echo "[$(date '+%F %T')] QUEUE 049 stable COMPLETE" | tee -a "$LOG"
