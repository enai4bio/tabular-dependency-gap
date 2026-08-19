#!/usr/bin/env bash
# 056 Task C downward sweep, QUEUE B of 3 (parallel, shares GPU 0 with queues A/C).
# Rebuilt clean (2026-08-17 20:xx). Disjoint from _A.sh and _C.sh -- see _A.sh header.
# Est. ~10.6h for this queue's share (largest of the 3, started first-ish to compensate).
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/jie/anaconda3/envs/tabbyflow/bin/python
CHALF="--config_path ef_vfm/configs/ef_vfm_configs_half.toml"
CQUARTER="--config_path ef_vfm/configs/ef_vfm_configs_quarter.toml"
CEIGHTH="--config_path ef_vfm/configs/ef_vfm_configs_eighth.toml"
export PYTHONPATH="/media/jie/expand_5t/7exp/next/ef-vfm:${PYTHONPATH:-}"
LOG=log_056_capacity_down_B.txt
: > "$LOG"

run () {  # dataname exp seed cfg
    local ds="$1" exp="$2" seed="$3" cfg="$4"
    if [ -f "ef_vfm/result/${ds}/${exp}/8000/samples.csv" ]; then
        echo "=== [$(date '+%F %T')] SKIP ${exp} ===" | tee -a "$LOG"; return
    fi
    echo "=== [$(date '+%F %T')] TRAIN ${exp} (${ds} seed=${seed} ${cfg:-1x}) ===" | tee -a "$LOG"
    $PY -u main.py --dataname "$ds" --mode train --exp_name "$exp" \
        --deterministic --seed "$seed" --no_wandb --gpu 0 $cfg \
        >> "$LOG" 2>&1 \
        || echo "=== WARN: ${exp} crashed ===" | tee -a "$LOG"
    echo "=== [$(date '+%F %T')] DONE ${exp} ===" | tee -a "$LOG"
}

run adult "adult_cap0.5x_s3" 3 "$CHALF"
run adult "adult_cap0.5x_s4" 4 "$CHALF"
run adult "adult_cap0.25x_s0" 0 "$CQUARTER"
run adult "adult_cap0.25x_s1" 1 "$CQUARTER"
run adult "adult_cap0.125x_s2" 2 "$CEIGHTH"

echo "[$(date '+%F %T')] QUEUE 056-B COMPLETE" | tee -a "$LOG"
