#!/usr/bin/env bash
# 056 Task C downward sweep, QUEUE C of 3 (parallel, shares GPU 0 with queues A/B).
# Rebuilt clean (2026-08-17 20:xx). Disjoint from _A.sh and _B.sh -- see _A.sh header.
# Est. ~9.35h for this queue's share.
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/jie/anaconda3/envs/tabbyflow/bin/python
CQUARTER="--config_path ef_vfm/configs/ef_vfm_configs_quarter.toml"
CEIGHTH="--config_path ef_vfm/configs/ef_vfm_configs_eighth.toml"
export PYTHONPATH="/media/jie/expand_5t/7exp/next/ef-vfm:${PYTHONPATH:-}"
LOG=log_056_capacity_down_C.txt
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

run adult "adult_cap0.25x_s2" 2 "$CQUARTER"
run adult "adult_cap0.25x_s3" 3 "$CQUARTER"
run adult "adult_cap0.25x_s4" 4 "$CQUARTER"
run adult "adult_cap0.125x_s3" 3 "$CEIGHTH"
run adult "adult_cap0.125x_s4" 4 "$CEIGHTH"

echo "[$(date '+%F %T')] QUEUE 056-C COMPLETE" | tee -a "$LOG"
