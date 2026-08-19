#!/usr/bin/env bash
# 056 Task C downward sweep, QUEUE A of 3 (parallel, shares GPU 0 with queues B/C).
# Rebuilt clean (2026-08-17 20:xx) after aborting the earlier 2-way split for a 3-way
# split. Items below are DISJOINT from queue_056_capacity_down_B.sh and _C.sh --
# every (dataset,width,seed) triplet appears in exactly ONE of the three files.
# Est. ~8.95h for this queue's share.
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/jie/anaconda3/envs/tabbyflow/bin/python
CHALF="--config_path ef_vfm/configs/ef_vfm_configs_half.toml"
CEIGHTH="--config_path ef_vfm/configs/ef_vfm_configs_eighth.toml"
export PYTHONPATH="/media/jie/expand_5t/7exp/next/ef-vfm:${PYTHONPATH:-}"
LOG=log_056_capacity_down_A.txt
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

run adult "adult_cap0.5x_s0" 0 "$CHALF"
run adult "adult_cap0.5x_s1" 1 "$CHALF"
run adult "adult_cap0.5x_s2" 2 "$CHALF"
run adult "adult_cap0.125x_s1" 1 "$CEIGHTH"

echo "[$(date '+%F %T')] QUEUE 056-A COMPLETE" | tee -a "$LOG"
