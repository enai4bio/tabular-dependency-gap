#!/usr/bin/env bash
# 048 Exp3: capacity midpoint width x2 (dim_t 2048, ~4x params).
# adult (required) then default (if time), 3 seeds each. Same-source: only dim_t
# differs from the 1x baseline. skip-if-done identical to queue_047.sh.
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/jie/anaconda3/envs/tabbyflow/bin/python
CFG2X=ef_vfm/configs/ef_vfm_configs_2x.toml
LOG=log_048_cap2x.txt
SEEDS="0 1 2"
: > "$LOG"

run_one () {
    local ds="$1" exp="$2" seed="$3"
    if [ -f "ef_vfm/result/${ds}/${exp}/8000/samples.csv" ]; then
        echo "=== [$(date '+%F %T')] SKIP ${exp}: 8000 samples exist ===" | tee -a "$LOG"
        return
    fi
    echo "=== [$(date '+%F %T')] TRAIN ${exp} (ds=${ds} seed=${seed} cfg=2x) ===" | tee -a "$LOG"
    $PY -u main.py --dataname "$ds" --mode train --exp_name "$exp" \
        --deterministic --seed "$seed" --no_wandb --gpu 0 --config_path "$CFG2X" \
        >> "$LOG" 2>&1 \
        || echo "=== WARN: ${exp} crashed (continuing) ===" | tee -a "$LOG"
    echo "=== [$(date '+%F %T')] DONE ${exp} ===" | tee -a "$LOG"
}

for S in $SEEDS; do run_one adult   "adult_cap2x_s${S}"   "$S"; done
for S in $SEEDS; do run_one default "default_cap2x_s${S}" "$S"; done

echo "[$(date '+%F %T')] QUEUE 048 cap2x COMPLETE" | tee -a "$LOG"
