#!/usr/bin/env bash
# 059 Task 1: bank downward capacity sweep, SINGLE-QUEUE serial mode (no parallelism).
# Switched from 3-way to serial after a fan failure caused a crash mid-run; user wants
# minimal thermal load while the repaired fan is being trusted again.
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/jie/anaconda3/envs/tabbyflow/bin/python
CHALF="--config_path ef_vfm/configs/ef_vfm_configs_half.toml"
CQUARTER="--config_path ef_vfm/configs/ef_vfm_configs_quarter.toml"
CEIGHTH="--config_path ef_vfm/configs/ef_vfm_configs_eighth.toml"
export PYTHONPATH="/media/jie/expand_5t/7exp/next/ef-vfm:${PYTHONPATH:-}"
LOG=log_059_bank_down_serial.txt
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

# smallest/fastest first, so partial progress is maximally informative if interrupted again
for S in 0 1 2 3 4; do run bank "bank_cap0.125x_s${S}" "$S" "$CEIGHTH"; done
for S in 0 1 2 3 4; do run bank "bank_cap0.25x_s${S}"  "$S" "$CQUARTER"; done
for S in 0 1 2 3 4; do run bank "bank_cap0.5x_s${S}"   "$S" "$CHALF"; done

echo "[$(date '+%F %T')] QUEUE 059-serial COMPLETE" | tee -a "$LOG"
