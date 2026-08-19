#!/usr/bin/env bash
# 047 finalize: training queue (serial, single GPU).
#  - 1x baselines use the ORIGINAL config (no --config_path).
#  - 4x capacity runs use ef_vfm_configs_4x.toml (dim_t x4) via --config_path.
# Priority order so the two deliverables become computable as early as possible:
#   1) default 1x   (multidataset diag headline + default capacity-1x)
#   2) adult   4x   (capacity check CORE -- must)
#   3) default 4x   (capacity second dataset)
#   4) magic   1x   (multidataset diag aux; test underpowered ~1900)
# adult 1x already exists (adult_base_s0..s4) -> reused, NOT retrained.
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/jie/anaconda3/envs/tabbyflow/bin/python
CFG4X=ef_vfm/configs/ef_vfm_configs_4x.toml
LOG=log_047_queue.txt
SEEDS="0 1 2"
: > "$LOG"

run_one () {
    local ds="$1" exp="$2" cfgflag="$3" seed="$4"
    if [ -f "ef_vfm/result/${ds}/${exp}/8000/samples.csv" ]; then
        echo "=== [$(date '+%F %T')] SKIP ${exp}: 8000 samples already exist ===" | tee -a "$LOG"
        return
    fi
    echo "=== [$(date '+%F %T')] TRAIN ${exp} (ds=${ds} seed=${seed} cfg=${cfgflag:-1x}) ===" | tee -a "$LOG"
    $PY -u main.py --dataname "$ds" --mode train --exp_name "$exp" \
        --deterministic --seed "$seed" --no_wandb --gpu 0 $cfgflag \
        >> "$LOG" 2>&1 \
        || echo "=== WARN: ${exp} crashed (continuing) ===" | tee -a "$LOG"
    echo "=== [$(date '+%F %T')] DONE ${exp} ===" | tee -a "$LOG"
}

# 1) default 1x
for S in $SEEDS; do run_one default "default_base_s${S}" "" "$S"; done
# 2) adult 4x
for S in $SEEDS; do run_one adult   "adult_cap4x_s${S}" "--config_path $CFG4X" "$S"; done
# 3) default 4x
for S in $SEEDS; do run_one default "default_cap4x_s${S}" "--config_path $CFG4X" "$S"; done
# 4) magic 1x
for S in $SEEDS; do run_one magic   "magic_base_s${S}" "" "$S"; done

echo "[$(date '+%F %T')] QUEUE 047 COMPLETE" | tee -a "$LOG"
