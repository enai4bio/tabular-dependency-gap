#!/usr/bin/env bash
# 049 bank capacity sweep: TabbyFlow 1x/2x/4x (dim_t 1024/2048/4096) on bank-marketing,
# STANDARD configs (same-source as 047/048 default). 3 seeds each, skip-if-done.
# s0 of all three caps first (to expose early whether 4x trains), then s1, s2.
# NOTE: 4x uses the original plateau scheduler; per 049 diagnosis it may collapse LR.
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/jie/anaconda3/envs/tabbyflow/bin/python
LOG=log_049_bank.txt
: > "$LOG"

run_one () {  # exp  cfgflag  seed
    local exp="$1" cfg="$2" seed="$3"
    if [ -f "ef_vfm/result/bank/${exp}/8000/samples.csv" ]; then
        echo "=== [$(date '+%F %T')] SKIP ${exp} ===" | tee -a "$LOG"; return
    fi
    echo "=== [$(date '+%F %T')] TRAIN ${exp} (seed=${seed} cfg=${cfg:-1x}) ===" | tee -a "$LOG"
    $PY -u main.py --dataname bank --mode train --exp_name "$exp" \
        --deterministic --seed "$seed" --no_wandb --gpu 0 $cfg \
        >> "$LOG" 2>&1 \
        || echo "=== WARN: ${exp} crashed (continuing) ===" | tee -a "$LOG"
    echo "=== [$(date '+%F %T')] DONE ${exp} ===" | tee -a "$LOG"
}

C2X="--config_path ef_vfm/configs/ef_vfm_configs_2x.toml"
C4X="--config_path ef_vfm/configs/ef_vfm_configs_4x.toml"
for S in 0 1 2; do
    run_one "bank_base_s${S}"   ""      "$S"
    run_one "bank_cap2x_s${S}"  "$C2X"  "$S"
    run_one "bank_cap4x_s${S}"  "$C4X"  "$S"
done
echo "[$(date '+%F %T')] QUEUE 049 bank COMPLETE" | tee -a "$LOG"
