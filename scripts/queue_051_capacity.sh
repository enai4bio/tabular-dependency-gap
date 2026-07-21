#!/usr/bin/env bash
# 051 P3+P4 capacity backfill to 5 seeds (STANDARD same-source configs). skip-if-done.
#  magic: base s3,s4 (1x) + cap2x s0..4 + cap4x s0..4  (all-new 2x/4x)
#  adult/bank/default: cap2x s3,s4 + cap4x s3,s4  (backfill to 5)
# Phase A = cheap 1x/2x first; Phase B = expensive 4x (168M) last.
# Convergence gate: per-seed final loss reported later; 4x may fail to converge
# (default 4x known to collapse LR; magic 4x may too) -> reported as-is, not hidden.
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/jie/anaconda3/envs/tabbyflow/bin/python
C2X="--config_path ef_vfm/configs/ef_vfm_configs_2x.toml"
C4X="--config_path ef_vfm/configs/ef_vfm_configs_4x.toml"
LOG=log_051_capacity.txt
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

# ---- Phase A: 1x/2x (cheap, trainable) ----
run magic magic_base_s3 3 ""
run magic magic_base_s4 4 ""
for S in 0 1 2 3 4; do run magic "magic_cap2x_s${S}" "$S" "$C2X"; done
for S in 3 4; do run adult   "adult_cap2x_s${S}"   "$S" "$C2X"; done
for S in 3 4; do run bank    "bank_cap2x_s${S}"    "$S" "$C2X"; done
for S in 3 4; do run default "default_cap2x_s${S}" "$S" "$C2X"; done

# ---- Phase B: 4x (168M, slow, may not converge) ----
for S in 0 1 2 3 4; do run magic "magic_cap4x_s${S}" "$S" "$C4X"; done
for S in 3 4; do run adult   "adult_cap4x_s${S}"   "$S" "$C4X"; done
for S in 3 4; do run bank    "bank_cap4x_s${S}"    "$S" "$C4X"; done
# default 4x s3/s4 REMOVED (2026-07-12): default 4x 决定按训练失败作废、不进论文,不再跑(省~15h)。
# for S in 3 4; do run default "default_cap4x_s${S}" "$S" "$C4X"; done

echo "[$(date '+%F %T')] QUEUE 051 capacity COMPLETE" | tee -a "$LOG"
