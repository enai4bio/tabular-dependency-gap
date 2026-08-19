#!/usr/bin/env bash
# 056 Task C: DOWNWARD capacity sweep (dim_t 1024 -> 512/256/128), 5 seeds, same-source.
# Positive-control for the existing upward sweep (048/051: 1x/2x/4x, dep flat).
# Mirrors queue_051_capacity.sh's idiom exactly: skip-if-done, --deterministic --seed N.
# Order: adult (mandatory, priority) then bank (time-permitting), smallest width first
# within each dataset (fastest to train, and the width most likely to reveal a genuine
# capacity effect if one exists) so partial runs still yield the most informative points.
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/jie/anaconda3/envs/tabbyflow/bin/python
# 056 §0 fix #2: `import src` in ef_vfm/main.py resolves via a stale site-packages .pth
# pointing at the dead pre-migration path (/media/jie/expand_5t/7exp_expand/next/ef-vfm).
# Without this, every training invocation below fails immediately with
# ModuleNotFoundError. Not touching the shared conda env; just prefixing PYTHONPATH here.
export PYTHONPATH="/media/jie/expand_5t/7exp/next/ef-vfm:${PYTHONPATH:-}"
CHALF="--config_path ef_vfm/configs/ef_vfm_configs_half.toml"
CQUARTER="--config_path ef_vfm/configs/ef_vfm_configs_quarter.toml"
CEIGHTH="--config_path ef_vfm/configs/ef_vfm_configs_eighth.toml"
LOG=log_056_capacity_down.txt
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

# ---- adult (mandatory, priority): 1/8x, 1/4x, 1/2x, all 5 seeds ----
for S in 0 1 2 3 4; do run adult "adult_cap0.125x_s${S}" "$S" "$CEIGHTH"; done
for S in 0 1 2 3 4; do run adult "adult_cap0.25x_s${S}"  "$S" "$CQUARTER"; done
for S in 0 1 2 3 4; do run adult "adult_cap0.5x_s${S}"   "$S" "$CHALF"; done

# ---- bank (time-permitting) ----
for S in 0 1 2 3 4; do run bank "bank_cap0.125x_s${S}" "$S" "$CEIGHTH"; done
for S in 0 1 2 3 4; do run bank "bank_cap0.25x_s${S}"  "$S" "$CQUARTER"; done
for S in 0 1 2 3 4; do run bank "bank_cap0.5x_s${S}"   "$S" "$CHALF"; done

echo "[$(date '+%F %T')] QUEUE 056 capacity-down COMPLETE" | tee -a "$LOG"
