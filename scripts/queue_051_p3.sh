#!/usr/bin/env bash
# 051 P2+P3: bank P3 and default P3 (litmus PASSed). Paired base-vs-fix, 5 seeds.
# base = standard 1x config (no --fix); fix = --fix --mechanism p3 (num<->cat cross).
# base_s0..2 already exist (049) -> only backfill base_s3,s4 + train fix_s0..4.
# skip-if-done. bank first (P2, required), then default (P3).
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/jie/anaconda3/envs/tabbyflow/bin/python
LOG=log_051_p3.txt
: > "$LOG"

run () {  # dataname exp seed extra
    local ds="$1" exp="$2" seed="$3" extra="$4"
    if [ -f "ef_vfm/result/${ds}/${exp}/8000/samples.csv" ]; then
        echo "=== [$(date '+%F %T')] SKIP ${exp} ===" | tee -a "$LOG"; return
    fi
    echo "=== [$(date '+%F %T')] TRAIN ${exp} (${ds} seed=${seed} ${extra:-base}) ===" | tee -a "$LOG"
    $PY -u main.py --dataname "$ds" --mode train --exp_name "$exp" \
        --deterministic --seed "$seed" --no_wandb --gpu 0 $extra \
        >> "$LOG" 2>&1 \
        || echo "=== WARN: ${exp} crashed ===" | tee -a "$LOG"
    echo "=== [$(date '+%F %T')] DONE ${exp} ===" | tee -a "$LOG"
}

FIX="--fix --mechanism p3"
for ds in bank default; do
    # backfill base seeds 3,4 (standard config)
    run "$ds" "${ds}_base_s3" 3 ""
    run "$ds" "${ds}_base_s4" 4 ""
    # P3 fix seeds 0..4
    for S in 0 1 2 3 4; do run "$ds" "${ds}_fix_s${S}" "$S" "$FIX"; done
done
echo "[$(date '+%F %T')] QUEUE 051 P3 COMPLETE" | tee -a "$LOG"
