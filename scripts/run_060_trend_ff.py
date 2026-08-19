"""060 task 1: recompute FF Trend AND FF LR-C2ST on 5 seeds for the metric-blindness
panel main.tex actually names at line 261/270 -- \textsc{adult}, \textsc{shoppers},
\textsc{bank}, \textsc{magic} (NOT default -- line 270 explicitly says "default was
not part of this panel"). Matches proper_metrics.evaluate_all's train_ref convention
(Trend/LR-C2ST computed against REAL TRAIN, their standard usage, not held-out test).

n (FF sample count) = len(test.csv) for every dataset, uniformly -- shoppers has no
generator-sample directory in this repo's current pipeline (ef_vfm/result/shoppers/
does not exist) so there is no generator-sample-count to borrow from for that one
dataset; using test-set size uniformly avoids mixing conventions across datasets, and
matches the "test n=..." sizing already documented in make_figs.py's own comments for
this exact panel.

NOTE: the original script/data that produced main.tex's specific numbers (LR-C2ST
0.999/1.000/0.995/0.997, Trend range 0.69--0.78) could not be located on disk
(step1_decomp_summary.csv, referenced by make_figs.py, no longer exists) -- this is a
FRESH recomputation with the current standard proper_metrics.py pipeline, not a
literal reproduction of whatever produced the original numbers.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import load_info, make_ff, sdmetrics_trend, lr_c2st, self_check, DATASETS

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
SEEDS = [0, 1, 2, 3, 4]
PANEL = ["adult", "shoppers", "bank", "magic"]  # main.tex line 261/270's actual panel, NOT default


def main():
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
    rows = []
    for ds in PANEL:
        info = load_info(f"data/{ds}/info.json")
        train = pd.read_csv(f"data/{ds}/train.csv")
        test = pd.read_csv(f"data/{ds}/test.csv")
        n = len(test)
        for seed in SEEDS:
            ff = make_ff(train, n, seed=seed)
            trend = sdmetrics_trend(train, ff, info)
            lr = lr_c2st(train, ff, info)
            rows.append(dict(dataset=ds, seed=seed, n=n, trend_ff=trend, lr_c2st_ff=lr))
            print(f"  {ds:10s} s{seed} n={n} FF Trend={trend:.4f} FF LR-C2ST={lr:.4f}")
    long = pd.DataFrame(rows)
    long.to_csv("step_060_trend_ff_long.csv", index=False)
    g = long.groupby("dataset")[["trend_ff", "lr_c2st_ff"]].agg(["mean", "std", "min", "max", "count"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]; g = g.reset_index()
    g.to_csv("step_060_trend_ff.csv", index=False)
    print("\n[SAVED] step_060_trend_ff.csv (+_long)")
    print(g.to_string(index=False))
    print(f"\nTrend overall range across panel: [{long.trend_ff.min():.4f}, {long.trend_ff.max():.4f}]")
    print(f"LR-C2ST overall range across panel: [{long.lr_c2st_ff.min():.4f}, {long.lr_c2st_ff.max():.4f}]")


if __name__ == "__main__":
    main()
