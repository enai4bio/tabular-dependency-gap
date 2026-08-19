"""051 P1/P3: metric-blindness for bank + magic.
For FF (lower) and TabbyFlow: Trend + LR-C2ST (blind metrics) vs XGB-C2ST (correct).
Point: FF keeps Trend/LR near-perfect (blind to dependency) while XGB-C2ST flags it.
3 seeds (existing TabbyFlow samples). Neutral numbers only.
Outputs: step_bank_blindness_051.csv, step_magic_blindness_051.csv (+_long).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import (load_info, xgb_c2st, sdmetrics_trend, lr_c2st,
                            make_ff, self_check, DATASETS)

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
SEEDS = [0, 1, 2]
DSS = {
    "bank":  {"train": "data/bank/train.csv",  "test": "data/bank/test.csv",  "info": "data/bank/info.json"},
    "magic": {"train": "data/magic/train.csv", "test": "data/magic/test.csv", "info": "data/magic/info.json"},
}


def tabby(ds, seed):
    return pd.read_csv(f"ef_vfm/result/{ds}/{ds}_base_s{seed}/8000/samples.csv")


def main():
    print("=" * 60); print("051 P1/P3 metric-blindness (bank + magic)"); print("=" * 60)
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
    for ds, cfg in DSS.items():
        info = load_info(cfg["info"]); train = pd.read_csv(cfg["train"]); test = pd.read_csv(cfg["test"])
        rows = []
        for seed in SEEDS:
            syn = tabby(ds, seed); n = len(syn)
            variants = {"FF": make_ff(train, n, seed=seed), "TabbyFlow": syn}
            for name, sdf in variants.items():
                xgb = xgb_c2st(test, sdf, info, seed=seed)
                trend = sdmetrics_trend(test, sdf, info)
                lr = lr_c2st(test, sdf, info)
                rows.append(dict(dataset=ds, variant=name, seed=seed,
                                 trend=trend, lr_c2st=lr, xgb_c2st=xgb))
                print(f"  {ds:6s} {name:10s} s{seed} Trend={trend:.4f} LR-C2ST={lr:.4f} XGB-C2ST={xgb:.4f}")
        long = pd.DataFrame(rows); long.to_csv(f"step_{ds}_blindness_051_long.csv", index=False)
        g = long.groupby(["dataset", "variant"])[["trend", "lr_c2st", "xgb_c2st"]].agg(["mean", "std"])
        g.columns = [f"{a}_{b}" for a, b in g.columns]
        g.reset_index().to_csv(f"step_{ds}_blindness_051.csv", index=False)
        print(f"  [SAVED] step_{ds}_blindness_051.csv")
    print("\nDONE.")


if __name__ == "__main__":
    main()
