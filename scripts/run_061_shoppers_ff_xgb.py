"""061 task 1: shoppers FF XGB-C2ST, 5 seeds, same FF generation as run_060_trend_ff.py
(same train, same n=len(test), same seeds -> make_ff is a deterministic RNG function,
so re-calling it with identical arguments reproduces the identical FF sample; no new
random source introduced). Reference set = held-out test, matching how the existing
adult/bank/magic FF XGB-C2ST numbers were computed (evaluate_all(sdf, test, test, ...)
in run_051_adult_seed5.py and 056's run_magic_diag5.py -- holdout_df=test).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import load_info, make_ff, xgb_c2st, self_check, DATASETS

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
SEEDS = [0, 1, 2, 3, 4]


def main():
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
    info = load_info("data/shoppers/info.json")
    train = pd.read_csv("data/shoppers/train.csv")
    test = pd.read_csv("data/shoppers/test.csv")
    n = len(test)  # identical to run_060_trend_ff.py's n for shoppers
    rows = []
    for seed in SEEDS:
        ff = make_ff(train, n, seed=seed)  # identical call -> identical FF sample as 060
        c2st = xgb_c2st(test, ff, info, seed=seed)
        rows.append(dict(dataset="shoppers", seed=seed, n=n, xgb_c2st_ff=c2st))
        print(f"  shoppers s{seed} n={n} FF XGB-C2ST={c2st:.4f}")
    long = pd.DataFrame(rows)
    long.to_csv("step_061_shoppers_ff_xgb_long.csv", index=False)
    print(f"\nmean={long.xgb_c2st_ff.mean():.4f} std={long.xgb_c2st_ff.std(ddof=1):.4f}")
    long.to_csv("step_061_shoppers_ff_xgb_long.csv", index=False)


if __name__ == "__main__":
    main()
