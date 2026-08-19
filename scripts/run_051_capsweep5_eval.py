"""051 capacity sweep to 5 seeds: eval all existing (dataset,cap,seed) TabbyFlow
samples -> full/marg/dep/dep_cross + minority F1. Robust: skips missing (runs
incrementally while 4x Phase B trains). 4 datasets x {1x,2x,4x} x 5 seeds.
Neutral numbers only. Output: step_capacity_sweep5_051.csv (+_long).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import load_info, evaluate_all, self_check, DATASETS
from run_a1_coltype import coltype_c2st

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
SEEDS = [0, 1, 2, 3, 4]
DSS = {
    "adult":   {"train": "data/adult/train.csv",   "test": "data/adult/test.csv",   "info": "data/adult/info.json"},
    "bank":    {"train": "data/bank/train.csv",    "test": "data/bank/test.csv",    "info": "data/bank/info.json"},
    "default": {"train": "data/default/train.csv", "test": "data/default/test.csv", "info": "data/default/info.json"},
    "magic":   {"train": "data/magic/train.csv",   "test": "data/magic/test.csv",   "info": "data/magic/info.json"},
}
CAP_EXP = {"1x": "{ds}_base_s{s}", "2x": "{ds}_cap2x_s{s}", "4x": "{ds}_cap4x_s{s}"}
PARAMS = {"1x": "10.6M", "2x": "42M", "4x": "168M"}


def main():
    print("=" * 60); print("051 capacity sweep -> 5 seeds (all datasets)"); print("=" * 60)
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
    rows = []
    for ds, cfg in DSS.items():
        info = load_info(cfg["info"]); train = pd.read_csv(cfg["train"]); test = pd.read_csv(cfg["test"])
        for cap in ["1x", "2x", "4x"]:
            for s in SEEDS:
                p = f"ef_vfm/result/{ds}/{CAP_EXP[cap].format(ds=ds, s=s)}/8000/samples.csv"
                if not os.path.exists(p):
                    continue
                syn = pd.read_csv(p)
                m = evaluate_all(syn, test, test, info, seed=s, do_old=False, train_ref=train)
                dc = m["xgb_c2st"] - coltype_c2st(test, syn, info, seed=s, mode="cross")
                rows.append(dict(dataset=ds, capacity=cap, params=PARAMS[cap], seed=s,
                                 full=m["xgb_c2st"], marg=m["marg_c2st"], dep=m["dep_c2st"],
                                 dep_cross=dc, mle_f1=m["mle_f1"], mle_recall=m["mle_recall"]))
                print(f"  {ds:8s} {cap} s{s} full={m['xgb_c2st']:.4f} dep={m['dep_c2st']:+.4f} dep_cross={dc:+.4f} f1={m['mle_f1']:.4f}")
    long = pd.DataFrame(rows); long.to_csv("step_capacity_sweep5_051_long.csv", index=False)
    met = ["full", "marg", "dep", "dep_cross", "mle_f1", "mle_recall"]
    g = long.groupby(["dataset", "capacity", "params"])[met].agg(["mean", "std", "count"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]; g = g.reset_index()
    g["_o"] = g.capacity.map({"1x": 0, "2x": 1, "4x": 2}); g = g.sort_values(["dataset", "_o"]).drop(columns="_o")
    g.to_csv("step_capacity_sweep5_051.csv", index=False)
    print("\n=== dep by dataset x capacity (n=seed count) ===")
    for _, r in g.iterrows():
        print(f"  {r['dataset']:8s} {r['capacity']} dep={r['dep_mean']:+.4f}±{r['dep_std']:.4f} (n={int(r['dep_count'])})")
    print("\n[SAVED] step_capacity_sweep5_051.csv (+_long)\nDONE.")


if __name__ == "__main__":
    main()
