"""048 Exp3 -- capacity sweep: assemble dep components at 1x / x2 / x4 (mean+/-std).

Reuses existing TabbyFlow samples:
  1x : {ds}_base_s{seed}   (10.6M params)
  x2 : {ds}_cap2x_s{seed}  (42M params, width x2)   [048]
  x4 : {ds}_cap4x_s{seed}  (168M params, width x4)  [047]
Metrics via proper_metrics: full / marg / dep / dep_cross. adult + default, 3 seeds.
Robust: skips (ds,cap,seed) with no samples so it can run incrementally.
Neutral numbers only -- no structure/capacity verdict (research lead's call).
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import load_info, evaluate_all, xgb_c2st, self_check, DATASETS
from run_a1_coltype import coltype_c2st

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
SEEDS = [0, 1, 2]
DS_CFG = {
    "adult":   {"train": "data/adult/train.csv",   "test": "data/adult/test.csv",   "info": "data/adult/info.json"},
    "default": {"train": "data/default/train.csv", "test": "data/default/test.csv", "info": "data/default/info.json"},
}
CAP_EXP = {"1x": "{ds}_base_s{s}", "2x": "{ds}_cap2x_s{s}", "4x": "{ds}_cap4x_s{s}"}
PARAMS = {"1x": "10.6M", "2x": "42M", "4x": "168M"}


def syn_path(ds, cap, s):
    p = f"ef_vfm/result/{ds}/{CAP_EXP[cap].format(ds=ds, s=s)}/8000/samples.csv"
    return p if os.path.exists(p) else None


def main():
    print("=" * 60); print("048 Exp3 capacity sweep 1x/x2/x4"); print("=" * 60)
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
    rows = []
    for ds, cfg in DS_CFG.items():
        info = load_info(cfg["info"]); train = pd.read_csv(cfg["train"]); test = pd.read_csv(cfg["test"])
        for cap in ["1x", "2x", "4x"]:
            for s in SEEDS:
                p = syn_path(ds, cap, s)
                if p is None:
                    continue
                syn = pd.read_csv(p)
                m = evaluate_all(syn, test, test, info, seed=s, do_old=False, train_ref=train)
                dep_cross = m["xgb_c2st"] - coltype_c2st(test, syn, info, seed=s, mode="cross")
                rows.append(dict(dataset=ds, capacity=cap, params=PARAMS[cap], seed=s,
                                 full=m["xgb_c2st"], marg=m["marg_c2st"], dep=m["dep_c2st"], dep_cross=dep_cross))
                print(f"  {ds:8s} {cap} ({PARAMS[cap]:>5}) s{s} full={m['xgb_c2st']:.4f} dep={m['dep_c2st']:+.4f} dep_cross={dep_cross:+.4f}")
    if not rows:
        print("no samples yet"); return
    long = pd.DataFrame(rows); long.to_csv("step_capacity_sweep_long.csv", index=False)
    g = long.groupby(["dataset", "capacity", "params"])[["full", "marg", "dep", "dep_cross"]].agg(["mean", "std", "count"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index()
    # order caps 1x<2x<4x
    g["_o"] = g.capacity.map({"1x": 0, "2x": 1, "4x": 2}); g = g.sort_values(["dataset", "_o"]).drop(columns="_o")
    g.to_csv("step_capacity_sweep.csv", index=False)
    print("\n[SAVED] step_capacity_sweep.csv (+ _long)")
    print(g[["dataset", "capacity", "params", "dep_mean", "dep_std", "dep_cross_mean", "dep_cross_std", "dep_mean_count" if "dep_mean_count" in g else "dep_count"]].to_string(index=False))
    print("\nDONE.")


if __name__ == "__main__":
    main()
