"""049 bank capacity sweep eval: full/marg/dep/dep_cross + minority F1 for
TabbyFlow 1x/2x/4x on bank-marketing, 3 seeds, per-seed + mean+/-std.
Reuses proper_metrics; neutral numbers only (no structure/capacity verdict).
Outputs: step_capacity_sweep_bank.csv (+ _long).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import load_info, evaluate_all, xgb_c2st, self_check, check_positive_class, DATASETS
from run_a1_coltype import coltype_c2st

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
SEEDS = [0, 1, 2]
CFG = {"train": "data/bank/train.csv", "test": "data/bank/test.csv", "info": "data/bank/info.json"}
CAP_EXP = {"1x": "bank_base_s{s}", "2x": "bank_cap2x_s{s}", "4x": "bank_cap4x_s{s}"}
PARAMS = {"1x": "10.6M", "2x": "42M", "4x": "168M"}
# final train loss (read from log_049_bank.txt)
TRAINLOSS = {("1x",0):13.0,("1x",1):13.0,("1x",2):13.0,
             ("2x",0):12.9,("2x",1):12.9,("2x",2):13.0,
             ("4x",0):12.9,("4x",1):12.9,("4x",2):13.0}


def main():
    print("=" * 60); print("049 bank capacity sweep eval"); print("=" * 60)
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
    info = load_info(CFG["info"]); train = pd.read_csv(CFG["train"]); test = pd.read_csv(CFG["test"])
    n_test = len(test)
    if n_test < 3000:
        print(f"WARN bank test n={n_test} < 3000")
    check_positive_class(info, test)
    rows = []
    for cap in ["1x", "2x", "4x"]:
        for s in SEEDS:
            p = f"ef_vfm/result/bank/{CAP_EXP[cap].format(s=s)}/8000/samples.csv"
            if not os.path.exists(p):
                print(f"MISSING {cap} s{s}"); continue
            syn = pd.read_csv(p)
            m = evaluate_all(syn, test, test, info, seed=s, do_old=False, train_ref=train)
            dep_cross = m["xgb_c2st"] - coltype_c2st(test, syn, info, seed=s, mode="cross")
            rows.append(dict(dataset="bank", capacity=cap, params=PARAMS[cap], seed=s,
                             final_train_loss=TRAINLOSS[(cap, s)],
                             full=m["xgb_c2st"], marg=m["marg_c2st"], dep=m["dep_c2st"],
                             dep_cross=dep_cross, mle_f1=m["mle_f1"], mle_recall=m["mle_recall"]))
            print(f"  {cap} ({PARAMS[cap]:>5}) s{s} loss={TRAINLOSS[(cap,s)]} full={m['xgb_c2st']:.4f} "
                  f"dep={m['dep_c2st']:+.4f} dep_cross={dep_cross:+.4f} f1={m['mle_f1']:.4f} rec={m['mle_recall']:.4f}")
    long = pd.DataFrame(rows); long.to_csv("step_capacity_sweep_bank_long.csv", index=False)
    met = ["final_train_loss", "full", "marg", "dep", "dep_cross", "mle_f1", "mle_recall"]
    g = long.groupby(["dataset", "capacity", "params"])[met].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]; g = g.reset_index()
    g["_o"] = g.capacity.map({"1x": 0, "2x": 1, "4x": 2}); g = g.sort_values("_o").drop(columns="_o")
    g.to_csv("step_capacity_sweep_bank.csv", index=False)
    print("\n[SAVED] step_capacity_sweep_bank.csv (+ _long)")
    print(g[["capacity", "params", "final_train_loss_mean", "dep_mean", "dep_std",
             "dep_cross_mean", "dep_cross_std", "full_mean", "mle_f1_mean"]].to_string(index=False))
    print("\nDONE.")


if __name__ == "__main__":
    main()
