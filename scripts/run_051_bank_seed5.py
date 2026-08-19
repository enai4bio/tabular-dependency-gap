"""051 P4 (free re-eval): bank diagnostic + 2a to 5 seeds (5 models already exist).
FF/TabbyFlow/Oracle: full/marg/dep/dep_cross + minority F1, and 2a moment gap. 5 seeds.
Outputs: step_bank_diag_seed5_051.csv, step_bank_order_seed5_051.csv (+_long).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import (load_info, evaluate_all, make_ff, make_oracle,
                            self_check, check_positive_class, DATASETS)
from run_a1_coltype import coltype_c2st
from run_048_order import moment_gap

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
SEEDS = [0, 1, 2, 3, 4]
CFG = {"train": "data/bank/train.csv", "test": "data/bank/test.csv", "info": "data/bank/info.json"}


def main():
    print("=" * 60); print("051 bank diagnostic + 2a -> 5 seeds (free re-eval)"); print("=" * 60)
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
    info = load_info(CFG["info"]); train = pd.read_csv(CFG["train"]); test = pd.read_csv(CFG["test"])
    check_positive_class(info, test)
    diag, order = [], []
    for seed in SEEDS:
        syn = pd.read_csv(f"ef_vfm/result/bank/bank_base_s{seed}/8000/samples.csv"); n = len(syn)
        variants = {"FF": make_ff(train, n, seed=seed), "TabbyFlow": syn, "Oracle": make_oracle(train, n, seed=seed)}
        for name, sdf in variants.items():
            m = evaluate_all(sdf, test, test, info, seed=seed, do_old=False, train_ref=train)
            dc = m["xgb_c2st"] - coltype_c2st(test, sdf, info, seed=seed, mode="cross")
            diag.append(dict(dataset="bank", variant=name, seed=seed, full=m["xgb_c2st"], marg=m["marg_c2st"],
                             dep=m["dep_c2st"], dep_cross=dc, mle_f1=m["mle_f1"], mle_recall=m["mle_recall"]))
            mg, cg, npr = moment_gap(test, sdf, info)
            order.append(dict(dataset="bank", variant=name, seed=seed, mean_gap=mg, corr_gap=cg, n_pairs=npr))
            print(f"  {name:10s} s{seed} full={m['xgb_c2st']:.4f} dep={m['dep_c2st']:+.4f} dep_cross={dc:+.4f} f1={m['mle_f1']:.4f} mean_gap={mg:.4f}")
    for rows, met, tag in [(diag, ["full","marg","dep","dep_cross","mle_f1","mle_recall"], "diag"),
                           (order, ["mean_gap","corr_gap"], "order")]:
        L = pd.DataFrame(rows); L.to_csv(f"step_bank_{tag}_seed5_051_long.csv", index=False)
        g = L.groupby(["dataset","variant"])[met].agg(["mean","std"]); g.columns=[f"{a}_{b}" for a,b in g.columns]
        g=g.reset_index(); g["_o"]=g.variant.map({"FF":0,"TabbyFlow":1,"Oracle":2}); g=g.sort_values("_o").drop(columns="_o")
        g.to_csv(f"step_bank_{tag}_seed5_051.csv", index=False)
    print("\n=== bank diag 5seed ==="); print(pd.read_csv("step_bank_diag_seed5_051.csv")[["variant","dep_mean","dep_std","dep_cross_mean","mle_f1_mean"]].to_string(index=False))
    print("=== bank 2a 5seed ==="); print(pd.read_csv("step_bank_order_seed5_051.csv")[["variant","mean_gap_mean","mean_gap_std","corr_gap_mean"]].to_string(index=False))
    print("\nDONE.")


if __name__ == "__main__":
    main()
