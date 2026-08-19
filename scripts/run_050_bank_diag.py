"""050: complete bank's diagnostic table.
Exp1: FF (lower) / TabbyFlow / Oracle (upper) reference rows for bank ->
      full/marg/dep/dep_cross + minority F1 (yes), 3 seeds mean+/-std (047 caliber).
Exp2: 2a moment matching (mean_gap/corr_gap) for FF/TabbyFlow/Oracle on bank (048 caliber).
Reuses proper_metrics + run_a1_coltype + run_048_order.moment_gap.
Neutral numbers only; no verdict.
Outputs: step_bank_diag.csv (+_long), step_bank_order.csv (+_long).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import (load_info, evaluate_all, make_ff, make_oracle,
                            self_check, check_positive_class, DATASETS)
from run_a1_coltype import coltype_c2st
from run_048_order import moment_gap

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
SEEDS = [0, 1, 2]
CFG = {"train": "data/bank/train.csv", "test": "data/bank/test.csv", "info": "data/bank/info.json"}


def tabby(seed):
    return pd.read_csv(f"ef_vfm/result/bank/bank_base_s{seed}/8000/samples.csv")


def main():
    print("=" * 60); print("050 bank diagnostic completion (FF/oracle + 2a)"); print("=" * 60)
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
    info = load_info(CFG["info"]); train = pd.read_csv(CFG["train"]); test = pd.read_csv(CFG["test"])
    check_positive_class(info, test)
    n_test = len(test)
    if n_test < 3000:
        print(f"NOTE bank test n={n_test}")

    diag_rows, order_rows = [], []
    for seed in SEEDS:
        syn = tabby(seed); n = len(syn)
        variants = {"FF": make_ff(train, n, seed=seed),
                    "TabbyFlow": syn,
                    "Oracle": make_oracle(train, n, seed=seed)}
        for name, sdf in variants.items():
            # Exp1 diagnostic
            m = evaluate_all(sdf, test, test, info, seed=seed, do_old=False, train_ref=train)
            dep_cross = m["xgb_c2st"] - coltype_c2st(test, sdf, info, seed=seed, mode="cross")
            diag_rows.append(dict(dataset="bank", variant=name, seed=seed,
                                  full=m["xgb_c2st"], marg=m["marg_c2st"], dep=m["dep_c2st"],
                                  dep_cross=dep_cross, mle_f1=m["mle_f1"], mle_recall=m["mle_recall"]))
            # Exp2 2a moment gap
            mg, cg, npairs = moment_gap(test, sdf, info)
            order_rows.append(dict(dataset="bank", variant=name, seed=seed,
                                   mean_gap=mg, corr_gap=cg, n_pairs=npairs))
            print(f"  {name:10s} s{seed} full={m['xgb_c2st']:.4f} dep={m['dep_c2st']:+.4f} "
                  f"dep_cross={dep_cross:+.4f} f1={m['mle_f1']:.4f} | mean_gap={mg:.4f} corr_gap={cg:.4f}")

    # save Exp1
    dl = pd.DataFrame(diag_rows); dl.to_csv("step_bank_diag_long.csv", index=False)
    met = ["full", "marg", "dep", "dep_cross", "mle_f1", "mle_recall"]
    g = dl.groupby(["dataset", "variant"])[met].agg(["mean", "std"]); g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index(); g["_o"] = g.variant.map({"FF": 0, "TabbyFlow": 1, "Oracle": 2}); g = g.sort_values("_o").drop(columns="_o")
    g.to_csv("step_bank_diag.csv", index=False)
    # save Exp2
    ol = pd.DataFrame(order_rows); ol.to_csv("step_bank_order_long.csv", index=False)
    go = ol.groupby(["dataset", "variant"])[["mean_gap", "corr_gap"]].agg(["mean", "std"]); go.columns = [f"{a}_{b}" for a, b in go.columns]
    go = go.reset_index(); go["_o"] = go.variant.map({"FF": 0, "TabbyFlow": 1, "Oracle": 2}); go = go.sort_values("_o").drop(columns="_o")
    go.to_csv("step_bank_order.csv", index=False)

    print("\n=== Exp1 bank diagnostic (3 seed mean±std) ===")
    for _, r in g.iterrows():
        print(f"  {r['variant']:10s} full={r['full_mean']:.4f} dep={r['dep_mean']:+.4f}±{r['dep_std']:.4f} "
              f"dep_cross={r['dep_cross_mean']:+.4f} f1={r['mle_f1_mean']:.4f}")
    print("=== Exp2 bank 2a moment gap (lower=more aligned) ===")
    for _, r in go.iterrows():
        print(f"  {r['variant']:10s} mean_gap={r['mean_gap_mean']:.4f}±{r['mean_gap_std']:.4f} corr_gap={r['corr_gap_mean']:.4f}")
    print("\n[SAVED] step_bank_diag.csv, step_bank_order.csv (+_long)\nDONE.")


if __name__ == "__main__":
    main()
