"""061 task 2: self-contained regeneration of fig_metric_blindness.pdf.

The original figure depended on step1_decomp_summary.csv, which no longer exists on
disk. This script recomputes all 12 numbers (4 datasets x {Trend, Linear C2ST,
XGB-C2ST}, all on the FF/zero-dependency reference, 5 seeds each) FROM RAW DATA using
only proper_metrics.py (unmodified, import-only) -- no dependency on any intermediate
CSV that might disappear again. Produces:
  - step_061_fig_metric_blindness_long.csv  (60 rows: 4 datasets x 3 metrics x 5 seeds)
  - step_061_fig_metric_blindness_summary.csv (12 rows: mean/std per dataset x metric)
  - figs/fig_metric_blindness_061.pdf

Panel = adult / shoppers / bank / magic (write/v22/main.tex line 261/270's actual
panel; default is NOT part of it). Trend/LR-C2ST computed against TRAIN (their
standard usage, matching proper_metrics.evaluate_all's train_ref convention).
XGB-C2ST computed against held-out TEST (matching how the existing adult/bank/magic
FF XGB-C2ST numbers in step_*_diag_seed5_051_long.csv / 056's magic backfill were
produced: evaluate_all(sdf, test, test, ...) -> holdout_df=test).
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import load_info, make_ff, sdmetrics_trend, lr_c2st, xgb_c2st, self_check, DATASETS

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
SEEDS = [0, 1, 2, 3, 4]
PANEL = ["adult", "shoppers", "bank", "magic"]
DISPLAY_NAME = {"adult": "adult", "shoppers": "shoppers", "bank": "bank", "magic": "magic"}


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
            xgb = xgb_c2st(test, ff, info, seed=seed)
            rows.append(dict(dataset=ds, seed=seed, n=n,
                              trend=trend, lr_c2st=lr, xgb_c2st=xgb))
            print(f"  {ds:10s} s{seed} Trend={trend:.4f} LR-C2ST={lr:.4f} XGB-C2ST={xgb:.4f}")

    long = pd.DataFrame(rows)
    long.to_csv("step_061_fig_metric_blindness_long.csv", index=False)

    met = ["trend", "lr_c2st", "xgb_c2st"]
    summ = long.groupby("dataset")[met].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reindex(PANEL).reset_index()
    summ.to_csv("step_061_fig_metric_blindness_summary.csv", index=False)
    print("\n[SAVED] step_061_fig_metric_blindness_summary.csv (+_long)")
    print(summ.to_string(index=False))

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(PANEL))
    width = 0.25
    labels = ["Trend (1.0 = perfect)", "Linear C2ST (1.0 = at chance)",
              "XGB-C2ST (1.0 = fully distinguishable)"]
    colors = ["#888888", "#d62728", "#1f77b4"]
    for i, (m, lab, col) in enumerate(zip(met, labels, colors)):
        means = summ[f"{m}_mean"].values
        stds = summ[f"{m}_std"].values
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3,
               label=lab, color=col)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAME[d].upper() for d in PANEL])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score of the zero-dependency FF reference")
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.15))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs("figs", exist_ok=True)
    fig.savefig("figs/fig_metric_blindness_063.pdf")
    print("[SAVED] figs/fig_metric_blindness_061.pdf")


if __name__ == "__main__":
    main()
