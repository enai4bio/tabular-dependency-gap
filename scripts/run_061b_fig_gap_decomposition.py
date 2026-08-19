"""061b: self-contained regeneration of fig_gap_decomposition.pdf (the headline
adult+shoppers marginal/dependency stacked-bar figure), replacing dependence on the
deleted step1_decomp_summary.csv, mirroring run_061_fig_metric_blindness.py's approach.

adult: FF/TabbyFlow/Oracle, 5 seeds each, from the existing standard pipeline
  (data/adult + ef_vfm/result/adult/adult_base_s{0..4}), matching
  step_adult_diag_seed5_051_long.csv exactly (recomputed fresh here, not read from
  that CSV, so this script has zero dependency on any other step_*.csv either).

shoppers: FF/TabbyFlow/Oracle all read from ef-vfm-diag/step1_decomp_long.csv (5 seeds
  each), a read-only historical directory (062/062reply correspondence, 2026-08-19).
  Independently re-verified before use (not taken on faith): recomputed mean/std for
  shoppers TabbyFlow dep_c2st from the 5 raw rows = +0.01664/0.01637, matching
  write/v22/main.tex line 289's "+0.017 +/- 0.016 (five seeds)" almost exactly; seed 0's
  xgb_c2st/marg_c2st (0.566813/0.523519) exactly match the single sample previously
  found at ef-vfm-dep/baselines/shoppers/samples.csv, confirming that file was just an
  unlabeled copy of seed 0 from this same 5-seed batch, not a separate/inconsistent run.
  This IS the batch that produced the paper's cited shoppers numbers.

  adult intentionally uses the CURRENT ef-vfm-fix pipeline (adult_base_s{0..4}), NOT
  ef-vfm-diag's adult rows -- the latter give dep=0.0532+/-0.0027, a DIFFERENT run/
  protocol vintage than the dep=0.050+/-0.007 already validated throughout 056-061 and
  actually cited in the paper's Table 1. Using ef-vfm-diag's adult number would
  introduce a NEW inconsistency with the already-published adult figure; using its
  shoppers number does not, because shoppers has no current-pipeline alternative and
  ef-vfm-diag's shoppers number is what the paper already cites (line 289/353).
  Each dataset therefore uses whichever source actually matches what is already
  published for it -- not a stylistic choice, and not "old vs new" as a general
  preference either way.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import load_info, evaluate_all, make_ff, make_oracle, self_check, DATASETS

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
SEEDS = [0, 1, 2, 3, 4]
SHOPPERS_HISTORICAL_SRC = "../ef-vfm-diag/step1_decomp_long.csv"  # read-only, import-only


def adult_rows():
    info = load_info("data/adult/info.json")
    train = pd.read_csv("data/adult/train.csv")
    test = pd.read_csv("data/adult/test.csv")
    rows = []
    for seed in SEEDS:
        syn = pd.read_csv(f"ef_vfm/result/adult/adult_base_s{seed}/8000/samples.csv")
        variants = {"FF": make_ff(train, len(syn), seed=seed), "TabbyFlow": syn,
                    "Oracle": make_oracle(train, len(syn), seed=seed)}
        for name, sdf in variants.items():
            m = evaluate_all(sdf, test, test, info, seed=seed, do_old=False, train_ref=train)
            rows.append(dict(dataset="adult", variant=name, seed=seed,
                              xgb_c2st=m["xgb_c2st"], marg_c2st=m["marg_c2st"], dep_c2st=m["dep_c2st"]))
    return rows


def shoppers_rows():
    """All three variants (FF/TabbyFlow/Oracle) read from the historical, read-only
    ef-vfm-diag/step1_decomp_long.csv (see module docstring for verification)."""
    hist = pd.read_csv(SHOPPERS_HISTORICAL_SRC)
    sub = hist[hist.dataset == "shoppers"][["variant", "seed", "xgb_c2st", "marg_c2st", "dep_c2st"]]
    rows = [dict(dataset="shoppers", **r) for r in sub.to_dict("records")]
    return rows


def main():
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
    rows = adult_rows() + shoppers_rows()
    long = pd.DataFrame(rows)
    long.to_csv("step_061b_gap_decomp_long.csv", index=False)

    g = long.groupby(["dataset", "variant"])[["xgb_c2st", "marg_c2st", "dep_c2st"]].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index()
    g.to_csv("step_061b_gap_decomp_summary.csv", index=False)
    print(g.to_string(index=False))

    # ---- figure (same visual design as make_figs.py::fig_gap_decomposition) ----
    df = g
    datasets = ["adult", "shoppers"]
    variants = ["FF", "TabbyFlow", "Oracle"]
    n_ds = len(datasets)
    width = 0.22
    x = np.arange(n_ds)
    fig, ax = plt.subplots(figsize=(8, 5))
    for vi, var in enumerate(variants):
        offsets = (vi - 1) * (width + 0.04) * np.ones(n_ds)
        segs_marg, segs_dep, errs_tot = [], [], []
        for ds in datasets:
            row = df[(df.dataset == ds) & (df.variant == var)]
            full_m = float(row.xgb_c2st_mean.iloc[0])
            marg_m = float(row.marg_c2st_mean.iloc[0])
            dep_m = float(row.dep_c2st_mean.iloc[0])
            full_s = float(row.xgb_c2st_std.iloc[0])
            segs_marg.append(max(0.0, marg_m - 0.5))
            segs_dep.append(max(0.0, dep_m))
            errs_tot.append(full_s)
        segs_marg = np.array(segs_marg); segs_dep = np.array(segs_dep); errs_tot = np.array(errs_tot)
        tops = segs_marg + segs_dep
        ax.bar(x + offsets, segs_marg, width, color="#4FC3F7", alpha=0.9,
               label="marginal gap" if vi == 0 else "")
        ax.bar(x + offsets, segs_dep, width, bottom=segs_marg, color="#EF5350", alpha=0.9,
               label="dependency gap" if vi == 0 else "")
        ax.errorbar(x + offsets, tops, yerr=errs_tot, fmt="none", ecolor="black", elinewidth=1.2, capsize=3)
        for xi, (top, err) in enumerate(zip(tops, errs_tot)):
            ax.text(x[xi] + offsets[xi], top + err + 0.005, var, ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["adult\n(test n=16,281)", "shoppers\n(test n=1,233)"])
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("XGB-C2ST discriminability above chance\n(gap = AUC - 0.5)", fontsize=10)
    ax.set_title("Gap Decomposition: marginal vs. dependency\n"
                 "(adult: current pipeline; shoppers: historical 5-seed run, ef-vfm-diag/)", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    marg_patch = mpatches.Patch(color="#4FC3F7", label="marginal gap (marg_c2st - 0.5)")
    dep_patch = mpatches.Patch(color="#EF5350", label="dependency gap (dep_c2st, clipped >= 0)")
    ax.legend(handles=[marg_patch, dep_patch], loc="upper right", fontsize=8)
    fig.tight_layout()
    os.makedirs("figs", exist_ok=True)
    fig.savefig("figs/fig_gap_decomposition_061.pdf")
    print("[SAVED] figs/fig_gap_decomposition_061.pdf")


if __name__ == "__main__":
    main()
