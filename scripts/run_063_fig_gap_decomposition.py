"""063: self-contained regeneration of fig_gap_decomposition.pdf, matching the ACTUAL
current figure (write/v22/figs/fig_gap_decomposition.pdf), which shows FOUR datasets
(ADULT/DEFAULT/BANK/MAGIC) x {FF, Tabby, Oracle} -- NOT adult+shoppers as the stale
ef-vfm-fix/make_figs.py::fig_gap_decomposition() / ef-vfm-diag/make_figs.py (same,
2-dataset) code would produce. Neither of those scripts is the true source of the
published figure; an exhaustive repo search found no other script matching the actual
4-dataset content either -- the original generator is genuinely not locatable. This is
a from-scratch reconstruction using the CURRENT, already-published 5-seed diagnostic
data for all four datasets (no historical/lost data needed here, unlike the shoppers
situation in fig_gap_decomposition's 2-dataset cousin / 061's mistaken attempt).

Data: step_{adult,default,bank}_diag_seed5_051_long.csv + step_magic_diag_seed5_long.csv
(the exact same files used throughout 056-061 for Table 1). magic's FF/Oracle rows are
the 5-seed values from 056 Task 0 (superseding the old 3-seed numbers baked into
whatever produced the currently-published magic bars).
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)

FILES = {
    "adult":   "step_adult_diag_seed5_051_long.csv",
    "default": "step_default_diag_seed5_051_long.csv",
    "bank":    "step_bank_diag_seed5_051_long.csv",
    "magic":   "step_magic_diag_seed5_long.csv",
}
DATASETS = ["adult", "default", "bank", "magic"]
VARIANTS = ["FF", "TabbyFlow", "Oracle"]  # display label "Tabby" in the figure only


def main():
    rows = []
    for ds, f in FILES.items():
        d = pd.read_csv(f)
        dep_col = "dep_c2st" if "dep_c2st" in d.columns else "dep"
        marg_col = "marg_c2st" if "marg_c2st" in d.columns else "marg"
        full_col = "xgb_c2st" if "xgb_c2st" in d.columns else "full"
        for var in VARIANTS:
            sub = d[d.variant == var]
            rows.append(dict(dataset=ds, variant=var, n=len(sub),
                              xgb_c2st_mean=sub[full_col].mean(), xgb_c2st_std=sub[full_col].std(ddof=1),
                              marg_c2st_mean=sub[marg_col].mean(), marg_c2st_std=sub[marg_col].std(ddof=1),
                              dep_c2st_mean=sub[dep_col].mean(), dep_c2st_std=sub[dep_col].std(ddof=1)))
    g = pd.DataFrame(rows)
    g.to_csv("step_063_gap_decomp_summary.csv", index=False)
    print(g.to_string(index=False))

    # ---- figure, matching the actual published style (uppercase labels, "Tabby",
    # legend inside upper area, minimal frame) ----
    display_var = {"FF": "FF", "TabbyFlow": "Tabby", "Oracle": "Oracle"}
    n_ds = len(DATASETS)
    n_var = len(VARIANTS)
    width = 0.22
    group_gap = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 3.2))  # matches original 518.4x230.4pt canvas (72pt/in)

    xticks, xticklabels = [], []
    for di, ds in enumerate(DATASETS):
        base_x = di * (n_var * width + group_gap)
        for vi, var in enumerate(VARIANTS):
            row = g[(g.dataset == ds) & (g.variant == var)].iloc[0]
            marg_seg = max(0.0, row.marg_c2st_mean - 0.5)
            dep_seg = max(0.0, row.dep_c2st_mean)
            err = row.xgb_c2st_std
            xi = base_x + vi * width
            ax.bar(xi, marg_seg, width * 0.92, color="#4C72B0",
                   label="marginal gap" if (di == 0 and vi == 0) else "")
            ax.bar(xi, dep_seg, width * 0.92, bottom=marg_seg, color="#C44E52",
                   label="dependency gap" if (di == 0 and vi == 0) else "")
            ax.errorbar(xi, marg_seg + dep_seg, yerr=err, fmt="none", ecolor="black",
                        elinewidth=1.0, capsize=2.5)
        xticks.append(base_x + width * (n_var - 1) / 2)
        xticklabels.append(ds.upper())

    ax.set_ylim(0.0, 0.55)
    y0, y1 = ax.get_ylim()
    sub_y = y0 - 0.028 * (y1 - y0)     # first tier: FF/Tabby/Oracle, just below axis
    ds_y = y0 - 0.11 * (y1 - y0)       # second tier: dataset name, further below
    for di, ds in enumerate(DATASETS):
        base_x = di * (n_var * width + group_gap)
        for vi, var in enumerate(VARIANTS):
            ax.annotate(display_var[var], (base_x + vi * width, sub_y),
                        ha="center", va="top", fontsize=7.5, annotation_clip=False)
        ax.annotate(ds.upper(), (base_x + width * (n_var - 1) / 2, ds_y),
                    ha="center", va="top", fontsize=11, annotation_clip=False)
    ax.set_xticks([])
    ax.set_ylabel("XGB-C2ST gap above chance\n(AUC $-$ 0.5)", fontsize=10)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.12))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs("figs", exist_ok=True)
    fig.savefig("figs/fig_gap_decomposition_063.pdf")
    print("[SAVED] figs/fig_gap_decomposition_063.pdf")


if __name__ == "__main__":
    main()
