"""Regenerate the two paper figures from the released CSVs (no training needed).

Figure 1 (fig_gap_decomposition): XGB-C2ST gap above chance, decomposed into
marginal + dependency segments, for FF / TabbyFlow / Oracle on four datasets.
Figure 2 (fig_metric_blindness): Trend / LR-C2ST / XGB-C2ST scores of the
zero-dependency FF reference on the four blindness-panel datasets.

Usage:  python make_paper_figs.py     (writes figs/fig_*.png/.pdf)
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "results"

# ---------- Figure 1: gap decomposition ----------
data = {}
for ds in ["adult", "default", "bank"]:
    d = pd.read_csv(f"{RES}/step_{ds}_diag_seed5_051_long.csv")
    data[ds] = d.groupby("variant").agg(full=("full", "mean"),
                                        fstd=("full", "std"),
                                        dep=("dep", "mean"))
# magic: FF/Oracle anchors are the 3-seed rows of the 047 multidataset run;
# the TabbyFlow trio uses the 5-seed capacity-sweep 1x rows (consistent with Table 1).
m = pd.read_csv(f"{RES}/step_multidataset_diag.csv")
mm = m[m.dataset == "magic"].set_index("variant")
g = pd.DataFrame({"full": mm["xgb_c2st_mean"], "fstd": mm["xgb_c2st_std"],
                  "dep": mm["dep_c2st_mean"]})
c = pd.read_csv(f"{RES}/step_capacity_sweep5_051_long.csv")
c1 = c[(c.dataset == "magic") & (c.capacity == "1x")]
g.loc["TabbyFlow", ["full", "fstd", "dep"]] = [c1["full"].mean(), c1["full"].std(), c1["dep"].mean()]
data["magic"] = g

variants = ["FF", "TabbyFlow", "Oracle"]
fig, ax = plt.subplots(figsize=(7.2, 3.2))
W, xs = 0.25, np.arange(len(data))
for i, v in enumerate(variants):
    marg, dep, err = [], [], []
    for ds in data:
        row = data[ds].loc[v]
        d_ = max(row["dep"], 0)
        marg.append(max(row["full"] - 0.5 - d_, 0)); dep.append(d_); err.append(row["fstd"])
    x = xs + (i - 1) * W
    ax.bar(x, marg, W, color="#4878CF", edgecolor="white",
           label="marginal gap" if i == 0 else None)
    ax.bar(x, dep, W, bottom=marg, color="#D65F5F", edgecolor="white",
           label="dependency gap" if i == 0 else None)
    ax.errorbar(x, np.array(marg) + np.array(dep), yerr=err, fmt="none",
                ecolor="black", capsize=2, lw=1)
for i, ds in enumerate(data):
    for j, v in enumerate(variants):
        ax.text(i + (j - 1) * W, -0.025, "Tabby" if v == "TabbyFlow" else v,
                ha="center", va="top", fontsize=6.5)
ax.set_xticks(xs); ax.set_xticklabels([d.upper() for d in data], fontsize=9)
ax.tick_params(axis="x", pad=14)
ax.set_ylabel("XGB-C2ST gap above chance\n(AUC $-$ 0.5)", fontsize=9)
ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.62, 0.98))
ax.set_ylim(0, 0.55); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
for ext in ["pdf", "png"]:
    plt.savefig(f"figs/fig_gap_decomposition.{ext}", dpi=300)
plt.close()

# ---------- Figure 2: metric blindness ----------
# adult/shoppers values come from the original diagnostic study (step1_decomp);
# bank/magic from the 051 blindness runs.
vals = {"adult": (0.764, 0.999, 0.991), "shoppers": (0.770, 1.000, 0.976)}
for ds in ["bank", "magic"]:
    b = pd.read_csv(f"{RES}/step_{ds}_blindness_051_long.csv")
    f = b[b.variant == "FF"]
    vals[ds] = (f["trend"].mean(), f["lr_c2st"].mean(), f["xgb_c2st"].mean())
labels = ["Trend (1.0 = perfect)", "LR-C2ST (1.0 = indistinguishable)",
          "XGB-C2ST (1.0 = fully distinguishable)"]
cols = ["#B0B0B0", "#D65F5F", "#4878CF"]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
xs, W = np.arange(len(vals)), 0.26
for k in range(3):
    ax.bar(xs + (k - 1) * W, [vals[d][k] for d in vals], W,
           color=cols[k], label=labels[k], edgecolor="white")
ax.axhline(1.0, color="black", lw=0.6, ls=":")
ax.set_xticks(xs); ax.set_xticklabels([d.upper() for d in vals], fontsize=9)
ax.set_ylabel("score of the zero-dependency\nFF reference", fontsize=9)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
for ext in ["pdf", "png"]:
    plt.savefig(f"figs/fig_metric_blindness.{ext}", dpi=300)
print("figures written to figs/")
