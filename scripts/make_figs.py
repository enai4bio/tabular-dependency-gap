"""040 腿二: 招牌图 + 指标盲区图。从 step1_decomp_summary.csv 读取。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

SUMM_PATH = os.path.join(BASE, "step1_decomp_summary.csv")


def save(fig, name):
    for ext in ["png", "pdf"]:
        p = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  saved figs/{name}.png  figs/{name}.pdf")


def fig_gap_decomposition(df):
    """招牌图: adult + shoppers 各三根堆叠条形 (边际段 + 依赖段) + 误差棒。"""
    datasets = ["adult", "shoppers"]
    variants = ["FF", "TabbyFlow", "Oracle"]
    colors_marg = ["#2196F3", "#2196F3", "#2196F3"]   # blue for marginal
    colors_dep  = ["#F44336", "#F44336", "#F44336"]   # red for dependency

    n_ds = len(datasets)
    n_var = len(variants)
    width = 0.22
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(8, 5))

    for vi, var in enumerate(variants):
        offsets = (vi - 1) * (width + 0.04) * np.ones(n_ds)
        segs_marg, segs_dep, errs_tot = [], [], []
        for ds in datasets:
            row = df[(df.dataset == ds) & (df.variant == var)]
            if row.empty:
                segs_marg.append(0); segs_dep.append(0); errs_tot.append(0)
                continue
            full_m = float(row.xgb_c2st_mean)
            marg_m = float(row.marg_c2st_mean)
            dep_m  = float(row.dep_c2st_mean)
            full_s = float(row.xgb_c2st_std)

            seg_marg = max(0.0, marg_m - 0.5)
            seg_dep  = max(0.0, dep_m)          # clip to 0
            segs_marg.append(seg_marg)
            segs_dep.append(seg_dep)
            errs_tot.append(full_s)

        segs_marg = np.array(segs_marg)
        segs_dep  = np.array(segs_dep)
        errs_tot  = np.array(errs_tot)
        tops = segs_marg + segs_dep

        # marginal segment (bottom)
        ax.bar(x + offsets, segs_marg, width, color="#4FC3F7", alpha=0.9,
               label="marginal gap" if vi == 0 else "")
        # dependency segment (stacked on top)
        ax.bar(x + offsets, segs_dep, width, bottom=segs_marg, color="#EF5350", alpha=0.9,
               label="dependency gap" if vi == 0 else "")
        # error bar on total
        ax.errorbar(x + offsets, tops, yerr=errs_tot,
                    fmt="none", ecolor="black", elinewidth=1.2, capsize=3)
        # label above bar
        for xi, (top, err) in enumerate(zip(tops, errs_tot)):
            ax.text(x[xi] + offsets[xi], top + err + 0.005, var,
                    ha="center", va="bottom", fontsize=7.5, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(test n={1 if d=='adult' else 1})" for d in datasets])
    # fix nice labels
    ax.set_xticklabels(["adult\n(test n=16,281)", "shoppers\n(test n=1,233)"])
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("XGB-C2ST discriminability above chance\n(gap = AUC − 0.5)", fontsize=10)
    ax.set_title("Gap Decomposition: marginal vs. dependency\n"
                 "TabbyFlow residual gap is ~50% dependency in adult\n"
                 "(shoppers dep_cst noisy at n=1,233 — see report)", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    marg_patch = mpatches.Patch(color="#4FC3F7", label="marginal gap (marg_c2st − 0.5)")
    dep_patch  = mpatches.Patch(color="#EF5350", label="dependency gap (dep_c2st, clipped ≥ 0)")
    ax.legend(handles=[marg_patch, dep_patch], loc="upper right", fontsize=8)

    # annotation: oracle ≈ 0, FF ≈ full dep
    for xi, ds in enumerate(datasets):
        for var in ["Oracle", "FF"]:
            row = df[(df.dataset == ds) & (df.variant == var)]
            if not row.empty:
                full_m = float(row.xgb_c2st_mean)
                dep_m  = float(row.dep_c2st_mean)
                seg_marg = max(0.0, float(row.marg_c2st_mean) - 0.5)
                seg_dep  = max(0.0, dep_m)
                top = seg_marg + seg_dep
                if var == "Oracle":
                    ax.annotate(f"oracle\n≈{full_m:.3f}", xy=(xi - width - 0.04, top),
                                xytext=(xi - 0.42, top + 0.04),
                                fontsize=6, color="gray",
                                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    fig.tight_layout()
    save(fig, "fig_gap_decomposition")
    plt.close(fig)


def fig_metric_blindness(df):
    """盲区图: FF / TabbyFlow / Oracle × 三个指标对比。"""
    datasets = ["adult", "shoppers"]
    variants  = ["FF", "TabbyFlow", "Oracle"]
    metrics_info = [
        # (column_mean, column_std, label, direction)
        ("xgb_c2st_mean", "xgb_c2st_std", "XGB-C2ST\n(lower = better)", "lower"),
        ("trend_mean",    "trend_std",     "SDMetrics Trend\n(higher = better)", "higher"),
        ("lr_c2st_mean",  "lr_c2st_std",   "LR-C2ST\n(higher = better)", "higher"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    colors = {"FF": "#EF5350", "TabbyFlow": "#42A5F5", "Oracle": "#66BB6A"}

    for ai, ds in enumerate(datasets):
        ax = axes[ai]
        n_var = len(variants)
        n_met = len(metrics_info)
        x = np.arange(n_met)
        width = 0.22

        for vi, var in enumerate(variants):
            row = df[(df.dataset == ds) & (df.variant == var)]
            vals, errs = [], []
            for col_m, col_s, *_ in metrics_info:
                if row.empty or col_m not in row.columns:
                    vals.append(float("nan")); errs.append(0)
                else:
                    vals.append(float(row[col_m]))
                    std_val = float(row[col_s]) if col_s in row.columns else 0
                    errs.append(std_val)

            offset = (vi - 1) * (width + 0.03)
            bars = ax.bar(x + offset, vals, width,
                          color=colors[var], alpha=0.85,
                          label=var if ai == 0 else "")
            ax.errorbar(x + offset, vals, yerr=errs,
                        fmt="none", ecolor="black", elinewidth=1.0, capsize=2)

        ax.set_xticks(x)
        ax.set_xticklabels([m[2] for m in metrics_info], fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Metric score", fontsize=9)
        ax.set_title(f"{ds}\n(test n={'16,281' if ds == 'adult' else '1,233'})", fontsize=10)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="C2ST null (0.5)")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    # shared legend on first axis
    handles = [mpatches.Patch(color=c, label=v) for v, c in colors.items()]
    handles.append(plt.Line2D([0], [0], color="gray", linestyle="--", lw=0.8, label="C2ST null (0.5)"))
    axes[0].legend(handles=handles, loc="lower right", fontsize=8)

    fig.suptitle(
        "Metric Blindness: Trend & LR-C2ST give FF (zero-dependency) high scores;\n"
        "only XGB-C2ST correctly detects FF's near-perfect discriminability",
        fontsize=9, y=1.02)
    fig.tight_layout()
    save(fig, "fig_metric_blindness")
    plt.close(fig)


def print_key_numbers(df):
    """Print key numbers for the results report."""
    print("\n" + "=" * 70)
    print("KEY NUMBERS FOR REPORT")
    print("=" * 70)
    for ds in ["adult", "shoppers"]:
        print(f"\n{ds}:")
        for var in ["FF", "TabbyFlow", "Oracle"]:
            row = df[(df.dataset == ds) & (df.variant == var)]
            if row.empty: continue
            full_m = float(row.xgb_c2st_mean)
            full_s = float(row.xgb_c2st_std)
            dep_m  = float(row.dep_c2st_mean)
            dep_s  = float(row.dep_c2st_std)
            gap    = full_m - 0.5
            ratio  = dep_m / gap if gap > 0.001 else float("nan")
            f1_m   = float(row.mle_f1_mean)
            rec_m  = float(row.mle_recall_mean) if "mle_recall_mean" in row.columns else float("nan")
            lr_m   = float(row.lr_c2st_mean)
            tr_m   = float(row.trend_mean)
            print(f"  {var:10s} full={full_m:.4f}±{full_s:.4f}  dep={dep_m:+.4f}±{dep_s:.4f}  "
                  f"ratio={ratio:.2f}  F1={f1_m:.4f}  recall={rec_m:.4f}  "
                  f"lr={lr_m:.4f}  trend={tr_m:.4f}")


def main():
    os.chdir(BASE)
    df = pd.read_csv(SUMM_PATH)
    print(f"Loaded summary: {len(df)} rows, columns: {list(df.columns)}")

    print_key_numbers(df)

    print("\nGenerating fig_gap_decomposition ...")
    fig_gap_decomposition(df)

    print("Generating fig_metric_blindness ...")
    fig_metric_blindness(df)

    print("\nDone. Figures in ef-vfm-diag/figs/")


if __name__ == "__main__":
    main()
