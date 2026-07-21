"""048 Exp2 -- order check: is the residual dependency gap 2nd-order or higher-order?

2a  Direct moment matching (cheapest; reuse existing TabbyFlow baseline samples):
    for every (num_i, cat_j=c): within-class num mean and corr(num_i, 1[cat_j=c]).
    Compare real_test vs {TabbyFlow, Oracle, FF}. Aggregate = mean abs standardized
    difference. If TabbyFlow's 2nd-order cross-moments already match real (gap ~ Oracle
    noise floor, << FF), the residual full-C2ST gap must be higher-order.

2b  GaussCopula headroom (supplement): fit a Gaussian copula on real TRAIN (captures ALL
    2nd-order structure; categoricals one-hot -> correlation, APPROXIMATE), then
    XGB-C2ST(real_test, GaussCopula) alongside FF/Oracle/TabbyFlow.
    GC C2ST ~ oracle(0.5) => 2nd-order dominates; still >0.5 => higher-order dominates.

Datasets: adult + default. Neutral output only (numbers, no structure/verdict).
Outputs: step_order.csv (both sub-experiments, mean+/-std).
"""
import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import (load_info, xgb_c2st, make_ff, make_oracle,
                            cat_columns, num_columns, self_check, DATASETS)

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

DS_CFG = {
    "adult":   {"train": "data/adult/train.csv",   "test": "data/adult/test.csv",   "info": "data/adult/info.json"},
    "default": {"train": "data/default/train.csv", "test": "data/default/test.csv", "info": "data/default/info.json"},
}
# 2a reuses existing TabbyFlow 1x samples (no retrain): adult has 5 seeds, default 3.
TABBY_SEEDS = {"adult": [0, 1, 2, 3, 4], "default": [0, 1, 2]}
COPULA_SEEDS = [0, 1, 2, 3, 4]           # 2b: copula generation seeds (5 both ds)
MIN_CLASS_N = 30                          # skip cat categories with <30 real_test support


def tabby_sample(ds, seed):
    p = f"ef_vfm/result/{ds}/{ds}_base_s{seed}/8000/samples.csv"
    return p if os.path.exists(p) else None


# ---------- 2a: cross-moment matching ----------
def cross_moment_vectors(df, num_idx, cat_idx, cats_support):
    """Return two dicts keyed by (i,j,c): within-class mean of num_i, and corr(num_i,1[cat_j=c])."""
    df = df.copy(); df.columns = range(len(df.columns))
    means, corrs = {}, {}
    for i in num_idx:
        xi = pd.to_numeric(df[i], errors="coerce").values.astype(float)
        for j in cat_idx:
            cj = df[j].astype(str).values
            for c in cats_support[j]:
                mask = (cj == c)
                if mask.sum() < 2:
                    means[(i, j, c)] = np.nan; corrs[(i, j, c)] = np.nan; continue
                means[(i, j, c)] = float(np.nanmean(xi[mask]))
                ind = mask.astype(float)
                if np.std(ind) < 1e-9 or np.nanstd(xi) < 1e-9:
                    corrs[(i, j, c)] = np.nan
                else:
                    corrs[(i, j, c)] = float(np.corrcoef(np.nan_to_num(xi, nan=np.nanmean(xi)), ind)[0, 1])
    return means, corrs


def moment_gap(real_df, syn_df, info):
    """Mean abs standardized within-class-mean diff and mean abs corr diff (syn vs real)."""
    num_idx = num_columns(info); cat_idx = cat_columns(info)
    r = real_df.copy(); r.columns = range(len(r.columns))
    s = syn_df.copy(); s.columns = range(len(s.columns))
    # categories with enough support in REAL
    cats_support = {}
    for j in cat_idx:
        vc = pd.Series(r[j].astype(str)).value_counts()
        cats_support[j] = [c for c, n in vc.items() if n >= MIN_CLASS_N]
    # real num std for standardization
    num_std = {i: float(np.nanstd(pd.to_numeric(r[i], errors="coerce"))) or 1.0 for i in num_idx}
    rm, rc = cross_moment_vectors(r, num_idx, cat_idx, cats_support)
    sm, sc = cross_moment_vectors(s, num_idx, cat_idx, cats_support)
    mean_diffs, corr_diffs = [], []
    for k in rm:
        if np.isnan(rm[k]) or np.isnan(sm[k]):
            continue
        i = k[0]
        mean_diffs.append(abs(sm[k] - rm[k]) / (num_std[i] if num_std[i] > 1e-9 else 1.0))
        if not (np.isnan(rc[k]) or np.isnan(sc[k])):
            corr_diffs.append(abs(sc[k] - rc[k]))
    return (float(np.mean(mean_diffs)) if mean_diffs else float("nan"),
            float(np.mean(corr_diffs)) if corr_diffs else float("nan"),
            len(mean_diffs))


def run_2a():
    rows = []
    for ds, cfg in DS_CFG.items():
        info = load_info(cfg["info"]); train = pd.read_csv(cfg["train"]); test = pd.read_csv(cfg["test"])
        for seed in TABBY_SEEDS[ds]:
            p = tabby_sample(ds, seed)
            if p is None:
                warnings.warn(f"{ds} s{seed}: no TabbyFlow sample"); continue
            syn = pd.read_csv(p)
            variants = {"TabbyFlow": syn,
                        "Oracle": make_oracle(train, len(syn), seed=seed),
                        "FF": make_ff(train, len(syn), seed=seed)}
            for name, sdf in variants.items():
                md, cd, npairs = moment_gap(test, sdf, info)
                rows.append(dict(dataset=ds, exp="2a_moment", variant=name, seed=seed,
                                 mean_gap=md, corr_gap=cd, n_pairs=npairs))
                print(f"  2a {ds:8s} {name:10s} s{seed}  mean_gap={md:.4f} corr_gap={cd:.4f} (pairs={npairs})")
    return rows


# ---------- 2b: GaussCopula headroom ----------
def run_2b():
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer
    rows = []
    for ds, cfg in DS_CFG.items():
        info = load_info(cfg["info"]); train = pd.read_csv(cfg["train"]); test = pd.read_csv(cfg["test"])
        train_named = train.copy(); test_named = test.copy()
        train_named.columns = [str(c) for c in train_named.columns]
        test_named.columns = [str(c) for c in test_named.columns]
        cats = set(cat_columns(info))
        md = SingleTableMetadata()
        md.detect_from_dataframe(train_named)
        num_dists = {}
        for ci, col in enumerate(train_named.columns):
            is_cat = ci in cats
            md.update_column(col, sdtype="categorical" if is_cat else "numerical")
            if not is_cat:
                num_dists[col] = "gaussian_kde"   # 048: KDE marginals to avoid long-tail (capital-gain) blowup
        try:
            synth = GaussianCopulaSynthesizer(md, numerical_distributions=num_dists)
            synth.fit(train_named)
        except Exception as e:
            print(f"[STOP] GaussCopula fit failed on {ds}: {e}")
            raise
        for seed in COPULA_SEEDS:
            try:
                np.random.seed(seed)
                gc = synth.sample(num_rows=len(test))
            except Exception as e:
                print(f"[STOP] GaussCopula sample failed on {ds} seed {seed}: {e}")
                raise
            gc.columns = range(len(gc.columns))
            c2st = xgb_c2st(test, gc, info, seed=seed)
            rows.append(dict(dataset=ds, exp="2b_gausscopula", variant="GaussCopula", seed=seed, c2st=c2st))
            print(f"  2b {ds:8s} GaussCopula s{seed}  C2ST={c2st:.4f}")
        # reference rows: FF / Oracle / TabbyFlow full C2ST (same real_test ref)
        for seed in COPULA_SEEDS:
            ff = make_ff(train, len(test), seed=seed)
            orc = make_oracle(train, len(test), seed=seed)
            rows.append(dict(dataset=ds, exp="2b_gausscopula", variant="FF", seed=seed,
                             c2st=xgb_c2st(test, ff, info, seed=seed)))
            rows.append(dict(dataset=ds, exp="2b_gausscopula", variant="Oracle", seed=seed,
                             c2st=xgb_c2st(test, orc, info, seed=seed)))
        for seed in TABBY_SEEDS[ds]:
            p = tabby_sample(ds, seed)
            if p is None: continue
            syn = pd.read_csv(p)
            rows.append(dict(dataset=ds, exp="2b_gausscopula", variant="TabbyFlow", seed=seed,
                             c2st=xgb_c2st(test, syn, info, seed=seed)))
        print(f"  -- {ds} 2b done --")
    return rows


def main():
    print("=" * 66); print("048 Exp2 order check (2a moment match + 2b GaussCopula)"); print("=" * 66)
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
    rows = run_2a() + run_2b()
    long = pd.DataFrame(rows)
    long.to_csv("step_order_long.csv", index=False)

    # summaries
    a = long[long.exp == "2a_moment"].groupby(["dataset", "variant"])[["mean_gap", "corr_gap"]].agg(["mean", "std", "count"])
    a.columns = [f"{x}_{y}" for x, y in a.columns]; a = a.reset_index(); a["exp"] = "2a_moment"
    b = long[long.exp == "2b_gausscopula"].groupby(["dataset", "variant"])[["c2st"]].agg(["mean", "std", "count"])
    b.columns = [f"{x}_{y}" for x, y in b.columns]; b = b.reset_index(); b["exp"] = "2b_gausscopula"
    out = pd.concat([a, b], ignore_index=True)
    out.to_csv("step_order.csv", index=False)
    print("\n[SAVED] step_order.csv (+ _long)")
    print("\n=== 2a moment gap (mean abs standardized diff, lower=more aligned) ===")
    print(a.to_string(index=False))
    print("\n=== 2b GaussCopula headroom (XGB-C2ST, 0.5=indistinguishable) ===")
    print(b.to_string(index=False))
    print("\nDONE.")


if __name__ == "__main__":
    main()
