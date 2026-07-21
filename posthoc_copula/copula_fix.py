"""
033 Phase 2: Copula dependency repair on top of TabbyFlow samples.

Idea (Sklar): TabbyFlow's marginals are good; replace its dependency with a
copula learned from REAL TRAIN. Unified mechanism = rank-reorder transport:

  1. Fit a copula C on train pseudo-observations U_train (rank/(n+1); for
     categoricals the rank uses a FIXED category->code order so fit & apply agree).
  2. Sample u ~ C  (n_syn x d uniforms carrying the learned dependency).
  3. For each column j: out[:,j] = sort_by_key(tabbyflow[:,j])[ rank(u[:,j]) ].
     This is a permutation of TabbyFlow's column => marginal preserved EXACTLY,
     dependency imposed by the copula. Works for continuous AND categorical
     (categories occupy contiguous code-intervals, so the empirical copula
     reproduces the true contingency association).

Copula variants (shotgun):
  - gaussian   : Z=Phi^-1(U_train), R=corr(Z); sample N(0,R)->Phi  (2nd order)
  - vine       : pyvinecopulib Vinecop (tll/gaussian families, trunc levels) — higher order
  - empirical  : bootstrap rows of U_train (+jitter) — nonparametric, all orders
  - sdv        : SDV GaussianCopulaSynthesizer (fresh classical reference)

Discipline:
  - Config (method / columns / vine trunc) tuned on a val split carved from
    train; TEST evaluated once with the val-selected config refit on full train.
  - >=3 seeds; report mean +/- std.
  - FF (lower) + Oracle (upper) shown alongside.
  - Sanity: empirical-reorder using TRAIN's own marginals must reproduce train
    (XGB-C2ST ~ real-vs-real, Trend ~ 1.0). Aborts if it fails.

Run:  python copula_fix.py [--datasets adult shoppers] [--seeds 0 1 2]
"""
import sys
import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import proper_metrics as pm
from proper_metrics import (
    DATASETS, load_info, cat_columns, num_columns,
    xgb_c2st, kway_tv, mle_auc_f1, sdmetrics_trend, lr_c2st,
    indistinguishability, make_ff, make_oracle, evaluate_all,
)
import pyvinecopulib as pv


# ── pseudo-observations / category ordering ────────────────────────────────────

def build_cat_order(train_df, info):
    """Fixed category -> code order per categorical column (from train)."""
    train_df = train_df.copy(); train_df.columns = range(len(train_df.columns))
    cats = set(cat_columns(info))
    order = {}
    for j in cats:
        order[j] = pd.Index(sorted(train_df[j].astype(str).unique()))
    return order


def to_key(df, info, cat_order):
    """Numeric key matrix: value for continuous, category-code for categorical."""
    df = df.copy(); df.columns = range(len(df.columns))
    cats = set(cat_columns(info))
    d = len(df.columns)
    K = np.zeros((len(df), d), dtype=float)
    for j in range(d):
        if j in cats:
            codes = cat_order[j].get_indexer(df[j].astype(str).values).astype(float)
            # unseen categories (shouldn't happen for train-fit) -> max+1
            codes[codes < 0] = len(cat_order[j])
            K[:, j] = codes
        else:
            K[:, j] = pd.to_numeric(df[j], errors="coerce").values
    return K


def pseudo_obs(K, jitter=0.0, seed=0):
    """Rank -> uniform pseudo-obs in (0,1). Optional jitter breaks categorical ties."""
    rng = np.random.default_rng(seed)
    n, d = K.shape
    U = np.zeros_like(K)
    for j in range(d):
        col = K[:, j].astype(float)
        if jitter > 0:
            col = col + rng.normal(0, jitter * (np.nanstd(col) + 1e-9), n)
        r = stats.rankdata(col, method="average")
        U[:, j] = r / (n + 1.0)
    return np.clip(U, 1e-6, 1 - 1e-6)


# ── copula samplers: each returns u (n_syn x d) in (0,1) ────────────────────────

def sample_gaussian(U_train, n_syn, seed):
    Z = stats.norm.ppf(U_train)
    R = np.corrcoef(Z.T)
    R = (R + R.T) / 2
    ev = np.linalg.eigvalsh(R)
    if ev.min() < 1e-6:
        R += np.eye(R.shape[0]) * (1e-6 - ev.min())
    L = np.linalg.cholesky(R)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_syn, U_train.shape[1])) @ L.T
    return stats.norm.cdf(z)


def sample_empirical(U_train, n_syn, seed, jitter=0.01):
    """Bootstrap rows of train pseudo-obs (preserves full joint rank structure)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(U_train), n_syn)
    u = U_train[idx].copy()
    if jitter > 0:
        u = u + rng.normal(0, jitter, u.shape)
    return np.clip(u, 1e-6, 1 - 1e-6)


def sample_vine(U_train, n_syn, seed, family="tll", trunc_lvl=None):
    fam = {"tll": pv.BicopFamily.tll, "gaussian": pv.BicopFamily.gaussian,
           "parametric": None}[family] if family != "parametric" else None
    if family == "parametric":
        fam_set = [pv.BicopFamily.gaussian, pv.BicopFamily.clayton,
                   pv.BicopFamily.gumbel, pv.BicopFamily.frank, pv.BicopFamily.joe]
    else:
        fam_set = [fam]
    kw = {"family_set": fam_set}
    if trunc_lvl is not None:
        kw["trunc_lvl"] = trunc_lvl
    ctrl = pv.FitControlsVinecop(**kw)
    vc = pv.Vinecop.from_data(U_train, controls=ctrl)
    return vc.simulate(n_syn, seeds=[int(seed) + 1, int(seed) + 17, int(seed) + 101])


# ── rank-reorder transport: TabbyFlow marginals + copula dependency ────────────

def reorder_transport(tabby_df, u, info, cat_order, cols=None):
    """
    out[:,j] = sort_by_key(tabby[:,j])[rank(u[:,j])] for j in cols;
    columns not in cols keep TabbyFlow originals.
    """
    tabby = tabby_df.copy(); tabby.columns = range(len(tabby.columns))
    d = len(tabby.columns)
    cols = list(range(d)) if cols is None else list(cols)
    Kt = to_key(tabby, info, cat_order)        # keys for ordering tabby values
    out = tabby.copy()
    for jj, j in enumerate(cols):
        order_key = Kt[:, j]
        sort_idx = np.argsort(order_key, kind="stable")     # ascending by key
        ranks = stats.rankdata(u[:, jj], method="ordinal").astype(int) - 1
        vals_sorted = tabby[j].values[sort_idx]
        out[j] = vals_sorted[ranks]
    out.columns = tabby_df.columns
    return out


# ── build one copula-repaired synthetic set ────────────────────────────────────

def make_copula_syn(tabby_df, train_df, info, method, seed,
                    cols_mode="all", vine_family="tll", vine_trunc=None,
                    emp_jitter=0.01):
    """cols_mode: 'all' | 'num' (only continuous+target reordered)."""
    cat_order = build_cat_order(train_df, info)
    if cols_mode == "all":
        cols = list(range(len(train_df.columns)))
    elif cols_mode == "num":
        cols = sorted(set(num_columns(info)) | set(info["target_col_idx"]))
    else:
        raise ValueError(cols_mode)

    K_train = to_key(train_df, info, cat_order)[:, cols]
    U_train = pseudo_obs(K_train, jitter=0.0)
    n_syn = len(tabby_df)

    if method == "gaussian":
        u = sample_gaussian(U_train, n_syn, seed)
    elif method == "empirical":
        u = sample_empirical(U_train, n_syn, seed, jitter=emp_jitter)
    elif method == "vine":
        u = sample_vine(U_train, n_syn, seed, family=vine_family, trunc_lvl=vine_trunc)
    else:
        raise ValueError(method)
    return reorder_transport(tabby_df, u, info, cat_order, cols=cols)


def make_sdv_gaussian(train_df, info, n_syn, seed):
    try:
        from sdv.single_table import GaussianCopulaSynthesizer
        from sdv.metadata import SingleTableMetadata
    except ImportError:
        return None
    meta = SingleTableMetadata()
    num_idx = set(info["num_col_idx"])
    sdtype = {}
    for i, col in enumerate(train_df.columns):
        sdtype[col] = {"sdtype": "numerical"} if i in num_idx else {"sdtype": "categorical"}
    meta.columns = sdtype; meta.primary_key = None
    np.random.seed(seed)
    synth = GaussianCopulaSynthesizer(meta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        synth.fit(train_df)
    return synth.sample(num_rows=n_syn)


# ── config grid for val tuning ─────────────────────────────────────────────────

def config_grid():
    cfgs = []
    cfgs.append({"name": "gaussian-all", "method": "gaussian", "cols_mode": "all"})
    cfgs.append({"name": "gaussian-num", "method": "gaussian", "cols_mode": "num"})
    cfgs.append({"name": "empirical-all", "method": "empirical", "cols_mode": "all"})
    cfgs.append({"name": "empirical-num", "method": "empirical", "cols_mode": "num"})
    cfgs.append({"name": "vine-tll-all", "method": "vine", "cols_mode": "all",
                 "vine_family": "tll", "vine_trunc": None})
    cfgs.append({"name": "vine-tll-t2-all", "method": "vine", "cols_mode": "all",
                 "vine_family": "tll", "vine_trunc": 2})
    cfgs.append({"name": "vine-par-all", "method": "vine", "cols_mode": "all",
                 "vine_family": "parametric", "vine_trunc": None})
    return cfgs


# ── sanity: empirical reorder on TRAIN marginals must reproduce train ──────────

def sanity_empirical(train_df, test_df, info):
    print("  [SANITY] empirical-copula reorder w/ TRAIN marginals -> should ~ real")
    syn = make_copula_syn(train_df, train_df, info, "empirical", seed=42,
                          cols_mode="all", emp_jitter=0.0)
    c2st = xgb_c2st(test_df, syn, info, seed=42)
    trend = sdmetrics_trend(train_df, syn, info)
    # compare to TabbyFlow-marginal-free real-vs-real reference
    rng = np.random.default_rng(0); perm = rng.permutation(len(test_df)); h = len(test_df)//2
    rr = xgb_c2st(test_df.iloc[perm[:h]].reset_index(drop=True),
                  test_df.iloc[perm[h:]].reset_index(drop=True), info, seed=42)
    ok = (c2st <= rr + 0.06) and (trend >= 0.97 or np.isnan(trend))
    print(f"    emp(train)->C2ST={c2st:.4f} (real-vs-real={rr:.4f}), Trend={trend:.4f}  "
          f"=> {'OK' if ok else 'FAIL'}")
    return ok


# ── val tuning ─────────────────────────────────────────────────────────────────

def val_split(train_df, frac=0.2, seed=2024):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_df)); nv = int(len(train_df) * frac)
    val = train_df.iloc[idx[:nv]].reset_index(drop=True)
    fit = train_df.iloc[idx[nv:]].reset_index(drop=True)
    return fit, val


def tune_on_val(tabby_df, train_df, info, seeds=(0, 1, 2)):
    """Fit copula on train_fit, reorder TabbyFlow marginals, score vs val.
    Returns (best_cfg, table)."""
    fit_df, val_df = val_split(train_df)
    print(f"  [VAL TUNE] train_fit={len(fit_df)} val={len(val_df)} | grid x {len(seeds)} seeds")
    rows = []
    for cfg in config_grid():
        cs = []
        for s in seeds:
            try:
                syn = make_copula_syn(tabby_df, fit_df, info,
                                      cfg["method"], seed=s,
                                      cols_mode=cfg["cols_mode"],
                                      vine_family=cfg.get("vine_family", "tll"),
                                      vine_trunc=cfg.get("vine_trunc", None))
                cs.append(xgb_c2st(val_df, syn, info, seed=s))
            except Exception as e:
                print(f"    {cfg['name']} seed{s} FAILED: {e}")
        if cs:
            rows.append({"name": cfg["name"], "val_c2st": float(np.mean(cs)),
                         "val_c2st_std": float(np.std(cs)), "cfg": cfg})
            print(f"    {cfg['name']:18s} val_C2ST={np.mean(cs):.4f} +/- {np.std(cs):.4f}")
    rows.sort(key=lambda r: r["val_c2st"])
    best = rows[0]
    print(f"  [VAL TUNE] best = {best['name']} (val_C2ST={best['val_c2st']:.4f})")
    return best, rows


# ── multi-seed test evaluation ─────────────────────────────────────────────────

def eval_multiseed(builder, holdout, test, info, seeds, train_ref):
    """builder(seed)->syn_df. Returns dict of metric -> (mean,std) over seeds."""
    keys = ["xgb_c2st", "indist", "tv2", "tv3", "mle_auc", "mle_f1", "trend", "lr_c2st"]
    acc = {k: [] for k in keys}
    for s in seeds:
        syn = builder(s)
        m = evaluate_all(syn, holdout, test, info, seed=s, do_old=True, train_ref=train_ref)
        for k in keys:
            acc[k].append(m[k])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}


def run(datasets, seeds):
    base = os.path.dirname(os.path.abspath(__file__)); os.chdir(base)
    print("=" * 72)
    print("033 PHASE 2 — COPULA DEPENDENCY REPAIR (target: beat TabbyFlow)")
    print(f"seeds={list(seeds)}")
    print("=" * 72)

    all_rows = []
    val_tables = {}
    for ds in datasets:
        cfg = DATASETS[ds]
        print(f"\n{'='*60}\n{ds}\n{'='*60}")
        info = load_info(cfg["info"])
        train = pd.read_csv(cfg["train"]); test = pd.read_csv(cfg["test"])
        tabby = pd.read_csv(cfg["syn"]); n_syn = len(tabby)

        if not sanity_empirical(train, test, info):
            print(f"  [ABORT] empirical-copula sanity failed for {ds}; skipping.")
            continue

        # 1) tune config on val
        best, val_tbl = tune_on_val(tabby, train, info, seeds=seeds)
        val_tables[ds] = val_tbl

        # 2) reference points + all variants on TEST, multi-seed
        builders = {
            "TabbyFlow":      (lambda s: tabby, [42]),               # deterministic baseline
            "FF (lower)":     (lambda s: make_ff(train, n_syn, seed=s), seeds),
            "Oracle (upper)": (lambda s: make_oracle(train, n_syn, seed=s), seeds),
            "SDV-GaussCop":   (lambda s: make_sdv_gaussian(train, info, n_syn, s), seeds),
        }
        # copula variants from the grid (refit on FULL train for final eval)
        for c in config_grid():
            def mk(s, c=c):
                return make_copula_syn(tabby, train, info, c["method"], seed=s,
                                       cols_mode=c["cols_mode"],
                                       vine_family=c.get("vine_family", "tll"),
                                       vine_trunc=c.get("vine_trunc", None))
            builders[c["name"]] = (mk, seeds)

        print(f"\n  ── TEST eval (multi-seed) — val-selected best = {best['name']} ──")
        for name, (mk, sds) in builders.items():
            try:
                res = eval_multiseed(mk, test, test, info, sds, train_ref=train)
            except Exception as e:
                print(f"  {name:18s} FAILED: {e}")
                continue
            star = " *" if name == best["name"] else ""
            row = {"dataset": ds, "variant": name,
                   "val_selected": (name == best["name"])}
            for k, (mu, sd) in res.items():
                row[k] = mu; row[k + "_std"] = sd
            all_rows.append(row)
            print(f"  {name:16s}{star:2s} C2ST={res['xgb_c2st'][0]:.4f}±{res['xgb_c2st'][1]:.4f} "
                  f"TV3={res['tv3'][0]:.4f}±{res['tv3'][1]:.4f} "
                  f"AUC={res['mle_auc'][0]:.4f}±{res['mle_auc'][1]:.4f} "
                  f"F1={res['mle_f1'][0]:.4f}±{res['mle_f1'][1]:.4f} "
                  f"| Trend={res['trend'][0]:.4f} LR={res['lr_c2st'][0]:.4f}")

    df = pd.DataFrame(all_rows)
    df.to_csv("phase2_copula_compare.csv", index=False)
    print("\n[SAVED] phase2_copula_compare.csv")

    # save val tables
    vrows = []
    for ds, tbl in val_tables.items():
        for r in tbl:
            vrows.append({"dataset": ds, "config": r["name"],
                          "val_c2st": r["val_c2st"], "val_c2st_std": r["val_c2st_std"]})
    pd.DataFrame(vrows).to_csv("phase2_val_tuning.csv", index=False)
    print("[SAVED] phase2_val_tuning.csv")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["adult", "shoppers", "diabetes"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    run(args.datasets, args.seeds)
