"""
033 Phase 1: PROPER metrics to re-measure how far TabbyFlow is from real.

Three correct tools (the 032 conclusion used broken Trend / LR-C2ST):
  1. XGBoost-C2ST  — tree classifier real-vs-syn; catches high-order interactions
                     that logistic regression is blind to. Report discriminator
                     AUC (0.5 = indistinguishable = good) + indistinguishability.
  2. FF (fully-factorized) blindness check — each real column shuffled
                     independently => zero dependency. A metric on which
                     TabbyFlow ~ FF is BLIND to dependency (throw it out);
                     a metric where TabbyFlow >> FF actually sees dependency.
  3. k-way TV       — explicit higher-order (k=3) interaction metric:
                     discretise columns, average total-variation distance of
                     the joint k-way histogram over many random k-subsets.

All fidelity metrics use REAL = held-out TEST as the reference (no circularity
with copula fitting, which only ever touches train). Downstream MLE trains on
synthetic and tests on the real test set.

Reusable as a library (import the functions) and runnable for Phase 1:
    python proper_metrics.py
Determinism self-check (seed=42, evaluate one sample 3x) runs first and aborts
on failure.
"""
import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_table import LogisticDetection
from eval.mle.mle import get_evaluator

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ── dataset registry ──────────────────────────────────────────────────────────
DATASETS = {
    "diabetes": {"train": "data/diabetes/train.csv", "test": "data/diabetes/test.csv",
                 "info": "data/diabetes/info.json", "syn": "baselines/diabetes/samples.csv"},
    "adult":    {"train": "data/adult/train.csv",    "test": "data/adult/test.csv",
                 "info": "data/adult/info.json",    "syn": "baselines/adult/samples.csv"},
    "shoppers": {"train": "data/shoppers/train.csv", "test": "data/shoppers/test.csv",
                 "info": "data/shoppers/info.json", "syn": "baselines/shoppers/samples.csv"},
}

SUBSET_SEED = 12345   # fixes which random k-subsets kway_tv uses (paired across variants)


# ── info / sdmetrics helpers (mirror headroom_diagnose) ────────────────────────

def load_info(path):
    with open(path) as f:
        info = json.load(f)
    info["metadata"]["columns"] = {int(k): v for k, v in info["metadata"]["columns"].items()}
    return info


def cat_columns(info):
    """Column indices treated as categorical (cat cols + target for classification)."""
    cat = list(info["cat_col_idx"])
    if info["task_type"] in ("binclass", "multiclass"):
        cat = cat + list(info["target_col_idx"])
    return sorted(cat)


def num_columns(info):
    num = list(info["num_col_idx"])
    if info["task_type"] == "regression":
        num = num + list(info["target_col_idx"])
    return sorted(num)


def reorder_for_sdmetrics(df, info):
    info = deepcopy(info)
    num_idx = deepcopy(info["num_col_idx"])
    cat_idx = deepcopy(info["cat_col_idx"])
    tgt_idx = deepcopy(info["target_col_idx"])
    if info["task_type"] == "regression":
        num_idx += tgt_idx
    else:
        cat_idx += tgt_idx
    df_r = pd.concat([df[num_idx], df[cat_idx]], axis=1)
    df_r.columns = range(len(df_r.columns))
    cols_meta = info["metadata"]["columns"]
    new_meta_cols = {i: cols_meta[orig] for i, orig in enumerate(num_idx + cat_idx)}
    return df_r, {"columns": new_meta_cols}


def sdmetrics_trend(real_df, syn_df, info):
    """Old pairwise Trend (column-pair correlation/contingency similarity)."""
    real_df = real_df.copy(); syn_df = syn_df.copy()
    real_df.columns = range(len(real_df.columns))
    syn_df.columns = range(len(syn_df.columns))
    real_r, meta = reorder_for_sdmetrics(real_df, info)
    syn_r, _ = reorder_for_sdmetrics(syn_df, info)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qr = QualityReport()
            qr.generate(real_r, syn_r, meta, verbose=False)
        return float(qr.get_properties()["Score"][1])   # Trend
    except Exception:
        return float("nan")


def lr_c2st(real_df, syn_df, info):
    """Old LR-based detection (LogisticDetection); 1.0 = indistinguishable."""
    real_df = real_df.copy(); syn_df = syn_df.copy()
    real_df.columns = range(len(real_df.columns))
    syn_df.columns = range(len(syn_df.columns))
    real_r, meta = reorder_for_sdmetrics(real_df, info)
    syn_r, _ = reorder_for_sdmetrics(syn_df, info)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(LogisticDetection.compute(real_data=real_r, synthetic_data=syn_r, metadata=meta))
    except Exception:
        return float("nan")


# ── encoding for tree-based C2ST ───────────────────────────────────────────────

def _ordinal_encode(real_df, syn_df, info):
    """Encode categoricals to integer codes (union of real+syn), numerics to float."""
    real_df = real_df.copy(); syn_df = syn_df.copy()
    real_df.columns = range(len(real_df.columns))
    syn_df.columns = range(len(syn_df.columns))
    cats = set(cat_columns(info))
    cols = list(range(len(real_df.columns)))
    R = np.zeros((len(real_df), len(cols)), dtype=float)
    S = np.zeros((len(syn_df), len(cols)), dtype=float)
    for j, c in enumerate(cols):
        if c in cats:
            rv = real_df[c].astype(str).values
            sv = syn_df[c].astype(str).values
            uniq = pd.Index(sorted(set(rv) | set(sv)))
            R[:, j] = uniq.get_indexer(rv)
            S[:, j] = uniq.get_indexer(sv)
        else:
            R[:, j] = pd.to_numeric(real_df[c], errors="coerce").values
            S[:, j] = pd.to_numeric(syn_df[c], errors="coerce").values
    return R, S


# ── XGBoost-C2ST (core) ────────────────────────────────────────────────────────

def xgb_c2st(real_df, syn_df, info, seed=42, n_splits=5, n_estimators=300, max_depth=6):
    """
    Classifier two-sample test with XGBoost (catches high-order interaction).
    Balance real/syn to equal size (subsample larger w/ seed), label real=1/syn=0,
    stratified CV, return out-of-fold discriminator AUC.
      AUC ~0.5 => indistinguishable (good);  AUC ->1.0 => easily told apart (bad).
    """
    rng = np.random.default_rng(seed)
    R, S = _ordinal_encode(real_df, syn_df, info)
    n = min(len(R), len(S))
    if len(R) > n:
        R = R[rng.choice(len(R), n, replace=False)]
    if len(S) > n:
        S = S[rng.choice(len(S), n, replace=False)]
    X = np.vstack([R, S])
    y = np.concatenate([np.ones(len(R)), np.zeros(len(S))])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            n_jobs=4, random_state=seed, eval_metric="logloss",
        )
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof)
    # canonicalise: discriminability is symmetric around 0.5
    auc = max(auc, 1.0 - auc)
    return float(auc)


def indistinguishability(auc):
    """Map C2ST AUC in [0.5,1] to [0,1]; 1.0 = perfectly indistinguishable."""
    return float(2.0 * (1.0 - auc))


def _shuffle_cols(df, seed):
    """Independently permute every column => destroys all dependency, keeps marginals."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].values[rng.permutation(len(out))]
    return out.reset_index(drop=True)


def marginal_c2st(real_df, syn_df, info, seed=42):
    """
    Marginal-only XGB-C2ST: independently shuffle every column in BOTH real and
    syn (dependency destroyed in both), then C2ST. Isolates the MARGINAL part of
    the full-C2ST gap. full_c2st - marginal_c2st = the dependency part.
    """
    return xgb_c2st(_shuffle_cols(real_df, seed + 1),
                    _shuffle_cols(syn_df, seed + 2), info, seed=seed)


# ── k-way TV (higher-order interaction metric) ─────────────────────────────────

def _discretize(real_df, syn_df, info, n_bins):
    """Discretise every column to integer codes using bin edges from REAL ref."""
    real_df = real_df.copy(); syn_df = syn_df.copy()
    real_df.columns = range(len(real_df.columns))
    syn_df.columns = range(len(syn_df.columns))
    cats = set(cat_columns(info))
    d = len(real_df.columns)
    Rc = np.zeros((len(real_df), d), dtype=np.int64)
    Sc = np.zeros((len(syn_df), d), dtype=np.int64)
    n_cells = np.zeros(d, dtype=np.int64)
    for j in range(d):
        if j in cats:
            rv = real_df[j].astype(str).values
            sv = syn_df[j].astype(str).values
            uniq = pd.Index(sorted(set(rv) | set(sv)))
            Rc[:, j] = uniq.get_indexer(rv)
            Sc[:, j] = uniq.get_indexer(sv)
            n_cells[j] = len(uniq)
        else:
            rv = pd.to_numeric(real_df[j], errors="coerce").values.astype(float)
            sv = pd.to_numeric(syn_df[j], errors="coerce").values.astype(float)
            qs = np.linspace(0, 1, n_bins + 1)[1:-1]
            edges = np.unique(np.nanquantile(rv, qs))
            Rc[:, j] = np.digitize(np.nan_to_num(rv, nan=np.nanmedian(rv)), edges)
            Sc[:, j] = np.digitize(np.nan_to_num(sv, nan=np.nanmedian(rv)), edges)
            n_cells[j] = len(edges) + 1
    return Rc, Sc, n_cells


def _subsets(d, k, n_subsets, seed):
    rng = np.random.default_rng(seed)
    from itertools import combinations
    allc = list(combinations(range(d), k))
    if len(allc) <= n_subsets:
        return allc
    idx = rng.choice(len(allc), n_subsets, replace=False)
    return [allc[i] for i in idx]


def kway_tv(real_df, syn_df, info, k=3, n_bins=4, n_subsets=200, subset_seed=SUBSET_SEED):
    """
    Mean total-variation distance of the joint k-way histogram between real
    and syn over random k-subsets of columns. 0 = identical k-way dependence;
    higher = worse. Deterministic given fixed subsets.
    """
    Rc, Sc, n_cells = _discretize(real_df, syn_df, info, n_bins)
    d = Rc.shape[1]
    if d < k:
        return float("nan")
    subs = _subsets(d, k, n_subsets, subset_seed)
    tvs = []
    nr, ns = len(Rc), len(Sc)
    for sub in subs:
        base = 1
        ridx = np.zeros(nr, dtype=np.int64)
        sidx = np.zeros(ns, dtype=np.int64)
        for c in sub:
            ridx += Rc[:, c] * base
            sidx += Sc[:, c] * base
            base *= n_cells[c]
        # joint distributions over the union of observed cells
        rcnt = np.bincount(ridx, minlength=base).astype(float)
        scnt = np.bincount(sidx, minlength=base).astype(float)
        pr = rcnt / rcnt.sum()
        ps = scnt / scnt.sum()
        tvs.append(0.5 * np.abs(pr - ps).sum())
    return float(np.mean(tvs))


# ── downstream MLE (train on syn -> test on real test) ─────────────────────────

def mle_auc_f1(syn_df, test_df, info, seed=42):
    train_np = syn_df.to_numpy()
    test_np = test_df.to_numpy()
    task = info["task_type"]
    ev = get_evaluator(task)
    if task == "regression":
        best_r2, best_rmse = ev(train_np, test_np, info, seed=seed)
        return float(best_rmse[0]["RMSE"]), float("nan"), float("nan")
    best_f1, f1_scores, best_auroc, _, _ = ev(train_np, test_np, info, seed=seed)
    auc = float(best_auroc[0]["roc_auc"])
    f1 = float(f1_scores[0].get("binary_f1", f1_scores[0].get("macro_f1", float("nan"))))
    recall = float(f1_scores[0].get("minority_recall", float("nan")))
    return auc, f1, recall


def check_positive_class(info, test_df):
    """Assert that sklearn average='binary' positive class == minority class."""
    if info["task_type"] != "binclass":
        return
    tgt = info["target_col_idx"][0]
    counts = test_df.iloc[:, tgt].value_counts()
    minority = counts.idxmin()
    # sklearn LabelEncoder encodes alphabetically; positive class = label with higher integer code
    labels = sorted(test_df.iloc[:, tgt].astype(str).unique())
    positive_label = labels[-1]  # highest lexicographic = positive in binary
    minority_label = str(minority)
    if positive_label != minority_label:
        warnings.warn(
            f"positive_class='{positive_label}' != minority_class='{minority_label}' "
            f"(counts={dict(counts)}). F1/recall may measure majority class.",
            stacklevel=2,
        )


def make_val_split(train_df, info, val_frac=0.12, seed=12345):
    """Stratified split: returns (train_sub, val) with val_frac of train as val."""
    from sklearn.model_selection import StratifiedShuffleSplit
    tgt = info["target_col_idx"][0]
    y = train_df.iloc[:, tgt].values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, val_idx = next(sss.split(train_df, y))
    return train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)


# ── reference-set builders ─────────────────────────────────────────────────────

def make_ff(train_df, n, seed=42):
    """Fully-factorized: each column independently resampled from its marginal."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in train_df.columns:
        out[c] = train_df[c].values[rng.integers(0, len(train_df), n)]
    return pd.DataFrame(out)


def make_oracle(train_df, n, seed=42):
    """Real marginals + real dependencies (subsample real train) — upper bound."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_df), n, replace=(n > len(train_df)))
    return train_df.iloc[idx].reset_index(drop=True)


# ── full metric bundle for one synthetic set ───────────────────────────────────

def evaluate_all(syn_df, holdout_df, test_df, info, seed=42,
                 kway_k=3, do_old=True, train_ref=None):
    """
    holdout_df = held-out real (test) -> NEW fidelity metrics (no copula-fit circularity).
    test_df    = real test            -> downstream MLE.
    train_ref  = real train           -> OLD metrics (Trend / LR-C2ST), their standard
                 usage; also avoids sdmetrics' unseen-category crash and reproduces the
                 "fake saturation" 032 reported. Defaults to holdout_df if None.
    """
    out = {}
    out["xgb_c2st"] = xgb_c2st(holdout_df, syn_df, info, seed=seed)
    out["indist"]   = indistinguishability(out["xgb_c2st"])
    out["marg_c2st"] = marginal_c2st(holdout_df, syn_df, info, seed=seed)
    out["dep_c2st"]  = float(out["xgb_c2st"] - out["marg_c2st"])  # dependency-attributable
    out["tv2"]      = kway_tv(holdout_df, syn_df, info, k=2)
    out["tv3"]      = kway_tv(holdout_df, syn_df, info, k=kway_k)
    auc, f1, rec = mle_auc_f1(syn_df, test_df, info, seed=seed)
    out["mle_auc"], out["mle_f1"], out["mle_recall"] = auc, f1, rec
    if do_old:
        ref = train_ref if train_ref is not None else holdout_df
        out["trend"]   = sdmetrics_trend(ref, syn_df, info)
        out["lr_c2st"] = lr_c2st(ref, syn_df, info)
    return out


# ── determinism self-check ─────────────────────────────────────────────────────

def self_check(cfg):
    print("[SELF-CHECK] evaluating one sample 3x with seed=42 ...")
    info = load_info(cfg["info"])
    syn = pd.read_csv(cfg["syn"]); test = pd.read_csv(cfg["test"])
    runs = []
    for _ in range(3):
        r = evaluate_all(syn, test, test, info, seed=42, do_old=False)
        runs.append((r["xgb_c2st"], r["tv3"], r["mle_auc"], r["mle_f1"]))
    ok = runs[0] == runs[1] == runs[2]
    if ok:
        print(f"  PASS  C2ST={runs[0][0]:.6f} TV3={runs[0][1]:.6f} "
              f"AUC={runs[0][2]:.6f} F1={runs[0][3]:.6f} (identical x3)")
    else:
        print("  FAIL  runs differ:")
        for r in runs:
            print("   ", r)
    return ok


# ── Phase 1 main: real headroom map ────────────────────────────────────────────

def run():
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    print("=" * 70)
    print("033 PHASE 1 — PROPER-METRIC HEADROOM MAP")
    print("=" * 70)

    if not self_check(DATASETS["adult"]):
        print("[ABORT] determinism self-check failed.")
        sys.exit(1)

    rows = []
    for ds, cfg in DATASETS.items():
        print(f"\n── {ds} ──")
        info = load_info(cfg["info"])
        train = pd.read_csv(cfg["train"]); test = pd.read_csv(cfg["test"])
        syn = pd.read_csv(cfg["syn"])
        n_ref = len(test)
        print(f"  train={len(train)} test={len(test)} syn={len(syn)}")

        # real-vs-real C2ST sanity: split test in half should be ~0.5
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(test)); h = len(test) // 2
        san_c2st = xgb_c2st(test.iloc[perm[:h]].reset_index(drop=True),
                            test.iloc[perm[h:]].reset_index(drop=True), info, seed=42)
        print(f"  [SANITY] real-vs-real XGB-C2ST AUC = {san_c2st:.4f} (want ~0.5)")

        variants = {
            "TabbyFlow": syn,
            "FF (lower)": make_ff(train, len(syn), seed=42),
            "Oracle (upper)": make_oracle(train, len(syn), seed=42),
        }
        for name, sdf in variants.items():
            m = evaluate_all(sdf, test, test, info, seed=42, do_old=True, train_ref=train)
            m.update({"dataset": ds, "variant": name, "san_rr_c2st": san_c2st})
            rows.append(m)
            print(f"  {name:16s} C2ST={m['xgb_c2st']:.4f} (marg={m['marg_c2st']:.4f} "
                  f"dep={m['dep_c2st']:+.4f}) TV3={m['tv3']:.4f} AUC={m['mle_auc']:.4f} "
                  f"F1={m['mle_f1']:.4f} | Trend={m['trend']:.4f} LR-C2ST={m['lr_c2st']:.4f}")

    df = pd.DataFrame(rows)
    cols = ["dataset", "variant", "xgb_c2st", "marg_c2st", "dep_c2st", "indist",
            "tv2", "tv3", "mle_auc", "mle_f1", "trend", "lr_c2st", "san_rr_c2st"]
    df = df[cols]
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 70)
    print("PHASE 1 HEADROOM MAP (real ref = held-out test)")
    print("=" * 70)
    print(df.to_string(index=False))
    df.to_csv("phase1_headroom_map.csv", index=False)
    print("\n[SAVED] phase1_headroom_map.csv")
    return df


if __name__ == "__main__":
    run()
