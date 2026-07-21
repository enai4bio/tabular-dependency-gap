"""041 Phase A1: 列类型 dep 归因 —— dep_numinv / dep_catinv / dep_cross (5 seed).

三个量:
  dep_numinv = full_c2st - c2st(只打乱数值列)   # num-num + num-cat 代价(粗判,重叠)
  dep_catinv = full_c2st - c2st(只打乱类别列)   # cat-cat + num-cat 代价(粗判,重叠)
  dep_cross  = full_c2st - c2st(block-shuffle)   # 干净单拎 num↔cat 交叉项

注意: dep_numinv + dep_catinv >= dep_full (num-cat 被两边各算一遍)——仅作粗判,
报告里明确标注"重叠,非 partition"。dep_cross 才是干净的交叉项。
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import (load_info, xgb_c2st, _shuffle_cols, _ordinal_encode,
                            cat_columns, num_columns, DATASETS)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

BASE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]


# ── extra dataset registry (default / magic not in proper_metrics.DATASETS) ──
EXTRA_DATASETS = {
    "default": {
        "train": "data/default/train.csv",
        "test":  "data/default/test.csv",
        "info":  "data/default/info.json",
    },
    "magic": {
        "train": "data/magic/train.csv",
        "test":  "data/magic/test.csv",
        "info":  "data/magic/info.json",
    },
}
ALL_DATASETS = {**DATASETS, **EXTRA_DATASETS}


def _shuffle_subset(df, col_indices, seed):
    """Independently permute only the specified columns (by position index)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    cols = list(df.columns)
    for i in col_indices:
        c = cols[i]
        out[c] = out[c].values[rng.permutation(len(out))]
    return out.reset_index(drop=True)


def _block_shuffle(df, num_idx, cat_idx, seed):
    """Block-shuffle: keep num-block order and cat-block order, permute alignment.

    Concretely: permute which num-row aligns with which cat-row.
    Preserves num-num and cat-cat joint distributions; destroys num-cat alignment.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    cols = list(df.columns)
    n = len(df)
    perm = rng.permutation(n)
    cat_cols = [cols[i] for i in cat_idx]
    # Move cat block rows to permuted positions (num block stays put)
    for c in cat_cols:
        out[c] = df[c].values[perm]
    return out.reset_index(drop=True)


def coltype_c2st(real_df, syn_df, info, seed, mode="numinv"):
    """
    mode='numinv': shuffle only num cols in BOTH real and syn, then C2ST.
    mode='catinv': shuffle only cat cols in BOTH real and syn.
    mode='cross':  block-shuffle BOTH (permute num-cat alignment).
    Returns AUC (canonicalized).
    """
    num_idx = num_columns(info)
    cat_idx = cat_columns(info)

    if mode == "numinv":
        r = _shuffle_subset(real_df, num_idx, seed + 10)
        s = _shuffle_subset(syn_df,  num_idx, seed + 20)
    elif mode == "catinv":
        r = _shuffle_subset(real_df, cat_idx, seed + 30)
        s = _shuffle_subset(syn_df,  cat_idx, seed + 40)
    elif mode == "cross":
        r = _block_shuffle(real_df, num_idx, cat_idx, seed + 50)
        s = _block_shuffle(syn_df,  num_idx, cat_idx, seed + 60)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return xgb_c2st(r, s, info, seed=seed)


def run_a1(syn_csv, ds_name, cfg):
    """Compute dep_numinv/dep_catinv/dep_cross for one dataset."""
    info = load_info(cfg["info"])
    test = pd.read_csv(cfg["test"])
    train = pd.read_csv(cfg["train"])
    syn  = pd.read_csv(syn_csv)

    num_idx = num_columns(info)
    cat_idx = cat_columns(info)
    has_cat = len(cat_idx) > 0

    rows = []
    for seed in SEEDS:
        # Full C2ST (from proper_metrics logic)
        full = xgb_c2st(test, syn, info, seed=seed)
        # Num-inverted: shuffle only num cols
        numinv = coltype_c2st(test, syn, info, seed, mode="numinv")
        dep_numinv = full - numinv

        if has_cat:
            catinv = coltype_c2st(test, syn, info, seed, mode="catinv")
            dep_catinv = full - catinv
            cross_c2st = coltype_c2st(test, syn, info, seed, mode="cross")
            dep_cross  = full - cross_c2st
        else:
            catinv = dep_catinv = cross_c2st = dep_cross = float("nan")

        row = dict(
            dataset=ds_name, seed=seed,
            full_c2st=full,
            numinv_c2st=numinv, dep_numinv=dep_numinv,
            catinv_c2st=catinv if has_cat else float("nan"),
            dep_catinv=dep_catinv,
            cross_c2st=cross_c2st if has_cat else float("nan"),
            dep_cross=dep_cross,
            n_num=len(num_idx), n_cat=len(cat_idx),
        )
        rows.append(row)
        print(f"  {ds_name:8s} s{seed} full={full:.4f} "
              f"dep_num={dep_numinv:+.4f} dep_cat={dep_catinv:+.4f} dep_cross={dep_cross:+.4f}")
    return rows


def summarize(rows):
    df = pd.DataFrame(rows)
    metrics = ["full_c2st", "dep_numinv", "dep_catinv", "dep_cross"]
    g = df.groupby("dataset")[metrics]
    s = g.agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    return df, s.reset_index()


def main():
    os.chdir(BASE)
    print("=" * 60)
    print("Phase A1: Column-type dependency attribution")
    print("NOTE: dep_numinv + dep_catinv >= dep_full (num-cat overlap)")
    print("      dep_cross is the CLEAN cross-type attribution.")
    print("=" * 60)

    # Datasets to run A1 on:
    # adult: use ef-vfm-dep baseline samples (already validated in 040)
    # shoppers: use ef-vfm-dep baseline samples (辅助)
    # default: needs trained samples -> use result/default/default_base_s0/samples.csv
    #          (will be skipped if not available yet)
    # magic: needs trained samples (no cat cols -> only dep_numinv)
    a1_configs = {
        "adult":    ("baselines/adult/samples.csv",    ALL_DATASETS["adult"]),
        "shoppers": ("baselines/shoppers/samples.csv", ALL_DATASETS["shoppers"]),
    }
    # Add default/magic if trained samples exist (main.py saves to ef_vfm/result/)
    for ds in ["default", "magic"]:
        for s in SEEDS:
            # Check final step first, then intermediate 2000
            for step in [8000, 4000, 6000, 2000]:
                p = f"ef_vfm/result/{ds}/{ds}_base_s{s}/{step}/samples.csv"
                if os.path.exists(p):
                    a1_configs[ds] = (p, ALL_DATASETS[ds])
                    print(f"  Found {ds} samples: {p}")
                    break
            if ds in a1_configs:
                break

    all_rows = []
    for ds_name, (syn_csv, cfg) in a1_configs.items():
        if not os.path.exists(syn_csv):
            print(f"  SKIP {ds_name}: {syn_csv} not found")
            continue
        print(f"\n--- {ds_name} ---")
        rows = run_a1(syn_csv, ds_name, cfg)
        all_rows.extend(rows)

    if not all_rows:
        print("No results — check that sample CSVs exist.")
        return

    long, summ = summarize(all_rows)
    long.to_csv("step_a1_coltype_dep_long.csv", index=False)
    summ.to_csv("step_a1_coltype_dep.csv", index=False)
    print("\n" + "=" * 60)
    print("SUMMARY (mean±std over 5 seeds)")
    print("=" * 60)
    for _, row in summ.iterrows():
        ds = row["dataset"]
        print(f"  {ds:8s}  full={row.full_c2st_mean:.4f}±{row.full_c2st_std:.4f}  "
              f"dep_num={row.dep_numinv_mean:+.4f}±{row.dep_numinv_std:.4f}  "
              f"dep_cat={row.dep_catinv_mean:+.4f}±{row.dep_catinv_std:.4f}  "
              f"dep_cross={row.dep_cross_mean:+.4f}±{row.dep_cross_std:.4f}")
    print("\n[SAVED] step_a1_coltype_dep.csv, step_a1_coltype_dep_long.csv")

    print("\nFIRING POINT RECOMMENDATION:")
    for _, row in summ.iterrows():
        ds = row["dataset"]
        cross = row.dep_cross_mean
        cat = row.dep_catinv_mean
        num = row.dep_numinv_mean
        if np.isnan(cross):
            print(f"  {ds:8s}: No categories → P1 (numeric only)")
        elif cross > 0.03:
            print(f"  {ds:8s}: dep_cross={cross:.3f} LARGE → P3 (num-cat cross) is primary target")
        elif cat > num * 1.5:
            print(f"  {ds:8s}: dep_cat={cat:.3f} >> dep_num={num:.3f} → P2 (categorical coupling)")
        else:
            print(f"  {ds:8s}: dep_num={num:.3f} ≈ dep_cat={cat:.3f} → P1 first (cheapest probe)")


if __name__ == "__main__":
    main()
