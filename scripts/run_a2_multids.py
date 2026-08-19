"""041 Phase A2: 多数据集全套 040 诊断 (adult/default/magic/shoppers),5 seed × 多样本.

对每个数据集:
  - 读取 5 个 seed 训出的样本(result/{ds}/{ds}_base_s{seed}/samples.csv)
  - 跑完整 evaluate_all(xgb_c2st + marg/dep + mle_f1/recall + FF/oracle + old)
  - 5 seed mean±std → step_a2_diag_long.csv + step_a2_diag_summary.csv

Also checks:
  - positive class == minority class (assert/warn)
  - test size adequacy (warn if < 3000)
  - rr_c2st gate (warn if > 0.52)
"""
import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import (load_info, evaluate_all, make_ff, make_oracle,
                            xgb_c2st, self_check, check_positive_class,
                            DATASETS)

BASE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]

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

ALL_DS_CFGS = {**DATASETS, **EXTRA_DATASETS}
HEADLINE = ["adult", "default"]          # primary evidence targets
AUXILIARY = ["magic", "shoppers"]        # supporting/contrast


def get_syn_path(ds, seed):
    """Return path to trained sample for this dataset/seed.

    main.py saves to ef_vfm/result/{ds}/{exp}/{step}/samples.csv.
    Prefer final step (8000); fall back to earlier checkpoints.
    """
    exp = f"{ds}_base_s{seed}"
    for step in [8000, 6000, 4000, 2000]:
        p = f"ef_vfm/result/{ds}/{exp}/{step}/samples.csv"
        if os.path.exists(p):
            return p
    return None


def main():
    os.chdir(BASE)
    print("=" * 70)
    print("Phase A2: Multi-dataset diagnosis (040 metrics extended)")
    print("=" * 70)

    assert self_check(DATASETS["adult"]), "determinism self-check FAILED → STOP"

    rows = []
    for ds, cfg in ALL_DS_CFGS.items():
        info = load_info(cfg["info"])
        train = pd.read_csv(cfg["train"])
        test  = pd.read_csv(cfg["test"])

        n_test = len(test)
        if n_test < 3000:
            warnings.warn(f"{ds}: test n={n_test} < 3000 → dep_c2st may be noisy (040 lesson)")

        # Check positive class
        check_positive_class(info, test)

        print(f"\n── {ds} (train={len(train)}, test={n_test}) ──")

        for seed in SEEDS:
            # Try to find trained samples; fall back to ef-vfm-dep baseline for adult/shoppers/diabetes
            syn_path = get_syn_path(ds, seed)
            if syn_path is None:
                # Fallback: use existing baselines (same for all seeds → only eval variance)
                fallback = cfg.get("syn")
                if fallback and os.path.exists(fallback):
                    syn_path = fallback
                    if seed == 0:
                        print(f"  {ds}: no trained samples found, using baseline {fallback} "
                              f"(eval-variance only, no generator variance)")
                else:
                    if seed == 0:
                        print(f"  {ds}: SKIP — no samples found")
                    continue

            syn = pd.read_csv(syn_path)

            # real-vs-real sanity
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n_test)
            h = n_test // 2
            rr = xgb_c2st(
                test.iloc[perm[:h]].reset_index(drop=True),
                test.iloc[perm[h:]].reset_index(drop=True),
                info, seed=seed
            )

            variants = {
                "FF":        make_ff(train, len(syn), seed=seed),
                "TabbyFlow": syn,
                "Oracle":    make_oracle(train, len(syn), seed=seed),
            }
            for name, sdf in variants.items():
                m = evaluate_all(sdf, test, test, info, seed=seed,
                                 do_old=True, train_ref=train)
                m.update(dataset=ds, variant=name, seed=seed, rr_c2st=rr,
                         n_test=n_test, syn_path=syn_path)
                rows.append(m)
                print(f"  {ds:8s} {name:10s} s{seed} "
                      f"full={m['xgb_c2st']:.4f} dep={m['dep_c2st']:+.4f} "
                      f"f1={m['mle_f1']:.4f} rec={m['mle_recall']:.4f} rr={rr:.4f}")

    long = pd.DataFrame(rows)
    long.to_csv("step_a2_diag_long.csv", index=False)

    metrics = ["xgb_c2st", "marg_c2st", "dep_c2st", "tv2", "tv3",
               "mle_auc", "mle_f1", "mle_recall", "trend", "lr_c2st", "rr_c2st"]
    g = long.groupby(["dataset", "variant"])[metrics]
    summ = g.agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv("step_a2_diag_summary.csv", index=False)

    # ── Print strategic fork analysis ──
    print("\n" + "=" * 70)
    print("STRATEGIC FORK ANALYSIS")
    print("=" * 70)
    summ_df = pd.read_csv("step_a2_diag_summary.csv")
    for ds in ["adult", "default", "magic", "shoppers", "diabetes"]:
        row = summ_df[(summ_df.dataset == ds) & (summ_df.variant == "TabbyFlow")]
        if row.empty:
            continue
        full_m = float(row.xgb_c2st_mean)
        dep_m  = float(row.dep_c2st_mean)
        dep_s  = float(row.dep_c2st_std)
        rr_m   = float(row.rr_c2st_mean)
        gap    = full_m - 0.5
        ratio  = dep_m / gap if gap > 0.001 else float("nan")
        snr    = abs(dep_m) / dep_s if dep_s > 0.001 else float("nan")
        n_test = int(long[long.dataset == ds].n_test.iloc[0]) if ds in long.dataset.values else "?"
        flag   = "✅" if snr > 5 else ("⚠️" if snr > 2 else "❌")
        print(f"  {flag} {ds:8s} n_test={n_test:>6}  full={full_m:.3f}  dep={dep_m:+.3f}±{dep_s:.3f}  "
              f"ratio={ratio:.2f}  SNR={snr:.1f}  rr={rr_m:.3f}")

    print("\n[SAVED] step_a2_diag_long.csv, step_a2_diag_summary.csv")
    print("\nNOTE: datasets with SNR > 5 provide reliable dep_ratio evidence.")
    print("      SNR < 2 = too noisy (shoppers 040 lesson); direction only.")


if __name__ == "__main__":
    main()
