"""Phase B evaluation: compare fix (P1+P3) vs baseline on adult + default.

For each dataset × variant (baseline/fix) × seed:
  - find best available samples (step 8000 > 6000 > 4000 > 2000)
  - evaluate: xgb_c2st, marg_c2st, dep_c2st, dep_cross (A1-style), mle_f1, mle_recall
  - 5 seed mean±std

Saves:
  step_b_eval_long.csv    all rows
  step_b_eval_summary.csv mean±std per (dataset, variant)
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import (load_info, evaluate_all, xgb_c2st, marginal_c2st,
                            DATASETS, self_check)
from run_a1_coltype import coltype_c2st   # reuse dep_cross computation

BASE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]

TARGET_DS = ["default", "adult"]
EXTRA_DS = {
    "default": {"train": "data/default/train.csv", "test": "data/default/test.csv",
                "info": "data/default/info.json"},
    "adult": DATASETS["adult"],
}


def get_syn_path(ds, exp_type, seed):
    """exp_type: 'base' or 'fix'"""
    exp = f"{ds}_{exp_type}_s{seed}"
    for step in [8000, 6000, 4000, 2000]:
        p = f"ef_vfm/result/{ds}/{exp}/{step}/samples.csv"
        if os.path.exists(p):
            return p, step
    # fallback: ef-vfm-dep baseline for adult/shoppers (base only)
    if exp_type == "base" and ds in DATASETS and DATASETS[ds].get("syn"):
        p = DATASETS[ds]["syn"]
        if os.path.exists(p):
            return p, "dep_baseline"
    return None, None


def eval_one(ds, exp_type, seed, cfg):
    syn_path, step = get_syn_path(ds, exp_type, seed)
    if syn_path is None:
        return None
    info = load_info(cfg["info"])
    test = pd.read_csv(cfg["test"])
    train = pd.read_csv(cfg["train"])
    syn = pd.read_csv(syn_path)

    m = evaluate_all(syn, test, test, info, seed=seed, do_old=False, train_ref=train)
    # dep_cross (P3-specific, A1-style block-shuffle)
    dep_cross = coltype_c2st(test, syn, info, seed=seed, mode="cross")
    dep_cross_val = m["xgb_c2st"] - dep_cross

    m.update(
        dataset=ds, variant=exp_type, seed=seed, step=step,
        dep_cross=dep_cross_val, syn_path=syn_path,
    )
    return m


def main():
    os.chdir(BASE)
    print("=" * 65)
    print("Phase B evaluation: fix vs baseline (dep_c2st, F1, dep_cross)")
    print("=" * 65)

    assert self_check(DATASETS["adult"]), "determinism self-check FAILED → STOP"

    rows = []
    for ds in TARGET_DS:
        cfg = EXTRA_DS[ds]
        info = load_info(cfg["info"])
        print(f"\n── {ds} ──")
        for exp_type in ["base", "fix"]:
            found = 0
            for seed in SEEDS:
                r = eval_one(ds, exp_type, seed, cfg)
                if r is None:
                    print(f"  SKIP {ds} {exp_type} s{seed}: no samples")
                    continue
                rows.append(r)
                found += 1
                print(f"  {ds:8s} {exp_type:4s} s{seed} step={r['step']:>6} "
                      f"full={r['xgb_c2st']:.4f} dep={r['dep_c2st']:+.4f} "
                      f"dep_cross={r['dep_cross']:+.4f} "
                      f"f1={r['mle_f1']:.4f} rec={r['mle_recall']:.4f}")
            if found == 0:
                print(f"  {ds} {exp_type}: NO SAMPLES FOUND — training may still be running")

    if not rows:
        print("\nNo results yet. Run after training completes.")
        return

    long = pd.DataFrame(rows)
    long.to_csv("step_b_eval_long.csv", index=False)

    metrics = ["xgb_c2st", "marg_c2st", "dep_c2st", "dep_cross",
               "mle_f1", "mle_recall", "tv2", "tv3"]
    g = long.groupby(["dataset", "variant"])[metrics]
    summ = g.agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv("step_b_eval_summary.csv", index=False)

    # ── Print comparison table ──
    print("\n" + "=" * 65)
    print("COMPARISON: fix vs baseline (mean±std, ≥3 seeds)")
    print("=" * 65)
    for ds in TARGET_DS:
        b_row = summ[(summ.dataset == ds) & (summ.variant == "base")]
        f_row = summ[(summ.dataset == ds) & (summ.variant == "fix")]
        if b_row.empty or f_row.empty:
            print(f"  {ds}: incomplete data")
            continue
        b = b_row.iloc[0]; f = f_row.iloc[0]
        dep_delta  = f.dep_c2st_mean  - b.dep_c2st_mean
        cross_delta = f.dep_cross_mean - b.dep_cross_mean
        f1_delta   = f.mle_f1_mean    - b.mle_f1_mean
        gate_dep = dep_delta < -0.005
        gate_f1  = f1_delta  > 0.005
        gate = "✅ PASS" if gate_dep and gate_f1 else "❌ FAIL"
        print(f"\n  {ds}: {gate}")
        print(f"    dep:   base={b.dep_c2st_mean:.4f}±{b.dep_c2st_std:.4f} "
              f"→ fix={f.dep_c2st_mean:.4f}±{f.dep_c2st_std:.4f}  Δ={dep_delta:+.4f}")
        print(f"    cross: base={b.dep_cross_mean:.4f}±{b.dep_cross_std:.4f} "
              f"→ fix={f.dep_cross_mean:.4f}±{f.dep_cross_std:.4f}  Δ={cross_delta:+.4f}")
        print(f"    F1:    base={b.mle_f1_mean:.4f}±{b.mle_f1_std:.4f} "
              f"→ fix={f.mle_f1_mean:.4f}±{f.mle_f1_std:.4f}  Δ={f1_delta:+.4f}")
        print(f"    (gate: dep↓={gate_dep}, F1↑={gate_f1})")

    # ── PAIRED per-seed Δ (PRIMARY gate) ───────────────────────────────────
    # base_i and fix_i share the SAME training+sampling seed → they differ ONLY
    # by the mechanism. The per-seed difference Δ_i = fix_i − base_i removes the
    # shared init/sampling noise (a paired test), which is far more powerful than
    # comparing two independent 5-means: a consistent small effect that survives
    # here can be invisible as "overlapping error bars" in the unpaired view.
    print("\n" + "=" * 65)
    print("PAIRED per-seed Δ (fix − base @ SAME seed) — PRIMARY gate")
    print("=" * 65)
    gate_metrics = ["dep_cross", "dep_c2st", "mle_f1", "mle_recall", "xgb_c2st"]
    paired_rows = []
    for ds in TARGET_DS:
        b = long[(long.dataset == ds) & (long.variant == "base")]
        f = long[(long.dataset == ds) & (long.variant == "fix")]
        m = b[["seed"] + gate_metrics].merge(
            f[["seed"] + gate_metrics], on="seed", suffixes=("_base", "_fix"))
        if m.empty:
            print(f"\n  {ds}: no paired seeds yet"); continue
        n = len(m)
        print(f"\n  {ds}  (n={n} paired seeds)")
        deltas = {}
        for met in gate_metrics:
            d = (m[f"{met}_fix"] - m[f"{met}_base"]).values
            deltas[met] = d
            mean_d = float(d.mean()); std_d = float(d.std(ddof=1)) if n > 1 else 0.0
            n_dn = int((d < 0).sum()); n_up = int((d > 0).sum())
            paired_rows.append(dict(dataset=ds, metric=met, n=n,
                                    mean_delta=mean_d, std_delta=std_d,
                                    n_down=n_dn, n_up=n_up))
            print(f"    {met:11s} Δ={mean_d:+.4f} ± {std_d:.4f}  "
                  f"(per-seed signs: {n_dn}↓ / {n_up}↑ of {n})")
        # paired P3 gate: want dep_cross↓ & F1↑ consistently (≥ n-1 seeds same sign),
        # and full (xgb_c2st) not meaningfully worse.
        need = max(n - 1, 1)            # allow ≤1 dissenting seed
        g_cross = deltas["dep_cross"].mean() < 0 and int((deltas["dep_cross"] < 0).sum()) >= need
        g_f1    = deltas["mle_f1"].mean()    > 0 and int((deltas["mle_f1"]    > 0).sum()) >= need
        g_full  = deltas["xgb_c2st"].mean() < 0.01
        verdict = "✅ MOVED" if (g_cross and g_f1 and g_full) else "❌ NOT MOVED (or inconsistent)"
        print(f"    → P3 paired gate: dep_cross↓={g_cross}, F1↑={g_f1}, "
              f"full not worse={g_full}  →  {verdict}")
    if paired_rows:
        pd.DataFrame(paired_rows).to_csv("step_b_paired_delta.csv", index=False)

    print("\n[SAVED] step_b_eval_long.csv, step_b_eval_summary.csv, step_b_paired_delta.csv")


if __name__ == "__main__":
    main()
