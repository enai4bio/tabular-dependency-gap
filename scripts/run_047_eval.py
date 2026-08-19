"""047 Step 3 evaluation -> the two TMLR deliverables.

Produces:
  step_multidataset_diag.csv   (product 1): per dataset (adult/default/magic),
       1x baseline, variants FF / TabbyFlow / Oracle, 3-seed mean+/-std of
       full / marg / dep / dep_cross / minority-F1 / recall  (+ rr sanity).
  step_capacity.csv            (product 2): adult/default, TabbyFlow only,
       1x vs 4x, the dep components (dep, dep_cross) 3-seed mean+/-std.

Reuses proper_metrics (xgb_c2st, marg/dep, FF/oracle, mle_f1/recall) and
run_a1_coltype.coltype_c2st (block-shuffle num<->cat 'cross').

Robust: silently skips (dataset,capacity,seed) combos with no samples, so it
can be run incrementally while the training queue is still going. Prints a
coverage report of what was found.

Discipline (047 sec.6): XGB-C2ST + dep components + minority F1/recall + FF/oracle.
3-seed mean+/-std; no hand-picked single seed. Determinism self-check first.
"""
import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import (load_info, evaluate_all, make_ff, make_oracle,
                            xgb_c2st, self_check, check_positive_class, DATASETS)
from run_a1_coltype import coltype_c2st

BASE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2]

# dataset cfg (paths)
DS_CFG = {
    "adult":   {"train": "data/adult/train.csv",   "test": "data/adult/test.csv",   "info": "data/adult/info.json"},
    "default": {"train": "data/default/train.csv", "test": "data/default/test.csv", "info": "data/default/info.json"},
    "magic":   {"train": "data/magic/train.csv",   "test": "data/magic/test.csv",   "info": "data/magic/info.json"},
}

# capacity -> exp_name template ; 1x adult reuses the existing *_base_s* runs
EXP_TMPL = {
    "1x": "{ds}_base_s{seed}",
    "4x": "{ds}_cap4x_s{seed}",
}

DIAG_DATASETS = ["adult", "default", "magic"]   # product 1 (1x only)
CAP_DATASETS  = ["adult", "default"]            # product 2 (1x vs 4x)


def syn_path(ds, cap, seed):
    exp = EXP_TMPL[cap].format(ds=ds, seed=seed)
    for step in [8000, 6000, 4000, 2000]:
        p = f"ef_vfm/result/{ds}/{exp}/{step}/samples.csv"
        if os.path.exists(p):
            return p, step
    return None, None


def metrics_for(syn, train, test, info, seed):
    """Full diagnostic bundle for one synthetic table vs real test."""
    m = evaluate_all(syn, test, test, info, seed=seed, do_old=False, train_ref=train)
    # dep_cross: block-shuffle num<->cat alignment (clean cross term)
    dep_cross_auc = coltype_c2st(test, syn, info, seed=seed, mode="cross")
    m["dep_cross"] = float(m["xgb_c2st"] - dep_cross_auc)
    return m


def main():
    os.chdir(BASE)
    print("=" * 70)
    print("047 Step 3 evaluation -> multidataset diag + capacity")
    print("=" * 70)
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"

    diag_rows, cap_rows, coverage = [], [], []

    for ds in DIAG_DATASETS:
        cfg = DS_CFG[ds]
        info = load_info(cfg["info"])
        train = pd.read_csv(cfg["train"]); test = pd.read_csv(cfg["test"])
        n_test = len(test)
        if n_test < 3000:
            warnings.warn(f"{ds}: test n={n_test} < 3000 -> dep noisy / underpowered (flag in report)")
        check_positive_class(info, test)
        print(f"\n-- {ds} (train={len(train)}, test={n_test}) --")

        # product 1: 1x diag with FF / TabbyFlow / Oracle variants
        for seed in SEEDS:
            p, step = syn_path(ds, "1x", seed)
            if p is None:
                coverage.append(dict(dataset=ds, cap="1x", seed=seed, found=False)); continue
            coverage.append(dict(dataset=ds, cap="1x", seed=seed, found=True, step=step))
            syn = pd.read_csv(p)
            # rr sanity
            rng = np.random.default_rng(seed); perm = rng.permutation(n_test); h = n_test // 2
            rr = xgb_c2st(test.iloc[perm[:h]].reset_index(drop=True),
                          test.iloc[perm[h:]].reset_index(drop=True), info, seed=seed)
            variants = {"FF": make_ff(train, len(syn), seed=seed),
                        "TabbyFlow": syn,
                        "Oracle": make_oracle(train, len(syn), seed=seed)}
            for name, sdf in variants.items():
                m = metrics_for(sdf, train, test, info, seed)
                m.update(dataset=ds, variant=name, capacity="1x", seed=seed,
                         step=step, rr_c2st=rr, n_test=n_test)
                diag_rows.append(m)
                print(f"  {ds:8s} {name:10s} s{seed} full={m['xgb_c2st']:.4f} "
                      f"dep={m['dep_c2st']:+.4f} cross={m['dep_cross']:+.4f} "
                      f"f1={m['mle_f1']:.4f} rec={m['mle_recall']:.4f} rr={rr:.4f}")

    # product 2: capacity (TabbyFlow only) 1x vs 4x
    print("\n" + "=" * 70 + "\nCapacity (1x vs 4x, TabbyFlow)\n" + "=" * 70)
    for ds in CAP_DATASETS:
        cfg = DS_CFG[ds]; info = load_info(cfg["info"])
        train = pd.read_csv(cfg["train"]); test = pd.read_csv(cfg["test"])
        for cap in ["1x", "4x"]:
            for seed in SEEDS:
                p, step = syn_path(ds, cap, seed)
                if p is None:
                    if cap == "4x":
                        coverage.append(dict(dataset=ds, cap=cap, seed=seed, found=False))
                    continue
                if cap == "4x":
                    coverage.append(dict(dataset=ds, cap=cap, seed=seed, found=True, step=step))
                syn = pd.read_csv(p)
                m = metrics_for(syn, train, test, info, seed)
                row = dict(dataset=ds, capacity=cap, seed=seed, step=step,
                           full=m["xgb_c2st"], marg=m["marg_c2st"],
                           dep=m["dep_c2st"], dep_cross=m["dep_cross"],
                           mle_f1=m["mle_f1"], mle_recall=m["mle_recall"])
                cap_rows.append(row)
                print(f"  {ds:8s} {cap} s{seed} full={row['full']:.4f} "
                      f"dep={row['dep']:+.4f} cross={row['dep_cross']:+.4f} f1={row['mle_f1']:.4f}")

    # ---- save product 1 ----
    if diag_rows:
        long = pd.DataFrame(diag_rows)
        long.to_csv("step_multidataset_diag_long.csv", index=False)
        met = ["xgb_c2st", "marg_c2st", "dep_c2st", "dep_cross",
               "mle_f1", "mle_recall", "tv2", "tv3", "rr_c2st"]
        g = long.groupby(["dataset", "variant"])[met].agg(["mean", "std"])
        g.columns = [f"{a}_{b}" for a, b in g.columns]
        g.reset_index().to_csv("step_multidataset_diag.csv", index=False)
        print("\n[SAVED] step_multidataset_diag.csv (+ _long)")

    # ---- save product 2 ----
    if cap_rows:
        capdf = pd.DataFrame(cap_rows)
        capdf.to_csv("step_capacity_long.csv", index=False)
        met = ["full", "marg", "dep", "dep_cross", "mle_f1", "mle_recall"]
        g = capdf.groupby(["dataset", "capacity"])[met].agg(["mean", "std"])
        g.columns = [f"{a}_{b}" for a, b in g.columns]
        g.reset_index().to_csv("step_capacity.csv", index=False)
        print("[SAVED] step_capacity.csv (+ _long)")

    # ---- coverage report ----
    cov = pd.DataFrame(coverage)
    print("\n" + "=" * 70 + "\nCOVERAGE (what was found)\n" + "=" * 70)
    if not cov.empty:
        for ds in DIAG_DATASETS:
            for cap in ["1x", "4x"]:
                sub = cov[(cov.dataset == ds) & (cov.cap == cap)]
                if sub.empty:
                    continue
                n_found = int(sub["found"].sum())
                print(f"  {ds:8s} {cap}: {n_found}/{len(sub)} seeds found")
    print("\nDONE.")


if __name__ == "__main__":
    main()
