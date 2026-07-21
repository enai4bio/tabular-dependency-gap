"""048 Exp1 -- SMOTE baseline: does SMOTE (linear interpolation) recover real dependency?

For adult + default, 5 seeds, mean+/-std:
  - SMOTENC oversample the minority class of TRAIN to balance -> augmented train set.
  - downstream: classifier trained on SMOTE-augmented set, F1/recall on REAL test
    (same pipeline as every other paper row: proper_metrics.mle_auc_f1).
  - dependency diagnostics: run proper_metrics on the SMOTE-augmented set vs real_test
    (full / marg / dep / dep_cross). dep is the key: is dependency recovered (dep~0)
    or still large (not recovered)?

Note (neutral): the SMOTE-augmented set is class-BALANCED whereas real_test is imbalanced,
so full/marg C2ST carry a class-prior mismatch; dep = full-marg cancels most of that
marginal effect, so dep/dep_cross are the comparable dependency signal. RealTrain row
(classifier on original train) is included as the F1 reference point.

Outputs: step_smote.csv (+ _long). Neutral numbers only; no win/lose verdict.
"""
import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import (load_info, evaluate_all, mle_auc_f1, xgb_c2st,
                            make_ff, make_oracle, self_check, check_positive_class,
                            DATASETS)
from run_a1_coltype import coltype_c2st
from imblearn.over_sampling import SMOTENC

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
SEEDS = [0, 1, 2, 3, 4]
DS_CFG = {
    "adult":   {"train": "data/adult/train.csv",   "test": "data/adult/test.csv",   "info": "data/adult/info.json"},
    "default": {"train": "data/default/train.csv", "test": "data/default/test.csv", "info": "data/default/info.json"},
}


def smote_augment(train, info, seed):
    """SMOTENC oversample minority to balance. Returns augmented full df (target last)."""
    tgt = info["target_col_idx"][0]
    ncols = train.shape[1]
    assert tgt == ncols - 1, f"expected target as last column, got {tgt} of {ncols}"
    X = train.iloc[:, :tgt].copy()
    y = train.iloc[:, tgt].copy()
    cat_in_X = [c for c in info["cat_col_idx"] if c < tgt]   # positions within X
    sm = SMOTENC(categorical_features=cat_in_X, random_state=seed, sampling_strategy="auto")
    X_res, y_res = sm.fit_resample(X, y)
    aug = pd.concat([pd.DataFrame(X_res).reset_index(drop=True),
                     pd.Series(y_res, name=tgt).reset_index(drop=True)], axis=1)
    aug.columns = range(aug.shape[1])
    return aug


def prior_match_subsample(aug, info, ref_df, seed):
    """Subsample the balanced SMOTE set back to real_test class ratio (keep all
    majority, downsample minority incl. synthetic). Removes the class-prior mismatch
    so dep=full-marg is comparable to TabbyFlow (which keeps the original ratio)."""
    tgt = info["target_col_idx"][0]
    rng = np.random.default_rng(seed + 100)
    ay = aug.iloc[:, tgt].astype(str).values
    p = ref_df.iloc[:, tgt].astype(str).value_counts(normalize=True)
    maj, mino = p.idxmax(), p.idxmin()
    maj_idx = np.where(ay == maj)[0]
    min_idx = np.where(ay == mino)[0]
    n_min_target = min(int(round(len(maj_idx) * p[mino] / p[maj])), len(min_idx))
    sel = np.concatenate([maj_idx, rng.choice(min_idx, n_min_target, replace=False)])
    return aug.iloc[sel].reset_index(drop=True)


def main():
    print("=" * 66); print("048 Exp1 SMOTE baseline (SMOTENC, 5 seed)"); print("=" * 66)
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
    rows = []
    for ds, cfg in DS_CFG.items():
        info = load_info(cfg["info"]); train = pd.read_csv(cfg["train"]); test = pd.read_csv(cfg["test"])
        check_positive_class(info, test)
        tgt = info["target_col_idx"][0]
        counts = train.iloc[:, tgt].value_counts()
        print(f"\n-- {ds}: train class counts = {dict(counts)} (balance via SMOTENC) --")
        for seed in SEEDS:
            try:
                aug = smote_augment(train, info, seed)
            except Exception as e:
                print(f"[STOP] SMOTENC failed on {ds} seed {seed}: {e}"); raise
            # downstream F1/recall on real test
            auc, f1, rec = mle_auc_f1(aug, test, info, seed=seed)
            # dependency diagnostics vs real_test
            m = evaluate_all(aug, test, test, info, seed=seed, do_old=False, train_ref=train)
            dep_cross = m["xgb_c2st"] - coltype_c2st(test, aug, info, seed=seed, mode="cross")
            rows.append(dict(dataset=ds, variant="SMOTE", seed=seed, n_aug=len(aug),
                             full=m["xgb_c2st"], marg=m["marg_c2st"], dep=m["dep_c2st"],
                             dep_cross=dep_cross, mle_f1=f1, mle_recall=rec, mle_auc=auc))
            print(f"  SMOTE  {ds:8s} s{seed} n_aug={len(aug):>6} full={m['xgb_c2st']:.4f} "
                  f"dep={m['dep_c2st']:+.4f} dep_cross={dep_cross:+.4f} f1={f1:.4f} rec={rec:.4f}")
            # prior-matched dep diagnostic (comparable to TabbyFlow; F1 stays the balanced one above)
            pm = prior_match_subsample(aug, info, test, seed)
            mp = evaluate_all(pm, test, test, info, seed=seed, do_old=False, train_ref=train)
            dep_cross_pm = mp["xgb_c2st"] - coltype_c2st(test, pm, info, seed=seed, mode="cross")
            rows.append(dict(dataset=ds, variant="SMOTE_priormatch", seed=seed, n_aug=len(pm),
                             full=mp["xgb_c2st"], marg=mp["marg_c2st"], dep=mp["dep_c2st"],
                             dep_cross=dep_cross_pm, mle_f1=np.nan, mle_recall=np.nan, mle_auc=np.nan))
            print(f"  SMOTE-pm {ds:6s} s{seed} n={len(pm):>6} full={mp['xgb_c2st']:.4f} "
                  f"dep={mp['dep_c2st']:+.4f} dep_cross={dep_cross_pm:+.4f}")
        # RealTrain reference (F1 without SMOTE)
        for seed in SEEDS:
            auc, f1, rec = mle_auc_f1(train, test, info, seed=seed)
            rows.append(dict(dataset=ds, variant="RealTrain", seed=seed, n_aug=len(train),
                             full=np.nan, marg=np.nan, dep=np.nan, dep_cross=np.nan,
                             mle_f1=f1, mle_recall=rec, mle_auc=auc))

    long = pd.DataFrame(rows)
    long.to_csv("step_smote_long.csv", index=False)
    met = ["full", "marg", "dep", "dep_cross", "mle_f1", "mle_recall", "mle_auc"]
    g = long.groupby(["dataset", "variant"])[met].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g.reset_index().to_csv("step_smote.csv", index=False)
    print("\n[SAVED] step_smote.csv (+ _long)")
    print(g.reset_index()[["dataset","variant","dep_mean","dep_std","dep_cross_mean","mle_f1_mean","mle_recall_mean"]].to_string(index=False))
    print("\nDONE.")


if __name__ == "__main__":
    main()
