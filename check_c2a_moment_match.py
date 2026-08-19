"""045 Check 2a — direct 2nd-order num<->cat moment match (CPU, zero training).

THE precondition for Option A (Cov-on-mu, a 2nd-order fix): does TabbyFlow ALREADY
match real's 2nd-order num<->cat cross moments? If yes, the residual XGB-C2ST gap is
higher-order -> Cov-on-mu has nothing to fix -> A dead (-> B). (044 + 033 prior.)

Statistic: for each (num_i, cat_j) pair, the vector over cat_j's categories of the
STANDARDIZED conditional mean E[num_i | cat_j=c] (standardized by real_test's global
num_i mean/std). This is exactly the linear/2nd-order num-cat association. Per-pair
discrepancy = freq-weighted RMS of (real - other) over categories; aggregate = mean
over all (num_i, cat_j) pairs.

Anchors (C2ST-style floor/upper):
  floor  = d(real_test, real_train)  -- two independent real samples (sampling noise)
  syn    = d(real_test, TabbyFlow)   -- 5 seeds, mean+/-std  (what we judge)
  FF     = d(real_test, FF)          -- columns shuffled => dependency destroyed (upper)

Verdict: syn ~ floor  => already matched => A DEAD.
         syn -> FF     => big 2nd-order error => A has headroom (go to check 1).
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import load_info, num_columns, cat_columns, make_ff

BASE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]


def load_pos(path):
    df = pd.read_csv(path)
    df.columns = range(len(df.columns))
    return df


def cond_means(df, num_idx, cat_idx, levels, num_mu, num_sd):
    """Return dict (i,j) -> vector over levels[j] of standardized E[num_i | cat_j=c]."""
    out = {}
    xs_cache = {i: (pd.to_numeric(df[i], errors="coerce").values - num_mu[i]) / num_sd[i]
                for i in num_idx}
    for j in cat_idx:
        cj = df[j].astype(str).values
        for i in num_idx:
            xs = xs_cache[i]
            vec = np.zeros(len(levels[j]))
            for ci, c in enumerate(levels[j]):
                mask = cj == c
                vec[ci] = xs[mask].mean() if mask.sum() > 0 else 0.0  # absent => global mean (0)
            out[(i, j)] = vec
    return out


def discrepancy(mD, mReal, num_idx, cat_idx, freq):
    perpair = []
    for i in num_idx:
        for j in cat_idx:
            d = mReal[(i, j)] - mD[(i, j)]
            perpair.append(float(np.sqrt(np.sum(freq[j] * d * d))))
    return float(np.mean(perpair)), perpair


def main():
    os.chdir(BASE)
    info = load_info("data/adult/info.json")
    num_idx = num_columns(info)
    cat_idx = cat_columns(info)        # includes target for binclass
    print("=" * 64)
    print("Check 2a: 2nd-order num<->cat moment match (adult)")
    print(f"  num cols={num_idx}  cat cols (incl target)={cat_idx}  "
          f"=> {len(num_idx)*len(cat_idx)} pairs")
    print("=" * 64)

    real = load_pos("data/adult/test.csv")
    train = load_pos("data/adult/train.csv")

    # reference stats from real_test
    num_mu = {i: float(pd.to_numeric(real[i], errors="coerce").mean()) for i in num_idx}
    num_sd = {i: float(pd.to_numeric(real[i], errors="coerce").std() + 1e-12) for i in num_idx}
    levels, freq = {}, {}
    for j in cat_idx:
        vc = real[j].astype(str).value_counts()
        levels[j] = list(vc.index)
        freq[j] = (vc.values / vc.values.sum()).astype(float)

    mReal = cond_means(real, num_idx, cat_idx, levels, num_mu, num_sd)

    # floor: real_test vs real_train
    mTrain = cond_means(train, num_idx, cat_idx, levels, num_mu, num_sd)
    d_floor, _ = discrepancy(mTrain, mReal, num_idx, cat_idx, freq)

    # upper: FF (columns independently shuffled -> num-cat dependency destroyed)
    ff = make_ff(train, len(train), seed=42)
    ff.columns = range(len(ff.columns))
    mFF = cond_means(ff, num_idx, cat_idx, levels, num_mu, num_sd)
    d_ff, _ = discrepancy(mFF, mReal, num_idx, cat_idx, freq)

    # syn: 5 TabbyFlow seeds
    d_syn = []
    for s in SEEDS:
        p = f"ef_vfm/result/adult/adult_base_s{s}/8000/samples.csv"
        if not os.path.exists(p):
            print(f"  MISSING {p}"); continue
        mS = cond_means(load_pos(p), num_idx, cat_idx, levels, num_mu, num_sd)
        d, _ = discrepancy(mS, mReal, num_idx, cat_idx, freq)
        d_syn.append(d)
        print(f"  syn s{s}: d={d:.4f}")
    d_syn = np.array(d_syn)

    print("-" * 64)
    print(f"  floor  d(real_test, real_train) = {d_floor:.4f}   (sampling-noise floor)")
    print(f"  syn    d(real_test, TabbyFlow)  = {d_syn.mean():.4f} ± {d_syn.std(ddof=1):.4f}")
    print(f"  FF     d(real_test, FF)         = {d_ff:.4f}   (dependency-destroyed upper)")
    # fraction of the floor->FF range that syn still misses
    frac = (d_syn.mean() - d_floor) / (d_ff - d_floor + 1e-12)
    print(f"  syn position floor->FF: {frac*100:.1f}%   (0% = matched, 100% = FF-like)")
    print("-" * 64)
    if d_syn.mean() <= d_floor + (d_ff - d_floor) * 0.15:
        print("  VERDICT: TabbyFlow ~ matches 2nd-order num<->cat moments (near floor)")
        print("           => residual gap is HIGHER-ORDER => Cov-on-mu has little to fix")
        print("           => Option A DEAD precondition (-> B). Confirm with 2b.")
    else:
        print("  VERDICT: TabbyFlow MISSES 2nd-order num<->cat moments (well above floor)")
        print("           => 2nd-order headroom exists => A precondition OK, go to check 1.")

    pd.DataFrame({
        "anchor": ["floor_train", "syn_mean", "syn_std", "FF_upper", "frac_floor_to_FF"],
        "value": [d_floor, float(d_syn.mean()), float(d_syn.std(ddof=1)), d_ff, float(frac)],
    }).to_csv("step_c2a_moment_match.csv", index=False)
    print("[SAVED] step_c2a_moment_match.csv")


if __name__ == "__main__":
    main()
