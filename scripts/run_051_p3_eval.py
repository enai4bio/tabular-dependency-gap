"""051 P2/P3: paired base-vs-fix (P3) evaluation. Per-seed Delta = fix - base
(same seed => only the mechanism differs). Report mean+/-std + sign-count for
dep_cross (P3's num<->cat target), minority F1/recall, full C2ST, dep.
Outcome-neutral: report whatever it is (P3 likely doesn't move the needle).
Usage: python run_051_p3_eval.py bank   (or default). Needs base_s0..4 + fix_s0..4.
Outputs: step_<ds>_p3_051.csv (+_long paired).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import load_info, evaluate_all, self_check, check_positive_class, DATASETS
from run_a1_coltype import coltype_c2st

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
SEEDS = [0, 1, 2, 3, 4]
DSS = {
    "bank":    {"train": "data/bank/train.csv",    "test": "data/bank/test.csv",    "info": "data/bank/info.json"},
    "default": {"train": "data/default/train.csv", "test": "data/default/test.csv", "info": "data/default/info.json"},
}


def metrics(syn, test, info, train, seed):
    m = evaluate_all(syn, test, test, info, seed=seed, do_old=False, train_ref=train)
    dc = m["xgb_c2st"] - coltype_c2st(test, syn, info, seed=seed, mode="cross")
    return dict(full=m["xgb_c2st"], dep=m["dep_c2st"], dep_cross=dc,
                mle_f1=m["mle_f1"], mle_recall=m["mle_recall"])


def main(ds):
    cfg = DSS[ds]
    print("=" * 60); print(f"051 P3 paired eval: {ds} (5 seed)"); print("=" * 60)
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
    info = load_info(cfg["info"]); train = pd.read_csv(cfg["train"]); test = pd.read_csv(cfg["test"])
    check_positive_class(info, test)
    rows = []
    for seed in SEEDS:
        bp = f"ef_vfm/result/{ds}/{ds}_base_s{seed}/8000/samples.csv"
        fp = f"ef_vfm/result/{ds}/{ds}_fix_s{seed}/8000/samples.csv"
        if not (os.path.exists(bp) and os.path.exists(fp)):
            print(f"  MISSING seed {seed} (base={os.path.exists(bp)} fix={os.path.exists(fp)})"); continue
        b = metrics(pd.read_csv(bp), test, info, train, seed)
        f = metrics(pd.read_csv(fp), test, info, train, seed)
        row = {"dataset": ds, "seed": seed}
        for k in ["full", "dep", "dep_cross", "mle_f1", "mle_recall"]:
            row[f"{k}_base"] = b[k]; row[f"{k}_fix"] = f[k]; row[f"{k}_delta"] = f[k] - b[k]
        rows.append(row)
        print(f"  s{seed} dep_cross {b['dep_cross']:+.4f}->{f['dep_cross']:+.4f} (Δ{f['dep_cross']-b['dep_cross']:+.4f}) "
              f"F1 {b['mle_f1']:.4f}->{f['mle_f1']:.4f} (Δ{f['mle_f1']-b['mle_f1']:+.4f}) "
              f"full {b['full']:.4f}->{f['full']:.4f} (Δ{f['full']-b['full']:+.4f})")
    long = pd.DataFrame(rows); long.to_csv(f"step_{ds}_p3_051_long.csv", index=False)
    # paired summary
    n = len(long); out = []
    for k in ["dep_cross", "mle_f1", "mle_recall", "full", "dep"]:
        d = long[f"{k}_delta"].values
        out.append(dict(metric=k, n=n, mean_delta=float(d.mean()),
                        std_delta=float(d.std(ddof=1)) if n > 1 else 0.0,
                        n_down=int((d < 0).sum()), n_up=int((d > 0).sum())))
    summ = pd.DataFrame(out); summ.to_csv(f"step_{ds}_p3_051.csv", index=False)
    print(f"\n=== {ds} P3 paired Δ (fix - base, n={n}) ===")
    for _, r in summ.iterrows():
        print(f"  {r['metric']:11s} Δ={r['mean_delta']:+.4f} ± {r['std_delta']:.4f}  ({int(r['n_down'])}↓/{int(r['n_up'])}↑)")
    # neutral P3 gate note (not a verdict): moved iff dep_cross down & F1 up consistently
    dc = long["dep_cross_delta"].values; f1 = long["mle_f1_delta"].values
    print(f"\n  [note] dep_cross↓ in {int((dc<0).sum())}/{n}, F1↑ in {int((f1>0).sum())}/{n} "
          f"(P3 'moved needle' would need both consistent; report as-is)")
    print("\nDONE.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bank")
