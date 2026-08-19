"""063 audit: adult P3 paired eval — same logic as run_051_p3_eval.py, adult was never
added to that script's DSS dict even though adult_base_s0-4 / adult_fix_s0-4 samples
both exist on disk. New file, does not modify run_051_p3_eval.py."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import load_info, evaluate_all, self_check, check_positive_class, DATASETS
from run_a1_coltype import coltype_c2st

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
SEEDS = [0, 1, 2, 3, 4]
cfg = {"train": "data/adult/train.csv", "test": "data/adult/test.csv", "info": "data/adult/info.json"}

def metrics(syn, test, info, train, seed):
    m = evaluate_all(syn, test, test, info, seed=seed, do_old=False, train_ref=train)
    dc = m["xgb_c2st"] - coltype_c2st(test, syn, info, seed=seed, mode="cross")
    return dict(full=m["xgb_c2st"], dep=m["dep_c2st"], dep_cross=dc,
                mle_f1=m["mle_f1"], mle_recall=m["mle_recall"])

assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
info = load_info(cfg["info"]); train = pd.read_csv(cfg["train"]); test = pd.read_csv(cfg["test"])
check_positive_class(info, test)
rows = []
for seed in SEEDS:
    bp = f"ef_vfm/result/adult/adult_base_s{seed}/8000/samples.csv"
    fp = f"ef_vfm/result/adult/adult_fix_s{seed}/8000/samples.csv"
    b = metrics(pd.read_csv(bp), test, info, train, seed)
    f = metrics(pd.read_csv(fp), test, info, train, seed)
    row = {"dataset": "adult", "seed": seed}
    for k in ["full", "dep", "dep_cross", "mle_f1", "mle_recall"]:
        row[f"{k}_base"] = b[k]; row[f"{k}_fix"] = f[k]; row[f"{k}_delta"] = f[k] - b[k]
    rows.append(row)
    print(f"  s{seed} dep_cross {b['dep_cross']:+.4f}->{f['dep_cross']:+.4f} (D{f['dep_cross']-b['dep_cross']:+.4f})")
long = pd.DataFrame(rows); long.to_csv("step_adult_p3_051_long.csv", index=False)
n = len(long); out = []
from scipy import stats
T = 2.776
for k in ["dep_cross", "mle_f1", "mle_recall", "full", "dep"]:
    d = long[f"{k}_delta"].values
    se = d.std(ddof=1)/np.sqrt(n)
    t, p = stats.ttest_1samp(d, 0)
    out.append(dict(metric=k, n=n, mean_delta=float(d.mean()), std_delta=float(d.std(ddof=1)),
                    ci_lo=float(d.mean()-T*se), ci_hi=float(d.mean()+T*se), t=float(t), p=float(p),
                    n_down=int((d<0).sum()), n_up=int((d>0).sum())))
summ = pd.DataFrame(out); summ.to_csv("step_adult_p3_051.csv", index=False)
print("\n=== adult P3 paired delta (fix-base, n=5) ===")
for _, r in summ.iterrows():
    print(f"  {r['metric']:11s} D={r['mean_delta']:+.4f}+-{r['std_delta']:.4f} CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] t={r['t']:.3f} p={r['p']:.4f} ({int(r['n_down'])}dn/{int(r['n_up'])}up)")
