"""056 Task B: evaluate all NFE-sweep sample files (adult+default) with the standard
proper_metrics pipeline. Reads result_056_nfe/manifest.csv, evaluates each path,
appends metrics, writes step_056_nfe_sweep_long.csv + summary + stats.
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import load_info, evaluate_all, self_check, check_positive_class, DATASETS
from run_a1_coltype import coltype_c2st

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
DS_CFG = {
    "adult":   {"train": "data/adult/train.csv",   "test": "data/adult/test.csv",   "info": "data/adult/info.json"},
    "default": {"train": "data/default/train.csv", "test": "data/default/test.csv", "info": "data/default/info.json"},
}


def main():
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
    man = pd.read_csv("result_056_nfe/manifest.csv")
    rows = []
    cache = {}
    for _, r in man.iterrows():
        ds = r["dataset"]; seed = int(r["seed"])
        if ds not in cache:
            info = load_info(DS_CFG[ds]["info"])
            train = pd.read_csv(DS_CFG[ds]["train"]); test = pd.read_csv(DS_CFG[ds]["test"])
            check_positive_class(info, test)
            cache[ds] = (info, train, test)
        info, train, test = cache[ds]
        syn = pd.read_csv(r["path"])
        m = evaluate_all(syn, test, test, info, seed=seed, do_old=False, train_ref=train)
        dc = m["xgb_c2st"] - coltype_c2st(test, syn, info, seed=seed, mode="cross")
        steps = r["steps"] if not pd.isna(r["steps"]) else np.nan
        label = "dopri5_default" if r["method"] == "dopri5_default" else f"euler{int(steps)}"
        rows.append(dict(dataset=ds, seed=seed, method=r["method"], steps=steps, label=label,
                          nfe=r["nfe"], time_s=r["time_s"],
                          full=m["xgb_c2st"], marg=m["marg_c2st"], dep=m["dep_c2st"],
                          dep_cross=dc, mle_f1=m["mle_f1"], mle_recall=m["mle_recall"]))
        print(f"  {ds:8s} {label:16s} s{seed} full={m['xgb_c2st']:.4f} dep={m['dep_c2st']:+.4f} "
              f"dep_cross={dc:+.4f} f1={m['mle_f1']:.4f} nfe={r['nfe']:.0f}")

    long = pd.DataFrame(rows)
    long.to_csv("step_056_nfe_sweep_long.csv", index=False)
    met = ["full", "marg", "dep", "dep_cross", "mle_f1", "mle_recall", "nfe", "time_s"]
    g = long.groupby(["dataset", "label"])[met].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]; g = g.reset_index()
    g.to_csv("step_056_nfe_sweep.csv", index=False)
    print("\n[SAVED] step_056_nfe_sweep.csv (+_long)")


if __name__ == "__main__":
    main()
