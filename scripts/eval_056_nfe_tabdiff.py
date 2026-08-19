"""056 Task B: evaluate all TabDiff NFE-sweep sample files (adult+default) with the
standard proper_metrics pipeline. Reads ../TabDiff/result_056_nfe/manifest.csv.
"""
import os, sys
import numpy as np, pandas as pd
from proper_metrics import load_info, evaluate_all, self_check, check_positive_class, DATASETS
from run_a1_coltype import coltype_c2st

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
DS_CFG = {
    "adult":   {"train": "data/adult/train.csv",   "test": "data/adult/test.csv",   "info": "data/adult/info.json"},
    "default": {"train": "data/default/train.csv", "test": "data/default/test.csv", "info": "data/default/info.json"},
}
TD = "../TabDiff"


def main():
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
    man = pd.read_csv(f"{TD}/result_056_nfe/manifest.csv")
    rows = []
    cache = {}
    for _, r in man.iterrows():
        ds = r["dataset"]; seed = int(r["seed"]); steps = int(r["steps"])
        if ds not in cache:
            info = load_info(DS_CFG[ds]["info"])
            train = pd.read_csv(DS_CFG[ds]["train"]); test = pd.read_csv(DS_CFG[ds]["test"])
            check_positive_class(info, test)
            cache[ds] = (info, train, test)
        info, train, test = cache[ds]
        syn = pd.read_csv(f"{TD}/{r['path']}")
        m = evaluate_all(syn, test, test, info, seed=seed, do_old=False, train_ref=train)
        dc = m["xgb_c2st"] - coltype_c2st(test, syn, info, seed=seed, mode="cross")
        label = f"steps{steps}"
        rows.append(dict(dataset=ds, seed=seed, steps=steps, label=label, time_s=r["time_s"],
                          full=m["xgb_c2st"], marg=m["marg_c2st"], dep=m["dep_c2st"],
                          dep_cross=dc, mle_f1=m["mle_f1"], mle_recall=m["mle_recall"]))
        print(f"  {ds:8s} {label:10s} s{seed} full={m['xgb_c2st']:.4f} dep={m['dep_c2st']:+.4f} "
              f"dep_cross={dc:+.4f} f1={m['mle_f1']:.4f}")

    long = pd.DataFrame(rows)
    long.to_csv("step_056_nfe_sweep_tabdiff_long.csv", index=False)
    met = ["full", "marg", "dep", "dep_cross", "mle_f1", "mle_recall", "time_s"]
    g = long.groupby(["dataset", "steps"])[met].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]; g = g.reset_index().sort_values(["dataset", "steps"])
    g.to_csv("step_056_nfe_sweep_tabdiff.csv", index=False)
    print("\n[SAVED] step_056_nfe_sweep_tabdiff.csv (+_long)")


if __name__ == "__main__":
    main()
