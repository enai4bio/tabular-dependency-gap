"""补 magic 的 FF/oracle 参照到 5 seed(此前047只做了3 seed),使 magic 与 adult/default/bank
一样全 5 seed。generator(TabbyFlow, magic_base_s0-4)本就5 seed。复用 proper_metrics+FF/oracle锚。
输出 step_magic_diag_seed5.csv(+_long,+_stats:TabbyFlow dep t检验/95%CI)。tabbyflow env。
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import (load_info, evaluate_all, make_ff, make_oracle,
                            self_check, check_positive_class, DATASETS)
from run_a1_coltype import coltype_c2st

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
SEEDS = [0, 1, 2, 3, 4]
CFG = {"train": "data/magic/train.csv", "test": "data/magic/test.csv", "info": "data/magic/info.json"}
T_975_df4 = 2.776


def main():
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
    info = load_info(CFG["info"]); train = pd.read_csv(CFG["train"]); test = pd.read_csv(CFG["test"])
    check_positive_class(info, test)
    rows = []
    for s in SEEDS:
        p = f"ef_vfm/result/magic/magic_base_s{s}/8000/samples.csv"
        syn = pd.read_csv(p); n = len(syn)
        variants = {"FF": make_ff(train, n, seed=s), "TabbyFlow": syn, "Oracle": make_oracle(train, n, seed=s)}
        for name, sdf in variants.items():
            m = evaluate_all(sdf, test, test, info, seed=s, do_old=False, train_ref=train)
            dc = m["xgb_c2st"] - coltype_c2st(test, sdf, info, seed=s, mode="cross")
            rows.append(dict(dataset="magic", variant=name, seed=s, full=m["xgb_c2st"], marg=m["marg_c2st"],
                             dep=m["dep_c2st"], dep_cross=dc, mle_f1=m["mle_f1"], mle_recall=m["mle_recall"]))
        print(f"  magic s{s} done")
    long = pd.DataFrame(rows); long.to_csv("step_magic_diag_seed5_long.csv", index=False)
    met = ["full", "marg", "dep", "dep_cross", "mle_f1", "mle_recall"]
    g = long.groupby(["dataset", "variant"])[met].agg(["mean", "std"]); g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index(); g["_o"] = g.variant.map({"FF": 0, "TabbyFlow": 1, "Oracle": 2})
    g = g.sort_values("_o").drop(columns="_o"); g.to_csv("step_magic_diag_seed5.csv", index=False)
    # TabbyFlow dep t检验+CI
    v = long[long.variant == "TabbyFlow"]["dep"].values
    t, pval = stats.ttest_1samp(v, 0); se = v.std(ddof=1)/np.sqrt(len(v))
    pd.DataFrame([dict(dataset="magic", variant="TabbyFlow", metric="dep", mean=v.mean(), std=v.std(ddof=1),
                       ci95_lo=v.mean()-T_975_df4*se, ci95_hi=v.mean()+T_975_df4*se, t_stat=t, p_value=pval, n=len(v))]
                 ).to_csv("step_magic_diag_seed5_stats.csv", index=False)
    print("\n=== magic 5 seed 诊断(对比047的3 seed) ===")
    for _, r in g.iterrows():
        print(f"  {r['variant']:10s} full={r['full_mean']:.4f} dep={r['dep_mean']:+.4f}±{r['dep_std']:.4f} dep_cross={r['dep_cross_mean']:+.4f} F1={r['mle_f1_mean']:.4f}")
    print(f"  TabbyFlow dep t检验: mean={v.mean():+.4f} 95%CI[{v.mean()-T_975_df4*se:+.4f},{v.mean()+T_975_df4*se:+.4f}] t={t:.3f} p={pval:.4f}")
    print("\n[SAVED] step_magic_diag_seed5.csv (+_long,+_stats)\nDONE.")


if __name__ == "__main__":
    main()
