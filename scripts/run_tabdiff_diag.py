"""TabDiff 诊断分解:FF/TabDiff/oracle 三行的 full/marg/dep/dep_cross/少数类F1,
adult/default 5种子 mean±std,加 dep/dep_cross 的单样本 t 检验(H0=0)+ 95%CI(t分布 df=4)。
复用现有 proper_metrics(判别器配置与 TabbyFlow 完全一致) + FF/oracle 锚(数据侧,与生成器无关)。
用 tabbyflow env 跑。输出 step_tabdiff_diag.csv(+_long)。
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import (load_info, evaluate_all, make_ff, make_oracle,
                            self_check, check_positive_class, DATASETS)
from run_a1_coltype import coltype_c2st

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
TD = "/media/jie/expand_5t/7exp/next/TabDiff/tabdiff/result"
SEEDS = [0, 1, 2, 3, 4]
DSS = {"adult":   {"train": "data/adult/train.csv",   "test": "data/adult/test.csv",   "info": "data/adult/info.json"},
       "default": {"train": "data/default/train.csv", "test": "data/default/test.csv", "info": "data/default/info.json"}}
T_975_df4 = 2.776  # t_{0.975, df=4} for 95% CI


def ci95(v):
    v = np.asarray(v); m = v.mean(); sd = v.std(ddof=1); se = sd / np.sqrt(len(v))
    t, p = stats.ttest_1samp(v, 0.0)
    return m, sd, m - T_975_df4 * se, m + T_975_df4 * se, float(t), float(p)


def main():
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
    diag_rows, stat_rows = [], []
    for ds, cfg in DSS.items():
        info = load_info(cfg["info"]); train = pd.read_csv(cfg["train"]); test = pd.read_csv(cfg["test"])
        check_positive_class(info, test)
        # 只在 5 个种子样本齐时才评
        paths = [f"{TD}/{ds}/{ds}_tabdiff_s{s}/8000/samples.csv" for s in SEEDS]
        if not all(os.path.exists(p) for p in paths):
            print(f"[skip] {ds}: TabDiff 样本未齐(需 5 种子),跳过"); continue
        print(f"\n== {ds} ==")
        for s, p in zip(SEEDS, paths):
            syn = pd.read_csv(p); n = len(syn)
            variants = {"FF": make_ff(train, n, seed=s), "TabDiff": syn, "Oracle": make_oracle(train, n, seed=s)}
            for name, sdf in variants.items():
                m = evaluate_all(sdf, test, test, info, seed=s, do_old=False, train_ref=train)
                dc = m["xgb_c2st"] - coltype_c2st(test, sdf, info, seed=s, mode="cross")
                diag_rows.append(dict(dataset=ds, variant=name, seed=s, full=m["xgb_c2st"], marg=m["marg_c2st"],
                                      dep=m["dep_c2st"], dep_cross=dc, mle_f1=m["mle_f1"], mle_recall=m["mle_recall"]))
            print(f"  s{s} done")
        # TabDiff 的 dep/dep_cross t 检验 + 95%CI
        sub = pd.DataFrame([r for r in diag_rows if r["dataset"] == ds and r["variant"] == "TabDiff"])
        for metric in ["dep", "dep_cross"]:
            m, sd, lo, hi, t, pval = ci95(sub[metric].values)
            stat_rows.append(dict(dataset=ds, variant="TabDiff", metric=metric, mean=m, std=sd,
                                  ci95_lo=lo, ci95_hi=hi, t_stat=t, p_value=pval, n=len(sub)))
            print(f"  TabDiff {metric}: {m:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] t={t:.2f} p={pval:.4f}")

    long = pd.DataFrame(diag_rows); long.to_csv("step_tabdiff_diag_long.csv", index=False)
    met = ["full", "marg", "dep", "dep_cross", "mle_f1", "mle_recall"]
    g = long.groupby(["dataset", "variant"])[met].agg(["mean", "std"]); g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index(); g["_o"] = g.variant.map({"FF": 0, "TabDiff": 1, "Oracle": 2})
    g = g.sort_values(["dataset", "_o"]).drop(columns="_o"); g.to_csv("step_tabdiff_diag.csv", index=False)
    pd.DataFrame(stat_rows).to_csv("step_tabdiff_diag_stats.csv", index=False)
    print("\n[SAVED] step_tabdiff_diag.csv (+_long, +_stats)\nDONE.")


if __name__ == "__main__":
    main()
