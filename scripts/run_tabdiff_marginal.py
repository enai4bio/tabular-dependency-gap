"""TabDiff 边际指标对照(读 dep 的前提):Shape/Trend(TabDiff训练时SDMetrics算,同原论文口径)
+ MLE(proper_metrics,分类AUC),adult/default 5种子 mean±std。用于对比原论文报告值,
证明 TabDiff 被正常训练。用 tabbyflow env 跑(proper_metrics)。
输出 step_tabdiff_marginal.csv。
"""
import os, sys, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import load_info, mle_auc_f1

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
TD = "/media/jie/expand_5t/7exp/next/TabDiff/tabdiff/result"
SEEDS = [0, 1, 2, 3, 4]
DSS = {"adult":   {"test": "data/adult/test.csv",   "info": "data/adult/info.json"},
       "default": {"test": "data/default/test.csv", "info": "data/default/info.json"}}


def main():
    rows = []
    for ds, cfg in DSS.items():
        info = load_info(cfg["info"]); test = pd.read_csv(cfg["test"])
        for s in SEEDS:
            base = f"{TD}/{ds}/{ds}_tabdiff_s{s}/8000"
            samp = f"{base}/samples.csv"
            if not os.path.exists(samp):
                continue
            # Shape/Trend: TabDiff训练时SDMetrics算的(同原论文)
            j = json.load(open(f"{base}/all_results.json"))
            shape, trend = j["density/Shape"], j["density/Trend"]
            # MLE: proper_metrics 分类AUC(syn训练,real test评估)
            syn = pd.read_csv(samp)
            auc, f1, rec = mle_auc_f1(syn, test, info, seed=s)
            rows.append(dict(dataset=ds, seed=s, Shape=shape, Trend=trend, MLE_AUC=auc, F1=f1, recall=rec))
            print(f"  {ds} s{s}: Shape={shape:.4f} Trend={trend:.4f} MLE-AUC={auc:.4f} F1={f1:.4f}")
    long = pd.DataFrame(rows); long.to_csv("step_tabdiff_marginal_long.csv", index=False)
    g = long.groupby("dataset")[["Shape", "Trend", "MLE_AUC", "F1", "recall"]].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g.reset_index().to_csv("step_tabdiff_marginal.csv", index=False)
    print("\n=== TabDiff 边际指标 mean±std ===")
    for _, r in g.reset_index().iterrows():
        print(f"  {r['dataset']}: Shape={r['Shape_mean']:.4f}±{r['Shape_std']:.4f} "
              f"Trend={r['Trend_mean']:.4f}±{r['Trend_std']:.4f} MLE-AUC={r['MLE_AUC_mean']:.4f}±{r['MLE_AUC_std']:.4f}")
    print("\n[SAVED] step_tabdiff_marginal.csv\nDONE.")


if __name__ == "__main__":
    main()
