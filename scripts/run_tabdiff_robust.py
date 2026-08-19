"""adult 判别器鲁棒性:用 2 组额外 XGB 配置(小/大)重跑 FF/TabbyFlow/TabDiff/oracle
的 full/dep/dep_cross,验证 dep 符号 + FF>gen>oracle 排序不随判别器容量变化。
主结果用默认配置(n_est=300,max_depth=6),此为附加检查。只 adult。5 种子 mean±std。
复用 proper_metrics 底层(xgb_c2st 带参) + shuffle helpers。tabbyflow env。
输出 step_tabdiff_robust.csv。
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import (load_info, xgb_c2st, _shuffle_cols, make_ff, make_oracle,
                            num_columns, cat_columns, self_check, DATASETS)
from run_a1_coltype import _block_shuffle

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
TD = "/media/jie/expand_5t/7exp/next/TabDiff/tabdiff/result"
EF = "ef_vfm/result/adult"
SEEDS = [0, 1, 2, 3, 4]
CFG = {"train": "data/adult/train.csv", "test": "data/adult/test.csv", "info": "data/adult/info.json"}
# 判别器配置:主结果=default(300,6);鲁棒性=small/large
XGB_CFGS = {"small(100,3)": dict(n_estimators=100, max_depth=3),
            "large(600,10)": dict(n_estimators=600, max_depth=10)}


def decomp(test, syn, info, seed, xgb_kw):
    """full/marg/dep/dep_cross 用指定 XGB 配置。"""
    num_idx, cat_idx = num_columns(info), cat_columns(info)
    full = xgb_c2st(test, syn, info, seed=seed, **xgb_kw)
    marg = xgb_c2st(_shuffle_cols(test, seed + 1), _shuffle_cols(syn, seed + 2), info, seed=seed, **xgb_kw)
    cross = xgb_c2st(_block_shuffle(test, num_idx, cat_idx, seed + 50),
                     _block_shuffle(syn, num_idx, cat_idx, seed + 60), info, seed=seed, **xgb_kw)
    return full, full - marg, full - cross  # full, dep, dep_cross


def main():
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
    info = load_info(CFG["info"]); train = pd.read_csv(CFG["train"]); test = pd.read_csv(CFG["test"])
    rows = []
    for cfg_name, xgb_kw in XGB_CFGS.items():
        print(f"\n== XGB {cfg_name} ==")
        for s in SEEDS:
            tabdiff = pd.read_csv(f"{TD}/adult/adult_tabdiff_s{s}/8000/samples.csv")
            tabby = pd.read_csv(f"{EF}/adult_base_s{s}/8000/samples.csv")
            n = len(tabdiff)
            variants = {"FF": make_ff(train, n, seed=s), "TabbyFlow": tabby,
                        "TabDiff": tabdiff, "Oracle": make_oracle(train, n, seed=s)}
            for name, sdf in variants.items():
                full, dep, dc = decomp(test, sdf, info, s, xgb_kw)
                rows.append(dict(xgb_cfg=cfg_name, variant=name, seed=s, full=full, dep=dep, dep_cross=dc))
            print(f"  s{s} done")
    long = pd.DataFrame(rows); long.to_csv("step_tabdiff_robust_long.csv", index=False)
    g = long.groupby(["xgb_cfg", "variant"])[["full", "dep", "dep_cross"]].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]; g = g.reset_index()
    g["_o"] = g.variant.map({"FF": 0, "TabbyFlow": 1, "TabDiff": 2, "Oracle": 3})
    g = g.sort_values(["xgb_cfg", "_o"]).drop(columns="_o"); g.to_csv("step_tabdiff_robust.csv", index=False)
    print("\n=== adult 判别器鲁棒性(full / dep, 5种子mean) ===")
    for _, r in g.iterrows():
        print(f"  [{r['xgb_cfg']:12s}] {r['variant']:10s} full={r['full_mean']:.4f} dep={r['dep_mean']:+.4f} dep_cross={r['dep_cross_mean']:+.4f}")
    print("\n[SAVED] step_tabdiff_robust.csv\nDONE.")


if __name__ == "__main__":
    main()
