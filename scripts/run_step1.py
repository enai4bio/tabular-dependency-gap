"""040 腿二: 多 seed 依赖分解诊断。复用 proper_metrics 的金标准实现。"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proper_metrics import (load_info, evaluate_all, make_ff, make_oracle,
                            xgb_c2st, self_check, DATASETS)

SEEDS = [0, 1, 2, 3, 4]
HEADLINE = ["adult", "shoppers"]          # diabetes 仅作噪声参照,不进 headline


def main():
    base = os.path.dirname(os.path.abspath(__file__)); os.chdir(base)

    # ── gate 1: 确定性自检(不过就停)──
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED → STOP"

    rows = []
    for ds, cfg in DATASETS.items():
        info = load_info(cfg["info"])
        train = pd.read_csv(cfg["train"]); test = pd.read_csv(cfg["test"])
        syn   = pd.read_csv(cfg["syn"])

        for seed in SEEDS:
            # ── gate 2: real-vs-real sanity(每 seed 都算,应 ≈0.5)──
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(test)); h = len(test) // 2
            rr = xgb_c2st(test.iloc[perm[:h]].reset_index(drop=True),
                          test.iloc[perm[h:]].reset_index(drop=True), info, seed=seed)

            variants = {
                "FF":        make_ff(train, len(syn), seed=seed),       # 下界, 每 seed 重建
                "TabbyFlow": syn,                                        # 固定单一实现
                "Oracle":    make_oracle(train, len(syn), seed=seed),   # 上界, 每 seed 重建
            }
            for name, sdf in variants.items():
                m = evaluate_all(sdf, test, test, info, seed=seed,
                                 do_old=True, train_ref=train)
                m.update(dataset=ds, variant=name, seed=seed, rr_c2st=rr)
                rows.append(m)
                print(f"{ds:9s} {name:10s} s{seed} "
                      f"full={m['xgb_c2st']:.4f} marg={m['marg_c2st']:.4f} "
                      f"dep={m['dep_c2st']:+.4f} f1={m['mle_f1']:.4f} "
                      f"rec={m['mle_recall']:.4f} "
                      f"trend={m['trend']:.4f} lr={m['lr_c2st']:.4f} rr={rr:.4f}")

    long = pd.DataFrame(rows)
    long.to_csv("step1_decomp_long.csv", index=False)

    # 汇总 mean±std
    metrics = ["xgb_c2st", "marg_c2st", "dep_c2st", "tv2", "tv3",
               "mle_auc", "mle_f1", "mle_recall", "trend", "lr_c2st", "rr_c2st"]
    g = long.groupby(["dataset", "variant"])[metrics]
    summ = g.agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv("step1_decomp_summary.csv", index=False)
    print("\n[SAVED] step1_decomp_long.csv, step1_decomp_summary.csv")

    # ── 打印 sanity gate 检查 ──
    print("\n" + "=" * 70)
    print("SANITY GATE CHECK")
    print("=" * 70)
    summ_df = pd.read_csv("step1_decomp_summary.csv")

    for ds in ["adult", "shoppers", "diabetes"]:
        for var in ["FF", "TabbyFlow", "Oracle"]:
            row = summ_df[(summ_df.dataset == ds) & (summ_df.variant == var)]
            if row.empty:
                continue
            full_m = float(row.xgb_c2st_mean)
            dep_m  = float(row.dep_c2st_mean)
            rr_m   = float(row.rr_c2st_mean)
            gap    = full_m - 0.5
            dep_ratio = dep_m / gap if gap > 0.001 else float("nan")
            print(f"  {ds:9s} {var:10s} full={full_m:.4f} dep={dep_m:+.4f} "
                  f"rr={rr_m:.4f} dep_ratio={dep_ratio:.2f}")

    print()
    print("Gate 2 (rr ≈ 0.5±0.02 for adult/shoppers):")
    for ds in ["adult", "shoppers"]:
        row = summ_df[(summ_df.dataset == ds) & (summ_df.variant == "TabbyFlow")]
        rr = float(row.rr_c2st_mean)
        ok = abs(rr - 0.5) <= 0.02
        print(f"  {ds}: rr_c2st_mean={rr:.4f}  {'PASS' if ok else 'FAIL !!!'}")

    print("Gate 3 (FF.dep >> Oracle.dep per dataset):")
    for ds in ["adult", "shoppers"]:
        ff_dep = float(summ_df[(summ_df.dataset == ds) & (summ_df.variant == "FF")].dep_c2st_mean)
        or_dep = float(summ_df[(summ_df.dataset == ds) & (summ_df.variant == "Oracle")].dep_c2st_mean)
        ok = ff_dep > 0.2 and abs(or_dep) < 0.02
        print(f"  {ds}: FF.dep={ff_dep:+.4f}  Oracle.dep={or_dep:+.4f}  {'PASS' if ok else 'FAIL !!!'}")

    print("Gate 4 (TabbyFlow full ∈ [0.58,0.62], dep_ratio ∈ [0.4,0.75]):")
    for ds in ["adult", "shoppers"]:
        row = summ_df[(summ_df.dataset == ds) & (summ_df.variant == "TabbyFlow")]
        full = float(row.xgb_c2st_mean)
        dep  = float(row.dep_c2st_mean)
        gap  = full - 0.5
        ratio = dep / gap if gap > 0.001 else float("nan")
        ok = (0.58 <= full <= 0.62) and (0.4 <= ratio <= 0.75)
        print(f"  {ds}: full={full:.4f} dep={dep:+.4f} ratio={ratio:.2f}  {'PASS' if ok else 'FAIL !!!'}")

    print("Gate 5 (FF: lr_c2st ≈ 1.0 >> xgb_c2st when xgb_c2st is high):")
    for ds in ["adult", "shoppers"]:
        row = summ_df[(summ_df.dataset == ds) & (summ_df.variant == "FF")]
        xgb = float(row.xgb_c2st_mean)
        lr  = float(row.lr_c2st_mean)
        tr  = float(row.trend_mean)
        ok = lr > 0.99 and xgb > 0.95
        print(f"  {ds}: FF xgb_c2st={xgb:.4f} lr_c2st={lr:.4f} trend={tr:.4f}  {'PASS' if ok else 'FAIL !!!'}")


if __name__ == "__main__":
    main()
