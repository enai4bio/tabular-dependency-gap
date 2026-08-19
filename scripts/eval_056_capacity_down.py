"""056 Task C: evaluate downward capacity sweep (1/2x, 1/4x, 1/8x width, 5 seeds each,
adult) with the standard proper_metrics pipeline, and pull final training loss + param
count from each run's log/config for the convergence check (spec C.2/C.3)."""
import os, sys, re, glob, pickle
import numpy as np, pandas as pd
from proper_metrics import load_info, evaluate_all, self_check, check_positive_class, DATASETS
from run_a1_coltype import coltype_c2st

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
WIDTHS = {"0.125x": 128, "0.25x": 256, "0.5x": 512}
SEEDS = [0, 1, 2, 3, 4]


def final_loss_and_params(exp):
    ckpt_dir = f"ef_vfm/ckpt/adult/{exp}"
    log_candidates = ["log_056_capacity_down_A.txt", "log_056_capacity_down_B.txt",
                       "log_056_capacity_down_C.txt", "log_056_capacity_down_pilot.txt"]
    final_loss = None
    for log in log_candidates:
        if not os.path.exists(log):
            continue
        with open(log, errors="ignore") as f:
            text = f.read()
        # isolate this exp's training block
        m = re.search(re.escape(f"TRAIN {exp} ") + r".*?(?=\n===|\Z)", text, re.S)
        if m:
            losses = re.findall(r"Epoch (\d+)/8000:\s*100%.*?TotalLoss=([0-9.]+)", m.group(0))
            if losses:
                final_loss = float(losses[-1][1])
                break
    with open(f"{ckpt_dir}/config.pkl", "rb") as f:
        cfg = pickle.load(f)
    return final_loss, cfg


def main():
    assert self_check(DATASETS["adult"]), "determinism self-check FAILED -> STOP"
    info = load_info("data/adult/info.json")
    train = pd.read_csv("data/adult/train.csv"); test = pd.read_csv("data/adult/test.csv")
    check_positive_class(info, test)

    rows = []
    for wname, dimt in WIDTHS.items():
        for seed in SEEDS:
            exp = f"adult_cap{wname}_s{seed}"
            syn = pd.read_csv(f"ef_vfm/result/adult/{exp}/8000/samples.csv")
            m = evaluate_all(syn, test, test, info, seed=seed, do_old=False, train_ref=train)
            dc = m["xgb_c2st"] - coltype_c2st(test, syn, info, seed=seed, mode="cross")
            final_loss, cfg = final_loss_and_params(exp)
            state = pickle_state = None
            model_pt = f"ef_vfm/ckpt/adult/{exp}/model_8000.pt"
            import torch
            sd = torch.load(model_pt, map_location="cpu")["vf_fn"]
            nparams = sum(v.numel() for v in sd.values())
            rows.append(dict(width=wname, dim_t=dimt, seed=seed, full=m["xgb_c2st"],
                              marg=m["marg_c2st"], dep=m["dep_c2st"], dep_cross=dc,
                              mle_f1=m["mle_f1"], mle_recall=m["mle_recall"],
                              final_loss=final_loss, n_params=nparams))
            print(f"  {wname:8s} s{seed} full={m['xgb_c2st']:.4f} dep={m['dep_c2st']:+.4f} "
                  f"dep_cross={dc:+.4f} f1={m['mle_f1']:.4f} loss={final_loss} params={nparams}")

    long = pd.DataFrame(rows)
    long.to_csv("step_056_capacity_down_long.csv", index=False)
    met = ["full", "marg", "dep", "dep_cross", "mle_f1", "mle_recall", "final_loss", "n_params"]
    g = long.groupby(["width", "dim_t"])[met].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]; g = g.reset_index()
    g = g.sort_values("dim_t")
    g.to_csv("step_056_capacity_down.csv", index=False)
    print("\n[SAVED] step_056_capacity_down.csv (+_long)")
    print(g[["width", "dep_mean", "dep_std", "final_loss_mean", "n_params_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
