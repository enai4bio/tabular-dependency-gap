"""056 Task B: standalone NFE (sampling step-count) sweep for TabDiff.

Purely ADDITIVE / read-only w.r.t. existing code: reuses UnifiedCtimeDiffusion,
UniModMLP, Model unmodified; only the `num_timesteps` value passed into a freshly
constructed UnifiedCtimeDiffusion instance is varied. Does not edit tabdiff/main.py
or tabdiff/trainer.py, so no default training/testing path is touched.

Env: /home/jie/anaconda3/envs/tabdiff/bin/python (this repo's TabDiff env).
Usage:
    cd TabDiff
    python run_056_nfe_resample_tabdiff.py --verify_only
    python run_056_nfe_resample_tabdiff.py
"""
import os, sys, pickle, glob, json, time, argparse
import numpy as np
import pandas as pd
import torch

# Match the global determinism flags tabdiff/main.py sets once at process start
# under --deterministic (lines ~97-115), which the published *_tabdiff_s* runs used.
# Must be set before any CUDA context / matmul happens.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BASE = os.path.dirname(os.path.abspath(__file__))
# Match queue_tabdiff.sh's actual training cwd (TabDiff/, the OUTER dir -- not
# TabDiff/tabdiff/), since main.py's `data/{dataname}` is CWD-relative while
# ckpt/result paths are relative to tabdiff/main.py's own __file__ location
# (hence the "tabdiff/" prefix below).
os.chdir(BASE)
sys.path.insert(0, BASE)

from tabdiff.modules.main_modules import UniModMLP, Model
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.trainer import split_num_cat_target, recover_data
from utils_train import TabDiffDataset

DATASETS = ["adult", "default"]
SEEDS = [0, 1, 2, 3, 4]
STEP_COUNTS = [10, 20, 50, 200, 500]  # 50 == the trained default (verified from config.pkl)
OUT_DIR = "result_056_nfe"
os.makedirs(OUT_DIR, exist_ok=True)


def load_diffusion(ds, seed, device, num_timesteps=None, ckpt_name="model_8000.pt"):
    """ckpt_name="model_8000.pt": the LIVE non-EMA weights at the final training step,
    matching how tabdiff/trainer.py:276 (evaluate_generation(ema=False)) produced the
    published result/{ds}/{ds}_tabdiff_s{seed}/8000/samples.csv (same pattern verified
    for EF-VFM: best_ema_model_*.pt is a DIFFERENT, EMA/best-loss checkpoint)."""
    ckpt_dir = f"tabdiff/ckpt/{ds}/{ds}_tabdiff_s{seed}"
    ckpt_path = f"{ckpt_dir}/{ckpt_name}"
    assert os.path.exists(ckpt_path), f"no checkpoint {ckpt_path}"
    with open(f"{ckpt_dir}/config.pkl", "rb") as f:
        raw_config = pickle.load(f)
    info = json.load(open(f"data/{ds}/info.json"))

    train_data = TabDiffDataset(ds, f"data/{ds}", info, y_only=False, isTrain=True,
                                 dequant_dist=raw_config["data"]["dequant_dist"],
                                 int_dequant_factor=raw_config["data"]["int_dequant_factor"])
    d_numerical, categories = train_data.d_numerical, train_data.categories

    params = dict(raw_config["unimodmlp_params"])
    params["d_numerical"] = d_numerical
    params["categories"] = (categories + 1).tolist()  # +1 mask category, matches main.py:159
    backbone = UniModMLP(**params)
    model = Model(backbone, **raw_config["diffusion_params"]["edm_params"]).to(device)

    diffusion_params = dict(raw_config["diffusion_params"])
    if num_timesteps is not None:
        diffusion_params["num_timesteps"] = num_timesteps  # ONLY thing we override
    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories, num_numerical_features=d_numerical,
        denoise_fn=model, y_only_model=None, device=device,
        **diffusion_params,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    diffusion._denoise_fn.load_state_dict(state["denoise_fn"])
    diffusion.num_schedule.load_state_dict(state["num_schedule"])
    diffusion.cat_schedule.load_state_dict(state["cat_schedule"])
    diffusion.eval()
    return diffusion, train_data, info, ckpt_path


def decode_to_df(x_gen, train_data, info):
    syn_num, syn_cat, syn_target = split_num_cat_target(
        x_gen, info, train_data.num_inverse, train_data.int_inverse, train_data.cat_inverse)
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)
    idx_name_mapping = {int(k): v for k, v in info["idx_name_mapping"].items()}
    syn_df.rename(columns=idx_name_mapping, inplace=True)
    return syn_df


@torch.no_grad()
def sample_all(diffusion, num_samples, batch_size):
    all_samples = []
    num_generated = 0
    while num_generated < num_samples:
        s = diffusion.sample(batch_size)
        mask_nan = torch.any(s.isnan(), dim=1)
        s = s[~mask_nan]
        all_samples.append(s)
        num_generated += s.shape[0]
    return torch.cat(all_samples, dim=0)[:num_samples]


def run_condition(ds, seed, device, num_timesteps, sample_batch_size):
    diffusion, train_data, info, ckpt_path = load_diffusion(ds, seed, device, num_timesteps=num_timesteps)
    real_path = f"synthetic/{ds}/real.csv"
    num_samples = len(pd.read_csv(real_path))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    t0 = time.time()
    x_gen = sample_all(diffusion, num_samples, sample_batch_size)
    dt = time.time() - t0
    syn_df = decode_to_df(x_gen, train_data, info)
    return syn_df, dt, ckpt_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify_only", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--steps", nargs="*", type=int, default=STEP_COUNTS)
    args = ap.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    manifest = []
    for ds in args.datasets:
        with open(f"tabdiff/ckpt/{ds}/{ds}_tabdiff_s{args.seeds[0]}/config.pkl", "rb") as f:
            sample_batch_size = pickle.load(f)["sample"]["batch_size"]
        for seed in args.seeds:
            # default-step (50) verification against published samples.csv
            syn_df, dt, ckpt_path = run_condition(ds, seed, device, num_timesteps=None,
                                                    sample_batch_size=sample_batch_size)
            out_path = f"{OUT_DIR}/{ds}_s{seed}_default50_samples.csv"
            syn_df.to_csv(out_path, index=False)
            print(f"[VERIFY] {ds} s{seed}: default(num_timesteps=50) time={dt:.1f}s "
                  f"n={len(syn_df)} ckpt={ckpt_path}")
            manifest.append(dict(dataset=ds, seed=seed, steps=50, time_s=dt, path=out_path))

            pub_path = f"tabdiff/result/{ds}/{ds}_tabdiff_s{seed}/8000/samples.csv"
            if os.path.exists(pub_path):
                pub = pd.read_csv(pub_path)
                syn_reloaded = pd.read_csv(out_path)
                same_shape = pub.shape == syn_reloaded.shape
                if same_shape:
                    try:
                        num_cols = pub.select_dtypes(include=[np.number]).columns
                        close = np.allclose(pub[num_cols].values, syn_reloaded[num_cols].values,
                                             equal_nan=True, atol=1e-4)
                    except Exception as e:
                        close = f"compare_failed: {e}"
                else:
                    close = "shape_mismatch"
                print(f"  [RED-LINE CHECK] vs published {pub_path}: shape_match={same_shape} "
                      f"numeric_allclose={close}")

            if args.verify_only:
                continue
            for ns in args.steps:
                if ns == 50:
                    continue  # already covered by the default-path run above
                syn_df_n, dt_n, _ = run_condition(ds, seed, device, num_timesteps=ns,
                                                    sample_batch_size=sample_batch_size)
                out_path_n = f"{OUT_DIR}/{ds}_s{seed}_steps{ns}_samples.csv"
                syn_df_n.to_csv(out_path_n, index=False)
                print(f"[SWEEP] {ds} s{seed} num_timesteps={ns}: time={dt_n:.1f}s")
                manifest.append(dict(dataset=ds, seed=seed, steps=ns, time_s=dt_n, path=out_path_n))

    pd.DataFrame(manifest).to_csv(f"{OUT_DIR}/manifest.csv", index=False)
    print(f"\n[SAVED] {OUT_DIR}/manifest.csv")
