"""056 Task B: standalone NFE (sampling step-count) sweep for EF-VFM/TabbyFlow.

Purely ADDITIVE / read-only w.r.t. existing code: does not import-and-modify
flow_model.py, main.py, or trainer.py; it re-implements the same sample()
logic locally (copied, not edited) so a custom fixed step-size can be passed
to odeint without changing any default call path or default parameter value.

Correction vs. the 056 spec (§B.0.1): the spec assumed the published adult/
default 1x dep numbers come from flow_model_fix.py's ExpVFMFix.sample(). They
do not -- queue_adult_base.sh / queue_047.sh train "*_base_s*" checkpoints
WITHOUT --fix, so they use plain UniModMLP + ExpVFM from flow_model.py (the
"Fix" class is only used for the separate P3 cross-coupling intervention,
"*_fix_*" checkpoints). Both classes' sample() use the identical
dopri5(rtol=1e-5,atol=1e-5)-default / euler-fallback logic, so the guidance
in the spec carries over unchanged -- this script targets flow_model.py's
ExpVFM, matching how *_base_s* was actually trained.

Usage:
    cd ef-vfm-fix
    python run_056_nfe_resample.py --verify_only     # red-line check only
    python run_056_nfe_resample.py                   # full sweep (adult+default, 5 seeds)
"""
import os, sys, pickle, glob, json, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint  # flow_model.py:4 uses this, not plain odeint

# Match the global determinism flags main.py sets once at process start under
# --deterministic (ef_vfm/main.py:97-115), which the published *_base_s* runs used.
# Must be set before any CUDA context / matmul happens, hence at import time here.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)
# WORKAROUND (not a repo/env change): ef_vfm/main.py and utils_train.py do `import src`,
# which historically resolved via a site-packages .pth pointing at the now-dead
# /media/jie/expand_5t/7exp_expand/next/ef-vfm path (pre-July-2026 migration). That .pth
# was never updated, so `import src` is currently broken for ANY ef-vfm-fix script,
# not just this one. We do not touch the shared conda env; we just add the correct,
# now-real path locally for this process.
sys.path.insert(0, "/media/jie/expand_5t/7exp/next/ef-vfm")

from ef_vfm.modules.main_modules import UniModMLP
from utils_train import EFVFMDataset
from ef_vfm.trainer import split_num_cat_target, recover_data

DATASETS = ["adult", "default"]
SEEDS = [0, 1, 2, 3, 4]
EULER_STEPS = [10, 20, 50, 200, 500]  # default (measured dopri5 NFE) reported separately
OUT_DIR = "result_056_nfe"
os.makedirs(OUT_DIR, exist_ok=True)


# ── local, unedited copies of Velocity / sample() math (flow_model.py) ─────
# (kept separate from the real class so this script cannot accidentally
#  change what flow_model.py's ExpVFM.sample() does for any other caller)

class Velocity(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        t = t * torch.ones(x.shape[0]).to(x.device)
        x_num = x[:, :self.model.d_numerical]
        x_cat = x[:, self.model.d_numerical:]
        mu, logits = self.model(x_num, x_cat, t)
        if self.model.d_numerical > 0:
            v_num = (mu - (1 - 0.01) * x_num) / (1 - (1 - 0.01) * t.unsqueeze(1))
        else:
            v_num = torch.zeros_like(x_num)
        if len(self.model.categories) > 0:
            v_cat_parts = []
            logit_idx = 0
            oh_idx = 0
            for k in self.model.categories:
                probs_k = F.softmax(logits[:, logit_idx:logit_idx + k], dim=-1)
                x_k = x_cat[:, oh_idx:oh_idx + k]
                v_k = (probs_k - (1 - 0.01) * x_k) / (1 - (1 - 0.01) * t.unsqueeze(1))
                v_cat_parts.append(v_k)
                logit_idx += k
                oh_idx += k
            v_cat = torch.cat(v_cat_parts, dim=1)
        else:
            v_cat = torch.zeros_like(x_cat)
        return torch.cat([v_num, v_cat], dim=1)


def load_flow(ds, seed, device, ckpt_name="model_8000.pt"):
    """ckpt_name="model_8000.pt" (the LIVE, non-EMA weights at the final training step)
    reproduces the training-time evaluate_generation(ema=False) call that wrote
    ef_vfm/result/{ds}/{ds}_base_s{seed}/8000/samples.csv (trainer.py:245-262,384-388).
    NOTE: "best_ema_model_*.pt" is a DIFFERENT (EMA, best-checkpoint) weight set only
    used by --mode test / report_test, not by the published 8000/samples.csv -- verified
    162/196 tensors differ between the two files for adult_base_s0."""
    ckpt_dir = f"ef_vfm/ckpt/{ds}/{ds}_base_s{seed}"
    ckpt_path = f"{ckpt_dir}/{ckpt_name}"
    assert os.path.exists(ckpt_path), f"no checkpoint {ckpt_path}"
    with open(f"{ckpt_dir}/config.pkl", "rb") as f:
        raw_config = pickle.load(f)
    info = json.load(open(f"data/{ds}/info.json"))
    train_data = EFVFMDataset(ds, f"data/{ds}", info, isTrain=True,
                               dequant_dist=raw_config["data"]["dequant_dist"],
                               int_dequant_factor=raw_config["data"]["int_dequant_factor"])
    d_numerical, categories = train_data.d_numerical, train_data.categories
    params = dict(raw_config["unimodmlp_params"])
    params["d_numerical"] = d_numerical
    params["categories"] = categories.tolist()
    model = UniModMLP(**params).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["vf_fn"])
    model.eval()
    num_classes = categories
    num_numerical_features = d_numerical
    return model, num_numerical_features, num_classes, ckpt_path, train_data, info


@torch.no_grad()
def sample_batch(model, num_numerical_features, num_classes, n, device,
                  method, step_size=None, rtol=1e-5, atol=1e-5, count_nfe=False):
    d_in = num_numerical_features + sum(num_classes)
    d_out = num_numerical_features + len(num_classes)
    x0 = torch.randn(n, d_in, device=device)
    t = torch.tensor([0.0, 0.999]).to(device)
    vf = Velocity(model)
    if method == "dopri5":
        try:
            traj = odeint(vf, x0, t, method="dopri5", rtol=rtol, atol=atol)
        except AssertionError:
            traj = odeint(vf, x0, t, method="euler", options={"step_size": 0.01})
    elif method == "euler":
        traj = odeint(vf, x0, t, method="euler", options={"step_size": step_size})
    else:
        raise ValueError(method)
    out = traj[-1]
    sample = torch.zeros(n, d_out, device=device, dtype=torch.float32)
    sample[:, :num_numerical_features] = out[:, :num_numerical_features].float()
    if sum(num_classes) != 0:
        idx = num_numerical_features
        for i, val in enumerate(num_classes):
            col = num_numerical_features + i
            sample[:, col] = torch.argmax(out[:, idx:idx + val], dim=1)
            idx += val
    return sample.cpu(), vf.nfe


def sample_all(model, num_numerical_features, num_classes, num_samples, batch_size, device,
               method, step_size=None):
    """Mirrors ExpVFM.sample_all's batching loop (flow_model.py:117), unmodified logic."""
    all_samples = []
    total_nfe = 0
    num_generated = 0
    while num_generated < num_samples:
        s, nfe = sample_batch(model, num_numerical_features, num_classes, batch_size, device,
                               method, step_size=step_size)
        mask_nan = torch.any(s.isnan(), dim=1)
        s = s[~mask_nan]
        all_samples.append(s)
        num_generated += s.shape[0]
        total_nfe += nfe
    x_gen = torch.cat(all_samples, dim=0)[:num_samples]
    return x_gen, total_nfe


def decode_to_df(x_gen, train_data, info):
    """Exact same decode path as trainer.py:454-465 (split_num_cat_target + recover_data
    + idx_name_mapping rename), imported not reimplemented, so a byte-for-byte comparison
    against published samples.csv is meaningful."""
    syn_num, syn_cat, syn_target = split_num_cat_target(
        x_gen, info, train_data.num_inverse, train_data.int_inverse, train_data.cat_inverse)
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)
    idx_name_mapping = {int(k): v for k, v in info["idx_name_mapping"].items()}
    syn_df.rename(columns=idx_name_mapping, inplace=True)
    return syn_df


def verify_default_path(ds, seed, device):
    """Red-line check: unmodified dopri5 default path, matched batching/seeding to
    trainer.py's sample_synthetic (torch.manual_seed(seed) immediately before sampling,
    batch_size from ef_vfm_configs.toml == 10000), must reproduce the exact published
    ef_vfm/result/{ds}/{ds}_base_s{seed}/8000/samples.csv after decoding."""
    model, dn, nc, ckpt_path, train_data, info = load_flow(ds, seed, device)
    real_path = f"synthetic/{ds}/real.csv"
    num_samples = len(pd.read_csv(real_path))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    t0 = time.time()
    x_gen, nfe = sample_all(model, dn, nc, num_samples, batch_size=10000, device=device,
                             method="dopri5")
    dt = time.time() - t0
    syn_df = decode_to_df(x_gen, train_data, info)
    return syn_df, nfe, dt, ckpt_path


def run_euler_condition(ds, seed, num_steps, device):
    model, dn, nc, ckpt_path, train_data, info = load_flow(ds, seed, device)
    real_path = f"synthetic/{ds}/real.csv"
    num_samples = len(pd.read_csv(real_path))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    t0 = time.time()
    x_gen, nfe = sample_all(model, dn, nc, num_samples, batch_size=10000, device=device,
                             method="euler", step_size=1.0 / num_steps)
    dt = time.time() - t0
    syn_df = decode_to_df(x_gen, train_data, info)
    return syn_df, nfe, dt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify_only", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--euler_steps", nargs="*", type=int, default=EULER_STEPS)
    args = ap.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    manifest = []
    for ds in args.datasets:
        for seed in args.seeds:
            syn_df, nfe, dt, ckpt_path = verify_default_path(ds, seed, device)
            print(f"[VERIFY] {ds} s{seed}: dopri5 default NFE(function calls)={nfe} "
                  f"time={dt:.1f}s n_samples={len(syn_df)} ckpt={ckpt_path}")
            out_path = f"{OUT_DIR}/{ds}_s{seed}_dopri5_default_samples.csv"
            syn_df.to_csv(out_path, index=False)
            manifest.append(dict(dataset=ds, seed=seed, method="dopri5_default", steps=np.nan,
                                  nfe=nfe, time_s=dt, path=out_path))

            pub_path = f"ef_vfm/result/{ds}/{ds}_base_s{seed}/8000/samples.csv"
            if os.path.exists(pub_path):
                pub = pd.read_csv(pub_path)
                syn_reloaded = pd.read_csv(out_path)  # re-read so dtypes match pub's CSV-inferred dtypes exactly
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
            for ns in args.euler_steps:
                syn_df_e, nfe_e, dt_e = run_euler_condition(ds, seed, ns, device)
                out_path_e = f"{OUT_DIR}/{ds}_s{seed}_euler{ns}_samples.csv"
                syn_df_e.to_csv(out_path_e, index=False)
                print(f"[SWEEP] {ds} s{seed} euler_steps={ns}: nfe={nfe_e} time={dt_e:.1f}s")
                manifest.append(dict(dataset=ds, seed=seed, method="euler", steps=ns,
                                      nfe=nfe_e, time_s=dt_e, path=out_path_e))

    pd.DataFrame(manifest).to_csv(f"{OUT_DIR}/manifest.csv", index=False)
    print(f"\n[SAVED] {OUT_DIR}/manifest.csv")
