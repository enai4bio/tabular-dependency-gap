"""
Litmus check — the permanent gate for Phase B dependency mechanisms.

PRINCIPLE (042): any "dependency fix" mechanism MUST change the quantities that
sample() actually consumes, otherwise it only reshapes the training loss and can
NOT change generated-sample dependencies. sample() integrates the velocity field
VelocityFix.forward(t, x), which builds:
    v_num  ← mu      (mean prediction)
    v_cat  ← logits  (categorical head output)
A mechanism "enters generation" iff it changes this velocity field. We test that
DIRECTLY (one deterministic forward on a fixed (t, x)), not by running the whole
noisy ODE — so the gate is exact and free of solver/seed confounds.

For mechanism under test:
    * P3 cross_head  -> MUST change v_cat   (logits → enters generation)
    * P1 L_head      -> does NOT change v   (dropped in VelocityFix → dead lever)

Exit 0 = PASS (safe to train).  Exit 1 = FAIL (does not enter generation — DO NOT TRAIN).

Usage:
    python litmus_check.py --mechanism p3     # the config we train for adult (MUST PASS)
    python litmus_check.py --mechanism p1p3   # P3 passes; documents P1's L is dead
    python litmus_check.py --mechanism p1     # the FAIL the gate exists to catch
"""
import argparse
import numpy as np
import torch

from ef_vfm.modules.main_modules_fix import UniModMLPFix
from ef_vfm.models.flow_model_fix import ExpVFMFix, VelocityFix


def build(mechanism, d_numerical=6, categories=(3, 4, 2), rank=2, device="cpu"):
    cats = np.array(categories)
    model = UniModMLPFix(
        d_numerical=d_numerical,
        categories=cats.tolist(),
        num_layers=2, d_token=4, n_head=1, factor=32, bias=True,
        dim_t=64, use_mlp=True, activation="gelu",
        rank=rank, mechanism=mechanism,
    ).to(device)
    flow = ExpVFMFix(
        num_classes=cats,
        num_numerical_features=d_numerical,
        vf_fn=model,
        device=torch.device(device),
        rank=rank,
    )
    flow.eval()
    return flow, model


def velocity_on_fixed_input(model, d_numerical, categories, n=32, seed=0):
    """One deterministic eval of the exact velocity field sample() integrates."""
    d_in = d_numerical + int(sum(categories))
    torch.manual_seed(seed)
    x = torch.randn(n, d_in)
    t = torch.full((n,), 0.5)
    vf = VelocityFix(model)
    with torch.no_grad():
        v = vf(t, x)                       # [n, d_in] = [v_num | v_cat]
    return v, d_numerical


def perturb_(module):
    with torch.no_grad():
        for p in module.parameters():
            p.add_(2.0 * torch.randn_like(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism", default="p3", choices=["p1p3", "p3", "p1"])
    args = ap.parse_args()

    print(f"=== LITMUS CHECK: mechanism='{args.mechanism}' ===")
    enable_p1 = "p1" in args.mechanism
    enable_p3 = "p3" in args.mechanism

    flow, model = build(args.mechanism)
    d_num = model.d_numerical
    cats = model.categories

    # ── structural assertions ─────────────────────────────────────────────
    if enable_p3:
        assert model.cross_head is not None, "P3 requested but cross_head is None"
    else:
        assert model.cross_head is None, "P3 disabled but cross_head exists"
    if enable_p1:
        assert model.L_head is not None, "P1 requested but L_head is None"
    else:
        assert model.L_head is None, "P1 disabled but L_head exists"
    print(f"  structural: cross_head={'on' if model.cross_head is not None else 'off'}, "
          f"L_head={'on' if model.L_head is not None else 'off'}")

    v_base, _ = velocity_on_fixed_input(model, d_num, cats)
    ok = True

    # ── functional litmus on the velocity field (deterministic) ────────────
    if enable_p3:
        _, m3 = build(args.mechanism); m3.load_state_dict(model.state_dict())
        perturb_(m3.cross_head)
        v_after, _ = velocity_on_fixed_input(m3, d_num, cats)
        v_cat_changed = not torch.allclose(v_base[:, d_num:], v_after[:, d_num:])
        print(f"  [P3] perturb cross_head -> v_cat changed: {v_cat_changed}")
        if not v_cat_changed:
            print("  FAIL: P3 cross_head does NOT affect v_cat — it would not enter generation.")
            ok = False
        else:
            print("  PASS: P3 enters generation (cross_head → logits → v_cat).")

    if enable_p1:
        _, m1 = build(args.mechanism); m1.load_state_dict(model.state_dict())
        perturb_(m1.L_head)
        v_after, _ = velocity_on_fixed_input(m1, d_num, cats)
        v_changed = not torch.allclose(v_base, v_after)
        print(f"  [P1] perturb L_head -> velocity changed: {v_changed}")
        if v_changed:
            print("  UNEXPECTED: L_head affected the velocity field — investigate.")
            ok = False
        else:
            print("  CONFIRMED-DEAD: L_head does NOT affect the velocity field "
                  "(VelocityFix ignores L). P1 cannot change sample dependencies; "
                  "its loss only reweights mu.")

    print()
    if args.mechanism == "p3":
        if ok:
            print("RESULT: PASS — P3-only enters generation. Safe to train (clean attribution).")
            return 0
        print("RESULT: FAIL — refuse to train.")
        return 1
    elif args.mechanism == "p1":
        print("RESULT: FAIL — P1-only is the case the gate exists to catch "
              "(loss-only, never enters generation). DO NOT train P1 in this form.")
        return 1
    else:  # p1p3
        print("RESULT: P3 enters generation; P1's L confirmed dead. "
              "For clean attribution train --mechanism p3.")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
