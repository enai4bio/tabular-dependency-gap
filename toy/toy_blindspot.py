"""
Leg-1 controlled toy (prompt/044 §7.4 + §7.4b): is EF-VFM's dependency loss a
SOFT finite-capacity effect, or a SHARP PROCEDURAL blind spot in its sampler
(finite-step ODE + per-column INDEPENDENT argmax)?

The TMLR recovery proof governs the VELOCITY FIELD (u affine in one-hot => v=v*),
NOT the actual sampler. So "does not recover at high capacity" must be attributed
across THREE confounds before it means anything:
  (a) capacity   -> sweep network width
  (b) ODE steps  -> sweep sampling steps
  (c) decode     -> per-column independent argmax vs joint decode; + residual
                    per-column confidence at t=1

Falsifiable prediction (044 §7.4b): XOR/checkerboard RECOVER under fine-step +
sufficient capacity (confirms lemma); under EF-VFM real sampler (few steps +
independent argmax) recover incompletely, with a gap that moves with steps/decode
but NOT capacity => procedural blind spot (the sharp leg-1, = what P3 fixes).

CRITICAL (blood discipline): dependency MUST be orthogonal to per-dim conditional
mean. NEVER Gaussian/linear copula (captured via the mean => false negative).
  * XOR        : C3 = C1 ^ C2  (pairwise independent, 3-way dependent) [primary]
  * checkerboard: 4x4 (zero linear corr, strong cell dependency, non-singular)
  * gaussian   : --exp sanity ONLY (should recover; proves the pipeline + that
                 linear dep is captured -- never a blind-spot demo).

Self-contained mirror of ef-vfm-fix/ef_vfm/models/flow_model.py (a=0.99).
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

A = 0.99
XOR_CODEWORDS = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])  # c3=c1^c2


# --------------------------- data ---------------------------
def sample_xor(n, rng):
    c1 = rng.integers(0, 2, n); c2 = rng.integers(0, 2, n)
    return np.stack([c1, c2, c1 ^ c2], axis=1).astype(np.int64)


def sample_checkerboard(n, rng, grid=4):
    out = []
    while sum(len(o) for o in out) < n:
        p = rng.uniform(-1, 1, size=(n, 2))
        i = np.clip(((p[:, 0] + 1) / (2.0 / grid)).astype(int), 0, grid - 1)
        j = np.clip(((p[:, 1] + 1) / (2.0 / grid)).astype(int), 0, grid - 1)
        out.append(p[(i + j) % 2 == 0])
    return np.concatenate(out)[:n].astype(np.float32)


def sample_gaussian(n, rng, rho=0.9):
    return rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n).astype(np.float32)


# --------------------------- model --------------------------
class MLP(nn.Module):
    def __init__(self, d_in, d_out, width, depth=3):
        super().__init__()
        layers, d = [], d_in + 1
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]; d = width
        layers += [nn.Linear(d, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t.reshape(-1, 1)], dim=1))


# --------------------------- continuous ---------------------
def train_continuous(data, width, epochs, bs, seed):
    torch.manual_seed(seed)
    d = data.shape[1]; net = MLP(d, d, width)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    X = torch.tensor(data); n = X.shape[0]
    for _ in range(epochs):
        x1 = X[torch.randint(0, n, (bs,))]
        t = torch.rand(bs, 1); eps = torch.randn_like(x1)
        xt = t * x1 + (1 - t) * eps
        loss = ((net(xt, t) - x1) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def sample_continuous(net, n, d, steps):
    x = torch.randn(n, d); ts = torch.linspace(0, 0.999, steps + 1)
    for k in range(steps):
        t = ts[k].expand(n)
        v = (net(x, t) - A * x) / (1 - A * t.reshape(-1, 1))
        x = x + v * (ts[k + 1] - ts[k])
    return x.numpy()


# --------------------------- categorical --------------------
def train_categorical(labels, K, width, epochs, bs, seed):
    torch.manual_seed(seed)
    m = labels.shape[1]; d = m * K; net = MLP(d, d, width)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    Y = torch.tensor(labels); n = Y.shape[0]
    for _ in range(epochs):
        y = Y[torch.randint(0, n, (bs,))]
        oh = torch.cat([F.one_hot(y[:, c], K).float() for c in range(m)], dim=1)
        t = torch.rand(bs, 1)
        xt = t * oh + (1 - t) * torch.randn_like(oh)
        logits = net(xt, t)
        loss = sum(F.cross_entropy(logits[:, c * K:(c + 1) * K], y[:, c]) for c in range(m))
        opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def sample_categorical(net, n, m, K, steps):
    """Returns (discrete_indep_argmax, final_continuous_state)."""
    d = m * K; x = torch.randn(n, d); ts = torch.linspace(0, 0.999, steps + 1)
    for k in range(steps):
        t = ts[k].expand(n)
        logits = net(x, t)
        parts = [(F.softmax(logits[:, c*K:(c+1)*K], -1) - A * x[:, c*K:(c+1)*K])
                 / (1 - A * t.reshape(-1, 1)) for c in range(m)]
        x = x + torch.cat(parts, 1) * (ts[k + 1] - ts[k])
    disc = torch.stack([x[:, c*K:(c+1)*K].argmax(1) for c in range(m)], 1).numpy()
    return disc, x.numpy()


# --------------------------- metrics ------------------------
def xor_diag(disc, cont):
    """Disentangle decode (c): independent argmax vs joint decode + confidence."""
    c1, c2, c3 = disc[:, 0], disc[:, 1], disc[:, 2]
    indep_dep = float(np.mean((c1 ^ c2) == c3))                  # EF-VFM real; true=1
    # per-column residual confidence at t=1 (1.0 = fully committed one-hot)
    probs = [np.exp(cont[:, c*2:(c+1)*2]) for c in range(3)]
    probs = [p / p.sum(1, keepdims=True) for p in probs]
    conf = float(np.mean([p.max(1).mean() for p in probs]))
    # joint decode: assign to nearest valid codeword by summed evidence
    ev = np.stack([cont[:, 0+cw[0]] + cont[:, 2+cw[1]] + cont[:, 4+cw[2]]
                   for cw in XOR_CODEWORDS], 1)                   # (n,4)
    j = ev.argmax(1)
    jdist = np.bincount(j, minlength=4) / len(j)                  # ~[.25]*4 if joint present
    joint_balance = float(1 - 0.5 * np.abs(jdist - 0.25).sum())  # 1=uniform, lower=skewed
    return indep_dep, conf, joint_balance, np.round(jdist, 3)


def cb_dep(samp, grid=4):
    p = np.clip(samp, -0.999, 0.999)
    i = np.clip(((p[:, 0] + 1) / (2.0 / grid)).astype(int), 0, grid - 1)
    j = np.clip(((p[:, 1] + 1) / (2.0 / grid)).astype(int), 0, grid - 1)
    return float(np.mean((i + j) % 2 == 0))                      # true=1, indep~0.5


def cb_oob(samp):
    return float(np.mean(np.any(np.abs(samp) > 1.0, axis=1)))    # leaked outside [-1,1]^2


def cb_capacity(widths, epochs, bs, n_data, n_samp, seed):
    """THE correct axis (peer catch): soft lemma is parameterized by CAPACITY, not
    epochs/steps. Hold training long & fixed; scale width. Two clean outcomes:
      in_cell -> 1.0 as width grows  => clean positive control, lemma confirmed.
      in_cell stuck ~0.70 at 4x width => stronger honest finding: at realistic
        capacity VFM practically cannot recover this affine-undetectable continuous
        high-order dependency. (Do NOT pre-write either; report what's true.)
    Boundary: ONLY this axis. No CFM-parameterization comparison (that's gilding)."""
    rng = np.random.default_rng(seed)
    data = sample_checkerboard(n_data, rng)
    print(f"\n=== CHECKERBOARD CAPACITY AXIS (depth=3, epochs={epochs}, steps=2000) ===")
    print("    true in_cell=1.0, independent baseline~0.5")
    print(f"  {'width':>6} | {'in_cell':>7} | {'oob_frac':>8}")
    for w in widths:
        net = train_continuous(data, w, epochs, bs, seed)
        samp = sample_continuous(net, n_samp, 2, 2000)
        print(f"  {w:>6} | {cb_dep(samp):>7.3f} | {cb_oob(samp):>8.3f}", flush=True)


def cb_recover(epochs, bs, n_data, n_samp, seed):
    """ONE focused toy: can a continuous dependency RECOVER given enough training?
    Stop rule (044 §7.4b discipline): recover => premise confirmed, lock & stop.
    Stuck at ~0.66 after genuine training => a REAL signal, do NOT hand-wave
    'undertrained' -- report it; it may threaten the soft lemma."""
    rng = np.random.default_rng(seed)
    data = sample_checkerboard(n_data, rng)
    print(f"\n=== CHECKERBOARD RECOVERY (width=512, epochs={epochs}) ===")
    print("    true in_cell=1.0, independent baseline~0.5")
    net = train_continuous(data, 512, epochs, bs, seed)
    print(f"  {'steps':>6} | {'in_cell':>7} | {'oob_frac':>8}")
    for s in [500, 2000]:
        samp = sample_continuous(net, n_samp, 2, s)
        print(f"  {s:>6} | {cb_dep(samp):>7.3f} | {cb_oob(samp):>8.3f}")


# --------------------------- decisive grid ------------------
def decisive(widths, step_list, epochs, bs, n_data, n_samp, seed, hi_w, hi_steps):
    rng = np.random.default_rng(seed)
    print(f"\n########## DECISIVE GRID (epochs={epochs}, hi_w={hi_w}, hi_steps={hi_steps}) ##########")
    print("Read converged HIGH-capacity rows only; early/low-cap lag is trivial.\n")

    # ---- XOR ----
    lab = sample_xor(n_data, rng)
    print("=== XOR: axis (a) CAPACITY  [steps=%d, independent argmax] ===" % hi_steps)
    print(f"  {'width':>6} | {'indep_dep':>9} | {'conf':>5} | {'joint_bal':>9} | jdist")
    for w in widths:
        net = train_categorical(lab, 2, w, epochs, bs, seed)
        d, c = sample_categorical(net, n_samp, 3, 2, hi_steps)
        dep, conf, jb, jd = xor_diag(d, c)
        print(f"  {w:>6} | {dep:>9.3f} | {conf:>5.2f} | {jb:>9.3f} | {jd}")

    print("\n=== XOR: axis (b) ODE STEPS  [width=%d, independent argmax] ===" % hi_w)
    net = train_categorical(lab, 2, hi_w, epochs, bs, seed)
    print(f"  {'steps':>6} | {'indep_dep':>9} | {'conf':>5} | {'joint_bal':>9}")
    for s in step_list:
        d, c = sample_categorical(net, n_samp, 3, 2, s)
        dep, conf, jb, _ = xor_diag(d, c)
        print(f"  {s:>6} | {dep:>9.3f} | {conf:>5.2f} | {jb:>9.3f}")
    print("  ^ (c) DECODE probe: if indep_dep<<1 but joint_bal~1 & conf low =>")
    print("    joint info present, INDEPENDENT decode destroys it = PROCEDURAL blind spot.")

    # ---- checkerboard (no decode confound: continuous) ----
    data = sample_checkerboard(n_data, rng)
    print("\n=== CHECKERBOARD: axis (a) CAPACITY  [steps=%d] === (true in-cell=1.0, indep~0.5)" % hi_steps)
    print(f"  {'width':>6} | {'in_cell':>7}")
    for w in widths:
        net = train_continuous(data, w, epochs, bs, seed)
        print(f"  {w:>6} | {cb_dep(sample_continuous(net, n_samp, 2, hi_steps)):>7.3f}")
    print("\n=== CHECKERBOARD: axis (b) ODE STEPS  [width=%d] ===" % hi_w)
    net = train_continuous(data, hi_w, epochs, bs, seed)
    print(f"  {'steps':>6} | {'in_cell':>7}")
    for s in step_list:
        print(f"  {s:>6} | {cb_dep(sample_continuous(net, n_samp, 2, s)):>7.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="decisive",
                    choices=["decisive", "xor", "checkerboard", "sanity",
                             "cb_recover", "cb_capacity"])
    ap.add_argument("--widths", type=int, nargs="+", default=[8, 32, 128, 512])
    ap.add_argument("--steps", type=int, nargs="+", default=[5, 20, 100, 500, 2000])
    ap.add_argument("--hi_w", type=int, default=512)
    ap.add_argument("--hi_steps", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--n_data", type=int, default=20000)
    ap.add_argument("--n_samp", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.widths, a.steps, a.epochs = [8, 128], [20, 200], 400
        a.hi_w, a.n_data, a.n_samp = 128, 4000, 4000
    if a.exp == "decisive":
        decisive(a.widths, a.steps, a.epochs, a.bs, a.n_data, a.n_samp,
                 a.seed, a.hi_w, a.hi_steps)
    elif a.exp == "cb_recover":
        cb_recover(a.epochs, a.bs, a.n_data, a.n_samp, a.seed)
    elif a.exp == "cb_capacity":
        cb_capacity(a.widths, a.epochs, a.bs, a.n_data, a.n_samp, a.seed)
    elif a.exp == "sanity":
        rng = np.random.default_rng(a.seed); data = sample_gaussian(a.n_data, rng, 0.9)
        print("[SANITY] Gaussian rho=0.9 -- should recover at all widths (NOT a blind-spot demo)")
        for w in a.widths:
            net = train_continuous(data, w, a.epochs, a.bs, a.seed)
            s = sample_continuous(net, a.n_samp, 2, a.hi_steps)
            print(f"  width={w:>4} rho_recovered={np.corrcoef(s.T)[0,1]:.3f}")
