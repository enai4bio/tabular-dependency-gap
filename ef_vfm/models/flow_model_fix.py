"""Phase B fix: ExpVFMFix + VelocityFix for P1 (low-rank cov) + P3 (num→cat cross)."""
import torch
import torch.nn.functional as F
import numpy as np
from torchdiffeq import odeint_adjoint as odeint

from ef_vfm.models.flow_model import ExpVFM, Velocity


class VelocityFix(Velocity):
    """Same as Velocity but handles 3-tuple output (num, cat, L_flat) from UniModMLPFix."""

    def forward(self, t, x):
        t = t * torch.ones(x.shape[0]).to(x.device)
        x_num = x[:, :self.model.d_numerical]
        x_cat = x[:, self.model.d_numerical:]
        mu, logits, _L_flat = self.model(x_num, x_cat, t)   # ignore L during ODE

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
            v_cat = torch.cat(v_cat_parts, dim=-1)
        else:
            v_cat = torch.zeros_like(x_cat)

        return torch.cat([v_num, v_cat], dim=1)


class ExpVFMFix(ExpVFM):
    """
    P1: _mvgloss uses low-rank covariance sigma = scale*(I + L L^T).
    P3: baked into UniModMLPFix.forward() (num→cat cross head).
    Sampling uses VelocityFix to handle 3-tuple network output.
    """

    def __init__(self, rank=2, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank

    def mixed_loss(self, x):
        b = x.shape[0]
        dev = x.device

        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:].long()

        t = torch.rand(b, device=dev, dtype=x_num.dtype)
        t = t[:, None]

        x_num_t = x_num
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = t * x_num + (1 - t) * noise

        x_cat_oh = self.to_one_hot(x_cat).float()
        x_cat_t = x_cat_oh
        if x_cat.shape[1] > 0:
            x_cat_t = t * x_cat_oh + (1 - t) * torch.randn_like(x_cat_oh)

        model_out_num, model_out_cat, model_out_L = self._vf_fn(x_num_t, x_cat_t, t.squeeze())

        d_loss = torch.zeros((1,)).float().to(dev)
        c_loss = torch.zeros((1,)).float().to(dev)

        if x_num.shape[1] > 0:
            c_loss = self._mvgloss_fix(model_out_num, model_out_L, x_num, t)

        if x_cat.shape[1] > 0:
            d_loss = self._absorbed_closs(model_out_cat, x_cat, self._vf_fn.categories)

        return d_loss.mean(), c_loss.mean()

    def _mvgloss_fix(self, mu_t, L_flat, x_num_t, t):
        """MVG NLL with low-rank cov: Sigma = scale*(I + L L^T).
        Woodbury identity: O(n*k*r) instead of O(n*k^3), no full k×k matrices.
          log|Σ| = k*log(scale) + log|I_r + L^T L|   (matrix det lemma)
          quad    = (1/scale) * [||r||^2 - r^T L (I_r+L^T L)^{-1} L^T r]
        """
        import math
        n, k = mu_t.shape
        dev = mu_t.device
        dt = mu_t.dtype

        t_val = t.view(n)
        scale = 1 - (1 - 0.01) * t_val ** 2   # [n]
        log_scale = torch.log(scale.clamp(min=1e-8))

        diff = x_num_t - mu_t                   # [n, k]
        diff_sq = (diff * diff).sum(-1)         # [n]

        if L_flat is not None:
            r = self.rank
            L = L_flat.reshape(n, k, r).to(dt)                       # [n, k, r]
            LtL = L.transpose(-1, -2) @ L                            # [n, r, r]
            M = torch.eye(r, device=dev, dtype=dt).unsqueeze(0) + LtL  # [n, r, r]
            log_det = k * log_scale + torch.logdet(M)                # [n]
            Lt_diff = (L.transpose(-1, -2) @ diff.unsqueeze(-1))     # [n, r, 1]
            corr = (L @ torch.linalg.solve(M, Lt_diff)).squeeze(-1)  # [n, k]
            quad = (diff_sq - (diff * corr).sum(-1)) / scale         # [n]
        else:
            log_det = k * log_scale
            quad = diff_sq / scale                                    # [n]

        log_prob = -0.5 * (k * math.log(2 * math.pi) + log_det + quad)
        return -log_prob.mean()

    @torch.no_grad()
    def sample(self, num_samples):
        dev = self.device
        dt = torch.float32
        d_in = self.num_numerical_features + sum(self.num_classes)
        d_out = self.num_numerical_features + len(self.num_classes)

        x0 = torch.randn(num_samples, d_in, device=dev)
        t = torch.tensor([0.0, 0.999]).to(dev)
        vf = VelocityFix(self._vf_fn)     # use VelocityFix for 3-tuple unpacking
        try:
            trajectory = odeint(vf, x0, t, method="dopri5", rtol=1e-5, atol=1e-5)
        except AssertionError:
            trajectory = odeint(vf, x0, t, method="euler", options={"step_size": 0.01})
        out = trajectory[-1]

        sample = torch.zeros(num_samples, d_out, device=dev, dtype=dt)
        sample[:, :self.num_numerical_features] = out[:, :self.num_numerical_features].to(torch.float32)
        if sum(self.num_classes) != 0:
            idx = self.num_numerical_features
            for i, val in enumerate(self.num_classes):
                col = self.num_numerical_features + i
                sample[:, col] = torch.argmax(out[:, idx:idx + val], dim=1)
                idx += val
                assert val >= sample[:, col].max() >= 0

        return sample.cpu()
