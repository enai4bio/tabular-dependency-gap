"""Phase B fix: UniModMLPFix adds P1 (low-rank covariance factor) and P3 (num→cat cross-coupling)."""
import torch
import torch.nn as nn
from ef_vfm.modules.main_modules import UniModMLP


class UniModMLPFix(UniModMLP):
    """
    P1: adds an L_head that outputs a low-rank factor L [bs, d_num*rank] for covariance.
    P3: adds a cross_head [d_token → sum_cats] that conditions categorical logits on
        the mean of numeric decoder tokens (explicit num→cat cross-coupling).

    Forward returns (x_num_pred, x_cat_pred, L_flat) instead of (x_num_pred, x_cat_pred).
    L_flat is None when d_numerical == 0.
    """

    def __init__(self, d_numerical, categories, *args, rank=2, mechanism='p1p3', **kwargs):
        super().__init__(d_numerical, categories, *args, **kwargs)
        d_token = kwargs.get("d_token", 4)
        self.rank = rank
        self.mechanism = mechanism
        enable_p1 = 'p1' in mechanism
        enable_p3 = 'p3' in mechanism
        sum_cats = int(sum(categories)) if len(categories) > 0 else 0

        # P1: low-rank factor head (numeric mean token → L factors).
        # NOTE: known-dead lever — L is dropped during sampling (flow_model_fix.py),
        # so it only reweights the mu gradient. Disabled when 'p1' not in mechanism.
        self.L_head = (
            nn.Linear(d_token, d_numerical * rank, bias=True)
            if enable_p1 and d_numerical > 0 else None
        )

        # P3: numeric summary → categorical logit adjustment (enters sampling via logits).
        self.cross_head = (
            nn.Linear(d_token, sum_cats, bias=False)
            if enable_p3 and d_numerical > 0 and sum_cats > 0 else None
        )

    def forward(self, x_num, x_cat, timesteps):
        e = self.tokenizer(x_num, x_cat)
        decoder_input = e[:, 1:, :]           # drop CLS token
        y = self.encoder(decoder_input)
        pred_y = self.mlp(y.reshape(y.shape[0], -1), timesteps)
        pred_e = self.decoder(pred_y.reshape(*y.shape))   # [bs, n_cols, d_token]
        x_num_pred, x_cat_pred_list = self.detokenizer(pred_e)

        # ── P3: numeric summary → categorical logit correction ─────────────────
        if self.cross_head is not None and self.d_numerical > 0:
            h_num = pred_e[:, :self.d_numerical, :]       # [bs, d_num, d_token]
            num_mean = h_num.mean(dim=1)                  # [bs, d_token]
            cross_adj = self.cross_head(num_mean)         # [bs, sum_cats]
            cum = 0
            for i, val in enumerate(self.categories):
                x_cat_pred_list[i] = x_cat_pred_list[i] + cross_adj[:, cum:cum + val]
                cum += val

        x_cat_pred = (
            torch.cat(x_cat_pred_list, dim=-1)
            if len(x_cat_pred_list) > 0
            else torch.zeros(x_num_pred.shape[0], 0, device=x_num_pred.device,
                             dtype=x_num_pred.dtype)
        )

        # ── P1: low-rank factor L ───────────────────────────────────────────────
        L_flat = None
        if self.L_head is not None and self.d_numerical > 0:
            h_num = pred_e[:, :self.d_numerical, :]       # [bs, d_num, d_token]
            num_mean = h_num.mean(dim=1)                  # [bs, d_token]
            L_flat = self.L_head(num_mean)                # [bs, d_num * rank]

        return x_num_pred, x_cat_pred, L_flat
