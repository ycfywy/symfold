# -*- coding: utf-8 -*-
"""
Bernoulli Discrete Flow Matching v4 — with Adaptive Density-Aware Loss

改进 vs v3:
1. Adaptive pos_weight: 根据 density head 预测的配对密度动态调整
2. Focal Loss 混合: 对高置信假阳性加大惩罚 (解决低密度过预测)
3. Density regression loss: 辅助任务，预测正确的配对密度
4. 保留: stacking + non-crossing physics loss
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Forward (Noising) — same as v3
# ============================================================

def sample_x_t_given_x_1(x_1: torch.Tensor, t: torch.Tensor,
                          rho_0: float = 0.005) -> torch.Tensor:
    B = x_1.shape[0]
    t_b = t.view(B, 1, 1, 1)
    p_one = t_b * x_1 + (1.0 - t_b) * rho_0
    return (torch.rand_like(x_1) < p_one).float()


def symmetrize_binary(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, x.transpose(-2, -1))


def symmetrize_logit(logit: torch.Tensor) -> torch.Tensor:
    return 0.5 * (logit + logit.transpose(-2, -1))


# ============================================================
# CTMC rates — same as v3
# ============================================================

def compute_ctmc_rates(x_t: torch.Tensor, p_x1: torch.Tensor,
                       t: torch.Tensor, rho_0: float = 0.005,
                       rate_clip: float = 50.0):
    B = x_t.shape[0]
    t_b = t.view(B, 1, 1, 1)
    eps = 1e-6
    p_xt_1 = (1.0 - t_b) * rho_0 + t_b * p_x1
    p_xt_0 = 1.0 - p_xt_1
    rate_01 = torch.clamp(p_x1 - rho_0, min=0.0) / (p_xt_0 + eps)
    rate_10 = torch.clamp(rho_0 - p_x1, min=0.0) / (p_xt_1 + eps)
    rate_01 = torch.clamp(rate_01, max=rate_clip)
    rate_10 = torch.clamp(rate_10, max=rate_clip)
    return rate_01, rate_10


# ============================================================
# Physics Losses — enhanced from v3
# ============================================================

class StackingLoss(nn.Module):
    """Encourage stacking continuity: (i,j) paired → (i+1,j-1) should be paired."""

    def __init__(self, weight: float = 0.05):
        super().__init__()
        self.weight = weight

    def forward(self, logit, contact_masks):
        if self.weight == 0:
            return torch.tensor(0.0, device=logit.device)
        prob = torch.sigmoid(logit)
        prob_shift = F.pad(prob[:, :, 1:, :-1], (1, 0, 0, 1))
        mask = contact_masks[:, :, 1:, :-1]
        mask = F.pad(mask, (1, 0, 0, 1))
        mask = mask * contact_masks
        stack_agreement = prob * prob_shift * mask
        loss = -stack_agreement.sum() / mask.sum().clamp(min=1.0)
        return self.weight * loss


class NonCrossingLoss(nn.Module):
    """Penalize multiple pairs per base (soft constraint for non-crossing)."""

    def __init__(self, weight: float = 0.02):
        super().__init__()
        self.weight = weight

    def forward(self, logit, contact_masks):
        if self.weight == 0:
            return torch.tensor(0.0, device=logit.device)
        prob = torch.sigmoid(logit) * contact_masks
        row_sum = prob.squeeze(1).sum(dim=-1)
        excess = F.relu(row_sum - 1.0)
        loss = excess.mean()
        return self.weight * loss


# ============================================================
# Adaptive Density-Aware Loss (NEW in v4)
# ============================================================

class BernoulliFlowLoss_v4(nn.Module):
    """
    L = BCE_adaptive + λ_stack * L_stack + λ_nc * L_nc + λ_density * L_density

    Key improvement: Adaptive pos_weight based on per-sample density.
    - High density samples (ppb > 0.5): normal pos_weight (~199)
    - Low density samples (ppb < 0.2): reduced pos_weight (~50-100)
    This prevents over-prediction on low-density RNA.

    Also adds focal-style modulation for hard negatives.
    """

    def __init__(self, rho_0: float = 0.005, time_weight: bool = True,
                 pos_weight_base: float = 199.0,
                 pos_weight_min: float = 50.0,
                 focal_gamma: float = 1.0,
                 stack_weight: float = 0.05,
                 nc_weight: float = 0.02,
                 density_weight: float = 0.1):
        super().__init__()
        self.rho_0 = rho_0
        self.time_weight = time_weight
        self.pos_weight_base = pos_weight_base
        self.pos_weight_min = pos_weight_min
        self.focal_gamma = focal_gamma
        self.density_weight = density_weight

        self.stacking_loss = StackingLoss(weight=stack_weight)
        self.nc_loss = NonCrossingLoss(weight=nc_weight)

    def _compute_adaptive_pos_weight(self, x_1, contact_masks):
        """
        Compute per-sample adaptive pos_weight based on GT density.
        Low density → lower pos_weight (less incentive to predict pairs).
        """
        B = x_1.shape[0]
        # Count GT pairs per sample
        with torch.no_grad():
            valid = contact_masks.squeeze(1)  # (B, L, L)
            L_eff = valid[:, 0, :].sum(dim=-1)  # effective length per sample
            gt_pairs = (x_1.squeeze(1) * valid).sum(dim=(-1, -2)) / 2  # symmetric
            pair_per_base = gt_pairs / L_eff.clamp(min=1)  # (B,)

            # Adaptive weight: interpolate between min and base based on density
            # density ~ 0.5+ → use base weight; density ~ 0 → use min weight
            alpha = (pair_per_base / 0.5).clamp(0, 1)  # 0→0, 0.5→1
            pos_weight = self.pos_weight_min + alpha * (self.pos_weight_base - self.pos_weight_min)
        return pos_weight.view(B, 1, 1, 1)  # (B, 1, 1, 1) for broadcasting

    def forward(self, logit, x_1, t, contact_masks,
                density_pred=None, return_gt_density=False):
        """
        Args:
            logit: (B, 1, L, L)
            x_1: (B, 1, L, L) ground truth
            t: (B,)
            contact_masks: (B, 1, L, L)
            density_pred: (B, 1) from density head (optional)
        """
        B, _, L, _ = logit.shape

        # --- Adaptive BCE ---
        pos_weight = self._compute_adaptive_pos_weight(x_1, contact_masks)

        # Manual BCE with per-sample pos_weight
        # BCE = -[pw * y * log(σ) + (1-y) * log(1-σ)]
        # Using log_sigmoid trick for numerical stability
        logsig = F.logsigmoid(logit)
        lognsig = F.logsigmoid(-logit)
        bce = -(pos_weight * x_1 * logsig + (1 - x_1) * lognsig)

        # Focal modulation: down-weight easy predictions
        if self.focal_gamma > 0:
            with torch.no_grad():
                p = torch.sigmoid(logit)
                pt = p * x_1 + (1 - p) * (1 - x_1)
                focal_w = (1 - pt) ** self.focal_gamma
            bce = bce * focal_w

        # Time weighting
        if self.time_weight:
            t_b = t.view(B, 1, 1, 1).clamp(0.0, 1.0 - 1e-3)
            w = 1.0 / (1.0 - t_b * (1.0 - self.rho_0))
            bce = bce * w

        # Mask: valid positions + |i-j| >= 3
        idx = torch.arange(L, device=logit.device)
        short_range_ok = ((idx.view(L, 1) - idx.view(1, L)).abs() >= 3)
        mask = contact_masks * short_range_ok.view(1, 1, L, L).float()

        bce = bce * mask
        bce_loss = bce.sum() / mask.sum().clamp(min=1.0)

        # --- Physics losses ---
        stack_loss = self.stacking_loss(logit, contact_masks)
        nc_loss = self.nc_loss(logit, contact_masks)

        # --- Density regression loss ---
        density_loss = torch.tensor(0.0, device=logit.device)
        gt_density = None
        if density_pred is not None and self.density_weight > 0:
            with torch.no_grad():
                valid = contact_masks.squeeze(1)
                L_eff = valid[:, 0, :].sum(dim=-1)
                gt_pairs = (x_1.squeeze(1) * valid).sum(dim=(-1, -2)) / 2
                gt_density = (gt_pairs / L_eff.clamp(min=1)).unsqueeze(1)  # (B, 1)
            density_loss = self.density_weight * F.mse_loss(density_pred, gt_density)

        total = bce_loss + stack_loss + nc_loss + density_loss
        loss_dict = {
            'bce': bce_loss.detach(),
            'stack': stack_loss.detach(),
            'nc': nc_loss.detach(),
            'density': density_loss.detach(),
        }

        if return_gt_density:
            return total, loss_dict, gt_density
        return total, loss_dict


# ============================================================
# Projection — same as v3
# ============================================================

def project_to_valid_contact_map(x: torch.Tensor, score: torch.Tensor,
                                  contact_masks: torch.Tensor,
                                  max_iters: int = None) -> torch.Tensor:
    """GPU greedy max-matching: symmetric + |i-j|>=3 + max 1 per row."""
    B, _, L, _ = x.shape
    device = x.device
    if max_iters is None:
        max_iters = L // 2

    idx = torch.arange(L, device=device)
    valid = ((idx.view(L, 1) - idx.view(1, L)).abs() >= 3).view(1, 1, L, L).float()
    valid = valid * contact_masks
    s = (x * (score + 1e-6)) * valid
    s = 0.5 * (s + s.transpose(-2, -1))

    out = torch.zeros_like(x)
    s_remain = s.clone()
    for _ in range(max_iters):
        flat = s_remain.view(B, L * L)
        max_val, max_idx = flat.max(dim=-1)
        if (max_val <= 0).all():
            break
        i = max_idx // L
        j = max_idx % L
        active = max_val > 0
        if active.any():
            bb = torch.arange(B, device=device)[active]
            ii = i[active]
            jj = j[active]
            out[bb, 0, ii, jj] = 1.0
            out[bb, 0, jj, ii] = 1.0
            s_remain[bb, 0, ii, :] = 0.0
            s_remain[bb, 0, :, ii] = 0.0
            s_remain[bb, 0, jj, :] = 0.0
            s_remain[bb, 0, :, jj] = 0.0
    return out


# ============================================================
# τ-leap sampling — same as v3 (cosine schedule)
# ============================================================

@torch.no_grad()
def sample_symfold_v4(network_fn, set_max_len: int, contact_masks: torch.Tensor,
                      num_samples: int = 1, num_steps: int = 20,
                      rho_0: float = 0.005,
                      physics_guidance_fn=None, energy_beta: float = 0.0,
                      project_final: bool = True, return_chain: bool = False):
    """τ-leap sampling with cosine schedule."""
    import math
    device = contact_masks.device
    L = set_max_len
    B = num_samples

    x_t = (torch.rand(B, 1, L, L, device=device) < rho_0).float()
    x_t = symmetrize_binary(x_t) * contact_masks

    chain = [x_t.clone()] if return_chain else None

    raw = [math.sin(math.pi * (k + 0.5) / (2 * num_steps)) for k in range(num_steps)]
    total = sum(raw)
    dt_list = [r / total for r in raw]

    p_x1_last = torch.zeros_like(x_t)
    t_cumulative = 0.0

    for k in range(num_steps):
        dt = dt_list[k]
        t_val = t_cumulative
        t_tensor = torch.full((B,), t_val, device=device)
        logit = network_fn(x_t, t_tensor)
        logit = symmetrize_logit(logit)

        if energy_beta > 0.0 and physics_guidance_fn is not None:
            grad = physics_guidance_fn(logit, x_t)
            logit = logit - energy_beta * grad

        p_x1 = torch.sigmoid(logit)
        p_x1 = 0.5 * (p_x1 + p_x1.transpose(-2, -1))
        p_x1_last = p_x1

        rate_01, rate_10 = compute_ctmc_rates(x_t, p_x1, t_tensor, rho_0=rho_0)
        f01 = torch.clamp(rate_01 * dt, max=1.0)
        f10 = torch.clamp(rate_10 * dt, max=1.0)
        flip01 = (torch.rand_like(f01) < f01) & (x_t < 0.5)
        flip10 = (torch.rand_like(f10) < f10) & (x_t > 0.5)
        x_t = torch.where(flip01, torch.ones_like(x_t), x_t)
        x_t = torch.where(flip10, torch.zeros_like(x_t), x_t)
        x_t = symmetrize_binary(x_t) * contact_masks

        t_cumulative += dt
        if return_chain:
            chain.append(x_t.clone())

    if project_final:
        x_final = project_to_valid_contact_map(x_t, p_x1_last, contact_masks)
    else:
        x_final = x_t

    if return_chain:
        return x_final, p_x1_last, chain
    return x_final, p_x1_last


# ============================================================
# Self-test
# ============================================================

if __name__ == '__main__':
    torch.manual_seed(0)
    B, L = 2, 16
    x_1 = (torch.rand(B, 1, L, L) > 0.99).float()
    x_1 = symmetrize_binary(x_1)

    print('1) Adaptive Loss test')
    logit = torch.randn(2, 1, 16, 16)
    cm = torch.ones(2, 1, 16, 16)
    density_pred = torch.rand(2, 1)
    loss_fn = BernoulliFlowLoss_v4()
    total, d = loss_fn(logit, x_1, torch.rand(2), cm, density_pred=density_pred)
    print(f'  total={total.item():.4f}, bce={d["bce"].item():.4f}, '
          f'stack={d["stack"].item():.4f}, nc={d["nc"].item():.4f}, '
          f'density={d["density"].item():.4f}')

    print('2) Projection test')
    x = torch.zeros(1, 1, L, L)
    score = torch.zeros(1, 1, L, L)
    x[0, 0, 2, 5] = 1; x[0, 0, 5, 2] = 1
    score[0, 0, 2, 5] = 0.9; score[0, 0, 5, 2] = 0.9
    out = project_to_valid_contact_map(x, score, cm[:1])
    print(f'  pairs after proj: {int(out.sum().item()) // 2}')

    print('discrete_flow_v4.py self-test passed')
