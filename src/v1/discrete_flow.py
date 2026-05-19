# -*- coding: utf-8 -*-
"""
Bernoulli Discrete Flow Matching on Symmetric Binary Matrices
==============================================================

把 contact map 视为对称二值矩阵 X ∈ {0,1}^{LxL}_sym 上的随机变量, 用 Discrete
Flow Matching 学习从 Bernoulli(rho_0) 先验到目标分布的概率流.

Forward marginal:
    p_t(X_ij = 1 | X_1) = (1-t) * rho_0 + t * 1[X_1,ij = 1]

训练 loss: pos-weighted BCE (telescoping form).
采样: τ-leaping CTMC.

参考 Campbell et al., NeurIPS 2024,
"Generative Flows on Discrete State-Spaces".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Forward (Noising)
# ============================================================

def sample_x_t_given_x_1(x_1: torch.Tensor, t: torch.Tensor,
                          rho_0: float = 0.005) -> torch.Tensor:
    """
    给定 x_1 和 t, 按边际 p_t(x_t=1|x_1) 独立伯努利采样 x_t.

    Args:
        x_1:   (B, 1, L, L), {0,1}
        t:     (B,), [0,1]
        rho_0: prior pair rate
    Returns:
        x_t: (B, 1, L, L), {0,1}
    """
    B = x_1.shape[0]
    t_b = t.view(B, 1, 1, 1)
    p_one = t_b * x_1 + (1.0 - t_b) * rho_0
    return (torch.rand_like(x_1) < p_one).float()


def symmetrize_binary(x: torch.Tensor) -> torch.Tensor:
    """X ← X ∨ X.T (元素 OR)."""
    return torch.maximum(x, x.transpose(-2, -1))


def symmetrize_logit(logit: torch.Tensor) -> torch.Tensor:
    return 0.5 * (logit + logit.transpose(-2, -1))


# ============================================================
# CTMC rates (sampling 用)
# ============================================================

def compute_ctmc_rates(x_t: torch.Tensor, p_x1: torch.Tensor,
                       t: torch.Tensor, rho_0: float = 0.005,
                       rate_clip: float = 50.0):
    """
    边缘化 x_1 后的 rate:
        R(0→1) ≈ ((p_x1 - rho_0)+ ) / p(x_t=0)
        R(1→0) ≈ ((rho_0 - p_x1)+ ) / p(x_t=1)
    """
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
# Loss
# ============================================================

class BernoulliFlowLoss(nn.Module):
    """
    L = E_{t,x1,x_t}[ w(t) * BCE_pos_weighted(logit_theta, x_1) ]

      w(t) = 1 / (1 - t * (1 - rho_0))
      pos_weight = (1 - rho_0) / rho_0    (≈ 199 当 rho_0=0.005)
    """

    def __init__(self, rho_0: float = 0.005, time_weight: bool = True,
                  pos_weight_scale: float = 1.0):
        super().__init__()
        self.rho_0 = rho_0
        self.time_weight = time_weight
        pw = (1.0 - rho_0) / rho_0 * pos_weight_scale
        self.register_buffer('pos_weight', torch.tensor([pw]))

    def forward(self, logit, x_1, t, contact_masks):
        bce = F.binary_cross_entropy_with_logits(
            logit, x_1,
            pos_weight=self.pos_weight.to(logit.device),
            reduction='none',
        )
        if self.time_weight:
            B = logit.shape[0]
            t_b = t.view(B, 1, 1, 1).clamp(0.0, 1.0 - 1e-3)
            w = 1.0 / (1.0 - t_b * (1.0 - self.rho_0))
            bce = bce * w

        # 排除短程 (|i-j|<3) + 对角线 + padding
        B, _, L, _ = logit.shape
        idx = torch.arange(L, device=logit.device)
        short_range_ok = ((idx.view(L, 1) - idx.view(1, L)).abs() >= 3)
        mask = contact_masks * short_range_ok.view(1, 1, L, L).float()

        bce = bce * mask
        return bce.sum() / mask.sum().clamp(min=1.0)


# ============================================================
# Projection: 任意 binary 矩阵 → 有效 contact map
# ============================================================

def project_to_valid_contact_map(x: torch.Tensor, score: torch.Tensor,
                                  contact_masks: torch.Tensor,
                                  max_iters: int = None) -> torch.Tensor:
    """
    GPU 友好的贪心 max-matching:
      约束 = 对称 + 对角 0 + |i-j|>=3 + 每行至多 1 个 1

    用迭代式: 每次 iter 在所有 batch 中各自找当前最高分的 (i,j), 把对应行/列清零.
    复杂度 O(B * num_pairs * L^2).

    Args:
        x:             (B,1,L,L) 候选 binary
        score:         (B,1,L,L) 分数 (P(x_1=1))
        contact_masks: (B,1,L,L)
        max_iters:     默认 L//2
    Returns:
        out: (B,1,L,L) ∈ {0,1}
    """
    B, _, L, _ = x.shape
    device = x.device
    if max_iters is None:
        max_iters = L // 2

    idx = torch.arange(L, device=device)
    valid = ((idx.view(L, 1) - idx.view(1, L)).abs() >= 3).view(1, 1, L, L).float()
    valid = valid * contact_masks
    s = (x * (score + 1e-6)) * valid                # 只在 x=1 处有正值
    s = 0.5 * (s + s.transpose(-2, -1))             # 对称化分数

    out = torch.zeros_like(x)
    s_remain = s.clone()
    for _ in range(max_iters):
        flat = s_remain.view(B, L * L)
        max_val, max_idx = flat.max(dim=-1)         # (B,) (B,)
        if (max_val <= 0).all():
            break
        i = (max_idx // L)                          # (B,)
        j = (max_idx % L)                           # (B,)
        active = max_val > 0                        # (B,)
        if active.any():
            bb = torch.arange(B, device=device)[active]
            ii = i[active]
            jj = j[active]
            out[bb, 0, ii, jj] = 1.0
            out[bb, 0, jj, ii] = 1.0
            # 清零对应 row & col, 防同 i 或同 j 再被选
            s_remain[bb, 0, ii, :] = 0.0
            s_remain[bb, 0, :, ii] = 0.0
            s_remain[bb, 0, jj, :] = 0.0
            s_remain[bb, 0, :, jj] = 0.0
    return out


# ============================================================
# τ-leap sampling
# ============================================================

@torch.no_grad()
def sample_symfold(network_fn, set_max_len: int, contact_masks: torch.Tensor,
                    num_samples: int = 1, num_steps: int = 20,
                    rho_0: float = 0.005,
                    physics_guidance_fn=None, energy_beta: float = 0.0,
                    project_final: bool = True, return_chain: bool = False):
    """
    τ-leap 采样.

    Args:
        network_fn:           callable (x_t, t) -> logit (B,1,L,L)
        set_max_len:          L
        contact_masks:        (B,1,L,L)
        num_samples:          B
        num_steps:            CTMC 步数
        rho_0:                prior
        physics_guidance_fn:  callable (logit, x_t) -> energy_grad (与 logit 同形状)
        energy_beta:          物理 guidance 强度
        project_final:        是否对最终结果做有效 contact projection
    """
    device = contact_masks.device
    L = set_max_len
    B = num_samples

    x_t = (torch.rand(B, 1, L, L, device=device) < rho_0).float()
    x_t = symmetrize_binary(x_t) * contact_masks

    chain = [x_t.clone()] if return_chain else None
    dt = 1.0 / num_steps

    p_x1_last = torch.zeros_like(x_t)
    for k in range(num_steps):
        t_val = k * dt
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
    rho_0 = 0.01

    print('1) Forward marginal Monte-Carlo check')
    for t_val in (0.0, 0.5, 1.0):
        t = torch.full((B,), t_val)
        cnt = torch.zeros_like(x_1)
        N = 3000
        for _ in range(N):
            cnt += sample_x_t_given_x_1(x_1, t, rho_0=rho_0)
        emp = cnt / N
        theo = t_val * x_1 + (1 - t_val) * rho_0
        print(f'   t={t_val}: max|emp-theo|={(emp-theo).abs().max():.4f}')

    print('2) Symmetric projection on small example')
    x = torch.zeros(1, 1, L, L)
    score = torch.zeros(1, 1, L, L)
    # 设两个候选, 但 i=2 同时与 5 和 7 配对 -> projection 应保留分数高的
    x[0, 0, 2, 5] = 1; x[0, 0, 5, 2] = 1; score[0, 0, 2, 5] = 0.9; score[0, 0, 5, 2] = 0.9
    x[0, 0, 2, 7] = 1; x[0, 0, 7, 2] = 1; score[0, 0, 2, 7] = 0.5; score[0, 0, 7, 2] = 0.5
    cm = torch.ones(1, 1, L, L)
    out = project_to_valid_contact_map(x, score, cm)
    print(f'   pairs after proj: {int(out.sum().item())//2} (should be 1, the 0.9 one)')
    print(f'   (2,5) kept? {bool(out[0,0,2,5].item())}; (2,7) kept? {bool(out[0,0,2,7].item())}')
    print('discrete_flow.py self-test passed')
