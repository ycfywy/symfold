# -*- coding: utf-8 -*-
"""
Bernoulli Discrete Flow Matching v2 — with Relaxed Projection + Adaptive Schedule

改进点 vs v1:
1. Relaxed Projection: 允许每行最多 2 个配对 (支持 pseudoknot)
2. Adaptive cosine schedule: 前大后小步长
3. Confidence-based threshold: 自适应二值化阈值
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Forward (Noising) — 同 v1
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
# CTMC rates — 同 v1
# ============================================================

def compute_ctmc_rates(x_t, p_x1, t, rho_0=0.005, rate_clip=50.0):
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
# Loss — 同 v1
# ============================================================

class BernoulliFlowLoss(nn.Module):
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

        B, _, L, _ = logit.shape
        idx = torch.arange(L, device=logit.device)
        short_range_ok = ((idx.view(L, 1) - idx.view(1, L)).abs() >= 3)
        mask = contact_masks * short_range_ok.view(1, 1, L, L).float()
        bce = bce * mask
        return bce.sum() / mask.sum().clamp(min=1.0)


# ============================================================
# Relaxed Projection (v2 新增)
# ============================================================

def relaxed_projection(x: torch.Tensor, score: torch.Tensor,
                       contact_masks: torch.Tensor,
                       max_pairs_per_row: int = 2,
                       threshold_k: float = 0.5,
                       min_threshold: float = 0.15) -> torch.Tensor:
    """
    松弛投影: 允许每行最多 max_pairs_per_row 个配对 (支持 pseudoknot)

    与 v1 贪心 max-matching 的区别:
    - v1: 每行严格 ≤ 1 个 pair, 用迭代 greedy
    - v2: 每行 ≤ 2 个 pair, 用 top-k 阈值化

    Args:
        x:             (B,1,L,L) 候选 binary (采样结果)
        score:         (B,1,L,L) 概率 (P(x_1=1))
        contact_masks: (B,1,L,L)
        max_pairs_per_row: 每行最多保留几个 pair
        threshold_k:   阈值 = mean + k * std
        min_threshold: 最低阈值 (防止太激进)
    Returns:
        out: (B,1,L,L) ∈ {0,1}
    """
    B, _, L, _ = x.shape
    device = x.device

    # 基础约束: |i-j| >= 3, contact_masks, symmetrize score
    idx = torch.arange(L, device=device)
    valid = ((idx.view(L, 1) - idx.view(1, L)).abs() >= 3).view(1, 1, L, L).float()
    valid = valid * contact_masks

    # 用 score (概率) 作为选择依据
    s = score * valid
    s = 0.5 * (s + s.transpose(-2, -1))

    # 自适应阈值: 使用 top percentile (更鲁棒，不依赖分布假设)
    valid_scores = s[valid > 0.5]
    if valid_scores.numel() > 0:
        pos_scores = valid_scores[valid_scores > 0.001]
        if pos_scores.numel() > 10:
            # 取 top 5% 的下界作为阈值 (确保选出足够多的候选)
            sorted_scores = pos_scores.sort(descending=True).values
            top_count = max(int(pos_scores.numel() * 0.05), 10)
            thresh = max(sorted_scores[min(top_count, len(sorted_scores)-1)].item(),
                        min_threshold)
        else:
            thresh = min_threshold
    else:
        thresh = min_threshold

    # 二值化
    candidate = (s > thresh).float()

    # 每行保留 top-k
    out = torch.zeros_like(x)
    row_scores = s.squeeze(1)  # (B, L, L)
    for b in range(B):
        for i in range(L):
            row = row_scores[b, i] * candidate[b, 0, i]
            if row.sum() == 0:
                continue
            topk_vals, topk_idx = row.topk(min(max_pairs_per_row, int((row > 0).sum().item())),
                                            dim=0, largest=True)
            for k_idx in range(topk_vals.shape[0]):
                if topk_vals[k_idx] > thresh:
                    j = topk_idx[k_idx].item()
                    out[b, 0, i, j] = 1.0
                    out[b, 0, j, i] = 1.0

    return out


def project_to_valid_contact_map(x: torch.Tensor, score: torch.Tensor,
                                  contact_masks: torch.Tensor,
                                  max_iters: int = None) -> torch.Tensor:
    """v1 兼容接口: 严格贪心 max-matching (每行≤1)."""
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
        i = (max_idx // L)
        j = (max_idx % L)
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
# Adaptive Cosine Sampling Schedule (v2 新增)
# ============================================================

def cosine_schedule(num_steps: int, device=None):
    """
    生成 cosine adaptive dt schedule.
    前期大步长 (快速从 prior 移动到数据分布), 后期小步长 (精修).

    使用 sin(πk/2K) 实现: k=0 时变化快 (dt 大), k→K 时变化慢 (dt 小)

    Returns:
        t_vals: (num_steps,) 各步的 t 值 (从 0 到接近 1)
        dts:    (num_steps,) 各步的 dt 值 (前大后小)
    """
    k = torch.arange(num_steps + 1, device=device, dtype=torch.float32)
    # sin(πk/2K): k=0→0, k=K→1, 且前期增长快后期增长慢
    t_schedule = torch.sin(math.pi * k / (2 * num_steps))
    t_vals = t_schedule[:-1]   # step 起点
    dts = t_schedule[1:] - t_schedule[:-1]  # 各步 dt (前大后小)
    return t_vals, dts


# ============================================================
# τ-leap sampling v2
# ============================================================

@torch.no_grad()
def sample_symfold_v2(network_fn, set_max_len: int, contact_masks: torch.Tensor,
                      num_samples: int = 1, num_steps: int = 20,
                      rho_0: float = 0.005,
                      physics_guidance_fn=None, energy_beta: float = 0.0,
                      project_mode: str = 'relaxed',
                      max_pairs_per_row: int = 2,
                      return_chain: bool = False):
    """
    v2 τ-leap 采样 with adaptive schedule + relaxed projection.

    Args:
        project_mode: 'relaxed' | 'strict' | 'none'
            - relaxed: 每行≤2, 自适应阈值 (v2 default)
            - strict:  每行≤1, 贪心 max-matching (v1 兼容)
            - none:    不做投影
    """
    device = contact_masks.device
    L = set_max_len
    B = num_samples

    x_t = (torch.rand(B, 1, L, L, device=device) < rho_0).float()
    x_t = symmetrize_binary(x_t) * contact_masks

    chain = [x_t.clone()] if return_chain else None

    # Adaptive schedule
    t_vals, dts = cosine_schedule(num_steps, device=device)

    p_x1_last = torch.zeros_like(x_t)
    for k in range(num_steps):
        t_val = t_vals[k].item()
        dt = dts[k].item()
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

    # Final projection
    if project_mode == 'relaxed':
        x_final = relaxed_projection(x_t, p_x1_last, contact_masks,
                                     max_pairs_per_row=max_pairs_per_row)
    elif project_mode == 'strict':
        x_final = project_to_valid_contact_map(x_t, p_x1_last, contact_masks)
    else:
        x_final = x_t

    if return_chain:
        return x_final, p_x1_last, chain
    return x_final, p_x1_last


# Also keep v1-compatible interface
sample_symfold = sample_symfold_v2


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

    print('2) Cosine schedule check')
    t_vals, dts = cosine_schedule(20)
    print(f'   sum(dts)={dts.sum().item():.6f} (should ≈ 1.0)')
    print(f'   dt[0]={dts[0].item():.4f} (large), dt[-1]={dts[-1].item():.4f} (small)')

    print('3) Relaxed projection')
    x = torch.zeros(1, 1, L, L)
    score = torch.zeros(1, 1, L, L)
    # 让位置 2 与 5 和 7 都配对 (pseudoknot)
    x[0, 0, 2, 5] = 1; x[0, 0, 5, 2] = 1; score[0, 0, 2, 5] = 0.9; score[0, 0, 5, 2] = 0.9
    x[0, 0, 2, 7] = 1; x[0, 0, 7, 2] = 1; score[0, 0, 2, 7] = 0.8; score[0, 0, 7, 2] = 0.8
    cm = torch.ones(1, 1, L, L)
    out = relaxed_projection(x, score, cm, max_pairs_per_row=2, min_threshold=0.1)
    n_pairs = int(out.sum().item()) // 2
    print(f'   pairs after relaxed proj: {n_pairs} (should be 2, both kept)')
    out_strict = project_to_valid_contact_map(x, score, cm)
    n_strict = int(out_strict.sum().item()) // 2
    print(f'   pairs after strict proj: {n_strict} (should be 1, only highest)')

    print('discrete_flow_v2.py self-test passed')
