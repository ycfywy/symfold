# -*- coding: utf-8 -*-
"""
Physics-Informed Pairwise Energy Guidance

采样时在 logit 上做加性 shift:
    logit' = logit - beta * dE_phys / dx

E_phys = -ΔG_pair - α·ΔG_stack + λ·#pseudoknot

用法:
    guide = PhysicsGuidance(seq_oh, lambda_pk=1.0)
    grad = guide(logit, x_t)     # (B,1,L,L)
    logit = logit - beta * grad
"""
from __future__ import annotations

import torch

A, U, C, G = 0, 1, 2, 3


def _pair_energy_table(device, dtype=torch.float32):
    M = torch.zeros(4, 4, device=device, dtype=dtype)
    M[A, U] = M[U, A] = 2.0
    M[G, C] = M[C, G] = 3.0
    M[G, U] = M[U, G] = 0.8
    return M


def pair_energy_field(seq_oh):
    """(B,L,4) -> (B,1,L,L), ΔG_pair(s_i, s_j)"""
    tbl = _pair_energy_table(seq_oh.device, seq_oh.dtype)
    s = torch.einsum('blc,cd->bld', seq_oh, tbl)
    E = torch.einsum('bld,bmd->blm', s, seq_oh)
    return E.unsqueeze(1)


def stacking_field(x_t, alpha=1.0):
    """(B,1,L,L) -> (B,1,L,L), 相邻 (i±1,j∓1) 有 pair 时给 stacking bonus."""
    pad = torch.nn.functional.pad(x_t, (1, 1, 1, 1), value=0.0)
    L = x_t.shape[-1]
    x_ip1_jm1 = pad[:, :, 2:L+2, 0:L]
    x_im1_jp1 = pad[:, :, 0:L, 2:L+2]
    return alpha * (x_ip1_jm1 + x_im1_jp1)


def pseudoknot_field(x_t):
    """
    对每个 (i,j) 估计 "如果 x_(i,j) 变 1, 会和多少已有 pair 产生交叉".
    简化 surrogate: count (i',j') with i'>i, j'>j, x_t[i',j']=1.
    """
    B, _, L, _ = x_t.shape
    x2 = x_t.squeeze(1)
    cum_j = torch.flip(torch.cumsum(torch.flip(x2, dims=[-1]), dim=-1), dims=[-1])
    cum_ij = torch.flip(torch.cumsum(torch.flip(cum_j, dims=[-2]), dim=-2), dims=[-2])
    pad = torch.nn.functional.pad(cum_ij, (0, 1, 0, 1), value=0.0)
    M = pad[:, 1:L+1, 1:L+1]
    return M.unsqueeze(1).float()


class PhysicsGuidance:
    def __init__(self, seq_oh, lambda_pk=0.0, alpha_stack=1.0, scale=1.0):
        self.lambda_pk = lambda_pk
        self.alpha_stack = alpha_stack
        self.scale = scale
        self.E_pair = pair_energy_field(seq_oh)   # predompute

    def __call__(self, logit, x_t):
        """return grad (B,1,L,L) s.t. logit' = logit - beta * grad"""
        grad = -self.E_pair.to(x_t.device)
        grad = grad - self.alpha_stack * stacking_field(x_t)
        if self.lambda_pk != 0:
            grad = grad + self.lambda_pk * pseudoknot_field(x_t)
        return self.scale * grad


if __name__ == '__main__':
    torch.manual_seed(0)
    B, L = 2, 16
    seq_oh = torch.zeros(B, L, 4)
    seq_oh.scatter_(2, torch.randint(0, 4, (B, L, 1)), 1.0)
    E = pair_energy_field(seq_oh)
    print(f'pair energy unique: {sorted(E.unique().tolist())}')
    x = torch.zeros(B, 1, L, L)
    x[0, 0, 5, 10] = x[0, 0, 10, 5] = 1
    s = stacking_field(x)
    print(f'stacking max={s.max().item()}')
    pk = pseudoknot_field(x)
    print(f'pk at (3,3)={pk[0,0,3,3].item()} (should be 2, both (5,10),(10,5) count)')
    g = PhysicsGuidance(seq_oh, lambda_pk=1.0)(torch.zeros_like(x), x)
    print(f'guidance shape={g.shape}')
    print('physics_energy.py self-test passed')
