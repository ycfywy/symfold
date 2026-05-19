# -*- coding: utf-8 -*-
"""
Symmetry-Equivariant Axial DiT (SE-DiT) Backbone

设计要点:
1. 小 patch (4) 保留像素级精度
2. Axial attention (row + col), 共享 QKV → 严格对称等变
3. UFold feature map 直接 patch-embed 加到 token (保空间)
4. AdaLN-Zero 全局调制 (时间 + FM/UFold pooling)
5. 输出对称化 + 短程 + 对角线 mask
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000.0) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class SharedAxialAttention(nn.Module):
    """row + col attention, 共享 QKV 权重."""

    def __init__(self, dim, num_heads=4, dim_head=32, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        inner = num_heads * dim_head
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def _attn(self, tokens):
        qkv = self.to_qkv(tokens).chunk(3, dim=-1)
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        a = (q @ k.transpose(-2, -1)) * self.scale
        a = F.softmax(a, dim=-1)
        out = a @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def forward(self, tokens):
        # tokens: (B, H, W, D)
        B, H, W, D = tokens.shape
        # row
        row = rearrange(tokens, 'b h w d -> (b h) w d')
        tokens = tokens + rearrange(self._attn(row), '(b h) w d -> b h w d', b=B)
        # col
        col = rearrange(tokens, 'b h w d -> (b w) h d')
        tokens = tokens + rearrange(self._attn(col), '(b w) h d -> b h w d', b=B)
        return tokens


class FFN(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SEDiTBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dim_head=32, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SharedAxialAttention(dim, num_heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FFN(dim, mlp_ratio, dropout)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, cond):
        # x:(B,H,W,D), cond:(B,D)
        sh1, sc1, g1, sh2, sc2, g2 = self.adaLN(cond).chunk(6, dim=-1)
        def expand(t):
            return t.view(t.shape[0], 1, 1, t.shape[-1])
        h = self.norm1(x) * (1 + expand(sc1)) + expand(sh1)
        h = self.attn(h)
        x = x + expand(g1) * h
        h = self.norm2(x) * (1 + expand(sc2)) + expand(sh2)
        h = self.ff(h)
        x = x + expand(g2) * h
        return x


class PatchEmbed2D(nn.Module):
    def __init__(self, in_channels, hidden_dim, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)   # (B,H,W,D)
        return self.norm(x)


class UnPatchify2D(nn.Module):
    def __init__(self, hidden_dim, out_channels, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = nn.Linear(hidden_dim, out_channels * patch_size * patch_size)

    def forward(self, tokens):
        B, H, W, D = tokens.shape
        P = self.patch_size
        x = self.proj(tokens).view(B, H, W, self.out_channels, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(B, self.out_channels, H * P, W * P)


class AxialPosEmbed(nn.Module):
    def __init__(self, max_side, dim):
        super().__init__()
        self.row_emb = nn.Parameter(torch.zeros(1, max_side, 1, dim))
        self.col_emb = nn.Parameter(torch.zeros(1, 1, max_side, dim))
        nn.init.trunc_normal_(self.row_emb, std=0.02)
        nn.init.trunc_normal_(self.col_emb, std=0.02)

    def forward(self, x):
        B, H, W, D = x.shape
        return x + self.row_emb[:, :H, :, :] + self.col_emb[:, :, :W, :]


class SEDiT(nn.Module):
    """
    Symmetry-Equivariant Axial DiT.

    Forward:
        x_t:    (B, 1, L, L), {0,1}
        t:      (B,), [0,1]
        fm_emb: (B, L, 640)
        fm_attn: (B, 240, L, L)
        seq_oh: (B, L, 4)
        u_cond: (B, cond_dim, L, L)
    Returns:
        logit: (B, 1, L, L)  (sigmoid 即 P(x_1=1))
    """

    def __init__(self,
                  hidden_dim: int = 192,
                  num_heads: int = 4,
                  dim_head: int = 48,
                  num_layers: int = 6,
                  patch_size: int = 4,
                  cond_dim: int = 8,
                  fm_emb_dim: int = 640,
                  fm_attn_dim: int = 240,
                  max_len: int = 640,
                  mlp_ratio: int = 4,
                  dropout: float = 0.1,
                  fm_proj_dim: int = 8,
                  fm_attn_proj_dim: int = 8,
                  xt_emb_dim: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # 输入分支
        self.x_t_embedding = nn.Embedding(2, xt_emb_dim)
        self.fm_emb_proj = nn.Sequential(
            nn.Linear(fm_emb_dim, 64), nn.GELU(), nn.Linear(64, fm_proj_dim))
        self.fm_attn_proj = nn.Sequential(
            nn.Conv2d(fm_attn_dim, 16, 1), nn.GELU(),
            nn.Conv2d(16, fm_attn_proj_dim, 1))
        in_channels = (xt_emb_dim + 2 * fm_proj_dim + fm_attn_proj_dim
                       + 2 * 4 + cond_dim)
        self.patch_embed = PatchEmbed2D(in_channels, hidden_dim, patch_size)
        self.u_patch_embed = nn.Conv2d(cond_dim, hidden_dim, patch_size, patch_size)
        self.pos_embed = AxialPosEmbed(max_len // patch_size + 1, hidden_dim)

        # 全局条件 → AdaLN
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.fm_global = nn.Sequential(
            nn.Linear(fm_emb_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.u_global = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.cond_fuse = nn.Linear(3 * hidden_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            SEDiTBlock(hidden_dim, num_heads, dim_head, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.unpatch = UnPatchify2D(hidden_dim, 1, patch_size)

    @staticmethod
    def _outer_concat(x):
        """(B, C, L) -> (B, 2C, L, L)"""
        B, C, L = x.shape
        xi = x.unsqueeze(-1).expand(-1, -1, -1, L)
        xj = x.unsqueeze(-2).expand(-1, -1, L, -1)
        return torch.cat([xi, xj], dim=1)

    def _global_cond(self, t, fm_emb, u_cond):
        te = self.time_mlp(t)
        fe = self.fm_global(fm_emb.mean(dim=1))
        ue = self.u_global(u_cond.mean(dim=(-1, -2)))
        return self.cond_fuse(torch.cat([te, fe, ue], dim=-1))

    def _build_features(self, x_t, fm_emb, fm_attn, seq_oh, u_cond):
        x_long = x_t.long().squeeze(1)
        x_emb = self.x_t_embedding(x_long).permute(0, 3, 1, 2)
        fm_proj = self.fm_emb_proj(fm_emb).permute(0, 2, 1)            # (B,d,L)
        fm_2d = self._outer_concat(fm_proj)                              # (B,2d,L,L)
        fm_attn_2 = self.fm_attn_proj(fm_attn)
        fm_attn_2 = 0.5 * (fm_attn_2 + fm_attn_2.transpose(-2, -1))
        seq_2d = self._outer_concat(seq_oh.permute(0, 2, 1))             # (B,8,L,L)
        f = torch.cat([x_emb, fm_2d, fm_attn_2, seq_2d, u_cond], dim=1)
        return 0.5 * (f + f.transpose(-2, -1))

    def forward(self, x_t, t, *, fm_emb, fm_attn, seq_oh, u_cond,
                contact_masks=None):
        B, _, L, _ = x_t.shape
        f = self._build_features(x_t, fm_emb, fm_attn, seq_oh, u_cond)
        tokens = self.patch_embed(f)
        u_tok = self.u_patch_embed(u_cond).permute(0, 2, 3, 1)
        tokens = tokens + u_tok
        tokens = self.pos_embed(tokens)
        cond = self._global_cond(t, fm_emb, u_cond)
        for blk in self.blocks:
            tokens = blk(tokens, cond)
        sh, sc = self.final_adaLN(cond).chunk(2, dim=-1)
        def expand(x): return x.view(x.shape[0], 1, 1, x.shape[-1])
        tokens = self.final_norm(tokens) * (1 + expand(sc)) + expand(sh)
        logit = self.unpatch(tokens)
        logit = 0.5 * (logit + logit.transpose(-2, -1))

        # 短程 + 对角线 + padding mask
        device = logit.device
        idx = torch.arange(L, device=device)
        short = (idx.view(L, 1) - idx.view(1, L)).abs() < 3
        logit = logit.masked_fill(short.view(1, 1, L, L), -10.0)
        if contact_masks is not None:
            logit = logit.masked_fill(contact_masks < 0.5, -10.0)
        return logit


if __name__ == '__main__':
    torch.manual_seed(0)
    B, L = 2, 32
    m = SEDiT(hidden_dim=64, num_heads=2, dim_head=16, num_layers=2,
              patch_size=4, cond_dim=8, max_len=64).eval()
    x_t = (torch.rand(B, 1, L, L) > 0.99).float()
    x_t = torch.maximum(x_t, x_t.transpose(-2, -1))
    fm_emb = torch.randn(B, L, 640)
    fm_attn = torch.randn(B, 240, L, L)
    fm_attn = 0.5 * (fm_attn + fm_attn.transpose(-2, -1))
    seq_oh = F.one_hot(torch.randint(0, 4, (B, L)), 4).float()
    u_cond = torch.randn(B, 8, L, L)
    u_cond = 0.5 * (u_cond + u_cond.transpose(-2, -1))
    cm = torch.ones(B, 1, L, L)
    t = torch.rand(B)
    with torch.no_grad():
        out = m(x_t, t, fm_emb=fm_emb, fm_attn=fm_attn, seq_oh=seq_oh,
                u_cond=u_cond, contact_masks=cm)
    diff = (out - out.transpose(-2, -1)).abs().max().item()
    print(f'symmetry: max|out-out.T|={diff:.6e}')
    n = sum(p.numel() for p in m.parameters())
    print(f'small SE-DiT params: {n:,}')
    print('se_dit.py self-test passed')
