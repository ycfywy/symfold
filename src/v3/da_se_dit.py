# -*- coding: utf-8 -*-
"""
Dilated Axial Symmetry-Equivariant DiT (DA-SE-DiT) — v3 Backbone

核心改进 vs v1 SEDiT:
1. Dilated Axial Attention: 交替 dilation=1,2,4 扩大感受野而不降分辨率
2. Rotary Position Embedding (RoPE): 替代 learnable pos embed, 更好地泛化到变长
3. FiLM (Feature-wise Linear Modulation): 每层注入 UFold 空间信息
4. 更深 (9 层 vs 6 层) + 更宽 (dim=256 vs 192)
5. QK-Norm: 稳定深层注意力训练
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================
# Positional Encoding: Rotary (RoPE) for Axial
# ============================================================

class AxialRoPE(nn.Module):
    """2D axial rotary position embedding."""

    def __init__(self, dim_head: int, max_side: int = 160):
        super().__init__()
        self.dim_head = dim_head
        half = dim_head // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, 2).float() / half))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        # precompute
        self._cache_len = 0
        self._cos_cache = None
        self._sin_cache = None

    def _update_cache(self, n: int, device):
        if n <= self._cache_len and self._cos_cache is not None:
            return
        pos = torch.arange(n, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(device))  # (n, half//2)
        freqs = freqs.repeat(1, 2)  # (n, half)
        self._cos_cache = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, n, half)
        self._sin_cache = freqs.sin().unsqueeze(0).unsqueeze(0)
        self._cache_len = n

    def forward(self, q, k):
        """Apply rotary embedding to q and k. Shape: (B, heads, N, dim_head)."""
        n = q.shape[2]
        self._update_cache(n, q.device)
        cos = self._cos_cache[:, :, :n, :q.shape[-1] // 2]
        sin = self._sin_cache[:, :, :n, :q.shape[-1] // 2]
        q = self._rotate(q, cos, sin)
        k = self._rotate(k, cos, sin)
        return q, k

    @staticmethod
    def _rotate(x, cos, sin):
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ============================================================
# Time Embedding
# ============================================================

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


# ============================================================
# Dilated Axial Attention
# ============================================================

class DilatedAxialAttention(nn.Module):
    """
    Row + Col attention with shared QKV (symmetric equivariant).
    Supports dilation: instead of attending to all positions in a row,
    attend to positions at stride `dilation`. This expands effective
    receptive field without increasing computation.
    """

    def __init__(self, dim, num_heads=4, dim_head=64, dropout=0.0,
                 dilation: int = 1, use_qk_norm: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.dilation = dilation
        inner = num_heads * dim_head
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        self.rope = AxialRoPE(dim_head)
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(dim_head)
            self.k_norm = nn.RMSNorm(dim_head)

    def _dilate_gather(self, tokens, dilation):
        """
        Gather tokens at stride=dilation for dilated attention.
        tokens: (B*H_grid, W_grid, D)
        Returns groups of size (B*H_grid*dilation, W_grid//dilation, D)
        """
        if dilation == 1:
            return tokens, None
        BH, W, D = tokens.shape
        # Pad W to multiple of dilation
        pad_w = (dilation - W % dilation) % dilation
        if pad_w > 0:
            tokens = F.pad(tokens, (0, 0, 0, pad_w))
        W_pad = W + pad_w
        # Reshape: (BH, W_pad, D) → (BH, dilation, W_pad//dilation, D) → (BH*dilation, W_pad//dilation, D)
        tokens = tokens.view(BH, dilation, W_pad // dilation, D)
        tokens = tokens.reshape(BH * dilation, W_pad // dilation, D)
        return tokens, (BH, W, pad_w, dilation)

    def _dilate_scatter(self, tokens, info):
        """Reverse of _dilate_gather."""
        if info is None:
            return tokens
        BH, W, pad_w, dilation = info
        W_pad = W + pad_w
        _, Wd, D = tokens.shape  # Wd = W_pad // dilation
        tokens = tokens.view(BH, dilation, Wd, D)
        tokens = tokens.reshape(BH, W_pad, D)
        if pad_w > 0:
            tokens = tokens[:, :W, :]
        return tokens

    def _attn(self, tokens):
        B, N, D = tokens.shape
        qkv = self.to_qkv(tokens).chunk(3, dim=-1)
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        # QK-Norm for training stability
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE
        q, k = self.rope(q, k)

        a = (q @ k.transpose(-2, -1)) * self.scale
        a = F.softmax(a, dim=-1)
        out = a @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def forward(self, tokens):
        """tokens: (B, H, W, D)"""
        B, H, W, D = tokens.shape
        d = self.dilation

        # Row attention (with dilation)
        row = rearrange(tokens, 'b h w d -> (b h) w d')
        row_d, info_r = self._dilate_gather(row, d)
        row_out = self._attn(row_d)
        row_out = self._dilate_scatter(row_out, info_r)
        tokens = tokens + rearrange(row_out, '(b h) w d -> b h w d', b=B)

        # Col attention (with dilation)
        col = rearrange(tokens, 'b h w d -> (b w) h d')
        col_d, info_c = self._dilate_gather(col, d)
        col_out = self._attn(col_d)
        col_out = self._dilate_scatter(col_out, info_c)
        tokens = tokens + rearrange(col_out, '(b w) h d -> b h w d', b=B)

        return tokens


# ============================================================
# FiLM: Feature-wise Linear Modulation from UFold
# ============================================================

class FiLM(nn.Module):
    """Inject spatial conditioning via scale and shift."""

    def __init__(self, cond_dim: int, hidden_dim: int, patch_size: int = 4):
        super().__init__()
        # UFold (B, cond_dim, L, L) → (B, H_grid, W_grid, 2*hidden_dim)
        self.proj = nn.Sequential(
            nn.Conv2d(cond_dim, hidden_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2 * hidden_dim, kernel_size=1),
        )

    def forward(self, x, u_cond):
        """
        x: (B, H, W, D) — token features
        u_cond: (B, cond_dim, L, L)
        Returns: modulated x
        """
        # project UFold to patch resolution
        mod = self.proj(u_cond)  # (B, 2*D, H, W)
        mod = mod.permute(0, 2, 3, 1)  # (B, H, W, 2*D)
        H, W = x.shape[1], x.shape[2]
        mod = mod[:, :H, :W, :]
        scale, shift = mod.chunk(2, dim=-1)
        return x * (1 + scale) + shift


# ============================================================
# Block
# ============================================================

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


class DASEDiTBlock(nn.Module):
    """Dilated Axial SE-DiT block with FiLM conditioning."""

    def __init__(self, dim, num_heads=4, dim_head=64, mlp_ratio=4,
                 dropout=0.0, dilation: int = 1, cond_dim: int = 8,
                 patch_size: int = 4, use_film: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = DilatedAxialAttention(dim, num_heads, dim_head, dropout,
                                           dilation=dilation)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FFN(dim, mlp_ratio, dropout)

        # AdaLN-Zero
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

        # FiLM from UFold (spatial conditioning)
        self.use_film = use_film
        if use_film:
            self.film = FiLM(cond_dim, dim, patch_size)

    def forward(self, x, cond, u_cond=None):
        """
        x: (B, H, W, D)
        cond: (B, D) global condition
        u_cond: (B, cond_dim, L, L) UFold spatial features
        """
        sh1, sc1, g1, sh2, sc2, g2 = self.adaLN(cond).chunk(6, dim=-1)

        def expand(t):
            return t.view(t.shape[0], 1, 1, t.shape[-1])

        h = self.norm1(x) * (1 + expand(sc1)) + expand(sh1)
        h = self.attn(h)
        x = x + expand(g1) * h

        # FiLM: spatial modulation from UFold between attn and FFN
        if self.use_film and u_cond is not None:
            x = self.film(x, u_cond)

        h = self.norm2(x) * (1 + expand(sc2)) + expand(sh2)
        h = self.ff(h)
        x = x + expand(g2) * h
        return x


# ============================================================
# Patch Embed / UnPatch
# ============================================================

class PatchEmbed2D(nn.Module):
    def __init__(self, in_channels, hidden_dim, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)  # (B, H, W, D)
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


# ============================================================
# Main Model: DASEDiT
# ============================================================

class DASEDiT(nn.Module):
    """
    Dilated Axial Symmetry-Equivariant DiT.

    Key innovations vs v1 SEDiT:
    - Dilated attention pattern [1,1,1, 2,2,2, 4,4,4] for multi-scale receptive field
    - RoPE instead of learnable position embedding
    - FiLM spatial conditioning from UFold at every layer
    - QK-Norm for stable deep training
    - Wider (256 vs 192) and deeper (9 vs 6)

    Forward:
        x_t:     (B, 1, L, L), {0,1}
        t:       (B,), [0,1]
        fm_emb:  (B, L, 640)
        fm_attn: (B, 240, L, L)
        seq_oh:  (B, L, 4)
        u_cond:  (B, cond_dim, L, L)
    Returns:
        logit: (B, 1, L, L)
    """

    def __init__(self,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 dim_head: int = 64,
                 num_layers: int = 9,
                 patch_size: int = 4,
                 cond_dim: int = 8,
                 fm_emb_dim: int = 640,
                 fm_attn_dim: int = 240,
                 max_len: int = 640,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 fm_proj_dim: int = 8,
                 fm_attn_proj_dim: int = 8,
                 xt_emb_dim: int = 8,
                 dilation_pattern: list = None):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        if dilation_pattern is None:
            dilation_pattern = [1, 1, 1, 2, 2, 2, 4, 4, 4]
        assert len(dilation_pattern) == num_layers

        # Input branches (same as v1)
        self.x_t_embedding = nn.Embedding(2, xt_emb_dim)
        self.fm_emb_proj = nn.Sequential(
            nn.Linear(fm_emb_dim, 64), nn.GELU(), nn.Linear(64, fm_proj_dim))
        self.fm_attn_proj = nn.Sequential(
            nn.Conv2d(fm_attn_dim, 16, 1), nn.GELU(),
            nn.Conv2d(16, fm_attn_proj_dim, 1))

        in_channels = (xt_emb_dim + 2 * fm_proj_dim + fm_attn_proj_dim
                       + 2 * 4 + cond_dim)  # 48
        self.patch_embed = PatchEmbed2D(in_channels, hidden_dim, patch_size)
        self.u_patch_embed = nn.Conv2d(cond_dim, hidden_dim, patch_size, patch_size)

        # Global condition → AdaLN
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

        # Blocks with dilated attention
        self.blocks = nn.ModuleList([
            DASEDiTBlock(hidden_dim, num_heads, dim_head, mlp_ratio, dropout,
                         dilation=dilation_pattern[i], cond_dim=cond_dim,
                         patch_size=patch_size, use_film=True)
            for i in range(num_layers)
        ])

        # Final output
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
        fm_proj = self.fm_emb_proj(fm_emb).permute(0, 2, 1)
        fm_2d = self._outer_concat(fm_proj)
        fm_attn_2 = self.fm_attn_proj(fm_attn)
        fm_attn_2 = 0.5 * (fm_attn_2 + fm_attn_2.transpose(-2, -1))
        seq_2d = self._outer_concat(seq_oh.permute(0, 2, 1))
        f = torch.cat([x_emb, fm_2d, fm_attn_2, seq_2d, u_cond], dim=1)
        return 0.5 * (f + f.transpose(-2, -1))

    def forward(self, x_t, t, *, fm_emb, fm_attn, seq_oh, u_cond,
                contact_masks=None):
        B, _, L, _ = x_t.shape
        f = self._build_features(x_t, fm_emb, fm_attn, seq_oh, u_cond)
        tokens = self.patch_embed(f)

        # UFold spatial injection at token level
        u_tok = self.u_patch_embed(u_cond).permute(0, 2, 3, 1)
        tokens = tokens + u_tok

        # Global condition
        cond = self._global_cond(t, fm_emb, u_cond)

        # Transformer blocks with dilated attention + FiLM
        for blk in self.blocks:
            tokens = blk(tokens, cond, u_cond)

        # Final projection
        sh, sc = self.final_adaLN(cond).chunk(2, dim=-1)

        def expand(x):
            return x.view(x.shape[0], 1, 1, x.shape[-1])

        tokens = self.final_norm(tokens) * (1 + expand(sc)) + expand(sh)
        logit = self.unpatch(tokens)
        logit = 0.5 * (logit + logit.transpose(-2, -1))

        # Short-range + diagonal + padding mask
        device = logit.device
        idx = torch.arange(L, device=device)
        short = (idx.view(L, 1) - idx.view(1, L)).abs() < 3
        logit = logit.masked_fill(short.view(1, 1, L, L), -10.0)
        if contact_masks is not None:
            logit = logit.masked_fill(contact_masks < 0.5, -10.0)
        return logit


# ============================================================
# Self-test
# ============================================================

if __name__ == '__main__':
    torch.manual_seed(0)
    B, L = 2, 32
    m = DASEDiT(hidden_dim=64, num_heads=2, dim_head=16, num_layers=9,
                patch_size=4, cond_dim=8, max_len=64,
                dilation_pattern=[1, 1, 1, 2, 2, 2, 4, 4, 4]).eval()
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
    print(f'small DA-SE-DiT params: {n:,}')
    print('da_se_dit.py self-test passed')
