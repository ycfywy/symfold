# -*- coding: utf-8 -*-
"""
Dilated Axial Symmetry-Equivariant DiT v4 (DA-SE-DiT-v4) — Enhanced Backbone

核心改进 vs v3:
1. Multi-Layer RNA-FM Features: 提取浅/中/深层 (layers 3,6,9,12) embedding，
   用 learnable weighted fusion 融合，捕获不同层次的 RNA 信息
2. Triangle Multiplicative Update: 受 AlphaFold2 启发，显式捕获三体约束
   (如果 i-k 配对且 k-j 配对，则影响 i-j 的预测)，后 3 层插入
3. Adaptive Density Head: 辅助预测 pair density，用于 loss 中动态调整 pos_weight
4. Gated FFN (SwiGLU): 替代 GELU FFN，更好的参数效率
5. 保留 v3 的: RoPE, QK-Norm, FiLM, Dilated Axial Attention, AdaLN-Zero
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================
# Positional Encoding: Rotary (RoPE) for Axial — same as v3
# ============================================================

class AxialRoPE(nn.Module):
    """2D axial rotary position embedding."""

    def __init__(self, dim_head: int, max_side: int = 160):
        super().__init__()
        self.dim_head = dim_head
        half = dim_head // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, 2).float() / half))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._cache_len = 0
        self._cos_cache = None
        self._sin_cache = None

    def _update_cache(self, n: int, device):
        if n <= self._cache_len and self._cos_cache is not None:
            return
        pos = torch.arange(n, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(device))
        freqs = freqs.repeat(1, 2)
        self._cos_cache = freqs.cos().unsqueeze(0).unsqueeze(0)
        self._sin_cache = freqs.sin().unsqueeze(0).unsqueeze(0)
        self._cache_len = n

    def forward(self, q, k):
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
# Time Embedding — same as v3
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
# Multi-Layer FM Feature Fusion (NEW in v4)
# ============================================================

class MultiLayerFMFusion(nn.Module):
    """
    Learnable weighted fusion of multi-layer RNA-FM representations.
    
    Extracts layers [3, 6, 9, 12] to capture:
    - Layer 3: 浅层 — 局部序列模式 (k-mer motifs, base composition)
    - Layer 6: 中浅层 — 局部结构倾向 (stem seeds, loop regions)
    - Layer 9: 中深层 — 中程依赖 (hairpin loops, internal loops)
    - Layer 12: 深层 — 全局结构语义 (domain-level folding)
    
    Uses learnable layer weights + per-layer projection + final MLP.
    """

    def __init__(self, fm_dim: int = 640, out_dim: int = 16, num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers
        # Learnable scalar weights for each layer (softmax normalized)
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        # Per-layer linear projection (captures layer-specific patterns)
        self.layer_projs = nn.ModuleList([
            nn.Linear(fm_dim, out_dim) for _ in range(num_layers)
        ])
        # Final fusion MLP
        self.fuse = nn.Sequential(
            nn.Linear(out_dim * num_layers, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, layer_reprs: list[torch.Tensor]):
        """
        layer_reprs: list of (B, L, 640) tensors, one per extracted layer.
        Returns: (B, L, out_dim)
        """
        # Softmax-weighted combination
        weights = F.softmax(self.layer_weights, dim=0)
        weighted = sum(w * rep for w, rep in zip(weights, layer_reprs))

        # Per-layer projections
        projs = [proj(rep) for proj, rep in zip(self.layer_projs, layer_reprs)]

        # Concatenate per-layer projections + MLP fusion
        concat = torch.cat(projs, dim=-1)  # (B, L, out_dim * num_layers)
        fused = self.fuse(concat)  # (B, L, out_dim)

        # Add weighted average component (residual connection for stability)
        proj_weighted = self.layer_projs[0](weighted)  # reuse first proj for avg
        return fused + proj_weighted


# ============================================================
# Triangle Multiplicative Update (NEW in v4, inspired by AF2)
# ============================================================

class TriangleMultiplicativeUpdate(nn.Module):
    """
    Captures three-body relationships: z[i,j] is influenced by
    z[i,k] and z[k,j] for all k (outgoing edges from i/j).

    Efficient implementation: uses low-rank projection + einsum.
    Complexity: O(L^2 * d) per patch-level token grid.
    
    This helps enforce: "if i pairs with k, then i shouldn't also pair with j"
    """

    def __init__(self, dim: int, tri_dim: int = 64):
        super().__init__()
        self.tri_dim = tri_dim
        # Project to lower dim for efficiency
        self.proj_left = nn.Linear(dim, tri_dim)
        self.proj_right = nn.Linear(dim, tri_dim)
        self.gate_left = nn.Sequential(nn.Linear(dim, tri_dim), nn.Sigmoid())
        self.gate_right = nn.Sequential(nn.Linear(dim, tri_dim), nn.Sigmoid())
        # Output projection
        self.norm = nn.LayerNorm(tri_dim)
        self.out_proj = nn.Linear(tri_dim, dim)
        self.out_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, z):
        """
        z: (B, H, W, D) — pair representations at patch level
        Returns: updated z with triangle information
        """
        B, H, W, D = z.shape
        # Outgoing edges: for z[i,j], aggregate info from z[i,k] * z[k,j]
        left = self.proj_left(z) * self.gate_left(z)   # (B, H, W, tri_dim)
        right = self.proj_right(z) * self.gate_right(z)  # (B, H, W, tri_dim)

        # Triangle update: sum_k left[i,k] * right[k,j]
        # = einsum('b i k d, b k j d -> b i j d')
        tri = torch.einsum('bild,bljd->bijd', left, right)  # (B, H, W, tri_dim)

        tri = self.norm(tri)
        tri = self.out_proj(tri)  # (B, H, W, D)
        gate = self.out_gate(z)
        return z + gate * tri


# ============================================================
# Dilated Axial Attention — same as v3
# ============================================================

class DilatedAxialAttention(nn.Module):
    """Row + Col attention with shared QKV (symmetric equivariant) + dilation."""

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
        if dilation == 1:
            return tokens, None
        BH, W, D = tokens.shape
        pad_w = (dilation - W % dilation) % dilation
        if pad_w > 0:
            tokens = F.pad(tokens, (0, 0, 0, pad_w))
        W_pad = W + pad_w
        tokens = tokens.view(BH, dilation, W_pad // dilation, D)
        tokens = tokens.reshape(BH * dilation, W_pad // dilation, D)
        return tokens, (BH, W, pad_w, dilation)

    def _dilate_scatter(self, tokens, info):
        if info is None:
            return tokens
        BH, W, pad_w, dilation = info
        W_pad = W + pad_w
        _, Wd, D = tokens.shape
        tokens = tokens.view(BH, dilation, Wd, D)
        tokens = tokens.reshape(BH, W_pad, D)
        if pad_w > 0:
            tokens = tokens[:, :W, :]
        return tokens

    def _attn(self, tokens):
        B, N, D = tokens.shape
        qkv = self.to_qkv(tokens).chunk(3, dim=-1)
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
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

        # Row attention
        row = rearrange(tokens, 'b h w d -> (b h) w d')
        row_d, info_r = self._dilate_gather(row, d)
        row_out = self._attn(row_d)
        row_out = self._dilate_scatter(row_out, info_r)
        tokens = tokens + rearrange(row_out, '(b h) w d -> b h w d', b=B)

        # Col attention
        col = rearrange(tokens, 'b h w d -> (b w) h d')
        col_d, info_c = self._dilate_gather(col, d)
        col_out = self._attn(col_d)
        col_out = self._dilate_scatter(col_out, info_c)
        tokens = tokens + rearrange(col_out, '(b w) h d -> b h w d', b=B)

        return tokens


# ============================================================
# FiLM: Feature-wise Linear Modulation — same as v3
# ============================================================

class FiLM(nn.Module):
    """Inject spatial conditioning via scale and shift."""

    def __init__(self, cond_dim: int, hidden_dim: int, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(cond_dim, hidden_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2 * hidden_dim, kernel_size=1),
        )

    def forward(self, x, u_cond):
        mod = self.proj(u_cond)
        mod = mod.permute(0, 2, 3, 1)
        H, W = x.shape[1], x.shape[2]
        mod = mod[:, :H, :W, :]
        scale, shift = mod.chunk(2, dim=-1)
        return x * (1 + scale) + shift


# ============================================================
# Gated FFN (SwiGLU) — NEW in v4
# ============================================================

class GatedFFN(nn.Module):
    """SwiGLU-style gated FFN, more expressive than GELU FFN."""

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        hidden = int(dim * mult * 2 / 3)  # Adjust for gate splitting
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ============================================================
# Block — enhanced with Triangle Update
# ============================================================

class DASEDiTBlock_v4(nn.Module):
    """v4 block: Dilated Axial Attn + FiLM + Gated FFN + optional Triangle Update."""

    def __init__(self, dim, num_heads=4, dim_head=64, mlp_ratio=4,
                 dropout=0.0, dilation: int = 1, cond_dim: int = 8,
                 patch_size: int = 4, use_film: bool = True,
                 use_triangle: bool = False, tri_dim: int = 64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = DilatedAxialAttention(dim, num_heads, dim_head, dropout,
                                           dilation=dilation)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = GatedFFN(dim, mlp_ratio, dropout)

        # AdaLN-Zero
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

        # FiLM
        self.use_film = use_film
        if use_film:
            self.film = FiLM(cond_dim, dim, patch_size)

        # Triangle Update (only in later blocks)
        self.use_triangle = use_triangle
        if use_triangle:
            self.tri_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.tri_update = TriangleMultiplicativeUpdate(dim, tri_dim=tri_dim)

    def forward(self, x, cond, u_cond=None):
        """
        x: (B, H, W, D)
        cond: (B, D) global condition
        u_cond: (B, cond_dim, L, L)
        """
        sh1, sc1, g1, sh2, sc2, g2 = self.adaLN(cond).chunk(6, dim=-1)

        def expand(t):
            return t.view(t.shape[0], 1, 1, t.shape[-1])

        # Attention
        h = self.norm1(x) * (1 + expand(sc1)) + expand(sh1)
        h = self.attn(h)
        x = x + expand(g1) * h

        # Triangle Update (before FiLM, captures structural constraints)
        if self.use_triangle:
            x = x + self.tri_update(self.tri_norm(x))

        # FiLM: spatial modulation from UFold
        if self.use_film and u_cond is not None:
            x = self.film(x, u_cond)

        # FFN (Gated)
        h = self.norm2(x) * (1 + expand(sc2)) + expand(sh2)
        h = self.ff(h)
        x = x + expand(g2) * h
        return x


# ============================================================
# Patch Embed / UnPatch — same as v3
# ============================================================

class PatchEmbed2D(nn.Module):
    def __init__(self, in_channels, hidden_dim, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
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
# Density Prediction Head (NEW in v4)
# ============================================================

class DensityHead(nn.Module):
    """
    Predicts pair_per_base density from global features.
    Used to adaptively adjust pos_weight during loss computation.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(self, global_feat):
        """global_feat: (B, D) → density: (B, 1)"""
        return self.head(global_feat)


# ============================================================
# Main Model: DASEDiT_v4
# ============================================================

class DASEDiT_v4(nn.Module):
    """
    DA-SE-DiT v4: Multi-Layer FM + Triangle Update + Adaptive Density.

    Key changes vs v3:
    - Input: multi-layer FM features (layers 3,6,9,12) fused via learnable weights
    - Blocks 7-9: add Triangle Multiplicative Update
    - FFN: SwiGLU gating
    - Output: additional density prediction head for adaptive pos_weight
    - Wider multi-layer FM input (16ch FM-2D vs v3's 16ch single-layer)

    Forward:
        x_t:     (B, 1, L, L), {0,1}
        t:       (B,), [0,1]
        fm_multi: list of (B, L, 640) — multi-layer FM embeddings [L3, L6, L9, L12]
        fm_attn: (B, 240, L, L)
        seq_oh:  (B, L, 4)
        u_cond:  (B, cond_dim, L, L)
    Returns:
        logit: (B, 1, L, L)
        density_pred: (B, 1) — predicted pair density (optional)
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
                 fm_multi_out_dim: int = 16,
                 max_len: int = 640,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 fm_attn_proj_dim: int = 8,
                 xt_emb_dim: int = 8,
                 dilation_pattern: list = None,
                 tri_start_layer: int = 6,
                 tri_dim: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.tri_start_layer = tri_start_layer

        if dilation_pattern is None:
            dilation_pattern = [1, 1, 1, 2, 2, 2, 4, 4, 4]
        assert len(dilation_pattern) == num_layers

        # Multi-layer FM fusion (NEW)
        self.fm_multi_fusion = MultiLayerFMFusion(
            fm_dim=fm_emb_dim, out_dim=fm_multi_out_dim, num_layers=4)

        # Input branches
        self.x_t_embedding = nn.Embedding(2, xt_emb_dim)
        # FM → 2D: multi-layer fused embedding projected to pair features
        fm_proj_dim = fm_multi_out_dim // 2  # outer_concat doubles it
        self.fm_emb_proj = nn.Sequential(
            nn.Linear(fm_multi_out_dim, fm_proj_dim),
            nn.GELU(),
        )
        self.fm_attn_proj = nn.Sequential(
            nn.Conv2d(fm_attn_dim, 16, 1), nn.GELU(),
            nn.Conv2d(16, fm_attn_proj_dim, 1))

        # in_channels = xt_emb(8) + fm_2d(2*fm_proj_dim=16) + fm_attn(8) + seq(8) + u_cond(8) = 48
        in_channels = (xt_emb_dim + 2 * fm_proj_dim + fm_attn_proj_dim
                       + 2 * 4 + cond_dim)
        self.patch_embed = PatchEmbed2D(in_channels, hidden_dim, patch_size)
        self.u_patch_embed = nn.Conv2d(cond_dim, hidden_dim, patch_size, patch_size)

        # Global condition → AdaLN
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        # Use multi-layer FM global (concat all layers then project)
        self.fm_global = nn.Sequential(
            nn.Linear(fm_multi_out_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.u_global = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.cond_fuse = nn.Linear(3 * hidden_dim, hidden_dim)

        # Blocks: Triangle update on layers >= tri_start_layer
        self.blocks = nn.ModuleList([
            DASEDiTBlock_v4(
                hidden_dim, num_heads, dim_head, mlp_ratio, dropout,
                dilation=dilation_pattern[i], cond_dim=cond_dim,
                patch_size=patch_size, use_film=True,
                use_triangle=(i >= tri_start_layer),
                tri_dim=tri_dim)
            for i in range(num_layers)
        ])

        # Final output
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.unpatch = UnPatchify2D(hidden_dim, 1, patch_size)

        # Density head (NEW)
        self.density_head = DensityHead(hidden_dim)

    @staticmethod
    def _outer_concat(x):
        """(B, C, L) -> (B, 2C, L, L)"""
        B, C, L = x.shape
        xi = x.unsqueeze(-1).expand(-1, -1, -1, L)
        xj = x.unsqueeze(-2).expand(-1, -1, L, -1)
        return torch.cat([xi, xj], dim=1)

    def _global_cond(self, t, fm_fused, u_cond):
        """Build global condition from time + FM global + UFold global."""
        te = self.time_mlp(t)
        fe = self.fm_global(fm_fused.mean(dim=1))  # (B, D)
        ue = self.u_global(u_cond.mean(dim=(-1, -2)))
        return self.cond_fuse(torch.cat([te, fe, ue], dim=-1))

    def _build_features(self, x_t, fm_fused, fm_attn, seq_oh, u_cond):
        """Build input feature tensor (B, in_ch, L, L)."""
        x_long = x_t.long().squeeze(1)
        x_emb = self.x_t_embedding(x_long).permute(0, 3, 1, 2)  # (B, 8, L, L)

        # Multi-layer FM → 2D projection
        fm_proj = self.fm_emb_proj(fm_fused).permute(0, 2, 1)  # (B, fm_proj_dim, L)
        fm_2d = self._outer_concat(fm_proj)  # (B, 2*fm_proj_dim, L, L)

        # FM attention maps
        fm_attn_2 = self.fm_attn_proj(fm_attn)
        fm_attn_2 = 0.5 * (fm_attn_2 + fm_attn_2.transpose(-2, -1))

        # Sequence 2D
        seq_2d = self._outer_concat(seq_oh.permute(0, 2, 1))

        f = torch.cat([x_emb, fm_2d, fm_attn_2, seq_2d, u_cond], dim=1)
        return 0.5 * (f + f.transpose(-2, -1))

    def forward(self, x_t, t, *, fm_multi: list, fm_attn, seq_oh, u_cond,
                contact_masks=None, return_density: bool = False):
        B, _, L, _ = x_t.shape

        # Multi-layer FM fusion
        fm_fused = self.fm_multi_fusion(fm_multi)  # (B, L, fm_multi_out_dim)

        # Build input features
        f = self._build_features(x_t, fm_fused, fm_attn, seq_oh, u_cond)
        tokens = self.patch_embed(f)

        # UFold spatial injection
        u_tok = self.u_patch_embed(u_cond).permute(0, 2, 3, 1)
        tokens = tokens + u_tok

        # Global condition
        cond = self._global_cond(t, fm_fused, u_cond)

        # Transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens, cond, u_cond)

        # Final projection
        sh, sc = self.final_adaLN(cond).chunk(2, dim=-1)

        def expand(x):
            return x.view(x.shape[0], 1, 1, x.shape[-1])

        tokens = self.final_norm(tokens) * (1 + expand(sc)) + expand(sh)
        logit = self.unpatch(tokens)
        logit = 0.5 * (logit + logit.transpose(-2, -1))

        # Short-range + padding mask
        device = logit.device
        idx = torch.arange(L, device=device)
        short = (idx.view(L, 1) - idx.view(1, L)).abs() < 3
        logit = logit.masked_fill(short.view(1, 1, L, L), -10.0)
        if contact_masks is not None:
            logit = logit.masked_fill(contact_masks < 0.5, -10.0)

        if return_density:
            density = self.density_head(cond)
            return logit, density
        return logit


# ============================================================
# Self-test
# ============================================================

if __name__ == '__main__':
    torch.manual_seed(0)
    B, L = 2, 32
    m = DASEDiT_v4(hidden_dim=64, num_heads=2, dim_head=16, num_layers=9,
                   patch_size=4, cond_dim=8, max_len=64, fm_multi_out_dim=16,
                   dilation_pattern=[1, 1, 1, 2, 2, 2, 4, 4, 4],
                   tri_start_layer=6, tri_dim=32).eval()
    x_t = (torch.rand(B, 1, L, L) > 0.99).float()
    x_t = torch.maximum(x_t, x_t.transpose(-2, -1))
    fm_multi = [torch.randn(B, L, 640) for _ in range(4)]
    fm_attn = torch.randn(B, 240, L, L)
    fm_attn = 0.5 * (fm_attn + fm_attn.transpose(-2, -1))
    seq_oh = F.one_hot(torch.randint(0, 4, (B, L)), 4).float()
    u_cond = torch.randn(B, 8, L, L)
    u_cond = 0.5 * (u_cond + u_cond.transpose(-2, -1))
    cm = torch.ones(B, 1, L, L)
    t = torch.rand(B)
    with torch.no_grad():
        out, density = m(x_t, t, fm_multi=fm_multi, fm_attn=fm_attn,
                         seq_oh=seq_oh, u_cond=u_cond, contact_masks=cm,
                         return_density=True)
    diff = (out - out.transpose(-2, -1)).abs().max().item()
    print(f'symmetry: max|out-out.T|={diff:.6e}')
    print(f'density shape: {density.shape}, values: {density.squeeze().tolist()}')
    n = sum(p.numel() for p in m.parameters())
    print(f'small DA-SE-DiT-v4 params: {n:,}')
    print('da_se_dit_v4.py self-test passed')
