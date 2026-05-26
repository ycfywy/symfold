# -*- coding: utf-8 -*-
"""
Dilated Axial Symmetry-Equivariant DiT v5 (DA-SE-DiT-v5)

核心改进 vs v4:
1. FM Fusion 输出维度: 16 → 64 (大幅减少信息压缩损失)
2. 密度条件注入: density embedding 加入全局条件，让模型知道该预测多少对
3. 输出精修 Conv: UnPatchify 后加 2 层 Conv 在 L×L 上精修 patch 边界
4. 保留 v4 全部: RoPE, QK-Norm, FiLM, Triangle Update, SwiGLU, AdaLN-Zero
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 复用 v4 的公共模块
from src.v4.da_se_dit import (
    AxialRoPE, SinusoidalTimeEmbedding,
    TriangleMultiplicativeUpdate, DASEDiTBlock_v4,
    DensityHead, UnPatchify2D, PatchEmbed2D,
)


# ============================================================
# Multi-Layer FM Feature Fusion v5 (wider: out_dim=64)
# ============================================================

class MultiLayerFMFusion_v5(nn.Module):
    """
    v5: 输出维度从 16 升到 64，减少 RNA-FM 的信息压缩损失。
    
    v4 是 4×640→16 (压缩比 160:1)
    v5 是 4×640→64 (压缩比 40:1) ← 4倍信息保留
    """

    def __init__(self, fm_dim: int = 640, out_dim: int = 64, num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.out_dim = out_dim

        # Learnable scalar weights for each layer (softmax normalized)
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

        # Per-layer linear projection
        self.layer_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fm_dim, out_dim * 2),
                nn.GELU(),
                nn.Linear(out_dim * 2, out_dim),
            ) for _ in range(num_layers)
        ])

        # Final fusion MLP (4×64=256 → 128 → 64)
        self.fuse = nn.Sequential(
            nn.Linear(out_dim * num_layers, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

        # Weighted average projection
        self.avg_proj = nn.Sequential(
            nn.Linear(fm_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, layer_reprs: list[torch.Tensor]):
        """
        layer_reprs: list of (B, L, 640) tensors, one per extracted layer.
        Returns: (B, L, out_dim=64)
        """
        # Softmax-weighted combination
        weights = F.softmax(self.layer_weights, dim=0)
        weighted = sum(w * rep for w, rep in zip(weights, layer_reprs))

        # Per-layer projections
        projs = [proj(rep) for proj, rep in zip(self.layer_projs, layer_reprs)]

        # Concatenate per-layer projections + MLP fusion
        concat = torch.cat(projs, dim=-1)  # (B, L, out_dim * 4 = 256)
        fused = self.fuse(concat)  # (B, L, 64)

        # Add weighted average component (residual)
        proj_weighted = self.avg_proj(weighted)  # (B, L, 64)
        return fused + proj_weighted


# ============================================================
# Output Refinement Conv (NEW in v5)
# ============================================================

class OutputRefineConv(nn.Module):
    """
    在 UnPatchify 后、在全分辨率 L×L 上做轻量精修。
    修正 patch 边界不连续性 + 利用局部上下文精调 logit。
    """

    def __init__(self, mid_ch: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, mid_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_ch, 1, 1),
        )
        # 初始化为近零，开始时等价于 identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, logit):
        """logit: (B, 1, L, L) → refined logit: (B, 1, L, L)"""
        return logit + self.net(logit)  # 残差连接


# ============================================================
# Main Model: DASEDiT_v5
# ============================================================

class DASEDiT_v5(nn.Module):
    """
    DA-SE-DiT v5: Wider FM + Density Conditioning + Output Refinement.

    Key changes vs v4:
    - FM Fusion output: 16 → 64 (4× more info from RNA-FM)
    - Density condition: predicted density embedded into global cond
    - Output refine: 2-layer Conv on full-resolution logit
    - Input channels: 48 → 80 (due to wider FM features)

    Forward:
        x_t:     (B, 1, L, L), {0,1}
        t:       (B,), [0,1]
        fm_multi: list of (B, L, 640) — multi-layer FM embeddings [L3, L6, L9, L12]
        fm_attn: (B, 240, L, L)
        seq_oh:  (B, L, 4)
        u_cond:  (B, cond_dim, L, L)
        density_hint: (B, 1) or None — optional density conditioning
    Returns:
        logit: (B, 1, L, L)
        density_pred: (B, 1)
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
                 fm_multi_out_dim: int = 64,
                 max_len: int = 640,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 fm_attn_proj_dim: int = 8,
                 xt_emb_dim: int = 8,
                 dilation_pattern: list = None,
                 tri_start_layer: int = 6,
                 tri_dim: int = 64,
                 refine_mid_ch: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.tri_start_layer = tri_start_layer
        self.fm_multi_out_dim = fm_multi_out_dim

        if dilation_pattern is None:
            dilation_pattern = [1, 1, 1, 2, 2, 2, 4, 4, 4]
        assert len(dilation_pattern) == num_layers

        # Multi-layer FM fusion (v5: wider output)
        self.fm_multi_fusion = MultiLayerFMFusion_v5(
            fm_dim=fm_emb_dim, out_dim=fm_multi_out_dim, num_layers=4)

        # Input branches
        self.x_t_embedding = nn.Embedding(2, xt_emb_dim)

        # FM → 2D: wider projection (64 → 32, outer_concat doubles to 64)
        fm_proj_dim = fm_multi_out_dim // 2  # 32
        self.fm_emb_proj = nn.Sequential(
            nn.Linear(fm_multi_out_dim, fm_proj_dim),
            nn.GELU(),
        )
        self.fm_attn_proj = nn.Sequential(
            nn.Conv2d(fm_attn_dim, 32, 1), nn.GELU(),
            nn.Conv2d(32, fm_attn_proj_dim, 1))

        # in_channels = xt_emb(8) + fm_2d(2*32=64) + fm_attn(8) + seq(8) + u_cond(8) = 96
        in_channels = (xt_emb_dim + 2 * fm_proj_dim + fm_attn_proj_dim
                       + 2 * 4 + cond_dim)
        self.patch_embed = PatchEmbed2D(in_channels, hidden_dim, patch_size)
        self.u_patch_embed = nn.Conv2d(cond_dim, hidden_dim, patch_size, patch_size)

        # Global condition → AdaLN (v5: add density embedding)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.fm_global = nn.Sequential(
            nn.Linear(fm_multi_out_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.u_global = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        # v5 NEW: density condition embedding
        self.density_emb = nn.Sequential(
            nn.Linear(1, hidden_dim // 4), nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim))
        # Fuse 4 sources (v4 was 3)
        self.cond_fuse = nn.Linear(4 * hidden_dim, hidden_dim)

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

        # v5 NEW: output refinement conv
        self.refine = OutputRefineConv(mid_ch=refine_mid_ch)

        # Density head
        self.density_head = DensityHead(hidden_dim)

    @staticmethod
    def _outer_concat(x):
        """(B, C, L) -> (B, 2C, L, L)"""
        B, C, L = x.shape
        xi = x.unsqueeze(-1).expand(-1, -1, -1, L)
        xj = x.unsqueeze(-2).expand(-1, -1, L, -1)
        return torch.cat([xi, xj], dim=1)

    def _global_cond(self, t, fm_fused, u_cond, density_hint=None):
        """Build global condition from time + FM global + UFold global + density."""
        te = self.time_mlp(t)
        fe = self.fm_global(fm_fused.mean(dim=1))  # (B, D)
        ue = self.u_global(u_cond.mean(dim=(-1, -2)))

        # v5: density conditioning
        if density_hint is not None:
            de = self.density_emb(density_hint)  # (B, D)
        else:
            # Use zeros when no density hint (training mode uses GT density)
            de = torch.zeros_like(te)

        return self.cond_fuse(torch.cat([te, fe, ue, de], dim=-1))

    def _build_features(self, x_t, fm_fused, fm_attn, seq_oh, u_cond):
        """Build input feature tensor (B, in_ch, L, L)."""
        x_long = x_t.long().squeeze(1)
        x_emb = self.x_t_embedding(x_long).permute(0, 3, 1, 2)  # (B, 8, L, L)

        # Multi-layer FM → 2D projection (wider: 64→32→outer→64ch)
        fm_proj = self.fm_emb_proj(fm_fused).permute(0, 2, 1)  # (B, 32, L)
        fm_2d = self._outer_concat(fm_proj)  # (B, 64, L, L)

        # FM attention maps
        fm_attn_2 = self.fm_attn_proj(fm_attn)
        fm_attn_2 = 0.5 * (fm_attn_2 + fm_attn_2.transpose(-2, -1))

        # Sequence 2D
        seq_2d = self._outer_concat(seq_oh.permute(0, 2, 1))

        f = torch.cat([x_emb, fm_2d, fm_attn_2, seq_2d, u_cond], dim=1)
        return 0.5 * (f + f.transpose(-2, -1))

    def forward(self, x_t, t, *, fm_multi: list, fm_attn, seq_oh, u_cond,
                contact_masks=None, density_hint=None, return_density: bool = False):
        B, _, L, _ = x_t.shape

        # Multi-layer FM fusion (v5: 64D output)
        fm_fused = self.fm_multi_fusion(fm_multi)  # (B, L, 64)

        # Build input features
        f = self._build_features(x_t, fm_fused, fm_attn, seq_oh, u_cond)
        tokens = self.patch_embed(f)

        # UFold spatial injection
        u_tok = self.u_patch_embed(u_cond).permute(0, 2, 3, 1)
        tokens = tokens + u_tok

        # Global condition (v5: includes density)
        cond = self._global_cond(t, fm_fused, u_cond, density_hint)

        # Transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens, cond, u_cond)

        # Final projection
        sh, sc = self.final_adaLN(cond).chunk(2, dim=-1)

        def expand(x):
            return x.view(x.shape[0], 1, 1, x.shape[-1])

        tokens = self.final_norm(tokens) * (1 + expand(sc)) + expand(sh)
        logit = self.unpatch(tokens)

        # v5 NEW: output refinement at full resolution
        logit = self.refine(logit)

        # Symmetrize
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
    m = DASEDiT_v5(hidden_dim=64, num_heads=2, dim_head=16, num_layers=9,
                   patch_size=4, cond_dim=8, max_len=64, fm_multi_out_dim=64,
                   dilation_pattern=[1, 1, 1, 2, 2, 2, 4, 4, 4],
                   tri_start_layer=6, tri_dim=32).eval()
    x_t = (torch.rand(B, 1, L, L) > 0.99).float()
    x_t = torch.maximum(x_t, x_t.transpose(-2, -1))
    t = torch.rand(B)
    fm_multi = [torch.randn(B, L, 640) for _ in range(4)]
    fm_attn = torch.randn(B, 240, L, L)
    seq_oh = torch.randn(B, L, 4)
    u_cond = torch.randn(B, 8, L, L)
    cm = torch.ones(B, 1, L, L)
    density = torch.rand(B, 1) * 0.5

    logit, dp = m(x_t, t, fm_multi=fm_multi, fm_attn=fm_attn,
                  seq_oh=seq_oh, u_cond=u_cond, contact_masks=cm,
                  density_hint=density, return_density=True)
    print(f'logit: {logit.shape}')  # (2, 1, 32, 32)
    print(f'density_pred: {dp.shape}')  # (2, 1)
    total_params = sum(p.numel() for p in m.parameters())
    print(f'total params: {total_params:,}')
