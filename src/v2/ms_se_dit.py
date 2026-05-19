# -*- coding: utf-8 -*-
"""
Multi-Scale Symmetry-Equivariant Axial DiT (MSEDiT) — v2 Backbone

改进点 vs v1 SEDiT:
1. U-Net 式结构: Encoder (L/4) → Downsample → Middle (L/8) → Upsample → Decoder (L/4)
2. Skip connection: encoder output 直接 concat/add 到 decoder
3. Local attention bias: 前 2 层对短程 pair 加 learnable bias
4. 更深: 3+2+3 = 8 blocks (vs v1 的 6 blocks)
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
    """Row + Col attention with shared QKV weights (symmetric equivariant)."""

    def __init__(self, dim, num_heads=4, dim_head=32, dropout=0.0,
                 local_bias: bool = False, local_window: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        inner = num_heads * dim_head
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        self.local_bias = local_bias
        if local_bias:
            # learnable bias for positions within local_window
            self.local_bias_param = nn.Parameter(torch.zeros(num_heads, local_window * 2 + 1))
            self.local_window = local_window

    def _get_local_bias(self, n, device):
        """Generate relative position bias for local attention."""
        if not self.local_bias:
            return 0.0
        w = self.local_window
        pos = torch.arange(n, device=device)
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)  # (n, n)
        rel_clamp = rel.clamp(-w, w) + w  # shift to [0, 2w]
        bias = self.local_bias_param[:, rel_clamp]  # (heads, n, n)
        return bias.unsqueeze(0)  # (1, heads, n, n)

    def _attn(self, tokens):
        B, N, D = tokens.shape
        qkv = self.to_qkv(tokens).chunk(3, dim=-1)
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        a = (q @ k.transpose(-2, -1)) * self.scale
        # add local bias
        if self.local_bias:
            a = a + self._get_local_bias(N, tokens.device)
        a = F.softmax(a, dim=-1)
        out = a @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def forward(self, tokens):
        B, H, W, D = tokens.shape
        # row attention
        row = rearrange(tokens, 'b h w d -> (b h) w d')
        tokens = tokens + rearrange(self._attn(row), '(b h) w d -> b h w d', b=B)
        # col attention
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


class MSEDiTBlock(nn.Module):
    """Multi-Scale SE-DiT Block with optional local attention bias."""

    def __init__(self, dim, num_heads=4, dim_head=32, mlp_ratio=4, dropout=0.0,
                 local_bias: bool = False, local_window: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SharedAxialAttention(dim, num_heads, dim_head, dropout,
                                         local_bias=local_bias, local_window=local_window)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FFN(dim, mlp_ratio, dropout)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, cond):
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


class Downsample2x(nn.Module):
    """2x downsample on token grid via stride-2 conv."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, H, W, D) -> (B, D, H, W) -> conv -> (B, D, H/2, W/2) -> (B, H/2, W/2, D)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return self.norm(x)


class Upsample2x(nn.Module):
    """2x upsample on token grid via transposed conv."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return self.norm(x)


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


class MSEDiT(nn.Module):
    """
    Multi-Scale Symmetry-Equivariant Axial DiT (v2).

    Architecture:
        Encoder (3 blocks, resolution L/patch) 
        → Downsample 2× 
        → Middle (2 blocks, resolution L/patch/2)
        → Upsample 2× 
        → Skip add from encoder
        → Decoder (3 blocks, resolution L/patch)
        → Output

    Forward:
        x_t:    (B, 1, L, L), {0,1}
        t:      (B,), [0,1]
        fm_emb: (B, L, 640)
        fm_attn: (B, 240, L, L)
        seq_oh: (B, L, 4)
        u_cond: (B, cond_dim, L, L)
    Returns:
        logit: (B, 1, L, L)
    """

    def __init__(self,
                 hidden_dim: int = 192,
                 num_heads: int = 4,
                 dim_head: int = 48,
                 num_layers_enc: int = 3,
                 num_layers_mid: int = 2,
                 num_layers_dec: int = 3,
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
                 local_bias_layers: int = 2,
                 local_window: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Input branches (same as v1)
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
        max_side = max_len // patch_size + 1
        self.pos_embed = AxialPosEmbed(max_side, hidden_dim)

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

        # Encoder blocks (with local bias on first `local_bias_layers`)
        self.encoder_blocks = nn.ModuleList([
            MSEDiTBlock(hidden_dim, num_heads, dim_head, mlp_ratio, dropout,
                        local_bias=(i < local_bias_layers), local_window=local_window)
            for i in range(num_layers_enc)
        ])

        # Downsample
        self.downsample = Downsample2x(hidden_dim)

        # Middle blocks (at L/8 resolution, larger effective receptive field)
        self.middle_blocks = nn.ModuleList([
            MSEDiTBlock(hidden_dim, num_heads, dim_head, mlp_ratio, dropout)
            for _ in range(num_layers_mid)
        ])

        # Upsample
        self.upsample = Upsample2x(hidden_dim)

        # Skip projection (encoder_out + upsampled → fuse)
        self.skip_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            MSEDiTBlock(hidden_dim, num_heads, dim_head, mlp_ratio, dropout)
            for _ in range(num_layers_dec)
        ])

        # Final
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
        u_tok = self.u_patch_embed(u_cond).permute(0, 2, 3, 1)
        tokens = tokens + u_tok
        tokens = self.pos_embed(tokens)
        cond = self._global_cond(t, fm_emb, u_cond)

        # Encoder
        for blk in self.encoder_blocks:
            tokens = blk(tokens, cond)
        enc_out = tokens  # save for skip

        # Downsample → Middle
        tokens = self.downsample(tokens)
        for blk in self.middle_blocks:
            tokens = blk(tokens, cond)

        # Upsample → Skip → Decoder
        tokens = self.upsample(tokens)
        # handle size mismatch after upsample (in case of odd grid)
        _, eH, eW, _ = enc_out.shape
        _, tH, tW, _ = tokens.shape
        if tH != eH or tW != eW:
            tokens = tokens[:, :eH, :eW, :]
        tokens = self.skip_proj(torch.cat([tokens, enc_out], dim=-1))

        for blk in self.decoder_blocks:
            tokens = blk(tokens, cond)

        # Final output
        sh, sc = self.final_adaLN(cond).chunk(2, dim=-1)
        def expand(x):
            return x.view(x.shape[0], 1, 1, x.shape[-1])
        tokens = self.final_norm(tokens) * (1 + expand(sc)) + expand(sh)
        logit = self.unpatch(tokens)

        # Symmetrize + mask
        logit = 0.5 * (logit + logit.transpose(-2, -1))
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
    m = MSEDiT(hidden_dim=64, num_heads=2, dim_head=16,
               num_layers_enc=2, num_layers_mid=1, num_layers_dec=2,
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
    print(f'MSEDiT params: {n:,}')
    print('ms_se_dit.py self-test passed')
