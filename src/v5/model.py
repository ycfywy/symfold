# -*- coding: utf-8 -*-
"""
SymFold v5 主模型: Wider FM + Density Conditioning + Output Refinement.

改进 vs v4:
1. FM Fusion 输出维度: 16 → 64 (减少 RNA-FM 信息压缩损失)
2. 密度条件注入: 训练时用 GT density，推理时用 DensityHead 预测
3. 输出精修 Conv: 2 层 Conv 在全分辨率修正 patch 边界
4. 降低 pos_weight_min: 50 → 20 (进一步抑制低密度过预测)
5. 密度引导采样: 推理时根据预测密度动态调整翻转阈值
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# symfold/ 作为项目根
SYMFOLD_SRC_V5 = os.path.dirname(os.path.abspath(__file__))
SYMFOLD_SRC = os.path.dirname(SYMFOLD_SRC_V5)
SYMFOLD_ROOT = os.path.dirname(SYMFOLD_SRC)
for p in (SYMFOLD_ROOT, SYMFOLD_SRC,
          os.path.join(SYMFOLD_SRC, 'models', 'condition', 'fm_conditioner')):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.condition.u_conditioner import Unet_conditioner, CH_FOLD
from models.condition.fm_conditioner.pretrained import load_model_and_alphabet_local

from .da_se_dit import DASEDiT_v5
from src.v4.discrete_flow import (
    BernoulliFlowLoss_v4, sample_x_t_given_x_1, symmetrize_binary,
    symmetrize_logit, compute_ctmc_rates,
    project_to_valid_contact_map,
)
from src.physics_energy import PhysicsGuidance

# RNA-FM layers to extract
FM_EXTRACT_LAYERS = [3, 6, 9, 12]


class SymFoldModel_v5(nn.Module):
    def __init__(self,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 dim_head: int = 64,
                 num_layers: int = 9,
                 patch_size: int = 4,
                 cond_dim: int = 8,
                 max_len: int = 640,
                 dp_rate: float = 0.1,
                 rho_0: float = 0.005,
                 pos_weight_base: float = 199.0,
                 pos_weight_min: float = 20.0,
                 focal_gamma: float = 1.5,
                 u_ckpt: str = 'ufold_train_alldata.pt',
                 num_families: int = 0,
                 dilation_pattern: list = None,
                 stack_weight: float = 0.05,
                 nc_weight: float = 0.02,
                 density_weight: float = 0.2,
                 tri_start_layer: int = 6,
                 tri_dim: int = 64,
                 fm_multi_out_dim: int = 64):
        super().__init__()
        self.rho_0 = rho_0
        self.num_families = num_families

        if dilation_pattern is None:
            dilation_pattern = [1, 1, 1, 2, 2, 2, 4, 4, 4]

        # 1) RNA-FM (frozen)
        cond_ckpt_path = os.path.join(SYMFOLD_ROOT, 'ckpt', 'cond_ckpt')
        fm_model, self.alphabet = load_model_and_alphabet_local(
            os.path.join(cond_ckpt_path, 'RNA-FM_pretrained.pth'))
        self.fm_conditioner = fm_model
        self.fm_conditioner.eval()
        for p_ in self.fm_conditioner.parameters():
            p_.requires_grad = False

        # 2) UFold (finetune)
        self.u_conditioner = Unet_conditioner(img_ch=17, output_ch=1)
        self.u_conditioner.load_state_dict(
            torch.load(os.path.join(cond_ckpt_path, u_ckpt),
                       map_location='cpu', weights_only=False))
        cond_out = nn.Conv2d(int(32 * CH_FOLD), cond_dim, 1, 1, 0)
        self.u_conditioner.Conv_1x1 = cond_out
        self.u_conditioner.requires_grad_(True)

        # 3) DA-SE-DiT v5 backbone
        self.backbone = DASEDiT_v5(
            hidden_dim=hidden_dim, num_heads=num_heads, dim_head=dim_head,
            num_layers=num_layers, patch_size=patch_size, cond_dim=cond_dim,
            max_len=max_len, dropout=dp_rate, fm_multi_out_dim=fm_multi_out_dim,
            dilation_pattern=dilation_pattern,
            tri_start_layer=tri_start_layer, tri_dim=tri_dim,
        )

        # 4) Adaptive Density-Aware Flow Loss (v5: lower min, higher focal)
        self.flow_loss = BernoulliFlowLoss_v4(
            rho_0=rho_0, time_weight=True,
            pos_weight_base=pos_weight_base, pos_weight_min=pos_weight_min,
            focal_gamma=focal_gamma,
            stack_weight=stack_weight, nc_weight=nc_weight,
            density_weight=density_weight)

    def get_alphabet(self):
        return self.alphabet

    # ----- Condition extraction -----

    @torch.no_grad()
    def get_fm(self, tokens, set_max_len):
        """Extract multi-layer RNA-FM features."""
        self.fm_conditioner.eval()
        with torch.amp.autocast('cuda', enabled=False):
            out = self.fm_conditioner(tokens, need_head_weights=False,
                                      repr_layers=FM_EXTRACT_LAYERS,
                                      return_contacts=True)

        fm_multi = []
        for layer_idx in FM_EXTRACT_LAYERS:
            emb = out['representations'][layer_idx][:, 1:-1, :]
            B, L_seq, D = emb.shape
            if L_seq < set_max_len:
                pad = torch.zeros(B, set_max_len - L_seq, D,
                                  device=emb.device, dtype=emb.dtype)
                emb = torch.cat([emb, pad], dim=1)
            fm_multi.append(emb)

        attn = out['attentions'][:, :, :, 1:-1, 1:-1]
        B, layers, heads, L_a, _ = attn.shape
        attn = attn.reshape(B, layers * heads, L_a, L_a)
        if L_a < set_max_len:
            pad = torch.zeros(B, layers * heads, set_max_len, set_max_len,
                              device=attn.device, dtype=attn.dtype)
            pad[:, :, :L_a, :L_a] = attn
            attn = pad
        return fm_multi, attn

    def get_ufold(self, data_fcn_2):
        with torch.amp.autocast('cuda', enabled=False):
            return self.u_conditioner(data_fcn_2.float())

    # ----- Training forward -----

    def forward(self, contact, data_fcn_2, tokens, contact_masks, set_max_len,
                seq_oh, family_label=None, adv_lambda: float = 0.0):
        B = contact.shape[0]
        device = contact.device

        fm_multi, fm_attn = self.get_fm(tokens, set_max_len)
        u_cond = self.get_ufold(data_fcn_2)

        t = torch.rand(B, device=device)
        x_1 = symmetrize_binary(contact)
        x_t = sample_x_t_given_x_1(x_1, t, rho_0=self.rho_0)
        x_t = symmetrize_binary(x_t) * contact_masks

        # v5: compute GT density as condition for training
        with torch.no_grad():
            valid = contact_masks.squeeze(1)
            L_eff = valid[:, 0, :].sum(dim=-1)
            gt_pairs = (x_1.squeeze(1) * valid).sum(dim=(-1, -2)) / 2
            gt_density = (gt_pairs / L_eff.clamp(min=1)).unsqueeze(1)  # (B, 1)

        logit, density_pred = self.backbone(
            x_t, t, fm_multi=fm_multi, fm_attn=fm_attn,
            seq_oh=seq_oh, u_cond=u_cond,
            contact_masks=contact_masks,
            density_hint=gt_density,  # v5: inject GT density during training
            return_density=True)

        total_loss, loss_dict = self.flow_loss(
            logit, x_1, t, contact_masks, density_pred=density_pred)

        loss_dict['total'] = total_loss.detach()
        return total_loss, loss_dict

    # ----- Inference -----

    @torch.no_grad()
    def sample(self, *, data_fcn_2, tokens, contact_masks, set_max_len,
               seq_oh, num_steps: int = 20, num_samples_per_input: int = 1,
               physics_beta: float = 0.0, physics_lambda_pk: float = 0.0,
               physics_alpha_stack: float = 1.0,
               density_guided: bool = True):
        """
        v5 采样: 支持密度引导。
        
        1. 先做一次快速前向获取 density prediction
        2. 将 predicted density 作为条件注入后续采样
        3. (可选) 根据密度动态调整翻转阈值
        """
        import math
        device = contact_masks.device
        B_real = data_fcn_2.shape[0]
        B = B_real * num_samples_per_input
        L = set_max_len

        fm_multi, fm_attn = self.get_fm(tokens, set_max_len)
        u_cond = self.get_ufold(data_fcn_2)

        # Repeat for multi-sample
        if num_samples_per_input > 1:
            fm_multi = [x.repeat(num_samples_per_input, 1, 1) for x in fm_multi]
            fm_attn = fm_attn.repeat(num_samples_per_input, 1, 1, 1)
            u_cond = u_cond.repeat(num_samples_per_input, 1, 1, 1)
            contact_masks = contact_masks.repeat(num_samples_per_input, 1, 1, 1)
            seq_oh = seq_oh.repeat(num_samples_per_input, 1, 1)

        # v5: get density prediction first (use t=0.5 as reference point)
        x_init = (torch.rand(B, 1, L, L, device=device) < self.rho_0).float()
        x_init = symmetrize_binary(x_init) * contact_masks
        t_half = torch.full((B,), 0.5, device=device)
        _, density_pred = self.backbone(
            x_init, t_half, fm_multi=fm_multi, fm_attn=fm_attn,
            seq_oh=seq_oh, u_cond=u_cond,
            contact_masks=contact_masks, density_hint=None,
            return_density=True)
        # density_pred: (B, 1) in [0, 1]

        # Build network_fn with density conditioning
        def network_fn(x_t, t_tensor):
            return self.backbone(
                x_t, t_tensor, fm_multi=fm_multi, fm_attn=fm_attn,
                seq_oh=seq_oh, u_cond=u_cond,
                contact_masks=contact_masks,
                density_hint=density_pred if density_guided else None)

        # τ-leap sampling with cosine schedule
        x_t = x_init
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

            if physics_beta > 0.0:
                pg = PhysicsGuidance(seq_oh, alpha_stack=physics_alpha_stack,
                                     lambda_pk=physics_lambda_pk)
                grad = pg(logit, x_t)
                logit = logit - physics_beta * grad

            p_x1 = torch.sigmoid(logit)
            p_x1 = 0.5 * (p_x1 + p_x1.transpose(-2, -1))
            p_x1_last = p_x1

            rate_01, rate_10 = compute_ctmc_rates(x_t, p_x1, t_tensor, rho_0=self.rho_0)

            # v5: density-guided rate damping for low-density sequences
            if density_guided:
                # Low density → reduce rate_01 (less 0→1 flipping)
                # density_pred ∈ [0,1], typical ~0.25 for ppb=0.5
                # damping: scale rate_01 by min(1, 2*density_pred)
                damp = (2.0 * density_pred).clamp(max=1.0)  # (B, 1)
                damp = damp.view(B, 1, 1, 1)
                rate_01 = rate_01 * damp

            f01 = torch.clamp(rate_01 * dt, max=1.0)
            f10 = torch.clamp(rate_10 * dt, max=1.0)
            flip01 = (torch.rand_like(f01) < f01) & (x_t < 0.5)
            flip10 = (torch.rand_like(f10) < f10) & (x_t > 0.5)
            x_t = torch.where(flip01, torch.ones_like(x_t), x_t)
            x_t = torch.where(flip10, torch.zeros_like(x_t), x_t)
            x_t = symmetrize_binary(x_t) * contact_masks

            t_cumulative += dt

        # Project
        x_final = project_to_valid_contact_map(x_t, p_x1_last, contact_masks)

        # Average over multi-samples
        if num_samples_per_input > 1:
            x_final = x_final.view(num_samples_per_input, B_real, 1, L, L).mean(0)
            x_final = (x_final > 0.5).float()
            x_final = symmetrize_binary(x_final)
            p_x1_last = p_x1_last.view(num_samples_per_input, B_real, 1, L, L).mean(0)

        return x_final, p_x1_last
