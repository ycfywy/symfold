# -*- coding: utf-8 -*-
"""
SymFold v4 主模型: Multi-Layer RNA-FM + UFold + DA-SE-DiT-v4 + Adaptive DFM.

改进 vs v3:
1. Multi-Layer RNA-FM: 提取 layers [3,6,9,12] 的 embedding，learnable fusion
   - 浅层(3): 局部序列 motif (k-mer)
   - 中层(6,9): 中程结构倾向 (hairpin, internal loop)
   - 深层(12): 全局折叠语义
2. Triangle Multiplicative Update: 后3层加入三体约束
3. Adaptive Density-Aware Loss: 低密度 RNA 降低 pos_weight 防止过预测
4. Gated FFN (SwiGLU): 更好的参数效率
5. Focal modulation: 对高置信假阳性加大惩罚
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# symfold/ 作为项目根
SYMFOLD_SRC_V4 = os.path.dirname(os.path.abspath(__file__))
SYMFOLD_SRC = os.path.dirname(SYMFOLD_SRC_V4)
SYMFOLD_ROOT = os.path.dirname(SYMFOLD_SRC)
for p in (SYMFOLD_ROOT, SYMFOLD_SRC,
          os.path.join(SYMFOLD_SRC, 'models', 'condition', 'fm_conditioner')):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.condition.u_conditioner import Unet_conditioner, CH_FOLD
from models.condition.fm_conditioner.pretrained import load_model_and_alphabet_local

from .da_se_dit import DASEDiT_v4
from .discrete_flow import (
    BernoulliFlowLoss_v4, sample_x_t_given_x_1, symmetrize_binary,
    sample_symfold_v4, project_to_valid_contact_map,
)
from src.physics_energy import PhysicsGuidance
from src.adversarial import FamilyClassifier


# RNA-FM layers to extract (浅/中/深)
FM_EXTRACT_LAYERS = [3, 6, 9, 12]


class SymFoldModel_v4(nn.Module):
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
                 pos_weight_min: float = 50.0,
                 focal_gamma: float = 1.0,
                 u_ckpt: str = 'ufold_train_alldata.pt',
                 num_families: int = 0,
                 dilation_pattern: list = None,
                 stack_weight: float = 0.05,
                 nc_weight: float = 0.02,
                 density_weight: float = 0.1,
                 tri_start_layer: int = 6,
                 tri_dim: int = 64,
                 fm_multi_out_dim: int = 16):
        super().__init__()
        self.rho_0 = rho_0
        self.num_families = num_families

        if dilation_pattern is None:
            dilation_pattern = [1, 1, 1, 2, 2, 2, 4, 4, 4]

        # 1) RNA-FM (frozen) — 提取多层
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

        # 3) DA-SE-DiT v4 backbone
        self.backbone = DASEDiT_v4(
            hidden_dim=hidden_dim, num_heads=num_heads, dim_head=dim_head,
            num_layers=num_layers, patch_size=patch_size, cond_dim=cond_dim,
            max_len=max_len, dropout=dp_rate, fm_multi_out_dim=fm_multi_out_dim,
            dilation_pattern=dilation_pattern,
            tri_start_layer=tri_start_layer, tri_dim=tri_dim,
        )

        # 4) Adaptive Density-Aware Flow Loss
        self.flow_loss = BernoulliFlowLoss_v4(
            rho_0=rho_0, time_weight=True,
            pos_weight_base=pos_weight_base, pos_weight_min=pos_weight_min,
            focal_gamma=focal_gamma,
            stack_weight=stack_weight, nc_weight=nc_weight,
            density_weight=density_weight)

        # 5) Family classifier (optional)
        self.fam_proj = None
        self.family_head = None
        if num_families > 0:
            self.fam_proj = nn.Linear(640, hidden_dim)
            self.family_head = FamilyClassifier(hidden_dim, num_families,
                                                 hidden=hidden_dim, dropout=dp_rate)

    def get_alphabet(self):
        return self.alphabet

    # ----- Condition extraction -----

    @torch.no_grad()
    def get_fm(self, tokens, set_max_len):
        """
        Extract multi-layer RNA-FM features.
        Returns:
            fm_multi: list of 4 tensors, each (B, set_max_len, 640)
            fm_attn: (B, 240, set_max_len, set_max_len)
        """
        self.fm_conditioner.eval()
        with torch.amp.autocast('cuda', enabled=False):
            out = self.fm_conditioner(tokens, need_head_weights=False,
                                      repr_layers=FM_EXTRACT_LAYERS,
                                      return_contacts=True)

        # Extract multi-layer embeddings
        fm_multi = []
        for layer_idx in FM_EXTRACT_LAYERS:
            emb = out['representations'][layer_idx][:, 1:-1, :]  # (B, L, 640)
            B, L_seq, D = emb.shape
            if L_seq < set_max_len:
                pad = torch.zeros(B, set_max_len - L_seq, D,
                                  device=emb.device, dtype=emb.dtype)
                emb = torch.cat([emb, pad], dim=1)
            fm_multi.append(emb)

        # Attention maps (same as v3: all layers × all heads)
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

        logit, density_pred = self.backbone(
            x_t, t, fm_multi=fm_multi, fm_attn=fm_attn,
            seq_oh=seq_oh, u_cond=u_cond,
            contact_masks=contact_masks, return_density=True)

        total_loss, loss_dict = self.flow_loss(
            logit, x_1, t, contact_masks, density_pred=density_pred)

        if (self.family_head is not None and family_label is not None
                and adv_lambda > 0):
            # Use layer 12 for family classification
            pooled = self.fam_proj(fm_multi[-1].mean(dim=1))
            fam_logit = self.family_head(pooled, adv_lambda)
            fam_loss = F.cross_entropy(fam_logit, family_label)
            total_loss = total_loss + adv_lambda * fam_loss
            loss_dict['fam_ce'] = fam_loss.detach()

        loss_dict['total'] = total_loss.detach()
        return total_loss, loss_dict

    # ----- Inference sample -----

    @torch.no_grad()
    def sample(self, data_fcn_2, tokens, contact_masks, set_max_len, seq_oh,
               num_steps: int = 20, num_samples_per_input: int = 1,
               physics_lambda_pk: float = 0.0,
               physics_alpha_stack: float = 1.0,
               physics_beta: float = 0.0,
               seeds: list = None):
        device = data_fcn_2.device
        B = data_fcn_2.shape[0]

        fm_multi, fm_attn = self.get_fm(tokens, set_max_len)
        u_cond = self.get_ufold(data_fcn_2)

        physics_fn = None
        if physics_beta > 0:
            physics_fn = PhysicsGuidance(seq_oh, lambda_pk=physics_lambda_pk,
                                          alpha_stack=physics_alpha_stack)

        def network_fn(x_t, t):
            return self.backbone(x_t, t, fm_multi=fm_multi, fm_attn=fm_attn,
                                  seq_oh=seq_oh, u_cond=u_cond,
                                  contact_masks=contact_masks)

        if num_samples_per_input == 1:
            return sample_symfold_v4(
                network_fn, set_max_len, contact_masks,
                num_samples=B, num_steps=num_steps, rho_0=self.rho_0,
                physics_guidance_fn=physics_fn, energy_beta=physics_beta,
                project_final=True)

        # Multi-seed voting
        all_preds, all_probs = [], []
        seeds = seeds or list(range(num_samples_per_input))
        for s in seeds:
            torch.manual_seed(s)
            x, p = sample_symfold_v4(
                network_fn, set_max_len, contact_masks,
                num_samples=B, num_steps=num_steps, rho_0=self.rho_0,
                physics_guidance_fn=physics_fn, energy_beta=physics_beta,
                project_final=True)
            all_preds.append(x)
            all_probs.append(p)
        stacked = torch.stack(all_preds, dim=0)
        voted = (stacked.mean(dim=0) > 0.5).float()
        mean_prob = torch.stack(all_probs, dim=0).mean(dim=0)
        return voted, mean_prob
