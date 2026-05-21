# -*- coding: utf-8 -*-
"""
SymFold v3 主模型: RNA-FM (frozen) + UFold (finetune) + DA-SE-DiT + Physics-Aware DFM.

改进 vs v1:
1. DA-SE-DiT: Dilated Axial attention (1,2,4) + RoPE + FiLM + QK-Norm
2. Physics-aware training loss: stacking + non-crossing penalty
3. Cosine sampling schedule
4. 更宽 (256) + 更深 (9层) backbone
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# symfold/ 作为项目根
SYMFOLD_SRC_V3 = os.path.dirname(os.path.abspath(__file__))
SYMFOLD_SRC = os.path.dirname(SYMFOLD_SRC_V3)
SYMFOLD_ROOT = os.path.dirname(SYMFOLD_SRC)
for p in (SYMFOLD_ROOT, SYMFOLD_SRC,
          os.path.join(SYMFOLD_SRC, 'models', 'condition', 'fm_conditioner')):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.condition.u_conditioner import Unet_conditioner, CH_FOLD
from models.condition.fm_conditioner.pretrained import load_model_and_alphabet_local

from .da_se_dit import DASEDiT
from .discrete_flow import (
    BernoulliFlowLoss_v3, sample_x_t_given_x_1, symmetrize_binary,
    sample_symfold_v3, project_to_valid_contact_map,
)
from src.physics_energy import PhysicsGuidance
from src.adversarial import FamilyClassifier


class SymFoldModel_v3(nn.Module):
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
                 pos_weight_scale: float = 1.0,
                 u_ckpt: str = 'ufold_train_alldata.pt',
                 num_families: int = 0,
                 dilation_pattern: list = None,
                 stack_weight: float = 0.05,
                 nc_weight: float = 0.02):
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

        # 3) DA-SE-DiT backbone
        self.backbone = DASEDiT(
            hidden_dim=hidden_dim, num_heads=num_heads, dim_head=dim_head,
            num_layers=num_layers, patch_size=patch_size, cond_dim=cond_dim,
            max_len=max_len, dropout=dp_rate,
            dilation_pattern=dilation_pattern,
        )

        # 4) Physics-Aware Loss
        self.flow_loss = BernoulliFlowLoss_v3(
            rho_0=rho_0, time_weight=True, pos_weight_scale=pos_weight_scale,
            stack_weight=stack_weight, nc_weight=nc_weight)

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
        self.fm_conditioner.eval()
        with torch.amp.autocast('cuda', enabled=False):
            out = self.fm_conditioner(tokens, need_head_weights=False,
                                      repr_layers=[12], return_contacts=True)
        fm_emb = out['representations'][12][:, 1:-1, :]
        B, L_seq, D = fm_emb.shape
        if L_seq < set_max_len:
            pad = torch.zeros(B, set_max_len - L_seq, D,
                              device=fm_emb.device, dtype=fm_emb.dtype)
            fm_emb = torch.cat([fm_emb, pad], dim=1)
        attn = out['attentions'][:, :, :, 1:-1, 1:-1]
        B, layers, heads, L_a, _ = attn.shape
        attn = attn.reshape(B, layers * heads, L_a, L_a)
        if L_a < set_max_len:
            pad = torch.zeros(B, layers * heads, set_max_len, set_max_len,
                              device=attn.device, dtype=attn.dtype)
            pad[:, :, :L_a, :L_a] = attn
            attn = pad
        return fm_emb, attn

    def get_ufold(self, data_fcn_2):
        with torch.amp.autocast('cuda', enabled=False):
            return self.u_conditioner(data_fcn_2.float())

    # ----- Training forward -----

    def forward(self, contact, data_fcn_2, tokens, contact_masks, set_max_len,
                seq_oh, family_label=None, adv_lambda: float = 0.0):
        B = contact.shape[0]
        device = contact.device

        fm_emb, fm_attn = self.get_fm(tokens, set_max_len)
        u_cond = self.get_ufold(data_fcn_2)

        t = torch.rand(B, device=device)
        x_1 = symmetrize_binary(contact)
        x_t = sample_x_t_given_x_1(x_1, t, rho_0=self.rho_0)
        x_t = symmetrize_binary(x_t) * contact_masks

        logit = self.backbone(x_t, t, fm_emb=fm_emb, fm_attn=fm_attn,
                              seq_oh=seq_oh, u_cond=u_cond,
                              contact_masks=contact_masks)

        total_loss, loss_dict = self.flow_loss(logit, x_1, t, contact_masks)

        if (self.family_head is not None and family_label is not None
                and adv_lambda > 0):
            pooled = self.fam_proj(fm_emb.mean(dim=1))
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

        fm_emb, fm_attn = self.get_fm(tokens, set_max_len)
        u_cond = self.get_ufold(data_fcn_2)

        physics_fn = None
        if physics_beta > 0:
            physics_fn = PhysicsGuidance(seq_oh, lambda_pk=physics_lambda_pk,
                                          alpha_stack=physics_alpha_stack)

        def network_fn(x_t, t):
            return self.backbone(x_t, t, fm_emb=fm_emb, fm_attn=fm_attn,
                                  seq_oh=seq_oh, u_cond=u_cond,
                                  contact_masks=contact_masks)

        if num_samples_per_input == 1:
            return sample_symfold_v3(
                network_fn, set_max_len, contact_masks,
                num_samples=B, num_steps=num_steps, rho_0=self.rho_0,
                physics_guidance_fn=physics_fn, energy_beta=physics_beta,
                project_final=True)

        # Multi-seed voting
        all_preds, all_probs = [], []
        seeds = seeds or list(range(num_samples_per_input))
        for s in seeds:
            torch.manual_seed(s)
            x, p = sample_symfold_v3(
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
