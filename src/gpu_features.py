# -*- coding: utf-8 -*-
"""
GPU 加速的 RNA 二级结构特征提取

把 RNADiffFold 中最慢的两个 CPU 操作移到 GPU 上:
1. creatmat: 配对概率矩阵 (原本 O(L²·30) Python 三层循环 → 向量化矩阵运算)
2. _get_data_fcn 中的 16 通道碱基对外积 → 简单的 einsum

这样 DataLoader 的 worker 只需要从 cPickle 读出 raw 数据 + 生成 contact map，
所有 17 通道 FCN 特征在 GPU 上现场算，CPU 不再是瓶颈。
"""

import torch
import numpy as np
from itertools import product

# 4×4 = 16 个通道排列
PERM_16 = list(product(range(4), range(4)))


def creatmat_gpu(seq_onehot, max_add=30):
    """
    GPU 向量化版本的 creatmat (严格保留原算法语义)
    
    原始算法:
      coef[i,j] = 0
      # 前向延伸: pair_score[i-add, j+add] for add=0,1,2,...
      # 只要遇到 score==0 就 break
      for add in range(max_add):
          if i-add < 0 or j+add >= L: break
          score = paired(seq[i-add], seq[j+add])
          if score == 0: break
          coef += score * gauss(add)
      
      # 反向延伸 (仅当前向有累加时才执行): pair_score[i+add, j-add] for add=1,2,...
      if coef > 0:
          for add in range(1, max_add):
              if i+add >= L or j-add < 0: break
              score = paired(seq[i+add], seq[j-add])
              if score == 0: break
              coef += score * gauss(add)
    
    向量化思路:
      1. 计算所有位置的碱基对得分 pair_score (B, L, L)
      2. 对于每个 (i, j)，沿 (-1, +1) 方向取 add=0..max_add-1 个值 → (B, L, L, max_add)
      3. 用 cumulative-or-product 模拟 "break on zero": 一旦某 add 为 0，后续都置为 0
         具体做法: mask[k] = (score[0]>0) & (score[1]>0) & ... & (score[k]>0)
         即 mask = cumprod(score > 0)
      4. coef = Σ score * gauss * mask
      5. 类似处理反向，最后乘上 (forward_coef > 0) 的 gate
    
    Args:
        seq_onehot: (B, L, 4) RNA one-hot
        max_add: 最大延伸距离
    Returns:
        mat: (B, L, L) 配对得分矩阵
    """
    B, L, _ = seq_onehot.shape
    device = seq_onehot.device
    dtype = seq_onehot.dtype
    
    # 构造碱基对得分查找矩阵 (4, 4)
    # 顺序: A=0, U=1, C=2, G=3
    score_table = torch.zeros((4, 4), device=device, dtype=dtype)
    score_table[0, 1] = 2.0   # A-U
    score_table[1, 0] = 2.0   # U-A
    score_table[3, 2] = 3.0   # G-C
    score_table[2, 3] = 3.0   # C-G
    score_table[3, 1] = 0.8   # G-U
    score_table[1, 3] = 0.8   # U-G
    
    s = torch.matmul(seq_onehot, score_table)  # (B, L, 4)
    pair_score = torch.matmul(s, seq_onehot.transpose(-1, -2))  # (B, L, L)
    
    # 高斯权重
    adds = torch.arange(max_add, device=device, dtype=dtype)
    gauss = torch.exp(-0.5 * adds * adds)  # (max_add,)
    
    # padding pair_score 用于偏移取值
    pad = max_add
    pair_padded = torch.nn.functional.pad(pair_score, (pad, pad, pad, pad))  # (B, L+2pad, L+2pad)
    
    # ===== 前向延伸: 取 (i-add, j+add) =====
    # 收集所有 add 的 score, shape: (B, L, L, max_add)
    fwd_scores = torch.empty((B, L, L, max_add), device=device, dtype=dtype)
    for add in range(max_add):
        si = pad - add
        sj = pad + add
        fwd_scores[:, :, :, add] = pair_padded[:, si:si+L, sj:sj+L]
    
    # 模拟 "break on zero": cumprod of (score > 0)
    nonzero_mask = (fwd_scores > 0).to(dtype)
    cum_mask = torch.cumprod(nonzero_mask, dim=-1)
    
    # forward coefficient
    fwd_coef = (fwd_scores * gauss.view(1, 1, 1, -1) * cum_mask).sum(dim=-1)
    
    # ===== 反向延伸: 取 (i+add, j-add), add=1..max_add-1 =====
    # 仅当 fwd_coef > 0 时才累加
    bwd_max = max_add - 1
    if bwd_max > 0:
        bwd_scores = torch.empty((B, L, L, bwd_max), device=device, dtype=dtype)
        for k, add in enumerate(range(1, max_add)):
            si = pad + add
            sj = pad - add
            bwd_scores[:, :, :, k] = pair_padded[:, si:si+L, sj:sj+L]
        
        bwd_nonzero = (bwd_scores > 0).to(dtype)
        bwd_cum = torch.cumprod(bwd_nonzero, dim=-1)
        bwd_gauss = gauss[1:].view(1, 1, 1, -1)
        bwd_coef = (bwd_scores * bwd_gauss * bwd_cum).sum(dim=-1)
        
        # gate by forward
        fwd_gate = (fwd_coef > 0).to(dtype)
        bwd_coef = bwd_coef * fwd_gate
        
        coef = fwd_coef + bwd_coef
    else:
        coef = fwd_coef
    
    return coef


def get_data_fcn_gpu(seq_onehot, set_max_len):
    """
    GPU 上生成 17 通道 FCN 特征
    
    Args:
        seq_onehot: (B, set_max_len, 4) padding 后的 one-hot 序列
        set_max_len: padding 长度
    Returns:
        data_fcn_2: (B, 17, set_max_len, set_max_len)
    """
    B, L, _ = seq_onehot.shape
    device = seq_onehot.device
    
    # 16 通道: 碱基对外积
    # outer[b, i, j, l1, l2] = seq[b, l1, i] * seq[b, l2, j]
    seq_t = seq_onehot.transpose(1, 2)  # (B, 4, L)
    outer = seq_t.unsqueeze(2).unsqueeze(4) * seq_t.unsqueeze(1).unsqueeze(3)  # (B,4,4,L,L)
    # reshape 到 (B, 16, L, L)
    data_fcn_16 = outer.reshape(B, 16, L, L)
    
    # 第 17 通道: creatmat
    data_fcn_1 = creatmat_gpu(seq_onehot).unsqueeze(1)  # (B, 1, L, L)
    
    data_fcn_2 = torch.cat([data_fcn_16, data_fcn_1], dim=1)
    return data_fcn_2
