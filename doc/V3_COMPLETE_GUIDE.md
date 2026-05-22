# V3 Complete Guide — DA-SE-DiT Architecture, Training & Evaluation

> SymFold v3: Dilated Axial SE-DiT + Physics-Aware Discrete Flow Matching

---

## 一、架构概览

```
RNA Sequence
    │
    ├─→ RNA-FM (frozen, 99.5M) → fm_emb (B, L, 640) + fm_attn (B, 240, L, L)
    │
    ├─→ UFold (finetune) → u_cond (B, 8, L, L)
    │
    └─→ Seq one-hot → seq_oh (B, L, 4)
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│            Feature Builder (48 channels)                  │
│  x_t_emb(8) + fm_2d(16) + fm_attn(8) + seq_2d(8) + u(8)│
│  → symmetrize → (B, 48, L, L)                           │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  PatchEmbed2D(48→256, patch=4)  → (B, L/4, L/4, 256)    │
│  + UFold patch embedding                                  │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│         9× DASEDiTBlock (Dilated Axial + FiLM)           │
│                                                          │
│  Block layout (per layer):                               │
│    AdaLN → DilatedAxialAttn(row+col) → FiLM → AdaLN → FFN │
│                                                          │
│  Dilation pattern: [1,1,1, 2,2,2, 4,4,4]                │
│  Global cond: time_emb + fm_global + ufold_global        │
│  Spatial cond: UFold FiLM at every layer                 │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Final: AdaLN → UnPatchify → symmetrize → mask → logit   │
│  logit: (B, 1, L, L) — per-element pairing score         │
└──────────────────────────────────────────────────────────┘
```

---

## 二、核心模块详解

### 2.1 Dilated Axial Attention (`DilatedAxialAttention`)

**文件**: `src/v3/da_se_dit.py`

核心思想：不降低分辨率就能看到更远的 token。

```python
# dilation=1: 标准 axial attention，每行所有位置互相关注
# dilation=2: 跳着看，关注 stride=2 的位置 (receptive field ×2)
# dilation=4: 关注 stride=4 的位置 (receptive field ×4)
```

实现方式：
1. 将 row tokens 按 dilation 分组：`(BH, W) → (BH*d, W//d)`
2. 在每组内做标准 attention
3. 还原回原始形状

**对称性保证**: Row attention 和 Col attention 使用相同的 QKV 投影（shared weights），确保 `f(M^T) = f(M)^T`。

### 2.2 Rotary Position Embedding (RoPE)

替代 v1 的 learnable position embedding：
- 更好地泛化到训练时未见过的序列长度
- 相对位置编码，天然支持变长输入
- 作用于 Q 和 K 向量的旋转

### 2.3 FiLM Spatial Conditioning

**文件**: `src/v3/da_se_dit.py` - `FiLM` class

UFold 提供逐像素级的条件信息（碱基对概率 prior），通过 FiLM 在每一层注入：

```python
# UFold (B, 8, L, L) → Conv(patch_size stride) → GELU → Conv(1x1) → scale, shift
# x = x * (1 + scale) + shift
```

vs v1 只用 UFold 的全局 pooling 作为 AdaLN 条件，丢失了空间细节。

### 2.4 AdaLN-Zero

每层都有 AdaLN-Zero 调制，条件来自 `time + fm_global + ufold_global` 的融合：

```python
cond = Linear(concat(time_emb, fm_emb.mean(), ufold.mean()))
shift, scale, gate = AdaLN(cond).chunk(6)  # 6 = 2 sublayers × 3 params
```

Gate 初始化为 0（Zero Init），确保初始化时每层是恒等变换。

### 2.5 QK-Norm

在 attention 计算前对 Q 和 K 做 RMSNorm，防止深层训练中 attention logits 爆炸。这对 9 层深网络尤为关键。

---

## 三、Discrete Flow Matching (v3 改进)

**文件**: `src/v3/discrete_flow.py`

### 3.1 前向加噪 (同 v1)

```python
p(x_t[i,j]=1 | x_1) = t * x_1[i,j] + (1-t) * ρ₀
```
- t=0: Bernoulli(ρ₀=0.005) 先验
- t=1: Ground truth contact map

### 3.2 训练 Loss: BCE + Physics

```python
total_loss = BCE_loss + λ_stack * stacking_loss + λ_nc * non_crossing_loss
```

| 分项 | 权重 | 作用 |
|------|:----:|------|
| BCE (pos-weighted) | 1.0 | 主损失，pos_weight≈199 |
| Stacking Loss | 0.05 | 鼓励连续配对 (stem 结构) |
| Non-Crossing Loss | 0.02 | 惩罚 row-sum > 1 (多配对) |

**Stacking Loss**: 如果 prob(i,j) 高，则 prob(i+1, j-1) 也应高 (stem 连续性)
**Non-Crossing Loss**: 每行概率和不应超过 1 (每个碱基最多一个 partner)

### 3.3 采样: Cosine-Schedule τ-leap

```python
# 步长: dt_k ∝ sin(π(k+0.5)/(2K))
# 前期大步长（粗粒度结构形成），后期小步长（精细调整）
```

### 3.4 投影: Strict Greedy Max-Matching

与 v1 完全相同的贪心算法：
1. 选 score 最高的 (i,j)
2. 标记 i 和 j 为已配对
3. 重复直到无有效候选

---

## 四、参数量统计

| 模块 | 参数量 | 状态 |
|------|-------:|:----:|
| RNA-FM Transformer (12 层) | 99.5M | 冻结 |
| UFold U-Net | 8.2M | 可训练 |
| DA-SE-DiT Backbone (9 层) | 13.2M | 可训练 |
| 条件投影 + 输出头 | 0.4M | 可训练 |
| **总计** | **121.3M** | — |
| **可训练** | **21.8M** | — |

---

## 五、训练配置与结果

### 5.1 训练配置

| 项目 | 配置 |
|------|------|
| 学习率 | 8e-5 → cosine decay |
| Warmup | 5 epochs |
| Epochs | 80 |
| Batch | L≤80→48, L≤160→24, L≤240→10, L≤320→6, L≤400→4 |
| 训练集 | 34,782 samples (RNAStrAlign + bpRNA + bpRNA-new) |
| 验证集 | bpRNA VL0 (1,299 samples) |
| 每 epoch | 2,812 batches, ~18 min |
| 总训练时长 | ~24.4 小时 |

### 5.2 训练曲线

| Epoch | Train Loss | Val F1 | Val Precision | Val Recall |
|:-----:|:----------:|:------:|:-------------:|:----------:|
| 1 | 0.044 | 0.432 | 0.340 | 0.648 |
| 13 | 0.012 | 0.500 | 0.412 | 0.693 |
| 33 | 0.007 | 0.542 | 0.455 | 0.721 |
| 55 | 0.005 | 0.575 | 0.497 | 0.730 |
| 73 | 0.004 | **0.603** | 0.527 | 0.745 |
| 79 | 0.004 | 0.598 | 0.522 | 0.743 |

**特点**: 全程稳定上升，无崩塌，未触发 early stop。

---

## 六、完整 Eval 结果 (best.pt = epoch 73)

### 6.1 v3 各数据集表现

| Dataset | N | Type | F1 | Precision | Recall | MCC | Time |
|---------|---:|:----:|:---:|:---------:|:------:|:---:|-----:|
| **RNAStrAlign** | 2,023 | ID | **0.939** | 0.920 | 0.961 | 0.940 | 6.6m |
| **ArchiveII** | 3,911 | OOD | **0.864** | 0.834 | 0.904 | 0.866 | 15.6m |
| **PDB_TS2** | 38 | OOD-hard | **0.807** | 0.873 | 0.759 | 0.810 | 0.04m |
| **PDB_TS1** | 60 | OOD-hard | **0.716** | 0.768 | 0.684 | 0.720 | 0.07m |
| **PDB_TS3** | 18 | OOD-hard | **0.666** | 0.738 | 0.611 | 0.669 | 0.03m |
| **bpRNA** | 1,304 | ID | **0.636** | 0.557 | 0.785 | 0.652 | 3.0m |
| **PDB_TS_hard** | 28 | OOD-hardest | **0.634** | 0.712 | 0.587 | 0.640 | 0.03m |

**Average F1: 0.752** | **Total eval time: ~26 min** (single sample, no multi-vote)

### 6.2 与 v1 和 RNADiffFold 对比

| Dataset | v3 F1 | v1 F1 | Δ(v3-v1) | RNADiffFold F1 | Δ(v3-RDF) |
|---------|:-----:|:-----:|:---------:|:--------------:|:---------:|
| RNAStrAlign | **0.939** | 0.921 | **+0.018** | 0.787 | **+0.152** |
| ArchiveII | **0.864** | 0.861 | **+0.003** | 0.740 | **+0.124** |
| PDB_TS2 | 0.807 | **0.832** | -0.025 | 0.733 | **+0.074** |
| PDB_TS1 | **0.716** | 0.675 | **+0.041** | 0.607 | **+0.109** |
| PDB_TS3 | **0.666** | 0.665 | **+0.001** | 0.635 | **+0.031** |
| bpRNA | 0.636 | **0.644** | -0.008 | 0.618 | **+0.018** |
| PDB_TS_hard | **0.634** | 0.596 | **+0.038** | 0.526 | **+0.108** |
| **Average** | **0.752** | 0.742* | **+0.010** | 0.657 | **+0.095** |

*v1 不含 bpRNA-new 的 8 数据集平均

### 6.3 分析

**v3 vs v1 优势**:
- RNAStrAlign: +1.8% (ID test 上从 0.921→0.939，接近 SOTA UFold 水平 0.96)
- PDB_TS1: +4.1% (OOD-hard 大幅提升)
- PDB_TS_hard: +3.8% (最难数据集大幅提升)
- 推理速度更快: 单 sample 无投票，total eval 26min vs RNADiffFold 119min

**v3 vs v1 劣势**:
- PDB_TS2: -2.5% (小样本数据集波动)
- bpRNA: -0.8% (可能因训练不够)

**v3 vs RNADiffFold 压倒性优势**:
- 全部 7 个数据集都超越 RNADiffFold
- 平均 F1: +9.5% (0.752 vs 0.657)
- 参数量: 21.8M vs 109M (可训练部分只有 1/5)
- 推理速度: ~4× faster (单 sample vs 10 sample 投票)

---

## 七、代码文件结构

```
src/v3/
├── __init__.py              # 包导出
├── model.py                 # SymFoldModel_v3: 整合 RNA-FM + UFold + DA-SE-DiT
├── da_se_dit.py             # DA-SE-DiT backbone (9 层 dilated axial + FiLM)
│   ├── AxialRoPE            #   2D 旋转位置编码
│   ├── SinusoidalTimeEmbedding  # 时间嵌入
│   ├── DilatedAxialAttention    # 核心: 带扩张率的行列注意力
│   ├── FiLM                 #   UFold 空间条件注入
│   ├── DASEDiTBlock         #   单个 Transformer 块
│   ├── PatchEmbed2D         #   L×L → (L/4)×(L/4) tokens
│   ├── UnPatchify2D         #   逆操作
│   └── DASEDiT              #   完整 backbone
└── discrete_flow.py         # v3 Discrete Flow Matching
    ├── sample_x_t_given_x_1 #   前向加噪
    ├── BernoulliFlowLoss_v3 #   Loss = BCE + Stacking + NonCrossing
    ├── StackingLoss         #   stem 连续性正则
    ├── NonCrossingLoss      #   row-sum 约束
    ├── project_to_valid_contact_map  # 贪心投影
    └── sample_symfold_v3    #   Cosine-schedule τ-leap 采样
```

---

## 八、关键设计决策与 v2 失败教训

| 决策 | v2 (失败) | v3 (成功) | 原因 |
|------|:---------:|:---------:|------|
| 架构 | U-Net (3+2+3) | Flat (9 层) | U-Net 下采样破坏对称等变性 |
| 投影 | Relaxed (top-5%) | Strict (greedy) | Relaxed 导致训练-推理 gap |
| 感受野 | 下采样扩大 | Dilated attention | 不降分辨率就能看更远 |
| Batch | 过大 (128) | 适中 (48) | 确保充分梯度步数 |
| UFold 注入 | 仅全局 pooling | FiLM 逐层空间注入 | 保留 UFold 空间细节 |
| 训练稳定性 | 无 QK-Norm | QK-Norm + Zero Init | 深层 attention 稳定 |
