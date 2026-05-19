# SymFold v2 — Multi-Scale Axial DiT + Relaxed Projection

> 2026-05-18 改进版本

## 改进动机 (来自 v1 失败分析)

| 问题 | 根因 | v2 解决方案 |
|------|------|------------|
| 长序列 Recall 低 | 单尺度 Axial Attn 信息传播不足 | **Multi-Scale U-DiT**: 加 1 级下采样层 (2×) 在中间，扩大感受野 |
| Pseudoknot/多stem差 | 贪心 max-matching 每行≤1 太严格 | **Relaxed Projection**: 允许 pseudoknot (每行≤2), threshold 自适应 |
| UFold 空间信息丢失 | 仅全局池化 → AdaLN | **Multi-Scale UFold Injection**: 在下采样层注入 UFold 中间特征 |
| 采样效率/质量 | 固定步长 τ-leap | **Adaptive Schedule**: cosine schedule dt, 前大后小 |

## 架构概述

```
SymFoldModel_v2 (~16M trainable params)
├── fm_conditioner    : RNA-FM (frozen, 同 v1)
├── u_conditioner     : UFold U-Net (finetune, 同 v1, 但额外导出中间层特征)
├── backbone          : MSEDiT (Multi-Scale Symmetry-Equivariant Axial DiT)
│   ├── PatchEmbed2D (patch=4, in=48ch → hidden=192)
│   ├── AxialPosEmbed
│   ├── Encoder Blocks × 3 (SEDiTBlock, resolution L/4)
│   ├── Downsample2x (conv stride=2, resolution L/8)
│   ├── Middle Blocks × 2 (SEDiTBlock, resolution L/8, 更大感受野)
│   ├── Upsample2x (transposed conv, back to L/4)
│   ├── Skip Connection (encoder → decoder)
│   ├── Decoder Blocks × 3 (SEDiTBlock, resolution L/4)
│   ├── FinalNorm + AdaLN
│   └── UnPatchify2D → logit (B,1,L,L)
├── flow_loss         : BernoulliFlowLoss (同 v1)
└── [optional] family_head
```

## 关键改进详解

### M1. Multi-Scale U-DiT (MSEDiT)

在 Axial DiT blocks 之间加一级 2× 下采样/上采样:
- **Encoder**: 3 blocks at L/4 resolution
- **Middle**: 2 blocks at L/8 resolution (感受野扩大 2×, 覆盖更长程依赖)
- **Decoder**: 3 blocks at L/4 resolution + skip connection from encoder

下采样用 2×2 stride-2 conv (在 token grid 上), 上采样用 transposed conv。

效果: 对 L=640 的序列, 中间层 token grid 只有 (640/4/2)² = 80² = 6400 tokens，
Axial attention 每行/列只有 80 个 token, 极大降低长程 pair 的距离。

### M2. Relaxed Projection (松弛投影)

v1 的贪心 max-matching 强制"每行≤1 个 1":
- 优点: 保证物理合法 (canonical base pairing)
- 缺点: 对 pseudoknot 完全不友好 (一个碱基可以参与 2 种配对)

v2 的松弛投影:
1. 先用网络输出 sigmoid(logit) 得到概率图
2. 用自适应阈值 (mean + k*std of positive predictions) 二值化
3. 每行最多保留 top-2 个预测 (允许 pseudoknot)
4. 仍保证对称性和 |i-j|≥3

### M3. Cosine Adaptive Sampling Schedule

v1: dt = 1/K (均匀)
v2: dt_k = (cos(π·k/(2K)) - cos(π·(k+1)/(2K))) / (1 - cos(π/(2K)))
    → 前期步长大 (快速探索), 后期步长小 (精修细节)

### M4. Local-Enhanced Attention

在 Encoder 的前 2 层加入局部窗口 attention (window_size=8):
对 |i-j| < 8 (短程) 的 token pair 额外加一个 bias, 帮助模型关注 stem 的连续配对模式。

## 参数量对比

| | v1 | v2 |
|--|:--:|:--:|
| Encoder blocks | 6 (flat) | 3 (L/4) + 2 (L/8) + 3 (L/4) = 8 total |
| Hidden dim | 192 | 192 |
| Heads | 4 | 4 |
| Patch | 4 | 4 |
| Extra modules | — | Downsample/Upsample conv, skip proj |
| **Total params** | ~13M | ~16M |
| **Inference speed** | 基准 | ~1.1× (略慢, 但 accuracy 值得) |

## 预期改进

| Dataset | v1 F1 | v2 预期 F1 | 提升来源 |
|---------|:-----:|:----------:|----------|
| RNAStrAlign | 0.921 | 0.93+ | Multi-scale 对长 stem 更好 |
| ArchiveII | 0.861 | 0.88+ | Multi-scale + relaxed proj |
| PDB_TS_hard | 0.596 | 0.65+ | Relaxed proj (允许 PK) + adaptive sample |
| bpRNA | 0.644 | 0.68+ | Local attention 帮助短 stem |
