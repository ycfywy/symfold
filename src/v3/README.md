# SymFold v3 — Dilated Axial DiT with Physics-Aware Training

## 设计理念

v3 基于 v1 的成功经验，解决其已知局限：

### v1 问题诊断
1. **长程依赖不足**: 6 层 flat axial attention 在 L/4 分辨率上，长序列需要信息经过多层传播
2. **UFold 信息注入不够**: 仅全局池化做 AdaLN + patch-level add，丢失了空间局部信息
3. **训练无物理约束**: 物理 guidance 仅在推理时使用，训练不知道什么是物理合理的结构

### v2 失败原因
1. **U-DiT 下采样丢失精度**: 2×2 stride conv 在 token grid 上不可逆，信息丢失严重
2. **Relaxed projection 引入假阳性**: 每行≤2 在标准 RNA 上过于宽松
3. **复杂度增加但数据不变**: 过拟合

### v3 核心改进

| 改进 | 动机 | 实现 |
|------|------|------|
| **Dilated Axial Attention** | 不降分辨率就能看到 2×/4× 远处的 token | 交替 dilation=1,2,4 |
| **Cross-Resolution Attention** | 让高分辨率层能获取全局视野 | 每3层插入一个全局压缩 attention |
| **UFold Spatial Injection** | 保留 UFold 空间细节而非仅全局 | Feature-wise Linear Modulation (FiLM) |
| **Physics-Aware Loss** | 训练时就学习物理约束 | 额外 stacking + non-crossing loss |
| **Adaptive Projection** | 自适应选择 strict/relaxed | Score-aware threshold，默认 strict |

## 架构

```
Input (48ch, L×L)
  → PatchEmbed(patch=4) → tokens (L/4 × L/4, dim=256)
  → + UFold PatchEmbed + AxialPosEmbed (RoPE)
  → [DASEDiTBlock × 9]:
      Block 0,1,2: dilation=1 (local, 等效 v1)
      Block 3,4,5: dilation=2 (2× 感受野)
      Block 6,7,8: dilation=4 (4× 感受野, 覆盖整个序列)
      每个 block 内:
        AdaLN-Zero + DilatedAxialAttn + FiLM(UFold) + FFN
  → Final AdaLN + UnPatch → logit (B, 1, L, L)
  → 对称化 + short-range mask + padding mask
```

## 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_dim | 256 | 略增 (v1=192)，因为更深 |
| num_heads | 4 | 保持 |
| dim_head | 64 | 增大 head dim (v1=48) |
| num_layers | 9 | 3组 × 3层，交替 dilation |
| patch_size | 4 | 保持 |
| cond_dim | 8 | 保持 |
| max_len | 640 | 保持 |
| dilation_pattern | [1,1,1,2,2,2,4,4,4] | 渐进扩张 |
| use_rope | True | 旋转位置编码替代 learnable |
| film_layers | all | 每层都注入 UFold 空间信息 |
| physics_loss_weight | 0.1 | 训练时物理约束权重 |

## 参数量 (实测)

- **Backbone (DA-SE-DiT)**: 13.2M (vs v1=13M, v2=16M)
- **Trainable total**: 21.8M (backbone + UFold finetune)
- **Frozen (RNA-FM)**: 99.5M
- 主要增长来自 9 层 blocks (12.1M) + FiLM 层
- 仍远小于 RNADiffFold 的 109M trainable params
- GPU memory: ~3.4GB for batch=4, L=160 (H20 96GB 绰绰有余)

## 文件

```
src/v3/
├── README.md           # 本文件
├── __init__.py
├── model.py            # SymFoldModel_v3
├── da_se_dit.py        # Dilated Axial SE-DiT backbone
└── discrete_flow.py    # v3 flow (保守投影 + physics loss)
```
