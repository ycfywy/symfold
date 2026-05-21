# V3 Rethinking — RNA 二级结构预测模型改进方案

> 基于 v2-fresh 训练崩塌的诊断，重新思考模型设计。

---

## 一、问题本质

RNA 二级结构预测本质上是：**给定序列 S = (s₁, ..., s_L)，预测配对矩阵 C ∈ {0,1}^{L×L}**

### 配对矩阵的硬约束
1. **对称性**: C[i,j] = C[j,i]
2. **对角排除**: C[i,j] = 0 if |i-j| < 4（hairpin loop 最少 3 nt）
3. **每行至多 1 个 1**（canonical, 无 pseudoknot）或至多 2 个（含 pseudoknot）
4. **嵌套性**（canonical）: 若 (i,j) 和 (k,l) 都配对，则 i<k<l<j 或 k<i<j<l（不交叉）
5. **碱基互补**: 只有 A-U, G-C, G-U 能配对

### 配对矩阵的稀疏性
- 典型 RNA: L=100 时约有 30~40 个配对 → 配对率 ρ ≈ 30/(100×100/2) ≈ 0.6%
- 矩阵极度稀疏（>99% 是 0），这是 class imbalance 的根源

---

## 二、v2-fresh 失败诊断

### 现象
- Train loss: 0.321 → 0.0088（持续下降）
- Val F1: 0.296 → 0.141（持续恶化）
- Val Precision: 0.493 → 0.622（越来越保守）
- Val Recall: 0.233 → 0.095（严重欠预测）
- Best model = epoch 1（几乎没学到东西）

### 根本原因分析

#### 1. Relaxed Projection 训练-推理不一致（最致命）

**训练时**: 网络学习预测 P(x₁=1|x_t, t)，loss 是 per-element BCE  
**推理时**: 网络输出经过 `relaxed_projection`（top-5% 阈值 + 每行 top-2）

问题：
- 训练 loss 鼓励模型对**每个位置**都给出准确概率
- 但 relaxed projection 只看 **top-5% 分数的下界**作为阈值
- 当模型学得更好（分数更 calibrated），高置信位置的分数提高，top-5% 阈值也水涨船高 → 反而过滤掉了中等置信的正确配对
- 结果: 模型越训越保守，recall 持续下降

#### 2. Batch Size 过大 → 有效训练步数不足

| 设置 | v1 | v2-fresh |
|------|:--:|:--------:|
| L≤80 batch | 16 | 128 |
| L≤160 batch | 8 | 64 |
| 每 epoch 梯度步数 | ~2200 | ~275 |
| 32 epochs 总步数 | ~70400 | **~8800** |

v2-fresh 的有效训练量仅为 v1 的 **1/8**。train loss 下降快不是因为学得好，而是因为看到的梯度步太少、每步更新的 batch 统计信息太强（类似 GAN 训练中 discriminator 过强的问题）。

#### 3. U-Net Down/Up 破坏对称等变性

MSEDiT 的 stride-2 Conv → TransposedConv 路径不严格保证 f(M^T) = f(M)^T。特别是当 patch grid 为奇数时，下采样再上采样会截断。v1 的 flat 6-layer 结构天然满足行列对称。

#### 4. Cosine Schedule 前期步长过大

Cosine schedule: dt_0 ≈ 0.078 vs 均匀 dt = 0.05。前期大步长 → rate×dt 接近 1 → 过多 flip → x_t 偏离正确轨迹。

---

## 三、改进方案

### 方案 A: 保守修复（基于 v2 代码微调）

**目标**: 在 v2 架构基础上修复关键 bug，验证是否能恢复性能

1. **推理时改用 strict projection**: 将 `project_mode` 从 `relaxed` 改为 `strict`（复用 v1 的 greedy max-matching）
2. **减小 batch size 到 v1 水平**: 或保持大 batch 但增加 epoch 到 400+
3. **去掉 U-Net 结构**: 改为 flat 8-layer SEDiT（v1 的 6 层 → 8 层，保持对称）
4. **恢复均匀 schedule**: dt = 1/20
5. **降低 pos_weight**: 当前 pos_weight ≈ 199 过大，尝试 50~100

**预期**: 应该能恢复到接近 v1 水平（F1 > 0.85）

### 方案 B: 架构重构（推荐，v3 新方向）

**核心思想**: 放弃 "预测整个配对矩阵" 的思路，改为 **结构化预测**

#### B1. Direct Score → 匈牙利匹配 (End-to-End)

```
序列 → Encoder → 配对分数矩阵 S[i,j] → 匈牙利/Sinkhorn 匹配 → 配对矩阵
```

- 网络直接输出 L×L 分数矩阵（不经过 flow matching）
- 使用**可微分匹配**（Sinkhorn OT 或 differentiable Hungarian）将分数转为满足约束的配对
- Loss 直接在最终配对矩阵上算 F1-aware loss（而非 per-element BCE）
- **优势**: 训练/推理完全一致，无 projection gap
- **风险**: Sinkhorn 的温度参数需要精调，梯度可能不稳定

#### B2. Flow Matching + 改进投影（渐进策略）

保留 Discrete Flow Matching 框架，但修复训练-推理 gap：

```python
# 关键改动: 训练时也加入 projection，让 loss 看到投影后的效果
def forward_with_projection(self, ...):
    logit = self.backbone(x_t, t, ...)
    p_x1 = sigmoid(logit)
    
    # Straight-Through Estimator: 前向走 projection，反向走连续梯度
    p_projected = differentiable_projection(p_x1, contact_masks)  # 新增!
    
    # Loss 同时计算在 raw logit 和 projected 上
    loss_raw = BCE(logit, x_1, pos_weight=...)      # 原始 loss
    loss_proj = BCE(p_projected, x_1)                # 投影后 loss
    loss = 0.7 * loss_raw + 0.3 * loss_proj
```

可微分投影的实现思路:
```python
def differentiable_projection(p, mask, tau=0.1):
    """Sinkhorn-like iterative normalization"""
    # 1. 对称化: p = (p + p.T) / 2
    # 2. 迭代行/列归一化 (类似 Sinkhorn):
    #    每行的 pair 概率之和 ≤ 1 (每个位置最多配一个)
    for _ in range(sinkhorn_iters):
        row_sum = p.sum(dim=-1, keepdim=True)
        p = p / row_sum.clamp(min=1.0)  # 如果 sum>1 则归一化
        p = (p + p.T) / 2               # 恢复对称性
    # 3. 温度缩放: p = sigmoid((p - 0.5) / tau)
    return p
```

#### B3. 两阶段方法 (Coarse-to-Fine)

```
Stage 1: 快速粗预测 (UFold-like CNN)
  - 1 forward pass → 粗粒度配对概率
  - 用 v1 strict projection 得到初始结构

Stage 2: Flow Matching 精修
  - 以 Stage 1 的输出作为 x_t 的初始化 (而非 Bernoulli(0.005))
  - 少步采样 (5~10 步) 精修
  - 关注 Stage 1 不确定的位置
```

**优势**: 
- Stage 1 提供强先验，大幅降低 Stage 2 的搜索空间
- Flow Matching 只需要微调少量位置，不容易崩塌
- 推理快（5步 vs 20步）

#### B4. 对比学习 + 结构损失（补充方案）

在 BCE 之外增加结构感知 loss:

```python
# 1. Row-constraint loss: 每行概率之和 ≈ 1 (one-pair-per-base 的软约束)
row_sum_loss = F.mse_loss(p.sum(dim=-1), target_row_sum)

# 2. Nesting loss: 惩罚 crossing pairs
crossing_penalty = detect_crossings(p) * lambda_cross

# 3. Stem-aware loss: 配对通常成堆出现 (stem)，鼓励连续配对
stem_bonus = compute_stem_continuity(p)

total_loss = bce_loss + 0.1 * row_sum_loss + 0.05 * crossing_penalty - 0.05 * stem_bonus
```

---

## 四、推荐路线 (v3)

### Phase 1: 快速验证 (1~2天)

1. **v2 + strict projection**: 不改任何训练代码，只在推理时用 v1 的 greedy 投影
   - 如果 F1 能到 0.5+ → 说明 backbone 没问题，问题全在 relaxed projection
   
2. **v2 + 小 batch + 长训**: batch 恢复 v1 水平，训 200 epochs
   - 验证是否是训练不足

### Phase 2: 架构优化 (3~5天)

基于 Phase 1 结论选择路线:

**若 strict projection 有效**:
- 走方案 B2: 加入 differentiable projection 到训练
- 多步骤: raw loss → 加入 row-constraint loss → 加入 projection loss

**若仍不行** (backbone 本身学歪了):
- 走方案 B3: Two-stage coarse-to-fine
- Stage 1 复用 UFold 的直接预测，Stage 2 用 flat SEDiT (v1 架构) 精修

### Phase 3: 极限优化 (可选)

- 增加采样步数到 50~100
- Multi-seed voting (5 次采样取平均)
- Physics guidance (自由能最小化)
- 集成 v1 + v3 的 ensemble

---

## 五、具体代码改动建议

### 5.1 快速验证: v2 模型 + strict projection

修改 `train/config/train_config_v2_fresh.json` 中的 sampling 配置:
```json
"sampling": {
    "num_steps": 20,
    "project_mode": "strict"  // ← 改这里
}
```

或直接修改 eval 代码，在调用 sample 时传入 `project_mode='strict'`。

### 5.2 v3 核心: Differentiable Row-Normalized Projection

```python
# src/v3/differentiable_projection.py

def sinkhorn_row_projection(logit, contact_masks, num_iters=5, tau=0.5):
    """
    可微分的行归一化投影
    保证每行概率之和 ≤ 1 (约等于每位置最多 1 个配对)
    """
    p = torch.sigmoid(logit / tau)  # 温度缩放的 sigmoid
    p = p * contact_masks           # mask padding + |i-j|<4
    
    for _ in range(num_iters):
        # 对称化
        p = (p + p.transpose(-2, -1)) / 2
        
        # 行归一化 (row sum ≤ 1)
        row_sum = p.sum(dim=-1, keepdim=True)  # (B, 1, L, 1)
        scale = (1.0 / row_sum.clamp(min=1.0))  # 只缩小不放大
        p = p * scale
        
        # 列归一化 (col sum ≤ 1, 由对称性保证)
        p = (p + p.transpose(-2, -1)) / 2
    
    return p
```

### 5.3 v3 Loss: 结构感知多任务

```python
# src/v3/structured_loss.py

class StructuredFlowLoss(nn.Module):
    def __init__(self, rho_0=0.005, pos_weight_scale=0.5, lambda_row=0.1):
        super().__init__()
        self.bce_loss = BernoulliFlowLoss(rho_0, pos_weight_scale=pos_weight_scale)
        self.lambda_row = lambda_row
    
    def forward(self, logit, x_1, t, contact_masks):
        # 1. 基础 BCE loss (降低 pos_weight)
        loss_bce = self.bce_loss(logit, x_1, t, contact_masks)
        
        # 2. Row-sum constraint loss
        p = torch.sigmoid(logit) * contact_masks
        row_sum = p.sum(dim=-1)  # (B, 1, L)
        # Ground truth: 每行要么有 1 个 pair (sum=1) 要么没有 (sum=0)
        gt_row_sum = (x_1.sum(dim=-1) > 0).float()  # (B, 1, L)
        loss_row = F.binary_cross_entropy(
            row_sum.clamp(0, 1), gt_row_sum, reduction='mean')
        
        return loss_bce + self.lambda_row * loss_row
```

### 5.4 v3 Backbone: 回归 Flat + 更深

```python
# src/v3/model.py 关键配置

model_config = {
    "hidden_dim": 256,       # 略增
    "num_heads": 8,          # 增加 head 数
    "dim_head": 32,          # 保持总宽度
    "num_layers": 8,         # flat 8层 (不用 U-Net!)
    "patch_size": 4,
    "cond_dim": 8,
    "max_len": 640,
    "dp_rate": 0.15,         # 略增 dropout 防过拟合
    "rho_0": 0.005,
    "pos_weight_scale": 0.5, # 降低 pos_weight (从 199 → ~100)
    "project_mode": "strict" # 回到严格投影
}
```

---

## 六、Baseline 对比与目标

| Model | RNAStrAlign F1 | ArchiveII F1 | bpRNA F1 | PDB_hard F1 |
|-------|:-----------:|:----------:|:-------:|:----------:|
| v1 (current best) | 0.921 | 0.861 | 0.644 | 0.596 |
| v2-fresh (failed) | 0.371 | 0.364 | 0.320 | 0.285 |
| **v3 目标** | **>0.93** | **>0.87** | **>0.70** | **>0.65** |
| UFold (参考) | 0.96 | 0.86 | — | — |
| MXfold2 (参考) | 0.93 | 0.78 | — | — |

### 改进信心来源
- v1 已经证明 Discrete Flow Matching + Axial DiT 能到 0.92 → 架构方向正确
- v2 失败的原因清晰（relaxed projection gap + 大 batch 训不足）→ 修复后应能恢复
- 加入结构约束 loss + 可微投影 → 应能超过 v1

---

## 七、实验计划

| 实验 | 改动 | 预期 | 优先级 |
|------|------|------|:------:|
| E1: v2 + strict proj (eval only) | 仅改 eval 投影 | F1 0.3→0.4+ | ★★★ |
| E2: v2 + 小 batch + 200ep | 减 batch, 增 epoch | F1 0.5+ | ★★★ |
| E3: flat 8L + strict proj | 去 U-Net, 加层 | F1 0.85+ | ★★☆ |
| E4: + row-constraint loss | 加结构损失 | F1 +0.02 | ★★☆ |
| E5: + diff projection training | B2 方案 | F1 +0.03 | ★☆☆ |
| E6: two-stage coarse-to-fine | B3 方案 | F1 0.93+ | ★☆☆ |

优先做 E1 (零成本验证)，如果 F1 提升明显则走 E3 → E4 路线。

---

## 八、总结

v2 失败的核心教训：
1. **训练与推理必须一致** — relaxed projection 只在推理用，训练从未见过它，gap 导致崩塌
2. **不要盲目加大 batch** — 有效梯度步数才是关键
3. **结构约束不是后处理** — 应该融入训练过程
4. **简单稳定优先** — v1 的 flat 架构 + 严格投影 + 均匀步长就是最好的 baseline

v3 的设计原则：
- **训练时就看到约束** (differentiable projection in the loop)
- **flat 深层 > 浅层 U-Net** (保持对称等变性)
- **结构感知 loss** (row-sum, stem continuity)
- **合理的 batch size** (确保充分训练)
