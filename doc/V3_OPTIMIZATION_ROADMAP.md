# V3 优化方向思考 — 从 0.752 到 0.85+

> 基于 v3 (80 epochs, avg F1=0.752) 的结果，思考进一步优化方向。

---

## 一、当前瓶颈诊断

### 1.1 各数据集分析

| Dataset | F1 | 瓶颈 | 可能原因 |
|---------|:--:|:----:|----------|
| RNAStrAlign | 0.939 | 接近天花板 | 已接近 UFold (0.96)，提升空间有限 |
| ArchiveII | 0.864 | Precision 偏低 (0.834) | 假阳性配对过多 |
| bpRNA | 0.636 | P=0.557, 大量假阳性 | bpRNA 含大量非标准结构，模型过度预测 |
| PDB_TS_hard | 0.634 | Recall 偏低 (0.587) | 复杂长程配对漏检 |
| PDB_TS1 | 0.716 | 均衡偏低 | OOD 泛化不足 |

### 1.2 整体问题

1. **训练不足**: 80 epochs 曲线仍在上升，未充分收敛
2. **Precision/Recall 不均衡**: bpRNA 上 P<<R，模型倾向过度预测
3. **OOD 泛化**: PDB 系列是从 3D 结构提取的，含 pseudoknot 和非标准配对
4. **Val 指标不全面**: 只用 bpRNA VL0 做 validation，model selection 不够

---

## 二、优化方向

### 方向 A: 继续训练 (最低风险, 预期 +2~5%)

**现状**: 80 epochs 后曲线仍在上升 (F1: 0.575→0.603 on val)

**方案**:
```json
{
  "epochs": 200,
  "lr": 4e-5,          // restart 用半 lr
  "warmup_epochs": 3,
  "patience": 40,
  "full_eval_every": 10  // 每 10 epoch 跑全量 eval
}
```

**预期**: 
- Val F1: 0.603 → 0.65+ 
- Test avg F1: 0.752 → 0.78+
- 依据: v1 训练 200+ epochs 才收敛，v3 80 epochs 相当于 v1 的 40%

### 方向 B: Multi-Val Model Selection (零成本, 预期 +1~3%)

**问题**: 当前只用 bpRNA VL0 选 best model，但 bpRNA 指标和其他数据集相关性不完美。

**方案**: 在 trainer 中加入定期全量 eval：

```python
# 每 10 epochs 在 PDB_TS1/2/3/hard + bpRNA_TS0 上跑快速 eval
# 以加权 F1 作为 model selection criterion
EVAL_WEIGHTS = {
    'bpRNA_VL0': 0.3,    # 快速验证
    'PDB_TS1': 0.15,     # OOD-hard
    'PDB_TS2': 0.10,     # OOD-hard  
    'PDB_TS3': 0.10,     # OOD-hard
    'PDB_TS_hard': 0.20, # 最难
    'RNAStrAlign_subset': 0.15,  # 随机 200 样本子集
}
```

**额外成本**: ~3 min per full eval (PDB 系列很小)

### 方向 C: 数据增强 (中等风险, 预期 +2~4%)

**问题**: 训练集 34K samples 可能不够，尤其对 OOD 数据集。

**方案 C1: 序列级增强**
```python
# 1. Random masking: 随机 mask 5-10% 的碱基 → 模型需从部分序列推断结构
# 2. Reverse complement: RC 序列的配对矩阵是转置 → 数据量 ×2
# 3. Subsequence cropping: 从长序列中截取连续子序列 → 增加短序列多样性
```

**方案 C2: Noise augmentation on contact map**
```python
# 在 ground truth 上加随机 flip: 以小概率翻转少量配对
# 模型需学会去噪 → 提升 robustness
contact_aug = contact.clone()
flip_prob = 0.01
flip_mask = torch.rand_like(contact_aug) < flip_prob
contact_aug = torch.where(flip_mask, 1 - contact_aug, contact_aug)
```

### 方向 D: Loss 改进 (中等风险, 预期 +1~3%)

#### D1. Focal Loss 替代 BCE

bpRNA 上 Precision 极低说明模型过度预测 → 假阳性置信度高但被 BCE 惩罚不够。

```python
# Focal Loss: 对 "easy negative" 降权，聚焦 hard examples
def focal_bce(logit, target, gamma=2.0):
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
    pt = p * target + (1-p) * (1-target)
    focal_weight = (1 - pt) ** gamma
    return (focal_weight * ce).mean()
```

#### D2. Adaptive pos_weight

当前 pos_weight=199 对所有 epoch 固定。随着模型变好，应该逐渐降低 pos_weight：

```python
# pos_weight annealing: 从 199 → 50 over training
pos_weight = 199 * (1 - 0.75 * epoch / total_epochs)  # 最终降到 ~50
```

#### D3. Structure-Aware F1 Loss (differentiable)

直接优化 F1 而非 BCE（需要 soft F1 近似）：

```python
def soft_f1_loss(pred_prob, target, eps=1e-6):
    tp = (pred_prob * target).sum()
    fp = (pred_prob * (1 - target)).sum()
    fn = ((1 - pred_prob) * target).sum()
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    return 1 - f1
```

### 方向 E: 架构微调 (中等风险, 预期 +2~5%)

#### E1. Triangle Multiplicative Update

受 AlphaFold2 启发，捕获三体关系 (如果 i 和 k 配对，则抑制 i 和 j 配对)：

```python
class TriangleUpdate(nn.Module):
    """z[i,j] += Σ_k gate(z[i,k]) * gate(z[k,j])"""
    # 在后 3 层 block 中加入
    # 复杂度: O(L³/patch³)，对于 patch=4, L=640 → 40³ ≈ 64K ops
```

**成本**: 每层增加 ~15% 计算量
**收益**: 更好地捕获 "每行至多一对" 的竞争关系

#### E2. Cross-Resolution Attention

每 3 层插入一个全局压缩 attention (类似 Perceiver):

```python
# 将 token grid (H, W) 压缩到 (H/2, W/2) 做 global attention
# 然后上采样回来 → 信息跨全局传播
class CrossResolutionAttn(nn.Module):
    def __init__(self, dim, compress_ratio=2):
        self.compress = nn.AvgPool2d(compress_ratio)
        self.global_attn = nn.MultiheadAttention(dim, 4)
        self.upsample = nn.Upsample(scale_factor=compress_ratio)
```

#### E3. 去 Patch 化 (短序列)

对 L≤160 的短序列，使用 patch_size=2 甚至 1：

```python
# patch=4: L=160 → 40×40 tokens, 每 token 16 个元素
# patch=2: L=160 → 80×80 tokens, 每 token 4 个元素 (信息更精细)
# patch=1: L=160 → 160×160 tokens (计算量 ×16, 可能过大)
```

对短序列 (L≤80) 用 patch=2 可能提升精度。

### 方向 F: 推理增强 (零训练成本, 预期 +1~3%)

#### F1. Multi-Seed Voting

```python
# 5 次独立采样 → 投票
preds = [sample(seed=s) for s in range(5)]
voted = (sum(preds) / 5 > 0.5).float()
# 预期: F1 +1~2% (RNADiffFold 用 10 次投票)
```

#### F2. 增加采样步数

```python
# num_steps: 20 → 50
# 更多步 = 更精细的去噪过程
# 预期: 小幅提升 (主要对难样本)
```

#### F3. Physics Guidance (推理时)

```python
# 开启 physics_beta > 0: 在采样时施加自由能梯度
# - stacking energy: 鼓励连续配对
# - WC bonus: 鼓励 AU/GC 配对
# - PK penalty: 惩罚交叉配对
physics_beta = 0.3  # 尝试 0.1~0.5
```

#### F4. Ensemble v1 + v3

```python
# v1 和 v3 的 logit 加权平均
logit_final = 0.6 * logit_v3 + 0.4 * logit_v1
# 两个模型架构不同，错误模式互补
```

### 方向 G: Row-Softmax 参数化 (高风险高回报, 预期 ±5~10%)

这是 V3_ARCHITECTURE_DESIGN.md 中描述的核心思想：

```python
# 当前: 每个 (i,j) 独立预测 sigmoid(logit) → BCE loss
# 改为: 每行输出 (L+1)-way softmax → per-row CE loss
# 优势: 天然约束每行至多一配对，无需后处理 projection
# 风险: 需要大量调参，可能训练初期不稳定
```

**建议**: 作为 v4 方向的独立实验。先在小规模 (L≤160) 上验证。

---

## 三、优先级排序

| 优先级 | 方向 | 预期收益 | 成本 | 风险 |
|:------:|------|:--------:|:----:|:----:|
| ★★★ | A: 继续训练到 200ep | +2~5% | 24h GPU | 低 |
| ★★★ | B: Multi-Val selection | +1~3% | 代码 1h | 低 |
| ★★★ | F1: Multi-seed voting | +1~2% | 推理时间 ×5 | 无 |
| ★★☆ | D1: Focal Loss | +1~3% | 代码 2h | 中 |
| ★★☆ | C1: 数据增强 | +2~4% | 代码 3h | 中 |
| ★★☆ | F3: Physics guidance | +0.5~2% | 调参 | 低 |
| ★☆☆ | E1: Triangle Update | +2~5% | 代码 1d | 中 |
| ★☆☆ | D3: Soft F1 Loss | +1~3% | 代码 3h | 中 |
| ★☆☆ | G: Row-Softmax (v4) | ±5~10% | 1w | 高 |

---

## 四、推荐实施路线

### Phase 1: 低风险快赢 (1-2天)

1. **继续训练到 200 epochs** (lr=4e-5, cosine restart)
2. **加入 full eval 作为 model selection** (每 10ep eval PDB 系列)
3. **推理时开 5-seed voting** (eval 脚本参数)

预期: avg F1 0.752 → 0.78+

### Phase 2: Loss + 数据 (3-5天)

4. **Focal Loss** 替代 BCE (解决 bpRNA 假阳性问题)
5. **Adaptive pos_weight** 衰减
6. **Reverse complement 增强** (数据量 ×2)

预期: avg F1 0.78 → 0.82+

### Phase 3: 架构优化 (5-7天)

7. **Triangle Multiplicative Update** (后 3 层)
8. **短序列去 patch** (L≤80 用 patch=2)
9. **Ensemble v1 + v3**

预期: avg F1 0.82 → 0.85+

### Phase 4: 探索性 (选做)

10. Row-Softmax 参数化实验
11. Two-stage coarse-to-fine
12. 更大模型 (512 dim, 12 层)

---

## 五、具体代码改动提纲

### 5.1 继续训练 (修改 config)

```json
// train/config/train_config_v3_continue.json
{
  "task_name": "260521-v3-continue",
  "training": {
    "epochs": 200,
    "lr": 4e-5,
    "warmup_epochs": 3,
    "auto_resume": true,
    "patience": 40,
    "full_eval_every": 10,
    "full_eval_sets": ["PDB_TS1", "PDB_TS2", "PDB_TS3", "PDB_TS_hard"]
  }
}
```

### 5.2 Full Eval in Training Loop

```python
# train/train_v3.py 新增
def run_full_eval(model, test_sets, device, config):
    """每 full_eval_every epochs 运行一次全量 eval"""
    results = {}
    for name in test_sets:
        loader = build_test_loader(name, config)
        f1 = evaluate_single_set(model, loader, device, config)
        results[name] = f1
    # 加权 F1
    weights = {'PDB_TS1': 0.25, 'PDB_TS2': 0.15, 'PDB_TS3': 0.15, 'PDB_TS_hard': 0.30, 'bpRNA_VL0': 0.15}
    weighted_f1 = sum(results.get(k, 0) * w for k, w in weights.items())
    return weighted_f1, results
```

### 5.3 Multi-Seed Voting (eval 参数)

```bash
python eval/eval.py --ckpt model/260520-v3-train/best.pt \
    --test_sets bpRNA,RNAStrAlign,ArchiveII,PDB_TS1,PDB_TS2,PDB_TS3,PDB_TS_hard \
    --num_samples 5 \
    --out_json output/260520-v3-train/eval_5vote.json
```

---

## 六、长期目标

| 目标 | 指标 | 路线 |
|------|:----:|------|
| 超越所有 test set 上 v1 | avg F1 > 0.77 | Phase 1 即可 |
| 接近 UFold (SOTA on RNAStrAlign) | RNAStrAlign F1 > 0.95 | Phase 2 |
| OOD 泛化优秀 | PDB_TS_hard F1 > 0.70 | Phase 2-3 |
| 论文级结果 | avg F1 > 0.85, 全部超 SOTA | Phase 3-4 |
