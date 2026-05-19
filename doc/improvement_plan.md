# SymFold 模型分析与改进方案

**生成时间**: 2026-05-15

## 1. 当前模型训练状态总结

### 1.1 训练完成情况
- **训练配置**: `260514-full-train-symfold`
- **已完成 epoch**: 80 / 80（全部完成）
- **最佳验证 F1**: 0.5654（heartbeat 记录）/ 0.5011（vis_samples_report epoch 29 时）
- **训练结束时间**: 2026-05-15 01:52:33
- **模型参数量**: ~13M trainable

### 1.2 验证集性能趋势

| Epoch | F1 | Precision | Recall | MCC |
|-------|------|-----------|--------|------|
| 1 | 0.4228 | 0.3351 | 0.631 | 0.4483 |
| 5 | 0.4360 | 0.3474 | 0.650 | 0.4625 |
| 15 | 0.4525 | 0.3645 | 0.656 | 0.4773 |
| 21 | 0.4838 | 0.3937 | 0.686 | 0.5081 |
| 25 | 0.4987 | 0.4109 | 0.691 | 0.5213 |
| 29 | 0.5011 | 0.4120 | 0.695 | 0.5238 |
| 最终 | ~0.5654 | - | - | - |

**关键观察**:
- F1 从 epoch 1 的 0.42 稳步上升到最终的 0.5654
- Precision 始终低于 Recall，说明 **假阳性（FP）偏多**
- 上升趋势尚未完全饱和，仍有改进空间

---

## 2. 模型核心问题分析

### 2.1 问题一：Precision 偏低，FP 过多

**现象**: 从可视化样本中观察到，模型预测的 contact map 中蓝色点（FP = 误预测的配对）较多，而绿色点（TP = 正确预测的配对）和红色点（FN = 遗漏的配对）的比例说明模型倾向于 **过度预测碱基配对**。

**根因分析**:
1. **极端类别不平衡**: contact map 中约 99.5% 的位置为 0（未配对），仅 0.5% 为 1（配对），先验 ρ₀=0.005。虽然 BernoulliFlowLoss 使用了 pos_weight ≈ 199 来补偿，但该权重可能 **过于激进**，导致模型对正样本过于敏感。
2. **采样步数不足**: 当前 inference 使用 `num_steps=20`，τ-leap CTMC 采样步数偏少，可能导致采样轨迹未能充分收敛到数据分布。
3. **Projection 贪心策略局限**: `project_to_valid_contact_map` 使用贪心最大匹配，每次选最高分的 pair 并清零对应行/列，这种策略在 FP 较多时可能选到错误的高分 pair。

**改进建议**:
- **降低 pos_weight_scale**: 将 `pos_weight_scale` 从 1.0 降至 0.5~0.7，减少对正样本的过度加权
- **增加采样步数**: 将 `num_steps` 从 20 增加到 30~50，给采样更多时间收敛
- **引入更优的 projection 算法**: 考虑使用匈牙利算法或 Sinkhorn-Knopp 等最优匹配替代贪心策略

### 2.2 问题二：训练曲线上升缓慢，收敛不充分

**现象**: 80 个 epoch 后 F1 仍在缓慢上升（从 0.50 → 0.56），说明模型尚未完全收敛。

**根因分析**:
1. **学习率策略过于简单**: 当前仅使用线性 warmup（3 epoch），之后保持恒定 lr=2e-4，缺乏 cosine annealing 或 step decay 等调度策略
2. **训练 epoch 数不足**: 80 epoch 对于这种复杂的生成模型来说可能不够
3. **缺乏数据增强**: 没有对 RNA 序列/结构进行增强（如反向互补、随机 masking 等）

**改进建议**:
- **引入 Cosine Annealing 学习率调度**: 使用 cosine annealing with warm restarts，让学习率在后期逐步降低
- **延长训练至 160~200 epoch**: 给模型更多时间收敛
- **降低后续训练学习率**: 从 last.pt 继续时使用 lr=5e-5（原 2e-4 的 1/4）
- **增加 dropout**: 将 `dp_rate` 从 0.1 提升到 0.15，增强泛化能力

### 2.3 问题三：模型容量与特征利用

**现象**: SE-DiT backbone 仅 6 层，hidden_dim=192，总参数约 13M，相对于 RNA 二级结构预测任务可能容量偏小。

**根因分析**:
1. **SE-DiT 层数偏少**: 6 层 axial attention 可能不足以捕获长程碱基配对关系
2. **RNA-FM 特征利用不充分**: RNA-FM 输出 640 维嵌入被压缩到 8 维（fm_proj_dim=8），信息损失严重
3. **UFold 条件信号较弱**: UFold 从 17ch FCN 特征映射到 8ch cond_dim，可能丢失空间细节

**改进建议**:
- **增加 SE-DiT 层数至 8~10 层**: 增强长程依赖建模能力
- **增大 fm_proj_dim**: 从 8 增大到 16 或 32，保留更多 RNA-FM 语义信息
- **增大 cond_dim**: 从 8 增大到 16，让 UFold 条件信号更丰富
- **引入跳跃连接**: 在 SE-DiT 层之间加入 skip connections，改善梯度流

### 2.4 问题四：Physics Guidance 未启用

**现象**: 训练和评估时 `physics_beta=0.0`，物理能量引导完全关闭。

**分析**: PhysicsGuidance 模块已实现 Turner 能量（WC/GU pair energy + stacking bonus + pseudoknot penalty），但在当前配置中未使用。

**改进建议**:
- **评估时启用 physics guidance**: 设置 `physics_beta=0.3~1.0`，利用碱基配对的热力学先验
- **设置 pseudoknot penalty**: `physics_lambda_pk=0.3~0.5`，抑制 pseudoknot 产生
- **保持 stacking bonus**: `physics_alpha_stack=1.0`，鼓励连续的螺旋区

---

## 3. 改进优先级排序

| 优先级 | 改进项 | 预期收益 | 实施难度 |
|--------|--------|----------|----------|
| P0 | 继续训练 (lr=5e-5, epochs→160) | 高 | 低 |
| P0 | 启用 Physics Guidance (评估时) | 高 | 低 |
| P1 | 降低 pos_weight_scale (0.5~0.7) | 中高 | 低 |
| P1 | 增加采样步数 (num_steps→30~50) | 中高 | 低 |
| P1 | 引入 Cosine Annealing LR | 中 | 中 |
| P2 | 增加 SE-DiT 层数 (6→8~10) | 中高 | 中 |
| P2 | 增大 fm_proj_dim (8→16~32) | 中 | 低 |
| P2 | 增大 cond_dim (8→16) | 中 | 低 |
| P3 | 改进 projection 算法 | 中 | 高 |
| P3 | 引入数据增强 | 中 | 中 |
| P3 | 增加 dropout (0.1→0.15) | 低中 | 低 |

---

## 4. 立即可执行的改进配置

### 4.1 继续训练配置 (`config/train_continue.json`)

已创建配置文件 `config/train_continue.json`，关键变更：
- `epochs`: 80 → 160（继续训练 80 个 epoch）
- `lr`: 2e-4 → 5e-5（降低学习率进行微调）
- `auto_resume`: false → true（从 last.pt 自动恢复）
- `dp_rate`: 0.1 → 0.15（增强正则化）
- `patience`: 15 → 20（更耐心等待改善）
- `num_steps`（采样）: 20 → 30（更多采样步数）
- `physics_beta`: 0.0 → 0.5（启用物理引导）
- `physics_lambda_pk`: 0.0 → 0.3（伪结惩罚）
- `model_save_dir`: 复用 `model/260514-full-train-symfold`（从 last.pt 恢复）
- `output_dir`: `output/260515-continue-train-symfold`（新输出目录）

### 4.2 启动命令

```bash
cd /root/aigame/dannyyan/RNADiffFold/symfold
bash scripts/run_train.sh config/train_continue.json
```

### 4.3 监控命令

```bash
# 查看训练日志
tail -f logs/260515-continue-train-symfold.log

# 查看心跳状态
cat logs/260515-continue-train-symfold.heartbeat | python -m json.tool

# 运行监控脚本
python scripts/monitor.py --task 260515-continue-train-symfold
```

---

## 5. 中期改进方案（需要代码修改）

### 5.1 引入 Cosine Annealing 学习率调度

**修改文件**: `train/train.py`

在 `warmup` 函数处，替换为 cosine annealing with warm restarts：
```python
import math

def cosine_annealing_lr(epoch, total_epochs, warmup_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

### 5.2 增大模型容量

**修改文件**: 创建新配置 `config/train_large.json`

关键参数变更：
- `num_layers`: 6 → 8
- `hidden_dim`: 192 → 256
- `fm_proj_dim`: 8 → 16
- `cond_dim`: 8 → 16
- `fm_attn_proj_dim`: 8 → 16

### 5.3 改进 Projection 算法

**修改文件**: `src/discrete_flow.py`

考虑替换 `project_to_valid_contact_map` 中的贪心策略为 top-k 过滤 + 非极大值抑制：
1. 先按 score 排序所有候选 pair
2. 使用非极大值抑制（类似目标检测 NMS）去除冲突的 pair
3. 保证每个碱基最多参与一个配对

---

## 6. 总结与下一步

当前 SymFold 模型在 80 epoch 训练后达到 best_f1=0.5654，仍有显著改进空间。**最紧迫的改进是继续训练（降低学习率微调）并在评估时启用物理能量引导**。中期来看，需要增大模型容量和改进学习率调度策略。长期可考虑引入更先进的 projection 算法和数据增强技术。

目标：通过上述改进，将 F1 提升至 0.65+ 水平。
