# SymFold v4 评估总结与 v5 设计方案

> 生成时间: 2026-05-25

---

## 一、v4 评估结果

### 1.1 各数据集性能

| Dataset | N | Type | F1 | Precision | Recall | MCC |
|---------|---:|:----:|:---:|:---------:|:------:|:---:|
| RNAStrAlign | 2,023 | ID | **0.941** | 0.920 | 0.967 | 0.942 |
| ArchiveII | 3,911 | OOD | **0.870** | 0.839 | 0.912 | 0.872 |
| PDB_TS2 | 38 | OOD-hard | 0.780 | 0.841 | 0.732 | 0.782 |
| PDB_TS1 | 60 | OOD-hard | 0.707 | 0.766 | 0.670 | 0.711 |
| bpRNA | 1,304 | ID | 0.638 | 0.560 | 0.783 | 0.653 |
| PDB_TS3 | 18 | OOD-hard | 0.630 | 0.714 | 0.574 | 0.635 |
| PDB_TS_hard | 28 | OOD-hardest | 0.608 | 0.685 | 0.562 | 0.614 |
| **Average** | | | **0.739** | | | |

### 1.2 核心问题总结

| 问题 | 严重度 | 影响范围 | 根因 |
|------|:------:|----------|------|
| **低密度过预测** | P0 | 78% 的失败 (380+ 样本) | pos_weight 偏高 + 训练集密度偏差 + FM 信息压缩过度 |
| **结构位置错误** | P1 | 130+ 样本 | 4×4 patch 分辨率不足 + 采样步数不够 |
| **Pseudoknots** | P2 | ~60 样本 | greedy projection 限制每行 1 配对 |
| **长序列退化** | P3 | ~40 样本 | 500+ nt 超出训练分布 |

### 1.3 关键数据

- ppb < 0.3 的 RNA: avg F1 = **0.34**，失败率 **72%**
- ppb ∈ [0.5, 0.6] (主体): avg F1 = **0.92**，失败率 **0.9%**
- 过预测 3×+ 的样本: avg F1 = **0.22**，共 118 条

---

## 二、v5 设计方案

### 2.1 设计目标

针对 v4 的三大失败模式，v5 的核心改动：

1. **解决低密度过预测** — 密度条件注入 + 密度引导采样
2. **提升位置精度** — 增大 FM 融合维度 + 输出精修卷积
3. **增强信息利用** — FM 压缩从 16D 升到 64D

### 2.2 架构改动清单

| 模块 | v4 | v5 | 改动理由 |
|------|----|----|----------|
| FM Fusion 输出维度 | 16 | **64** | v4 压缩 160:1 太狠，大量 RNA-FM 信息丢失 |
| 输入通道 | 48ch | **80ch** | FM 2D 从 16→64ch (outer_concat 后 128→保留 64) |
| Density 条件注入 | 仅辅助 loss | **注入全局条件 + 影响采样** | 让模型知道该预测多少对 |
| 输出精修 | 无 | **2 层 Conv 在 L×L 精修** | 修正 patch 边界伪影 |
| 采样 | 固定 threshold | **密度自适应 threshold** | 低密度 RNA 少翻转 |
| pos_weight | min=50 | **min=20** | 进一步降低低密度过预测 |
| 验证集 | bpRNA VL0 only | **bpRNA VL0 + RNAStrAlign val** | 更好的泛化信号 |

### 2.3 参数量估计

| 模块 | v4 参数 | v5 参数 | 增量 |
|------|:-------:|:-------:|:----:|
| FM Fusion | ~55K | ~175K | +120K |
| PatchEmbed (输入通道增大) | ~3.1M | **~5.1M** | +2.0M |
| 输出精修 Conv | 0 | ~4K | +4K |
| 其他不变 | ~22M | ~22M | 0 |
| **实际日志** | — | **122.4M total / 22.87M trainable** | RNA-FM 冻结 |

---

## 三、v5 代码文件

| 文件 | 作用 |
|------|------|
| `src/v5/da_se_dit.py` | DASEDiT_v5 backbone (Wider FM + Density Cond + Refine Conv) |
| `src/v5/model.py` | SymFoldModel_v5 主模型 (训练+密度引导采样) |
| `src/v5/README.md` | v5 架构说明 |
| `train/config/train_config_v5.json` | v5 训练配置 |

---

## 四、v5 关键设计详解

### 4.1 解决低密度过预测 (P0)

**三重防线**：

```
防线1: 训练时
  - pos_weight_min: 50 → 20 (低密度 RNA 的正样本权重更低)
  - focal_gamma: 1.0 → 1.5 (更强地聚焦 hard examples)
  - density_weight: 0.1 → 0.2 (更重视密度预测准确性)
  - density 作为条件注入 (训练时用 GT density)

防线2: 模型内部
  - 密度 embedding 注入全局条件 (让模型知道目标密度)
  - 全局条件: time + FM_global + UFold_global + density → Linear(1024→256)

防线3: 采样时
  - 先预测 density，再条件化后续采样
  - 低密度 → damping rate_01 (减少 0→1 翻转)
  - damping = clamp(2×density_pred, max=1.0)
  - 效果: density=0.1 的 RNA，翻转率降低 80%
```

### 4.2 提升信息利用 (FM 压缩从 160:1 到 40:1)

```
v4: 4×640 → MultiLayerFMFusion → 16D → proj(16→8) → outer(8→16ch)
    信息瓶颈: 每层只有 4D 的表达空间

v5: 4×640 → MultiLayerFMFusion_v5 → 64D → proj(64→32) → outer(32→64ch)
    每层有 16D 的表达空间，4× 提升

改进:
  - 每层投影: Linear(640→128→64) (加了非线性，v4 只有 Linear(640→16))
  - 加权平均: Linear(640→128→64) (比 v4 的 reuse first proj 更好)
  - 融合 MLP: Linear(256→128→64) (处理 4×64=256D 的 concat)
```

### 4.3 输出精修 Conv

```python
class OutputRefineConv(nn.Module):
    def __init__(self):
        self.net = nn.Sequential(
            Conv2d(1, 16, k=3, pad=1), GELU,   # 感受野: 3×3
            Conv2d(16, 16, k=3, pad=1), GELU,  # 感受野: 5×5 (两层叠加)
            Conv2d(16, 1, k=1),                 # 压回 1ch, zero-init
        )
    
    def forward(self, logit):
        return logit + self.net(logit)  # 残差! 初始等于 identity
```

**作用**：
- 修正 4×4 patch 边界的不连续跳变
- 利用 5×5 局部上下文精调 logit
- zero-init 确保训练开始时不影响 v4 已学到的表示

### 4.4 密度引导采样流程

```
推理 Pipeline:
1. 初始化 x_0 ~ Bernoulli(0.005)
2. 快速前向: backbone(x_0, t=0.5) → density_pred  (一次额外推理)
3. 条件化采样 (20步):
   for k in range(20):
     logit = backbone(x_t, t, density_hint=density_pred)
     p_x1 = sigmoid(logit)
     rate_01, rate_10 = CTMC_rates(...)
     rate_01 = rate_01 * clamp(2 * density_pred, max=1)  ← 密度阻尼
     x_t = apply flips
4. Greedy projection → 有效接触图
```

---

## 五、预期改进

| 指标 | v4 | v5 预期 | 改进来源 |
|------|:--:|:-------:|----------|
| bpRNA F1 | 0.638 | **0.70+** | 低密度过预测大幅减少 |
| ArchiveII F1 | 0.870 | **0.89+** | FM 信息利用率提升 |
| PDB_TS1 F1 | 0.707 | **0.73+** | 输出精修 + FM 宽度 |
| 低密度(ppb<0.3) avg F1 | 0.34 | **0.55+** | 三重防线 |
| 过预测 3x+ 样本数 | 118 | **<30** | 密度引导采样 |

---

## 六、训练计划与实时追踪

### 6.1 训练命令

```bash
cd /root/aigame/dannyyan/RNADiffFold/symfold
source /root/aigame/dannyyan/miniconda3/bin/activate RNADiffFold_torch260
bash scripts/run_train_v5.sh
```

当前任务名：`260525-161400-v5-train`

### 6.2 每 20 epoch 完整 eval

`train/train_v5.py` 已集成 full eval：

```json
{
  "full_eval_enabled": true,
  "full_eval_every": 20,
  "full_eval_sets": "bpRNA,RNAStrAlign,bpRNA-new,ArchiveII,PDB_TS1,PDB_TS2,PDB_TS3,PDB_TS_hard",
  "full_eval_vis_samples_per_set": 3
}
```

每次完整 eval 会输出：

| 文件 | 说明 |
|------|------|
| `output/260525-161400-v5-train/full_eval/e020/full_eval_e020.json` | 第 20 epoch 完整指标 JSON |
| `output/260525-161400-v5-train/full_eval/e020/FULL_EVAL_REPORT_e020.md` | 第 20 epoch Markdown 报告 |
| `output/260525-161400-v5-train/full_eval/e020/full_eval_f1_bar_e020.png` | 当前 epoch 各数据集 F1 柱状图 |
| `output/260525-161400-v5-train/full_eval/e020/vis/*.png` | 每个数据集若干 GT/Pred 可视化 |
| `output/260525-161400-v5-train/full_eval/full_eval_history.json` | 所有完整 eval 的历史 |
| `output/260525-161400-v5-train/full_eval/full_eval_f1_trend.png` | 完整 eval F1 趋势图 |

同时 `output/260525-161400-v5-train/curves.png` 会额外包含 `Full Eval Avg F1` 曲线。
