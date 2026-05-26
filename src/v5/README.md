# SymFold v5: DA-SE-DiT-v5

## 改进 vs v4

| 特性 | v4 | v5 | 改动理由 |
|------|----|----|----------|
| FM Fusion 输出 | 16D | **64D** | 压缩比 160:1→40:1，保留更多 RNA-FM 信息 |
| 输入通道 | 48ch | **96ch** | FM 2D 分支从 16ch 增至 64ch |
| 密度条件 | 仅辅助 loss | **注入全局条件 + 引导采样** | 让模型知道该预测多少对 |
| 输出精修 | 无 | **3 层 Conv 在 L×L 精修** | 修正 patch 边界伪影 |
| pos_weight_min | 50 | **20** | 进一步抑制低密度过预测 |
| focal_gamma | 1.0 | **1.5** | 更强的 hard example mining |
| density_weight | 0.1 | **0.2** | 更重视密度预测的准确性 |
| 采样策略 | 固定 | **密度自适应 rate damping** | 低密度 RNA 减少 0→1 翻转 |

## 核心设计

### 1. 密度条件注入 (训练+推理)

- **训练时**: 将 GT density (pairs/base) 作为第 4 个全局条件注入 AdaLN
- **推理时**: 先跑一次 density head 预测 density，再用预测值条件化后续采样
- **作用**: 模型不再需要"猜"该预测多少对，直接被告知目标密度

### 2. 密度引导采样

推理时根据预测密度动态调整 CTMC 翻转率：
```python
# 低密度序列 → 减少 0→1 翻转（抑制过预测）
damp = clamp(2 * density_pred, max=1.0)
rate_01 = rate_01 * damp
```

### 3. 输出精修 Conv

在 UnPatchify 后加一个轻量残差 ConvNet：
```python
self.refine = nn.Sequential(
    Conv2d(1→16, k=3), GELU,
    Conv2d(16→16, k=3), GELU,
    Conv2d(16→1, k=1),   # zero-init
)
logit = logit + refine(logit)  # 残差
```

## 文件结构

```
src/v5/
├── __init__.py
├── README.md          # 本文件
├── da_se_dit.py       # DASEDiT_v5 backbone
└── model.py           # SymFoldModel_v5 主模型
```

## 参数量

实际启动日志：

- Total params: 122,396,020
- Trainable params: 22,874,474
- 冻结 RNA-FM: ~99.5M
