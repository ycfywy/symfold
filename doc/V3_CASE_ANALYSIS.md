# V3 Eval Case Analysis — 表现好/差样本的特征分析

> 基于 v3 best.pt (epoch 73) 在 7 个测试集上的 eval 详细结果
> 分析脚本: `scripts/analyze_eval_cases.py`
> 数据源: `output/260520-v3-train/eval_detailed.json`

---

## 一、核心发现

### 1.1 决定 F1 的最关键因素: **Pair Density (配对密度)**

| 指标 | 与 F1 的相关性 (bpRNA) | 说明 |
|------|:---------------------:|------|
| **Pair per base** | **r=+0.675** | ✅ 最强正相关 |
| Pred/GT ratio | r=-0.499 | 过度预测 → F1 差 |
| GT pairs 数量 | r=+0.257 | 配对越多越好 |
| 序列长度 | r=-0.057 | 几乎无关 |

**关键洞察**: 模型对 **低配对密度** 的 RNA (pair_per_base < 0.3) 表现极差。这些 RNA 大部分碱基未配对 (loop 多, stem 少)，模型倾向于过度预测配对。

### 1.2 过度预测是主要失败模式

| 数据集 | 过度预测率 | 过度预测 F1 | 正常预测 F1 |
|--------|:----------:|:-----------:|:-----------:|
| bpRNA | **42.9%** | 0.448 | 0.778 |
| RNAStrAlign | 6.2% | 0.737 | 0.951 |
| ArchiveII | 12.2% | 0.601 | 0.901 |

bpRNA 上近一半样本被过度预测 (pred > 1.5× gt)！

### 1.3 PDB_TS_hard: 高配对密度 + 模型欠预测

PDB_TS_hard 的失败模式与 bpRNA 相反:
- Bad samples 的 pair_per_base = **1.015** (极高密度, 含 pseudoknots)
- 模型 pred_ratio = **0.672** (欠预测)
- 这些 RNA 含有大量非标准配对 (pseudoknots, base triples)，模型未学到

---

## 二、各数据集分组分析

### 2.1 bpRNA (N=1304, avg F1=0.636)

| 分组 | 数量 | 占比 | 特征 |
|------|:----:|:----:|------|
| Good (F1≥0.8) | 397 | 30.4% | 短序列 (119), 高密度 (ppb=0.51), pred/gt≈1.15 |
| Medium (0.5~0.8) | 562 | 43.1% | 中等长度, 中密度 |
| Bad (F1<0.5) | 345 | **26.5%** | 中等长度 (136), **低密度 (ppb=0.27)**, 严重过预测 (pred/gt=**2.97**) |
| Zero (F1=0) | ~10 | ~0.8% | 极低密度 (ppb<0.2), 全部预测错误 |

**Bad 样本的核心特征**:
1. **低配对密度** (pair_per_base=0.27 vs good=0.51): 大部分碱基是 unpaired loop
2. **严重过度预测** (pred/gt=2.97): 模型预测了 3 倍于 GT 的配对数
3. **Precision 极低** (0.215 vs good=0.854): 大量假阳性
4. **Recall 尚可** (0.481): 真配对大部分找到了，但淹没在假阳性中

### 2.2 RNAStrAlign (N=2023, avg F1=0.939)

| 分组 | 数量 | 占比 | 特征 |
|------|:----:|:----:|------|
| Good (F1≥0.8) | 1831 | **90.5%** | 高密度 (ppb≈0.55), pred/gt≈1.04 |
| Bad (F1<0.5) | 82 | 4.1% | 长序列 (205), 低密度 (ppb=0.27) |
| Perfect (F1=1) | 674 | **33.3%** | 标准 tRNA/rRNA 结构 |

RNAStrAlign 表现好因为: 数据集本身配对密度高 (多是标准 tRNA, 5S rRNA)。

### 2.3 ArchiveII (N=3911, avg F1=0.864)

| 分组 | 数量 | 占比 | 特征 |
|------|:----:|:----:|------|
| Good (F1≥0.8) | 2883 | 73.7% | 标准结构 RNA |
| Bad (F1<0.5) | 299 | 7.6% | 低密度, 长序列, 过预测 |

### 2.4 PDB 系列

**PDB_TS_hard (N=28, avg F1=0.634) — 失败模式不同**:
- Bad samples 有 **高** pair_per_base (1.015): 含 pseudoknots/base triples
- 模型 **欠预测** (pred_ratio=0.672): 对非标准配对漏检
- 这些 RNA 来自 3D 结构，含有模型从未见过的复杂配对模式

---

## 三、F1 与序列长度的关系

| 长度区间 | bpRNA F1 | ArchiveII F1 | RNAStrAlign F1 |
|:--------:|:--------:|:------------:|:--------------:|
| [0, 80) | 0.724 | 0.917 | 0.970 |
| [80, 160) | 0.605 | 0.855 | 0.940 |
| [160, 240) | 0.590 | 0.838 | 0.927 |
| [240, 320) | 0.646 | 0.870 | 0.923 |
| [320, 480) | 0.646 | 0.853 | 0.908 |
| [480, 640) | 0.662 | 0.848 | 0.889 |

**观察**: 
- 长度对 F1 影响较小 (r≈-0.06)
- 80-160 区间 F1 偏低，可能因为这个区间训练 batch 最大，样本多样性导致

---

## 四、F1 与配对密度的关系 (最关键)

对 bpRNA 按 pair_per_base 分 bin:

| Pair/Base | N | Mean F1 | Bad Rate |
|:---------:|---:|:-------:|:--------:|
| [0, 0.2) | 104 | 0.243 | **76.9%** |
| [0.2, 0.4) | 471 | 0.503 | 40.3% |
| [0.4, 0.6) | 495 | 0.736 | 13.3% |
| [0.6, 0.8) | 203 | 0.845 | 3.9% |
| [0.8, 1.0) | 31 | 0.884 | 0.0% |

**结论**: 
- pair_per_base < 0.2: **76.9% 的样本 F1<0.5**！几乎全军覆没
- pair_per_base ≥ 0.6: 只有 3.9% 的样本 F1<0.5，表现优秀
- 这是一个 **3× 的 F1 差距** (0.243 vs 0.884)

---

## 五、失败样本的共同模式

### 模式 1: 低密度 + 过度预测 (bpRNA 的主要问题)

```
典型失败 case: bpRNA_RFAM_11730
- Length=125, GT pairs=2 (只有 2 个配对！)
- Pred pairs=43 (预测了 43 个！)
- pair_per_base = 0.03 (极低)
- F1=0.0

原因: 这个 RNA 几乎全是 loop (只有 1 个极短的 stem)
      模型的 pos_weight=199 鼓励预测配对 → 对几乎全 unpaired 的序列严重过预测
```

### 模式 2: 非标准配对 + 欠预测 (PDB_TS_hard 的主要问题)

```
典型失败 case: 5NWQ-1-A  
- Length=41, GT pairs=29 (pair_per_base=1.41, 超过 1.0！)
- Pred pairs=16 (只预测了 55%)
- F1=0.311

原因: pair_per_base > 1.0 意味着存在大量 pseudoknots (每个碱基平均参与 >1 个配对)
      模型的 strict projection 强制每行至多 1 个配对 → 天然无法处理 pseudoknots
```

### 模式 3: Precision 极低 (最常见)

```
bpRNA_RFAM_26260: L=175, GT=7, Pred=47, F1=0.0
- 7 个真配对，模型预测了 47 个，但一个都没对上
- 可能是 RNA 的实际配对位置极不典型 (非 Watson-Crick)
```

---

## 六、根因分析

### 6.1 为什么低密度 RNA 表现差？

1. **pos_weight=199 的偏差**: 训练时正样本权重 199×，鼓励模型预测更多 1 → 对低密度序列严重过预测
2. **训练数据分布**: 训练集平均 pair_per_base ≈ 0.4-0.5，低密度 (ppb<0.2) 样本占比小
3. **projection 不适应**: greedy max-matching 会尽可能多地配对 → 对低密度 RNA 不友好
4. **模型缺乏 "unpaired prior"**: 当序列大部分应该 unpaired 时，模型没有机制表达这一点

### 6.2 为什么 PDB_TS_hard 表现差？

1. **Pseudoknots**: strict projection 每行至多 1 配对，天然无法表达交叉配对
2. **非 Watson-Crick 配对**: 3D 结构中存在 GU wobble 以外的非标准配对 (Hoogsteen 等)
3. **OOD**: 训练集全是从序列预测的 2D 结构标注，PDB 是 3D 结构提取，标注标准不同
4. **小样本**: 仅 28 个样本，统计波动大

### 6.3 为什么 bpRNA 比 RNAStrAlign/ArchiveII 差？

| | bpRNA | RNAStrAlign | ArchiveII |
|-|:-----:|:-----------:|:---------:|
| Avg pair/base | 0.38 | 0.55 | 0.48 |
| Low-density (ppb<0.3) | 44% | 12% | 18% |
| 数据来源 | Rfam 多样化 | tRNA/rRNA 为主 | 标准 RNA |

bpRNA 含有大量来自 Rfam 的非标准 RNA (lncRNA, riboswitch 等)，配对密度普遍偏低。

---

## 七、改进建议 (基于本分析)

### 7.1 解决低密度过预测 (最高优先)

**方案 A: Adaptive pos_weight per sample**
```python
# 根据序列特征动态调整 pos_weight
# 低密度序列用更小的 pos_weight
estimated_density = ufold_prior.sum() / (L * L)  # UFold 给的 prior
pos_weight_adaptive = base_pos_weight * (estimated_density / 0.005)
```

**方案 B: 加入 "配对数量回归" 辅助任务**
```python
# 额外预测总配对数 → 约束投影阶段
num_pairs_pred = model.predict_num_pairs(fm_emb)  # 回归头
# projection 时限制最大配对数 = num_pairs_pred * 1.2
```

**方案 C: 投影时引入 density-aware threshold**
```python
# 当前: 贪心选所有 score > 0 的配对
# 改进: 根据预估密度设定 minimum score threshold
if estimated_low_density:
    threshold = 0.7  # 更严格
else:
    threshold = 0.3  # 正常
```

### 7.2 解决 PDB_TS_hard 欠预测

**方案**: 允许 relaxed projection (每行至多 2 配对) 用于 PDB 评估
```python
# 评估时: 如果检测到高密度，使用 relaxed 模式
if pred_density > 0.8:
    project_mode = 'relaxed_2'  # 每行至多 2
```

### 7.3 训练数据重采样

```python
# 低密度样本权重提高 (当前被高密度样本主导)
sample_weight = 1.0 / max(pair_per_base, 0.1)  # 低密度权重更高
```

---

## 八、可复用分析命令

```bash
# 运行完整 case 分析
python scripts/analyze_eval_cases.py output/260520-v3-train/eval_detailed.json

# 输出:
#   - 控制台: 统计摘要
#   - output/260520-v3-train/case_analysis.json: 详细 JSON
#   - output/260520-v3-train/case_analysis_plots.png: 可视化图

# 对比两次 eval 结果:
# python scripts/analyze_eval_cases.py output/<new_eval>/eval_detailed.json
```

---

## 九、总结

| 失败类型 | 占比 | 根因 | 修复优先级 |
|----------|:----:|------|:----------:|
| 低密度过预测 | ~43% (bpRNA) | pos_weight=199 + 无 density prior | ★★★ |
| 非标准配对漏检 | ~29% (PDB) | strict projection + OOD | ★★☆ |
| 长序列精度下降 | ~10% | 注意力稀释 | ★☆☆ |

**最大的改进杠杆**: 解决低密度 RNA 的过预测问题 (影响 bpRNA 26.5% 的样本)。如果能把 bpRNA bad rate 从 26.5% 降到 10%，bpRNA F1 预计从 0.636 → 0.75+。
