# SymFold v4 — Multi-Layer FM + Triangle Update + Adaptive Density Loss

## 核心改进 (vs v3)

### 1. Multi-Layer RNA-FM Feature Extraction

v3 只取 RNA-FM 第12层（最后一层）的 embedding。v4 提取 **4 层** (3, 6, 9, 12) 的 representation，
用 learnable weighted fusion 融合：

```
Layer 3  (浅层): 局部序列 motif (k-mer, base composition)
Layer 6  (中浅): 局部结构倾向 (stem seeds, loop regions)
Layer 9  (中深): 中程依赖 (hairpin loops, internal loops)
Layer 12 (深层): 全局折叠语义 (domain-level folding)
```

**融合方式**:
- Softmax-weighted scalar combination (learnable per-layer weights)
- Per-layer linear projection → concat → MLP fusion
- Residual connection for stability

**动机**: RNA-FM 不同层捕获不同粒度的信息。浅层对局部碱基组成敏感，
深层对全局结构更敏感。融合多层相当于数据增强 + 更丰富的特征。

### 2. Triangle Multiplicative Update (后3层)

受 AlphaFold2 启发，在 backbone 的后 3 层 (layer 6, 7, 8) 插入 Triangle Update:

```
z[i,j] += gate * Σ_k proj_left(z[i,k]) * proj_right(z[k,j])
```

**作用**: 显式捕获三体约束 —— 如果 i 已经与 k 配对（z[i,k] 高），
则 i 与其他 j 的配对概率应该降低。这比 v3 的 NonCrossingLoss 更直接有效。

### 3. Adaptive Density-Aware Loss

v3 的核心问题：对低密度 RNA (pair_per_base < 0.2) 严重过预测。
原因是固定 pos_weight=199 对所有样本一视同仁。

v4 引入:
- **Per-sample adaptive pos_weight**: 低密度样本用更低的 pos_weight (50-100)
- **Focal modulation**: 下调 easy negatives 的权重，聚焦 hard examples
- **Density regression head**: 辅助任务预测配对密度，指导投影阶段

### 4. Gated FFN (SwiGLU)

替代 v3 的 GELU FFN:
```python
# v3: x → Linear → GELU → Linear
# v4: x → SiLU(Linear1) * Linear2 → Linear3  (SwiGLU)
```
参数效率更高，在相同 FLOPs 下表现更好。

## 架构对比

| 组件 | v3 | v4 |
|------|----|----|
| RNA-FM layers | 第12层 only | **layers [3,6,9,12]** fused |
| FM fusion | 无 | **Learnable weighted + MLP** |
| FFN | GELU | **SwiGLU (gated)** |
| Triangle Update | 无 | **后3层 (layer 6-8)** |
| pos_weight | 固定 199 | **自适应 50-199** |
| Focal Loss | 无 | **γ=1.0** |
| Density Head | 无 | **辅助回归任务** |
| 其余 | Dilated Axial + RoPE + QK-Norm + FiLM | 保留 |

## 参数量预估

| 组件 | v3 params | v4 params | Δ |
|------|:---------:|:---------:|:-:|
| RNA-FM (frozen) | 99.5M | 99.5M | 0 |
| Multi-Layer Fusion | 0 | ~0.16M | +0.16M |
| UFold | ~0.5M | ~0.5M | 0 |
| Backbone (9 blocks) | 13.2M | ~15.5M | +2.3M |
| Triangle Update (×3) | 0 | ~0.8M | +0.8M |
| Density Head | 0 | ~33K | +33K |
| **Total trainable** | **21.8M** | **~25.1M** | **+3.3M** |

## 预期效果

根据 V3_CASE_ANALYSIS 的根因分析:

1. **bpRNA F1 提升**: Adaptive pos_weight + Focal → 低密度过预测问题缓解
   - 预期: bpRNA F1 0.636 → 0.70+
2. **PDB 系列 F1 提升**: Multi-layer FM 提供更丰富的序列上下文
   - 预期: PDB_TS_hard 0.634 → 0.67+
3. **整体 F1**: Triangle Update 强化结构一致性
   - 预期: avg F1 0.752 → 0.79+

## 文件结构

```
src/v4/
├── __init__.py
├── model.py         # SymFoldModel_v4
├── da_se_dit.py     # DASEDiT_v4 backbone
├── discrete_flow.py # Adaptive loss + sampling
└── README.md        # 本文件
```

## 训练

```bash
cd /root/aigame/dannyyan/RNADiffFold/symfold
bash scripts/run_train_v4.sh
# 或手动:
python -u train/train_v4.py train/config/train_config_v4.json
```
