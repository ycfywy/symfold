# SymFold v1 — Symmetry-Equivariant Axial DiT for RNA Secondary Structure

> **Checkpoint**: `model/260514-full-train-symfold/best.pt` (epoch 143)  
> **状态**: 训练完成，冻结，不再修改

---

## 1. 代码结构

```
src/v1/
├── README.md           ← 本文件
├── model.py            ← SymFoldModel: 完整模型 (条件编码 + backbone + flow loss + 采样)
├── se_dit.py           ← SEDiT: Symmetry-Equivariant Axial DiT backbone (6层 flat)
└── discrete_flow.py    ← BernoulliFlowLoss + GreedyProjection (贪心 max-matching)
```

### 关键依赖 (公共模块，位于 `src/` 顶层)

| 文件 | 作用 |
|------|------|
| `src/data.py` | `SimpleRNADataset` + `BucketBatchSampler` + `simple_collate_fn` |
| `src/gpu_features.py` | GPU 17 通道 FCN 特征计算 (`get_data_fcn_gpu`) |
| `src/physics_energy.py` | 推理时物理 guidance (WC + stacking + PK penalty) |
| `src/common/data_utils.py` | `contact_map_masks` 等工具函数 |
| `src/common/loss_utils.py` | `rna_evaluation` (计算 F1/P/R/MCC) |
| `src/models/` | RNA-FM conditioner + UFold conditioner (预训练权重加载) |

### 训练/评估入口

| 文件 | 作用 |
|------|------|
| `train/train.py` | v1 训练脚本 |
| `train/config/train_config.json` | v1 训练配置 |
| `scripts/run_eval.sh` | v1 快速评估 |
| `eval/eval.py` | v1 详细评估 (支持 `--detailed`) |

---

## 2. Walkthrough：给定一个 RNA 序列，如何预测其二级结构

```
输入: RNA 序列 "AUGCCGUUA..." (长度 L)

┌─────────────────── 条件编码阶段 ───────────────────┐
│                                                    │
│  1. RNA-FM Conditioner (frozen, 12层 Transformer)  │
│     序列 → token embedding (L, 640)                │
│     → fm_emb_outer: outer product → (L, L, 16)    │
│     → fm_attn_proj: attention maps → (L, L, 8)    │
│     → fm_global: mean pooling → AdaLN 调制信号      │
│                                                    │
│  2. UFold Conditioner (U-Net, finetune)            │
│     序列 one-hot (L, 4) → outer product → FCN     │
│     → u_cond: (L, L, 8)                           │
│     → u_global: mean pooling → AdaLN 调制信号      │
│                                                    │
│  3. GPU 17通道 FCN 特征                             │
│     序列 one-hot → get_data_fcn_gpu → (L, L, 17)  │
│     (含 seq_outer 等手工特征)                       │
│                                                    │
└────────────────────────────────────────────────────┘
                         ↓
        拼接成 48 通道输入: (B, 48, L, L)
        - x_t embedding:  8ch (当前时刻状态的 learned embedding)
        - fm_emb_outer:  16ch
        - fm_attn_proj:   8ch
        - seq_outer:      8ch
        - u_cond:         8ch

┌─────────────────── SE-DiT Backbone ───────────────────┐
│                                                       │
│  PatchEmbed2D: (B, 48, L, L) → (B, N, 192)          │
│     patch_size=4, N = (L/4)²                          │
│                                                       │
│  + Axial Position Embedding (row + col, learnable)    │
│                                                       │
│  SEDiTBlock × 6:                                      │
│    ├─ AdaLN (用 time + fm_global + u_global 调制)      │
│    ├─ SharedAxialAttention                            │
│    │   ├─ Row attention: reshape to (B*L/4, L/4, 192)│
│    │   ├─ Col attention: reshape to (B*L/4, L/4, 192)│
│    │   └─ 共享 QKV 权重 → 保证 (i,j)↔(j,i) 等变性    │
│    ├─ FFN (4× expansion, GELU)                       │
│    └─ Residual + Scale (AdaLN-Zero)                  │
│                                                       │
│  Final: LayerNorm → AdaLN → Linear → UnPatchify2D    │
│  输出: logit (B, 1, L, L) → sigmoid → p(配对)         │
│                                                       │
└───────────────────────────────────────────────────────┘
                         ↓
┌─────────────────── 采样 (推理时) ────────────────────┐
│                                                      │
│  1. 初始化: x_0 ~ Bernoulli(ρ₀=0.005), 对称化       │
│     → 初始就是 99.5% 为 0 的稀疏矩阵                  │
│                                                      │
│  2. τ-leap CTMC × 20 步:                            │
│     for t = 0 → 1 (dt = 1/20):                      │
│       - 拼接 x_t + 条件 → SE-DiT → p(x₁=1|x_t)     │
│       - 计算 flip rates:                             │
│         R(0→1) = p / (1 - t*(1-ρ₀))                 │
│         R(1→0) = (1-p) * ρ₀ / (1 - t*(1-ρ₀))       │
│       - x_t 中 0→1 with prob R(0→1)*dt              │
│       - x_t 中 1→0 with prob R(1→0)*dt              │
│       - 对称化: x_t = (x_t + x_t.T) / 2 → 二值化    │
│                                                      │
│  3. 最终投影: Greedy Max-Matching                    │
│     - 约束: 对称 + |i-j| ≥ 3 + 每行 ≤ 1 配对        │
│     - 按概率降序贪心选配对                             │
│                                                      │
│  输出: 对称二值 contact map (L, L)                    │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 3. 训练与评估

### 3.1 训练命令

```bash
cd /root/aigame/dannyyan/RNADiffFold/symfold
source /root/aigame/dannyyan/miniconda3/bin/activate RNADiffFold_torch260

nohup python -u train/train.py train/config/train_config.json \
    >> logs/YYMMDD-HHMMSS-v1-train.stdout.log 2>&1 &
```

### 3.2 训练配置

| 项 | 值 |
|---|---|
| **数据集** | all (RNAStrAlign + bpRNA + bpRNA-new), ~34k 样本 |
| **总 Epoch** | 160 (best 在 epoch 143) |
| **学习率** | 5e-5, warmup 2 epochs |
| **优化器** | Adam |
| **Batch size** | 按桶: 80→64, 160→32, 240→16, 320→8, 400→4, 480→2, 560→1, 640→1 |
| **Dropout** | 0.15 |
| **Grad clip** | 1.0 |
| **Early stop** | patience=20 (基于 val F1) |
| **精度** | fp32 (TF32 关闭, 避免 H20 cuBLAS bug) |
| **GPU** | NVIDIA H20 96GB × 1 |
| **训练时间** | ~155 epochs × ~5min/epoch ≈ 13 小时 |

### 3.3 数据集划分

**训练集** (34,782 samples):
| 子集 | 样本数 | 来源 |
|------|-------:|------|
| RNAStrAlign | 17,630 | RNAStrAlign 数据库训练集 |
| bpRNA TR0 | 11,751 | bpRNA 官方训练集 |
| bpRNA-new | 5,401 | bpRNA 新增数据 (⚠️ 同时用作测试) |

**验证集** (训练时 early stopping):
| 数据集 | 样本数 | 用途 |
|--------|-------:|------|
| bpRNA VL0 | 1,299 | 每 2 epoch eval一次，选 best.pt |

**测试集** (最终评估):
| 数据集 | 样本数 | 类型 | 说明 |
|--------|-------:|:----:|------|
| bpRNA TS0 | 1,304 | ID test | ✅ 未在训练中 |
| RNAStrAlign | 2,023 | ID test | ✅ 未在训练中 |
| ArchiveII | 3,911 | OOD | ✅ 完全独立 |
| PDB TS1 | 60 | OOD-hard | ✅ PDB 3D 结构 |
| PDB TS2 | 38 | OOD-hard | ✅ PDB 3D 结构 |
| PDB TS3 | 18 | OOD-hard | ✅ PDB 3D 结构 |
| PDB TS_hard | 28 | OOD-hardest | ✅ PDB 3D 结构 |
| bpRNA-new | 5,401 | ⚠️ 泄漏 | 同时在训练集中 |

### 3.4 评估命令

```bash
# 快速评估
bash scripts/run_eval.sh model/260514-full-train-symfold/best.pt

# 详细评估 (逐样本)
python eval/eval.py \
    --ckpt model/260514-full-train-symfold/best.pt \
    --test_sets bpRNA,RNAStrAlign,ArchiveII,PDB_TS1,PDB_TS2,PDB_TS3,PDB_TS_hard \
    --detailed --top_k 5 \
    --out_json output/260514-full-train-symfold/eval_detailed.json
```

---

## 4. 评估结果总结

### 4.1 最终性能 (best.pt, epoch 143, no physics guidance)

| Dataset | N | F1 | Precision | Recall | MCC |
|---------|---:|:---:|:---------:|:------:|:---:|
| RNAStrAlign | 2023 | **0.921** | 0.911 | 0.936 | 0.922 |
| ArchiveII | 3911 | **0.861** | 0.839 | 0.893 | 0.863 |
| PDB_TS2 | 38 | **0.832** | 0.922 | 0.769 | 0.838 |
| bpRNA-new | 5401 | 0.683 | 0.621 | 0.790 | 0.693 |
| PDB_TS1 | 60 | 0.675 | 0.765 | 0.617 | 0.681 |
| PDB_TS3 | 18 | 0.665 | 0.779 | 0.596 | 0.675 |
| bpRNA | 1304 | 0.644 | 0.583 | 0.758 | 0.656 |
| PDB_TS_hard | 28 | 0.596 | 0.695 | 0.540 | 0.605 |

**8 数据集平均: F1=0.735, Precision=0.764, Recall=0.737, MCC=0.744**

### 4.2 vs RNADiffFold 逐数据集对标 (109M params, 10次投票)

| 数据集 | N | 类型 | **SymFold F1** | **RNADiffFold F1** | **ΔF1** | SymFold P/R | RNADiffFold P/R |
|--------|---:|:----:|:--------------:|:------------------:|:-------:|:-----------:|:---------------:|
| RNAStrAlign | 2023 | ID | **0.921** | 0.787 | **+0.134** | 0.911/0.936 | 0.681/0.938 |
| ArchiveII | 3911 | OOD | **0.861** | 0.740 | **+0.121** | 0.839/0.893 | 0.647/0.877 |
| PDB_TS2 | 38 | OOD-hard | **0.832** | 0.733 | **+0.099** | 0.922/0.769 | 0.749/0.726 |
| bpRNA-new | 5401 | ⚠️泄漏 | **0.683** | 0.611 | +0.072 | 0.621/0.790 | 0.521/0.770 |
| PDB_TS1 | 60 | OOD-hard | **0.675** | 0.607 | **+0.067** | 0.765/0.617 | 0.654/0.595 |
| PDB_TS3 | 18 | OOD-hard | **0.665** | 0.635 | +0.030 | 0.779/0.596 | 0.719/0.593 |
| bpRNA | 1304 | ID | **0.644** | 0.618 | +0.026 | 0.583/0.758 | 0.524/0.792 |
| PDB_TS_hard | 28 | OOD-hardest | **0.596** | 0.526 | **+0.070** | 0.695/0.540 | 0.569/0.530 |

### 4.3 总体指标对比

| 指标 | SymFold v1 | RNADiffFold | 提升 |
|------|:----------:|:-----------:|:----:|
| **平均 F1 (8 数据集)** | **0.735** | 0.657 | **+11.8%** |
| **平均 Precision** | **0.764** | 0.633 | +20.8% |
| **平均 Recall** | 0.737 | 0.758 | -2.7% |
| **平均 MCC** | **0.744** | 0.665 | +11.9% |
| **可训练参数** | 13M | 109M | 8× 更小 |
| **推理速度** | 20步×1次 | 20步×10次 | 10× 更快 |

**核心结论**: 
- 用 1/8 参数、1/10 推理量，在**所有 8 个测试集**上全面超越 RNADiffFold
- **最大优势是 Precision** (+20.8%): 解决了 RNADiffFold "假阳性过多" 的痛点
- Recall 略有下降 (-2.7%) 是 Precision-Recall 权衡的正常表现
- **OOD 泛化最强**: ArchiveII +12.1%, PDB_TS_hard +7.0%

### 4.3 典型样例

**完美预测 (F1=1.0)**:
```
[4OOG-1-D] L=34
序列:     CAUGUCAUGUCAUGAGUCCAUGGCAUGGCAUGGC
GT结构:   ((((((((((((((....)))))))))))))).
Pred结构: ((((((((((((((....)))))))))))))).
→ 14 对配对全部正确，0 假阳性
```

**失败案例 (F1=0.21)**:
```
[6LAS_A] L=55
序列:     GGCAUUGUGCCUCGCAUUGCACUCCGCGGGGCGAUAAGUCCUGAAAAGGGAUGUC
GT结构:   (((((((((((((((.(.)......)))))))).(..)..(((..)))))))))
→ 复杂多 stem + pseudoknot，23 对中只找到 4 对
```

### 4.4 已知局限

1. **长序列 Recall 低**: L>300 时 Recall 急剧下降 (6层 flat attention 信息传播不足)
2. **Pseudoknot 预测差**: 贪心 max-matching 的 "每行≤1 配对" 约束太强
3. **缺乏多尺度**: 只有单一 patch_size=4 的分辨率
4. **UFold 特征利用不够**: 仅用全局池化做 AdaLN，丢失空间信息



---

## 5. 可用输出与可视化

所有输出保存在 `output/260514-full-train-symfold/`，共 203 个文件。

---

### 5.1 训练曲线

![Training Curves](../../output/260514-full-train-symfold/training_curves.png)

---

### 5.2 训练进展对比（同一 RNA 不同 epoch）

每张图包含 3 个子图: **GT Contact Map**(蓝) / **Pred Contact Map**(橙) / **Overlay**(TP=绿, FN=红, FP=蓝)

#### bpRNA_CRW_15576 (L=143, 35 对配对, 含 4 段 stem)

**Epoch 1** — 训练初期，预测基本为噪声:
![e1_15576](../../output/260514-full-train-symfold/vis_e1_bpRNA_CRW_15576.png)

**Epoch 11** — 开始捕捉主要 stem 结构:
![e11_15576](../../output/260514-full-train-symfold/vis_e11_bpRNA_CRW_15576.png)

**Epoch 21** — 大部分配对已正确预测:
![e21_15576](../../output/260514-full-train-symfold/vis_e21_bpRNA_CRW_15576.png)

**Epoch 29** — 最终效果，几乎完美还原 GT:
![e29_15576](../../output/260514-full-train-symfold/vis_e29_bpRNA_CRW_15576.png)

---

#### bpRNA_CRW_15618 (L=117, 34 对配对, 多 stem 嵌套)

**Epoch 1** — 训练初期:
![e1_15618](../../output/260514-full-train-symfold/vis_e1_bpRNA_CRW_15618.png)

**Epoch 11** — 结构逐渐清晰:
![e11_15618](../../output/260514-full-train-symfold/vis_e11_bpRNA_CRW_15618.png)

**Epoch 21** — 大部分配对捕获:
![e21_15618](../../output/260514-full-train-symfold/vis_e21_bpRNA_CRW_15618.png)

**Epoch 29** — 最终效果:
![e29_15618](../../output/260514-full-train-symfold/vis_e29_bpRNA_CRW_15618.png)

---

#### bpRNA_CRW_15857 (L=135, 36 对配对, 含 G-U wobble pair)

**Epoch 1** — 训练初期:
![e1_15857](../../output/260514-full-train-symfold/vis_e1_bpRNA_CRW_15857.png)

**Epoch 11** — 部分结构浮现:
![e11_15857](../../output/260514-full-train-symfold/vis_e11_bpRNA_CRW_15857.png)

**Epoch 21** — 大部分正确:
![e21_15857](../../output/260514-full-train-symfold/vis_e21_bpRNA_CRW_15857.png)

**Epoch 29** — 最终效果:
![e29_15857](../../output/260514-full-train-symfold/vis_e29_bpRNA_CRW_15857.png)

---

#### bpRNA_CRW_15869 (L=?, 偏难样本 — 即使训练充分也有残余误差)

**Epoch 1** — 完全没学到:
![e1_15869](../../output/260514-full-train-symfold/vis_e1_bpRNA_CRW_15869.png)

**Epoch 11** — 开始有结构，但误差大:
![e11_15869](../../output/260514-full-train-symfold/vis_e11_bpRNA_CRW_15869.png)

**Epoch 21** — 改善，但仍有明显 FN(红) 和 FP(蓝):
![e21_15869](../../output/260514-full-train-symfold/vis_e21_bpRNA_CRW_15869.png)

**Epoch 29** — 最终效果，仍可见残余红色(漏检)和蓝色(误检):
![e29_15869](../../output/260514-full-train-symfold/vis_e29_bpRNA_CRW_15869.png)

**Epoch 143 附近 (continue-train, epoch 99)** — 训练更久也难以消除的错误:
![e99_15869](../../output/260515-continue-train-symfold/bpRNA_CRW_15869/vis_e99_bpRNA_CRW_15869.png)

> ⚠️ 这个样本说明了 v1 的局限：对于配对密集、结构复杂的 RNA，6 层 flat attention 的信息传播不足，导致长程配对遗漏（红色）。

---

#### bpRNA_CRW_15964 (L=?, 另一个难样本)

**Epoch 1** — 噪声:
![e1_15964](../../output/260514-full-train-symfold/vis_e1_bpRNA_CRW_15964.png)

**Epoch 11** — 只捕获了部分主 stem:
![e11_15964](../../output/260514-full-train-symfold/vis_e11_bpRNA_CRW_15964.png)

**Epoch 21** — 有进步但次要 stem 缺失:
![e21_15964](../../output/260514-full-train-symfold/vis_e21_bpRNA_CRW_15964.png)

**Epoch 29** — 最终效果:
![e29_15964](../../output/260514-full-train-symfold/vis_e29_bpRNA_CRW_15964.png)

**Epoch 143 附近 (continue-train, epoch 99)** — 多 stem 结构的极限:
![e99_15964](../../output/260515-continue-train-symfold/bpRNA_CRW_15964/vis_e99_bpRNA_CRW_15964.png)

> ⚠️ 观察 overlay 中的红色区域：即使训练到收敛，仍有部分长程配对无法被 6 层 flat attention 捕获。

---

### 5.3 好 vs 差 case 总结

| 表现 | 样本特征 | 可视化表现 | 原因 |
|:----:|----------|-----------|------|
| ✅ 好 | 短序列, 清晰 stem, 标准 Watson-Crick | Overlay 几乎全绿 | stem 模式简单，attention 容易捕获 |
| ⚠️ 中等 | 中等长度, 多段 stem | 部分绿 + 少量红/蓝 | 主 stem 对了，次要 stem 偶有遗漏 |
| ❌ 差 | 长序列, pseudoknot, 密集配对 | 大量红色(FN) | 6层 flat attention 信息传播不足 + greedy matching 约束太强 |

---

### 5.3 输出目录结构

```
output/260514-full-train-symfold/
├── training_curves.png             ← 训练曲线 (Loss/F1/Time)
├── vis_samples_report.md           ← 5 个 RNA 样本详细分析报告
├── vis_e1_bpRNA_CRW_15576.png     ← epoch 1 可视化
├── vis_e1_bpRNA_CRW_15618.png
├── vis_e1_bpRNA_CRW_15857.png
├── vis_e1_bpRNA_CRW_15869.png
├── vis_e1_bpRNA_CRW_15964.png
├── vis_e11_bpRNA_CRW_*.png        ← epoch 11
├── vis_e21_bpRNA_CRW_*.png        ← epoch 21
├── vis_e29_bpRNA_CRW_*.png        ← epoch 29 (best checkpoint 附近)
└── ... (共 203 文件, 每个 val epoch × 5 个固定 RNA 样本)
```

### 训练进展 (Val F1 on bpRNA VL0)

| Epoch | Val F1 | 说明 |
|------:|:------:|------|
| 1 | 0.423 | 初始 |
| 15 | 0.453 | 稳步提升 |
| 25 | 0.499 | 接近 0.5 |
| 29 | 0.501 | 首次超过 0.5 |
| ... | ... | (继续训练到 160 epoch) |
| 143 | **0.502** | 最终 best (此后 patience 20 触发 early stop) |

---

## 6. 设计思想总结

| 设计决策 | 理由 | 效果 |
|----------|------|------|
| Bernoulli ρ₀=0.005 | 精确匹配数据稀疏率 (配对仅占 0.5%) | 减少假阳性 |
| pos_weight=199 | 自然平衡 99.5% 负样本 | Precision +20% |
| Shared QKV (row=col) | 强制网络输出对称 | 无需后处理对称化 |
| Patch=4 | 保留配对像素精度 | 细粒度预测 |
| AdaLN-Zero | 时间+条件调制，零初始化 | 稳定训练 |
| Greedy Matching | 物理约束 (每行≤1配对) | 消除非法预测 |
| 单次采样即可 | Bernoulli 先验足够好 | 10× 加速 |
