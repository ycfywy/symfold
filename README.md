# SymFold — Symmetry-Constrained Discrete Flow Matching for RNA Secondary Structure

> **SymFold** puts RNA secondary structure (contact map) prediction as a generative modeling problem on **symmetric binary matrices**, solved via **Discrete Flow Matching** with a **Symmetry-Equivariant Axial DiT** backbone.

---

## Quick Overview

```
RNA sequence → [RNA-FM + UFold conditioners] → DA-SE-DiT predicts P(pair)
             → τ-leap CTMC sampling (20 steps) → strict projection → contact map
```

**Key innovations (v5, current):**
1. **Bernoulli Discrete Flow Matching** on symmetric matrices (not Gaussian diffusion)
2. **Dilated Axial SE-DiT-v5** — 9-layer flat backbone with alternating dilation (1/2/4) + Triangle Multiplicative Update (layers 6-8)
3. **Wider Multi-Layer RNA-FM Fusion** — `fm_multi_out_dim=64` (vs v4's 16), preserving fine-grained RNA-FM features
4. **Density Conditioning** — GT density injected during training; predicted density guides sampling
5. **OutputRefineConv** — 3-layer Conv residual refinement at full L×L resolution after UnPatchify
6. **Density-guided Sampling** — Suppresses over-prediction for low-density RNA
7. **Adaptive Density-Aware Loss** — Per-sample adaptive pos_weight (20-199) + Focal Loss (γ=1.5)
8. **SwiGLU Gated FFN** — Higher parameter efficiency than standard GELU FFN
9. **UFold Spatial Injection (FiLM)** — Feature-wise Linear Modulation preserving spatial conditioning details
10. **Strict Greedy Projection** — consistent train/inference projection

---

## Results

### Version History

| Version | Architecture | Status | Val F1 | Notes |
|:-------:|:------------|:------:|:------:|:------|
| **v5** | DA-SE-DiT-v5 (wider FM + density cond + refine conv) | **Training (55/120ep)** | **0.798** | Faster convergence, density-guided sampling |
| v4 | DA-SE-DiT-v4 (9L, triangle, multi-FM, SwiGLU) | Training (66/120ep) | 0.616 | Multi-layer FM + AF2 triangle |
| v3 | DA-SE-DiT (9L flat, dilation 1/2/4) | Completed (80ep) | 0.603 | Avg test F1=0.752 |
| v1 | SEDiT (6L flat) | Completed | 0.644 | Baseline, avg test F1=0.742 |
| v2 | MSEDiT (3+2+3 U-shape) | ❌ Failed | 0.296 | Collapsed: relaxed projection gap |

### SymFold v5 Evaluation Results (epoch 40/120, single sample, density-guided)

| Dataset | N | Type | v5 F1 | v4 F1 | v3 F1 | v1 F1 | RNADiffFold F1 |
|---------|---:|:----:|:-----:|:-----:|:-----:|:-----:|:--------------:|
| RNAStrAlign | 2023 | ID | 0.917 | **0.941** | 0.939 | 0.921 | 0.787 |
| ArchiveII | 3911 | OOD | 0.840 | **0.870** | 0.864 | 0.861 | 0.740 |
| PDB_TS2 | 38 | OOD-hard | **0.831** | 0.780 | 0.807 | 0.832 | 0.733 |
| PDB_TS1 | 60 | OOD-hard | 0.695 | 0.707 | **0.716** | 0.675 | 0.607 |
| PDB_TS3 | 18 | OOD-hard | 0.658 | 0.630 | **0.666** | 0.665 | 0.635 |
| bpRNA-new | 5401 | OOD | **0.593** | — | — | — | — |
| bpRNA | 1304 | ID | 0.593 | 0.638 | 0.636 | **0.644** | 0.618 |
| PDB_TS_hard | 28 | OOD-hardest | 0.578 | 0.608 | **0.634** | 0.596 | 0.526 |

**Note**: v5 results at epoch 40/120 (only 1/3 trained). Already exceeds v4 on PDB_TS2 (0.831 vs 0.780). Val F1=0.783 at epoch 35 far surpasses v4's 0.616 and v3's 0.603. Training ongoing.

### SymFold v1 vs RNADiffFold (8 benchmarks, single sample, no physics guidance)

| Dataset | N | Type | SymFold F1 | RNADiffFold F1 | Δ |
|---------|---:|:----:|:----------:|:--------------:|:---:|
| RNAStrAlign | 2023 | ID | **0.921** | 0.787 | +0.134 |
| ArchiveII | 3911 | OOD | **0.861** | 0.740 | +0.121 |
| PDB_TS2 | 38 | OOD-hard | **0.832** | 0.733 | +0.099 |
| bpRNA-new | 5401 | OOD-easy | **0.683** | 0.611 | +0.072 |
| PDB_TS1 | 60 | OOD-hard | **0.675** | 0.607 | +0.068 |
| PDB_TS3 | 18 | OOD-hard | **0.665** | 0.635 | +0.030 |
| bpRNA | 1304 | ID | **0.644** | 0.618 | +0.026 |
| PDB_TS_hard | 28 | OOD-hardest | **0.596** | 0.526 | +0.070 |

**Average F1: 0.735 vs 0.657 (+11.8%)**. With 1/8 parameters (13M vs 109M) and 10× faster inference.

### v3 Evaluation Results (80 epochs, single sample, no physics guidance)

| Dataset | N | Type | v3 F1 | v1 F1 | RNADiffFold F1 |
|---------|---:|:----:|:-----:|:-----:|:--------------:|
| RNAStrAlign | 2023 | ID | **0.939** | 0.921 | 0.787 |
| ArchiveII | 3911 | OOD | **0.864** | 0.861 | 0.740 |
| PDB_TS2 | 38 | OOD-hard | 0.807 | **0.832** | 0.733 |
| PDB_TS1 | 60 | OOD-hard | **0.716** | 0.675 | 0.607 |
| PDB_TS3 | 18 | OOD-hard | **0.666** | 0.665 | 0.635 |
| bpRNA | 1304 | ID | 0.636 | **0.644** | 0.618 |
| PDB_TS_hard | 28 | OOD-hardest | **0.634** | 0.596 | 0.526 |
| **Average** | | | **0.752** | 0.742 | 0.657 |

**v3 vs RNADiffFold: +14.5% avg F1** (0.752 vs 0.657), with 1/5 trainable parameters (21.8M vs 109M) and 4× faster inference (single sample, no multi-vote).

### v4 Evaluation Results (epoch 65/120, single sample, no physics guidance)

| Dataset | N | Type | v4 F1 | v3 F1 | v1 F1 | RNADiffFold F1 |
|---------|---:|:----:|:-----:|:-----:|:-----:|:--------------:|
| RNAStrAlign | 2023 | ID | **0.941** | 0.939 | 0.921 | 0.787 |
| ArchiveII | 3911 | OOD | **0.870** | 0.864 | 0.861 | 0.740 |
| PDB_TS2 | 38 | OOD-hard | 0.780 | **0.807** | 0.832 | 0.733 |
| PDB_TS1 | 60 | OOD-hard | 0.707 | **0.716** | 0.675 | 0.607 |
| bpRNA | 1304 | ID | **0.638** | 0.636 | 0.644 | 0.618 |
| PDB_TS3 | 18 | OOD-hard | 0.630 | **0.666** | 0.665 | 0.635 |
| PDB_TS_hard | 28 | OOD-hardest | 0.608 | **0.634** | 0.596 | 0.526 |
| **Average** | | | **0.739** | **0.752** | 0.742 | 0.657 |

**Note**: v4 results above are from the `"all"` dataset mode (epoch 65/120). Future training uses `"standard"` mode (bpRNA TR0 + RNAStrAlign only, no bpRNA-new leakage). bpRNA-new is now a proper OOD test set.

### Data Split Design

```
Training:   bpRNA TR0 (11,751) + RNAStrAlign train (17,630) = 29,381 samples
Validation: bpRNA VL0 (1,299) + RNAStrAlign val (~2,000) = ~3,299 samples
Test:       bpRNA TS0 | RNAStrAlign test | bpRNA-new | ArchiveII | PDB TS1/2/3/hard
```

- **No data leakage**: bpRNA-new removed from training, used only for evaluation
- **Both val sets**: early stopping uses combined bpRNA VL0 + RNAStrAlign val for better generalization signal

---

## Installation

### Requirements

- Python 3.10+ (tested on 3.12)
- PyTorch 2.6.0+ with CUDA 12.4
- GPU with ≥24GB VRAM (tested on NVIDIA H20 96GB)
- **Note**: TF32 must be disabled on H20 (cuBLAS SIGFPE bug)

```bash
# Create conda environment
conda create -n symfold python=3.12 -y
conda activate symfold

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt
```

Key dependencies: `einops` (tensor rearrangement), `scipy` (sparse matrix), `pandas` (ct file parsing), `matplotlib` (visualization), `scikit-learn` (metrics), `tqdm`, `PyYAML`, `Pillow`.

### Download Pretrained Weights

Place them in `ckpt/cond_ckpt/`:

| File | Size | Description | Source |
|------|------|-------------|--------|
| `RNA-FM_pretrained.pth` | 1.2 GB | RNA Foundation Model (12-layer Transformer) | [RNA-FM](https://github.com/ml4bio/RNA-FM) |
| `ufold_train_alldata.pt` | 34 MB | UFold U-Net pretrained on all RNA data | [UFold](https://github.com/uci-cbcl/UFold) |

```bash
mkdir -p ckpt/cond_ckpt
# Download RNA-FM
wget -O ckpt/cond_ckpt/RNA-FM_pretrained.pth <RNA-FM_URL>
# Download UFold
wget -O ckpt/cond_ckpt/ufold_train_alldata.pt <UFold_URL>
```

### Download Data

Place datasets in `data/`:

#### Training Data (used in `dataset: "standard"` mode)

| Directory | Size | Contents |
|-----------|------|----------|
| `data/preprocess/RNAStrAlign/` | 121 MB | RNAStrAlign train (preprocessed, binned cPickle) |
| `data/preprocess/bpRNA/` | 63 MB | bpRNA TR0 (preprocessed, binned cPickle) |

#### Validation Data (for early stopping, NOT seen during training)

| File | Size | Contents |
|------|------|----------|
| `data/bpRNA/VL0.cPickle` | 6 MB | bpRNA VL0 (1,299 samples) |
| `data/RNAStrAlign/val.cPickle` | 12 MB | RNAStrAlign val (subset) |

#### Test Data (completely independent evaluation)

| File | Samples | Type | Description |
|------|--------:|:----:|-------------|
| `data/bpRNA/TS0.cPickle` | 1,304 | ID test | bpRNA official test |
| `data/RNAStrAlign/test.cPickle` | 2,023 | ID test | RNAStrAlign official test |
| `data/ArchiveII/archiveII.cPickle` | 3,911 | OOD | Completely independent RNA families |
| `data/PDB/TS1.cPickle` | 60 | OOD-hard | PDB 3D structure-derived |
| `data/PDB/TS2.cPickle` | 38 | OOD-hard | PDB 3D structure-derived |
| `data/PDB/TS3.cPickle` | 18 | OOD-hard | PDB 3D structure-derived |
| `data/PDB/TS_hard.cPickle` | 28 | OOD-hardest | PDB hardest subset |
| `data/bpRNA-new/bpRNAnew.cPickle` | 5,401 | OOD | bpRNA new additions (NOT in training) |

#### Raw Data (for reference only)

| File | Description |
|------|-------------|
| `data/bpRNA/TR0.cPickle` | Raw training: bpRNA TR0 (11,751 samples) |
| `data/RNAStrAlign/train.cPickle` | Raw training: RNAStrAlign (17,630) |

#### Data Format

All `.cPickle` files are Python pickle 序列化的 `list[RNA_SS_data]`，其中 `RNA_SS_data` 是一个 namedtuple：

```python
import collections
RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq seq_raw length name pairs')
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `seq` | `np.ndarray (L, 4)` | one-hot 编码的序列 (A=[1,0,0,0], U=[0,1,0,0], C=[0,0,1,0], G=[0,0,0,1]) |
| `seq_raw` | `str` | 原始 RNA 序列字符串，如 `"AUGCGC..."` |
| `length` | `int` | 序列长度 |
| `name` | `str` | 样本 ID，如 `"bpRNA_CRW_15573"` |
| `pairs` | `list of [i, j]` | 碱基对列表 (0-indexed)，如 `[[0, 118], [1, 117], ...]` |

#### Original Data Sources

| 数据集 | 来源 | 原始格式 | 下载地址 |
|--------|------|----------|----------|
| bpRNA (TR0/VL0/TS0) | bpRNA database | `.ct` 文件 (connectivity table) | https://bprna.cgrb.oregonstate.edu/ |
| bpRNA-new | bpRNA 新增数据 | `.ct` 文件 | 同上 |
| RNAStrAlign | RNA Structure Alignment | `.ct` 文件 | https://rna.urmc.rochester.edu/pub/RNAStrAlign.tar.gz |
| ArchiveII | RNA Archive II | `.ct` 文件 | https://rna.urmc.rochester.edu/pub/archiveII.tar.gz |
| PDB (TS1/TS2/TS3/TS_hard) | RCSB PDB 3D 结构 | `.pdb` + 工具提取 | https://www.rcsb.org/ |

**`.ct` 文件格式** (connectivity table，每行一个碱基):
```
1  G  0  2  72  1
2  C  1  3  71  2
...
```
列含义: `编号(1-idx)  碱基  前一编号  后一编号  配对编号(0=未配对)  编号`

#### Data Preprocessing Pipeline (从原始数据复现)

原始 `.ct` 文件 → cPickle 的转换管道:

```
1. 原始 .ct 文件 → RNA_SS_data namedtuple (已由上游项目完成)
   - 解析函数: common/data_utils.py :: get_pairings(data)
   - 序列编码: common/data_utils.py :: seq_encoding(string) → (L, 4) one-hot
   - 打包为 list[RNA_SS_data] 后 pickle.dump → data/{dataset}/{split}.cPickle

2. Raw cPickle → Binned cPickle (按长度分桶，训练用)
   - 脚本: preprocess_data/binning_alldata.py
   - 输入: data/bpRNA/TR0.cPickle, data/RNAStrAlign/train.cPickle 等
   - 输出: data/preprocess/{dataset}/bpRNA-pdb_{bin_len}_{idx}.cPickle
   - 分桶规则: 步进 80 (80, 160, 240, ...)
   - 每桶 batch: 80→128, 160→64, 240-320→16, 320-640→4, 640-1280→2, >1280→1
```

**复现命令:**
```bash
cd /root/aigame/dannyyan/RNADiffFold
python preprocess_data/binning_alldata.py    # 生成 data/preprocess/ 训练数据
python preprocess_data/binning_testsets.py   # 生成 data/preprocess_test/ (可选)
```

#### Training Data Loading Flow

训练时 **不需要** 预先计算 17 通道 FCN 特征，全部在 GPU 实时计算:

```
data/preprocess/{dataset}/*.cPickle   (binned, 格式 A)
        ↓  src/data.py :: build_index()
扁平索引 [(path, sample_idx, seq_len), ...]
        ↓  src/data.py :: SimpleRNADataset.__getitem__()
encode_one_sample() → contact_map (L,L) + seq_one_hot (L,4)
        ↓  src/data.py :: simple_collate_fn()
batch tensors + RNA-FM tokens
        ↓  src/gpu_features.py :: get_data_fcn_gpu()  [GPU 实时计算]
17 通道 FCN 特征 (B, 17, L, L) = 16ch 碱基对外积 + 1ch 配对概率得分
        ↓
模型前向推理
```

---

## Project Structure

```
symfold/
├── README.md              # This file
├── CLAUDE.md              # AI assistant guidelines
├── requirements.txt       # Python dependencies
├── .gitignore
│
├── src/                   # All source code
│   ├── v1/                # v1: SEDiT (6-layer flat, greedy projection) — baseline
│   │   ├── README.md
│   │   ├── model.py       #   SymFoldModel
│   │   ├── se_dit.py      #   Symmetry-Equivariant Axial DiT
│   │   └── discrete_flow.py
│   ├── v2/                # v2: MSEDiT (U-shape 3+2+3) — deprecated
│   │   ├── README.md
│   │   ├── model.py       #   SymFoldModel_v2
│   │   ├── ms_se_dit.py   #   Multi-Scale Axial DiT
│   │   └── discrete_flow.py
│   ├── v3/                # v3: DA-SE-DiT (9-layer dilated axial)
│   │   ├── README.md
│   │   ├── model.py       #   SymFoldModel_v3
│   │   ├── da_se_dit.py   #   Dilated Axial SE-DiT
│   │   └── discrete_flow.py  # Strict projection + Physics loss
│   ├── v4/                # v4: DA-SE-DiT-v4 (multi-FM + triangle + adaptive loss)
│   │   ├── README.md
│   │   ├── model.py       #   SymFoldModel_v4
│   │   ├── da_se_dit.py   #   DASEDiT_v4 (triangle update + SwiGLU)
│   │   └── discrete_flow.py  # Adaptive density loss + Focal
│   ├── v5/                # ★ v5: DA-SE-DiT-v5 (wider FM + density cond + refine conv) — current
│   │   ├── README.md
│   │   ├── model.py       #   SymFoldModel_v5
│   │   └── da_se_dit.py   #   DASEDiT_v5
│   ├── data.py            # Shared: Dataset / BucketBatchSampler
│   ├── gpu_features.py    # Shared: GPU 17-channel FCN features
│   ├── physics_energy.py  # Shared: Physics guidance (WC + stacking + PK)
│   ├── adversarial.py     # Shared: Family-adversarial GRL
│   ├── common/            # Utilities (data_utils, loss_utils)
│   ├── datasets/          # Data loading (cPickle reader)
│   └── models/            # Conditioners (RNA-FM, UFold)
│
├── train/                 # Training scripts
│   ├── config/            #   JSON configs
│   ├── train.py           #   v1 trainer
│   ├── train_v2.py        #   v2 trainer (deprecated)
│   ├── train_v3.py        #   v3 trainer
│   ├── train_v4.py        #   v4 trainer
│   └── train_v5.py        #   ★ v5 trainer (current, full eval every 20 epochs)
│
├── eval/                  # Evaluation
│   └── eval.py            #   Multi-dataset eval (supports --detailed)
│
├── scripts/               # Shell scripts
│   ├── run_train.sh
│   ├── run_train_v2.sh
│   ├── run_train_v3.sh
│   ├── run_train_v4.sh    #   ★ v4 training launcher
│   └── run_eval.sh
│
├── doc/                   # Documentation & reports
│   ├── MODEL_ARCHITECTURE_EVOLUTION.md  # Full architecture walkthrough (v1→v4)
│   ├── DISCRETE_FLOW_MATCHING.md        # ★ DFM tutorial with numerical examples
│   ├── V5_DESIGN.md                     # v5 design rationale
│   └── V4_FAILURE_ANALYSIS.md           # v4 PDB underperformance analysis
├── ckpt/                  # Pretrained weights (not in git)
├── data/                  # Datasets (not in git)
├── model/                 # Saved checkpoints (not in git)
├── logs/                  # Training logs (not in git)
└── output/                # Visualizations & eval results (not in git)
```

---

## Usage

### Training

```bash
cd symfold

# v5 (current, ~27M trainable params, full eval every 20 epochs)
python -u train/train_v5.py train/config/train_config_v5.json
# or use the launch script:
bash scripts/run_train_v5.sh

# v4 (~25.1M trainable params, ~19min/epoch on H20)
python -u train/train_v4.py train/config/train_config_v4.json
bash scripts/run_train_v4.sh

# v3 (~21.8M trainable params, ~18min/epoch on H20)
python -u train/train_v3.py train/config/train_config_v3.json
bash scripts/run_train_v3.sh

# v1 (baseline, ~13M params, ~20min/epoch on H20)
python -u train/train.py train/config/train_config.json
```

Training outputs (saved to `output/<task_name>/`):
- `curves.png` — Loss / Val F1 / Epoch time curves (updated every epoch)
- `history.json` — Full training history
- `vis_e{N}_{sample}.png` — GT vs Pred visualization during validation

### Evaluation

```bash
# Quick eval on all test sets
bash scripts/run_eval.sh model/<task>/best.pt

# Detailed eval (per-sample sequence, structure, TP/FP/FN analysis)
python eval/eval.py \
    --ckpt model/<task>/best.pt \
    --test_sets bpRNA,RNAStrAlign,ArchiveII,PDB_TS1,PDB_TS2,PDB_TS3,PDB_TS_hard \
    --detailed --top_k 5 \
    --out_json output/<task>/eval_detailed.json

# Eval on validation sets (to check overfitting)
python eval/eval.py \
    --ckpt model/<task>/best.pt \
    --test_sets bpRNA_VL0,RNAStrAlign_val

# With physics guidance
python eval/eval.py \
    --ckpt model/<task>/best.pt \
    --test_sets PDB_TS1 \
    --physics_beta 0.5 --physics_lambda_pk 0.0 \
    --num_steps 20
```

### Inference on a single sequence

```python
import torch
from src.v4.model import SymFoldModel_v4

model = SymFoldModel_v4().cuda()
ckpt = torch.load('model/260522-v4-train/best.pt', map_location='cuda', weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

# Prepare input (see src/data.py for data pipeline)
# pred, prob = model.sample(data_fcn_2, tokens, contact_masks, set_max_len, seq_oh)
```

---

## Method

### Bernoulli Discrete Flow Matching

Forward marginal (per position pair):
```
p_t(X_ij = 1 | X_1) = (1-t) · ρ_0 + t · 1[X_1,ij = 1]
```

- `t=0`: Prior Bernoulli(ρ₀=0.005) ≈ dataset pairing rate
- `t=1`: Ground truth contact map
- Training: pos-weighted BCE with time weighting `w(t) = 1/(1-t(1-ρ₀))`
- Sampling: τ-leap CTMC with closed-form rates

### Architecture (v5: DA-SE-DiT-v5, current)

```
Input (48ch) → PatchEmbed(4) → [DilatedAxialAttn + SwiGLU-FFN + AdaLN + TriangleUpdate(L6-8)] ×9 → UnPatch → OutputRefineConv(3L) → logit
                                 dilation: [1,1,1, 2,2,2, 4,4,4]
                                 triangle multiplicative update on layers 6-8
```

Key improvements over v4:
- **Wider RNA-FM Fusion**: `fm_multi_out_dim=64` (v4=16), 4× more capacity to preserve multi-layer RNA-FM information
- **Density Conditioning**: Injects ground-truth pairing density during training via AdaLN; predicts density at inference for guided sampling
- **Density-guided Sampling**: Dynamically scales 0→1 flip rate based on predicted density, suppressing over-prediction for sparse RNAs
- **OutputRefineConv**: 3-layer residual convolution at full L×L after UnPatchify, refining patch boundary artifacts
- **Stronger Low-density Control**: `pos_weight_min=20` (v4=50), `focal_gamma=1.5` (v4=1.0)
- 9 layers, hidden_dim=256, 4 heads, dim_head=64 (total trainable: ~27M)

### Architecture (v4: DA-SE-DiT-v4)

```
Input (48ch) → PatchEmbed(4) → [DilatedAxialAttn + SwiGLU-FFN + AdaLN + TriangleUpdate(L6-8)] ×9 → UnPatch → logit
                                 dilation: [1,1,1, 2,2,2, 4,4,4]
                                 triangle multiplicative update on layers 6-8
```

Key improvements over v3:
- **Multi-Layer RNA-FM Fusion**: Extracts layers [3,6,9,12] with learnable softmax-weighted combination + per-layer projection + MLP fusion — captures local motifs (shallow) to global folding (deep)
- **Triangle Multiplicative Update**: AF2-inspired ternary constraint on layers 6-8, explicitly modeling mutual exclusion between base pairs
- **Adaptive Density-Aware Loss**: Per-sample adaptive pos_weight (50-199) based on pairing density + Focal Loss (γ=1.0)
- **SwiGLU Gated FFN**: `SiLU(W1·x) * W2·x → W3` replaces GELU FFN for better parameter efficiency
- **Density Regression Head**: Auxiliary task predicting pairing density to guide projection
- 9 layers, hidden_dim=256, 4 heads, dim_head=64 (backbone: ~15.5M, triangle: ~0.8M, total trainable: ~25.1M)

### Architecture (v3: DA-SE-DiT)

```
Input (48ch) → PatchEmbed(4) → [DilatedAxialAttn + FFN + AdaLN] ×9 → UnPatch → logit
                                 dilation: [1,1,1, 2,2,2, 4,4,4]
```

Key improvements over v1:
- **Dilated Axial Attention**: alternating dilation rates (1/2/4) capture multi-scale dependencies without downsampling — avoids v2's U-Net symmetry-breaking issue
- **Cross-Resolution Attention**: global compressed attention inserted every 3 layers
- **UFold FiLM Injection**: Feature-wise Linear Modulation preserves spatial conditioning details from UFold
- **Physics-Aware Loss**: stacking continuity + non-crossing penalties during training
- **Strict Projection**: greedy max-matching (same as v1), ensuring train/inference consistency
- 9 layers, hidden_dim=256, 4 heads, dim_head=64 (backbone: 13.2M params, total trainable: 21.8M)

### Architecture (v1: SEDiT, baseline)

```
Input (48ch) → PatchEmbed(4) → [AxialAttn + FFN + AdaLN] ×6 → UnPatch → logit
```

- Shared QKV for row/col attention → O(L³) complexity, strict symmetry
- AdaLN-Zero conditioning on time + RNA-FM global + UFold global
- pos_weight = (1-ρ₀)/ρ₀ ≈ 199

### Architecture (v2: MSEDiT, deprecated)

```
Input → PatchEmbed → Encoder(×3) → Downsample2× → Middle(×2) → Upsample2× → Skip+Decoder(×3) → logit
```

- U-shape for multi-scale: middle blocks see L/8 resolution (2× larger receptive field)
- Local attention bias on first 2 encoder layers
- Relaxed projection: allows up to 2 pairs per row (supports pseudoknots)

---

## Citation

If you use this code, please cite:

```bibtex
@article{symfold2026,
  title={SymFold: Symmetry-Constrained Discrete Flow Matching with Physics-Guided Sampling for RNA Secondary Structure Prediction},
  author={Yan, Danny},
  year={2026},
  note={In preparation for NeurIPS 2026 / ICLR 2027}
}
```

---

## License

MIT License
