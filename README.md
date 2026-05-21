# SymFold — Symmetry-Constrained Discrete Flow Matching for RNA Secondary Structure

> **SymFold** puts RNA secondary structure (contact map) prediction as a generative modeling problem on **symmetric binary matrices**, solved via **Discrete Flow Matching** with a **Symmetry-Equivariant Axial DiT** backbone.

---

## Quick Overview

```
RNA sequence → [RNA-FM + UFold conditioners] → DA-SE-DiT predicts P(pair)
             → τ-leap CTMC sampling (20 steps) → strict projection → contact map
```

**Key innovations (v3, current):**
1. **Bernoulli Discrete Flow Matching** on symmetric matrices (not Gaussian diffusion)
2. **Dilated Axial SE-DiT (DA-SE-DiT)** — 9-layer flat backbone with alternating dilation (1/2/4) for multi-scale long-range dependencies without resolution loss
3. **UFold Spatial Injection (FiLM)** — Feature-wise Linear Modulation preserving spatial conditioning details
4. **Physics-Aware Training Loss** — stacking continuity + non-crossing penalties during training
5. **Strict Greedy Projection** — consistent train/inference projection (lesson from v2's failure)

---

## Results

### Version History

| Version | Architecture | Status | Val F1 (bpRNA VL0) | Notes |
|:-------:|:------------|:------:|:-------------------:|:------|
| **v3** | DA-SE-DiT (9L flat, dilation 1/2/4) | **Training (epoch 56/80)** | **0.575** ↑ | Still improving, no plateau |
| v1 | SEDiT (6L flat) | Completed | 0.644 | Baseline, solid |
| v2 | MSEDiT (3+2+3 U-shape) | ❌ Failed | 0.296 | Collapsed: relaxed projection gap |

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

### v3 Training Progress (ongoing)

v3 val F1 is steadily climbing with no signs of plateau:

| Epoch | Train Loss | Val F1 | Val Precision | Val Recall |
|:-----:|:----------:|:------:|:-------------:|:----------:|
| 1 | 0.044 | 0.432 | 0.340 | 0.648 |
| 19 | 0.011 | 0.511 | 0.423 | 0.703 |
| 39 | 0.006 | 0.552 | 0.468 | 0.726 |
| 55 | 0.005 | **0.575** | 0.497 | 0.730 |

*Full test-set evaluation will be done once v3 training completes (80 epochs).*

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.6.0+ with CUDA 12.4
- GPU with ≥24GB VRAM (tested on NVIDIA H20 96GB)

```bash
# Create conda environment
conda create -n symfold python=3.10 -y
conda activate symfold

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt
```

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

| Directory | Size | Contents |
|-----------|------|----------|
| `data/preprocess/RNAStrAlign/` | 121 MB | Training set (preprocessed cPickle) |
| `data/preprocess/bpRNA/` | 63 MB | Training set |
| `data/preprocess/bpRNA-new/` | 22 MB | Training set |
| `data/bpRNA/TS0.cPickle` | — | Test: bpRNA (1304 samples) |
| `data/bpRNA/VL0.cPickle` | — | Validation (1299 samples) |
| `data/RNAStrAlign/test.cPickle` | — | Test: RNAStrAlign (2023) |
| `data/ArchiveII/archiveII.cPickle` | — | Test: ArchiveII (3911) |
| `data/PDB/TS1~TS3,TS_hard.cPickle` | — | Test: PDB OOD-hard (18~60) |
| `data/bpRNA-new/bpRNAnew.cPickle` | — | Test: bpRNA-new (5401) |

Data format: Python cPickle files containing `RNA_SS_data` namedtuples with fields `(seq, seq_raw, length, name, pairs)`.

Original data sources:
- **bpRNA**: [bpRNA database](https://bprna.cgrb.oregonstate.edu/)
- **RNAStrAlign**: [RNAStrAlign](https://rna.urmc.rochester.edu/pub/RNAStrAlign.tar.gz)
- **ArchiveII**: [ArchiveII](https://rna.urmc.rochester.edu/pub/archiveII.tar.gz)
- **PDB**: Extracted from RCSB PDB 3D structures

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
│   ├── v3/                # ★ v3: DA-SE-DiT (9-layer dilated axial) — current
│   │   ├── README.md
│   │   ├── model.py       #   SymFoldModel_v3
│   │   ├── da_se_dit.py   #   Dilated Axial SE-DiT
│   │   └── discrete_flow.py  # Strict projection + Physics loss
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
│   └── train_v3.py        #   ★ v3 trainer (current)
│
├── eval/                  # Evaluation
│   └── eval.py            #   Multi-dataset eval (supports --detailed)
│
├── scripts/               # Shell scripts
│   ├── run_train.sh
│   ├── run_train_v2.sh
│   ├── run_train_v3.sh    #   ★ v3 training launcher
│   └── run_eval.sh
│
├── doc/                   # Documentation & reports
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

# v3 (current, ~21.8M trainable params, ~18min/epoch on H20)
python -u train/train_v3.py train/config/train_config_v3.json
# or use the launch script:
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
    --test_sets bpRNA,ArchiveII,PDB_TS1,PDB_TS2,PDB_TS3,PDB_TS_hard \
    --detailed --top_k 5 \
    --out_json output/<task>/eval_detailed.json

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
from src.v3.model import SymFoldModel_v3

model = SymFoldModel_v3().cuda()
ckpt = torch.load('model/260520-v3-train/best.pt', map_location='cuda')
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

### Architecture (v3: DA-SE-DiT, current)

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
