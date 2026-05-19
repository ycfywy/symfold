# -*- coding: utf-8 -*-
"""
SymFold 源码包

目录结构:
  src/
  ├── __init__.py          # 本文件
  ├── data.py              # 公共: Dataset / BucketBatchSampler / collate_fn
  ├── gpu_features.py      # 公共: GPU 17通道 FCN 特征
  ├── physics_energy.py    # 公共: 物理 guidance (WC + stacking + pseudoknot)
  ├── adversarial.py       # 公共: Family-Adversarial GRL
  ├── v1/                  # v1 版本 (原始, 已训练完成)
  │   ├── model.py         #   SymFoldModel (RNA-FM + UFold + SEDiT + FM)
  │   ├── se_dit.py        #   Symmetry-Equivariant Axial DiT (6 层 flat)
  │   └── discrete_flow.py #   Bernoulli FM + Greedy max-matching (每行≤1)
  └── v2/                  # v2 版本 (改进)
      ├── model.py         #   SymFoldModel_v2 (Multi-Scale backbone)
      ├── ms_se_dit.py     #   Multi-Scale Axial DiT (3+2+3 U 型)
      └── discrete_flow.py #   Relaxed projection (每行≤2) + Cosine schedule
"""
