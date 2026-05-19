# CLAUDE.md — SymFold 项目指南

> 本文件供 AI 助手阅读，描述项目结构、开发规范和用户偏好。

---

## 项目概述

SymFold 是一个基于 **Discrete Flow Matching** 的 RNA 二级结构预测模型，位于：
```
/root/aigame/dannyyan/RNADiffFold/symfold/
```

它是 RNADiffFold 的改进版本，使用 Bernoulli Flow Matching + Symmetry-Equivariant Axial DiT 架构。

---

## 目录结构规范

symfold/ 顶层**只允许**以下目录/文件：
```
symfold/
├── README.md          # 项目文档
├── CLAUDE.md          # AI 助手指南 (本文件)
├── ckpt/              # 预训练权重 (符号链接)
├── data/              # 数据集 (符号链接)
├── doc/               # ★ 文档 (对比报告、改进计划、技术文档等，除 README 和 CLAUDE 外的所有文档)
├── eval/              # ★ 评估/测试脚本 (所有评估代码放这里)
├── logs/              # ★ 所有后台任务的日志
├── model/             # ★ 训练产出的 checkpoint
├── output/            # ★ 可视化输出、eval JSON、训练曲线、GPU 监控
├── scripts/           # 运行脚本 (bash + python 工具脚本)
├── src/               # ★ 所有核心代码
└── train/             # 训练入口 + 配置
```

### 各目录职责与命名规范

| 目录 | 存放内容 | 命名格式 | 示例 |
|------|----------|----------|------|
| `eval/` | 评估/测试脚本 | `eval.py`, `eval_v2.py` 等 | `eval/eval_v2.py` |
| `doc/` | 所有文档 (除 README.md 和 CLAUDE.md) | 描述性名称 | `doc/EVAL_COMPARISON_REPORT.md` |
| `logs/` | 所有后台任务的日志 | `YYMMDD-HHMMSS-任务名/` **子目录** | `logs/260519-132200-v2-fresh/*.log` |
| `model/` | 训练产出的 checkpoint | `YYMMDD-HHMMSS-任务名/` 子目录 | `model/260519-132200-v2-fresh/best.pt` |
| `output/` | 可视化、eval 结果、训练曲线 | `YYMMDD-HHMMSS-任务名/` 子目录 | `output/260519-132200-v2-fresh/curves.png` |
| `scripts/` | 工具脚本 (GPU 监控、eval 启动等) | 描述性名称 | `scripts/gpu_monitor.py` |

### ★ 核心原则

1. **时间+任务名 + 子目录归档**: `logs/`, `model/`, `output/` 下的所有产出必须以 `YYMMDD-HHMMSS-任务名/` 子目录组织
2. **logs 子目录**: 同一任务的所有日志文件（.log, .heartbeat, .stdout.log, .stderr.log, gpu_monitor 等）放在同一个子目录下
3. **评估代码放 `eval/`**: 不要把 eval 脚本散落在其他地方
4. **文档放 `doc/`**: README.md 和 CLAUDE.md 放顶层，其余所有文档（报告、计划、分析）放 `doc/`
5. **模型放 `model/`**: 每次训练任务一个子目录，内含 `best.pt` + `last.pt`
6. **输出放 `output/`**: 每次训练/评估任务一个子目录，内含曲线、history、可视化、eval JSON

示例目录结构:
```
logs/
├── 260518-161300-v2-train/       ← 一个任务的所有日志
│   ├── 260518-161300-v2-train.log
│   ├── 260518-161300-v2-train.stdout.log
│   ├── 260518-161300-v2-train.stderr.log
│   └── 260518-161300-v2-train.heartbeat
├── 260519-132200-v2-fresh/       ← 另一个任务
│   ├── 260519-132200-v2-fresh.log
│   ├── 260519-132200-v2-fresh.stdout.log
│   ├── 260519-132200-v2-fresh.stderr.log
│   ├── 260519-132200-v2-fresh.heartbeat
│   ├── 260519-132200-v2-fresh-eval.log
│   └── 260519-gpu-monitor.log
```

### src/ 内部结构

```
src/
├── __init__.py
├── data.py              # 公共: Dataset / BucketBatchSampler
├── gpu_features.py      # 公共: GPU 17通道 FCN 特征
├── physics_energy.py    # 公共: 物理 guidance
├── adversarial.py       # 公共: GRL 分类器
├── common/              # 公共工具 (data_utils, loss_utils)
├── datasets/            # 数据加载 (data_generator, _CompatUnpickler)
├── models/              # 条件编码器 (RNA-FM, UFold)
├── v1/                  # ★ v1 版本模型代码
│   ├── model.py         #   SymFoldModel
│   ├── se_dit.py        #   SEDiT backbone (6层 flat)
│   └── discrete_flow.py #   Greedy max-matching
└── v2/                  # ★ v2 版本模型代码
    ├── model.py         #   SymFoldModel_v2
    ├── ms_se_dit.py     #   MSEDiT backbone (3+2+3 U型)
    └── discrete_flow.py #   Relaxed projection + Cosine schedule
```

**规则**：
- 版本专属代码（model, backbone, flow）放 `src/v1/`, `src/v2/`, `src/v3/`...
- 公共模块放 `src/` 顶层或 `src/common/`, `src/datasets/`, `src/models/`
- 不要在 symfold/ 顶层散落 .py 文件或其他代码目录

---

## 环境

- **conda env**: `/root/aigame/dannyyan/miniconda3/envs/RNADiffFold_torch260`
- **Python**: 3.12
- **PyTorch**: 2.6.0 + cu124
- **GPU**: NVIDIA H20 (96 GB), device `cuda:0`
- **重要**: 全程 fp32，TF32 必须关闭（H20 上有 cuBLAS SIGFPE bug）

激活环境：
```bash
source /root/aigame/dannyyan/miniconda3/bin/activate RNADiffFold_torch260
```

---

## 运行命令

### 训练
```bash
cd /root/aigame/dannyyan/RNADiffFold/symfold
source /root/aigame/dannyyan/miniconda3/bin/activate RNADiffFold_torch260

# 命名规范: YYMMDD-HHMMSS-任务名
# 例如: 260519-132200-v2-fresh

# v1 训练
nohup python -u train/train.py train/config/train_config.json >> logs/<task>.stdout.log 2>> logs/<task>.stderr.log &

# v2 全新训练 (auto_resume=false, batch翻倍, vis_samples=10)
nohup python -u train/train_v2.py train/config/train_config_v2_fresh.json >> logs/260519-132200-v2-fresh.stdout.log 2>> logs/260519-132200-v2-fresh.stderr.log &

# v2 恢复训练 (auto_resume=true, 从 last.pt 恢复)
nohup python -u train/train_v2.py train/config/train_config_v2.json >> logs/<task>.stdout.log 2>> logs/<task>.stderr.log &
```

### GPU 监控 (后台持续运行)
```bash
# 后台启动 GPU 监控，每10秒采集一次，每5分钟绘制一次曲线
nohup python scripts/gpu_monitor.py --output output/260519-132200-v2-fresh --interval 10 --plot_every 30 >> logs/260519-gpu-monitor.log 2>&1 &

# 查看实时监控图: output/260519-132200-v2-fresh/gpu_monitor_live.png
# 停止: kill <PID>  (优雅退出, 退出前自动保存)
```

### 评估
```bash
# 基础评估
bash scripts/run_eval.sh model/260514-full-train-symfold/best.pt

# 详细评估 (逐样本 + 序列 + GT/Pred结构)
python eval/eval.py --ckpt model/xxx/best.pt --test_sets PDB_TS1,PDB_TS3 --detailed --out_json output/<task>-eval-detailed.json
```

### 监控训练
```bash
# 查看训练日志
tail -f logs/260519-132200-v2-fresh/260519-132200-v2-fresh.log

# 查看 heartbeat (实时训练状态)
cat logs/260519-132200-v2-fresh/260519-132200-v2-fresh.heartbeat

# 查看训练历史
cat output/260519-132200-v2-fresh/history.json

# 查看 GPU 监控图
# 直接在 IDE 中打开: output/260519-132200-v2-fresh/gpu_monitor_live.png
```

---

## 用户偏好和要求

### 任务与日志规范（重要！）
1. **所有后台任务必须有日志**: 无论是训练、评估、GPU 监控还是其他任何后台运行的脚本，都必须将 stdout/stderr 重定向到 `logs/<任务子目录>/` 下
2. **日志按任务归档到子目录**: 同一任务的所有日志放在 `logs/YYMMDD-HHMMSS-任务名/` 子目录下
   - 训练日志: `logs/260519-132200-v2-fresh/260519-132200-v2-fresh.log`
   - eval 日志: `logs/260519-132200-v2-fresh/260519-132200-v2-fresh-eval.log`
   - GPU 监控: `logs/260519-132200-v2-fresh/260519-gpu-monitor.log`
   - heartbeat: `logs/260519-132200-v2-fresh/260519-132200-v2-fresh.heartbeat`
3. **禁止直接用 `python -c` 跑长时间任务**: 必须写成独立脚本，方便复用和追踪
4. **日志必须 append 模式 (`>>`)**: 不要覆盖已有日志，方便回溯历史
5. **启动后必须确认**: 每次启动后台任务后，必须验证进程存在 + 日志正常输出

示例模板:
```bash
cd /root/aigame/dannyyan/RNADiffFold/symfold
source /root/aigame/dannyyan/miniconda3/bin/activate RNADiffFold_torch260

# 格式: nohup python -u <script> <args> >> logs/YYMMDD-HHMMSS-任务名.log 2>&1 &
nohup python -u scripts/eval_v2.py --ckpt model/xxx/best.pt \
    >> logs/260519-185500-v2-eval.log 2>&1 &

# 确认
ps aux | grep eval_v2 | grep -v grep
tail -5 logs/260519-185500-v2-eval.log
```

### 代码组织
1. **版本管理**: 模型迭代用 `src/v1/`, `src/v2/`, `src/v3/` 区分，不要覆盖旧版本
2. **自包含**: symfold/ 目录要能独立运行，所有依赖（common, models, datasets）都放在 `src/` 内，不依赖外部路径或 PYTHONPATH
3. **顶层整洁**: symfold/ 顶层只放上述列出的目录，不要散落代码文件

### 训练要求
1. **充分利用 GPU**: H20 有 96GB 显存，batch size 要尽量开大
2. **记录可视化**: 训练过程必须输出到output下：
   - Loss 曲线图 (每 epoch 更新)
   - Epoch 时间统计
   - Val 时的 case 可视化 (GT vs Pred 对比图)
   - 训练历史 JSON (方便后续分析)
3. **后台运行**: 训练要挂后台 (nohup)，确认跑起来后再退出
4. **命名规范**: logs/ 和 output/ 下的所有文件必须以 `YYMMDD-HHMMSS-任务名` 格式命名
   - 时间精确到秒，例如: `260518-143500-v2-train.stdout.log`
   - output 中的曲线/history 同理: `260518-143500-v2-train_curves.png`
   - model/ 下的 checkpoint 目录也用此格式: `model/260518-143500-v2-train/`

### 评估要求
1. **详细模式**: eval 脚本需要支持 `--detailed`，输出逐样本的：序列、GT/Pred dot-bracket 结构、TP/FP/FN、最好/最差样本展示
2. **对比文档**: 每次新版本训练完，要写对比报告放 `doc/`

### 其他
- 每个版本目录下写 README.md 说明架构
- 数据集和预训练权重用符号链接，不要复制大文件

---

## 数据集

数据通过符号链接引用 `data -> /root/aigame/dannyyan/RNADiffFold/data`

### 训练集 (Train) — 共 34,782 samples

从 `data/preprocess/` 加载，为预处理过的分 bin 数据（按序列长度分 batch）：

| 子集 | 文件 | 样本数 | 说明 |
|------|------|-------:|------|
| RNAStrAlign | `data/preprocess/RNAStrAlign/` | 17,630 | RNAStrAlign 训练集 |
| bpRNA TR0 | `data/preprocess/bpRNA/` | 11,751 | bpRNA 官方训练集 |
| bpRNA-new | `data/preprocess/bpRNA-new/` | 5,401 | ⚠️ bpRNA 新增数据（全量） |

### 验证集 (Val) — 训练时每 2 个 epoch 做一次 eval

| 数据集 | 文件 | 样本数 | 说明 |
|--------|------|-------:|------|
| bpRNA VL0 | `data/bpRNA/VL0.cPickle` | 1,299 | bpRNA 官方验证集，用于 early stopping |

### 测试集 (Eval) — 训练完后独立评估

| 数据集 | 文件 | 样本数 | 类型 | 说明 |
|--------|------|-------:|:----:|------|
| bpRNA TS0 | `data/bpRNA/TS0.cPickle` | 1,304 | ID test | ✅ 未在训练中出现 |
| RNAStrAlign | `data/RNAStrAlign/test.cPickle` | 2,023 | ID test | ✅ 未在训练中出现 |
| ArchiveII | `data/ArchiveII/archiveII.cPickle` | 3,911 | OOD | ✅ 完全独立数据 |
| bpRNA-new | `data/bpRNA-new/bpRNAnew.cPickle` | 5,401 | ⚠️ 泄漏 | 同时在训练集中 |
| PDB TS1 | `data/PDB/TS1.cPickle` | 60 | OOD-hard | ✅ PDB 3D 结构提取 |
| PDB TS2 | `data/PDB/TS2.cPickle` | 38 | OOD-hard | ✅ PDB 3D 结构提取 |
| PDB TS3 | `data/PDB/TS3.cPickle` | 18 | OOD-hard | ✅ PDB 3D 结构提取 |
| PDB TS_hard | `data/PDB/TS_hard.cPickle` | 28 | OOD-hardest | ✅ PDB 3D 结构提取 |

### ⚠️ 数据泄漏警告

**bpRNA-new** 被同时用作训练集和测试集（5,401 samples 完全重叠）。
在论文/报告中需标注该数据集为 "seen during training"，其 eval 结果不能作为泛化能力的证据。

### bpRNA 系列说明

| 名称 | 来源 | 划分 |
|------|------|------|
| bpRNA TR0/VL0/TS0 | bpRNA 数据库官方划分 | TR0=训练, VL0=验证, TS0=测试 ✅ |
| bpRNA-new | bpRNA 数据库后续新增 | 整体用于训练 + 整体用于测试 ⚠️ |

---

## 当前状态

- **v1**: 训练完成，best.pt 在 `model/260514-full-train-symfold/best.pt`
- **v2 (旧)**: 已废弃，配置在 `train/config/train_config_v2.json`，模型在 `model/260518-161300-v2-train/`
- **v2 (全新)**: ★ 正在训练中，从零开始
  - 配置: `train/config/train_config_v2_fresh.json`
  - 任务名: `260519-132200-v2-fresh`
  - 模型目录: `model/260519-132200-v2-fresh/`
  - 输出目录: `output/260519-132200-v2-fresh/`
  - 日志: `logs/260519-132200-v2-fresh.log`
  - 特点: batch翻倍, auto_resume=false, vis_samples=10, 只保留 last.pt + best.pt
  - 可视化: 按 RNA 名字建子文件夹 (`output/260519-132200-v2-fresh/vis/<rna_name>/epoch_XX.png`)
  - GPU监控: 后台运行 `scripts/gpu_monitor.py`，图在 `output/260519-132200-v2-fresh/gpu_monitor_live.png`
  - 进度: epoch 3/80 (截至 2026-05-19 14:14)

### 恢复/继续训练 v2-fresh 的步骤

1. 先确认训练进程是否还在运行:
```bash
ps aux | grep train_v2 | grep -v grep
```

2. 如果进程已死，直接重启（会自动从 last.pt 恢复）:
```bash
cd /root/aigame/dannyyan/RNADiffFold/symfold
source /root/aigame/dannyyan/miniconda3/bin/activate RNADiffFold_torch260

# 注意: 需要先将配置中 auto_resume 改为 true
# 编辑 train/config/train_config_v2_fresh.json，将 "auto_resume": false 改为 true

nohup python -u train/train_v2.py train/config/train_config_v2_fresh.json >> logs/260519-132200-v2-fresh.stdout.log 2>> logs/260519-132200-v2-fresh.stderr.log &
```

3. 确认 GPU 监控也在运行:
```bash
ps aux | grep gpu_monitor | grep -v grep
# 如果没有运行，重新启动:
nohup python scripts/gpu_monitor.py --output output/260519-132200-v2-fresh --interval 10 --plot_every 30 >> logs/260519-gpu-monitor.log 2>&1 &
```

4. 验证训练正常:
```bash
tail -5 logs/260519-132200-v2-fresh.log
cat logs/260519-132200-v2-fresh.heartbeat
```

### v1 评估结果 (baseline)

| Dataset | F1 | Precision | Recall |
|---------|:---:|:---------:|:------:|
| RNAStrAlign | 0.921 | 0.911 | 0.936 |
| ArchiveII | 0.861 | 0.839 | 0.893 |
| bpRNA | 0.644 | 0.583 | 0.758 |
| PDB_TS_hard | 0.596 | 0.695 | 0.540 |

### v2 改进点

1. Multi-Scale Axial DiT (U型 3+2+3 blocks，增强长程依赖)
2. Relaxed Projection (每行≤2 配对，支持 pseudoknot)
3. Adaptive Cosine Sampling Schedule (前大后小步长)
4. Local Attention Bias (前 2 层加短程 bias)
