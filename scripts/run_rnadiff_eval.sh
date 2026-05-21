#!/usr/bin/env bash
# 跑 RNADiffFold 原版模型在所有测试集上的 eval，用于与 SymFold v1 对比
# 日志输出到 symfold/logs/ 目录
set -e

SYMFOLD_ROOT="/root/aigame/dannyyan/RNADiffFold/symfold"
RNADIFF_ROOT="/root/aigame/dannyyan/RNADiffFold"
EVAL_DIR="$RNADIFF_ROOT/evaluation"
LOG_DIR="$SYMFOLD_ROOT/logs/260520-105600-rnadiff-eval"
CONFIG="$EVAL_DIR/config.json"

mkdir -p "$LOG_DIR"

source /root/aigame/dannyyan/miniconda3/bin/activate RNADiffFold_torch260

DATASETS="bpRNA RNAStrAlign bpRNA-new ArchiveII PDB_TS1 PDB_TS2 PDB_TS3 PDB_TS_hard"

echo "=== RNADiffFold Eval Start: $(date) ===" >> "$LOG_DIR/260520-105600-rnadiff-eval.log"

for DS in $DATASETS; do
    echo "[$(date)] Evaluating dataset: $DS" >> "$LOG_DIR/260520-105600-rnadiff-eval.log"

    # 修改 config.json 的 dataset 字段
    python -c "
import json
with open('$CONFIG', 'r') as f:
    cfg = json.load(f)
cfg['data']['dataset'] = '$DS'
with open('$CONFIG', 'w') as f:
    json.dump(cfg, f, indent=2)
"

    # 跑 eval
    cd "$EVAL_DIR"
    PYTHONPATH="$RNADIFF_ROOT:$RNADIFF_ROOT/models/condition/fm_conditioner" \
        python -u eval.py >> "$LOG_DIR/260520-105600-rnadiff-eval.log" 2>&1

    echo "[$(date)] Done: $DS" >> "$LOG_DIR/260520-105600-rnadiff-eval.log"
    echo "" >> "$LOG_DIR/260520-105600-rnadiff-eval.log"
done

echo "=== RNADiffFold Eval Finish: $(date) ===" >> "$LOG_DIR/260520-105600-rnadiff-eval.log"
