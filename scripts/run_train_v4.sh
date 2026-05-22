#!/bin/bash
# SymFold v4 训练启动脚本
# 用法: bash scripts/run_train_v4.sh

set -e

cd /root/aigame/dannyyan/RNADiffFold/symfold
source /root/aigame/dannyyan/miniconda3/bin/activate RNADiffFold_torch260

TASK="260522-v4-train"
CONFIG="train/config/train_config_v4.json"

# 创建目录
mkdir -p logs/${TASK}
mkdir -p model/${TASK}
mkdir -p output/${TASK}

echo "[$(date)] Starting SymFold v4 training: ${TASK}"
echo "Config: ${CONFIG}"

nohup python -u train/train_v4.py ${CONFIG} \
    >> logs/${TASK}/${TASK}.stdout.log \
    2>> logs/${TASK}/${TASK}.stderr.log &

PID=$!
echo "[$(date)] Started PID=${PID}"
echo "Logs: logs/${TASK}/${TASK}.stdout.log"

# 等待几秒确认启动
sleep 5
if ps -p ${PID} > /dev/null 2>&1; then
    echo "[OK] Process running (PID=${PID})"
    tail -3 logs/${TASK}/${TASK}.stdout.log 2>/dev/null || true
else
    echo "[FAIL] Process died"
    tail -20 logs/${TASK}/${TASK}.stderr.log 2>/dev/null || true
    exit 1
fi
