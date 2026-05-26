#!/bin/bash
# SymFold v5 训练启动脚本
# 用法: bash scripts/run_train_v5.sh

set -e

cd /root/aigame/dannyyan/RNADiffFold/symfold
source /root/aigame/dannyyan/miniconda3/bin/activate RNADiffFold_torch260

TASK="260525-161400-v5-train"
CONFIG="train/config/train_config_v5.json"

mkdir -p logs/${TASK}
mkdir -p model/${TASK}
mkdir -p output/${TASK}

echo "[$(date)] Starting SymFold v5 training: ${TASK}"
echo "Config: ${CONFIG}"
echo "Full eval: every 20 epochs -> output/${TASK}/full_eval/"

nohup python -u train/train_v5.py ${CONFIG} \
    >> logs/${TASK}/${TASK}.stdout.log \
    2>> logs/${TASK}/${TASK}.stderr.log &

PID=$!
echo "[$(date)] Started PID=${PID}"
echo "Logs: logs/${TASK}/${TASK}.stdout.log"

sleep 8
if ps -p ${PID} > /dev/null 2>&1; then
    echo "[OK] Process running (PID=${PID})"
    tail -5 logs/${TASK}/${TASK}.stdout.log 2>/dev/null || true
else
    echo "[FAIL] Process died"
    tail -40 logs/${TASK}/${TASK}.stderr.log 2>/dev/null || true
    exit 1
fi
