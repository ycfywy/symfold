#!/usr/bin/env bash
# SymFold v3 训练启动脚本
#
# 用法:
#   bash scripts/run_train_v3.sh [config_path]
#   默认 config: train/config/train_config_v3.json

set -e
cd "$(dirname "$0")/.."

CONFIG="${1:-train/config/train_config_v3.json}"
TASK=$(python -c "import json; print(json.load(open('$CONFIG'))['task_name'])")
LOG_DIR="logs/$TASK"

mkdir -p "$LOG_DIR"

source /root/aigame/dannyyan/miniconda3/bin/activate RNADiffFold_torch260

echo "Starting SymFold v3 training: $TASK"
echo "Config: $CONFIG"
echo "Logs: $LOG_DIR/"

nohup python -u train/train_v3.py "$CONFIG" \
    >> "$LOG_DIR/$TASK.stdout.log" \
    2>> "$LOG_DIR/$TASK.stderr.log" &

PID=$!
echo "PID: $PID"
sleep 2

if ps -p $PID > /dev/null 2>&1; then
    echo "Process running. Check logs:"
    echo "  tail -f $LOG_DIR/$TASK.log"
    echo "  cat $LOG_DIR/$TASK.heartbeat"
else
    echo "ERROR: Process died immediately. Check stderr:"
    tail -20 "$LOG_DIR/$TASK.stderr.log"
fi
