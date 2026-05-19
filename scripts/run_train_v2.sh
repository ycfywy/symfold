#!/usr/bin/env bash
# SymFold v2 训练启动脚本
#
# 用法:
#   bash scripts/run_train_v2.sh
#   bash scripts/run_train_v2.sh config/train_config_v2.json

set -e
cd "$(dirname "$0")/.."

CONFIG="${1:-config/train_config_v2.json}"
TASK=$(python -c "import json; print(json.load(open('$CONFIG'))['task_name'])")

echo "Starting SymFold v2 training: task=$TASK config=$CONFIG"
echo "Logs: logs/${TASK}.{log,stderr.log}"

mkdir -p logs model output

source /root/aigame/dannyyan/miniconda3/bin/activate RNADiffFold_torch260
export PYTHONPATH=..
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 显式关闭 TF32/BF16, 纯 fp32
export NVIDIA_TF32_OVERRIDE=0

setsid bash -c "
    source /root/aigame/dannyyan/miniconda3/bin/activate RNADiffFold_torch260
    export PYTHONPATH=..
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NVIDIA_TF32_OVERRIDE=0
    exec python -u train/train_v2.py $CONFIG
" < /dev/null \
  > "logs/${TASK}.stdout.log" \
  2> "logs/${TASK}.stderr.log" &

echo "PID=$! PGID=$(ps -o pgid= -p $!)"
echo "Monitor: tail -f logs/${TASK}.log"
echo "Heartbeat: cat logs/${TASK}.heartbeat"
