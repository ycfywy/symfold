#!/usr/bin/env bash
# 启动 SymFold 训练 (后台 + 独立进程组 + 独立 stderr)
#
# 必须用 setsid + 独立 stderr, 避免:
#   1) codebuddy/ssh shell 退出时连带杀进程
#   2) C 层 SIGFPE 错误信息被丢失
#
# 用法:
#   bash scripts/run_train.sh
#   bash scripts/run_train.sh config/another_config.json

set -e
cd "$(dirname "$0")/.."

CONFIG="${1:-config/train_config.json}"
TASK_NAME=$(python -c "import json; print(json.load(open('$CONFIG'))['task_name'])")

mkdir -p logs

setsid bash -c "
    source /root/aigame/dannyyan/miniconda3/bin/activate RNADiffFold_torch260
    export PYTHONPATH=..
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    exec python -u train/train.py '$CONFIG'
" < /dev/null \
  > "logs/${TASK_NAME}.stdout.log" \
  2> "logs/${TASK_NAME}.stderr.log" &

PID=$!
sleep 2
echo "Launched SymFold training:"
echo "  PID:    $PID"
echo "  task:   $TASK_NAME"
echo "  stdout: logs/${TASK_NAME}.stdout.log"
echo "  stderr: logs/${TASK_NAME}.stderr.log"
echo "  log:    logs/${TASK_NAME}.log"
echo "  heartbeat: logs/${TASK_NAME}.heartbeat"
echo
echo "  follow with: tail -f logs/${TASK_NAME}.log"
ps -eo pid,pgid,sid,etime,cmd | grep -E "python.*train.py" | grep -v grep | head -5 || true
