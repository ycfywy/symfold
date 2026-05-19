#!/usr/bin/env bash
# 在所有标准测试集上 eval SymFold
#
# 用法:
#   bash scripts/run_eval.sh model/<task>/best.pt
#   bash scripts/run_eval.sh model/<task>/best.pt 20 1 0.0  # num_steps=20, samples=1, beta=0
#   bash scripts/run_eval.sh model/<task>/best.pt 20 5 0.5  # 多 seed + physics guidance

set -e
cd "$(dirname "$0")/.."

CKPT="${1:?需要 ckpt 路径作为第一参数}"
NUM_STEPS="${2:-20}"
NUM_SAMPLES="${3:-1}"
PHYSICS_BETA="${4:-0.0}"
LAMBDA_PK="${5:-0.0}"

source /root/aigame/dannyyan/miniconda3/bin/activate RNADiffFold_torch260

mkdir -p output/eval_results
OUT_JSON="output/eval_results/$(basename ${CKPT%.pt})_eval.json"

python eval/eval.py \
    --ckpt "$CKPT" \
    --test_sets bpRNA,RNAStrAlign,bpRNA-new,ArchiveII,PDB_TS1,PDB_TS2,PDB_TS3,PDB_TS_hard \
    --num_steps $NUM_STEPS \
    --num_samples $NUM_SAMPLES \
    --physics_beta $PHYSICS_BETA \
    --physics_lambda_pk $LAMBDA_PK \
    --out_json "$OUT_JSON"

echo
echo "Results JSON: $OUT_JSON"
