#!/bin/bash
# LatentRAG on Qwen3-0.6B-Base, one GPU. Usage (from anywhere):
#   RUN_DIR=/path/to/run bash scripts/training/train_latentrag_qwen3-0.6b-base_qwen3-embedding-0.6b.sh [hydra overrides...]
# Rerunning with the same RUN_DIR resumes from its latest checkpoint.
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export WANDB_PROJECT="latentrag"
export WANDB_MODE="${WANDB_MODE:-offline}"
# 8 GB GPU: reduce fragmentation (variable-length batches)
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [ -z "$RUN_DIR" ]; then
  echo "Set RUN_DIR to the run's output folder first, e.g. RUN_DIR=\$PWD/data/runs/main"
  exit 1
fi
echo "RUN_DIR=$RUN_DIR"

MODEL_CONFIG="latentrag_qwen3-0.6b-base"
RETRIEVER_CONFIG="qwen3-embedding-0.6b"
DATA_TAG="Search-R1_Qwen2.5-7B"
GPU_ENV_CONFIG="${MODEL_CONFIG}_${RETRIEVER_CONFIG}/gpu_1_8g"
DATA_PATH="${DATA_PATH:-$(pwd)/data/reasoning_steps_subset20k}"
TRAIN_FILES="['nq/train/${DATA_TAG}_${RETRIEVER_CONFIG}_maxturns_4_topk_3_negk_15_reasoning_steps.json', 'hotpotqa/train/${DATA_TAG}_${RETRIEVER_CONFIG}_maxturns_4_topk_3_negk_15_reasoning_steps.json']"

torchrun --rdzv-endpoint 127.0.0.1:12346 \
  --nnodes 1 --nproc_per_node 1 src/train.py \
  --config-path $(pwd)/config \
  --config-name ${MODEL_CONFIG} \
  "+data.data_tag='${DATA_TAG}'" \
  "+data.data_path='${DATA_PATH}'" \
  +data.train_files="$TRAIN_FILES" \
  +data.eval_files=null \
  "retriever=${RETRIEVER_CONFIG}" \
  "retriever.corpus_path='$(pwd)/data/corpus'" \
  gpu_env="${GPU_ENV_CONFIG}" \
  "$@"
