#!/bin/bash
# Qwen3-0.6B port: REST retriever on the CPU IVF-PQ index with exact re-ranking.
# Same API as launch_qwen3-embedding-0.6b.sh (port 8001). Extra Hydra overrides pass through, e.g.
#   bash scripts/launch_retriever_server/launch_qwen3-embedding-0.6b_ivfpq.sh ivfpq.load_encoder=false
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

python src/launch_retriever_server_ivfpq.py \
  --config-path "$(pwd)/config" \
  --config-name retriever_server_ivfpq \
  "$@"
