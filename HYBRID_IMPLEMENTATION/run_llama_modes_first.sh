#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc
conda activate nayak_env
cd /home2/venkatesh.nayak/work/Hybrid_Implementation

export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=180
export HF_HUB_ETAG_TIMEOUT=30
export HYBRID_CACHE_ROOT=/tmp/hybrid_cache
export PYTHONDONTWRITEBYTECODE=1

INPUT="climate_hallucination_dataset_test_230_v2.json"
TIMEOUT_SECS="7200"

run_one () {
  MODEL="$1"
  MODE="$2"
  OUT="eval/hybrid_${MODEL}_${MODE}_full_dataset_answers.json"

  echo "============================================================"
  echo "START: model=${MODEL} mode=${MODE}"
  echo "OUTPUT: ${OUT}"
  echo "============================================================"

  timeout "${TIMEOUT_SECS}" python eval/run_full_dataset.py \
    --input "${INPUT}" \
    --model "${MODEL}" \
    --retrieval-mode "${MODE}" \
    --use-gpu \
    --output "${OUT}"

  echo "DONE: model=${MODEL} mode=${MODE}"
}

# Run all modes with Llama first (as requested)
run_one llama seq_rag_first
run_one llama parallel
run_one qwen seq_rag_first
run_one qwen parallel

echo "ALL MODES COMPLETED"
