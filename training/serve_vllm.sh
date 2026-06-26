#!/usr/bin/env bash
# Serve the fine-tuned planner behind an OpenAI-compatible API with vLLM.
#
#   MODE=lora   (default) 4-bit base (the exact weights the adapter was trained on) + LoRA adapter,
#               served as "fedsearch-qwen3.8-27b"; fits a 40 GB GPU
#   MODE=merged  merged 16-bit weights from training/merge_adapter.py
#   MODE=base    the untuned base model (for the before/after comparison in the eval)
#
# Structured output: PydanticAI sends response_format=json_schema and vLLM enforces it with guided decoding.
# Tool-call flags are included so FEDSEARCH_LLM_OUTPUT_MODE=tool also works.
# Needs a recent vLLM with Qwen3.5/3.8 support (0.16 lacks it).
set -euo pipefail

BASE=${BASE:-unsloth/Qwen3.8-27B-unsloth-bnb-4bit}
ADAPTER=${ADAPTER:-outputs/qwen3.8-27b-planner/adapter}
MERGED=${MERGED:-outputs/qwen3.8-27b-planner/merged}
NAME=${NAME:-fedsearch-qwen3.8-27b}
PORT=${PORT:-8000}
COMMON=(--port "$PORT" --max-model-len 8192 --gpu-memory-utilization 0.90
        --enable-auto-tool-choice --tool-call-parser hermes --enable-prefix-caching)

case "${MODE:-lora}" in
  lora)   exec vllm serve "$BASE" --quantization bitsandbytes --served-model-name base --enable-lora --max-lora-rank 16 \
               --lora-modules "$NAME=$ADAPTER" "${COMMON[@]}" ;;
  merged) exec vllm serve "$MERGED" --served-model-name "$NAME" "${COMMON[@]}" ;;
  base)   exec vllm serve "$BASE" --quantization bitsandbytes --served-model-name base "${COMMON[@]}" ;;
  *) echo "unknown MODE"; exit 1 ;;
esac
