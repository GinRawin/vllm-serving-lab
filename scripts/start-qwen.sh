#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# FlashInfer's sampler requires nvcc for JIT compilation. Use vLLM's
# supported alternative sampler on machines without the CUDA toolkit.
export VLLM_USE_FLASHINFER_SAMPLER=0

exec "$project_dir/.venv/bin/vllm" serve Qwen/Qwen3-0.6B \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.8 \
    "$@"
