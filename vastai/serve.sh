#!/usr/bin/env bash
# vLLM serve line for nvidia/Qwen3.8-Flash-Next-NVFP4 on 2x RTX PRO 6000.
# Edit flags here, then: bash /workspace/serve.sh  (logs to /workspace/serve.log)
set -euo pipefail
export HF_HOME="${HF_HOME:-/workspace/hf}"
MODEL="${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
EXTRA="${EXTRA:-}"   # e.g. EXTRA='--speculative-config {"method":"mtp","num_speculative_tokens":3} --no-enable-flashinfer-autotune'

pkill -f 'vllm serve' || true
sleep 2
exec vllm serve "$MODEL" \
  --served-model-name Qwen3.8-Flash-Next-NVFP4 \
  --tensor-parallel-size 2 \
  --quantization modelopt \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --max-num-seqs "${MAX_SEQS:-12}" \
  --max-num-batched-tokens "${MAX_BATCHED:-8192}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --mamba-ssm-cache-dtype bfloat16 \
  --async-scheduling \
  --long-prefill-token-threshold "${LPT:-1024}" \
  --host 0.0.0.0 --port 8000 \
  $EXTRA \
  >> /workspace/serve.log 2>&1
