#!/usr/bin/env bash
# Vast.ai onstart script (ssh launch mode replaces the image entrypoint with this).
# Pulls the model to /workspace/hf and starts vLLM in the background.
set -uo pipefail
env | grep _ >> /etc/environment   # make VAST_* / HF_HOME visible to ssh sessions
mkdir -p /workspace/hf
export HF_HOME=/workspace/hf
{
  echo "=== $(date -u) onstart"; nvidia-smi
  hf download nvidia/Qwen3.8-Flash-Next-NVFP4 || huggingface-cli download nvidia/Qwen3.8-Flash-Next-NVFP4
  echo "=== $(date -u) download done"
} >> /workspace/onstart.log 2>&1
# serve.sh is not in the image: paste it in via ssh/scp, or inline its contents here.
[ -f /workspace/serve.sh ] && nohup bash /workspace/serve.sh &
