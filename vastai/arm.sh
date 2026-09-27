#!/usr/bin/env bash
# Restart vLLM on the Vast box with a named tuning arm and record kill->ready time.
# Usage: bash vastai/arm.sh <arm-name> [EXTRA flags...]   (env MAX_MODEL_LEN, GPU_UTIL, MAX_SEQS, MAX_BATCHED, LPT pass through)
set -uo pipefail
NAME="${1:?arm name}"; shift
EXTRA="$*"
HOST=151.237.25.16; PORT=33800; API=http://$HOST:34118
SSH="ssh -i $HOME/.ssh/vastai -o StrictHostKeyChecking=no -p $PORT root@$HOST"
LOG=/Users/alessandro/Development/Qwen3.8-Next-Flash-Dual-RTX6000/vastai/boots.tsv
[ -f "$LOG" ] || printf "arm\tkill_utc\tready_utc\tsecs\textra\n" > "$LOG"
t0=$(date -u +%s); kill_utc=$(date -u +%H:%M:%S)
$SSH "mv /workspace/serve.log /workspace/serve-before-$NAME.log 2>/dev/null; : > /workspace/serve.log; export EXTRA='$EXTRA' MAX_MODEL_LEN='${MAX_MODEL_LEN:-131072}' GPU_UTIL='${GPU_UTIL:-0.90}' MAX_SEQS='${MAX_SEQS:-12}' MAX_BATCHED='${MAX_BATCHED:-8192}' LPT='${LPT:-1024}'; setsid nohup bash /workspace/serve.sh </dev/null >/dev/null 2>&1 & sleep 2; echo relaunched" 2>/dev/null
for i in $(seq 1 120); do
  sleep 5
  if curl -sf -m 4 "$API/health" >/dev/null && $SSH "grep -q 'startup complete' /workspace/serve.log" 2>/dev/null; then break; fi
  if $SSH "tail -50 /workspace/serve.log | grep -q -E 'Engine core initialization failed|Traceback'" 2>/dev/null; then echo "BOOT FAILED — see serve.log"; $SSH "grep -E 'Error|Traceback' /workspace/serve.log | tail -5"; exit 1; fi
done
t1=$(date -u +%s); ready_utc=$(date -u +%H:%M:%S)
printf "%s\t%s\t%s\t%s\t%s\n" "$NAME" "$kill_utc" "$ready_utc" "$((t1-t0))" "$EXTRA" >> "$LOG"
echo "$NAME ready in $((t1-t0)) s"
$SSH "grep -E 'NvFp4 MoE backend|Fp8 MoE backend|GPU KV cache size' /workspace/serve.log | tail -3 | cut -c1-160"
