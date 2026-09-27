#!/usr/bin/env bash
# Smoke test against the Vast instance. Usage: bash vastai/smoke.sh http://151.237.25.16:34118
set -euo pipefail
URL="${1:?base url}"
M=Qwen3.8-Flash-Next-NVFP4
curl -sf "$URL/v1/models" | python3 -c "import sys,json; print('models:', [m['id'] for m in json.load(sys.stdin)['data']])"
t0=$(date +%s.%N)
curl -sf "$URL/v1/chat/completions" -H 'content-type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a haiku about PCIe.\"}],\"max_tokens\":400}" \
 | python3 -c "
import sys,json,time; d=json.load(sys.stdin); c=d['choices'][0]['message']
print('reasoning:', (c.get('reasoning_content') or c.get('reasoning') or '')[:200].replace('\n',' '))
print('content:', c['content'][:300]); print('usage:', d['usage'])"
t1=$(date +%s.%N); echo "wall: $(python3 -c "print(round($t1-$t0,2))") s"
echo "--- spec-decode / cache metrics:"
curl -sf "$URL/metrics" | grep -E "^vllm:(spec_decode_num_(draft|accepted)_tokens_total|prefix_cache_(hits|queries)_total|generation_tokens_total)" | head
