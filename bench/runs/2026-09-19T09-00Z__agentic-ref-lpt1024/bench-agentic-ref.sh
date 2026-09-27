#!/bin/bash
# First agentic reference on the lpt1024 boot: bench.py's exact loadgen flags for PROTOCOLS["agentic"]
# (sweep 2,3 · 65000 prompt · 2300 unique · max_tokens 4096 · 4 turns), warm-up + 3 counted reps on the
# same seeds as every kit arm (20260920 base). No quality gates: same boot as lpt1024, already gated.
set -u
cd ~/bench-suite || exit 1
D=trial-agentic-ref-lpt1024; rm -rf $D; mkdir -p $D/warmup $D/counted
MODEL=Qwen3.8-Flash-Next-NVFP4
base="python3 -u loadgen.py --url http://127.0.0.1:8000/v1/chat/completions --metrics-url http://127.0.0.1:8000/metrics --model $MODEL --sweep 2,3 --prompt-tokens 65000 --unique-tokens 2300 --max-tokens 4096 --workload agentic --turns 4"
echo "$(date -u +%FT%TZ) warmup (seed 20260920)"
$base --tail-seed 20260920 --outdir $D/warmup > $D/warmup.log 2>&1 || echo "warmup FAILED rc=$?"
for s in 20260921 20260922 20260923; do
  echo "$(date -u +%FT%TZ) counted rep seed $s"
  $base --tail-seed $s --outdir $D/counted > $D/rep-$s.log 2>&1 || echo "rep $s FAILED rc=$?"
done
echo "$(date -u +%FT%TZ) BENCH-DONE"
