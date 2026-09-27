# Results — Qwen3.8-Flash-Next-NVFP4 on 2× RTX PRO 6000 (Vast.ai 51798033)

One row per (run, concurrency level). `agentic` workload = `bench/loadgen.py --workload agentic`
(65k-token tool-calling coding session, 2.3k uncached tokens per turn, natural stops, server-default
sampling). Raw JSON per run in `runs/<name>/`. Spark reference rows from `bench/README.md`
(lpt1024 boot, hibrid48 checkpoint, MTP k=3).

Columns: **TTFT** is loadgen client-side p50 over all turns (turn 0 is a cold 64k prefill, later
turns are warm); **decode** is tok/s per request (client); **agg** is aggregate tok/s; **prefill**
is tok/s on computed (uncached) tokens; **accept** is MTP tokens/step and per-position %.

## Baseline config (boot #3)

`serve.sh`: TP=2, modelopt, max-model-len 131072, max-num-seqs 12, max-num-batched-tokens 8192,
gpu-util 0.90, mamba-ssm-cache bf16, async-scheduling, long-prefill-token-threshold 1024,
MTP k=3, no flashinfer autotune, qwen3 reasoning parser, qwen3_coder tool parser.
MoE: FLASHINFER_CUTLASS NVFP4; draft MoE: TRITON FP8 (untuned config).

| run | c | turns×reps | TTFT p50 / p95 | decode /req | agg | prefill | accept | cache hit | out p50 | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Spark ref | 2 | 4×3 | 2.45 / 6.6 s (warm) | 47.6 | — | 1,525 | 2.52 · 69/48/34 | 95 % | 612 | production hardware |
| Spark ref | 3 | 4×3 | 2.64 / 4.9 s (warm) | 35.2 | — | 1,555 | 2.48 · 69/47/33 | 94 % | 562 | |
| agentic-c2-mtp3-boot3 | 2 | 2×1 | 9.98 / 10.46 s (½ cold) · warm 0.68 s | 109.7 | 87 | 6,558 | 2.58 · 69/47/35 | 48 % | 459 | 19:36 UTC; only 2 turns so half the requests are cold |
| sweep1-baseline-boot3 | 1 | 3×1 | 0.68 / 0.73 s | 190.1 | 125 | 7,285 | 3.01 · 72/50/39 | 97 % | 216 | 19:39 UTC; prefix warm from earlier run |
| sweep1-baseline-boot3 | 2 | 3×1 | 0.86 / 1.73 s | 138.3 | 211 | 7,790 | 2.65 · 69/51/37 | 93 % | — | |
| sweep1-baseline-boot3 | 4 | 3×1 | 1.15 / 2.05 s | 104.4 | 269 | 4,897 | 2.64 · 69/48/33 | 94 % | — | |
| sweep1-baseline-boot3 | 6 | 3×1 | 1.02 / 2.53 s | 91.9 | 332 | 4,052 | 2.64 · 68/47/33 | 95 % | — | |
| sweep1-baseline-boot3 | 8 | 3×1 | 0.94 / 2.47 s | 87.1 | 468 | 4,746 | 2.65 · 70/50/36 | 95 % | — | |
| sweep1-baseline-boot3 | 12 | 3×1 | 0.96 / 3.10 s | 70.5 | 517 | 4,129 | 2.61 · 69/48/34 | 94 % | — | max-num-seqs=12 ⇒ ceiling of this config; queue 0.01 s |

## seqs48 config (boot #4): max-num-seqs 48, max-num-batched-tokens 16384, otherwise baseline

| run | c | turns×reps | TTFT p50 / p95 | decode /req | agg | prefill | accept | cache hit | notes |
|---|---|---|---|---|---|---|---|---|---|
| sweep2-seqs48 | 8 | 3×1 | 1.19 / **32.1** s | 76.3 | 171 | 2,263 | 2.84 · 71/51/37 | 64 % | **cold cache after restart**: 8 simultaneous 64k prefills; the 9am case |
| sweep2-seqs48 | 12 | 3×1 | 1.03 / 2.92 s | 74.4 | 482 | 3,460 | 2.83 · 71/52/37 | 95 % | = sweep1 c=12 (70.5) |
| sweep2-seqs48 | 16 | 3×1 | 1.35 / 4.47 s | 51.9 | 511 | 3,237 | 2.76 · 70/51/37 | 94 % | |
| sweep2-seqs48 | 24 | 3×1 | 1.14 / 4.03 s | 47.7 | 513 | 3,323 | 2.77 · 71/50/37 | 95 % | = Spark at c=2 (47.6) |
| sweep2-seqs48 | 32 | 3×1 | 1.31 / 6.58 s | 45.7 | 566 | 3,276 | 2.79 · 71/51/37 | 93 % | |
| sweep2-seqs48 | 48 | 3×1 | 1.18 / 6.88 s | 42.9 | 576 | 3,696 | 2.71 · 70/50/36 | 94 % | no preemption, queue 0; server-side running count below |

### same config, loadgen run **on the box** (sweep3) — valid to c=24; c≥32 capped by loadgen's thread pool (see log)

| run | c | turns×reps | TTFT p50 / p95 | decode /req | agg | prefill | accept | cache hit | server max running | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| sweep3-onbox-seqs48 | 16 | 3×1 | 0.66 / 3.69 s | 62.8 | 430 | 3,308 | 2.75 · 70/49/35 | 95 % | 24 | laptop run said 51.9 |
| sweep3-onbox-seqs48 | 24 | 3×1 | 0.85 / 4.97 s | 45.6 | 654 | 2,627 | 2.75 · 70/50/36 | 95 % | 24 | ≈ Spark c=2 per user |
| sweep3-onbox-seqs48 | 32 | 3×1 | 1.02 / 9.46 s | 31.2 | 625 | 2,172 | 2.78 · 71/52/39 | 94 % | 32 | KV 27 % |
| sweep3-onbox-seqs48 | 48 | 3×1 | 0.89 / 11.63 s | 32.2 | 725 | 2,203 | 2.73 · 70/50/36 | 94 % | **32** | client capped at 32 in flight — rerun after loadgen fix |

### same config, on-box, loadgen executor fixed (sweep4) — **valid at every level**

| run | c | turns×reps | TTFT p50 / p95 | queue | decode /req | agg | prefill | accept | cache hit | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| sweep4-onbox-seqs48-fixed | 32 | 3×1 | 1.17 / 5.64 s | 0.02 s | 34.3 | 727 | 2,224 | 2.78 · 71/51/38 | 95 % | |
| sweep4-onbox-seqs48-fixed | 48 | 3×1 | 1.07 / 9.44 s | 0.31 s | 25.2 | 730 | 1,897 | 2.78 · 71/50/37 | 95 % | aggregate saturated ≈ 730 tok/s; queueing begins |

**Ceiling, baseline kernels (FLASHINFER_CUTLASS + MTP k=3):** aggregate decode saturates at ~730 tok/s.
Holding the Spark c=2 experience (≥ 45 tok/s per user, TTFT p95 ≤ 5 s) ⇒ **~24 concurrent coding
sessions**; at the Spark c=3 experience (≥ 35 tok/s) ⇒ **~32**. KV cache was 27 % used at c=32 (prefix-shared
workload); distinct 65k contexts would fit ~40 before preemption.

## Arm `mtp2` (boot #7): MTP k=2, otherwise seqs48 config, on-box, executor fixed

| run | c | turns×reps | TTFT p50 / p95 | decode /req | agg | prefill | accept | cache hit | vs k=3 (same level) |
|---|---|---|---|---|---|---|---|---|---|
| arm-mtp2 | 4 | 3×1 | 0.73 / 17.6 s | 93.6 | 104 | 3,993 | 2.46 · 75/54 | 64 % | throwaway, cold cache |
| arm-mtp2 | 12 | 3×1 | 0.65 / 3.04 s | 68.0 | 566 | 3,616 | 2.40 · 71/51 | 95 % | 74.4 → **−9 %** |
| arm-mtp2 | 24 | 3×1 | 0.79 / 4.71 s | 43.9 | 592 | 2,681 | 2.40 · 71/51 | 95 % | 45.6 → −4 % |
| arm-mtp2 | 32 | 3×1 | 0.93 / 12.2 s | 32.3 | 726 | 1,889 | 2.39 · 71/51 | 93 % | 34.3 → −6 %, TTFT p95 worse |

**Verdict: k=3 wins at every level.** The third draft position's 37 % acceptance still buys more than its
forward costs, even at saturation. Keep `num_speculative_tokens: 3`.

## Arm `nomtp` (boot #8): no speculative decoding, otherwise seqs48 config, on-box

| run | c | turns×reps | TTFT p50 / p95 | decode /req | agg | cache hit | vs MTP k=3 (decode · TTFT p95) |
|---|---|---|---|---|---|---|---|
| arm-nomtp | 4 | 3×1 | 0.44 / 16.2 s | 76.0 | 100 | 65 % | throwaway, cold cache |
| arm-nomtp | 12 | 3×1 | 0.51 / 1.27 s | 51.7 | 392 | 96 % | 74.4 · 2.92 s → **−31 % decode, −56 % TTFT p95** |
| arm-nomtp | 24 | 3×1 | 0.59 / 3.30 s | 37.9 | 553 | 96 % | 45.6 · 4.97 s → −17 % decode, −34 % TTFT p95 |
| arm-nomtp | 32 | 3×1 | 0.60 / 4.24 s | 32.9 | 711 | 96 % | 34.3 · 5.64 s → −4 % decode, −25 % TTFT p95 |

**Verdict:** MTP k=3 is worth +44 % per-user decode at c=12 and +20 % at c=24, at the price of a longer
TTFT tail (the draft head runs on every prefill chunk and the untuned Triton FP8 config). At c=32 the box
is compute-saturated and the draft no longer pays. For a 24-seat target, keep MTP; tuning the draft's
Triton config (`E=512,N=320,…block_shape=[64,64].json`) is the obvious next win for TTFT.

## Arm `batch32k-ctx262k` (boot #9): max-num-batched-tokens 32768, max-model-len 262144, MTP k=3, on-box

| run | c | turns×reps | TTFT p50 / p95 | decode /req | agg | prefill | accept | cache hit | vs 16k batch (same level) |
|---|---|---|---|---|---|---|---|---|---|
| arm-batch32k-ctx262k | 4 | 3×1 | 0.48 / 18.9 s | 99.8 | 151 | 4,099 | 2.66 · 70/49/34 | 65 % | throwaway, cold cache |
| arm-batch32k-ctx262k | 12 | 3×1 | 0.71 / 5.19 s | 59.7 | 476 | 3,701 | 2.72 · 70/51/38 | 94 % | 74.4 · 2.92 s → **−20 % decode, TTFT p95 +78 %** |
| arm-batch32k-ctx262k | 24 | 3×1 | 0.93 / 6.00 s | 42.1 | 634 | 2,952 | 2.74 · 70/50/37 | 94 % | 45.6 · 4.97 s → −8 %, TTFT p95 +21 % |
| arm-batch32k-ctx262k | 32 | 3×1 | 0.99 / 6.65 s | 38.8 | 757 | 2,125 | 2.72 · 70/49/35 | 95 % | 34.3 · 5.64 s → **+13 % decode**, TTFT p95 +18 % |

**Verdict:** a 32k prefill budget lets cold chunks crowd out decode steps below saturation (worse at
12 and 24) and only pays once the box is compute-bound (32). 262k context fits: KV 2,401,301 tokens,
9 simultaneous max-length requests. Keep 16k for a ≤24-seat deployment; 32k if running at the edge.

## Winner for a coding-agent deployment on 2× RTX PRO 6000

`seqs48` config = TP 2, FLASHINFER_CUTLASS NVFP4, MTP k=3, max-num-seqs 48, max-num-batched-tokens
16384, gpu-util 0.90, max-model-len 131072 (262144 also fits), async scheduling, LPT 1024.
