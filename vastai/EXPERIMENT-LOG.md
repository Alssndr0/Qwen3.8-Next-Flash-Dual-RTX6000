# Experiment log — 2× RTX PRO 6000 on Vast.ai vs the DGX Spark pair

Structured record for the write-up. All times UTC. Raw sources: `logs/onstart.clean.log`
(image boot + model pull), `logs/serve-mtp3.log` (both vLLM boots, appended), `runs/` for
loadgen output. Spark comparison figures come from `../bench/README.md`.

## Machine

| | Vast.ai 51798033 (offer 48905148) | DGX Spark pair (production) |
|---|---|---|
| GPU | 2× NVIDIA RTX PRO 6000 Blackwell **Server Edition**, 97,887 MiB each, SM120 | 2× GB10 (SM121), 128 GB unified each |
| interconnect | PCIe gen5 x16, same NUMA node, **no NVLink** | 2 boxes over ConnectX (RDMA), Ray head + worker |
| host | AMD EPYC (64c/128t visible), 252 GB RAM, 300 GB overlay disk | — |
| driver / CUDA | 595.84 / 13.0 (image 13.0.2) | — |
| location | Bulgaria, BG | on-prem |
| price | $3.12/h GPU + $0.13/GB-mo disk ⇒ **$3.18/h** all-in; $1.37/TB down | capex |
| software | `vllm/vllm-openai:nightly-d05da62e…` = vLLM `0.29.1rc1.dev422+gd05da62e9`, torch 2.13.0+cu130 | `myllmbox/qwen38-flash-next-cluster-vllm` v5, vLLM 0.29.0 |
| checkpoint | `nvidia/Qwen3.8-Flash-Next-NVFP4` stock, 123.6 GiB, TP=2 | `myllmbox/Qwen3.8-Flash-Next-hibrid48` (custom), TP across 2 nodes |
| MoE kernel | `FLASHINFER_CUTLASS` NVFP4 **W4A4** (auto; TRTLLM + CuTeDSL rejected on SM120) | `marlin` W4A16 |

## Timeline

| clock (UTC) | Δ | event |
|---|---|---|
| 19:15:13 | 0 | `vastai create instance` accepted (contract 51798033) |
| 19:15:39 | +0:26 | status `loading`, docker layers pulling |
| 19:17:29 | +2:16 | image loaded (9.7 GB), container `running`, onstart starts |
| 19:17:29 | | `hf download` of 25 files / 123.6 GiB begins |
| 19:22:59 | +7:46 | download done — **5.5 min ≈ 3.0 Gb/s** (host advertised 2.2) |
| 19:23:0x | | boot #1 (no MTP) starts |
| 19:23:38 | | engine init |
| 19:25:05 | | weights loaded: **64.5 s** (cold page cache) |
| 19:25:40 | | KV profile done: 44.8 GiB/GPU, 3,488,939 tokens |
| 19:26:11 | +10:58 | API up — **boot #1 = ~3 min** from serve start |
| 19:26:3x | | smoke test OK (haiku), streaming probe 108 tok/s |
| 19:27:28 | | SIGTERM boot #1 (restart for MTP) |
| 19:27:37 | | boot #2 (MTP k=3) API process up |
| 19:28:02 | | engine init |
| 19:29:26 | | weights loaded: **50.0 s** (warm page cache) |
| 19:29:30 | | draft (MTP) weights loaded: 3.4 s; FP8 blocks 128→64 for TP shard |
| 19:30:04 | | KV profile: 42.9 GiB/GPU, 2,864,545 tokens |
| 19:30:31 | | API up — **boot #2 = 2 min 54 s** kill-to-ready |
| 19:31:3x | | loadgen agentic c=2 → all HTTP 400 (`tool_choice` needs `--enable-auto-tool-choice`) |
| 19:3x | | boot #3: + `--enable-auto-tool-choice --tool-call-parser qwen3_coder` (see below) |

Spark reference for the same operation: **~12 min per production boot** (`bench/README.md`,
"two more 12-minute production boots").

## Measurements

### Single stream, 800-token reasoning reply, 60-token prompt, streaming

| config | TTFT | decode tok/s | tokens/step | accept pos 0/1/2 |
|---|---|---|---|---|
| boot #1, no MTP | 0.18 s | 108.3 (×2 runs identical) | 1.0 | — |
| boot #2, MTP k=3 | 0.14–0.16 s | 137.1 / 139.4 | 1.99 | 53 / 30 / 16 % |
| Spark, MTP k=3 (prefill-profile decode, 700 tok) | — | 67 | — | — |

MTP metrics after the two probes: 2,409 drafts, 795 accepted (428 / 240 / 127 by position).
Spark's agentic-workload reference is 2.52 tok/step at 69 / 48 / 34 %, on the hibrid48 checkpoint
and a tool-calling prompt — different workload, do not compare until the agentic run exists.

GPU state during single-stream decode: 87.8 GB used per card, ~170 W per card (600 W cap).

## Issues hit

1. `serve.sh`'s `pkill -f 'vllm serve'` makes the ssh session that launches it exit 255; harmless,
   the new server comes up. Use `setsid nohup … </dev/null`.
2. `tool_choice: "auto"` in the agentic workload → HTTP 400 without `--enable-auto-tool-choice
   --tool-call-parser qwen3_coder`. The Spark serve line has both; the first recipe here did not.
3. Triton FP8 MoE for the MTP draft has no tuned config for this GPU
   (`E=512,N=320,…RTX_PRO_6000_Blackwell_Server_Edition,dtype=fp8_w8a8,block_shape=[64,64].json`).
4. `workload.py agentic --turn N` prints the body *without* `tools` (by design, line 728), so a
   hand-replayed turn 400s for a different reason than loadgen's. Misleading when debugging.

## Cost so far

Credit 9.97 → 9.01 at 19:32 UTC (17 min ≈ $0.96, matches $3.18/h + transfer).

## Config facts (boot #3 onward)

- sampling: server-default `generation_config.json` = temperature 1.0, top_k 20, top_p 0.95
  (Spark carries `--override-generation-config '{"top_p":0.95}'`; same effective sampling)
- attention: Qwen4Exp QSA (sparse) kernels warmed for decode shapes (1..4,1) and sparse shapes up to (64,33,8)
- cudagraph capture sizes: 1,2,4,8,16,24,…,96 (default set; Spark tuned 12 FULL decode graphs)
- `b12x` (SM120 kernels) **not installed** in the nightly image; `pip install "vllm[b12x]"` is an arm
- FlashInfer 0.6.18.post1 + cu130 JIT cache present
- boot #3: kill 19:33:20 → API 19:36:32 = **3 min 12 s** (`boots.tsv` records later arms)
- boot #4 (`seqs48`: max-num-seqs 48, max-num-batched-tokens 16384, MTP k=3): kill 19:43:45 → API 19:47:20
  = **215 s**; 43 piecewise + 25 full cudagraph sizes instead of 15 + 7, KV cache 39.2 GiB / 2,619,243 tokens

## Finding: client-side bottleneck above c≈16 when loadgen runs from the laptop

Server log during sweep2 (10 s samples): at nominal c=24/32/48 the engine reports **Running ≤ 18,
Waiting 0, KV usage ≤ 17 %**. The server is idle-capable; the client is the limiter — every turn
uploads a ~230 KB prompt from the laptop over a home uplink, so ≥ 24 sessions serialise on upload
and TTFT is measured through it. sweep2 rows for c ≥ 24 are therefore *lower bounds* only.
Fix: run `loadgen.py` on the box (as the Spark runs did on the head), copied to `/workspace/bench`.
Spark bench note applies here too: the client must sit next to the server.

## Tuning arms planned (each = one ~3.5 min reboot via `arm.sh`, then on-box sweep at the levels that matter)

1. `b12x` — `--moe-backend b12x` (SM120-native NVFP4 MoE; installed 19:49 UTC, cutlass-dsl re-pinned 4.7.1)
2. `mtp2` — MTP k=2: at high c the third draft position accepts only ~36 %, so its cost may exceed its gain
3. `lpt4096` — `--long-prefill-token-threshold 4096` for cold-prefill throughput (the 9am case)
4. `ctx262k` — `--max-model-len 262144` to confirm the full native context fits (KV usage peaked at 17 % at c=48)

**Correction (20:03 UTC):** the in-flight cap is not the network. Server "Running" maxes at exactly
18 from the laptop (14 cores → default `ThreadPoolExecutor` = min(32, 14+4) = 18) and exactly 32
on the box (256 threads → 32). `loadgen.py` drives blocking HTTP through asyncio's default executor,
so nominal c above the pool size is silently serialised. This also bounds every Spark `bench` run
ever made at c ≤ 18 from a laptop-class head — worth checking in `bench/README.md` history.
Fix: size the executor to c + margin.
- boot #5 `b12x` (`--moe-backend b12x` + MTP k=3): kill 20:04:14 → **FAILED 20:05:45** —
  `moe_backend='b12x' is not supported for FP8 MoE`. The flag is global and the MTP draft experts are
  block-FP8, so b12x and MTP are mutually exclusive in this build. Server was down 20:05–20:15
  (arm.sh's failure grep missed the message; fixed). Retrying b12x without MTP.
- boot #6 `b12x-nomtp` (`--moe-backend b12x`, no MTP): kill 20:14:50 → weights loaded 20:16:23 →
  **FAILED 20:17:03** during profiling: `Cuda error custom_all_reduce.cuh:164 'an illegal memory access
  was encountered'` on both ranks, engine `RuntimeError: cancelled`. b12x 1.3.0 + vLLM nightly d05da62 at
  TP=2 is not usable; consistent with vLLM excluding it from auto-selection ("upstream CUTLASS SM121 MMA op
  guard"). **b12x arm abandoned.** ~13 min of box time spent on the two b12x attempts.
- boot #7 `mtp2` (CUTLASS, MTP k=2, seqs48/16k): kill 20:21:15 → API 20:24:05 = **170 s**; KV 2,686,601 tokens
  (k=2 draft needs fewer slots than k=3's 2,619,243)
- plan trimmed by Alessandro at 20:20: remaining arms = mtp2 → nomtp → batch32k+ctx262k; cold-prefill case
  and repeats dropped
- boot #8 `nomtp` (CUTLASS, no speculative decoding, seqs48/16k): kill 20:28:22 → API 20:31:01 = **159 s**;
  KV 3,314,682 tokens (no draft slots)
- boot #9 `batch32k-ctx262k` (CUTLASS, MTP k=3, max-num-seqs 48, max-num-batched-tokens 32768,
  max-model-len 262144): kill 20:35:28 → API 20:38:15 = **167 s**; KV 2,401,301 tokens = 9.1× 262k requests. Credit $5.71 at 20:35 UTC.

## Teardown

- all `runs/`, `logs/` (every serve log, onstart log, GPU sampler CSV) copied off the box 20:42 UTC
- `vastai destroy instance 51798033` at **20:43:27 UTC**; verified: 0 instances, volumes `[]`, endpoints `[]`,
  workergroups `[]`. Nothing left billing.
- **Session cost: credit 9.97 → 5.23 = $4.74** for 19:15:13 → 20:43:27 (**88 min**), i.e. $3.18/h + $0.18 model pull
  + transfer. 9 boots (2 failed), 8 load-test runs, 5 configurations measured.
