# Qwen3.8-Flash-Next-NVFP4 on 2× RTX PRO 6000 (Vast.ai)

Research notes, 2026-09-20. Nothing here has been run on a rented box yet.

## What we are serving

`nvidia/Qwen3.8-Flash-Next-NVFP4` (HF, not gated, license: NVIDIA Open Model License
over Qwen Community License 1.0):

| | |
|---|---|
| architecture | `Qwen4ExpForConditionalGeneration` — 48 layers (36 GatedDeltaNet linear + 12 sparse full attention), 512 experts / top-10 + shared expert, gated residual (`hc_count` 4), PLE n-gram embedding, 1-layer MTP head, vision encoder |
| params | 125B total / 6B active, plus 51B n-gram embedding and 4B MTP |
| precision | routed experts W4A4 NVFP4; attention + shared experts BF16; MTP experts 128×128 block FP8; PLE per-tensor FP8 |
| checkpoint | **132.7 GB**, 11 safetensors shards |
| context | 262,144 native |
| vLLM | needs commit `d4d703c` (2026-09-03) for the FP8 PLE, and PR #55513 (merged 2026-09-08) for MTP. **Both are in `v0.29.1rc0`, `v0.30.0rc1/rc2` and every `nightly` image since.** |

NVIDIA's card says B200/B300 with TP=8. That is what they tested, not a hard requirement:
the checkpoint is 133 GB and two RTX PRO 6000 give 192 GB, so weights fit at TP=2 with
~25 GB per card left for KV cache, activations and CUDA graphs. Nothing in the model
config forces TP=8 (24 attention heads, 2 KV heads, 512 experts — all divisible by 2).

## How this differs from the Spark deployment

The bench suite in `../bench` was written against the current production box: two DGX
Sparks (GB10, SM121) running `myllmbox/qwen38-flash-next-cluster-vllm` (vLLM 0.29.0) and
serving a *custom* checkpoint `myllmbox/Qwen3.8-Flash-Next-hibrid48` under the name
`Qwen3.8-Flash-Next-NVFP4`, with `--moe-backend marlin` (W4A16, no real FP4 tensor cores).
Production numbers to beat, from `bench/README.md`: prefill ~2.1k tok/s, decode ~67 tok/s
single-stream, MTP k=3 at 2.65 tokens/step, TTFT p50 1.3–2.0 s at 65k prompts.

RTX PRO 6000 is SM120 (compute cap 12.0), 96 GB GDDR7, ~1.4 TB/s each — roughly 5× the
memory bandwidth of a GB10. The interesting questions for the experiment:

1. Does the stock NVIDIA checkpoint load and run at TP=2 on SM120 at all?
2. Which NVFP4 MoE kernel does vLLM pick on SM120 (auto order in
   `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`: flashinfer_trtllm →
   flashinfer_cutlass → cutedsl → vllm_cutlass → marlin → emulation; the SM120-specific
   `b12x`/`flashinfer_b12x` backend is opt-in only via `--moe-backend`). Look for the
   `Using ... backend for NVFP4 MoE` log line at boot and A/B `--moe-backend
   flashinfer_cutlass`, `b12x` (needs `pip install "vllm[b12x]"`), `marlin`.
3. Does MTP (k=1 per NVIDIA, k=3 as on Spark) work at TP=2 without
   `--enable-expert-parallel`? NVIDIA needs EP at TP=8 so the 128×128 FP8 blocks of the MTP
   experts (intermediate 640) stay whole; at TP=2 each shard is 320 wide, still >128, so it
   may just work. Try without EP first, add `--enable-expert-parallel` if loading fails.
4. Decode and prefill tok/s vs Spark at the `agentic` workload, using the existing harness.

## Vast.ai: what the docs say

CLI: `curl -fsSL https://vast.ai/install.sh | bash` or `pip install vastai`. Key from
https://cloud.vast.ai/manage-keys/ then `vastai set api-key KEY`; `vastai show user` to
verify. `vastai search offers` works **without** a key. Register an SSH key once:
`vastai create ssh-key ~/.ssh/id_ed25519.pub`.

Launch modes: `--ssh` (Vast replaces the image entrypoint; your serve command goes in
`--onstart-cmd` or `--onstart file`), `--jupyter`, or entrypoint mode (`--entrypoint` /
`--args`, no SSH). Ports: `--env '-p 8000:8000'` maps a container port; Vast assigns an
external port, read it from `vastai show instance ID` or `$VAST_TCP_PORT_8000` inside the
container. Disk is fixed at creation (`--disk GB`) and cannot grow. Use versioned image
tags. `vastai stop` keeps storage billing; `vastai destroy` ends everything.

Search filter fields we care about: `gpu_name` (spaces → `_`), `num_gpus`, `gpu_ram` (MB),
`cuda_vers`, `driver_version`, `disk_space`, `inet_down`, `pcie_bw`, `bw_nvlink`,
`geolocation`, `verified`, `reliability`, `dph_total`. Sort with `-o dph_total`.

## Dual RTX PRO 6000 offers on 2026-09-20

Vast names the card `RTX PRO 6000 WS` (workstation, 600 W) or `RTX PRO 6000 S` (server
edition). 15 rentable 2-GPU offers, all verified, all CUDA ≥13.0, no NVLink on any of
them (expected — the card has none; TP=2 traffic goes over PCIe). Hosts also bill
internet transfer per TB and stopped-instance storage per GB-month; the "pull" column is
what the one-off 133 GB model download costs on that host. Benchmark traffic afterwards is
tiny (an agentic sweep moves well under 1 GB), so only the pull matters.

| id | gpu | $/h | $/TB down | 133 GB pull | disk $/GB/mo | PCIe | down Mb/s | where |
|---|---|---|---|---|---|---|---|---|
| 42515614 | WS | 2.91 | 2.73 | 0.36 | 0.20 | gen5 x16 | 854 | New York |
| 42515647 | WS | 2.85 | 2.73 | 0.36 | 0.20 | gen5 x16 | 815 | New York |
| 48905148 | S | 3.12 | 1.37 | 0.18 | 0.13 | gen5 x16 | 2201 | Bulgaria |
| 51032797 | S | 3.33 | 1.37 | 0.18 | 0.08 | gen5 x16 | 4273 | Switzerland |
| 47849041 | S | 2.80 | 2.67 | 0.35 | 0.20 | gen5 x16 | 1132 | Australia |
| 51775833 | WS | 2.94 | 40.96 | 5.45 | 0.80 | gen5 x16 | 3634 | Norway |
| 44623529 | S | 3.74 | 40.00 | 5.32 | 0.53 | gen5 x16 | 7910 | Czechia |
| 50041229 | WS | 2.27 | 13.33 | 1.77 | 0.87 | gen4 x16 | 322 | Beijing |
| 51246926 | WS | 2.71 | 6.67 | 0.89 | 0.27 | gen5 **x8** | 842 | Oregon |

Pick criteria: gen5 x16 on both cards (TP=2 all-reduce every layer), ≥1 Gb/s down (133 GB
model pull: ~20 min at 1 Gb/s, ~3 min at 8 Gb/s), ≥300 GB disk (image ~10–17 GB + model
133 GB + HF cache headroom), CUDA ≥13.0 (the vLLM `nightly` image is CUDA 13.0.2). The
bandwidth charge never exceeds two hours of GPU time, so it is not a deciding factor on its
own, but the fast Czechia/Norway/Hungary hosts charge $40/TB and their hourly price moves
(Czechia went 2.99 → 3.74 within the hour). Bulgaria 48905148 is the balanced pick: 2.2 Gb/s
(~8 min pull), $0.18 for the pull, $0.13/GB-month if stopped. New York 42515614 is the
cheapest x16 offer with a sane bandwidth rate; note its 278 GB disk is below the 300 GB
we want, so check `disk_space` before renting.

Re-run the search (IDs churn):

```bash
vastai search offers 'gpu_name in ["RTX_PRO_6000_WS","RTX_PRO_6000_S"] num_gpus=2 rentable=true pcie_bw>40 inet_down>1000 disk_space>300 internet_down_cost_per_tb<10' -o dph_total
```

## Recipe

Image: `vllm/vllm-openai:nightly` (9.7 GB, CUDA 13.0.2, built with `TORCH_CUDA_ARCH_LIST`
including 12.0, FlashInfer JIT cache included). Pin the exact tag that is current when you
rent, e.g. `vllm/vllm-openai:nightly-d05da62e9ccdf8e342b15bf6785d83224cc165af`
(2026-09-20), and record it — the bench suite's provenance rule applies here too.

```bash
# 1. rent (manual, per the plan); --disk 300, ssh mode, expose 8000
vastai create instance OFFER_ID \
  --image vllm/vllm-openai:nightly-d05da62e9ccdf8e342b15bf6785d83224cc165af \
  --disk 300 --ssh --direct \
  --env '-p 8000:8000 -e HF_HOME=/workspace/hf' \
  --onstart vastai/onstart.sh \
  --label qwen38-nvfp4-2xrtx6000

# 2. watch it come up
vastai show instance ID          # status loading -> running, external port for 8000
vastai ssh-url ID                # ssh root@HOST -p PORT
ssh ... 'tail -f /workspace/serve.log'
```

`onstart.sh` (next to this file) pulls the model with `hf download` to `/workspace/hf` and
starts vLLM in the background; `serve.sh` is the serve line by itself so it can be re-run
by hand after edits. First boot is the model pull plus torch.compile and CUDA-graph
capture, so allow 20–40 min before the health check answers.

Serve line, first attempt (adapted from NVIDIA's TP=8 command and the Spark flags):

```bash
vllm serve nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name Qwen3.8-Flash-Next-NVFP4 \
  --tensor-parallel-size 2 \
  --quantization modelopt \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --max-model-len 131072 \
  --max-num-seqs 12 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --mamba-ssm-cache-dtype bfloat16 \
  --async-scheduling \
  --long-prefill-token-threshold 1024 \
  --port 8000
```

Then, once that boots: add `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
(with `--no-enable-flashinfer-autotune`, which NVIDIA requires with spec decode), then raise
`--max-model-len` to 262144, then sweep `--moe-backend`.

Smoke test from the laptop (`PORT` from `vastai show instance`):

```bash
curl http://HOST:PORT/v1/models
curl http://HOST:PORT/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next-NVFP4","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

Then point `bench/loadgen.py --workload agentic` at it. `bench.py run` will not work as-is:
its provenance capture scp's into the Spark head/worker and reads `.env.dspark`, none of
which exist here. Plan: a `vastai` provenance backend that records the image digest, offer
id, `nvidia-smi` and the serve line from `serve.sh`.

## Cost

~$3/h for the pair. A full session (boot 30 min, MTP and backend A/Bs, an agentic sweep at
c=2,3 with 3 reps ≈ 1 h) is about 2–3 h, so ~$10 per experiment day if the box is destroyed
after. Keep it `stop`ped (disk only, ~$0.5/GB/month on most hosts) only if the model cache
is worth more than 20 min of re-download.

## Sources

- Model card: https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4
- vLLM PR #55513 (MTP fix): https://github.com/vllm-project/vllm/pull/55513
- vLLM b12x SM120 backend docs: `docs/features/quantization/b12x.md` on vLLM main
- Vast.ai CLI quick start: https://docs.vast.ai/cli/get-started
- Vast.ai templates / launch modes: https://docs.vast.ai/instances/templates, https://docs.vast.ai/instances/launch-modes
- RTX PRO 6000 + vLLM community recipe (Qwen3.6, single-GPU, CUDA 13): https://github.com/lastloop-ai/vllm-blackwell-guide
- NVFP4 on RTX PRO 6000 throughput write-up: https://jarvislabs.ai/blog/nvfp4-rtxpro-6000

## Session log

### 2026-09-20 — first rent, instance 51798033 (offer 48905148, Bulgaria)

- created 20:15 UTC+2 with the command above; `running` at 20:17 (image pull ~2.5 min)
- box: 2× `NVIDIA RTX PRO 6000 Blackwell Server Edition` 97,887 MiB, driver 595.84, PCIe gen5 x16,
  GPU0↔GPU1 `NODE` (same NUMA, no NVLink), 300 GB disk, $3.18/h all-in
- image reports vLLM `0.29.1rc1.dev422+gd05da62e9`, torch 2.13.0+cu130
- ssh: `ssh -i ~/.ssh/vastai -p 33800 root@151.237.25.16` (direct) or `ssh5.vast.ai:38032` (proxy);
  vLLM API on `http://151.237.25.16:34118`
- model pull started 20:18 at ~2 Gb/s; `serve.sh` scp'd to `/workspace/serve.sh` before the pull
  finished so `onstart.sh` starts it automatically
- pull done 19:23 UTC (5.5 min, 124 GB on disk); vLLM ready 19:26 — weights 65 s, compile + graphs ~2.5 min
- **auto-selected NVFP4 MoE backend: `FLASHINFER_CUTLASS`** (TRTLLM and CuTeDSL rejected on SM120); W4A4, not Marlin
- KV cache 44.8 GiB per GPU = 3,488,939 tokens (26.6× 131k requests); 87.8 GB used per card at util 0.90
- smoke (no MTP, single stream, 800 tok, 59-token prompt): **TTFT 0.18 s, decode 108 tok/s**, ~170 W per card
  — Spark reference is 67 tok/s *with* MTP k=3
- next: `EXTRA='--speculative-config {"method":"mtp","num_speculative_tokens":3} --no-enable-flashinfer-autotune' bash /workspace/serve.sh`
- **MTP k=3 boot (19:27–19:30 UTC, 3 min)**: draft experts loaded at TP=2 *without* `--enable-expert-parallel`
  — vLLM refined the FP8 block scales 128×128 → 64×64 for the 320-wide shard, draft MoE on `TRITON` FP8
  with an untuned default config (no `E=512,N=320,...RTX_PRO_6000...fp8_w8a8,block_shape=[64,64].json`).
  KV cache 42.9 GiB / 2,864,545 tokens.
- MTP probe (same 800-token reasoning prompt, 2 runs): **TTFT 0.15 s, decode 137–139 tok/s** (vs 108 without)
  — acceptance 795/2409 drafts = **1.99 tokens/step**, per position 53 / 30 / 16 %. Spark's agentic
  reference is 2.52 tok/step at 69/48/34 %, but on a different checkpoint and workload, so not yet comparable;
  candidates: server-default sampling temperature, the 64×64 scale refinement, the untuned Triton config.
