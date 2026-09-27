# Spark bench — the suite to run after every config change

Model-agnostic: it discovers which of the two deployments is up
([Qwen/vLLM](../qwen3.8-flash-next), [DeepSeek/DSpark](../deepseek-v4-flash-0731)) and captures
its provenance before measuring
anything. **`./bench.py run` does not restart the deployment** — it measures whatever is
running now. (It said the opposite until 2026-09-13; audited against the code, the run path
only scp's the harness, `pkill`s stray `loadgen.py`, and drives the four gates.) It is still
not read-only in effect: it drives the box to c=10 for ~20 minutes and will slow anything
else using the server. Restart first, by hand, when an arm needs a clean boot.

```bash
./bench.py run --ref <label>   # capture provenance, run the suite, store results
./bench.py list                # one line per stored run
./bench.py dashboard           # regenerate dashboard.html
./bench.py restore <ref>       # print (or --apply) the steps back to a run's config
```

Open `dashboard.html` in a browser:

- **click a row** to expand it — every metric at every concurrency level the run measured,
  not just the `c=5` median the summary row shows;
- **click a column header** to sort by it (runs with no scoreable data always sort last);
- **tick two checkboxes** to pop up a diff of their serve flags, env and hotfixes against
  the metric delta, per level as well as at `c=5` — `Esc` or `✕` closes it and clears the
  pair;
- filter by ref / model / sha / note, or hide everything that is not `full` provenance.

## Why this exists

On 2026-08-14 we upgraded `~/dspark-mia` from `cbd719f` to `103af68c` and needed to know
whether it regressed. The only baseline was two `loadgen` runs from 08-12, and **nothing
recorded what config produced them**. Reconstructing it took git-reflog archaeology and
still ended ambiguous: those runs were served under `model=deepseek-v4-flash-0731`, a name
that 404s on the current deployment — proving `.env.dspark` had changed, with no record of
how. Config identity could only be *inferred* from matching cache-hit and prefill numbers.

The cost: a solid finding (issue #27's prefill serialization costs us 24%, proven by a
same-day A/B) sat next to an unfalsifiable one (a residual +11% at c=5) and separating them
would have taken two more 12-minute production boots.

So the rule here is: **a run without provenance is worse than no run** — it looks like
evidence and cannot be checked. `bench.py run` captures provenance *first* and aborts if
any field is missing.

## What gets recorded

`runs/<UTC>__<ref>/manifest.json`, plus `env.dspark` and `local.diff` verbatim:

- `dspark_sha` + `local_diff` — the commit *and* uncommitted local edits (we carry a
  `--override-generation-config '{"top_p":0.95}'` tune that lives only in the working tree)
- `serve_cmd` and parsed `serve_flags` — the resolved serve line, where env defaults,
  compose substitution and CLI overrides have all collapsed. This is ground truth.
- `image` digest, `vllm_version`, `retention_interval`
- `hotfixes` **per rank** — which in-container patches applied. We deliberately hold #26
  and #27 with mounted no-op files, so *absent* is the signal a hold took. Checked on both
  ranks because the worker is a flat directory the head scp's into, not a git checkout;
  capture aborts if the two ranks disagree.
- `served_model_name` — the field whose silent change broke the 08-12 comparison
- `bench_sha` — versions the harness itself, so a harness bug is attributable too

## The suite

| test | gate | catches |
|---|---|---|
| `smoke` | served name present in `/v1/models` | wrong model, dead server |
| `toolprobe` | **10/10 exact tool names** | the #26 prefix-cache corruption — the model paraphrases tool names (`run_shell` → `run_command`/`exec`) with no error, crash or latency change. Nothing else here would see it, and agentic tool calling is the whole opencode workload. |
| `loadgen` | 0 preemptions, all requests complete | throughput/TTFT regressions. Identical request shapes with `ignore_eos` **and pinned prompt seeds** (since 2026-09-12) **on a real code prompt** (since 2026-09-13), so a difference between runs is the config. Before the pin, every run drew a fresh 2,700-token tail and MTP acceptance moved tok/s by up to 17% between reps of one config — runs older than 2026-09-12 carry that noise. Acceptance is now recorded per level (`accept_len` from the server counters, `accept_len_client` from the SSE stream) so content can never again be a hidden variable. |
| `decode_shapes` | recorded, no numeric gate yet | spec-decode/MTP acceptance regressions that loadgen's single output shape cannot see. Gate once enough runs exist to set one honestly. |

Protocol, from `opencode-bench/SESSION-2026-08-12.md`: 18000 prompt tokens (the measured
production mean), 2700 unique tail (keeps prefix-cache hit near the real 80–90%, measured
84.3%), sweep
`1,5,6,10` (c=6 added 2026-09-12 — the production target is six concurrent agents), one
discarded warm-up sweep then 3 counted reps, each rep on its own pinned tail seed
(`TAIL_SEED_BASE + i`) so every arm sees the same three prompt sets. Re-running the same
ref on the same boot finds those tails in the prefix cache; the runner warns when
`cache_hit_pct` says so. The runner kills stray
`loadgen.py` first — an orphan silently competed for the GPU and produced one junk arm
(score 332.9) on 08-12.

### Prompt content: `code` since 2026-09-13

`workload.py` holds the prompt generators, selected by `bench.py run --workload` (default
`WORKLOAD = "code"`) and recorded in the manifest as `workload.kind`. The third kind,
`agentic`, is [below](#prompt-content-agentic-since-2026-09-18--the-production-shape).

Until 2026-09-13 the suite ran one prompt, now called **`filler`**: 18,000 tokens drawn
uniformly from a 38-word list, task *"Summarize the above."*, with `ignore_eos` forcing
300 tokens past the model's natural stop. It is reproducible to the byte, which is what
it was built for — but MTP acceptance is a property of *content*, and on that prompt the
drafter accepts 43% of draft positions against 50% on our live opencode traffic and the
80–97% bilikaz measure on dense code. So `accept_len ≈ 2.3` in every stored run is the
**prompt generator's** number, not the engine's, and any arm that only pays off on
structured content was invisible to this suite.

**`code`** is a synthetic Python service repo (~15.2k shared tokens, hot in the prefix
cache, modelling opencode's real ~11.8k system prompt) plus a per-request cold module and
an implementation task whose answer runs well past `--max-tokens` — so `ignore_eos` never
forces off-distribution continuation. The generated code is not semantically meaningful
and does not need to be: what the drafter sees is the *statistical* structure of code.

Measured paired on one boot, same seed, same prompt size, same output length:

| workload | accept | tok/s @ c=1 | steps/s |
|---|---|---|---|
| `filler` | 2.32 | 46.4 | 20.0 |
| `code` | **3.18** | **63.9** | **20.1** |

**steps/s is unchanged** — the engine does identical work and only acceptance moves,
which is exactly the confound the change removes. Per draft position that is 44% → 73%,
into the band where K=4 (issue #8, closed against on filler's 43%) is worth re-testing.

Consequences:

- **Runs before 2026-09-13 are `filler` and are not comparable to `code` runs.**
  `compare.py` refuses to pair across kinds rather than printing a delta that is really
  the prompt generator. Re-run the reference arm to rebaseline.
- `filler` stays available (`--workload filler`) and is byte-for-byte unchanged, so any
  stored arm can still be re-run on its own content.
- `code` is generated to a character budget via `workload.CHARS_PER_TOKEN`, measured
  against the live tokenizer (4.1725, `POST /tokenize`). loadgen warns when the served
  `prompt_tokens` drifts >5% from target; re-measure with
  `./workload.py --calibrate <url> <model>`.

### Prompt content: `agentic` since 2026-09-18 — the production shape

`code` fixed the *content* problem; it did not fix the *shape*. Prometheus on the head, read
over the 46.6 h after the `lpt1024` boot (2026-09-16 18:31 → 09-18 17:05 UTC, 6,409
requests, four workstation clients), says production looks like this per request:

| | production | `code` protocol |
|---|---|---|
| prompt tokens, mean | 65.9k (57% > 50k, 23% > 100k, 65 requests > 200k) | 18k |
| prefix-cache hit | 94.2% | ~78% |
| uncached tokens per request | mean 3.8k, median ~1.5k | 2.7k, fixed |
| generation tokens | mean 535, tail to 2–5k, ~half reasoning | 300, forced |
| turns ending in a tool call | ~80% | 0 (no tools) |
| sampling | none sent → server `generation_config` | temperature 0 |
| concurrency, by request-time | c=1 32%, c=2 27%, c=3 21%, c=4 11%, c≥5 10% | 1,5,6,10 |
| MTP acceptance, tokens/step | **2.65** (71 / 53 / 42% per position) | **3.15** (80 / 64 / 51%) |

That last row is the reason this kind exists: on `code` every decode number the suite has
produced is ~16% optimistic against what a user gets, and the per-position curve — which
is what decides whether a K+1th draft position pays — is a different curve. The
concurrency row is the other reason: 68% of a production request's life is spent sharing
the box with one to three others, so c=2–3 is where an arm has to win, not c=1 or c=10.

**`agentic`** is one turn of a multi-turn agent session, sent as chat `messages` with a
`tools` array of eight harness-style tools (`read_file`, `edit_file`, `bash`, `grep`, …):

- a harness system prompt and a task (English, ~1.6k tokens), then ~60k tokens of canned
  *exploration* tool loops, all seeded `0` — identical on every run, every session, every
  arm, so it is resident in the prefix cache the way a developer's session is;
- then the session's own tool loops, seeded by `(tail_seed, session)`: turn *t* repeats
  turn *t−1*'s messages verbatim and appends one `(assistant tool_call, tool result)`
  pair, so the only cold tokens are that result plus the 832-token block the engine
  recomputes on every hit (MTP drops it — `prefix-cache-last-block-drop-is-mtp-by-design`).
  Result sizes are drawn from a three-band distribution (short grep/listing, a whole
  module, a big file or test log): builder estimate **mean 3.4k, median 1.9k** uncached
  per turn, turn-0 prompt ~66k;
- the assistant halves of the loops are canned, not the model's replies: a rep has to be
  byte-identical across arms for `compare.py`'s pairing, and a sampled reply is not. The
  cost is that the model's real previous reply is never in the next prompt, so ~100–300
  tokens per turn that production would hit are computed here. Recorded, not corrected.

The run model changes with it (`loadgen.py`): each concurrency slot is a **session of
`--turns` sequential requests** (default 4), turn *t+1* sent when turn *t* finishes, so
after the first burst prefills stagger the way production's do. Turns **stop naturally**
(`ignore_eos` off — forcing 300 tokens past a tool call is not production text), sampling
parameters are **omitted** so the server's `generation_config` applies exactly as it does
for opencode, and a per-request `seed` is pinned so a rep is reproducible on one engine.
TPOT is per token, so `tps_per_req` stays comparable; wall-clock and aggregate tok/s are no
longer fixed-shape and are reported, not gated.

```bash
./bench.py run --ref <label> --workload agentic            # sweep 2,3 · 4 turns · 3 reps
./bench.py run --ref <label> --workload agentic --sweep 1,2,3,6
python3 loadgen.py --workload agentic --sweep 2 --turns 2 --max-tokens 4096 \
    --prompt-tokens 65000 --unique-tokens 2300 --model Qwen3.8-Flash-Next-NVFP4   # on the head
python3 workload.py agentic --turn 2 | head                # print one turn, no server
```

What each level now records on top of the usual: **per-position acceptance** (from
`vllm:spec_decode_num_accepted_tokens_per_pos_total`, kept apart in `metrics.scrape`),
**prefill tok/s on computed tokens** and **uncached tokens per request** (the production
read was 1.5–2.0k tok/s and 3.8k), output p50/p95, tool-call rate, reasoning share of
steps, finish reasons. `loadgen` prints them against the production figures after every
level; `compare.py` pairs `prefill_tps`, `uncached`, `tool_call`, `out_p50` and prints the
per-position curves side by side. The cache-contamination line is per kind
(`workload.contamination_hit_pct`: 92% for `code`, ~97% for `agentic`, whose honest hit is
~95%).

The chars/token ratios were measured against the served tokenizer on 2026-09-19
(`./workload.py --calibrate-agentic <url> <model>`, vLLM `/tokenize`, CPU-side): the
line-numbered reads and grep hits that make up ~85% of the prompt tokenise at 3.4–3.5
chars/token, not the 4.17 of plain code, so the offline build was 17% larger than its
estimate; `workload.py` now carries three ratios (tool results, JSON, prose) and `loadgen`
warns at >5% drift if they go stale. Not yet done, because it needs the live server:
whether the model's turns actually end
in a tool call ~80% of the time on this task (printed per level as `tool_call`). **`agentic` runs are not comparable to
`code` or `filler` runs** — `compare.py` refuses, as before. The dashboard's summary row
is the median at `TARGET_C` (5); for agentic runs regenerate with `TARGET_C=3
./bench.py dashboard` or read the expanded per-level table.

### First `agentic` reference — 2026-09-19, the lpt1024 boot

`runs/2026-09-19T09-00Z__agentic-ref-lpt1024` (loadgen driven by hand on the head,
`bench-agentic-ref.sh` in the run dir, warm-up + 3 reps on seeds 20260921-23, same boot as
`*__lpt1024`). Medians over the reps; the reference every agentic arm pairs against:

| | production (Prometheus, 46.6 h) | c=2 | c=3 |
|---|---|---|---|
| tokens per step | 2.65 | 2.52 | 2.48 |
| acceptance per position | 71 / 53 / 42 % | 69 / 48 / 34 % | 69 / 47 / 33 % |
| output tokens, mean | 535 | 612 | 562 |
| turns ending in a tool call | ~80 % | 100 % | 100 % |
| reasoning share of steps | ~54 % of tokens | 78 % | 84 % |
| uncached tokens per request | 3.8k | 3.4k | 4.5k |
| prefill tok/s | 1.5–2.0k | 1,525 | 1,555 |
| TTFT p50 / p95 | 1.3–2.0 s / 4–5 s | 2.45 / 6.6 s | 2.64 / 4.9 s |
| decode tok/s per stream | ~47 (c=2), ~35 (c=3) from TPOT | 47.6 | 35.2 |
| prefix hit | 94.2 % | 95.2 % | 93.6 % |

The shape lands: acceptance is within 5 % of production (the `code` workload sat at 3.15
tokens/step, 20 % high), prefill runs at production speed, the hit rate matches, and every
turn ends in a tool call. Known deltas: reasoning is heavier than production and the tool-call
rate is 100 % rather than ~80 %, both because the canned task always has another loop to run;
uncached per request at c=3 runs high because the result-size draw has a fat tail
(`_RESULT_SIZES`). Rep-to-rep spread is 45.6–49.0 tok/s at c=2 and 34.3–38.6 at c=3 — ~7 %,
wider than `code`'s, because natural stops make each rep's output length content-driven.
Judge an arm on tokens/step, per-position acceptance, prefill tok/s and TTFT p50 together, not
on tok/s per request alone.

### Prefill profile — 2026-09-19, read-only on the lpt1024 boot

TTFT in production is prefill (queue ≈ 0), and prefill runs at ~2k tok/s. `probe_prefill.py`
with `gpusample.sh` on both boxes (`runs/2026-09-19T09-40Z__prefill-profile`: `phases.json`,
`head.csv`, `worker.csv`) asked what bounds it. Single stream, fully cold prompts, `max_tokens` 1:

| cold prompt | prefill s | tok/s | head GPU busy | SM MHz | W | engine CPU | link |
|---|---|---|---|---|---|---|---|
| 2,332 | 1.38 | 1,687 | — | | | | |
| 8,501 | 3.88 | 2,191 | 80 % | 2,509 | 50 | Worker 202 % + Core 100 % | 1.25 GB/s |
| 32,817 | 14.97 | 2,192 | 92 % | 2,538 | 53 | same | 1.21 GB/s |
| 65,046 | 30.83 | 2,110 | **96 %** (min 95) | 2,505 | 54–56 | same | 1.16 GB/s |
| 65k warm (982 uncached) | 0.57 | 1,713 | | | | | |
| decode, 700 tok | 260 steps in 10.4 s = 25 steps/s | 67 tok/s | 87 % | 2,529 | 34 | Worker 197 % + Core 101 % | 0.20 GB/s |

The worker box reads the same: 94–96 % busy, 55–57 W, 203 % CPU. Idle is 0 %, 2,405 MHz, 9.5 W.

What that rules out, in order:

- **Context length.** 8k → 65k is flat at 2.1–2.2k tok/s, so per-token cost is constant
  (~0.46 ms/token on the pair). The 12 full-attention layers use the indexer (budget 2,048)
  and the 36 linear-attention layers are linear, so nothing here is quadratic. Long prompts are
  slow only because they are long.
- **Chunking.** The per-token cost matches lpt2048 (2,635 tok/s) and the uncapped 8,192-token
  chunks before it within 20 %, so the 832-token chunk is not what holds the rate.
- **CPU / launch.** The GPU executes kernels 96 % of the time with the SM clock at its ceiling;
  the engine's two hot threads sit at the same 200 % + 100 % they hold during decode. There is
  no idle gap to reclaim, as with decode (memory `decode-is-launch-bound-not-bandwidth-bound`).
- **The link.** 1.2 GB/s during prefill is exactly the TP all-reduce volume (48 layers × 2 ×
  2,560 × 2 B ≈ 490 KB/token) on a ~13 GB/s link; ~7,500 all-reduces of 4 MB in 31 s cost a
  few seconds at most.
- **The API.** The warm 65k repeat spends 0.57 s in the engine and 0.67 s end to end: ~0.1 s of
  tokenising and HTTP per 65k prompt.

What it leaves: **the kernels themselves.** ~5 B active parameters per token (10 routed + 1
shared expert of 2,560 × 640 × 3 over 48 layers, plus the GDN and attention projections) is
~10 GFLOP/token; at 2.1k tok/s that is ~21 TFLOP/s on the pair, ~10 TFLOP/s per GB10, under
10 % of its dense bf16 tensor peak and a fraction of its NVFP4 peak. And the box draws 55 W
while "96 % busy" against 34 W in decode and ~9 W idle, so the SMs are occupied but not fed:
kernels that are small, serial or sync-bound rather than compute- or bandwidth-bound. The
suspects, none separable without a per-kernel trace: the Marlin W4A16 MoE path (a decode
kernel; at 832 tokens × 10 of 512 experts each expert GEMM sees ~16 rows), the Triton GDN
chunked scan, the indexer top-k, and the per-layer NCCL all-reduce spin.

**Next step, and it is a boot arm, not a read-only one:** the same recipe with the torch
profiler configured (`--profiler-config`; no behaviour change until `/start_profile` is called),
one cold prefill under the profiler, and the kernel table. That decides between the MoE backend
(the kit's `moe-backend marlin` is +5 % on decode steps; a NVFP4 tensor-core path could be a
multiple on prefill) and the GDN kernels, which no serve flag reaches. Until then the TTFT
lever stays unpicked.

### The kernel table — 2026-09-19, the `prof` boot (lpt1024 + the inert profiler flag)

`bilikaz-cluster-recipe/config/recipe.prof.yaml` is lpt1024 plus
`profiler-config '{"profiler":"torch",...}'` (this image has no `VLLM_TORCH_PROFILER_DIR`; the
profiler object is built lazily on the first `POST /start_profile`). `profile_windows.py` on the
head drove two windows and copied the tables aside
(`runs/2026-09-19T10-30Z__prefill-kernels`): **W1** one cold 8,488-token prefill, `max_tokens` 1
(4.07 s under the profiler vs 3.88 s without: the overhead is small); **W2** a warm 2k prompt plus
96 decode tokens with `ignore_eos`. Memory dipped ~1.5 GiB on each box during a window and came
back. Rank 0's W1 table, self CUDA time by kernel family (3.87 s of CUDA in 4.04 s of wall):

| family | ms | % | what the rows say |
|---|---|---|---|
| Marlin W4A16 MoE GEMM (routed experts) | 1,380 | **36** | 784 calls × 1.60 ms = 8 chunks × 49 layers × {gate_up, down}. gate_up streams 419 MB of expert weight per rank per layer → **262 GB/s, the HBM roofline**; down (K = 320) streams 210 MB → 131 GB/s |
| NCCL all-reduce (`RING_LL`) | 584 | 15 | 1,070 × 0.53 ms; 5.2 MB each (1,024 × 2,560 × 2 B) → ~10 GB/s, the link |
| hyper-connection kernels (`hc_gate_mix`, `hc_combine_norm`) | 444 | 11.5 | elementwise on the residual streams, ~270 µs per call |
| dense bf16 GEMM (cutlass / nvjet via `aten::mm`) | 438 | 11 | 4,288 calls, 119 µs avg: router, indexer and hc projections |
| sparse attention + indexer (`qsa`, persistent top-k) | 292 | 7.5 | the 12 full-attention layers |
| dense Marlin (GDN/attention projections, shared expert) | 247 | 6.4 | |
| GDN (chunk scan, conv1d, norms) | 150–270 | 4–7 | the 36 linear-attention layers |
| MoE sum / act / gating | 132 | 3.4 | |
| elementwise, copies, rest | ~250 | 6.5 | |

Also in the table: `Command Buffer Full`, 917 ms over 3,846 events, the CPU thread blocked because
the GPU launch queue was full. Prefill is GPU-bound end to end, the opposite of decode.

What it settles:

- **GDN is not the prefill cost** (4–7 %). Dead as an arm, which is good: no flag reaches it.
- **The MoE backend arm is bounded at ~+8 % prefill.** gate_up is already at the weight-bandwidth
  roofline, so no kernel beats it; only the down GEMM has headroom (half rate), 49 × 0.8 ms ≈
  39 ms of a 440 ms chunk. The kit measured Marlin +5 % on decode steps over `flashinfer_cutlass`.
  Not worth a boot for TTFT.
- **Prefill's biggest term is weight re-streaming per chunk.** Every 1,024-token chunk touches all
  512 experts (10,240 assignments), i.e. 31 GB of NVFP4 weight per rank per chunk, 157 ms of the
  ~440 ms. That is the mechanism behind the lpt trade (2048 → 1024 cost 8 % of prefill rate): the
  cost scales as 1/chunk. The 0.29 scheduler applies `long-prefill-token-threshold`
  unconditionally (`v1/core/sched/scheduler.py` lines 590 and 984), so a lone request on an idle
  box (70 % of production time) also prefills in 1,024-token chunks and streams the experts 8×
  more often than the 8,192 budget would. A cap that applies only while another request is
  running would give the idle-box TTFT the uncapped rate at no stall cost. That is an image
  patch, not a flag.
- **The all-reduce is link-bound at 15 %.** 5.2 MB in 0.53 ms is the ~13 GB/s link, not spin.
  Only a topology change (pipeline instead of tensor parallel) moves it; not on the table.
- The remaining half (hyper-connections, dense bf16 GEMMs, indexer, projections) is a dozen
  mid-size families with no single owner and no serve flag.

W2 (`w2-decode-profiler_out_0.txt`) mixes the 2,248-token warm prefill with the 96 decode steps
(graph replays `execute_context_0(0)_generation_1(4)`, 4,205 of them), so it is not a clean decode
table; the decode picture stays the one in memory `qwen-decode-lever-is-the-draft-head`.

### The decode table — 2026-09-19, same boot, single stream

`profile_decode.py` (idle gate, streaming request on a cached 2k prompt, profiler started after the
first token, stopped ~200 tokens later: 82 steps, 3.08 tokens/step, acceptance 69 %) gives a clean
decode table on both ranks (`w3-decode-rank{0,1}.txt` in the same run dir). Rank 0, 42.7 ms per
step (23.4 steps/s), attributed with the checkpoint's tensor headers (`model-*.safetensors`):

| family | ms/step | % | what the rows and the checkpoint say |
|---|---|---|---|
| bf16 linears (cutlass split-K + gemv) | 16.0 | **37** | ~2.45 GB of bf16 weight per rank per step: the hyper-connection `input_mix_weight_{down,up}` (`ReplicatedLinear`, `quant_config=None`, **1.34 GB replicated on both ranks**, 217 split-K GEMMs/step), attention q/k/v/o (12 layers, 0.6 GB/rank), shared experts (0.24), router (0.13), the MTP layer's own bf16 (0.14 × 3 passes) |
| routed experts, Marlin NVFP4 | 9.4 | 22 | 102 GEMMs/step = 48 layers × 2 + 3 draft passes × 2 |
| dense Marlin NVFP4 | 5.9 | 14 | 72 GDN projections (2.2 ms) + **4 lm_head reads of 179 MB/rank: 1 verify + 3 draft (3.6 ms)** |
| NCCL all-reduce | 5.0 | 12 | 107 × 47 µs of 5 KB each: pure latency (rank 1 shows 3.3 ms, the rest is rank 0 waiting) |
| elementwise, gating, GDN decode, attention, copies | ~6.4 | 15 | |

The MTP draft has no head of its own in the checkpoint: `nvidia/mtp.py` builds a `ParallelLMHead`
with the model's quant config and loads `lm_head.weight`, the NVFP4 one. So the draft head is
**3 × 179 MB ≈ 2.7 ms, 6 % of the step** on this kit, not the 19 % measured on the Mia fork's
bf16 head. A ranked 16k draft vocabulary (the fork's patch, ported to `mtp.py`) is worth ~+6 %
steps/s single-stream and ~+2 % at c=5.

The decode lever has moved to the bf16 remainder, and the largest block is the hyper-connection
matrices: 48 layers × 4 × 6.55 MB, replicated, 201 GEMMs per step. Per-kernel durations from the
rank-0 trace (`w3-rank0-trace`, parsed on the laptop) at the decode M of 4 rows:

| GEMM | bytes | kernel | µs | GB/s | per step |
|---|---|---|---|---|---|
| hc up / down (bf16) | 6.55 MB | cutlass wmma split-K | 33.5 / 34.2 | ~195 | 201 → 6.9 ms |
| attention qkv (bf16) | 34 MB | cutlass | 152 | 224 | 13 → 2.0 ms |
| attention o_proj (bf16) | 15.7 MB | cutlass | 70 | 224 | 12 → 0.9 ms |
| shared expert gate_up / down (bf16) | 3.3 / 1.6 MB | cutlass split-K | 23 / 11 | 140–150 | 51 → 1.8 ms |
| router gate (bf16) | 2.6 MB | cutlass split-K + reduce | 33 | 80 | 49 → 1.6 ms |
| lm_head (NVFP4) | 179 MB | Marlin | 735 | 244 | 4 → 2.9 ms |
| GDN in_proj qkv+z (NVFP4) | 10.5 MB | Marlin | 58 | ~205 | 36 → 2.1 ms |
| GDN out_proj (NVFP4) | 2.0 MB | Marlin | 24 | ~94 | 36 → 0.9 ms |

Marlin has a floor of roughly 20 µs at these sizes (the persistent kernel on 48 SMs), so the
byte-based estimate does not carry through. Quantizing the hyper-connections to NVFP4 takes each
GEMM from ~34 µs to ~22 µs (1.85 MB with scales, between the two GDN points): **−2.4 ms/step**.
The attention projections give another **−1.8 ms**. The shared-expert down (1.6 MB, 11 µs in
cutlass) and the router would get *slower* under Marlin, so they stay bf16. Net: about **−4 ms of
42.7, +10 % steps/s single-stream, roughly +5 % at c=5**, plus ~1.9 GB less resident weight per
rank. A third of what the byte budget promised. Quality cost of NVFP4 on the sigmoid-gate
matrices still unmeasured.

The planned in-container microbench (`hc_probe.py`) could not run: a second CUDA context next to
the serving process fails at creation with `cudaErrorMemoryAllocation` on both boxes' shape of
memory (25 GiB MemAvailable, 1.5 GiB MemFree), so no GPU side-experiment is possible while the
cluster serves. Anything at the kernel level goes through the profiler endpoints or a trial boot.

Everything runs **on the head node**. `loadgen` must hit `127.0.0.1:8000`; driving it from
a laptop measures the LAN as much as the engine, and the WSL→box path returns empty replies
while the box itself serves 200s.

## Reading the numbers honestly

Score is `5·ttft_p95 + 1500/tok-s-per-req` at c=5 (`agg.py`), lower is better — 5 LLM calls
per task, 1500 output tokens across the whole task. Scores are computed live from the stored
loadgen JSON, so changing those constants rescales every run at once; only score numbers
quoted *outside* the suite go stale. With 2 reps
per run an exact permutation test cannot go below p=0.167, so **nothing here reaches
significance on its own**. Trust: non-overlapping ranges, a mechanism (the queue-vs-prefill
split is what identified #27), and same-day A/Bs. Distrust cross-day comparisons — that is
what this suite exists to make checkable.

## Legacy rows

`./bench.py import-legacy` brought in 14 pre-suite arms from `opencode-bench/bench_results`,
all flagged `unverified` and visually distinct on the dashboard. Two known traps recorded in
their manifests:

- the four `mia-*-0812` arms were served as `deepseek-v4-flash-0731` — internally consistent
  with each other, **not** comparable to anything served as `deepseek-v4-flash-dspark`
- arms predating the 08-12 `loadgen` TTFT fix have null timings and cannot be scored at all;
  they show as `n=0`

## Stdlib only

No venv, no dependencies. That constraint is what lets the same files run unchanged on the
head node. `metrics.py` is `scrape`/`server_delta`/`DEFAULT_METRICS` extracted from
`opencode-bench/opencode_bench.py` so we do not drag an `opencode` CLI dependency into a
suite that never runs agents.

Self-check: `python3 test_loadgen.py` (all three prompt builders, the agentic session
runner and request-body policy, SSE parsing including tool-call and reasoning deltas,
per-position acceptance from the counters — no network) and `python3 test_dashboard.py`
(the per-level median the expanded row shows).
