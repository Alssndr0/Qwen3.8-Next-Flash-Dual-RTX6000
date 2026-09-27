#!/usr/bin/env python3
"""vllm /metrics scraping — the only honest source of the TTFT/prefill/decode split.

Extracted verbatim from opencode-bench/opencode_bench.py (DEFAULT_METRICS, scrape,
server_delta). loadgen.py and probe_decode_shapes.py need exactly these three; importing
them from the full agentic harness would drag in an `opencode` CLI dependency for a suite
that never runs agents.

Client-side timings say what a caller felt. These counters say what the engine did — the
queue-vs-prefill split here is what identified the issue #27 prefill serialization on
2026-08-14, when both configs looked identical from the client side.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request

DEFAULT_METRICS = "http://127.0.0.1:8000/metrics"


def scrape(url: str, timeout: float = 5.0) -> dict[str, float] | None:
    """Prometheus text format -> {metric_name: value}, labels collapsed by sum."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError):
        return None
    out: dict[str, float] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, val = line.rpartition(" ")
        try:
            v = float(val)
        except ValueError:
            continue
        # Per-position acceptance is the one labelled series worth keeping apart: it
        # is what says whether a K+1th draft position would pay. Stored under
        # `<name>[pos=k]` next to the label-collapsed sum.
        pos = _POS.search(name)
        if pos:
            key = f"{name[: name.index('{')]}[pos={pos.group(1)}]"
            out[key] = out.get(key, 0.0) + v
        name = re.sub(r"\{[^}]*\}", "", name).strip()
        out[name] = out.get(name, 0.0) + v
    return out


_POS = re.compile(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{.*position="(\d+)"')


def server_delta(before: dict | None, after: dict | None, wall_s: float) -> dict | None:
    """Counter deltas over one concurrency level -> per-request means."""
    if not before or not after:
        return None

    def d(k: str) -> float:
        return after.get(k, 0.0) - before.get(k, 0.0)

    def mean(stem: str) -> float | None:
        n = d(f"{stem}_count")
        return (d(f"{stem}_sum") / n) if n > 0 else None

    reqs = d("vllm:time_to_first_token_seconds_count")
    gen = d("vllm:generation_tokens_total")
    prompt = d("vllm:prompt_tokens_total")
    cached = d("vllm:prompt_tokens_cached_total")
    tpot = mean("vllm:request_time_per_output_token_seconds")
    # Spec-decode acceptance over the window. accepted/drafts is the mean number of
    # draft tokens accepted per verify step; +1 for the target's own token gives tokens
    # per engine step, the same figure stepprobe reads off the SSE stream. Without this
    # a content-driven acceptance swing (17% between two reps on 2026-09-06) is
    # indistinguishable from a config effect -- see TUNING.md #12.
    drafts = d("vllm:spec_decode_num_drafts_total")
    accepted = d("vllm:spec_decode_num_accepted_tokens_total")
    per_pos = []
    if drafts > 0:
        k = 0
        while f"vllm:spec_decode_num_accepted_tokens_per_pos_total[pos={k}]" in (after or {}):
            per_pos.append(d(f"vllm:spec_decode_num_accepted_tokens_per_pos_total[pos={k}]") / drafts)
            k += 1
    # Prefill throughput on the tokens the engine actually computed (prompt minus the
    # prefix-cache hit); the 2026-09-18 production read was 1.5-2.0k tok/s, alone or not.
    computed = d("vllm:request_prefill_kv_computed_tokens_sum")
    prefill_s = d("vllm:request_prefill_time_seconds_sum")
    return {
        "requests": reqs,
        "ttft_mean": mean("vllm:time_to_first_token_seconds"),
        "queue_mean": mean("vllm:request_queue_time_seconds"),
        "prefill_mean": mean("vllm:request_prefill_time_seconds"),
        "decode_mean": mean("vllm:request_decode_time_seconds"),
        "tpot_mean": tpot,
        "decode_tps_per_req": (1.0 / tpot) if tpot else None,
        "prompt_tokens": prompt,
        "gen_tokens": gen,
        "cache_hit_pct": (cached / prompt * 100) if prompt else None,
        "gen_tps_server": (gen / wall_s) if wall_s else None,
        "prompt_tps_server": (prompt / wall_s) if wall_s else None,
        "preemptions": d("vllm:num_preemptions_total"),
        "spec_drafts": drafts,
        "spec_accepted": accepted,
        "accept_len": (accepted / drafts + 1.0) if drafts > 0 else None,
        "accept_per_pos": per_pos or None,
        "uncached_tokens_mean": (computed / reqs) if reqs > 0 else None,
        "prefill_tps": (computed / prefill_s) if prefill_s > 0 else None,
    }
