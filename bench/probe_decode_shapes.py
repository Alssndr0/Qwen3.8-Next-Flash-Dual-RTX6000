#!/usr/bin/env python3
"""
probe_decode_shapes.py — decode rate vs draft acceptance, by output shape.

Adapted from jvr0x/dgx-spark-bench
(recipes/deepseek-v4-flash-0731-dual/probe-decode-shapes.py), rewritten onto
stdlib + this repo's `scrape` so it runs with no venv, like loadgen.py.

Why this exists: on a DSpark spec-decode profile there is no single decode rate.
The engine step rate is near-constant; throughput is that rate multiplied by how
many drafted tokens survive verification, and acceptance is a property of the
TEXT BEING GENERATED. loadgen.py measures one shape only — filler prompt forced
past EOS — which is the least draftable end of the range. This maps the range so
loadgen's numbers can be read as a floor rather than an estimate.

All rates come from vLLM's own counters (generation_tokens_total over
request_decode_time_seconds_sum), so the client is not in the measurement path.
That is also why this does not need to stream.

    ./probe_decode_shapes.py            # one pass, ~3 min, single-stream
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

from metrics import DEFAULT_METRICS, scrape

URL = "http://10.150.0.50:8000/v1/chat/completions"
MODEL = "deepseek-v4-flash-dspark"

GEN = "vllm:generation_tokens_total"
DEC_S = "vllm:request_decode_time_seconds_sum"
DRAFTS = "vllm:spec_decode_num_drafts_total"
DRAFT_TOK = "vllm:spec_decode_num_draft_tokens_total"
ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
KEYS = (GEN, DEC_S, DRAFTS, DRAFT_TOK, ACCEPTED)

FILLER = "Summarise the following text.\n\n" + "benchmark " * 192
LIST_INSTR = "\nReturn exactly 128 numbered lowercase English words, then stop."
PROSE = (
    "Explain how tensor parallelism splits a transformer layer across two GPUs, "
    "then give a short worked example."
)
CODE = (
    "Write a Python function that parses a Prometheus text-format metrics body "
    "into a dict, with a docstring and inline comments."
)


def snapshot(metrics_url: str) -> dict[str, float]:
    m = scrape(metrics_url, timeout=30) or {}
    return {k: m.get(k, 0.0) for k in KEYS}


def busy(metrics_url: str) -> float:
    m = scrape(metrics_url, timeout=10) or {}
    return m.get("vllm:num_requests_running", 0.0) + m.get(
        "vllm:num_requests_waiting", 0.0
    )


def run(label: str, prompt: str, metrics_url: str, **params) -> dict | None:
    """One request; reports server-measured decode rate and draft acceptance."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], **params}
    before = snapshot(metrics_url)
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            resp = json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"{label:<44} FAILED: {e}")
        return None
    wall = time.perf_counter() - t0
    after = snapshot(metrics_url)

    d = {k: after[k] - before[k] for k in KEYS}
    out_tok = resp.get("usage", {}).get("completion_tokens", 0)
    # A step emits the accepted draft tokens plus the always-emitted bonus token,
    # so tokens/step is (accepted + drafts) / drafts.
    per_step = (d[ACCEPTED] + d[DRAFTS]) / d[DRAFTS] if d[DRAFTS] else float("nan")
    accept = d[ACCEPTED] / d[DRAFT_TOK] if d[DRAFT_TOK] else float("nan")
    rate = d[GEN] / d[DEC_S] if d[DEC_S] else float("nan")
    print(
        f"{label:<44} out={out_tok:>4}  decode={rate:6.1f} tok/s  "
        f"accept={accept:6.1%}  tok/step={per_step:4.2f}  "
        f"steps/s={rate / per_step if per_step else float('nan'):5.1f}  wall={wall:5.1f}s"
    )
    return {
        "label": label,
        "out_tokens": out_tok,
        "decode_tok_s": rate,
        "acceptance": accept,
        "tok_per_step": per_step,
        "steps_s": rate / per_step if per_step else None,
        "wall_s": wall,
    }


def main() -> int:
    # ponytail: rebind the globals rather than thread url/model through run() and its
    # nine call sites. Single-shot CLI, no concurrency, so there is nothing to race.
    global URL, MODEL
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-url", default=DEFAULT_METRICS)
    p.add_argument("--url", default=URL, help="chat/completions endpoint")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", default=None, help="write results JSON here")
    p.add_argument("--force", action="store_true", help="run even if server is busy")
    args = p.parse_args()
    URL, MODEL = args.url, args.model

    if not args.force and (n := busy(args.metrics_url)) > 0:
        print(f"server busy ({n:.0f} req in flight) — this probe adds load. Use --force.")
        return 1

    rows = []
    run("warmup (discarded)", PROSE, args.metrics_url, max_tokens=64, temperature=0.0)

    print("\n-- highly draftable output --")
    rows.append(run("numbered word list, t=0.6 natural", FILLER + LIST_INSTR,
                    args.metrics_url, max_tokens=600, temperature=0.6, top_p=0.95))
    rows.append(run("numbered word list, t=1.0 forced 512", FILLER + LIST_INSTR,
                    args.metrics_url, max_tokens=512, min_tokens=512,
                    ignore_eos=True, temperature=1.0))

    print("\n-- free-form prose --")
    rows.append(run("prose answer, t=0.0 natural", PROSE, args.metrics_url,
                    max_tokens=512, temperature=0.0))
    rows.append(run("prose answer, t=1.0 natural", PROSE, args.metrics_url,
                    max_tokens=512, temperature=1.0))

    print("\n-- code output (closest to real opencode traffic) --")
    rows.append(run("code answer, t=0.0 natural", CODE, args.metrics_url,
                    max_tokens=512, temperature=0.0))

    print("\n-- loadgen.py's shape (filler prompt, forced past EOS) --")
    rows.append(run("filler summarise, forced 300", FILLER, args.metrics_url,
                    max_tokens=300, min_tokens=300, ignore_eos=True, temperature=1.0))
    rows.append(run("filler summarise, forced 512", FILLER, args.metrics_url,
                    max_tokens=512, min_tokens=512, ignore_eos=True, temperature=1.0))

    rows = [r for r in rows if r]
    if rows:
        lo = min(rows, key=lambda r: r["decode_tok_s"])
        hi = max(rows, key=lambda r: r["decode_tok_s"])
        print(
            f"\nrange: {lo['decode_tok_s']:.1f} tok/s ({lo['label']}) "
            f"-> {hi['decode_tok_s']:.1f} tok/s ({hi['label']}) "
            f"= {hi['decode_tok_s'] / lo['decode_tok_s']:.2f}x from output shape alone"
        )
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"url": URL, "model": MODEL, "rows": rows}, f, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
