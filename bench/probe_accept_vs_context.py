#!/usr/bin/env python3
"""
probe_accept_vs_context.py — draft acceptance vs prompt length, content held fixed.

Why this exists: the recipe's headline MTP numbers (72.8% acceptance, 52.1 tok/s
batch-1) were measured at ~1k context, and its own draft-vocab table reports 56.5%
for the same config at 1k. Our workload is 18k prompts, where we measure 48-50%
greedy. Those are three different numbers for one engine, and the only variable
this probe holds still is the one nobody varied deliberately: how much context the
drafter is conditioned on.

Same task text at every length; the prefix is loadgen's filler, so the shared-prefix
shape matches the suite. Greedy throughout — at temperature the rejection sampler
accepts stochastically and acceptance drops ~10 points regardless (see
decode-shapes), which would swamp the effect being measured.

Per-position acceptance comes from vLLM's own per-pos counter, label-preserved:
metrics.scrape() sums labels away, which is exactly the detail that matters here.

    ./probe_accept_vs_context.py            # ~6 min, single-stream
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

from loadgen import _filler
from metrics import DEFAULT_METRICS, scrape
from probe_decode_shapes import CODE, PROSE

URL = "http://10.150.0.50:8000/v1/chat/completions"
MODEL = "Qwen3.8-Flash-Next-NVFP4"

GEN = "vllm:generation_tokens_total"
DEC_S = "vllm:request_decode_time_seconds_sum"
DRAFTS = "vllm:spec_decode_num_drafts_total"
DRAFT_TOK = "vllm:spec_decode_num_draft_tokens_total"
ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos"
KEYS = (GEN, DEC_S, DRAFTS, DRAFT_TOK, ACCEPTED)

TASKS = {"prose": PROSE, "code": CODE}


def scrape_per_pos(url: str, timeout: float = 15.0) -> dict[int, float]:
    """{position: accepted_total}. Kept separate from metrics.scrape, which
    collapses labels by sum — here the label IS the measurement."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError):
        return {}
    out: dict[int, float] = {}
    for line in body.splitlines():
        if not line.startswith(PER_POS):
            continue
        name, _, val = line.rpartition(" ")
        m = re.search(r'position="(\d+)"', name)
        if not m:
            continue
        try:
            out[int(m.group(1))] = out.get(int(m.group(1)), 0.0) + float(val)
        except ValueError:
            continue
    return out


def snapshot(metrics_url: str) -> tuple[dict[str, float], dict[int, float]]:
    m = scrape(metrics_url, timeout=30) or {}
    return {k: m.get(k, 0.0) for k in KEYS}, scrape_per_pos(metrics_url)


def busy(metrics_url: str) -> float:
    m = scrape(metrics_url, timeout=10) or {}
    return m.get("vllm:num_requests_running", 0.0) + m.get(
        "vllm:num_requests_waiting", 0.0
    )


def run(label: str, prompt: str, metrics_url: str, **params) -> dict | None:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], **params}
    before, before_pos = snapshot(metrics_url)
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            resp = json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"{label:<34} FAILED: {e}")
        return None
    wall = time.perf_counter() - t0
    after, after_pos = snapshot(metrics_url)

    d = {k: after[k] - before[k] for k in KEYS}
    usage = resp.get("usage", {})
    per_step = (d[ACCEPTED] + d[DRAFTS]) / d[DRAFTS] if d[DRAFTS] else float("nan")
    accept = d[ACCEPTED] / d[DRAFT_TOK] if d[DRAFT_TOK] else float("nan")
    rate = d[GEN] / d[DEC_S] if d[DEC_S] else float("nan")
    # Each position is drafted once per step, so the denominator is the drafts delta.
    pos = {
        p: (after_pos.get(p, 0.0) - before_pos.get(p, 0.0)) / d[DRAFTS]
        for p in sorted(set(before_pos) | set(after_pos))
    } if d[DRAFTS] else {}
    pos_s = " ".join(f"p{p}={v:5.1%}" for p, v in sorted(pos.items()))
    print(
        f"{label:<34} prompt={usage.get('prompt_tokens', 0):>6} "
        f"out={usage.get('completion_tokens', 0):>4}  decode={rate:6.1f} tok/s  "
        f"accept={accept:6.1%}  tok/step={per_step:4.2f}  "
        f"steps/s={rate / per_step if per_step else float('nan'):5.1f}  {pos_s}"
    )
    return {
        "label": label,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens", 0
        ),
        "out_tokens": usage.get("completion_tokens", 0),
        "decode_tok_s": rate,
        "acceptance": accept,
        "tok_per_step": per_step,
        "steps_s": rate / per_step if per_step else None,
        "acceptance_by_pos": {str(p): v for p, v in sorted(pos.items())},
        "wall_s": wall,
    }


def main() -> int:
    global URL, MODEL
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-url", default=DEFAULT_METRICS)
    p.add_argument("--url", default=URL, help="chat/completions endpoint")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", default=None, help="write results JSON here")
    p.add_argument("--force", action="store_true", help="run even if server is busy")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument(
        "--contexts",
        default="0,1000,4000,8000,18000",
        help="approx prefix sizes in tokens; 0 is the bare task, 18000 the suite's prompt size",
    )
    args = p.parse_args()
    URL, MODEL = args.url, args.model

    if not args.force and (n := busy(args.metrics_url)) > 0:
        print(f"server busy ({n:.0f} req in flight) — this probe adds load. Use --force.")
        return 1
    if not scrape_per_pos(args.metrics_url):
        print(f"no {PER_POS} counter on {args.metrics_url} — is MTP enabled?")
        return 1

    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]
    run("warmup (discarded)", PROSE, args.metrics_url, max_tokens=64, temperature=0.0)

    rows = []
    for name, task in TASKS.items():
        for n in contexts:
            # Same seed as loadgen's shared prefix: this text is what the suite's
            # 18k prompts are actually made of, and it stays hot across the sweep.
            prefix = (
                f"Reference material:\n\n{_filler(n, seed=0)}\n\n---\n\n" if n else ""
            )
            r = run(
                f"{name} @ {n or 'bare'} ctx",
                prefix + task,
                args.metrics_url,
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
            if r:
                r["task"], r["ctx_target"] = name, n
                rows.append(r)

    if rows:
        base = {r["ctx_target"]: r for r in rows if r["task"] == "prose"}
        lo, hi = min(rows, key=lambda r: r["acceptance"]), max(
            rows, key=lambda r: r["acceptance"]
        )
        print(
            f"\nacceptance range: {lo['acceptance']:.1%} ({lo['label']}) "
            f"-> {hi['acceptance']:.1%} ({hi['label']})"
        )
        if base:
            steps = [r["steps_s"] for r in rows if r["steps_s"]]
            print(
                f"steps/s across the sweep: {min(steps):.1f}-{max(steps):.1f} "
                "(flat means acceptance is the only decode lever)"
            )
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"url": URL, "model": MODEL, "rows": rows}, f, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
