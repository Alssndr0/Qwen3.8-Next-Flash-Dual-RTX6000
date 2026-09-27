#!/usr/bin/env python3
"""probe_mixed.py — what a decoding stream feels when a long prefill arrives.

TUNING.md #13. Neither instrument we own can see this: bench/ averages the stall into a
window that contains the prefill, and stepprobe waits for the engine to go idle before
every rep. Yet 60% of production windows contain prefill, and at MAX_NUM_BATCHED_TOKENS=8192
one chunk is one engine step during which every other stream on the box emits nothing.

Shape (after MiaAI-Lab's bench/mixed.py): N streams decoding a short prompt with ignore_eos,
then one long, fully-unique prompt fired into them. Each stream's SSE chunk timestamps are
kept, and one chunk is one engine step here (stepprobe, 2026-09-07), so the inter-chunk gap
is the step time the stream experienced. Gaps are split into three phases:

    before   the decoders alone
    during   from the moment the long request is sent until its first token arrives
    after    the long request has finished prefill and is decoding alongside the others

RUNS ON THE HEAD NODE against 127.0.0.1:8000. Stdlib only.

    python3 probe_mixed.py --ref before-chunk --prompt-tokens 65536
    python3 probe_mixed.py --compare A B
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import threading
import time
import urllib.request

from workload import _filler  # moved out of loadgen.py in 47f15cb
from metrics import scrape

BASE = "http://127.0.0.1:8000"
DECODER_PROMPT = "Write a long, detailed essay on the history of computing. Be verbose."


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def served_model():
    return json.loads(get(BASE + "/v1/models"))["data"][0]["id"]


def running_now():
    m = scrape(BASE + "/metrics") or {}
    return m.get("vllm:num_requests_running", 0.0) + m.get("vllm:num_requests_waiting", 0.0)


def stream(model, prompt, max_tokens, stamps, done, effort="none"):
    """Collect a wall-clock stamp per content chunk into `stamps`; set done[0] at the end."""
    body = {
        "model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0,
        "ignore_eos": True, "reasoning_effort": effort,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                if not raw.startswith(b"data: ") or b"[DONE]" in raw:
                    continue
                try:
                    d = json.loads(raw[6:])
                except Exception:
                    continue
                if d.get("usage"):
                    usage = d["usage"]
                ch = d.get("choices") or []
                delta = (ch[0].get("delta") or {}) if ch else {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    stamps.append(time.perf_counter())
    finally:
        done[0] = usage or {"error": True}


def gap_stats(gaps):
    if len(gaps) < 2:
        return {"n": len(gaps)}
    q = st.quantiles(gaps, n=100) if len(gaps) >= 100 else None
    s = sorted(gaps)
    pct = lambda p: (q[p - 1] if q else s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))])
    return {"n": len(gaps), "p50_ms": pct(50) * 1000, "p95_ms": pct(95) * 1000,
            "p99_ms": pct(99) * 1000, "max_ms": max(gaps) * 1000,
            "steps_per_s": len(gaps) / sum(gaps),
            "gaps_over_1s": sum(1 for g in gaps if g > 1.0),
            "gaps_over_250ms": sum(1 for g in gaps if g > 0.25)}


def phase(stamps, t0, t1):
    """Gaps between consecutive chunks that both fall inside [t0, t1)."""
    ts = [t for t in stamps if t0 <= t < t1]
    return [ts[i] - ts[i - 1] for i in range(1, len(ts))]


def one_run(model, n_dec, prompt_tokens, settle_s, dec_tokens, seed):
    for _ in range(60):
        if running_now() == 0:
            break
        time.sleep(2)
    stamps = [[] for _ in range(n_dec)]
    dones = [[None] for _ in range(n_dec)]
    ths = [threading.Thread(target=stream, args=(model, DECODER_PROMPT + f" (stream {i})",
                                                 dec_tokens, stamps[i], dones[i]), daemon=True)
           for i in range(n_dec)]
    t_start = time.perf_counter()
    for t in ths:
        t.start()
    time.sleep(settle_s)

    # The long prompt: fully unique so nothing is served from the prefix cache. ~1 word
    # per token with loadgen's word list; vLLM reports the real count in usage.
    long_prompt = f"[mixed {seed}] " + _filler(prompt_tokens, seed=seed) + "\n\nSummarize the above in one sentence."
    lstamps, ldone = [], [None]
    t_fire = time.perf_counter()
    lt = threading.Thread(target=stream, args=(model, long_prompt, 32, lstamps, ldone), daemon=True)
    lt.start()
    while not lstamps and lt.is_alive():
        time.sleep(0.05)
    t_first = lstamps[0] if lstamps else time.perf_counter()
    lt.join()
    t_long_end = time.perf_counter()
    for t in ths:
        t.join()
    t_end = time.perf_counter()

    out = {"decoders": n_dec, "prompt_tokens_requested": prompt_tokens,
           "long_prompt_tokens": (ldone[0] or {}).get("prompt_tokens"),
           "long_ttft_s": t_first - t_fire,
           "long_prefill_tok_s": ((ldone[0] or {}).get("prompt_tokens") or 0) / (t_first - t_fire),
           "settle_s": settle_s, "decoder_tokens": [(d[0] or {}).get("completion_tokens") for d in dones],
           "phases": {}}
    for name, (a, b) in {"before": (t_start, t_fire), "during": (t_fire, t_first),
                         "after": (t_first, t_end)}.items():
        gaps = [g for s in stamps for g in phase(s, a, b)]
        out["phases"][name] = gap_stats(gaps)
        out["phases"][name]["window_s"] = b - a
        # tokens the decoders actually delivered inside the window, summed over streams
        out["phases"][name]["chunks_per_stream_per_s"] = (
            sum(len([t for t in s if a <= t < b]) for s in stamps) / n_dec / (b - a)) if b > a else None
    # The stall as one number: the longest single gap any decoder saw while the prefill ran.
    dur = [g for s in stamps for g in phase(s, t_fire, t_long_end)]
    out["worst_gap_during_prefill_ms"] = max(dur) * 1000 if dur else None
    return out


def show(r):
    print(f"  long prompt {r['long_prompt_tokens']} tok, TTFT {r['long_ttft_s']:.2f}s "
          f"({r['long_prefill_tok_s']:.0f} tok/s prefill); worst decoder gap during prefill "
          f"{(r['worst_gap_during_prefill_ms'] or 0):.0f} ms")
    print(f"  {'phase':7s} {'win s':>6} {'steps':>6} {'p50':>7} {'p95':>7} {'p99':>7} {'max':>8} {'>250ms':>6} {'>1s':>4} {'chunk/s':>8}")
    for name, p in r["phases"].items():
        if p.get("n", 0) < 2:
            print(f"  {name:7s} {p.get('window_s', 0):6.1f} {p.get('n', 0):6.0f}   (too few chunks)")
            continue
        print(f"  {name:7s} {p['window_s']:6.1f} {p['n']:6.0f} {p['p50_ms']:7.1f} {p['p95_ms']:7.1f} "
              f"{p['p99_ms']:7.1f} {p['max_ms']:8.1f} {p['gaps_over_250ms']:6.0f} {p['gaps_over_1s']:4.0f} "
              f"{p['chunks_per_stream_per_s']:8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref")
    ap.add_argument("--decoders", type=int, default=2)
    ap.add_argument("--prompt-tokens", default="65536,18000",
                    help="comma list of long-prompt sizes to fire, one run each")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--settle", type=float, default=12.0, help="seconds of clean decode before firing")
    ap.add_argument("--decoder-tokens", type=int, default=3000)
    ap.add_argument("--out", default="/tmp/mixed")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    if a.compare:
        A = json.load(open(os.path.join(a.out, a.compare[0] + ".json")))
        B = json.load(open(os.path.join(a.out, a.compare[1] + ".json")))
        for size in A["runs"]:
            if size not in B["runs"]:
                continue
            for name, X in ((a.compare[0], A), (a.compare[1], B)):
                print(f"\n== {name} @ {size} tok (median of {len(X['runs'][size])} reps) ==")
                show(median_run(X["runs"][size]))
        return
    if not a.ref:
        ap.error("--ref required")
    model = served_model()
    res = {"ref": a.ref, "model": model, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "decoders": a.decoders, "runs": {}}
    m = scrape(BASE + "/metrics") or {}
    print(f"model={model}  decoders={a.decoders}  reps={a.reps}  running now={running_now():.0f}")
    for size in [int(x) for x in a.prompt_tokens.split(",")]:
        res["runs"][str(size)] = []
        for i in range(a.reps):
            print(f"\n== long prompt {size} tok, rep {i + 1}/{a.reps} ==", flush=True)
            r = one_run(model, a.decoders, size, a.settle, a.decoder_tokens, seed=int(time.time()) + i)
            res["runs"][str(size)].append(r)
            show(r)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, a.ref + ".json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"\nstored: {path}")


def median_run(runs):
    """Element-wise median of the reported numbers, for --compare."""
    if len(runs) == 1:
        return runs[0]
    out = json.loads(json.dumps(runs[0]))
    def med(key_path):
        vals = []
        for r in runs:
            v = r
            for k in key_path:
                v = v.get(k) if isinstance(v, dict) else None
            if isinstance(v, (int, float)):
                vals.append(v)
        return st.median(vals) if vals else None
    for k in ("long_ttft_s", "long_prefill_tok_s", "worst_gap_during_prefill_ms"):
        out[k] = med((k,))
    for ph in out["phases"]:
        for k in list(out["phases"][ph]):
            out["phases"][ph][k] = med(("phases", ph, k))
    return out


if __name__ == "__main__":
    main()
