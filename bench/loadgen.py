#!/usr/bin/env python3
"""
loadgen.py — deterministic synthetic load for tuning the vllm serving config.

opencode_bench.py measures what developers actually experience, but 20-50% of its
agents answer without calling a tool, and the ones that work range 5-11 steps. That
variance swamps the ~10% config effects we are trying to tune. This fires identically
shaped requests instead, so a difference between two runs is the config.

The trick that makes it work is `ignore_eos`: every request emits exactly --max-tokens
tokens, so time-per-output-token is comparable across runs.

Prompt content comes from workload.py and is selected with --workload. It is not a
cosmetic choice: MTP acceptance is a property of content, so `filler`, `code` and
`agentic` runs measure different things and must never be compared to each other.

`agentic` (2026-09-18) changes the run model, not just the content: each concurrency
slot is a SESSION of --turns sequential requests, every turn a real tool loop (chat
messages + tools) whose prompt is the previous turn's plus one tool result -- the
production shape (see workload.py). Turns stop naturally (ignore_eos off: a forced
continuation past a tool call is not production text), sampling params are omitted so
the server's generation_config applies exactly as it does for opencode, and the
per-request `seed` is pinned so a rep is reproducible on the same engine. TPOT is still
per token, so tok/s per request stays comparable; wall-clock and aggregate tok/s are not
fixed-shape any more and are reported, not gated.

Examples
--------
# concurrency curve against the current server
./loadgen.py --sweep 1,2,4,8,16,24

# quick smoke test
./loadgen.py --sweep 1 --max-tokens 50 --warmup 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from metrics import DEFAULT_METRICS, scrape, server_delta
from workload import KINDS, build_prompt, build_turn

DEFAULT_URL = "http://10.150.0.50:8000/v1/chat/completions"
DEFAULT_MODEL = "deepseek-v4-flash-dspark"



# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #


class _Counter:
    """Hands out request indices that never repeat within a run."""

    def __init__(self, start: int = 0):
        self.n = start

    def take(self) -> int:
        self.n += 1
        return self.n


@dataclass
class Req:
    index: int
    ok: bool
    ttft_s: float | None  # to first content delta — a real client-side TTFT
    total_s: float
    out_tokens: int
    prompt_tokens: int
    cached_tokens: int
    error: str = ""
    deltas: int = 0  # SSE content chunks; vLLM emits one per engine step here
    # agentic only: which session and turn this request was, whether the model ended
    # on a tool call (production: ~80% of turns), how many chunks were reasoning, the
    # finish reason, and the builder's own prompt-size estimate for the drift check.
    turn: int = 0
    session: int = 0
    tool_call: bool = False
    reasoning_deltas: int = 0
    finish: str = ""
    est_prompt_tokens: int | None = None

    @property
    def accept_len(self) -> float | None:
        """Tokens per engine step, from the wire: out_tokens / SSE chunks.

        On this deployment one SSE chunk carries one step's accepted run of
        speculative tokens (verified 2026-09-07, stepprobe), so this is the MTP
        acceptance length as the client sees it, cross-checkable against the
        server's spec_decode counters in Level.server.
        """
        if not self.ok or self.deltas < 1 or self.out_tokens < 1:
            return None
        return self.out_tokens / self.deltas

    @property
    def tpot_s(self) -> float | None:
        """Seconds per output token after the first."""
        if not self.ok or self.ttft_s is None or self.out_tokens < 2:
            return None
        return (self.total_s - self.ttft_s) / (self.out_tokens - 1)


@dataclass
class Level:
    level: int
    wall_s: float
    reqs: list[Req] = field(default_factory=list)
    server: dict | None = None

    def stats(self) -> dict:
        good = [r for r in self.reqs if r.ok]

        def pct(xs, p):
            xs = sorted(xs)
            if not xs:
                return None
            k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
            return xs[k]

        ttfts = [r.ttft_s for r in good if r.ttft_s is not None]
        tpots = [r.tpot_s for r in good if r.tpot_s is not None]
        accepts = [r.accept_len for r in good if r.accept_len is not None]
        out = sum(r.out_tokens for r in good)
        deltas = sum(r.deltas for r in good)
        finishes: dict[str, int] = {}
        for r in good:
            finishes[r.finish or "?"] = finishes.get(r.finish or "?", 0) + 1
        return {
            "level": self.level,
            "launched": len(self.reqs),
            "succeeded": len(good),
            "wall_s": self.wall_s,
            "ttft_p50": pct(ttfts, 50),
            "ttft_p95": pct(ttfts, 95),
            "total_p50": pct([r.total_s for r in good], 50),
            "total_p95": pct([r.total_s for r in good], 95),
            "tpot_mean": statistics.fmean(tpots) if tpots else None,
            "tps_per_req": (1 / statistics.fmean(tpots)) if tpots else None,
            "out_tokens": out,
            "tps_aggregate": (out / self.wall_s) if self.wall_s else None,
            "out_tokens_mean": (out / len(good)) if good else None,
            "prompt_tokens_mean": statistics.fmean([r.prompt_tokens for r in good])
            if good
            else None,
            "accept_len_client": statistics.fmean(accepts) if accepts else None,
            # agentic shape checks: natural stops make these vary, so they are recorded
            # per level rather than assumed. reasoning_share is by SSE chunk, not token.
            "out_tokens_p50": pct([r.out_tokens for r in good], 50),
            "out_tokens_p95": pct([r.out_tokens for r in good], 95),
            "tool_call_rate": (sum(r.tool_call for r in good) / len(good)) if good else None,
            "reasoning_share": (sum(r.reasoning_deltas for r in good) / deltas) if deltas else None,
            "finish_reasons": finishes,
            "turns": max((r.turn for r in good), default=0) + 1,
            "server": self.server,
        }


# --------------------------------------------------------------------------- #
# one streamed request
# --------------------------------------------------------------------------- #


def request_body(args, prompt, index: int) -> dict:
    """The JSON body for one request. `prompt` is a string (filler/code) or a
    build_turn() dict (agentic: messages + tools)."""
    fields = prompt if isinstance(prompt, dict) else {"messages": [{"role": "user", "content": prompt}]}
    body = {
        "model": args.model,
        "messages": fields["messages"],
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if fields.get("tools"):
        body["tools"] = fields["tools"]
        body["tool_choice"] = fields.get("tool_choice", "auto")
    # Sampling. filler/code pin greedy (temperature 0, seed 0) so a rep is a pure
    # function of the prompt. agentic omits temperature/top_p/top_k: production clients
    # send none, so the server's generation_config applies (the model's own 0.7 /
    # top_k 20 / top_p 0.8 -- the "greedy +8%" of 2026-09-16 is NOT what users get),
    # and pins a per-request seed so the draw is reproducible on the same engine.
    temperature = args.temperature
    if temperature is None and args.workload != "agentic":
        temperature = 0.0
    if temperature is not None:
        body["temperature"] = temperature
    body["seed"] = 0 if args.workload != "agentic" else (args.tail_seed * 1000 + index) % (2**31)
    # vllm extension: never stop early, so every request emits exactly max_tokens.
    # Without this, TPOT is not comparable between runs -- for a fixed-shape prompt.
    # An agentic turn ends on a tool call; forcing 300 more tokens past it is not
    # production text and would put the acceptance we are measuring off-distribution.
    ignore_eos = {"on": True, "off": False}.get(args.ignore_eos, args.workload != "agentic")
    if ignore_eos:
        body["ignore_eos"] = True
    return body


def _post_stream(args, prompt, index: int, turn: int = 0, session: int = 0) -> Req:
    """Blocking; run via asyncio.to_thread. Streams SSE and stamps the first delta."""
    body = request_body(args, prompt, index)
    est = prompt.get("est_tokens") if isinstance(prompt, dict) else None
    req = urllib.request.Request(
        args.url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
    )
    t0 = time.perf_counter()
    ttft = None
    n_deltas = 0
    n_reasoning = 0
    tool_call = False
    finish = ""
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    # Reasoning deltas count too: under DEFAULT_THINKING != off the
                    # model streams these and `content` stays empty, so keying on
                    # content alone leaves ttft None for the whole run. The DSpark
                    # 0.25.2 image emits "reasoning"; other vLLM builds emit
                    # "reasoning_content" — accept either.
                    delta = ch.get("delta") or {}
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    if delta.get("tool_calls"):
                        # The tool parser streams the call as tool_calls deltas; each is
                        # still one engine step's worth of accepted tokens.
                        tool_call = True
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        n_deltas += 1
                    elif (
                        delta.get("content")
                        or delta.get("reasoning")
                        or delta.get("reasoning_content")
                    ):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        n_deltas += 1
                        if not delta.get("content"):
                            n_reasoning += 1
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return Req(index, False, ttft, time.perf_counter() - t0, 0, 0, 0, str(e)[:200],
                   turn=turn, session=session, est_prompt_tokens=est)

    details = usage.get("prompt_tokens_details") or {}
    return Req(
        index=index,
        ok=True,
        ttft_s=ttft,
        total_s=time.perf_counter() - t0,
        # prefer vllm's count; fall back to counting deltas
        out_tokens=usage.get("completion_tokens") or n_deltas,
        prompt_tokens=usage.get("prompt_tokens", 0),
        cached_tokens=details.get("cached_tokens", 0),
        deltas=n_deltas,
        turn=turn,
        session=session,
        tool_call=tool_call,
        reasoning_deltas=n_reasoning,
        finish=finish,
        est_prompt_tokens=est,
    )


def _run_session(args, session: int) -> list[Req]:
    """agentic: --turns sequential requests of one session, each turn's prompt being
    the previous turn's plus one tool loop, as an agent harness would send them."""
    out = []
    for t in range(args.turns):
        body = build_turn(args.prompt_tokens, args.unique_tokens, session,
                          args.tail_seed, args.workload, t)
        r = _post_stream(args, body, session, turn=t, session=session)
        out.append(r)
        if not r.ok:
            break
    return out


async def run_level(args, level: int, counter: "_Counter") -> Level:
    print(f"\n=== concurrency {level} — {level} requests ===", flush=True)
    before = scrape(args.metrics_url) if args.metrics_url else None
    t0 = time.perf_counter()
    # Indices come from a counter that spans the whole run, never restarting per
    # level. Restarting it meant c=4 re-served c=2's prompts from the prefix cache,
    # so hit rate climbed 83.8% -> 91.6% down the sweep and levels were not
    # comparable to one another.
    idx = [counter.take() for _ in range(level)]
    if args.workload == "agentic":
        # `level` sessions run side by side, each one --turns requests in sequence.
        # Turn 1+ of every session starts when its own turn 0 ends, so after the first
        # burst the prefills stagger the way production's do (queue ~0 there).
        groups = await asyncio.gather(*(asyncio.to_thread(_run_session, args, i) for i in idx))
        reqs = [r for g in groups for r in g]
    else:
        reqs = await asyncio.gather(
            *(
                asyncio.to_thread(
                    _post_stream,
                    args,
                    build_prompt(
                        args.prompt_tokens, args.unique_tokens, i,
                        args.tail_seed, args.workload,
                    ),
                    i,
                )
                for i in idx
            )
        )
    wall = time.perf_counter() - t0
    after = scrape(args.metrics_url) if args.metrics_url else None

    for r in sorted(reqs, key=lambda x: (x.index, x.turn)):
        tag = f"#{r.index:>2}" if args.workload != "agentic" else f"s{r.session:>2}.t{r.turn}"
        if not r.ok:
            print(f"  [{tag}] FAILED  {r.error}", flush=True)
            continue
        ttft = f"{r.ttft_s:.2f}s" if r.ttft_s is not None else "—"
        tpot = f"{1 / r.tpot_s:5.1f}" if r.tpot_s else "    —"
        extra = ""
        if args.workload == "agentic":
            extra = (f"  {'tool_call' if r.tool_call else r.finish or 'stop':<9} "
                     f"steps={r.deltas:>4} think={r.reasoning_deltas:>4}")
        print(
            f"  [{tag}] ttft={ttft:<7} total={r.total_s:6.1f}s  "
            f"out={r.out_tokens:>4} prompt={r.prompt_tokens:>6} "
            f"(cached {r.cached_tokens:>6})  {tpot} tok/s{extra}",
            flush=True,
        )
    return Level(level, wall, list(reqs), server_delta(before, after, wall))


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--sweep", default="1,2,4,8,16,24", help="concurrency levels")
    p.add_argument(
        "--prompt-tokens",
        type=int,
        default=18000,
        help="approx prompt size; 18000 is the measured production mean",
    )
    p.add_argument(
        "--unique-tokens",
        type=int,
        default=2700,
        help="per-request tail, keeps prefix-cache hit near the measured 80-90%%",
    )
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument(
        "--workload",
        choices=KINDS,
        default="code",
        help="prompt content. 'code' (default since 2026-09-13) is a synthetic Python "
        "repo plus an implementation task; 'filler' is the original word-salad prompt "
        "every run before then used; 'agentic' (2026-09-18) is a multi-turn tool-loop "
        "session at production shape (~65k prompt, tools, reasoning, natural stops). "
        "Content sets MTP acceptance, so kinds are NOT comparable -- compare.py refuses "
        "to pair them.",
    )
    p.add_argument(
        "--turns",
        type=int,
        default=4,
        help="agentic only: sequential turns per session at each concurrency level",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="sampling temperature. Default: 0 for filler/code (greedy, reproducible); "
        "OMITTED for agentic so the server's generation_config applies, as it does for "
        "every production client",
    )
    p.add_argument(
        "--ignore-eos",
        choices=("auto", "on", "off"),
        default="auto",
        help="auto = on for filler/code (fixed output shape), off for agentic (turns end "
        "on their tool call, as in production)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="discarded requests first; a cold engine JIT-compiles triton kernels "
        "during inference and those spikes would land in the measurement",
    )
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--cooldown", type=float, default=20.0)
    p.add_argument("--metrics-url", default=DEFAULT_METRICS)
    p.add_argument("--outdir", default="bench_results")
    p.add_argument(
        "--tail-seed",
        type=int,
        default=None,
        help="seed for the unique tail. Defaults to the wall clock, which keeps tails "
        "out of the prefix cache but changes the CONTENT every run -- and content moves "
        "MTP acceptance by up to 17%% (TUNING.md #12). bench.py pins it per rep so arms "
        "are compared on identical prompts.",
    )
    args = p.parse_args(argv)
    args.levels = [int(x) for x in args.sweep.split(",") if x.strip()]
    if args.tail_seed is None:
        args.tail_seed = int(time.time())
    return args


async def main_async(args) -> int:
    # asyncio.to_thread uses the default executor, whose size is min(32, cpu+4): 18 on a
    # 14-core laptop, 32 on the Vast box. Any level above that was silently serialised
    # (server "Running" capped at exactly the pool size; found 2026-09-20). Size it to c.
    from concurrent.futures import ThreadPoolExecutor
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=max(args.levels) + args.warmup + 8))
    if args.warmup:
        print(f"warmup: {args.warmup} request(s), discarded…", flush=True)
        if args.workload == "agentic":
            # One session per warm-up slot: primes the shared prefix (a cold ~60k
            # prefill, ~30 s at 2k tok/s) and lets the JIT see the tool-call shapes.
            await asyncio.gather(
                *(asyncio.to_thread(_run_session, args, -1 - i) for i in range(args.warmup))
            )
        else:
            await asyncio.gather(
                *(
                    asyncio.to_thread(
                        _post_stream,
                        args,
                        build_prompt(
                            args.prompt_tokens, args.unique_tokens, -1 - i,
                            args.tail_seed, args.workload,
                        ),
                        -1 - i,
                    )
                    for i in range(args.warmup)
                )
            )

    levels: list[Level] = []
    counter = _Counter()
    for i, lv in enumerate(args.levels):
        levels.append(await run_level(args, lv, counter))
        s = levels[-1].stats()
        print(
            f"--- c={lv}: {s['succeeded']}/{s['launched']} ok, "
            f"ttft p50 {s['ttft_p50'] or 0:.2f}s p95 {s['ttft_p95'] or 0:.2f}s, "
            f"{s['tps_per_req'] or 0:.2f} tok/s per req, "
            f"{s['tps_aggregate'] or 0:.1f} tok/s aggregate, wall {s['wall_s']:.1f}s, "
            f"accept {s['accept_len_client'] or 0:.2f} tok/step",
            flush=True,
        )
        served = s.get("prompt_tokens_mean") or 0
        good = [r for r in levels[-1].reqs if r.ok]
        if args.workload == "agentic":
            # Turns grow the prompt, so compare against the builder's own estimate for
            # the turns that actually ran, not --prompt-tokens.
            ests = [r.est_prompt_tokens for r in good if r.est_prompt_tokens]
            expected = statistics.fmean(ests) if ests else args.prompt_tokens
            hint = ("workload.CHARS_PER_TOKEN_TOOL is off: re-measure with "
                    "`./workload.py --calibrate-agentic <url> <model>`")
        else:
            expected = args.prompt_tokens
            hint = ("workload.CHARS_PER_TOKEN is stale: re-measure with "
                    "`./workload.py --calibrate <url> <model>`")
        if s["succeeded"] and served and abs(served / expected - 1) > 0.05:
            print(
                f"    WARNING: served prompt_tokens {served:.0f} is "
                f"{100 * (served / expected - 1):+.1f}% off the expected {expected:.0f}. "
                f"For --workload {args.workload} that means {hint}.",
                flush=True,
            )
        if args.workload == "agentic" and s["succeeded"]:
            print(
                f"    shape: out p50 {s['out_tokens_p50']} p95 {s['out_tokens_p95']}, "
                f"tool_call {100 * (s['tool_call_rate'] or 0):.0f}%, "
                f"reasoning {100 * (s['reasoning_share'] or 0):.0f}% of steps, "
                f"finish {s['finish_reasons']}  (production: gen mean 535, ~80% tool "
                f"calls, ~half reasoning)",
                flush=True,
            )
            sv = s["server"]
            if sv and sv.get("accept_per_pos"):
                pos = " / ".join(f"{100 * x:.0f}%" for x in sv["accept_per_pos"])
                print(f"    accept per position: {pos}  (production 71 / 53 / 42%), "
                      f"uncached/req {sv.get('uncached_tokens_mean') or 0:.0f}, "
                      f"prefill {sv.get('prefill_tps') or 0:.0f} tok/s", flush=True)
        ignore_eos = {"on": True, "off": False}.get(args.ignore_eos, args.workload != "agentic")
        if s["succeeded"] and ignore_eos and s["out_tokens_mean"] != args.max_tokens:
            print(
                f"    WARNING: mean output {s['out_tokens_mean']:.1f} != "
                f"--max-tokens {args.max_tokens}. ignore_eos is not being honoured, "
                f"so TPOT is NOT comparable across runs.",
                flush=True,
            )
        sv = s["server"]
        if sv:
            g = lambda k, u="s": "—" if sv[k] is None else f"{sv[k]:.2f}{u}"
            print(
                f"    server[{sv['requests']:.0f} req]: TTFT {g('ttft_mean')} "
                f"(queue {g('queue_mean')} + prefill {g('prefill_mean')}), "
                f"decode {g('decode_mean')}, {g('decode_tps_per_req', '')} tok/s/req, "
                f"{g('gen_tps_server', '')} tok/s total, cache hit {g('cache_hit_pct', '%')}, "
                f"preempt {sv['preemptions']:.0f}, accept {g('accept_len', '')}",
                flush=True,
            )
        if i < len(args.levels) - 1 and args.cooldown:
            await asyncio.sleep(args.cooldown)

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"loadgen-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "url": args.url,
                    "model": args.model,
                    "prompt_tokens": args.prompt_tokens,
                    "unique_tokens": args.unique_tokens,
                    "max_tokens": args.max_tokens,
                    "workload": args.workload,
                    "turns": args.turns if args.workload == "agentic" else None,
                    "temperature": args.temperature,
                    "ignore_eos": args.ignore_eos,
                    "warmup": args.warmup,
                    "tail_seed": args.tail_seed,
                    "levels": args.levels,
                    "started": datetime.now().isoformat(timespec="seconds"),
                },
                "summary": [lv.stats() for lv in levels],
                "requests": [asdict(r) for lv in levels for r in lv.reqs],
            },
            indent=2,
        )
    )
    print(f"\nraw: {path}")
    return 0


def main() -> int:
    try:
        return asyncio.run(main_async(parse_args()))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
