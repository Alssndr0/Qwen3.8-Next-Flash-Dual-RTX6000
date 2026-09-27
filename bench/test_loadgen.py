#!/usr/bin/env python3
"""Self-check for loadgen's prompt builder and SSE parsing. Run: python3 test_loadgen.py

No network: the streaming path is exercised by feeding _post_stream a fake urlopen.
"""

import json
import types

import loadgen
from loadgen import Level, Req, build_prompt, parse_args

# --- prompt builder -----------------------------------------------------------
# Both workloads must hold the same invariants: deterministic per (index, tail_seed),
# a tail that moves with either, and a shared prefix that never moves -- that prefix is
# what the server's prefix cache keys on, and it models opencode's ~11.8k system prompt.
for kind in ("filler", "code"):
    a = build_prompt(4000, 800, index=0, tail_seed=7, kind=kind)
    b = build_prompt(4000, 800, index=0, tail_seed=7, kind=kind)
    c = build_prompt(4000, 800, index=1, tail_seed=7, kind=kind)
    assert a == b, f"{kind}: same (index, tail_seed) must give a byte-identical prompt"
    assert a != c, f"{kind}: different index must give a different prompt"

    # a different run seed must produce a different tail, or the prefix cache serves it
    # and prefill is measured as free
    d = build_prompt(4000, 800, index=0, tail_seed=8, kind=kind)
    assert d != a, f"{kind}: different tail_seed must change the tail"

    # the divergence must come only at the tail, so the shared head stays cacheable
    common = 0
    for x, y in zip(a, c):
        if x != y:
            break
        common += 1
    assert common > len(a) * 0.6, f"{kind}: shared prefix too short: {common}/{len(a)}"
    seed_common = len(__import__("os").path.commonprefix([a, d]))
    assert seed_common > len(a) * 0.6, f"{kind}: tail_seed moved the shared prefix"

    # degenerate splits must not explode
    assert len(build_prompt(400, 0, 0, kind=kind)) > 0
    assert len(build_prompt(0, 0, 0, kind=kind)) > 0
    assert len(build_prompt(50, 500, 0, kind=kind)) > 0  # unique larger than total

# filler's own markers, unchanged since 2026-08-12 -- stored runs must stay reproducible
a = build_prompt(1000, 200, index=0, tail_seed=7, kind="filler")
c = build_prompt(1000, 200, index=1, tail_seed=7, kind="filler")
assert "[request 7:0]" in a and "[request 7:1]" in c
assert a.endswith("Summarize the above.")
assert build_prompt(100, 0, 0, kind="filler").count("[request") == 0
assert build_prompt(0, 0, 0, kind="filler") == "Summarize the above."

# code must end on the task, and the task must be a code task -- the whole point is that
# the answer runs past --max-tokens so ignore_eos never forces off-distribution text
code = build_prompt(4000, 800, index=0, tail_seed=7, kind="code")
assert code.rstrip().endswith("Return the complete updated file.")
assert "retry_with_backoff" in code and "@dataclass(slots=True)" in code

# an unknown kind must fail loudly rather than silently falling back
try:
    build_prompt(1000, 200, 0, 7, kind="nonsense")
except SystemExit:
    pass
else:  # pragma: no cover
    raise AssertionError("unknown workload must raise")

# --workload must reach args, and default to code
assert parse_args(["--sweep", "1"]).workload == "code"
assert parse_args(["--sweep", "1", "--workload", "filler"]).workload == "filler"

# --- request indices must never repeat within a run ---------------------------
# A repeated index means a repeated prompt, which the prefix cache then serves for
# free, so prefill looks cheaper at later levels and the levels stop being
# comparable. This is what made cache hit climb 83.8% -> 91.6% down a sweep.
from loadgen import _Counter  # noqa: E402

counter = _Counter()
seen, prompts = set(), set()
for lvl in (1, 2, 4, 8):
    for _ in range(lvl):
        i = counter.take()
        assert i not in seen, f"index {i} reused at level {lvl}"
        seen.add(i)
        prompts.add(build_prompt(1000, 200, i, tail_seed=5))
assert len(seen) == 1 + 2 + 4 + 8
assert len(prompts) == len(seen), "every request must get a distinct prompt"
# warmup uses negative indices, so it must not collide with the counter's
assert -1 not in seen and -2 not in seen

# --- SSE parsing --------------------------------------------------------------
STREAM = [
    b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',  # no content: not TTFT
    b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n',
    b"\n",  # keepalive blank line
    b'data: {"choices":[{"delta":{"content":"lo"}}]}\n',
    b"data: {malformed\n",  # must be skipped, not crash
    b'data: {"choices":[],"usage":{"prompt_tokens":17000,"completion_tokens":300,'
    b'"prompt_tokens_details":{"cached_tokens":14000}}}\n',
    b"data: [DONE]\n",
    b'data: {"choices":[{"delta":{"content":"AFTER-DONE"}}]}\n',  # must not be read
]


class FakeResp:
    def __iter__(self):
        return iter(STREAM)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


sent = {}


def fake_urlopen(req, timeout=None):
    sent["body"] = json.loads(req.data)
    return FakeResp()


loadgen.urllib.request.urlopen = fake_urlopen
r = loadgen._post_stream(parse_args(["--max-tokens", "300"]), "prompt text", 7)

assert r.ok and r.index == 7, r
assert r.ttft_s is not None and r.ttft_s >= 0, "TTFT must be stamped on first content"
assert r.out_tokens == 300, r.out_tokens  # from usage, not the 2 deltas seen
assert r.prompt_tokens == 17000 and r.cached_tokens == 14000, r
assert r.total_s >= r.ttft_s
# one SSE content chunk per engine step: 300 tokens over 2 chunks is the acceptance
# length as the client sees it (TUNING.md #12 -- content noise must be visible)
assert r.deltas == 2 and r.accept_len == 150.0, (r.deltas, r.accept_len)

# the request body must carry the flags that make measurements comparable
assert sent["body"]["ignore_eos"] is True, "ignore_eos missing -> TPOT incomparable"
assert sent["body"]["stream"] is True
assert sent["body"]["stream_options"]["include_usage"] is True
assert sent["body"]["temperature"] == 0
assert sent["body"]["max_tokens"] == 300

# --- reasoning-only stream ----------------------------------------------------
# DEFAULT_THINKING != off streams reasoning_content and leaves content empty. With
# ignore_eos the whole response can be reasoning, so keying TTFT on content alone
# left ttft None for every request against the DSpark recipe's default profile.
class ReasoningResp(FakeResp):
    def __iter__(self):
        yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
        yield b'data: {"choices":[{"delta":{"content":""}}]}\n'  # empty: not TTFT
        yield b'data: {"choices":[{"delta":{"reasoning":"think"}}]}\n'
        yield b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":300}}\n'
        yield b"data: [DONE]\n"


loadgen.urllib.request.urlopen = lambda req, timeout=None: ReasoningResp()
rr = loadgen._post_stream(parse_args([]), "p", 2)
assert rr.ttft_s is not None, "TTFT must be stamped on reasoning_content deltas too"

# --- error path ---------------------------------------------------------------
def boom(req, timeout=None):
    raise OSError("connection refused")


loadgen.urllib.request.urlopen = boom
bad = loadgen._post_stream(parse_args([]), "p", 1)
assert not bad.ok and "refused" in bad.error and bad.out_tokens == 0, bad

# --- stats --------------------------------------------------------------------
# 300 tokens, ttft 1s, total 4s -> 3s over 299 tokens
lv = Level(4, wall_s=10.0, reqs=[Req(i, True, 1.0, 4.0, 300, 17000, 14000) for i in range(4)])
lv.reqs.append(Req(9, False, None, 0.5, 0, 0, 0, "err"))
s = lv.stats()
assert s["succeeded"] == 4 and s["launched"] == 5
assert s["out_tokens"] == 1200 and s["out_tokens_mean"] == 300
assert abs(s["tpot_mean"] - 3.0 / 299) < 1e-9, s["tpot_mean"]
assert abs(s["tps_per_req"] - 299 / 3.0) < 1e-9
assert abs(s["tps_aggregate"] - 120.0) < 1e-9  # 1200 tokens / 10s
assert s["ttft_p50"] == 1.0
assert s["accept_len_client"] is None  # no deltas recorded on these synthetic reqs
lv2 = Level(1, 1.0, [Req(0, True, 1.0, 4.0, 300, 10, 0, deltas=120)])
assert abs(lv2.stats()["accept_len_client"] - 2.5) < 1e-9

# --- server acceptance from the spec-decode counters ------------------------
from metrics import server_delta  # noqa: E402

before = {"vllm:spec_decode_num_drafts_total": 1000.0,
          "vllm:spec_decode_num_accepted_tokens_total": 1200.0}
after = {"vllm:spec_decode_num_drafts_total": 1100.0,
         "vllm:spec_decode_num_accepted_tokens_total": 1350.0}
sd = server_delta(before, after, 1.0)
assert sd["spec_drafts"] == 100.0 and sd["spec_accepted"] == 150.0
assert abs(sd["accept_len"] - 2.5) < 1e-9, sd["accept_len"]  # 150/100 + 1
assert server_delta(before, before, 1.0)["accept_len"] is None

# a single-token response has no TPOT to report, and must not divide by zero
assert Req(0, True, 1.0, 2.0, 1, 10, 0).tpot_s is None
assert Level(1, 1.0, [Req(0, True, 1.0, 2.0, 1, 10, 0)]).stats()["tpot_mean"] is None

print("ok")

# --- agentic: production-shaped tool-loop sessions (2026-09-18) ---------------
from workload import build_turn, contamination_hit_pct  # noqa: E402

t0 = build_turn(65000, 2300, 1, tail_seed=7, kind="agentic", turn=0)
t1 = build_turn(65000, 2300, 1, tail_seed=7, kind="agentic", turn=1)
assert t0 == build_turn(65000, 2300, 1, tail_seed=7, kind="agentic", turn=0), "must be deterministic"
# turn t+1 = turn t + one loop, verbatim: that is what makes the previous turn a prefix hit
assert t1["messages"][: len(t0["messages"])] == t0["messages"], "turn t must prefix turn t+1"
assert len(t1["messages"]) == len(t0["messages"]) + 2, "one (call, result) pair per turn"
assert t0["messages"][0]["role"] == "system" and t0["messages"][1]["role"] == "user"
assert t0["messages"][-1]["role"] == "tool", "the model must answer a tool result"
assert t0["messages"][-2].get("tool_calls"), "…that a canned assistant call asked for"
assert t0["messages"][-2]["tool_calls"][0]["id"] == t0["messages"][-1]["tool_call_id"]
assert 5 <= len(t0["tools"]) <= 10 and all(t["type"] == "function" for t in t0["tools"])
assert t0["tool_choice"] == "auto"
# shared block identical across seeds and sessions; the session part moves with both
n_shared = 2 + 2 * sum(1 for m in t0["messages"] if m["role"] == "tool"
                       and m["tool_call_id"].startswith("call_shared"))
other_seed = build_turn(65000, 2300, 1, tail_seed=8, kind="agentic", turn=0)
other_sess = build_turn(65000, 2300, 2, tail_seed=7, kind="agentic", turn=0)
assert other_seed["messages"][:n_shared] == t0["messages"][:n_shared], "tail_seed moved the shared block"
assert other_sess["messages"][:n_shared] == t0["messages"][:n_shared], "session moved the shared block"
assert other_seed["messages"][n_shared:] != t0["messages"][n_shared:]
assert other_sess["messages"][n_shared:] != t0["messages"][n_shared:]
assert n_shared > 0.9 * len(t0["messages"]), "the shared block must dominate the prompt"
# size: turn 0 lands near --prompt-tokens (builder's estimate; served count is the truth)
assert abs(t0["est_tokens"] / 65000 - 1) < 0.08, t0["est_tokens"]
# degenerate: no session loops, tiny budgets
assert build_turn(2000, 0, 1, 7, "agentic", 0)["messages"][-1]["role"] in ("tool", "user")
assert build_turn(0, 0, 1, 7, "agentic", 0)["messages"][1]["role"] == "user"
# filler/code go through build_turn as one user message, unchanged
assert build_turn(1000, 200, 0, 7, "filler")["messages"] == [
    {"role": "user", "content": build_prompt(1000, 200, 0, 7, "filler")}]
assert contamination_hit_pct("code", 18000, 2700) == 92.0
assert 96 < contamination_hit_pct("agentic", 65000, 2300) < 99

# request body policy: agentic omits sampling params (server generation_config applies,
# as for every production client), never forces past EOS, pins a per-request seed,
# carries tools; code/filler keep greedy + ignore_eos exactly as before
from loadgen import request_body  # noqa: E402

ag = parse_args(["--workload", "agentic", "--tail-seed", "5", "--max-tokens", "4096"])
body = request_body(ag, t0, 3)
assert "temperature" not in body and "top_p" not in body and "top_k" not in body, body.keys()
assert "ignore_eos" not in body
assert body["tools"] is t0["tools"] and body["tool_choice"] == "auto"
assert body["seed"] == 5 * 1000 + 3 and body["max_tokens"] == 4096
assert body["messages"] is t0["messages"]
assert request_body(ag, t0, 4)["seed"] != body["seed"]
ag_t = parse_args(["--workload", "agentic", "--temperature", "0.6", "--ignore-eos", "on"])
bt = request_body(ag_t, t0, 3)
assert bt["temperature"] == 0.6 and bt["ignore_eos"] is True
cd = request_body(parse_args(["--workload", "code"]), "p", 1)
assert cd["temperature"] == 0 and cd["seed"] == 0 and cd["ignore_eos"] is True and "tools" not in cd
assert parse_args([]).turns == 4 and parse_args(["--turns", "2"]).turns == 2


# a streamed agentic turn: reasoning deltas, then a tool call streamed as tool_calls
# deltas, finish_reason tool_calls. TTFT on the first reasoning delta; every delta is
# one engine step; the turn is marked as ending in a tool call.
class ToolCallResp(FakeResp):
    def __iter__(self):
        yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
        yield b'data: {"choices":[{"delta":{"reasoning_content":"let me"}}]}\n'
        yield b'data: {"choices":[{"delta":{"reasoning_content":" check"}}]}\n'
        yield b'data: {"choices":[{"delta":{"content":"I will read the file."}}]}\n'
        yield (b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function",'
               b'"function":{"name":"read_file","arguments":"{\\"path\\""}}]}}]}\n')
        yield (b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
               b'{"arguments":": \\"a.py\\"}"}}]}}]}\n')
        yield b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n'
        yield (b'data: {"choices":[],"usage":{"prompt_tokens":66000,"completion_tokens":420,'
               b'"prompt_tokens_details":{"cached_tokens":62000}}}\n')
        yield b"data: [DONE]\n"


sent_bodies = []


def fake_agentic_urlopen(req, timeout=None):
    sent_bodies.append(json.loads(req.data))
    return ToolCallResp()


loadgen.urllib.request.urlopen = fake_agentic_urlopen
r = loadgen._post_stream(ag, t0, 3, turn=1, session=3)
assert r.ok and r.tool_call and r.finish == "tool_calls", r
assert r.deltas == 5 and r.reasoning_deltas == 2, (r.deltas, r.reasoning_deltas)
assert r.out_tokens == 420 and r.prompt_tokens == 66000 and r.cached_tokens == 62000
assert r.turn == 1 and r.session == 3 and r.est_prompt_tokens == t0["est_tokens"]
assert r.ttft_s is not None

# a session runs --turns requests in order, each turn's messages one loop longer
sent_bodies.clear()
ag2 = parse_args(["--workload", "agentic", "--turns", "3", "--tail-seed", "5"])
sess = loadgen._run_session(ag2, 4)
assert [x.turn for x in sess] == [0, 1, 2] and all(x.session == 4 for x in sess)
lens = [len(b["messages"]) for b in sent_bodies]
assert lens[1] == lens[0] + 2 and lens[2] == lens[1] + 2, lens
assert sent_bodies[1]["messages"][: lens[0]] == sent_bodies[0]["messages"], "turn 1 must extend turn 0"
assert all("temperature" not in b and "ignore_eos" not in b and b["tools"] for b in sent_bodies)
assert len({b["seed"] for b in sent_bodies}) == 1, "one seed per session, so a rep is reproducible"

# level stats carry the shape checks
lv = Level(2, 30.0, sess)
s = lv.stats()
assert s["tool_call_rate"] == 1.0 and s["turns"] == 3
assert abs(s["reasoning_share"] - 2 / 5) < 1e-9
assert s["finish_reasons"] == {"tool_calls": 3}
assert s["out_tokens_p50"] == 420 and s["out_tokens_p95"] == 420

# per-position acceptance and prefill throughput from the server counters
from metrics import scrape  # noqa: E402

TEXT = b"""# HELP x
vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 1000.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 1660.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} 710.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="1"} 530.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="2"} 420.0
vllm:request_prefill_kv_computed_tokens_sum{engine="0",model_name="m"} 34000.0
vllm:request_prefill_time_seconds_sum{engine="0",model_name="m"} 17.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="m"} 10.0
"""


class MetricsResp:
    def read(self):
        return TEXT

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


import metrics as _metrics  # noqa: E402

_metrics.urllib.request.urlopen = lambda url, timeout=None: MetricsResp()
after = scrape("http://x/metrics")
assert after["vllm:spec_decode_num_accepted_tokens_per_pos_total"] == 1660.0, "collapsed sum kept"
assert after["vllm:spec_decode_num_accepted_tokens_per_pos_total[pos=2]"] == 420.0
before = {k: 0.0 for k in after}
sd = server_delta(before, after, 10.0)
assert sd["accept_per_pos"] == [0.71, 0.53, 0.42], sd["accept_per_pos"]
assert abs(sd["accept_len"] - 2.66) < 1e-9
assert sd["prefill_tps"] == 2000.0 and sd["uncached_tokens_mean"] == 3400.0
assert server_delta(before, before, 1.0)["accept_per_pos"] is None

print("ok (agentic)")
