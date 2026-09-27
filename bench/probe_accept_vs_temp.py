#!/usr/bin/env python3
"""Speed vs sampling temperature at PRODUCTION shape: reasoning_effort=medium (what every dev runs),
single stream, same two prompts per condition, read off vllm:spec_decode_* deltas. Conditions:
prod = the checkpoint's generation_config (t=1.0, top_p .95, top_k 20) i.e. what opencode gets today."""
import json, sys, time, urllib.request
API = "http://localhost:8000/v1/chat/completions"; MET = "http://localhost:8000/metrics"
MODEL = "Qwen3.8-Flash-Next-NVFP4"; EFFORT = "medium"; MAX_TOKENS = 1500; REPS = 2
def scrape():
    txt = urllib.request.urlopen(MET, timeout=10).read().decode()
    d, pos = {}, {}
    for line in txt.splitlines():
        if line.startswith("#"): continue
        if line.startswith("vllm:spec_decode_num_drafts_total"): d["drafts"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total"): d["dtok"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"): d["acc"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            p = int(line.split('position="')[1].split('"')[0]); pos[p] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:num_requests_running"): d["running"] = float(line.rsplit(" ", 1)[1])
    return d, pos
PROMPTS = {
 "code": "Write a complete Python module implementing an LRU cache with per-entry TTL, thread-safe, with type hints, docstrings, a small CLI demo under __main__, and pytest tests in the same file guarded by an import check. Include edge cases: zero TTL, capacity 1, concurrent eviction.",
 "agentic": "You are refactoring a FastAPI service. The file app/routes/users.py has three endpoints that each open their own SQLAlchemy session and duplicate the same error handling. Explain the plan step by step, then write the refactored module with a shared dependency for the session and a decorator for the error handling, and list the follow-up changes needed in tests.",
}
COND = {"prod": {}, "t0.7": {"temperature": 0.7}, "t0.6": {"temperature": 0.6}, "greedy": {"temperature": 0.0}}
def run(prompt, extra):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS,
            "stream": False, "reasoning_effort": EFFORT}
    body.update(extra)
    req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); r = json.loads(urllib.request.urlopen(req, timeout=900).read()); dt = time.time() - t0
    m = r["choices"][0]["message"]
    return r["usage"]["completion_tokens"], dt, len(m.get("reasoning_content") or m.get("reasoning") or ""), len(m.get("content") or "")
d0, _ = scrape()
if d0.get("running", 0) > 0: sys.exit(f"server busy: running={d0['running']}")
run(PROMPTS["code"], {})  # one warm request, discarded
rows = []
for rep in range(REPS):
    for name, prompt in PROMPTS.items():
        for cond, extra in COND.items():
            a, pa = scrape(); toks, dt, rlen, clen = run(prompt, extra); b, pb = scrape()
            drafts = b["drafts"] - a["drafts"]; dtok = b["dtok"] - a["dtok"]; acc = b["acc"] - a["acc"]
            per_pos = [(pb[i] - pa[i]) / drafts if drafts else float("nan") for i in sorted(pa)]
            row = {"rep": rep, "prompt": name, "cond": cond, "tokens": toks, "secs": round(dt, 1),
                   "tok_s": round(toks / dt, 1), "reasoning_chars": rlen, "content_chars": clen,
                   "accept_rate": round(acc / dtok, 4) if dtok else None,
                   "accept_len": round(acc / drafts + 1, 3) if drafts else None,
                   "per_pos": [round(x, 3) for x in per_pos]}
            rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open("/tmp/accept-temp-probe-medium.json", "w"), indent=1)
print("== summary (mean over reps and prompts), reasoning_effort=%s" % EFFORT)
for cond in COND:
    rs = [r for r in rows if r["cond"] == cond and r["accept_len"]]
    print(f"{cond:8s} accept_len {sum(r['accept_len'] for r in rs)/len(rs):.3f}  accept_rate {sum(r['accept_rate'] for r in rs)/len(rs):.3f}  "
          f"tok/s {sum(r['tok_s'] for r in rs)/len(rs):.1f}  per_pos {[round(sum(r['per_pos'][i] for r in rs)/len(rs),3) for i in range(3)]}")
print("PROBE-DONE")
