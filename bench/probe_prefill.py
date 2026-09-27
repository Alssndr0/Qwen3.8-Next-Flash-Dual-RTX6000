#!/usr/bin/env python3
"""probe_prefill.py -- where prefill time goes, read-only, single stream, RUNS ON THE HEAD.

Four fully-cold prompts (2k/8k/32k/65k, max_tokens 1) give prefill tok/s vs length; a warm
repeat gives the API floor; a 700-token ignore_eos decode is the comparison phase; a second
cold 65k gives the sampler two prefill windows. Start `gpusample.sh <out> <pids>` on both boxes
first (0.5 s: gpu util, sm MHz, W, engine CPU ticks, RDMA bytes); phases.json carries the epoch
windows to cut the samples by. First run and its reading: bench/README.md, 2026-09-19.
"""
import json, sys, time, urllib.request
sys.path.insert(0, "/home/nvidia/bench-suite")
import workload as W
from metrics import scrape
URL="http://127.0.0.1:8000/v1/chat/completions"; MET="http://127.0.0.1:8000/metrics"; MODEL="Qwen3.8-Flash-Next-NVFP4"
SEED=20260919
def post(body, timeout=600):
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t0=time.time(); r=json.load(urllib.request.urlopen(req,timeout=timeout)); return time.time()-t0, r
def d(a,b,k): return b.get(k,0)-a.get(k,0)
phases=[]
def run(name, body):
    m0=scrape(MET); t0=time.time(); lat,r=post(body); t1=time.time(); m1=scrape(MET)
    u=r.get("usage",{})
    steps=d(m0,m1,"vllm:iteration_tokens_total_count")
    pf=d(m0,m1,"vllm:request_prefill_time_seconds_sum"); q=d(m0,m1,"vllm:request_queue_time_seconds_sum")
    pt=d(m0,m1,"vllm:prompt_tokens_total"); pc=d(m0,m1,"vllm:prompt_tokens_cached_total"); gt=d(m0,m1,"vllm:generation_tokens_total")
    dec=d(m0,m1,"vllm:request_decode_time_seconds_sum")
    rec=dict(name=name,t0=t0,t1=t1,latency=round(lat,3),prompt=pt,cached=pc,gen=gt,steps=steps,prefill_s=round(pf,3),queue_s=round(q,3),decode_s=round(dec,3),
             prefill_tps=round((pt-pc)/pf,1) if pf else None, tok_per_step=round((pt-pc)/steps,1) if steps else None,
             completion_tokens=u.get("completion_tokens"))
    phases.append(rec); print(json.dumps(rec), flush=True)
    time.sleep(3)
# cold prefills, fully unique content (shared prefix = 0), max_tokens 1
for i,n in enumerate([2048, 8192, 32768, 65536]):
    text=W.build_prompt(n, n, 90+i, SEED, "code")
    run(f"cold-{n}", {"model":MODEL,"messages":[{"role":"user","content":text}],"max_tokens":1,"temperature":0})
# same 65k again: warm (prefix hit) -> the API/tokenize/scheduling floor
text=W.build_prompt(65536, 65536, 93, SEED, "code")
run("warm-65536", {"model":MODEL,"messages":[{"role":"user","content":text}],"max_tokens":1,"temperature":0})
# steady decode baseline: short warm prompt, 700 tokens, ignore_eos
text=W.build_prompt(2048, 2048, 90, SEED, "code")
run("decode-700", {"model":MODEL,"messages":[{"role":"user","content":text+"\n\nExplain this module in detail."}],"max_tokens":700,"ignore_eos":True,"temperature":0})
# one more cold 65k so the sampler has two prefill windows
text=W.build_prompt(65536, 65536, 94, SEED, "code")
run("cold-65536-b", {"model":MODEL,"messages":[{"role":"user","content":text}],"max_tokens":1,"temperature":0})
json.dump(phases, open("/home/nvidia/bench-suite/prof/phases.json","w"), indent=1)
print("PROFILE-DONE")
