#!/usr/bin/env python3
"""Two profiler windows on the head against 127.0.0.1:8000, stdlib only.
  W1 prefill: POST /start_profile, one fully cold 8k prompt (max_tokens 1), POST /stop_profile.
  W2 decode : a second start (may be refused by torch; then only W1 exists), warm 2k prompt + 96 tokens, stop.
Between windows the kernel tables (profiler_out_<rank>.txt) are copied aside, since stop overwrites them.
"""
import json, sys, time, urllib.request, shutil, glob, os
sys.path.insert(0, "/home/nvidia/bench-suite")
import workload as W
from metrics import scrape
BASE="http://127.0.0.1:8000"; MODEL="Qwen3.8-Flash-Next-NVFP4"; SEED=20260919; PROF=os.path.expanduser("~/qwen38-cluster-recipe/cache/prof")
def post(path, body=None, timeout=900):
    req=urllib.request.Request(BASE+path,data=json.dumps(body).encode() if body is not None else b"",headers={"Content-Type":"application/json"},method="POST")
    t0=time.time(); r=urllib.request.urlopen(req,timeout=timeout); raw=r.read(); return time.time()-t0, (json.loads(raw) if raw.strip().startswith(b"{") else raw[:200])
def mem(): 
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): return round(int(l.split()[1])/1048576,1)
def window(name, body):
    print(f"== {name}: MemAvailable {mem()} GiB", flush=True)
    t,r=post("/start_profile"); print(f"  start_profile {t:.2f}s {r}", flush=True)
    m0=scrape(BASE+"/metrics"); lat,resp=post("/v1/chat/completions", body); m1=scrape(BASE+"/metrics")
    pt=m1.get("vllm:prompt_tokens_total",0)-m0.get("vllm:prompt_tokens_total",0); pc=m1.get("vllm:prompt_tokens_cached_total",0)-m0.get("vllm:prompt_tokens_cached_total",0)
    pf=m1.get("vllm:request_prefill_time_seconds_sum",0)-m0.get("vllm:request_prefill_time_seconds_sum",0)
    print(f"  request {lat:.2f}s  prompt {pt:.0f} cached {pc:.0f} prefill {pf:.2f}s  completion {resp.get('usage',{}).get('completion_tokens') if isinstance(resp,dict) else resp}", flush=True)
    t,r=post("/stop_profile"); print(f"  stop_profile {t:.2f}s {r}  MemAvailable {mem()} GiB", flush=True)
    time.sleep(5)
    for f in glob.glob(PROF+"/profiler_out_*.txt"):
        shutil.copy(f, f"{PROF}/{name}-{os.path.basename(f)}")
    print("  files:", sorted(os.path.basename(f) for f in glob.glob(PROF+"/*")), flush=True)
text=W.build_prompt(8192, 8192, 200, SEED, "code")
window("w1-prefill", {"model":MODEL,"messages":[{"role":"user","content":text}],"max_tokens":1,"temperature":0})
warm=W.build_prompt(2048, 2048, 201, SEED, "code")
post("/v1/chat/completions", {"model":MODEL,"messages":[{"role":"user","content":warm}],"max_tokens":1,"temperature":0})  # warm its prefix first
window("w2-decode", {"model":MODEL,"messages":[{"role":"user","content":warm+"\n\nExplain this module in detail."}],"max_tokens":96,"ignore_eos":True,"temperature":0})
print("WINDOWS-DONE")
