#!/usr/bin/env python3
"""One decode-only profiler window on the head against 127.0.0.1:8000, stdlib only.
Idle gate, then a streaming request on the prompt W2 already cached; the profiler starts after the
first streamed token (prefill is over) and stops ~150 tokens later, before the request ends.
Copies profiler_out_*.txt aside as w3-decode-*."""
import json, sys, time, urllib.request, shutil, glob, os, threading
sys.path.insert(0, "/home/nvidia/bench-suite")
import workload as W
from metrics import scrape
BASE="http://127.0.0.1:8000"; MODEL="Qwen3.8-Flash-Next-NVFP4"; SEED=20260919; PROF=os.path.expanduser("~/qwen38-cluster-recipe/cache/prof")
def post(path, body=None, timeout=900):
    req=urllib.request.Request(BASE+path,data=json.dumps(body).encode() if body is not None else b"",headers={"Content-Type":"application/json"},method="POST")
    t0=time.time(); r=urllib.request.urlopen(req,timeout=timeout); raw=r.read(); return time.time()-t0, raw[:120]
def mem():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): return round(int(l.split()[1])/1048576,1)
m=scrape(BASE+"/metrics")
if m.get("vllm:num_requests_running",0) or m.get("vllm:num_requests_waiting",0):
    print("NOT IDLE", m.get("vllm:num_requests_running"), m.get("vllm:num_requests_waiting")); sys.exit(2)
warm=W.build_prompt(2048, 2048, 201, SEED, "code")
body={"model":MODEL,"messages":[{"role":"user","content":warm+"\n\nExplain this module in detail."}],"max_tokens":260,"ignore_eos":True,"temperature":0,"stream":True}
first=threading.Event(); done=threading.Event(); n=[0]; t_first=[0.0]
def run():
    req=urllib.request.Request(BASE+"/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"},method="POST")
    r=urllib.request.urlopen(req,timeout=300)
    for line in r:
        if line.startswith(b"data:") and b"[DONE]" not in line:
            n[0]+=1
            if not first.is_set(): t_first[0]=time.time(); first.set()
    done.set()
print(f"== w3-decode: MemAvailable {mem()} GiB", flush=True)
th=threading.Thread(target=run); t0=time.time(); th.start()
first.wait(120); print(f"  first chunk after {t_first[0]-t0:.2f}s", flush=True)
m0=scrape(BASE+"/metrics"); t,r=post("/start_profile"); print(f"  start_profile {t:.2f}s {r}", flush=True); ts=time.time()
while n[0] < 200 and not done.is_set(): time.sleep(0.05)
t,r=post("/stop_profile"); te=time.time(); m1=scrape(BASE+"/metrics"); print(f"  stop_profile {t:.2f}s {r}  window {te-ts:.2f}s  chunks seen {n[0]}  MemAvailable {mem()} GiB", flush=True)
th.join(); print(f"  request done {time.time()-t0:.2f}s, {n[0]} chunks", flush=True)
for k in ["vllm:iteration_tokens_total_count","vllm:spec_decode_num_drafts_total","vllm:spec_decode_num_draft_tokens_total","vllm:spec_decode_num_accepted_tokens_total","vllm:generation_tokens_total","vllm:prompt_tokens_total","vllm:prompt_tokens_cached_total"]:
    print(f"  {k}: {m1.get(k,0)-m0.get(k,0):.0f}", flush=True)
time.sleep(5)
for f in glob.glob(PROF+"/profiler_out_*.txt"): shutil.copy(f, f"{PROF}/w3-decode-{os.path.basename(f)}")
print("  files:", sorted(os.path.basename(f) for f in glob.glob(PROF+"/*")), flush=True); print("W3-DONE")
