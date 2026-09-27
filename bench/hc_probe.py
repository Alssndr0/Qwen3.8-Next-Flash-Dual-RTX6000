"""Decode-shape microbench inside the serving image: bf16 F.linear (what runs now for the hyper-connection
matrices) vs the same shape through vLLM's ModelOpt NVFP4 linear method (Marlin W4A16 on the GB10).
Random weights; only kernel time matters. Shapes: hc down [320 x 10240], hc up [10240 x 320], plus the
attention q_proj [12288/2 x 2560] and shared-expert gate [640/2 x 2560] shards for reference."""
import torch, time, json, inspect, sys, os
import torch.nn.functional as F
torch.cuda.set_device(0); dev="cuda"
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.linear import ReplicatedLinear
cfg=None
for mk in [lambda: ModelOptNvFp4Config.from_config({"quantization":{"quant_algo":"NVFP4","kv_cache_quant_algo":None,"group_size":16,"exclude_modules":[]}}),
           lambda: ModelOptNvFp4Config.from_config({"quant_algo":"NVFP4","kv_cache_quant_algo":None,"group_size":16,"exclude_modules":[],"quant_method":"modelopt"}),
           lambda: ModelOptNvFp4Config(True, None, [], 16)]:
    try: cfg=mk(); break
    except Exception as e: print("cfg attempt failed:", repr(e)[:160])
print("quant config:", type(cfg).__name__, flush=True)
def timeit(fn, iters=300):
    for _ in range(20): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/iters*1000  # us
def make_q(K,N):
    layer=ReplicatedLinear(K,N,bias=False,quant_config=cfg,prefix="probe").to(dev)
    with torch.no_grad():
        for name,p in list(layer.named_parameters()):
            if p.dtype==torch.uint8: p.data=torch.randint(0,255,p.shape,dtype=torch.uint8,device=dev)
            elif p.dtype==torch.float8_e4m3fn: p.data=(torch.rand(p.shape,device=dev)*0.5+0.5).to(torch.float8_e4m3fn)
            elif p.numel()<=4: p.data=torch.ones_like(p)
            else: p.data=torch.randn_like(p)
    layer.quant_method.process_weights_after_loading(layer)
    m=layer.quant_method; print("  method:", type(m).__name__, {k:v for k,v in vars(m).items() if isinstance(v,(bool,str,int))}, flush=True)
    return layer
print(f"GPU {torch.cuda.get_device_name(0)}  mem alloc {torch.cuda.memory_allocated()/2**20:.0f} MiB", flush=True)
for label,K,N in [("hc down  [K=10240 -> N=320]",10240,320),("hc up    [K=320 -> N=10240]",320,10240),("attn q_proj shard [2560 -> 6144]",2560,6144),("shared gate shard [2560 -> 320]",2560,320)]:
    w=torch.randn(N,K,dtype=torch.bfloat16,device=dev); q=make_q(K,N)
    print(f"== {label}: weight bf16 {N*K*2/2**20:.1f} MiB, nvfp4 {N*K*0.5/2**20:.1f} MiB (+scales {N*K/16/2**20:.1f})", flush=True)
    for M in [1,4,8,12]:
        x=torch.randn(M,K,dtype=torch.bfloat16,device=dev)
        tb=timeit(lambda: F.linear(x,w)); tq=timeit(lambda: q(x)[0] if isinstance(q(x),tuple) else q(x))
        print(f"   M={M:2d}  bf16 {tb:7.1f} us ({N*K*2/tb/1e3:6.0f} GB/s)   nvfp4-marlin {tq:7.1f} us ({N*K*0.5/tq/1e3:6.0f} GB/s)   speedup {tb/tq:4.2f}x", flush=True)
    del w,q; torch.cuda.empty_cache()
print(f"peak mem alloc {torch.cuda.max_memory_allocated()/2**20:.0f} MiB"); print("PROBE-DONE")
