#!/usr/bin/env python3
"""compare.py — A/B two bench runs rep-by-rep, paired on tail_seed.

Since 2026-09-12 bench.py pins one tail seed per counted rep, so rep i of arm A and rep i of
arm B were served the SAME prompts. Pairing them removes the content component of the
spread (acceptance follows content) and leaves the config delta plus boot noise. Unpaired
medians are printed too, for runs that predate the pin.

    ./compare.py <run_dir_or_ref A> <run_dir_or_ref B>
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def find(ref):
    if os.path.isdir(ref):
        return ref
    hits = sorted(glob.glob(f"{HERE}/runs/*__{ref}"))
    if not hits:
        sys.exit(f"no run matching {ref!r}")
    return hits[-1]


def load(d):
    rows, kinds = {}, set()
    for f in sorted(glob.glob(f"{d}/loadgen-*.json")):
        j = json.load(open(f))
        seed = j["meta"].get("tail_seed")
        # Runs before 2026-09-13 predate the field and were all filler.
        kinds.add(j["meta"].get("workload", "filler"))
        for lv in j["summary"]:
            srv = lv.get("server") or {}
            rows[(seed, lv["level"])] = {
                "tps": lv["tps_per_req"], "agg": lv["tps_aggregate"],
                "ttft95": lv["ttft_p95"], "ttft50": lv["ttft_p50"],
                "cache": srv.get("cache_hit_pct"), "accept": srv.get("accept_len"),
                "prefill": srv.get("prefill_mean"), "queue": srv.get("queue_mean"),
                "score": 5 * lv["ttft_p95"] + 1500 / lv["tps_per_req"],
                # agentic extras (None on fixed-shape kinds)
                "prefill_tps": srv.get("prefill_tps"),
                "uncached": srv.get("uncached_tokens_mean"),
                "tool_call": lv.get("tool_call_rate"),
                "out_p50": lv.get("out_tokens_p50"),
                "pos": srv.get("accept_per_pos"),
            }
    return rows, kinds


def pct(a, b):
    return (b / a - 1) * 100 if a else float("nan")


def main():
    da, db = find(sys.argv[1]), find(sys.argv[2])
    (A, ka), (B, kb) = load(da), load(db)
    na, nb = os.path.basename(da).split("__", 1)[1], os.path.basename(db).split("__", 1)[1]
    # Content sets MTP acceptance, and acceptance sets tok/s. Pairing a `code` arm
    # against a `filler` one measures the prompt generator, not the config -- exactly
    # the class of unfalsifiable comparison this suite exists to prevent.
    if ka != kb:
        sys.exit(
            f"refusing to compare: {na} ran workload {sorted(ka)} and {nb} ran "
            f"{sorted(kb)}. Content sets MTP acceptance, so these measure different "
            f"things. Re-run one arm with the other's workload."
        )
    if len(ka) > 1:
        sys.exit(f"refusing to compare: {na} mixes workloads {sorted(ka)} across its reps")
    print(f"workload: {sorted(ka)[0]}")
    levels = sorted({lv for _, lv in A} | {lv for _, lv in B})
    seeds = sorted({s for s, _ in A} & {s for s, _ in B})
    print(f"A = {na}\nB = {nb}\npaired seeds: {seeds or 'none (unpaired only)'}\n")
    agentic = sorted(ka)[0] == "agentic"
    metrics = [("tps", "higher"), ("score", "lower"), ("ttft95", "lower"),
               ("agg", "higher"), ("accept", "flat"), ("cache", "flat")]
    if agentic:
        # Natural stops: output length and tool-call rate are part of what an arm did,
        # and prefill tok/s is the TTFT lever production data points at.
        metrics += [("prefill_tps", "higher"), ("uncached", "flat"),
                    ("tool_call", "flat"), ("out_p50", "flat")]
    for metric, better in metrics:
        print(f"== {metric} ({better} is better) ==")
        print(f"  {'c':>3} {'A med':>8} {'B med':>8} {'unpaired':>9}   paired per-seed deltas")
        for lv in levels:
            va = [A[k][metric] for k in A if k[1] == lv and A[k][metric] is not None]
            vb = [B[k][metric] for k in B if k[1] == lv and B[k][metric] is not None]
            if not va or not vb:
                continue
            paired = [pct(A[(s, lv)][metric], B[(s, lv)][metric]) for s in seeds
                      if (s, lv) in A and (s, lv) in B and A[(s, lv)][metric] and B[(s, lv)][metric]]
            pstr = "  ".join(f"{p:+5.1f}%" for p in paired)
            pmed = f"  -> median {st.median(paired):+5.1f}%" if paired else ""
            print(f"  {lv:>3} {st.median(va):8.2f} {st.median(vb):8.2f} {pct(st.median(va), st.median(vb)):+8.1f}%   {pstr}{pmed}")
        print()
    if agentic:
        # Per draft position, first paired seed per level: the K question is decided
        # here, not on the mean. Arms with different K print different lengths.
        print("== acceptance per draft position (A | B), first paired seed ==")
        for lv in levels:
            for s in seeds:
                pa, pb = A.get((s, lv), {}).get("pos"), B.get((s, lv), {}).get("pos")
                if pa and pb:
                    fa = " / ".join(f"{100 * x:.0f}%" for x in pa)
                    fb = " / ".join(f"{100 * x:.0f}%" for x in pb)
                    print(f"  {lv:>3}  {fa}  |  {fb}")
                    break
        print()


if __name__ == "__main__":
    main()
