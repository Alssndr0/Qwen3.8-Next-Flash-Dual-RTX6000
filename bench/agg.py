#!/usr/bin/env python3
"""Aggregate loadgen runs per arm and permutation-test two arms.

score = 5*ttft_p95 + 1500/tps_per_req at c=5 — the objective from
archive-pre-0731/experiment.py:score(), rounded on 2026-08-14 from the measured 4.4
calls / 1260 tokens. 1500 is still the budget for a WHOLE task (~300 per call over 5
calls), not per call — multiplying a per-call figure by STEPS is the bug that form
was written to fix. Inlined rather than imported: that module pulls in a whole search
space we do not want here.

    ./agg.py <arm_dir> [<arm_dir2>]
"""
import glob
import itertools
import json
import statistics
import sys

import os
STEPS, TASK_OUT_TOKENS = 5, 1500
TARGET_C = int(os.environ.get("TARGET_C", 5))


def arm(d):
    rows = []
    # loadgen-*.json only: a run dir also holds manifest.json, toolprobe.json and
    # decode-shapes.json, none of which carry a "summary".
    for f in sorted(glob.glob(f"{d}/loadgen-*.json")):
        j = json.load(open(f))
        lv = next((x for x in j["summary"] if x["level"] == TARGET_C), None)
        if not lv:
            continue
        # Runs predating the 2026-08-12 loadgen TTFT fix stamped null timings; they
        # cannot be scored. Skip rather than crash — the dashboard shows them as n=0.
        if lv.get("ttft_p95") is None or not lv.get("tps_per_req"):
            continue
        srv = lv.get("server") or {}
        rows.append(
            dict(
                file=f.split("/")[-1],
                score=STEPS * lv["ttft_p95"] + TASK_OUT_TOKENS / lv["tps_per_req"],
                ttft_p95=lv["ttft_p95"],
                tps_req=lv["tps_per_req"],
                tps_agg=lv["tps_aggregate"],
                ok=f"{lv['succeeded']}/{lv['launched']}",
                out=lv["out_tokens_mean"],
                preempt=srv.get("preemptions", -1),
                cache=srv.get("cache_hit_pct", -1),
                queue=srv.get("queue_mean", -1),
                prefill=srv.get("prefill_mean", -1),
                accept=srv.get("accept_len") or lv.get("accept_len_client") or -1,
                seed=j.get("meta", {}).get("tail_seed"),
                workload=j.get("meta", {}).get("workload", "filler"),
            )
        )
    return rows


def spread(v):
    return (max(v) - min(v)) / statistics.median(v) * 100 if v else 0


def report(name, rows):
    if not rows:
        print(f"{name}: no runs")
        return []
    s = [r["score"] for r in rows]
    print(f"\n=== {name} — {len(rows)} runs @ c={TARGET_C} ===")
    for r in rows:
        print(
            f"  {r['file'][-20:]} score {r['score']:7.1f}  ttft_p95 {r['ttft_p95']:6.2f}"
            f"  tps/req {r['tps_req']:6.2f}  agg {r['tps_agg']:6.1f}"
            f"  ok {r['ok']}  out {r['out']:.0f}  preempt {r['preempt']:.0f}"
            f"  cache {r['cache']:.1f}%  q {r['queue']:.2f}s  pf {r['prefill']:.2f}s"
            f"  accept {r['accept']:.2f}  seed {r['seed']}"
        )
    for k in ("score", "ttft_p95", "tps_req", "tps_agg"):
        v = [r[k] for r in rows]
        print(
            f"  {k:9s} median {statistics.median(v):7.2f}  "
            f"[{min(v):.2f}-{max(v):.2f}]  spread {spread(v):.1f}%"
        )
    return s


def perm(a, b):
    """Exact permutation test on the difference in means."""
    obs = abs(statistics.mean(a) - statistics.mean(b))
    pool, n, hits, tot = a + b, len(a), 0, 0
    for idx in itertools.combinations(range(len(pool)), n):
        left = [pool[i] for i in idx]
        right = [pool[i] for i in range(len(pool)) if i not in idx]
        tot += 1
        if abs(statistics.mean(left) - statistics.mean(right)) >= obs - 1e-9:
            hits += 1
    return hits, tot, hits / tot


if __name__ == "__main__":
    arms = [(d.rstrip("/").split("/")[-1], arm(d)) for d in sys.argv[1:]]
    scored = [(n, report(n, r)) for n, r in arms]
    if len(scored) == 2 and all(s for _, s in scored):
        (na, a), (nb, b) = scored
        hits, tot, p = perm(a, b)
        d = (statistics.median(b) - statistics.median(a)) / statistics.median(a) * 100
        print(f"\n{nb} vs {na}: median delta {d:+.1f}%")
        print(f"exact permutation test on means: p = {p:.3f} ({hits}/{tot} splits)")
