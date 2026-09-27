#!/usr/bin/env python3
"""qualityprobe.py — the accuracy gate a precision change has to pass (TUNING.md #14).

We had never run an accuracy eval on this deployment. That was tolerable while every arm
was a scheduling change; it is not tolerable for #10 (bf16 recurrent state), which changes
the numerics of a state carried across the whole sequence. Two probes, both deterministic
so the same items run on every arm and a --compare shows exactly which items flipped:

  gsm8k    the first 200 items of the GSM8K test set (data/gsm8k-200.jsonl, the subset size
           used in sglang#35860). temperature 0, thinking off by default, scored on the
           final number.
  needles  a passcode buried in ~32k and ~96k tokens of prose at 5/50/95% depth, N samples
           per cell with fixed seeds -- MATCHED sample counts on both arms, because Mia
           found the 95%-depth needle at 32k flaky regardless of dtype and a single pass
           cannot tell dtype from luck.

Neither probe reaches significance on its own at these sizes. What they can do is catch a
collapse (a bf16 state that drifts loses whole-number arithmetic and long recall first),
and show per-item flips so a 2-item swing is readable as noise rather than a trend.

RUNS ON THE HEAD NODE against 127.0.0.1:8000. Stdlib only.

    python3 qualityprobe.py --ref baseline            # both probes, ~15 min
    python3 qualityprobe.py --ref x --probes gsm8k
    python3 qualityprobe.py --compare baseline mamba-bf16
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
HERE = os.path.dirname(os.path.abspath(__file__))

SENTENCES = [
    "The harbour master logged the tide at dawn and noted that the wind had shifted to the east.",
    "A small bakery on the corner sells rye loaves that are scored by hand before the oven.",
    "The committee postponed its vote until the survey of the northern field was complete.",
    "Every spring the river rises a metre and the ferry runs from the upper landing instead.",
    "The librarian catalogued the donated maps by decade and then by the region they showed.",
    "Rain delayed the roofing crew for three days, so the scaffolding stayed up over the weekend.",
    "The orchard's oldest tree still produces fruit, though the apples are smaller than they were.",
    "A freight train passes the village twice a night, and the dogs have long stopped barking at it.",
    "The museum's east wing was closed for repainting and reopened with the textile collection.",
    "Students measured the length of the shadow at noon each week to plot the changing season.",
    "The lighthouse keeper's ledger records fog on eleven of the thirty days of that month.",
    "Two cargo barges were moored below the bridge waiting for the lock to be repaired.",
    "The town band rehearses on Thursdays in the hall behind the fire station.",
    "A survey of the marsh found more heron nests than in any year since the count began.",
    "The council agreed to replace the wooden footbridge with a steel one of the same width.",
    "Frost came late that year, and the last of the tomatoes were picked in the second week of November.",
]


def post(body, timeout=1800):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def served_model():
    with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as r:
        return json.loads(r.read().decode())["data"][0]["id"]


def ask(model, prompt, max_tokens, effort):
    t0 = time.perf_counter()
    d = post({"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0, "max_tokens": max_tokens, "reasoning_effort": effort})
    msg = d["choices"][0]["message"]
    return {"text": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "finish": d["choices"][0].get("finish_reason"), "usage": d.get("usage", {}),
            "wall_s": time.perf_counter() - t0}


# ---------------------------------------------------------------- gsm8k
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def gold_answer(ans: str) -> float:
    return float(ans.split("####")[-1].strip().replace(",", ""))


def predicted(text: str):
    m = re.search(r"[Ff]inal [Aa]nswer\s*[:=]?\s*\$?\s*(-?\d[\d,]*\.?\d*)", text)
    if not m:
        nums = NUM.findall(text)
        if not nums:
            return None
        m_val = nums[-1]
    else:
        m_val = m.group(1)
    try:
        return float(m_val.replace(",", "").rstrip("."))
    except ValueError:
        return None


def run_gsm8k(model, effort, conc, max_tokens):
    items = [json.loads(l) for l in open(os.path.join(HERE, "data", "gsm8k-200.jsonl"))]
    prompt = ("{q}\n\nSolve the problem step by step, then give the final numeric answer on "
              "the last line in the form 'Final answer: <number>'.")

    def one(i):
        it = items[i]
        r = ask(model, prompt.format(q=it["question"]), max_tokens, effort)
        g = gold_answer(it["answer"])
        p = predicted(r["text"])
        ok = p is not None and abs(p - g) < 1e-6 * max(1.0, abs(g))
        return {"i": i, "ok": ok, "gold": g, "pred": p, "finish": r["finish"],
                "out_tokens": r["usage"].get("completion_tokens"), "wall_s": r["wall_s"]}

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(conc) as ex:
        rows = sorted(ex.map(one, range(len(items))), key=lambda r: r["i"])
    n_ok = sum(r["ok"] for r in rows)
    trunc = sum(1 for r in rows if r["finish"] == "length")
    print(f"  gsm8k: {n_ok}/{len(rows)} = {100 * n_ok / len(rows):.1f}%   truncated {trunc}   "
          f"mean out {sum(r['out_tokens'] or 0 for r in rows) / len(rows):.0f} tok   "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    return {"n": len(rows), "correct": n_ok, "acc": n_ok / len(rows), "truncated": trunc,
            "effort": effort, "max_tokens": max_tokens, "items": rows}


# ---------------------------------------------------------------- needles
def haystack(n_words, seed):
    rng = random.Random(seed)
    out, n = [], 0
    while n < n_words:
        s = rng.choice(SENTENCES)
        out.append(s)
        n += len(s.split())
    return out


def run_needles(model, cells, conc):
    """cells: list of (size_tokens, depth_pct, sample_idx)."""
    def one(cell):
        size, depth, k = cell
        seed = size * 1000 + depth * 10 + k
        rng = random.Random(seed)
        code = "".join(rng.choice("0123456789") for _ in range(7))
        colour = rng.choice(["amber", "cobalt", "crimson", "ivory", "jade", "onyx", "saffron", "teal"])
        needle = f"Note for the records: the access passcode for the {colour} vault is {code}."
        sents = haystack(int(size * 0.72), seed)  # ~0.72 words/token for this prose
        pos = int(len(sents) * depth / 100)
        sents.insert(pos, needle)
        prompt = ("Read the following notes carefully.\n\n" + " ".join(sents) +
                  f"\n\nQuestion: what is the access passcode for the {colour} vault? "
                  "Reply with the passcode only.")
        r = ask(model, prompt, 64, "none")
        found = code in (r["text"] + r["reasoning"])
        return {"size": size, "depth": depth, "k": k, "ok": found, "code": code,
                "prompt_tokens": r["usage"].get("prompt_tokens"), "wall_s": r["wall_s"],
                "reply": r["text"][:80]}

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(conc) as ex:
        rows = list(ex.map(one, cells))
    by = {}
    for r in rows:
        by.setdefault((r["size"], r["depth"]), []).append(r["ok"])
    for (size, depth), oks in sorted(by.items()):
        toks = [r["prompt_tokens"] for r in rows if r["size"] == size][0]
        print(f"  needles {size:>6} tok (real {toks}) depth {depth:>2}%: {sum(oks)}/{len(oks)}", flush=True)
    print(f"  needles total {sum(r['ok'] for r in rows)}/{len(rows)}   {time.perf_counter() - t0:.0f}s")
    return {"n": len(rows), "found": sum(r["ok"] for r in rows), "items": rows}


# ---------------------------------------------------------------- compare
def compare(out_dir, a, b):
    A = json.load(open(os.path.join(out_dir, a + ".json")))
    B = json.load(open(os.path.join(out_dir, b + ".json")))
    print(f"{'probe':10s} {a:>18s} {b:>18s}")
    if "gsm8k" in A and "gsm8k" in B:
        ga, gb = A["gsm8k"], B["gsm8k"]
        print(f"{'gsm8k':10s} {ga['correct']:>14d}/{ga['n']:<3d} {gb['correct']:>14d}/{gb['n']:<3d}")
        ia = {r["i"]: r["ok"] for r in ga["items"]}
        ib = {r["i"]: r["ok"] for r in gb["items"]}
        lost = sorted(i for i in ia if ia[i] and not ib.get(i, False))
        won = sorted(i for i in ia if not ia[i] and ib.get(i, False))
        print(f"           items right in {a} but wrong in {b}: {lost}")
        print(f"           items wrong in {a} but right in {b}: {won}")
        print(f"           truncated: {ga['truncated']} vs {gb['truncated']}")
    if "needles" in A and "needles" in B:
        na, nb = A["needles"], B["needles"]
        print(f"{'needles':10s} {na['found']:>14d}/{na['n']:<3d} {nb['found']:>14d}/{nb['n']:<3d}")
        ca = {(r["size"], r["depth"], r["k"]): r["ok"] for r in na["items"]}
        cb = {(r["size"], r["depth"], r["k"]): r["ok"] for r in nb["items"]}
        for key in sorted(set(ca) | set(cb)):
            if ca.get(key) != cb.get(key):
                print(f"           {key}: {ca.get(key)} -> {cb.get(key)}")
    if "toolprobe" in A or "toolprobe" in B:
        print(f"{'toolprobe':10s} {str(A.get('toolprobe')):>18s} {str(B.get('toolprobe')):>18s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref")
    ap.add_argument("--probes", default="gsm8k,needles")
    ap.add_argument("--effort", default="none", help="reasoning_effort for gsm8k (none/low/medium/xhigh)")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--conc", type=int, default=6, help="6 = the production concurrency target")
    ap.add_argument("--needle-cells", default="32768:5,98304:3",
                    help="size:samples per depth; depths are 5/50/95%%")
    ap.add_argument("--out", default="/tmp/quality")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    if a.compare:
        compare(a.out, *a.compare)
        return
    if not a.ref:
        ap.error("--ref required")
    model = served_model()
    res = {"ref": a.ref, "model": model, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(f"model={model}  ref={a.ref}  probes={a.probes}")
    if "gsm8k" in a.probes:
        res["gsm8k"] = run_gsm8k(model, a.effort, a.conc, a.max_tokens)
    if "needles" in a.probes:
        cells = []
        for spec in a.needle_cells.split(","):
            size, n = (int(x) for x in spec.split(":"))
            cells += [(size, d, k) for d in (5, 50, 95) for k in range(n)]
        res["needles"] = run_needles(model, cells, min(a.conc, 2))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, a.ref + ".json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"stored: {path}")


if __name__ == "__main__":
    main()
