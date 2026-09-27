#!/usr/bin/env python3
"""bench.py — the suite we run after every config change.

    ./bench.py run --ref <label>     capture provenance, run the suite, store results
    ./bench.py list                  one line per stored run
    ./bench.py dashboard             regenerate dashboard.html
    ./bench.py restore <ref>         print (or --apply) the steps back to a run's config
    ./bench.py import-legacy         pull in pre-suite opencode-bench arms, flagged

Runs execute ON THE HEAD NODE. loadgen has to hit 127.0.0.1:8000 — driving it from a
laptop measures the LAN as much as the engine, and the WSL->box path proved flaky on
2026-08-14 (empty replies while the box itself served 200s).

Stdlib only, no venv: that is what lets the same files run on the head node unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import provenance
from provenance import HEAD, ProvenanceError
from workload import KINDS, contamination_hit_pct

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
REMOTE = "/home/nvidia/bench-suite"

# The documented protocol from opencode-bench/SESSION-2026-08-12.md: identical request
# shapes, one discarded warm-up sweep, then counted reps. 18000 prompt tokens is the
# measured production mean; 2700 unique keeps the prefix-cache hit near the real 80-90%.
HEAD_TZ = "Europe/Amsterdam"  # head node's clock; legacy loadgen stamps are naive local
# c=6 added 2026-09-12: the production target is six concurrent agentic coding sessions,
# and at MTP=3 that is a 24-token verify batch -- a different CUDA graph from c=5's 20.
SWEEP = "1,5,6,10"
PROMPT_TOKENS, UNIQUE_TOKENS, MAX_TOKENS = 18000, 2700, 300
REPS = 3

# Prompt content (workload.py). Until 2026-09-13 this was hardcoded to `filler`: 18,000
# tokens drawn from a 38-word list, task "Summarize the above.", with ignore_eos forcing
# 300 tokens past the natural stop. MTP acceptance is a property of content, and on that
# prompt the drafter accepts 43% of draft positions against 50% on our live opencode
# traffic and 80-97% that bilikaz measure on dense code -- so filler's accept_len ~2.3
# was the prompt generator's number, not the engine's, and any arm that only pays off on
# structured content was invisible to this suite. `code` is a synthetic Python repo plus
# an implementation task whose answer runs well past MAX_TOKENS, so ignore_eos never
# forces off-distribution continuation. Runs of the two kinds are NOT comparable; the
# kind is in the manifest and compare.py refuses to pair across it.
WORKLOAD = "code"

# Per-kind protocol. `code` is the 2026-08-12 shape above. `agentic` (2026-09-18) is the
# production shape Prometheus measured over the 46.6 h after the lpt1024 boot: ~65k
# prompt, ~3.4k uncached per turn, tools + reasoning, natural stops, and c=2-3 -- the
# levels 68% of production request-time is spent at (c=1 is 32%, c>=5 is 10%). c=1 and
# c=6 are available with --sweep. max_tokens is a ceiling, not the output length: turns
# stop on their tool call (production gen mean 535, tail to 2-5k). turns=4 gives 8-12
# counted requests per level per rep at the default sweep.
PROTOCOLS = {
    "filler": dict(sweep=SWEEP, prompt_tokens=PROMPT_TOKENS, unique_tokens=UNIQUE_TOKENS,
                   max_tokens=MAX_TOKENS, turns=None),
    "code": dict(sweep=SWEEP, prompt_tokens=PROMPT_TOKENS, unique_tokens=UNIQUE_TOKENS,
                 max_tokens=MAX_TOKENS, turns=None),
    "agentic": dict(sweep="2,3", prompt_tokens=65000, unique_tokens=2300,
                    max_tokens=4096, turns=4),
}

# The unique tail of every prompt is seeded, and until 2026-09-12 the seed was the wall
# clock (loadgen.py), so every rep -- and every arm -- ran on different content. Content
# moves MTP acceptance, and acceptance moved tok/s by 17% between two reps of an identical
# config on 2026-09-06 (TUNING.md #12). Pinning one seed per rep gives every arm the same
# three prompt sets, so a delta between arms is the config; the rep-to-rep spread inside an
# arm is the content spread and is now the same in every arm. The warm-up sweep gets its
# own seed so it never pre-warms a counted rep's tails in the prefix cache.
TAIL_SEED_BASE = 20260912


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, **kw)


def remote(cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
    p = sh(["ssh", "-o", "BatchMode=yes", HEAD, cmd], capture_output=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"remote failed: {cmd}\n{p.stderr}")
    return p


def bench_sha() -> str:
    p = sh(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True)
    sha = p.stdout.strip() or "(uncommitted)"
    dirty = sh(["git", "-C", str(HERE), "status", "--porcelain", "."], capture_output=True)
    return sha + ("-dirty" if dirty.stdout.strip() else "")


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def cmd_run(args) -> int:
    print("== provenance ==")
    try:
        prov = provenance.capture(bench_sha(), args.recipe)
    except ProvenanceError as e:
        print(f"ABORT: {e}", file=sys.stderr)
        print(
            "\nA result without provenance is worse than no result — it looks like\n"
            "evidence and cannot be checked. Fix the deployment, then re-run.",
            file=sys.stderr,
        )
        return 2
    print(f"  launch  {prov['deployment']}  ({prov['container']})")
    label = "recipe" if prov["deployment"] == "qwen38fn" else "dspark"
    print(f"  {label:6s}  {prov['dspark_sha'][:8]}"
          f"{'  (dirty)' if prov['dspark_dirty'] else ''}")
    print(f"  vllm    {prov['vllm_version']}")
    print(f"  model   {prov['model']}  as  {prov['served_model_name']}")
    print(f"  hotfix  {', '.join(prov['hotfixes'][HEAD]) or '(none)'}")
    print(f"  seqs {prov['max_num_seqs']}  batched {prov['max_num_batched_tokens']}  "
          f"long-prefill {prov['long_prefill_token_threshold']}  retention {prov['retention_interval']}")
    proto = dict(PROTOCOLS[args.workload])
    if args.sweep:
        proto["sweep"] = args.sweep
    prov["workload"] = {
        "sweep": proto["sweep"], "prompt_tokens": proto["prompt_tokens"],
        "unique_tokens": proto["unique_tokens"], "max_tokens": proto["max_tokens"],
        "turns": proto["turns"], "reps": REPS, "kind": args.workload,
        "tail_seed_base": args.tail_seed,
        "tail_seeds": {"warmup": args.tail_seed,
                       "counted": [args.tail_seed + i + 1 for i in range(REPS)]},
    }
    print(f"  seeds   warm-up {args.tail_seed}, counted {prov['workload']['tail_seeds']['counted']}")
    print(f"  content workload={args.workload}  sweep {proto['sweep']}  prompt {proto['prompt_tokens']}"
          f"  unique {proto['unique_tokens']}  max_tokens {proto['max_tokens']}"
          f"{'  turns ' + str(proto['turns']) if proto['turns'] else ''}")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%MZ")
    rundir = RUNS / f"{stamp}__{args.ref}"
    rundir.mkdir(parents=True, exist_ok=True)
    (rundir / prov["config_file"]).write_text(prov.pop("config_text"))
    (rundir / "local.diff").write_text(prov.pop("local_diff"))
    prov["ref"] = args.ref

    # A mounted file is config. The 2026-09-06 draft-vocabulary arm differed from the
    # one before it mainly in the 92 KB file behind /etc/vllm-draft-vocab.txt, and
    # storing only its hash would record that the arms differ without being able to put
    # the file back. Small enough to keep verbatim; the hash in the manifest is what
    # proves this copy is the one that ran.
    vocab = (prov.get("overlay", {}).get(HEAD, {}).get("binds", {})
             .get("/etc/vllm-draft-vocab.txt") or {}).get("source")
    if vocab:
        sh(["scp", "-q", f"{HEAD}:{vocab}", str(rundir / "draft_vocab.txt")], check=False)
        print(f"  vocab   {vocab} -> draft_vocab.txt")

    print(f"\n== sync -> {HEAD}:{REMOTE} ==")
    remote(f"mkdir -p {REMOTE}")
    files = ["metrics.py", "loadgen.py", "workload.py", "probe_decode_shapes.py",
             "toolprobe.py", "agg.py"]
    sh(["scp", "-q", *[str(HERE / f) for f in files], f"{HEAD}:{REMOTE}/"], check=True)

    # An orphaned loadgen from an earlier invocation silently competes for the GPU and
    # produced one junk arm on 08-12 (score 332.9). Cheap to prevent, expensive to spot.
    remote("pkill -f loadgen.py || true", check=False)

    gates: dict[str, dict] = {}
    for name, fn in (
        ("smoke", _smoke),
        ("toolprobe", _toolprobe),
        ("loadgen", _loadgen),
        ("decode_shapes", _decode_shapes),
    ):
        if name in args.skip:
            print(f"\n== {name}: SKIPPED (--skip) ==")
            gates[name] = {"ok": None, "skipped": True}
            continue
        print(f"\n== {name} ==")
        gates[name] = fn(prov, rundir)
        state = gates[name].get("ok")
        print(f"-- {name}: {'PASS' if state else 'RECORDED' if state is None else 'FAIL'}")
        if state is False and not args.keep_going:
            print(f"\n{name} failed its gate. Stopping (use --keep-going to continue).")
            break

    prov["gates"] = {k: v.get("ok") for k, v in gates.items()}
    prov["results"] = gates
    (rundir / "manifest.json").write_text(json.dumps(prov, indent=1))
    print(f"\nstored {rundir.relative_to(HERE.parent)}")

    failed = [k for k, v in prov["gates"].items() if v is False]
    print("VERDICT:", "FAIL — " + ", ".join(failed) if failed else "PASS")
    return 1 if failed else 0


def _smoke(prov, rundir) -> dict:
    """Cheapest possible check that we are talking to the model we think we are."""
    name = prov["served_model_name"]
    p = remote(
        f"curl -s --max-time 30 http://127.0.0.1:8000/v1/models", check=False
    )
    try:
        served = [m["id"] for m in json.loads(p.stdout)["data"]]
    except Exception:  # noqa: BLE001
        return {"ok": False, "detail": f"/v1/models unreadable: {p.stdout[:200]}"}
    if name not in served:
        return {"ok": False, "detail": f"expected {name!r}, server lists {served}"}
    return {"ok": True, "served": served}


def _toolprobe(prov, rundir) -> dict:
    out = f"{REMOTE}/toolprobe.json"
    p = sh(
        ["ssh", "-o", "BatchMode=yes", HEAD,
         f"cd {REMOTE} && PROBE_URL=http://127.0.0.1:8000 "
         f"PROBE_MODEL={prov['served_model_name']} PROBE_OUT={out} "
         f"python3 toolprobe.py {prov['ref']} 10"],
        capture_output=True,
    )
    sys.stdout.write(p.stdout)
    sh(["scp", "-q", f"{HEAD}:{out}", str(rundir / "toolprobe.json")], check=False)
    try:
        return json.loads((rundir / "toolprobe.json").read_text())
    except Exception:  # noqa: BLE001
        return {"ok": False, "detail": "toolprobe produced no JSON"}


def _loadgen(prov, rundir) -> dict:
    w = prov["workload"]
    base = (
        f"python3 -u loadgen.py --url http://127.0.0.1:8000/v1/chat/completions "
        f"--metrics-url http://127.0.0.1:8000/metrics --model {prov['served_model_name']} "
        f"--sweep {w['sweep']} --prompt-tokens {w['prompt_tokens']} "
        f"--unique-tokens {w['unique_tokens']} --max-tokens {w['max_tokens']} "
        f"--workload {w['kind']}"
        + (f" --turns {w['turns']}" if w.get("turns") else "")
    )
    seeds = w["tail_seeds"]
    # Same seeds twice on one boot serve the cold tails from the prefix cache; the honest
    # hit depends on the shape (code ~85%, agentic ~95%), so the line does too.
    contaminated = contamination_hit_pct(w["kind"], w["prompt_tokens"], w["unique_tokens"])
    remote(f"rm -rf {REMOTE}/warmup {REMOTE}/counted", check=False)
    print("  warm-up sweep (discarded — a cold engine JITs triton kernels mid-inference)")
    remote(f"cd {REMOTE} && {base} --tail-seed {seeds['warmup']} --outdir {REMOTE}/warmup >/dev/null")
    for i, seed in enumerate(seeds["counted"]):
        print(f"  counted rep {i + 1}/{REPS}  (tail_seed {seed})")
        remote(f"cd {REMOTE} && {base} --tail-seed {seed} --outdir {REMOTE}/counted >/dev/null")
    sh(["scp", "-q", f"{HEAD}:{REMOTE}/counted/*.json", str(rundir)], check=False)

    files = sorted(rundir.glob("loadgen-*.json"))
    if len(files) < REPS:
        return {"ok": False, "detail": f"expected {REPS} reps, got {len(files)}"}
    preempt, bad, warn = 0.0, [], []
    by_level: dict[int, list[dict]] = {}
    for f in files:
        for lv in json.loads(f.read_text())["summary"]:
            srv = lv.get("server") or {}
            preempt += srv.get("preemptions", 0) or 0
            if lv["succeeded"] != lv["launched"]:
                bad.append(f"c={lv['level']} {lv['succeeded']}/{lv['launched']}")
            # With pinned seeds a second run of the SAME ref on the same boot finds its
            # tails already in the prefix cache and measures prefill as free. The shared
            # prefix alone is ~85% of the prompt; anything above that is contamination.
            hit = srv.get("cache_hit_pct")
            if hit is not None and hit > contaminated:
                warn.append(f"{f.name} c={lv['level']} cache_hit {hit:.1f}% (> {contaminated:.1f}%)")
            by_level.setdefault(lv["level"], []).append({**lv, "server": srv})
    if warn:
        print("  WARNING: tails served from the prefix cache — same seeds already run on this boot?")
        for w in warn:
            print(f"    {w}")
    _print_levels(by_level)
    return {
        "ok": preempt == 0 and not bad,
        "preemptions": preempt,
        "incomplete": bad,
        "cache_contaminated": warn,
        "reps": [f.name for f in files],
        "levels": _level_medians(by_level),
    }


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def _level_medians(by_level: dict[int, list[dict]]) -> dict:
    out = {}
    for c, rows in sorted(by_level.items()):
        out[str(c)] = {
            "tps_per_req": _median([r["tps_per_req"] for r in rows]),
            "tps_aggregate": _median([r["tps_aggregate"] for r in rows]),
            "ttft_p50": _median([r["ttft_p50"] for r in rows]),
            "ttft_p95": _median([r["ttft_p95"] for r in rows]),
            "cache_hit_pct": _median([r["server"].get("cache_hit_pct") for r in rows]),
            "accept_len": _median([r["server"].get("accept_len") for r in rows]),
            "accept_len_client": _median([r.get("accept_len_client") for r in rows]),
            # agentic shape (None on fixed-shape kinds): what the level actually looked like
            "tool_call_rate": _median([r.get("tool_call_rate") for r in rows]),
            "out_tokens_p50": _median([r.get("out_tokens_p50") for r in rows]),
            "uncached_tokens_mean": _median([r["server"].get("uncached_tokens_mean") for r in rows]),
            "prefill_tps": _median([r["server"].get("prefill_tps") for r in rows]),
            "accept_per_pos": next((r["server"].get("accept_per_pos") for r in rows
                                    if r["server"].get("accept_per_pos")), None),
            "score": _median([5 * r["ttft_p95"] + 1500 / r["tps_per_req"]
                              for r in rows if r["ttft_p95"] and r["tps_per_req"]]),
        }
    return out


def _print_levels(by_level: dict[int, list[dict]]) -> None:
    f = lambda v, w=6, d=2: f"{v:{w}.{d}f}" if isinstance(v, (int, float)) else "—".rjust(w)
    print(f"  {'c':>3} {'tps/req':>8} {'agg':>7} {'ttft50':>7} {'ttft95':>7} "
          f"{'cache%':>7} {'accept':>7} {'score':>7}   per-rep tps/req")
    for c, m in _level_medians(by_level).items():
        reps = " ".join(f(r["tps_per_req"], 5, 1) for r in by_level[int(c)])
        print(f"  {c:>3} {f(m['tps_per_req'], 8)} {f(m['tps_aggregate'], 7, 1)} "
              f"{f(m['ttft_p50'], 7)} {f(m['ttft_p95'], 7)} {f(m['cache_hit_pct'], 7, 1)} "
              f"{f(m['accept_len'], 7)} {f(m['score'], 7, 1)}   {reps}")


def _decode_shapes(prov, rundir) -> dict:
    """No numeric gate yet — recorded until enough runs exist to set one honestly."""
    out = f"{REMOTE}/decode-shapes.json"
    p = remote(
        f"cd {REMOTE} && python3 probe_decode_shapes.py "
        f"--url http://127.0.0.1:8000/v1/chat/completions "
        f"--model {prov['served_model_name']} "
        f"--metrics-url http://127.0.0.1:8000/metrics --out {out}",
        check=False,
    )
    sys.stdout.write(p.stdout[-2000:])
    sh(["scp", "-q", f"{HEAD}:{out}", str(rundir / "decode-shapes.json")], check=False)
    return {"ok": None, "recorded": (rundir / "decode-shapes.json").exists()}


# --------------------------------------------------------------------------- #
# list / restore / import
# --------------------------------------------------------------------------- #


def load_runs() -> list[dict]:
    out = []
    for d in sorted(RUNS.glob("*/")):
        mf = d / "manifest.json"
        if mf.exists():
            m = json.loads(mf.read_text())
            m["_dir"] = d
            out.append(m)
    return out


def cmd_list(args) -> int:
    for m in load_runs():
        badge = "full " if m.get("provenance") == "full" else "UNVER"
        gates = m.get("gates", {})
        mark = "".join(
            "." if gates.get(k) is None else "P" if gates[k] else "F"
            for k in ("smoke", "toolprobe", "loadgen", "decode_shapes")
        )
        print(f"{m['_dir'].name:44s} {badge}  {mark}  {(m.get('dspark_sha') or '—')[:8]:8s} "
              f"{m.get('served_model_name') or '?'}")
    return 0


def cmd_restore(args) -> int:
    m = next((r for r in load_runs() if r["ref"] == args.ref or r["_dir"].name == args.ref), None)
    if not m:
        print(f"no run matching {args.ref!r} — try `bench.py list`", file=sys.stderr)
        return 1
    if m.get("provenance") != "full":
        print(f"{args.ref} has {m.get('provenance')} provenance; cannot restore from it.",
              file=sys.stderr)
        return 1
    d = m["_dir"]
    # Whichever launcher produced the run, the other one's containers have to come down
    # first — they both want the whole GPU, and a half-stopped pair is the slowest way
    # to discover that.
    stop_compose = f"ssh {HEAD} 'cd {provenance.DSPARK_DIR} && ./stop-deepseek-v4-flash-dspark.sh'"
    # --all, not a bare `--cluster both`: stop needs a TARGET or --all, and the target
    # would be a cluster id that changes every launch.
    stop_sparkrun = "sparkrun stop --all --cluster both || true"
    stop_qwen38fn = f"ssh {HEAD} 'cd {provenance.QWEN38FN_DIR} && ./stop.sh' || true"
    QDIR = provenance.QWEN38FN_DIR

    if m.get("deployment") == "sparkrun":
        recipe = d / "recipe.yaml"
        steps = [
            stop_compose,
            stop_qwen38fn,
            stop_sparkrun,
            f"sparkrun run {recipe} --cluster both --no-follow",
        ]
    elif m.get("deployment") == "qwen38fn":
        # --no-download, not --launch: the checkpoint is already in both HF caches, but
        # the sync step is also what verifies the worker still has it. --launch skips
        # that check and fails ~7 min into a load instead.
        steps = [
            stop_sparkrun,
            stop_compose,
            stop_qwen38fn,
            f"ssh {HEAD} 'cd {QDIR} && git checkout {m['dspark_sha']}'",
            f"scp {d / 'env.qwen38fn'} {HEAD}:{QDIR}/.env",
        ]
        if m.get("dspark_dirty"):
            steps.append(
                f"scp {d / 'local.diff'} {HEAD}:/tmp/local.diff && "
                f"ssh {HEAD} 'cd {QDIR} && git apply /tmp/local.diff'"
            )
        # The recipe takes the draft vocabulary from MTP_DRAFT_VOCAB, and the arms that
        # measured it were launched with that on the command line -- so a captured
        # `.env` can be complete and still boot on the full vocabulary, which is a
        # silent -21% at c=1 and looks like a bad boot. Push the file back and pin it.
        vocab = d / "draft_vocab.txt"
        if vocab.exists():
            dest = f"{QDIR}/draft_vocab_restored.txt"
            steps.append(f"scp -q {vocab} {HEAD}:{dest}")
            steps.append(
                f"ssh {HEAD} \"grep -q '^MTP_DRAFT_VOCAB=' {QDIR}/.env "
                f"|| echo 'MTP_DRAFT_VOCAB=\\\"{dest}\\\"' >> {QDIR}/.env\""
            )
        elif "/etc/vllm-draft-vocab.txt" in (
            m.get("overlay", {}).get(provenance.HEAD, {}).get("binds", {})
        ):
            print(f"# WARNING: {args.ref} ran with a draft vocabulary mounted but the "
                  "file was not stored beside it.\n"
                  "# This restore cannot reproduce it. Use "
                  "qwen3.8-flash-next/bootstrap.sh instead.",
                  file=sys.stderr)
        elif "overlay" not in m:
            print(f"# WARNING: {args.ref} predates overlay capture (added 2026-09-06). "
                  "Its `.env` and\n"
                  "# local.diff were captured, but nothing recorded what was mounted "
                  "into the container,\n"
                  "# so this restore reproduces the recipe and not necessarily the run. "
                  "For the known-good\n"
                  "# config use qwen3.8-flash-next/bootstrap.sh.",
                  file=sys.stderr)
        steps.append(f"ssh {HEAD} 'cd {QDIR} && ./start.sh --no-download'")
    else:
        steps = [
            stop_sparkrun,
            stop_qwen38fn,
            stop_compose,
            f"ssh {HEAD} 'cd {provenance.DSPARK_DIR} && git checkout {m['dspark_sha']}'",
            f"scp {d / 'env.dspark'} {HEAD}:{provenance.DSPARK_DIR}/.env.dspark",
        ]
        if m.get("dspark_dirty"):
            steps.append(
                f"scp {d / 'local.diff'} {HEAD}:/tmp/local.diff && "
                f"ssh {HEAD} 'cd {provenance.DSPARK_DIR} && git apply /tmp/local.diff'"
            )
        steps.append(
            f"ssh {HEAD} 'cd {provenance.DSPARK_DIR} && ./start-deepseek-v4-flash-dspark.sh'"
        )

    print(f"# restore {m['_dir'].name}  ({m['model']} as {m['served_model_name']})")
    for s in steps:
        print(s)
    if not args.apply:
        print("\n# nothing executed. Re-run with --apply to perform these steps.")
        print("# this stops production and takes ~12 min (two model loads).")
        return 0
    for s in steps:
        print(f"\n+ {s}")
        if sh(["bash", "-c", s]).returncode != 0:
            print("step failed — stopping", file=sys.stderr)
            return 1
    return 0


LEGACY = Path("/home/alessandroalviani/dgx-cluster/opencode-bench/bench_results")
# The 08-12 arms were served under this name; it 404s on the current deployment, which
# proves .env.dspark changed in between. Recorded so the trap is documented, not
# rediscovered — it is why these rows can never be a trusted baseline.
LEGACY_NOTE = (
    "Pre-suite run: no provenance was captured. Config identity can only be inferred "
    "from the workload params and observed cache-hit/prefill numbers."
)


def _legacy_utc(started: str) -> datetime:
    """Legacy loadgen stamped `datetime.now()` — naive HEAD-NODE local time.

    New manifests are UTC, so mixing the two makes the dashboard's chronological order
    lie (a 10:27 CEST run sorted above an 08:53 UTC run that came after it). ZoneInfo
    gets DST right per-date; a fixed offset would not.
    """
    from zoneinfo import ZoneInfo

    naive = datetime.fromisoformat(started)
    return naive.replace(tzinfo=ZoneInfo(HEAD_TZ)).astimezone(timezone.utc)


def cmd_import_legacy(args) -> int:
    n = 0
    for arm in sorted(LEGACY.glob("*/")):
        files = sorted(arm.glob("loadgen-*.json"))
        if not files:
            continue
        first = json.loads(files[0].read_text())["meta"]
        stamp = _legacy_utc(first["started"]).strftime("%Y-%m-%dT%H-%MZ")
        rundir = RUNS / f"{stamp}__legacy-{arm.name}"
        rundir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, rundir / f.name)
        note = LEGACY_NOTE
        if first["model"] != "deepseek-v4-flash-dspark":
            note += (f" NOTE: served as {first['model']!r}, which 404s on the current "
                     "deployment — the serving config demonstrably differed.")
        (rundir / "manifest.json").write_text(json.dumps({
            "ref": f"legacy-{arm.name}",
            "provenance": "unverified",
            "captured_utc": first["started"],
            "served_model_name": first["model"],
            "model": None, "dspark_sha": None, "vllm_version": None,
            "serve_flags": {}, "hotfixes": {},
            "workload": {k: first.get(k) for k in
                         ("prompt_tokens", "unique_tokens", "max_tokens", "levels",
                          "workload")},
            "note": note,
            "gates": {}, "results": {},
        }, indent=1))
        n += 1
        print(f"imported {rundir.name}  ({len(files)} runs, served as {first['model']})")
    print(f"\n{n} legacy arms imported, all flagged `unverified`.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="capture provenance, run the suite, store results")
    r.add_argument("--ref", required=True, help="label for this run, e.g. 103af68c-no26-no27")
    r.add_argument("--skip", default="", help="comma-separated test names to skip")
    r.add_argument("--keep-going", action="store_true", help="continue past a failed gate")
    r.add_argument("--tail-seed", type=int, default=TAIL_SEED_BASE,
                   help="base seed for the prompt tails (warm-up = base, rep i = base+i). "
                        "Leave at the default so every arm runs identical prompts.")
    r.add_argument("--workload", choices=KINDS, default=WORKLOAD,
                   help=f"prompt kind and protocol (PROTOCOLS). Default {WORKLOAD}. "
                        "'agentic' is the production shape: ~65k prompt, tool loop, "
                        "reasoning, natural stops, sweep 2,3.")
    r.add_argument("--sweep", default=None,
                   help="override the kind's concurrency levels, e.g. 1,2,3,6")
    r.add_argument(
        "--recipe",
        help="sparkrun recipe that launched the deployment. Required when the box is "
        "served by sparkrun rather than the ~/dspark-mia compose launcher — there is "
        "no checkout to read an identity off, so the recipe is the identity.",
    )
    r.set_defaults(fn=cmd_run)

    sub.add_parser("list", help="one line per stored run").set_defaults(fn=cmd_list)

    d = sub.add_parser("dashboard", help="regenerate dashboard.html")
    d.set_defaults(fn=lambda a: __import__("dashboard").build())

    s = sub.add_parser("restore", help="print (or --apply) the steps back to a config")
    s.add_argument("ref")
    s.add_argument("--apply", action="store_true", help="actually perform the steps")
    s.set_defaults(fn=cmd_restore)

    sub.add_parser("import-legacy", help="import pre-suite opencode-bench arms"
                   ).set_defaults(fn=cmd_import_legacy)

    args = p.parse_args()
    if getattr(args, "skip", None) is not None:
        args.skip = {s for s in str(args.skip).split(",") if s}
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
