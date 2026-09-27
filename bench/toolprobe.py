#!/usr/bin/env python3
"""toolprobe.py — tool-name fidelity gate.

Two tools, one applicable, max_tokens 800, N runs. PASS requires the model to emit the
tool name EXACTLY as declared, with valid JSON arguments.

Why this shape: upstream issue #39 (MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) is a
prefix-cache bug that manifests as the model *paraphrasing* tool names it can see —
`run_shell` -> `run_command` / `exec` / `ssh` — plus outright confabulations. Two sites
independently bisected it to the issue #26 hotfix. It does not show up as an error, a
crash, or a latency change; only as wrong names and blank turns. Nothing else in this
suite would catch it, and agentic tool calling is the entire opencode workload.

Deliberately distinctive names: a paraphrase of `run_shell` is unambiguous, whereas a
generic `search` could be "corrected" to something plausible for innocent reasons.

Exit code 0 only on N/N. Usage: toolprobe.py [label] [n]
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

BASE = os.environ.get("PROBE_URL", "http://127.0.0.1:8000")
URL = BASE + "/v1/chat/completions"
MODEL = os.environ.get("PROBE_MODEL", "deepseek-v4-flash-dspark")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command on the host and return stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_logs",
            "description": "Search application logs for a pattern.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
]
NAMES = {t["function"]["name"] for t in TOOLS}
EXPECT = "run_shell"
PROMPT = "Check how much free disk space is on /var. Use the tools available to you."


def once() -> tuple[bool, str]:
    body = json.dumps(
        {
            "model": MODEL,
            "tools": TOOLS,
            "tool_choice": "auto",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 800,
            "temperature": 1.0,
            "top_p": 0.95,
        }
    ).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    ch = d["choices"][0]
    msg, fin = ch["message"], ch["finish_reason"]
    calls = msg.get("tool_calls") or []
    if not calls:
        # issue #39 also reported blank turns: finish_reason=length, empty content.
        content = (msg.get("content") or "").strip()
        return False, f"no tool_call (finish={fin}, content_len={len(content)})"
    name = calls[0]["function"]["name"]
    try:
        json.loads(calls[0]["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return False, f"{name}: invalid JSON arguments"
    if name not in NAMES:
        return False, f"CONFABULATED NAME: {name!r}"
    if name != EXPECT:
        return False, f"wrong tool chosen: {name} (expected {EXPECT})"
    return True, name


def probe(n: int = 10) -> dict:
    results = []
    for i in range(n):
        try:
            good, detail = once()
        except Exception as e:  # noqa: BLE001 — any failure is a failed probe
            good, detail = False, f"{type(e).__name__}: {e}"
        results.append({"i": i + 1, "pass": good, "detail": detail})
        print(f"  {i + 1:2d} {'PASS' if good else 'FAIL'}  {detail}", flush=True)
    passed = sum(r["pass"] for r in results)
    return {
        "test": "toolprobe",
        "n": n,
        "passed": passed,
        "gate": "all",
        "ok": passed == n,
        "model": MODEL,
        "url": URL,
        "results": results,
    }


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    summary = probe(n)
    print(f"\n[{label}] tool-name fidelity: {summary['passed']}/{n}")
    out = os.environ.get("PROBE_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(summary, f, indent=1)
        print(f"wrote {out}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
