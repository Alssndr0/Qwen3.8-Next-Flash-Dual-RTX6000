#!/usr/bin/env python3
"""provenance.py — capture exactly what produced a number.

On 2026-08-14 the only baseline available was two loadgen runs from 08-12. Nothing
recorded the config behind them. Reconstructing it took git-reflog archaeology and still
ended ambiguous: those runs were served under `model=deepseek-v4-flash-0731`, a name that
404s today, proving .env.dspark had changed — so config identity could only be inferred
from matching cache-hit and prefill numbers, never confirmed.

Every field here is a command that was run by hand that day. A run without provenance is
worse than no run: it looks like evidence and cannot be checked.

Design rule: capture FAILS the run. A partial manifest is the exact trap this replaces.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HEAD = "10.150.0.50"
WORKER = "10.150.0.51"
DSPARK_DIR = "/home/nvidia/dspark-mia"
CONTAINER = "deepseek-v4-flash-vllm-dspark-1"
QWEN38FN_DIR = "/home/nvidia/qwen38-fn-vllm"
QWEN38FN_CONTAINER = "vllm-fn"

# The box can be served two ways and they have different identity sources. The compose
# launcher's identity is a git checkout on the head; a sparkrun-launched recipe has no
# checkout at all — the runtime is built inside the container from a commit pinned in
# the recipe's pre_exec. Capturing the wrong one is worse than capturing nothing: the
# ~/dspark-mia checkout still sits there at 103af68c and would happily stamp a
# MiaAI-Lab sha onto a tonyd2wild run.
SPARKRUN_HEAD_CONTAINER = "_node_0"  # sparkrun's vllm-distributed rank-0 name suffix
VLLM_SITE_PACKAGES = "/opt/env/lib/python3.12/site-packages/vllm"
SPARKRUN_SERVE_SCRIPT = "/tmp/sparkrun_serve.sh"


class ProvenanceError(RuntimeError):
    """Raised when any field cannot be captured — aborts the run."""


def ssh(host: str, cmd: str, *, allow_empty: bool = False) -> str:
    p = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, cmd],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if p.returncode != 0:
        raise ProvenanceError(f"{host}: `{cmd}` exited {p.returncode}: {p.stderr.strip()}")
    out = p.stdout.strip()
    if not out and not allow_empty:
        raise ProvenanceError(f"{host}: `{cmd}` returned nothing")
    return out


def _env_value(env_text: str, key: str) -> str | None:
    """Last uncommented assignment wins — matches how docker compose --env-file reads it."""
    val = None
    for line in env_text.splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            val = v.strip()
    return val


def parse_serve_cmd(cmd: str) -> dict:
    """`vllm serve <model> --flag value --bool-flag …` -> {flag: value, _model: str}.

    The serve line is the one place every knob is resolved — env defaults, compose
    substitution and CLI overrides have all collapsed by then. Diffing two of these is
    the fastest way to answer "did the config actually change?".
    """
    toks = cmd.split()
    flags: dict[str, str | bool] = {}
    try:
        i = toks.index("serve")
        flags["_model"] = toks[i + 1]
    except (ValueError, IndexError):
        pass
    i = 0
    while i < len(toks):
        if toks[i].startswith("--"):
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if nxt is None or nxt.startswith("--"):
                flags[toks[i]] = True
                i += 1
            else:
                flags[toks[i]] = nxt
                i += 2
        else:
            i += 1
    return flags


def detect(host: str = HEAD) -> tuple[str, str]:
    """Which launcher is serving the box right now -> (deployment, container name).

    Checked against running containers rather than trusting a flag: the failure this
    prevents is a manifest that names the ~/dspark-mia checkout while the GPU is
    actually running a recipe built from somewhere else entirely.
    """
    names = ssh(host, "docker ps --format '{{.Names}}'", allow_empty=True).split()
    if CONTAINER in names:
        return "compose", CONTAINER
    # The Qwen/vLLM recipe names its rank-0 and rank-1 containers identically, so the
    # name alone does not say which rank this is — but we only ever detect on HEAD.
    if QWEN38FN_CONTAINER in names:
        return "qwen38fn", QWEN38FN_CONTAINER
    node0 = [n for n in names if n.endswith(SPARKRUN_HEAD_CONTAINER)]
    if len(node0) == 1:
        return "sparkrun", node0[0]
    if node0:
        raise ProvenanceError(f"{host}: several sparkrun rank-0 containers: {node0}")
    raise ProvenanceError(
        f"{host}: no serving container found (looked for {CONTAINER!r}, "
        f"{QWEN38FN_CONTAINER!r} and *{SPARKRUN_HEAD_CONTAINER}). "
        f"Running: {names or '(none)'}"
    )


def _identity_compose(prov: dict) -> None:
    """Identity of the ~/dspark-mia compose deployment: a git checkout on the head."""
    env_text = ssh(HEAD, f"cat {DSPARK_DIR}/.env.dspark")
    local_diff = ssh(HEAD, f"cd {DSPARK_DIR} && git diff", allow_empty=True)
    prov["dspark_sha"] = ssh(HEAD, f"git -C {DSPARK_DIR} rev-parse HEAD")
    prov["dspark_dirty"] = bool(local_diff)
    prov["config_file"] = "env.dspark"
    prov["config_text"] = env_text
    prov["local_diff"] = local_diff
    for key in ("SERVED_MODEL_NAME", "DSPARK_REVISION", "MAX_NUM_SEQS",
                "MAX_NUM_BATCHED_TOKENS", "MAX_MODEL_LEN", "MTP_NUM_TOKENS",
                "DEFAULT_THINKING", "LONG_PREFILL_TOKEN_THRESHOLD"):
        prov[key.lower()] = _env_value(env_text, key)


def _identity_sparkrun(prov: dict, recipe: str) -> None:
    """Identity of a sparkrun-launched recipe: the recipe file plus the commit its
    pre_exec builds the in-container runtime from.

    There is no checkout to read a sha off, and the container's own image is only the
    public base — the overlay commit is the thing that actually decides what code runs,
    so that is what goes in `dspark_sha`. The whole recipe is stored verbatim beside it,
    which is stronger provenance than the compose path's env+diff pair.
    """
    text = Path(recipe).read_text()
    m = re.search(r"^\s*COMMIT=([0-9a-f]{40})\s*$", text, re.M)
    if not m:
        raise ProvenanceError(
            f"{recipe}: no `COMMIT=<40-hex>` in the pre_exec overlay step. That line is "
            "the only record of which source the runtime was built from."
        )
    prov["dspark_sha"] = m.group(1)
    prov["dspark_dirty"] = False  # the recipe is stored whole; there is no base to diff
    prov["config_file"] = "recipe.yaml"
    prov["config_text"] = text
    prov["local_diff"] = ""
    prov["recipe_path"] = str(Path(recipe).resolve())
    prov["recipe_sha256"] = hashlib.sha256(text.encode()).hexdigest()

    # No .env to read these off — take them from the resolved serve line, which is
    # where every knob has collapsed anyway.
    f = prov["serve_flags"]
    prov["served_model_name"] = f.get("--served-model-name")
    prov["dspark_revision"] = None
    prov["max_num_seqs"] = f.get("--max-num-seqs")
    prov["max_num_batched_tokens"] = f.get("--max-num-batched-tokens")
    prov["max_model_len"] = f.get("--max-model-len")
    prov["long_prefill_token_threshold"] = f.get("--long-prefill-token-threshold")
    spec = json.loads(f.get("--speculative-config", "{}").strip("'\""))
    prov["mtp_num_tokens"] = str(spec.get("num_speculative_tokens", ""))
    kw = json.loads(f.get("--default-chat-template-kwargs", "{}").strip("'\""))
    prov["default_thinking"] = "on" if kw.get("thinking") else "off"


def _identity_qwen38fn(prov: dict) -> None:
    """Identity of the ~/qwen38-fn-vllm deployment: a git checkout on the head.

    Same shape as the compose path — a MiaAI-Lab recipe checkout plus the site `.env`
    it was launched with — but the knobs are read off the resolved serve line rather
    than out of `.env`. This recipe renders serve flags from `.env` through several
    layers of defaulting (`.env` beats the environment, `start.sh` supplies its own
    fallbacks, `EXTRA_VLLM_ARGS` is appended last, YaRN is force-disabled at or below
    native context), so `.env` states an intent and only the serve line states what
    the engine got. The `.env` is still stored verbatim beside the run.
    """
    env_text = ssh(HEAD, f"cat {QWEN38FN_DIR}/.env")
    local_diff = ssh(HEAD, f"cd {QWEN38FN_DIR} && git diff", allow_empty=True)
    prov["dspark_sha"] = ssh(HEAD, f"git -C {QWEN38FN_DIR} rev-parse HEAD")
    prov["dspark_dirty"] = bool(local_diff)
    prov["config_file"] = "env.qwen38fn"
    prov["config_text"] = env_text
    prov["local_diff"] = local_diff
    prov["recipe_repo"] = "MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks"

    f = prov["serve_flags"]
    prov["served_model_name"] = f.get("--served-model-name")
    prov["dspark_revision"] = None
    prov["max_num_seqs"] = f.get("--max-num-seqs")
    prov["max_num_batched_tokens"] = f.get("--max-num-batched-tokens")
    prov["max_model_len"] = f.get("--max-model-len")
    # Not a knob this recipe exposes. Recorded as absent rather than omitted so it is
    # distinguishable from "set to None" when diffed against a dspark run.
    prov["long_prefill_token_threshold"] = f.get("--long-prefill-token-threshold", "(absent)")
    spec = json.loads(f.get("--speculative-config", "{}").strip("'\""))
    prov["mtp_num_tokens"] = str(spec.get("num_speculative_tokens", ""))
    # This build has no --default-chat-template-kwargs and thinking cannot be turned off
    # server-side; it is a per-request flag only.
    prov["default_thinking"] = "on"
    # Serving-shape facts that have no dspark equivalent, so they would otherwise vanish
    # from the record even though they move every number in the suite.
    prov["kv_cache_dtype"] = f.get("--kv-cache-dtype")
    prov["gpu_memory_utilization"] = f.get("--gpu-memory-utilization")
    prov["expert_parallel"] = bool(f.get("--enable-expert-parallel"))


# Env vars worth recording: the ones a recipe or a launch can set. The image sets a lot
# more (NVIDIA_REQUIRE_CUDA alone is 2 KB of driver matrix) and none of it varies while
# the image digest is pinned, which the manifest already records.
OVERLAY_ENV_PREFIXES = (
    "VLLM_", "NCCL_", "GLOO_", "TP_", "HF_", "TRANSFORMERS_",
    "OMP_", "TORCH_", "FLASHINFER_", "MBX_", "CUDA_",
)


def _overlay(host: str, ctr: str) -> dict:
    """What is mounted into a running container, and what those files contain.

    Added 2026-09-06 after the four Qwen arms of that day proved un-restorable. They
    captured a byte-identical `.env` and a byte-identical `local.diff` while their
    resolved serve lines differed, because the arms were driven by command-line
    overrides -- and the draft-vocabulary arm differed from its predecessor mainly in a
    92 KB file mounted at /etc/vllm-draft-vocab.txt that nothing recorded at all. Two
    runs whose only difference is the contents of a mounted file were indistinguishable
    in the record, and `bench.py restore` would have rebuilt the same server for both.

    Keyed by mount TARGET, not source: the head mounts its overlay out of the recipe
    checkout and the worker out of /tmp/vllm-overlay, so sources legitimately differ
    across ranks while contents must not.
    """
    binds = json.loads(
        ssh(host, f"docker inspect {ctr} --format '{{{{json .HostConfig.Binds}}}}'")
    ) or []
    env_list = json.loads(
        ssh(host, f"docker inspect {ctr} --format '{{{{json .Config.Env}}}}'")
    ) or []

    out: dict[str, dict] = {}
    for b in binds:
        parts = b.split(":")
        if len(parts) < 2:
            continue
        source, target = parts[0], parts[1]
        out[target] = {"source": source, "sha256": None}

    # One round trip for every hash. Directory mounts (the HF and vLLM caches) hash to
    # nothing, which is correct -- their identity is the model snapshot, recorded
    # separately, not a tree digest we would have to keep stable.
    srcs = " ".join(f"'{v['source']}'" for v in out.values())
    if srcs:
        hashes = ssh(
            host,
            f"for p in {srcs}; do "
            f"if [ -f \"$p\" ]; then sha256sum \"$p\"; fi; done",
            allow_empty=True,
        )
        by_src = {}
        for line in hashes.splitlines():
            digest, _, path = line.partition("  ")
            by_src[path.strip()] = digest.strip()
        for v in out.values():
            v["sha256"] = by_src.get(v["source"])

    env = {}
    for item in env_list:
        k, _, v = item.partition("=")
        if k.startswith(OVERLAY_ENV_PREFIXES):
            env[k] = v
    return {"binds": out, "env": env}


def capture(bench_sha: str, recipe: str | None = None) -> dict:
    """Full provenance for the currently-running deployment. Raises on any gap."""
    deployment, container = detect()
    if deployment == "sparkrun" and not recipe:
        raise ProvenanceError(
            f"{container} is a sparkrun deployment; pass --recipe <file> so the run "
            "records which recipe produced it. There is no checkout to infer it from."
        )

    # Where the resolved serve line lives differs by launcher. compose runs vllm as the
    # container command, so `.Config.Cmd` is it. sparkrun runs `sleep infinity` as PID 1
    # and execs the real thing from a generated script — inspecting Cmd there returns
    # the base64 sleep shim, which parses to no model at all.
    if deployment == "compose":
        serve_cmd = ssh(
            HEAD, f"docker inspect {container} --format '{{{{join .Config.Cmd \" \"}}}}'"
        )
    elif deployment == "qwen38fn":
        # This image sets Entrypoint ["vllm","serve"] and passes the model plus every
        # flag as Cmd, so Cmd alone starts at the model and has no `serve` token for
        # parse_serve_cmd to anchor on -- it would find no model and abort the run.
        # The resolved line is entrypoint + cmd; capture both.
        serve_cmd = ssh(
            HEAD,
            f"docker inspect {container} --format "
            "'{{join .Config.Entrypoint \" \"}} {{join .Config.Cmd \" \"}}'",
        )
    else:
        serve_cmd = ssh(HEAD, f"docker exec {container} cat {SPARKRUN_SERVE_SCRIPT}")

    prov = {
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": "full",
        "bench_sha": bench_sha,
        "deployment": deployment,
        "container": container,
        "serve_cmd": serve_cmd,
        "image": ssh(HEAD, f"docker inspect {container} --format '{{{{.Config.Image}}}}'"),
        "image_id": ssh(HEAD, f"docker inspect {container} --format '{{{{.Image}}}}'"),
        "vllm_version": ssh(
            HEAD,
            f"docker exec {container} python3 -c "
            "'import vllm; print(vllm.__version__)'",
        ),
        # getattr-with-default, not a bare attribute read: the retention interval is a
        # MiaAI-Lab fork addition (the #26 hybrid prefix-cache work), so on any other
        # runtime `vllm.envs.__getattr__` raises and the whole capture used to abort.
        # "the knob does not exist here" is a fact about the deployment, not a gap in
        # the record — the gap would be failing to tell it apart from "set to None".
        "retention_interval": ssh(
            HEAD,
            f"docker exec {container} python3 -c "
            "'import vllm.envs as e; "
            "print(getattr(e, \"VLLM_PREFIX_CACHE_RETENTION_INTERVAL\", \"(absent)\"))'",
        ),
    }

    # Which in-container patches actually applied, per rank. The compose path's #26/#27
    # holds are mounted no-op files, so "absent" is the signal that a hold took; the
    # sparkrun path builds its runtime in pre_exec and marks it the same way. Both must
    # be checked on BOTH ranks — the worker is not a checkout, so the two can silently
    # disagree.
    # compose gives both ranks the same container name; sparkrun ranks them, so the
    # worker must be probed as _node_1. Querying the head's name on the worker returns
    # "no such container", which `|| true` turns into an empty set — a missing patch and
    # a mistyped container name would have looked identical.
    per_host = {HEAD: container, WORKER: container}
    if deployment == "sparkrun":
        per_host[WORKER] = container[: -len(SPARKRUN_HEAD_CONTAINER)] + "_node_1"

    prov["containers"] = per_host
    prov["hotfixes"] = {}
    for host, ctr in per_host.items():
        marks = ssh(
            host,
            f"docker logs {ctr} 2>&1 | grep -oE '\\[issue[0-9]+-hotfix\\]' "
            "| sort -u | tr -d '[]' || true",
            allow_empty=True,
        )
        applied = {m for m in marks.split() if m}
        # sparkrun's overlay runs via `docker exec` in pre_exec, so none of its output
        # reaches `docker logs`. The stamp file it leaves behind is the only per-rank
        # evidence that the runtime was actually rebuilt — and which commit from.
        stamp = ssh(
            host,
            f"docker exec {ctr} bash -lc "
            f"'ls -a {VLLM_SITE_PACKAGES} 2>/dev/null | grep sparkrun-dspark-overlay' || true",
            allow_empty=True,
        )
        applied |= {s.lstrip(".") for s in stamp.split() if s.strip()}
        prov["hotfixes"][host] = sorted(applied)

    # What is mounted in, and what it contains. See _overlay: this is the field that
    # makes two arms differing only in a mounted file tellable apart, and it is what
    # `bench.py restore` needs in order to put that file back.
    prov["overlay"] = {h: _overlay(h, c) for h, c in per_host.items()}

    # Model comes from the resolved serve line, not .env: .env picks between
    # DSPARK_MODEL_OFFICIAL and DSPARK_MODEL_ABLITERATED via the ABLITERATED flag, so
    # no single env key holds the answer. serve_flags is also what the dashboard diffs.
    prov["serve_flags"] = parse_serve_cmd(prov["serve_cmd"])
    prov["model"] = prov["serve_flags"].get("_model")
    if not prov["model"]:
        raise ProvenanceError("could not parse the model out of the serve command")

    if deployment == "compose":
        _identity_compose(prov)
    elif deployment == "qwen38fn":
        _identity_qwen38fn(prov)
    else:
        _identity_sparkrun(prov, recipe)

    if prov["hotfixes"][HEAD] != prov["hotfixes"][WORKER]:
        raise ProvenanceError(
            f"rank hotfix sets differ — head={prov['hotfixes'][HEAD]} "
            f"worker={prov['hotfixes'][WORKER]}. Split-brain config; fix before benching."
        )

    # Same rule for the overlay: the worker's copies are scp'd into /tmp by start.sh, so
    # a re-launch that failed to refresh them leaves the ranks running different code
    # with no other symptom. Compare content by mount target; sources differ by design.
    def _by_target(host: str) -> dict:
        return {t: v["sha256"] for t, v in prov["overlay"][host]["binds"].items()}

    head_o, worker_o = _by_target(HEAD), _by_target(WORKER)
    if head_o != worker_o:
        differing = sorted(set(head_o) ^ set(worker_o)) or sorted(
            t for t in head_o if head_o[t] != worker_o.get(t)
        )
        raise ProvenanceError(
            f"rank overlays differ at {differing} — head and worker are running "
            "different mounted files. Split-brain config; fix before benching."
        )
    return prov


if __name__ == "__main__":
    import sys

    print(json.dumps(capture("(cli)", sys.argv[1] if len(sys.argv) > 1 else None), indent=1))
