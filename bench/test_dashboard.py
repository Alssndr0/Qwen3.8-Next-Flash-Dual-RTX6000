#!/usr/bin/env python3
"""Checks for the per-level reduction the dashboard's expanded row shows.

Run: python3 test_dashboard.py

levels_for() is the one piece of real logic in dashboard.py: it medians every metric
across reps at every concurrency level. Everything else is string building.
"""
import json
import tempfile
from pathlib import Path

import agg
import dashboard


def _rep(path: Path, rows):
    """rows: (level, ttft_p95, tps_per_req, queue_mean|None)"""
    path.write_text(json.dumps({"summary": [
        {"level": c, "launched": c, "succeeded": c, "ttft_p95": t, "ttft_p50": t,
         "total_p95": t, "tps_per_req": s, "tps_aggregate": s and s * c,
         "tpot_mean": s and 1 / s,
         "out_tokens_mean": 300, "prompt_tokens_mean": 18000,
         "server": {"queue_mean": q, "prefill_mean": 1.0, "cache_hit_pct": 80.0,
                    "preemptions": 0}}
        for c, t, s, q in rows]}))


def main():
    # The scored level follows agg.TARGET_C, so the fixture does too — otherwise moving
    # TARGET_C (10 -> 5 on 2026-08-14) silently skips the summary/level agreement check.
    T = agg.TARGET_C
    OTHER = 1 if T != 1 else 2
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _rep(d / "loadgen-1.json", [(OTHER, 2.0, 50.0, 0.1), (T, 10.0, 10.0, 0.5)])
        _rep(d / "loadgen-2.json", [(OTHER, 4.0, 30.0, 0.3), (T, 20.0, 20.0, -1)])
        _rep(d / "loadgen-3.json", [(OTHER, 6.0, 40.0, 0.2)])
        # unscoreable: pre-2026-08-12 loadgen stamped nulls. Must not appear at all.
        _rep(d / "loadgen-4.json", [(T + 100, None, None, 0.1)])
        lv = {r["c"]: r for r in dashboard.levels_for(d)}

        assert set(lv) == {OTHER, T}, lv.keys()
        assert lv[OTHER]["n"] == 3 and lv[T]["n"] == 2
        assert lv[OTHER]["ttft_p95"] == 4.0                  # median of 2,4,6
        assert lv[OTHER]["tps_req"] == 40.0                  # median of 50,30,40
        assert lv[OTHER]["ok"] == f"{3 * OTHER}/{3 * OTHER}"
        assert lv[T]["ok"] == f"{2 * T}/{2 * T}"
        assert lv[T]["queue"] == 0.5, lv[T]["queue"]         # -1 sentinel dropped
        assert lv[T]["score"] == agg.STEPS * 15.0 + agg.TASK_OUT_TOKENS / 15.0

        # the summary row and the c=TARGET_C level must tell the same story
        m = dashboard.metrics_for(d)
        assert abs(m["ttft_p95"] - lv[T]["ttft_p95"]) < 1e-9
        assert abs(m["tps_req"] - lv[T]["tps_req"]) < 1e-9

        # a directory with no loadgen files is empty, not an exception
        assert dashboard.levels_for(Path(td) / "nope") == []
    print("ok")


if __name__ == "__main__":
    main()
