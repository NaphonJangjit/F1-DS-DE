"""Autonomous overnight pipeline: pre-warm -> rebuild -> forest-of-forests -> report.

Run once in background:  python _run_all.py
Each step writes its own log file; the orchestrator prints progress markers to stdout.
A failed step is logged but does NOT abort the pipeline (the rebuild re-fetches any
cache gaps left by pre-warm, and the report degrades gracefully if results are missing).
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("pre-warm cache",       "_prewarm_cache.py",     "prewarm_run.log"),
    ("rebuild outcomes",     "_rebuild_outcomes.py",  "rebuild_outcomes_run.log"),
    ("forest-of-forests",    "_fforest.py",           "fforest_run.log"),
    ("write report",         "_write_report.py",      "write_report_run.log"),
]


def main():
    t0 = time.time()
    for name, script, log in STEPS:
        print(f"\n{'=' * 70}\nSTEP: {name}  ({script})\n{'=' * 70}", flush=True)
        t1 = time.time()
        with open(ROOT / log, "w", encoding="utf-8") as fh:
            r = subprocess.run([sys.executable, script], cwd=ROOT,
                               stdout=fh, stderr=subprocess.STDOUT)
        print(f"STEP '{name}' exit={r.returncode}  ({time.time() - t1:.0f}s; "
              f"total {time.time() - t0:.0f}s)", flush=True)
        if r.returncode != 0:
            print(f"  !! step failed (exit {r.returncode}); continuing anyway — "
                  f"see {log}", flush=True)

    print(f"\nALL STEPS DONE in {time.time() - t0:.0f}s", flush=True)
    print("Outputs: data_*_out.parquet, fforest_results.json, fif_report.md", flush=True)


if __name__ == "__main__":
    main()
