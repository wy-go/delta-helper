#!/usr/bin/env python3
"""Synthetic pool-run unit: burn one core for SECONDS, then write a JSON record (thread variables, affinity, host) to
$POOL_RUN_TMP (set when pool-run has --out) or the path given, and print it. Usage: pool_unit.py SECONDS [OUT]"""
import json
import os
import socket
import sys
import time

secs = float(sys.argv[1])
out = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("POOL_RUN_TMP")
t0, x = time.time(), 0
while time.time() - t0 < secs:
    x += 1
rec = {k: os.environ.get(k) for k in ("POOL_RUN_UNIT", "POOL_RUN_WORKER", "POOL_RUN_THREADS", "OMP_NUM_THREADS",
                                      "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
rec.update({"host": socket.gethostname(), "pid": os.getpid(), "affinity": len(os.sched_getaffinity(0)),
            "nice": os.nice(0), "seconds": round(time.time() - t0, 2), "loops": x})
try:
    import numpy  # noqa: F401
    from threadpoolctl import threadpool_info
    rec["blas_threads"] = sorted(set(i["num_threads"] for i in threadpool_info()))
except Exception:
    pass
if out:
    with open(out, "w") as f:
        json.dump(rec, f)
print(json.dumps(rec))
