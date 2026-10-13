#!/usr/bin/env python3
"""End-to-end self-test of bin/pool-run on this machine (login node or job). Prints one PASS/FAIL line per check with its
evidence. Run: python3 tests/test_pool_run.py [--root DIR] [--py PYTHON]
  --root  directory for the test pools (default /projects/bgwo/wanyu/pool_run_selftest; any shared directory works)
  --py    Python that runs the synthetic units (an env with numpy + threadpoolctl also reports BLAS threads)
Each run uses a fresh <root>/<host>-<time>/ directory. Units are synthetic: burn one core for a few seconds and write JSON.
"""
import argparse
import glob
import json
import os
import signal
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
POOL_RUN = os.path.join(HERE, "..", "bin", "pool-run")
UNIT = os.path.join(HERE, "pool_unit.py")
RESULTS = []


def check(name, ok, evidence):
    RESULTS.append(ok)
    print("%s %s: %s" % ("PASS" if ok else "FAIL", name, evidence))
    sys.stdout.flush()


def units_file(path, n, secs, py, compound=False, prefix="u"):
    with open(path, "w") as f:
        for i in range(n):
            s = secs[i % len(secs)] if isinstance(secs, list) else secs
            cmd = "%s %s %s" % (py, UNIT, s)                   # writes $POOL_RUN_TMP when pool-run has --out
            if compound:
                cmd = cmd + "; true"                      # bash stays as the parent: tests orphan cleanup
            f.write("%s%02d\t%s\n" % (prefix, i, cmd))
    return path


def pr(args, **kw):
    return subprocess.run([sys.executable, POOL_RUN] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True, **kw)


def start(args, log):
    return subprocess.Popen([sys.executable, POOL_RUN] + args, stdout=open(log, "w"), stderr=subprocess.STDOUT)


def results(pool):
    p = os.path.join(pool, "results.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def group_pids(pgid):
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                if int(open("/proc/%s/stat" % d).read().rsplit(")", 1)[1].split()[2]) == int(pgid):
                    out.append(int(d))
            except (OSError, IndexError, ValueError):
                pass
    return out


def wait_for(cond, timeout, step=0.2):
    t = time.time()
    while time.time() - t < timeout:
        if cond():
            return True
        time.sleep(step)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/projects/bgwo/wanyu/pool_run_selftest")
    ap.add_argument("--py", default=sys.executable)
    a = ap.parse_args()
    host = socket.gethostname().split(".")[0]
    base = os.path.join(a.root, "%s-%s" % (host, time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(base)
    print("host %s, slurm job %s, affinity %d of %d cores, load %s; test dir %s" % (
        host, os.environ.get("SLURM_JOB_ID"), len(os.sched_getaffinity(0)), os.cpu_count(),
        " ".join("%.1f" % x for x in os.getloadavg()), base))

    # 1. workers capped at free cores; refused with none free
    U = units_file(os.path.join(base, "units_cap.txt"), 4, 1, a.py)
    r = pr(["--pool", os.path.join(base, "cap"), "--units", U, "--workers", "100000", "--dry-run"])
    line = [x for x in r.stdout.splitlines() if "workers" in x]
    plan = [x for x in r.stdout.splitlines() if "free" in x]
    w = int(line[-1].split("workers ")[1].split()[0]) if line else -1
    free = int(plan[0].split("free ")[1].split(";")[0].split()[0]) if plan else -1
    check("workers capped at free cores", 0 < w <= max(free, 0) and w < 100000,
          "requested 100000 -> %d workers; plan: %s" % (w, " | ".join(dict.fromkeys(x.split("] ", 1)[1] for x in plan + line))))
    r = pr(["--pool", os.path.join(base, "cap"), "--units", U, "--threads", "100000"])
    check("refuses when no worker fits", r.returncode == 3, "rc %d: %s" % (r.returncode, r.stderr.strip()[:200]))

    # 2. a run: pinned threads, results streaming, temp-then-rename outputs
    pool = os.path.join(base, "run")
    U = units_file(os.path.join(base, "units_run.txt"), 8, [1, 2, 3, 4], a.py)
    outpat = os.path.join(base, "run_out", "{id}.json")
    t0 = time.time()
    p = start(["--pool", pool, "--units", U, "--out", outpat, "--workers", "4"], os.path.join(base, "run.log"))
    seen = []
    while p.poll() is None:
        n = len(results(pool))
        if not seen or n > seen[-1][1]:
            seen.append((round(time.time() - t0, 1), n))
        time.sleep(0.1)
    t_exit = round(time.time() - t0, 1)
    res = results(pool)
    outs = sorted(glob.glob(os.path.join(base, "run_out", "*.json")))
    tmps = glob.glob(os.path.join(base, "run_out", "*.tmp.*"))
    recs = [json.load(open(o)) for o in outs]
    pins = set((r["OMP_NUM_THREADS"], r["MKL_NUM_THREADS"], r["OPENBLAS_NUM_THREADS"], r["NUMEXPR_NUM_THREADS"]) for r in recs)
    blas = set(tuple(r.get("blas_threads", [])) for r in recs)
    check("run finished", p.returncode == 0 and len(res) == 8 and all(x["status"] == "done" for x in res),
          "rc %d, %d result lines, outputs %d, leftover temp files %d" % (p.returncode, len(res), len(outs), len(tmps)))
    check("threads pinned to 1 per worker", pins == {("1", "1", "1", "1")} and blas in ({()}, {(1,)}),
          "OMP/MKL/OPENBLAS/NUMEXPR seen by units: %s; BLAS threads reported by threadpoolctl: %s; nice %s" % (
              sorted(pins), sorted(blas), sorted(set(r["nice"] for r in recs))))
    check("at most 4 units at once", max(int(r["POOL_RUN_WORKER"]) for r in recs) <= 3,
          "worker slots used %s" % sorted(set(int(r["POOL_RUN_WORKER"]) for r in recs)))
    first_line = seen[1][0] if len(seen) > 1 else None
    check("results log streams unit by unit", first_line is not None and first_line < t_exit - 1 and len(seen) >= 4,
          "(seconds since launch, result lines): %s; launcher exited at %.1f s" % (seen, t_exit))
    r = pr(["--pool", pool, "--units", U, "--out", outpat])
    check("resume skips finished units", r.returncode == 0 and "nothing to do" in r.stdout,
          r.stdout.strip().splitlines()[-1].split("] ", 1)[-1])

    # 3. a second launcher is refused by the lock
    pool = os.path.join(base, "lock")
    U = units_file(os.path.join(base, "units_lock.txt"), 4, 6, a.py)
    p = start(["--pool", pool, "--units", U, "--workers", "2"], os.path.join(base, "lock_a.log"))
    wait_for(lambda: glob.glob(os.path.join(pool, "locks", "*.lock")), 15)
    time.sleep(1)
    r2 = pr(["--pool", pool, "--units", U, "--workers", "2"])
    r3 = pr(["--pool", pool, "--units", U, "--workers", "2", "--lane", "another-lane"])
    p.wait()
    check("second launcher refused by the lock", r2.returncode == 4, "rc %d: %s" % (r2.returncode, r2.stderr.strip()))
    insl = bool(os.environ.get("SLURM_JOB_ID"))
    check("second lane on the same host refused outside Slurm" if not insl else "another lane name inside a job",
          (r3.returncode == 4) if not insl else r3.returncode in (0, 6),
          "rc %d: %s" % (r3.returncode, (r3.stderr.strip() or r3.stdout.strip().splitlines()[-1])[:220]))
    ran = [x["unit"] for x in results(pool) if x["status"] == "done"]
    check("first launcher unaffected; every unit ran exactly once", sorted(ran) == ["u00", "u01", "u02", "u03"] and
          (p.returncode == 0 if not insl else p.returncode in (0, 6)),
          "first launcher rc %d (6 = it left units to the other live launcher); units run: %s by launchers %s" % (
              p.returncode, sorted(ran), sorted(set(x["launcher_pid"] for x in results(pool)))))

    # 4a. a killed unit process: the unit fails and is retried
    pool = os.path.join(base, "killunit")
    U = units_file(os.path.join(base, "units_killunit.txt"), 1, 6, a.py)
    p = start(["--pool", pool, "--units", U, "--retries", "1"], os.path.join(base, "killunit.log"))
    cpath = os.path.join(pool, "claims", "u00.json")
    wait_for(lambda: os.path.exists(cpath) and "pgid" in (open(cpath).read() or "{}"), 15)
    time.sleep(1)
    pg = json.load(open(cpath))["pgid"]
    os.kill(pg, signal.SIGKILL)
    p.wait()
    res = results(pool)
    check("killed unit process -> failed attempt, then rerun", [x["status"] for x in res] == ["failed", "done"],
          "; ".join("attempt %d %s %s" % (x["attempt"], x["status"], x.get("note", "")) for x in res))

    # 4b. a killed launcher: its claims go stale (pid gone), orphans are cleaned, a new launcher reruns the units
    pool = os.path.join(base, "killlauncher")
    U = units_file(os.path.join(base, "units_killlauncher.txt"), 2, 8, a.py, compound=True)
    p = start(["--pool", pool, "--units", U, "--workers", "2"], os.path.join(base, "killlauncher_a.log"))
    claims = lambda: glob.glob(os.path.join(pool, "claims", "*.json"))  # noqa: E731
    wait_for(lambda: len(claims()) == 2 and all("pgid" in open(c).read() for c in claims()), 15)
    # wait until bash has started the python unit (bash -c may first run BASH_ENV, e.g. an lmod refresh)
    def unit_started(pg):
        for x in group_pids(pg):
            try:
                if b"pool_unit.py" in open("/proc/%d/cmdline" % x, "rb").read() and x != pg:
                    return True
            except OSError:
                pass
        return False
    wait_for(lambda: all(unit_started(json.load(open(c))["pgid"]) for c in claims()), 15)
    first_pid = p.pid
    p.send_signal(signal.SIGKILL)
    p.wait()
    time.sleep(1)
    left = [json.load(open(c)) for c in claims()]
    orphans = [x for c in left for x in group_pids(c["pgid"])]
    t1 = time.time()
    r = pr(["--pool", pool, "--units", U, "--workers", "2"])
    res = results(pool)
    took = [x.split("] ", 1)[1] for x in r.stdout.splitlines() if "took over" in x]
    check("killed launcher -> claims stale, orphans killed, units rerun", r.returncode == 0 and len(res) == 2 and
          len(orphans) >= 1 and not any(group_pids(c["pgid"]) for c in left) and
          all(x["status"] == "done" and x["launcher_pid"] != first_pid for x in res),
          "claims left by killed pid %d: %d; orphaned unit processes still running before the rerun: %d; second launcher rc %d in %.0f s; %s" % (
              first_pid, len(left), len(orphans), r.returncode, time.time() - t1, " | ".join(took)))

    # 4c. claims from another host: stale by heartbeat age (taken over) or live (left alone)
    pool = os.path.join(base, "remote")
    U = units_file(os.path.join(base, "units_remote.txt"), 2, 1, a.py)
    os.makedirs(os.path.join(pool, "claims"))
    for uid, age in (("u00", 600), ("u01", 5)):
        c = os.path.join(pool, "claims", uid + ".json")
        json.dump({"host": "other-host.example", "pid": 1, "lane": "other", "unit": uid}, open(c, "w"))
        os.utime(c, (time.time() - age, time.time() - age))
    r = pr(["--pool", pool, "--units", U, "--stale-seconds", "180"])
    res = results(pool)
    check("another host's claim: stale by heartbeat taken over, live one left", r.returncode == 6 and
          [x["unit"] for x in res] == ["u00"],
          "rc %d (6 = units left to live launchers); ran %s; %s" % (
              r.returncode, [x["unit"] for x in res], " | ".join(x.split("] ", 1)[1] for x in r.stdout.splitlines()
                                                                 if "took over" in x or "held by" in x)))

    # 5. --time-one
    pool = os.path.join(base, "timeone")
    U = units_file(os.path.join(base, "units_timeone.txt"), 40, 2, a.py)
    r = pr(["--pool", pool, "--units", U, "--time-one"])
    lines = [x for x in r.stdout.splitlines() if x.startswith("time-one")]
    check("--time-one measures one unit and projects the wall", r.returncode == 0 and len(results(pool)) == 1 and
          len(lines) >= 2, " | ".join(lines[:2]))

    # 6. /scratch refusal on a host whose /scratch hangs (emulated here by naming this host)
    env = dict(os.environ, POOL_RUN_SCRATCH_HANG_HOSTS=host)
    r = pr(["--pool", "/scratch/bgwo/wanyu/pool_run_never_created", "--units", U, "--dry-run"], env=env)
    check("refuses /scratch work paths on a hanging host", r.returncode == 7,
          "rc %d: %s" % (r.returncode, r.stderr.strip()[:160]))

    print("%d of %d checks passed; test dir %s" % (sum(RESULTS), len(RESULTS), base))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
