#!/usr/bin/env python3
"""pool-run: run independent units in parallel on one machine without oversubscribing it.

Units come from a units file (one per line: "ID<TAB>COMMAND", or just "COMMAND", whose id is then a hash of it; blank and
"#" lines skipped) or from a command template over a list of items (--template 'python fit.py {item} > {tmp}' --items FILE).
Placeholders in commands: {id} {item} {out} {tmp} {pool} {log}.

What it guarantees by default:
  - workers = min(requested, free cores // threads per worker), where free cores = cores - load, rounded down after
    allowing 0.25 core of measurement noise (an 8-core job measuring 0.2 busy keeps 8 workers). Cores are
    the CPUs this process may use (its affinity, so a Slurm allocation counts only its own cores; and any cgroup quota).
    Load is max(1-min load average, CPUs busy now) on a whole machine, and the CPUs busy now within the allocation
    otherwise. It refuses to start with no free core, and never runs more workers than free cores, unless --force.
    On a login node (dt-login*, gh-login*, cc-login*) workers are capped at --login-max (4).
  - library threads pinned per worker (OMP, MKL, OpenBLAS, NumExpr, BLIS, vecLib, Numba, loky/joblib, Rayon, Polars,
    TensorFlow; LightGBM and XGBoost follow OMP_NUM_THREADS), and POOL_RUN_THREADS set for code that passes n_jobs or
    num_threads explicitly. --bind also pins each worker to its own CPUs.
  - per-unit claims in <pool>/claims/<id>.json (host, launcher pid, unit process group, lane); a claim is stale when, on
    its own host, the launcher is gone (unit processes it left behind are killed at takeover), or, from another host, its heartbeat
    (the file's mtime, refreshed every --heartbeat seconds) is older than --stale-seconds. Stale claims are taken over
    under a short takeover lock, so copies on several hosts or lanes never run a live unit twice. Orphaned unit processes
    of a dead launcher on this host are killed before the takeover.
  - outputs temp-then-rename: with --out PATTERN, a unit writes {tmp}; on exit 0 the file is fsynced and renamed to {out}.
    A unit is finished when {out} exists (with --out) or <pool>/done/<id>.json exists. Finished units are skipped, so a
    rerun of the same command resumes.
  - a single-instance lock per pool and lane in <pool>/locks/<lane>.lock (host, pid, heartbeat); the lane is the host,
    plus the Slurm job id inside a job. A second launcher for the same lane exits, and so does one started on the same
    host outside Slurm under another lane name, unless --force.
  - one JSON line per finished unit appended to <pool>/results.jsonl (and printed), as each unit ends.
  - on dt-login02 it refuses work paths on /scratch (pool, units, outputs, results, cwd): /scratch I/O there can hang for
    minutes; run on dt-login03 or in a job.
  - nice 19 outside Slurm jobs (login nodes, lent machines); --nice N overrides.

Modes:
  pool-run --pool DIR --units FILE [--out PATTERN] [--workers N]      run (resumable)
  pool-run --pool DIR --units FILE --time-one                          run one unit, print its time and the projected wall
  pool-run --pool DIR --units FILE --dry-run                           print cores, load, workers and pending units only
  pool-run --pool DIR [--units FILE] --status                          counts, live and stale claims, locks, recent rate

Exit codes: 0 every unit finished (here or elsewhere); 1 some units failed; 2 usage error; 3 no free cores;
4 lock held by another launcher; 5 stopped by a signal; 6 units left to other live launchers (without --linger);
7 refused: /scratch work paths on a host whose /scratch I/O hangs (dt-login02 by default; --allow-scratch-here).
"""
import argparse
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time

HOST = socket.gethostname()
PID = os.getpid()
LOGIN_RE = re.compile(r"^(dt-login|gh-login|cc-login|login)\d*")
# hosts where /scratch I/O is known to hang (dt-login02: tar -x into /scratch hung 4-7 min, 2026-10-13); override with
# POOL_RUN_SCRATCH_HANG_HOSTS="host1 host2" (empty string disables)
SCRATCH_HANG_HOSTS = os.environ.get("POOL_RUN_SCRATCH_HANG_HOSTS", "dt-login02").split()
SCRATCH_PREFIXES = ("/scratch/", "/work/hdd/", "/work/nvme/")
NOISE_CORES = 0.25         # busy CPU below this is measurement noise (the launcher itself), not a lost core
PIN_VARS = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "NUMEXPR_MAX_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "LOKY_MAX_CPU_COUNT",
            "RAYON_NUM_THREADS", "POLARS_MAX_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"]


# ----------------------------------------------------------------------------------------------------------- small utils
def now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def say(msg, quiet=False):
    if not quiet:
        sys.stdout.write("[pool-run %s] %s\n" % (dt.datetime.now().strftime("%H:%M:%S"), msg))
        sys.stdout.flush()


def write_atomic(path, text):
    tmp = "%s.tmp.%s.%d" % (path, HOST, PID)
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError):
        return False


def proc_start_ticks(pid):
    """Start time of a process in clock ticks since boot (guards against pid reuse), or None."""
    try:
        with open("/proc/%d/stat" % int(pid)) as f:
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def same_process(pid, ticks):
    if not pid or not pid_alive(pid):
        return False
    t = proc_start_ticks(pid)
    return ticks is None or t is None or t == ticks


def group_members(pgid):
    """Pids of this user's processes in process group pgid (scanning /proc)."""
    out, uid = [], os.getuid()
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            if os.stat("/proc/" + d).st_uid != uid:
                continue
            with open("/proc/%s/stat" % d) as f:
                fields = f.read().rsplit(")", 1)[1].split()
            if int(fields[2]) == int(pgid):
                out.append(int(d))
        except (OSError, ValueError, IndexError):
            continue
    return out


def proc_env_has(pid, key, value):
    try:
        with open("/proc/%d/environ" % pid, "rb") as f:
            return ("%s=%s" % (key, value)).encode() in f.read().split(b"\0")
    except OSError:
        return False


def sanitize(s):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("._")
    return s[:120] or "unit"


def on_scratch(path):
    """True when path lies on Delta's /scratch (= /work/hdd). Checks the plain path first, so no /scratch I/O happens
    for paths that are visibly there; otherwise resolves symlinks (which may touch the filesystem)."""
    if not path:
        return False
    a = os.path.abspath(path).rstrip("/") + "/"
    if a.startswith(SCRATCH_PREFIXES):
        return True
    try:
        return (os.path.realpath(a).rstrip("/") + "/").startswith(SCRATCH_PREFIXES)
    except OSError:
        return False


def scratch_hang_check(args):
    """Refuse (by default) to run on a host whose /scratch I/O hangs when any work path is on /scratch."""
    short = HOST.split(".")[0]
    if short not in SCRATCH_HANG_HOSTS or args.allow_scratch_here:
        return None
    paths = {"pool": args.pool, "units": args.units, "items": args.items, "results": args.results,
             "cwd": args.cwd or os.getcwd(), "out": os.path.dirname(args.out) if args.out else None}
    bad = ["%s %s" % (k, v) for k, v in paths.items() if v and on_scratch(v)]
    if not bad:
        return None
    return ("pool-run: %s: /scratch I/O on this host can hang for minutes (2026-10-13: tar -x into /scratch hung "
            "4-7 min). Work paths on /scratch: %s. Run on dt-login03 (ssh dt-login03) or inside a job "
            "(cpu-interactive, cpu-preempt, ICC); --allow-scratch-here overrides.\n" % (short, "; ".join(bad)))


# ------------------------------------------------------------------------------------------------------------ CPU budget
def cgroup_quota_cores():
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            q, p = f.read().split()
        if q != "max":
            return float(q) / float(p)
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            q = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            p = int(f.read())
        if q > 0:
            return q / float(p)
    except (OSError, ValueError):
        pass
    return None


def cpu_times(cpus):
    """{cpu: (busy, total)} jiffies from /proc/stat for the given cpu ids."""
    res = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            parts = line.split()
            c = int(parts[0][3:])
            if c not in cpus:
                continue
            v = [int(x) for x in parts[1:]]
            idle = v[3] + (v[4] if len(v) > 4 else 0)            # idle + iowait
            total = sum(v[:8])                                    # guest time is already inside user
            res[c] = (total - idle, total)
    return res


def busy_cpus(cpus, seconds=1.0):
    a = cpu_times(cpus)
    time.sleep(seconds)
    b = cpu_times(cpus)
    busy = 0.0
    for c in a:
        if c in b:
            dt_total = b[c][1] - a[c][1]
            if dt_total > 0:
                busy += (b[c][0] - a[c][0]) / float(dt_total)
    return busy


def cpu_budget(sample_seconds=1.0):
    aff = sorted(os.sched_getaffinity(0))
    machine = os.cpu_count() or len(aff)
    cores = float(len(aff))
    q = cgroup_quota_cores()
    if q is not None:
        cores = min(cores, q)
    busy = busy_cpus(set(aff), sample_seconds)
    whole = len(aff) >= machine and q is None
    load1 = os.getloadavg()[0]
    load = max(load1, busy) if whole else busy
    free = int(math.floor(cores - load + NOISE_CORES))
    basis = ("whole machine: %d cores, 1-min load %.1f, busy now %.1f" % (cores, load1, busy) if whole else
             "allocation: %g of %d cores (affinity%s), busy now within it %.1f" % (
                 cores, machine, ", cgroup quota" if q is not None else "", busy))
    return {"cores": cores, "load": load, "free": max(free, 0), "whole_machine": whole, "basis": basis, "cpus": aff}


def plan_workers(args, budget):
    t = max(1, args.threads)
    cap = budget["free"] // t
    notes = []
    if args.mem_per_worker:
        avail_gb = mem_available_gb()
        if avail_gb is not None:
            mcap = int(avail_gb // args.mem_per_worker)
            if mcap < cap:
                notes.append("memory caps workers at %d (%.0f GB available, %.1f GB each)" % (mcap, avail_gb, args.mem_per_worker))
                cap = mcap
    if LOGIN_RE.match(HOST) and cap > args.login_max:
        notes.append("login node: capped at --login-max %d" % args.login_max)
        cap = args.login_max
    req = args.workers if args.workers else cap
    if args.force:
        w = req if args.workers else max(cap, 1)
        if w > cap:
            notes.append("--force: %d workers exceed the %d that fit" % (w, cap))
    else:
        if req > cap:
            notes.append("requested %d workers, capped at %d (free cores // threads%s)" % (
                req, cap, ", login cap" if LOGIN_RE.match(HOST) and cap == args.login_max else ""))
        w = min(req, cap)
    return w, cap, notes


def mem_available_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576.0
    except OSError:
        pass
    return None


# ----------------------------------------------------------------------------------------------------------------- units
def load_units(args):
    units, seen = [], set()

    def add(uid, cmd, item=None):
        uid = sanitize(uid)
        if uid in seen:
            sys.exit("pool-run: duplicate unit id %r" % uid)
        seen.add(uid)
        units.append({"id": uid, "cmd": cmd, "item": item if item is not None else uid})

    if args.units:
        with open(args.units) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                if "\t" in line:
                    uid, cmd = line.split("\t", 1)
                else:
                    uid, cmd = "u" + hashlib.sha1(line.encode()).hexdigest()[:10], line
                add(uid, cmd)
    elif args.template:
        items = list(args.item or [])
        if args.items:
            with open(args.items) as f:
                items += [x.strip() for x in f if x.strip() and not x.lstrip().startswith("#")]
        for it in items:
            uid = sanitize(it)
            if uid in seen:
                uid = "%s_%s" % (uid, hashlib.sha1(it.encode()).hexdigest()[:6])
            add(uid, args.template, it)
    return units


class Pool:
    def __init__(self, args):
        self.args = args
        self.dir = os.path.abspath(args.pool)
        for sub in ("claims", "done", "locks", "logs"):
            os.makedirs(os.path.join(self.dir, sub), exist_ok=True)
        self.results = os.path.abspath(args.results) if args.results else os.path.join(self.dir, "results.jsonl")
        job = os.environ.get("SLURM_JOB_ID")
        self.in_slurm = bool(job)
        self.lane = sanitize(args.lane or (HOST.split(".")[0] + ("-job" + job if job else "")))
        self.lock_path = os.path.join(self.dir, "locks", self.lane + ".lock")
        self.my_ticks = proc_start_ticks(PID)

    # paths
    def claim_path(self, uid):
        return os.path.join(self.dir, "claims", uid + ".json")

    def done_path(self, uid):
        return os.path.join(self.dir, "done", uid + ".json")

    def out_path(self, u):
        if not self.args.out:
            return None
        return os.path.abspath(self.args.out.replace("{id}", u["id"]).replace("{item}", u["item"]))

    def is_done(self, u):
        o = self.out_path(u)
        if o is not None:
            return os.path.exists(o)
        return os.path.exists(self.done_path(u["id"]))

    # staleness of a claim or lock record
    def is_stale(self, rec, path):
        try:
            age = time.time() - os.stat(path).st_mtime
        except OSError:
            return True
        if not isinstance(rec, dict):
            return age > self.args.stale_seconds            # unreadable or half-written: judge by age only
        if rec.get("host") == HOST:              # same host: stale once the launcher is gone (orphaned unit processes
            return not same_process(rec.get("pid"), rec.get("pid_ticks"))   # are killed at takeover; nobody would rename
                                                                            # their output)
        return age > self.args.stale_seconds

    def kill_orphans(self, rec, uid):
        """A dead launcher on this host may leave unit processes running; kill them (verified by their environment)."""
        if not isinstance(rec, dict) or rec.get("host") != HOST or same_process(rec.get("pid"), rec.get("pid_ticks")):
            return []
        pg = rec.get("pgid")
        if not pg:
            return []
        members = [p for p in group_members(pg) if proc_env_has(p, "POOL_RUN_UNIT", uid)]
        if members:
            try:
                os.killpg(int(pg), signal.SIGTERM)
                time.sleep(2)
                if group_members(pg):
                    os.killpg(int(pg), signal.SIGKILL)
            except OSError:
                pass
        return members

    def try_create(self, path, rec):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            json.dump(rec, f)
            f.flush()
            os.fsync(f.fileno())
        return True

    def acquire(self, path, rec, uid=None):
        """Create path exclusively; take over a stale holder under a takeover lock. Returns (ok, holder_record)."""
        if self.try_create(path, rec):
            return True, None
        cur = read_json(path)
        if not self.is_stale(cur, path):
            return False, cur
        tk = path + ".takeover"
        try:
            os.mkdir(tk)
        except FileExistsError:
            try:
                if time.time() - os.stat(tk).st_mtime > 120:
                    os.rmdir(tk)
            except OSError:
                pass
            return False, cur
        try:
            cur = read_json(path)
            if os.path.exists(path) and not self.is_stale(cur, path):
                return False, cur
            killed = self.kill_orphans(cur, uid) if uid else []
            if os.path.exists(path):
                os.replace(path, "%s.stale.%d" % (path, int(time.time())))
            ok = self.try_create(path, rec)
            if ok:
                say("took over stale %s held by %s pid %s%s" % (
                    os.path.relpath(path, self.dir), (cur or {}).get("host"), (cur or {}).get("pid"),
                    ", killed orphan pids %s" % killed if killed else ""), self.args.quiet)
            return ok, None if ok else read_json(path)
        finally:
            try:
                os.rmdir(tk)
            except OSError:
                pass

    def owns(self, path):
        rec = read_json(path)
        return isinstance(rec, dict) and rec.get("host") == HOST and rec.get("pid") == PID

    def touch(self, path):
        try:
            t = time.time()
            os.utime(path, (t, t))
        except OSError:
            pass

    def release(self, path):
        if self.owns(path):
            try:
                os.remove(path)
            except OSError:
                pass

    # lock
    def take_lock(self):
        rec = {"host": HOST, "pid": PID, "pid_ticks": self.my_ticks, "lane": self.lane, "started": now_iso(),
               "slurm_job": os.environ.get("SLURM_JOB_ID"), "argv": sys.argv}
        ok, cur = self.acquire(self.lock_path, rec)
        if not ok:
            age = time.time() - os.stat(self.lock_path).st_mtime if os.path.exists(self.lock_path) else -1
            sys.stderr.write("pool-run: another launcher holds %s: host %s pid %s, heartbeat %.0f s ago; exiting\n" % (
                self.lock_path, (cur or {}).get("host"), (cur or {}).get("pid"), age))
            sys.exit(4)
        if not self.in_slurm and not self.args.force:
            for fn in os.listdir(os.path.join(self.dir, "locks")):
                p = os.path.join(self.dir, "locks", fn)
                if not fn.endswith(".lock") or p == self.lock_path:
                    continue
                other = read_json(p)
                if isinstance(other, dict) and other.get("host") == HOST and not other.get("slurm_job") \
                        and not self.is_stale(other, p):
                    self.release(self.lock_path)
                    sys.stderr.write("pool-run: launcher pid %s already runs this pool on this host (lane %s); "
                                     "exiting (--force to run anyway)\n" % (other.get("pid"), other.get("lane")))
                    sys.exit(4)

    def append_result(self, rec):
        line = json.dumps(rec, sort_keys=True) + "\n"
        fd = os.open(self.results, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            try:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
            except (OSError, ImportError):
                pass
            os.write(fd, line.encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        if not self.args.quiet:
            sys.stdout.write("RESULT " + line)
            sys.stdout.flush()


# --------------------------------------------------------------------------------------------------------------- running
def set_pdeathsig():
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)                # PR_SET_PDEATHSIG: die with the launcher
    except OSError:
        pass


class Runner:
    def __init__(self, pool, units, workers, budget):
        self.pool, self.args = pool, pool.args
        self.units, self.workers, self.budget = units, workers, budget
        self.running = {}            # uid -> dict(proc, unit, start, slot, claim, tmp, out, attempt, log)
        self.attempts = {}
        self.failed, self.finished, self.done_ids, self.last_rec = set(), 0, set(), None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.free_slots = list(range(workers))
        self.lost_lock = False

    def env_for(self, u, slot, out, tmp, log):
        env = dict(os.environ)
        if not self.args.no_pin:
            for k in PIN_VARS:
                env[k] = str(self.args.threads)
        env.update({"POOL_RUN_UNIT": u["id"], "POOL_RUN_ITEM": u["item"], "POOL_RUN_POOL": self.pool.dir,
                    "POOL_RUN_THREADS": str(self.args.threads), "POOL_RUN_WORKER": str(slot),
                    "POOL_RUN_LANE": self.pool.lane, "POOL_RUN_LOG": log})
        if out:
            env["POOL_RUN_OUT"], env["POOL_RUN_TMP"] = out, tmp
        return env

    def cpus_for(self, slot):
        cpus, t = self.budget["cpus"], self.args.threads
        sel = cpus[slot * t:(slot + 1) * t]
        return set(sel) if sel else None

    def start(self, u, slot):
        p = self.pool
        claim = p.claim_path(u["id"])
        out = p.out_path(u)
        tmp = "%s.tmp.%s.%d" % (out, HOST.split(".")[0], PID) if out else ""
        log = os.path.join(p.dir, "logs", u["id"] + ".log")
        rec = {"host": HOST, "pid": PID, "pid_ticks": p.my_ticks, "lane": p.lane, "unit": u["id"], "started": now_iso(),
               "slurm_job": os.environ.get("SLURM_JOB_ID")}
        ok, _ = p.acquire(claim, rec, uid=u["id"])
        if not ok:
            return False
        if p.is_done(u):                                   # finished elsewhere between our check and the claim
            p.release(claim)
            return False
        if out:
            os.makedirs(os.path.dirname(out), exist_ok=True)
        cmd = u["cmd"]
        for k, v in (("{id}", u["id"]), ("{item}", u["item"]), ("{out}", out or ""), ("{tmp}", tmp),
                     ("{pool}", p.dir), ("{log}", log)):
            cmd = cmd.replace(k, v)
        cpus = self.cpus_for(slot) if self.args.bind else None

        def pre():
            os.setsid()
            set_pdeathsig()
            if cpus:
                os.sched_setaffinity(0, cpus)

        logf = open(log, "a")
        logf.write("=== %s %s pid %d slot %d attempt %d\n$ %s\n" % (now_iso(), HOST, PID, slot,
                                                                    self.attempts.get(u["id"], 0) + 1, cmd))
        logf.flush()
        proc = subprocess.Popen(["bash", "-c", cmd], stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=self.env_for(u, slot, out, tmp, log), preexec_fn=pre, close_fds=True,
                                cwd=self.args.cwd or None)
        logf.close()
        rec.update({"pgid": proc.pid, "child_pid": proc.pid, "cmd": cmd[:500]})
        try:                       # record the unit's process group in the claim, in place (same inode, never renamed)
            with open(claim, "r+") as f:
                f.seek(0)
                f.truncate()
                json.dump(rec, f)
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            pass
        self.attempts[u["id"]] = self.attempts.get(u["id"], 0) + 1
        with self.lock:
            self.running[u["id"]] = {"proc": proc, "unit": u, "start": time.time(), "start_iso": now_iso(), "slot": slot,
                                     "claim": claim, "tmp": tmp, "out": out, "log": log,
                                     "attempt": self.attempts[u["id"]]}
        return True

    def finish(self, uid, rc, timed_out=False):
        p = self.pool
        r = self.running.pop(uid)
        secs = time.time() - r["start"]
        status, note = ("done", "") if rc == 0 else ("failed", "exit %d" % rc if rc > 0 else "killed by signal %d" % -rc)
        if timed_out:
            status, note = "failed", "timeout after %d s" % self.args.unit_timeout
        if status == "done" and r["out"]:
            if os.path.exists(r["tmp"]):
                with open(r["tmp"], "rb") as f:
                    os.fsync(f.fileno())
                os.replace(r["tmp"], r["out"])
            elif not os.path.exists(r["out"]):
                status, note = "failed", "exit 0 but no output at {tmp} or {out}"
        elif r["tmp"] and os.path.exists(r["tmp"]):
            try:
                os.remove(r["tmp"])
            except OSError:
                pass
        rec = {"unit": uid, "status": status, "rc": rc, "seconds": round(secs, 3), "start": r["start_iso"],
               "end": now_iso(), "host": HOST, "lane": p.lane, "launcher_pid": PID, "slot": r["slot"],
               "attempt": r["attempt"], "out": r["out"], "log": r["log"], "slurm_job": os.environ.get("SLURM_JOB_ID")}
        if note:
            rec["note"] = note
        if status == "done":
            write_atomic(p.done_path(uid), json.dumps(rec))
            self.finished += 1
            self.done_ids.add(uid)
            self.failed.discard(uid)
        else:
            self.failed.add(uid)
        p.append_result(rec)
        self.last_rec = rec
        p.release(r["claim"])
        self.free_slots.append(r["slot"])
        return rec

    def heartbeat_loop(self):
        while not self.stop.wait(self.args.heartbeat):
            p = self.pool
            if not p.owns(p.lock_path):
                say("lost the pool lock %s; no new units will start" % p.lock_path)
                self.lost_lock = True
            p.touch(p.lock_path)
            with self.lock:
                claims = [r["claim"] for r in self.running.values()]
            for c in claims:
                if p.owns(c):
                    p.touch(c)
            if self.budget["whole_machine"] and not self.args.quiet:
                l1 = os.getloadavg()[0]
                if l1 > self.budget["cores"] * 1.05:
                    say("WARNING load average %.1f exceeds %d cores on %s" % (l1, self.budget["cores"], HOST))

    def kill_all(self, sig=signal.SIGTERM):
        with self.lock:
            procs = [r["proc"] for r in self.running.values()]
        for pr in procs:
            try:
                os.killpg(pr.pid, sig)
            except OSError:
                pass

    def run(self, max_units=None):
        """Start units as slots free up; scan the unit list when a slot frees or every 5 s (few stats on Lustre)."""
        a, p = self.args, self.pool
        hb = threading.Thread(target=self.heartbeat_loop, daemon=True)
        hb.start()
        deadline = time.time() + a.stop_new_after if a.stop_new_after else None
        todo = list(self.units)
        started, last_scan, dirty, waiting_logged = 0, 0.0, True, False
        held = []
        try:
            while True:
                for uid, r in list(self.running.items()):             # reap finished units
                    rc = r["proc"].poll()
                    if rc is None and a.unit_timeout and time.time() - r["start"] > a.unit_timeout:
                        try:
                            os.killpg(r["proc"].pid, signal.SIGKILL)
                        except OSError:
                            pass
                        rc = r["proc"].wait()
                        self.finish(uid, rc, timed_out=True)
                        dirty = True
                    elif rc is not None:
                        self.finish(uid, rc)
                        dirty = True
                may_start = not self.lost_lock and (deadline is None or time.time() < deadline) and \
                    (max_units is None or started < max_units)
                if not may_start:
                    if not self.running:
                        break
                    time.sleep(0.2)
                    continue
                if self.free_slots and (dirty or time.time() - last_scan > 5):
                    dirty, last_scan, keep, held = False, time.time(), [], []
                    for u in todo:
                        uid = u["id"]
                        if uid in self.running:
                            keep.append(u)
                            continue
                        if uid in self.done_ids or self.attempts.get(uid, 0) > a.retries:
                            continue                                   # finished, or given up in this launch
                        if not self.free_slots or (max_units is not None and started >= max_units):
                            keep.append(u)
                            continue
                        if p.is_done(u):
                            continue
                        slot = self.free_slots.pop(0)
                        if self.start(u, slot):
                            started += 1
                            keep.append(u)
                        else:
                            self.free_slots.insert(0, slot)
                            if not p.is_done(u):
                                held.append(uid)
                                keep.append(u)
                    todo = keep
                if not self.running:
                    if dirty:
                        continue                                       # rescan with every slot free before deciding
                    if not todo or (max_units is not None and started >= max_units):
                        break
                    if not a.linger:
                        say("%d units are held by other live launchers; leaving them (--linger to wait and take over "
                            "stale claims)" % len(held), a.quiet)
                        return 6
                    if not waiting_logged:
                        say("%d units held by other launchers; waiting for them or for stale claims" % len(held), a.quiet)
                        waiting_logged = True
                    time.sleep(min(30, a.stale_seconds / 3.0))
                    dirty = True
                    continue
                time.sleep(0.2)
        finally:
            self.stop.set()
        return 1 if self.failed else 0


# ----------------------------------------------------------------------------------------------------------------- modes
def status(args):
    pdir = os.path.abspath(args.pool)
    pool = Pool(args)
    units = load_units(args) if (args.units or args.template) else []
    done_ids = set(f[:-5] for f in os.listdir(os.path.join(pdir, "done")) if f.endswith(".json"))
    print("pool %s" % pdir)
    claims = [f for f in os.listdir(os.path.join(pdir, "claims")) if f.endswith(".json")]
    live, stale = [], []
    for f in claims:
        path = os.path.join(pdir, "claims", f)
        rec = read_json(path)
        (stale if pool.is_stale(rec, path) else live).append((f[:-5], rec or {}))
    for fn in sorted(os.listdir(os.path.join(pdir, "locks"))):
        if fn.endswith(".lock"):
            path = os.path.join(pdir, "locks", fn)
            rec = read_json(path) or {}
            print("lock %-40s host %s pid %s heartbeat %.0f s ago%s" % (
                fn, rec.get("host"), rec.get("pid"), time.time() - os.stat(path).st_mtime,
                " STALE" if pool.is_stale(rec, path) else ""))
    if units:
        done = sum(1 for u in units if pool.is_done(u))
        print("units %d: finished %d, running %d (live claims), stale claims %d, not finished %d" % (
            len(units), done, len(live), len(stale), len(units) - done))
    else:
        print("finished %d, live claims %d, stale claims %d" % (len(done_ids), len(live), len(stale)))
    for uid, rec in live:
        print("  running %s on %s (lane %s, launcher pid %s)" % (uid, rec.get("host"), rec.get("lane"), rec.get("pid")))
    for uid, rec in stale:
        print("  stale claim %s from %s pid %s" % (uid, rec.get("host"), rec.get("pid")))
    res = pool.results
    if os.path.exists(res):
        last = {}
        recent, cutoff = 0, time.time() - 600
        with open(res) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                last[r["unit"]] = r
        for r in last.values():
            try:
                if dt.datetime.fromisoformat(r["end"]).timestamp() > cutoff and r["status"] == "done":
                    recent += 1
            except (KeyError, ValueError):
                pass
        failed = sorted(u for u, r in last.items() if r.get("status") != "done" and u not in done_ids)
        print("results log %s: %d units, %d finished in the last 10 min, %d whose last attempt failed%s" % (
            res, len(last), recent, len(failed), (": " + " ".join(failed[:20])) if failed else ""))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n", 2)[2])
    ap.add_argument("--pool", required=True, help="pool directory (shared storage when several hosts or lanes cooperate)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--units", help="units file: 'ID<TAB>COMMAND' or 'COMMAND' per line")
    src.add_argument("--template", help="command template; {item} is each item")
    ap.add_argument("--items", help="file with one item per line (with --template)")
    ap.add_argument("--item", action="append", help="an item (repeatable; with --template)")
    ap.add_argument("--out", help="output path pattern with {id} or {item}; units write {tmp}, renamed to it on exit 0")
    ap.add_argument("--workers", type=int, default=0, help="requested workers (default: every free core)")
    ap.add_argument("--threads", type=int, default=1, help="library threads per worker (default 1)")
    ap.add_argument("--force", action="store_true", help="allow more workers than free cores, and a second launcher on this host")
    ap.add_argument("--no-pin", action="store_true", help="do not set the thread environment variables")
    ap.add_argument("--bind", action="store_true", help="also pin each worker to its own CPUs (sched_setaffinity)")
    ap.add_argument("--nice", type=int, default=None, help="nice level (default 19 outside a Slurm job, else 0)")
    ap.add_argument("--login-max", type=int, default=4, help="worker cap on a login node (default 4)")
    ap.add_argument("--mem-per-worker", type=float, default=0, help="GB per worker; caps workers by MemAvailable")
    ap.add_argument("--lane", help="lane name for the pool lock (default: host, plus -job<ID> inside Slurm)")
    ap.add_argument("--results", help="results log (default <pool>/results.jsonl)")
    ap.add_argument("--retries", type=int, default=1, help="extra attempts per failed unit within this launch (default 1)")
    ap.add_argument("--unit-timeout", type=float, default=0, help="seconds before a unit is killed (default none)")
    ap.add_argument("--stop-new-after", type=float, default=0, help="seconds after launch to stop starting new units")
    ap.add_argument("--heartbeat", type=float, default=30, help="seconds between heartbeats (default 30)")
    ap.add_argument("--stale-seconds", type=float, default=180, help="heartbeat age that makes another host's claim stale")
    ap.add_argument("--linger", action="store_true", help="when only other launchers' units remain, wait to take over stale ones")
    ap.add_argument("--cwd", help="working directory for units")
    ap.add_argument("--allow-scratch-here", action="store_true",
                    help="run on a host listed in POOL_RUN_SCRATCH_HANG_HOSTS (default dt-login02) with /scratch work paths")
    ap.add_argument("--time-one", action="store_true", help="run one unit, print its time and the projected wall time")
    ap.add_argument("--dry-run", action="store_true", help="print the plan only")
    ap.add_argument("--status", action="store_true", help="print the pool's state")
    ap.add_argument("--quiet", action="store_true", help="only the final summary on stdout")
    args = ap.parse_args(argv)
    if args.heartbeat * 3 > args.stale_seconds:
        ap.error("--stale-seconds must be at least 3 x --heartbeat")
    msg = scratch_hang_check(args)
    if msg:
        sys.stderr.write(msg)
        return 7
    if args.status:
        return status(args)
    if not (args.units or args.template):
        ap.error("give --units FILE or --template CMD with --items/--item")
    if args.template and not (args.items or args.item):
        ap.error("--template needs --items FILE or --item X")

    units = load_units(args)
    pool = Pool(args)
    nice = args.nice if args.nice is not None else (0 if pool.in_slurm else 19)
    if nice:
        try:
            os.nice(max(0, nice - os.nice(0)))
        except OSError:
            pass
    budget = cpu_budget()
    workers, cap, notes = plan_workers(args, budget)
    pending = [u for u in units if not pool.is_done(u)]
    say("%s; %s on %s (lane %s), nice %d" % (budget["basis"], "free %d" % budget["free"], HOST, pool.lane, os.nice(0)),
        args.quiet)
    for n in notes:
        say(n, args.quiet)
    say("units %d, already finished %d, to run %d; workers %d x %d thread%s; threads pinned: %s" % (
        len(units), len(units) - len(pending), len(pending), workers, args.threads, "" if args.threads == 1 else "s",
        "no" if args.no_pin else "yes (%s=%d ...)" % (PIN_VARS[0], args.threads)), args.quiet)
    if args.dry_run:
        return 0
    if workers < 1:
        sys.stderr.write("pool-run: no free cores on %s (%s); refusing to start (--force to override)\n" % (HOST, budget["basis"]))
        return 3
    if not pending:
        say("nothing to do", args.quiet)
        return 0

    pool.take_lock()
    runner = Runner(pool, pending, 1 if args.time_one else workers, budget)

    def on_signal(signum, frame):
        say("signal %d: stopping %d running units and releasing claims" % (signum, len(runner.running)))
        runner.stop.set()
        runner.kill_all(signal.SIGTERM)
        deadline = time.time() + 10
        while time.time() < deadline and any(r["proc"].poll() is None for r in runner.running.values()):
            time.sleep(0.2)
        runner.kill_all(signal.SIGKILL)
        for r in list(runner.running.values()):
            pool.release(r["claim"])
        pool.release(pool.lock_path)
        os._exit(5)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGHUP, on_signal)
    t0 = time.time()
    try:
        if args.time_one:
            rc = runner.run(max_units=1)
            done = [runner.last_rec] if runner.last_rec and runner.last_rec["status"] == "done" else []
            if not done:
                print("time-one: the unit did not finish (see the results log %s); no projection" % pool.results)
                return 1
            t = done[0]["seconds"]
            rest = len(pending) - 1
            w = max(workers, 1)
            rounds = int(math.ceil(rest / float(w))) if rest else 0
            wall = rounds * t
            print("time-one: unit %s took %.1f s (alone, %d thread%s)" % (done[0]["unit"], t, args.threads,
                                                                          "" if args.threads == 1 else "s"))
            print("time-one: %d units left, %d workers here (%s), %d rounds -> projected wall %.0f s (%.2f h) on this "
                  "lane; with k identical lanes about %.2f/k h" % (rest, w, budget["basis"], rounds, wall, wall / 3600.0,
                                                                   wall / 3600.0))
            print("time-one: measured alone; units running side by side share memory bandwidth, so treat this as a "
                  "lower bound")
            return rc
        rc = runner.run()
    finally:
        pool.release(pool.lock_path)
    left = [u["id"] for u in units if not pool.is_done(u)]
    say("finished %d units here in %.0f s; %d failed%s; %d of %d units not finished" % (
        runner.finished, time.time() - t0, len(runner.failed),
        (" (" + " ".join(sorted(runner.failed)[:10]) + ")") if runner.failed else "", len(left), len(units)))
    return rc


if __name__ == "__main__":
    sys.exit(main())
