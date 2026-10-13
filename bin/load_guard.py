#!/usr/bin/env python3
"""load-guard: one-line alerts when a machine is oversubscribed or misused, for watchdog loops.

Checks, for each host (`local`, or an ssh alias such as dt-login03, campuscluster, cloudlab-*):
  load         the load average (--load-window 1, 5 or 15 minutes; default 5) exceeds the host's core count;
  copies       more than --max-copies processes of one script run for the user (default: the host's core count);
               the script is the first non-option argument of an interpreter (python, bash, ...), else the program;
  login-cpu    on a login node (dt-login*, gh-login*, cc-login*), one process of the user has used more than
               --login-cpu-minutes CPU minutes (default 10) at an average of at least --min-util of a core (default
               0.2, so long-lived light loops and editors are not flagged);
  pool-lock    the same pool-run lock is held twice: two pool-run launchers for one pool and lane on a host, two
               launchers for one pool on a host outside Slurm, or a launcher whose pool lock names another holder;
               stale locks (holder gone) are reported as notes;
  scratch-io   a process (any user) in uninterruptible I/O wait (state D) on /scratch (= /work/hdd) for more than
               --dstate-minutes (default 2), judged from its arguments, working directory and open files (own
               processes) or a Lustre kernel wait channel; a process counts as stuck while it stays in D without using
               CPU time between checks. D-state first sightings are kept in a state file
               (--state-file, default ~/.cache/load-guard/dstate.json) so successive watchdog runs measure duration.
               On --dstate-any-hosts (default dt-login02, whose /scratch I/O hung for 4-7 min on 2026-10-13) every
               stuck D-state process is reported, with "unknown path" when no evidence is readable.
  unreachable  the host could not be reached.

Output: "ALERT <host> <check>: <message>" per problem, and (without --alerts-only) "OK <host>: ..." per healthy host.
Exit code 0 when there is no alert, 1 otherwise. --watch SECONDS loops forever, printing alerts with a time stamp when
they first appear and again every --repeat-minutes while they persist (for a Monitor or watchdog).

As a library: `from load_guard import check_hosts; for a in check_hosts(["local", "dt-login03"]): print(a.line())`.
"""
import argparse
import datetime as dt
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time

LOGIN_RE = re.compile(r"^(dt-login|gh-login|cc-login|login)\d*")
INTERPRETERS = re.compile(r"^(python[0-9.]*|bash|sh|zsh|dash|Rscript|R|perl|julia|node|ruby|java|pypy[0-9.]*)$")
SCRATCH_RE = re.compile(r"(^|[\s=:'\"])/(scratch|work/hdd|work/nvme)/")
SCRATCH_FS_NAME = "dltawork"          # Lustre file system behind Delta /scratch (= /work/hdd); kernel flush threads carry it
# hosts where /scratch I/O is known to hang: every stuck D-state process there is reported, with "path unknown" when no
# evidence is readable (other users' /proc entries are private); dt-login02, 2026-10-13
DSTATE_ANY_HOSTS = "dt-login02"
LUSTRE_WCHAN = re.compile(r"^(ll_|lov_|osc_|mdc_|cl_|ptlrpc|ldlm|lustre|lmv_|vvp_|ksocknal|lnet)")
DEFAULT_IGNORE = r"cursor-server|vscode-server|\.codex/|/claude( |$)|^ssh: |sshd|^-?bash$|tmux|systemd"

GATHER = r"""
echo __HOST; hostname
echo __NPROC; nproc --all
echo __LOAD; cat /proc/loadavg
echo __ME; id -un
echo __PS
ps -eo pid=,ppid=,pgid=,user:32=,stat=,etimes=,time=,wchan:40=,lstart=,args= -ww 2>/dev/null
echo __D
for p in $(ps -eo pid=,stat= | awk '$2 ~ /^D/ {print $1}'); do
  echo "__DP $p"
  readlink /proc/$p/cwd 2>/dev/null
  for f in /proc/$p/fd/*; do readlink "$f" 2>/dev/null; done | grep -E '^/(scratch|work)/' | head -5
done
echo __END
"""


class Alert(object):
    def __init__(self, host, check, msg, key=None, level="ALERT"):
        self.host, self.check, self.msg, self.level = host, check, msg, level
        self.key = key or "%s|%s|%s" % (host, check, msg)

    def line(self):
        return "%s %s %s: %s" % (self.level, self.host, self.check, self.msg)

    def __repr__(self):
        return self.line()


def run_cmd(host, script, timeout):
    if host in ("local", "localhost", socket.gethostname(), socket.gethostname().split(".")[0]):
        argv = ["bash", "-c", script]
    else:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, "bash -c %s" % shlex.quote(script)]
    try:
        r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
                           universal_newlines=True)
    except subprocess.TimeoutExpired:
        return None, "timed out after %d s" % timeout
    except OSError as e:
        return None, str(e)
    if r.returncode != 0 and "__END" not in r.stdout:
        return None, (r.stderr.strip().splitlines() or ["exit %d" % r.returncode])[-1][:200]
    return r.stdout, None


def cpu_minutes(t):
    """ps 'time' ([DD-]HH:MM:SS) -> minutes."""
    days = 0
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d)
    parts = [int(x) for x in t.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return days * 1440 + h * 60 + m + s / 60.0


def parse(out):
    sec, data = None, {"ps": [], "d": {}}
    cur_d = None
    for line in out.splitlines():
        if line.startswith("__") and line.split()[0] in ("__HOST", "__NPROC", "__LOAD", "__ME", "__PS", "__D", "__END"):
            sec = line.split()[0]
            continue
        if line.startswith("__DP "):
            cur_d = int(line.split()[1])
            data["d"][cur_d] = []
            sec = "__DP"
            continue
        if sec == "__HOST":
            data["host"] = line.strip()
        elif sec == "__NPROC":
            data["cores"] = int(line.strip())
        elif sec == "__LOAD":
            data["load"] = [float(x) for x in line.split()[:3]]
        elif sec == "__ME":
            data["me"] = line.strip()
        elif sec == "__PS":
            f = line.split(None, 13)
            if len(f) < 13:
                continue
            # pid ppid pgid user stat etimes time wchan lstart(5 fields) args
            args = f[13] if len(f) > 13 else ""
            data["ps"].append({"pid": int(f[0]), "ppid": int(f[1]), "pgid": int(f[2]), "user": f[3], "stat": f[4],
                               "etimes": int(f[5]), "time": f[6], "wchan": f[7], "lstart": " ".join(f[8:13]),
                               "args": args})
        elif sec == "__DP" and cur_d is not None and line.strip():
            data["d"][cur_d].append(line.strip())
    return data


def script_key(args):
    toks = args.split()
    if not toks or args.startswith("["):
        return None
    exe = os.path.basename(toks[0])
    if not INTERPRETERS.match(exe):
        return exe
    i = 1
    while i < len(toks):
        t = toks[i]
        if t in ("-m",) and i + 1 < len(toks):
            return "%s -m %s" % (exe, toks[i + 1])
        if t in ("-c", "-e"):
            return "%s %s" % (exe, t)
        if t.startswith("-"):
            i += 1
            continue
        return t
    return exe


def arg_value(toks, flag):
    for i, t in enumerate(toks):
        if t == flag and i + 1 < len(toks):
            return toks[i + 1]
        if t.startswith(flag + "="):
            return t.split("=", 1)[1]
    return None


def is_pool_launcher(args):
    toks = args.split()
    names = [os.path.basename(t) for t in toks[:3]]
    if not any(n in ("pool-run", "pool_run.py") for n in names):
        return None
    if "--status" in toks or "--dry-run" in toks:
        return None
    return toks


def read_locks(host, pools, timeout):
    """{pool: [lock records]} read on the host (shared storage is read locally when the host is local)."""
    if not pools:
        return {}
    script = "".join("for f in %s/locks/*.lock; do [ -f \"$f\" ] && { echo \"__F %s $f $(( $(date +%%s) - $(stat -c %%Y \"$f\") ))\"; "
                     "cat \"$f\"; echo; }; done; " % (shlex.quote(p), shlex.quote(p)) for p in pools) + "echo __END"
    out, err = run_cmd(host, script, timeout)
    res = {}
    if out is None:
        return res
    cur = None
    for line in out.splitlines():
        if line.startswith("__F "):
            _, pool, path, age = line.split(" ", 3)
            cur = {"pool": pool, "path": path, "age": float(age)}
            res.setdefault(pool, []).append(cur)
        elif cur is not None and line.strip().startswith("{"):
            try:
                cur.update(json.loads(line))
            except ValueError:
                pass
    return res


def load_state(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(path, state):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = "%s.tmp.%d" % (path, os.getpid())
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, path)
    except OSError:
        pass


def check_host(host, max_copies=None, load_window=5, login_cpu_minutes=10.0, min_util=0.2, user=None,
               all_users=False, ignore=DEFAULT_IGNORE, pools=(), dstate_minutes=2.0, dstate_state=None,
               dstate_any_hosts=DSTATE_ANY_HOSTS, timeout=60, verbose=True):
    """Return (alerts, summary line) for one host. dstate_state is a dict updated in place (first-seen D times)."""
    out, err = run_cmd(host, GATHER, timeout)
    if out is None:
        return [Alert(host, "unreachable", err, key="%s|unreachable" % host)], None
    d = parse(out)
    name = d.get("host", host)
    short = name.split(".")[0]
    label = host if host not in ("local", "localhost") else short
    cores = d.get("cores", 0)
    load = d.get("load", [0, 0, 0])
    me = user or d.get("me")
    alerts = []
    ign = re.compile(ignore) if ignore else None
    mine = [p for p in d["ps"] if all_users or p["user"] == me]

    # load
    li = {1: 0, 5: 1, 15: 2}[load_window]
    if cores and load[li] > cores:
        alerts.append(Alert(label, "load", "%d-min load average %.1f exceeds %d cores (1/5/15 min: %.1f %.1f %.1f)" % (
            load_window, load[li], cores, load[0], load[1], load[2]), key="%s|load" % label))

    # copies of one script
    limit = max_copies if max_copies else cores
    counts = {}
    for p in mine:
        k = script_key(p["args"])
        if k:
            counts.setdefault((p["user"], k), []).append(p)
    for (u, k), ps in sorted(counts.items(), key=lambda kv: -len(kv[1])):
        if limit and len(ps) > limit:
            running = sum(1 for p in ps if p["stat"].startswith("R"))
            alerts.append(Alert(label, "copies", "%d copies of %s by %s (%d running) exceed %d (cores %d); pgids %s" % (
                len(ps), k, u, running, limit, cores, " ".join(str(x) for x in sorted(set(p["pgid"] for p in ps))[:8])),
                key="%s|copies|%s|%s" % (label, u, k)))

    # heavy process on a login node
    if LOGIN_RE.match(short):
        for p in mine:
            if ign and ign.search(p["args"]):
                continue
            cm = cpu_minutes(p["time"])
            util = cm * 60.0 / max(p["etimes"], 1)
            if cm > login_cpu_minutes and util >= min_util:
                alerts.append(Alert(label, "login-cpu", "pid %d (%s) used %.0f CPU-min over %.0f min (%.0f%% of a core) "
                                    "on a login node: %s" % (p["pid"], p["user"], cm, p["etimes"] / 60.0, 100 * util,
                                                             p["args"][:120]),
                                    key="%s|login-cpu|%d" % (label, p["pid"])))

    # pool-run locks
    launchers = {}
    for p in d["ps"]:
        if p["user"] != me:
            continue
        toks = is_pool_launcher(p["args"])
        if toks is None:
            continue
        pool = arg_value(toks, "--pool")
        if pool:
            launchers.setdefault(pool, []).append((p, arg_value(toks, "--lane")))
    for pool, ls in launchers.items():
        by_lane = {}
        for p, lane in ls:
            by_lane.setdefault(lane, []).append(p)
        for lane, ps in by_lane.items():
            # children of a launcher (between fork and exec) share its args; keep top-level launchers only
            pids = set(p["pid"] for p in ps)
            tops = [p for p in ps if p["ppid"] not in pids]
            if len(tops) > 1:
                alerts.append(Alert(label, "pool-lock", "pool %s lane %s is run by %d launchers on this host: pids %s" % (
                    pool, lane or "(default)", len(tops), " ".join(str(p["pid"]) for p in tops)),
                    key="%s|pool-lock|%s|%s" % (label, pool, lane)))
    all_pools = sorted(set(list(launchers) + list(pools)))
    locks = read_locks(host, all_pools, timeout) if all_pools else {}
    live_pids = set(p["pid"] for p in d["ps"])
    for pool in all_pools:
        recs = locks.get(pool, [])
        here = [r for r in recs if r.get("host") == name]
        live_here = [r for r in here if r.get("pid") in live_pids]
        for r in here:
            if r.get("pid") not in live_pids:
                alerts.append(Alert(label, "pool-lock", "stale lock %s: holder pid %s is gone (the next launcher takes "
                                    "it over)" % (r["path"], r.get("pid")), level="NOTE",
                                    key="%s|stale|%s" % (label, r["path"])))
        nonslurm = [r for r in live_here if not r.get("slurm_job")]
        if len(nonslurm) > 1:
            alerts.append(Alert(label, "pool-lock", "pool %s is locked twice on this host outside Slurm: %s" % (
                pool, ", ".join("lane %s pid %s" % (r.get("lane"), r.get("pid")) for r in nonslurm)),
                key="%s|pool-lock2|%s" % (label, pool)))
        holders = set(r.get("pid") for r in live_here)
        for p, lane in launchers.get(pool, []):
            if p["etimes"] > 30 and p["pid"] not in holders and p["ppid"] not in holders and recs:
                alerts.append(Alert(label, "pool-lock", "launcher pid %d runs pool %s without holding a lock (holders: %s)" % (
                    p["pid"], pool, ", ".join("%s pid %s" % (r.get("host", "?").split(".")[0], r.get("pid")) for r in recs)),
                    key="%s|pool-nolock|%s|%d" % (label, pool, p["pid"])))

    # processes in uninterruptible I/O wait on /scratch
    nowt = time.time()
    seen_now = set()
    if dstate_state is None:
        dstate_state = {}
    for p in d["ps"]:
        if not p["stat"].startswith("D"):
            continue
        paths = d["d"].get(p["pid"], [])
        evidence = []
        if SCRATCH_RE.search(" " + p["args"]):
            evidence.append("arguments")
        if any(x.startswith(("/scratch/", "/work/hdd/", "/work/nvme/")) or x in ("/scratch", "/work/hdd") for x in paths):
            evidence.append("open files or cwd")
        if SCRATCH_FS_NAME in p["args"]:
            evidence.append("kernel flush thread of %s" % SCRATCH_FS_NAME)
        lustre = bool(LUSTRE_WCHAN.match(p["wchan"] or ""))
        any_host = short in [h.strip() for h in (dstate_any_hosts.split(",") if isinstance(dstate_any_hosts, str)
                                                 else dstate_any_hosts)]
        if not evidence and not lustre and not any_host:
            continue
        key = "%s|%d|%s" % (name, p["pid"], p["lstart"])
        seen_now.add(key)
        cpu = cpu_minutes(p["time"])
        rec = dstate_state.get(key)
        if not isinstance(rec, list) or cpu - rec[1] > 2.0 / 60:      # first sighting, or it used CPU since: not stuck
            rec = [nowt, cpu]
        dstate_state[key] = rec
        dur = nowt - rec[0]
        if dur >= dstate_minutes * 60:
            where = ("/scratch (%s)" % ", ".join(evidence) if evidence else
                     "Lustre (wait channel %s; path unknown)" % p["wchan"] if lustre else
                     ("a path not shown on /scratch (cwd and open files: %s; %s's /scratch I/O is known to hang)" % (
                         ", ".join(paths[:3]), short)) if paths else
                     "an unknown path (no readable evidence; %s's /scratch I/O is known to hang)" % short)
            alerts.append(Alert(label, "scratch-io", "pid %d (%s) in uninterruptible I/O wait on %s for at least %.1f min, "
                                "wait channel %s: %s" % (p["pid"], p["user"], where, dur / 60.0, p["wchan"], p["args"][:120]),
                                key="%s|scratch-io|%d" % (label, p["pid"])))
    for k in [k for k in dstate_state if k.startswith(name + "|") and k not in seen_now]:
        del dstate_state[k]

    nmine = len([p for p in d["ps"] if p["user"] == me])
    top = max(counts.items(), key=lambda kv: len(kv[1])) if counts else None
    summary = "OK %s: load %.1f/%.1f/%.1f on %d cores, %d processes of %s%s, %d processes in D state" % (
        label, load[0], load[1], load[2], cores, nmine, me,
        (", most copies %d of %s" % (len(top[1]), top[0][1])) if top else "",
        sum(1 for p in d["ps"] if p["stat"].startswith("D")))
    return alerts, summary


def check_hosts(hosts, state_file=None, **kw):
    """Run every check on every host; returns a list of Alert (NOTE-level items included)."""
    state_file = state_file or os.path.expanduser("~/.cache/load-guard/dstate.json")
    state = load_state(state_file)
    alerts = []
    for h in hosts:
        a, _ = check_host(h, dstate_state=state, **kw)
        alerts += a
    save_state(state_file, state)
    return alerts


def expand_hosts(spec):
    hosts = []
    for h in spec.split(","):
        h = h.strip()
        if not h:
            continue
        if "*" in h:
            try:
                with open(os.path.expanduser("~/.ssh/config")) as f:
                    names = [w for line in f if line.strip().lower().startswith("host ")
                             for w in line.split()[1:] if "*" not in w]
            except OSError:
                names = []
            pat = re.compile("^" + re.escape(h).replace(r"\*", ".*") + "$")
            hosts += [n for n in names if pat.match(n)]
        else:
            hosts.append(h)
    return hosts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n", 2)[2])
    ap.add_argument("--hosts", default="local", help="comma list: local, ssh aliases, or patterns like 'cloudlab-*' "
                    "matched against ~/.ssh/config (default local)")
    ap.add_argument("--max-copies", type=int, default=0, help="copies of one script allowed (default: core count)")
    ap.add_argument("--load-window", type=int, choices=(1, 5, 15), default=5)
    ap.add_argument("--login-cpu-minutes", type=float, default=10.0)
    ap.add_argument("--min-util", type=float, default=0.2, help="average core use a login process needs to be flagged")
    ap.add_argument("--user", help="user whose processes are checked (default: the login user on each host)")
    ap.add_argument("--all-users", action="store_true", help="copies and login-cpu checks over every user")
    ap.add_argument("--ignore", default=DEFAULT_IGNORE, help="regex of process arguments the login-cpu check skips")
    ap.add_argument("--pool", action="append", default=[], help="pool-run directory whose locks are checked (repeatable); "
                    "pools of running launchers are found automatically")
    ap.add_argument("--dstate-minutes", type=float, default=2.0)
    ap.add_argument("--dstate-any-hosts", default=DSTATE_ANY_HOSTS,
                    help="hosts (short names, comma list) where every stuck D-state process is reported (default dt-login02)")
    ap.add_argument("--state-file", default=os.path.expanduser("~/.cache/load-guard/dstate.json"))
    ap.add_argument("--timeout", type=int, default=60, help="seconds per host")
    ap.add_argument("--alerts-only", action="store_true", help="print alerts only (no OK or NOTE lines)")
    ap.add_argument("--watch", type=float, default=0, help="loop every SECONDS, printing new alerts with a time stamp")
    ap.add_argument("--repeat-minutes", type=float, default=30, help="with --watch, repeat a persisting alert this often")
    a = ap.parse_args(argv)
    hosts = expand_hosts(a.hosts)
    kw = dict(max_copies=a.max_copies, load_window=a.load_window, login_cpu_minutes=a.login_cpu_minutes,
              min_util=a.min_util, user=a.user, all_users=a.all_users, ignore=a.ignore, pools=a.pool,
              dstate_minutes=a.dstate_minutes, dstate_any_hosts=a.dstate_any_hosts.split(","), timeout=a.timeout)
    last_shown = {}
    while True:
        state = load_state(a.state_file)
        seen = set()
        any_alert = False
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for h in hosts:
            alerts, summary = check_host(h, dstate_state=state, **kw)
            for al in alerts:
                seen.add(al.key)
                if al.level == "ALERT":
                    any_alert = True
                elif a.alerts_only:
                    continue
                if a.watch:
                    t = last_shown.get(al.key)
                    if t is not None and time.time() - t < a.repeat_minutes * 60:
                        continue
                    last_shown[al.key] = time.time()
                    print("%s %s" % (stamp, al.line()))
                else:
                    print(al.line())
            if summary and not a.alerts_only and not a.watch:
                print(summary)
        save_state(a.state_file, state)
        for k in [k for k in last_shown if k not in seen]:       # resolved: show again if it comes back
            del last_shown[k]
        sys.stdout.flush()
        if not a.watch:
            return 1 if any_alert else 0
        time.sleep(a.watch)


if __name__ == "__main__":
    sys.exit(main())
