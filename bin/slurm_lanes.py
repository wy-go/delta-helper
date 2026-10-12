"""Shared code for submit-everywhere and dedup-reaper: lanes, clusters, group files, job states, the duplicate rule.

A "group" is one unit of work submitted as several copies (one per lane). Its JSON file lives in <state-dir>/groups/ and records
every job id the tool submitted; the reaper only ever cancels job ids listed there, and only after checking that the job's name
is still the group id (guards against a reused id on another cluster).
"""
import datetime as dt
import json
import os
import random
import re
import socket
import subprocess
import time

DEFAULT_STATE_DIR = "/projects/bgwo/wanyu/coord_reaper"

# How each cluster is reached from a Delta login node. None = run locally.
CLUSTERS = {
    "delta": None,
    "icc": ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "campuscluster"],
    "deltaai": ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "gh-login.delta.ncsa.illinois.edu"],
}

# Partitions never to use (owner rules): too expensive.
FORBIDDEN_PARTITIONS = {"ghx4-interactive", "gpuH200x8-interactive"}

# Preset lane lists: "partition:account:cluster:hardware:kind"; kind = normal | preempt.
PRESETS = {
    "delta-cpu": ["cpu-interactive:bgwo-delta-cpu:delta:cpu:normal", "cpu-preempt:bgwo-delta-cpu:delta:cpu:preempt",
                  "cpu:bgwo-delta-cpu:delta:cpu:normal"],
    "icc-cpu": ["secondary:fyy-ic:icc:cpu:preempt", "IllinoisComputes:fyy-ic:icc:cpu:normal", "cs:cs-fyy:icc:cpu:normal"],
    "a100": ["gpuA100x4-interactive:bgvu-delta-gpu:delta:a100:normal", "gpuA100x4-preempt:bgvu-delta-gpu:delta:a100:preempt",
             "gpuA100x4:bgvu-delta-gpu:delta:a100:normal", "gpuA100x8-interactive:bgvu-delta-gpu:delta:a100:normal",
             "gpuA100x8:bgvu-delta-gpu:delta:a100:normal"],
    "a40": ["gpuA40x4-interactive:bgvu-delta-gpu:delta:a40:normal", "gpuA40x4-preempt:bgvu-delta-gpu:delta:a40:preempt",
            "gpuA40x4:bgvu-delta-gpu:delta:a40:normal"],
    "h200": ["gpuH200x8:bgvu-delta-gpu:delta:h200:normal"],
    "mi100": ["gpuMI100x8-interactive:bgvu-delta-gpu:delta:mi100:normal", "gpuMI100x8:bgvu-delta-gpu:delta:mi100:normal"],
    "gh200": ["ghx4:bgwo-dtai-gh:deltaai:gh200:normal"],
}

HARDWARE_GUESS = [(r"^gpuA100", "a100"), (r"^gpuA40", "a40"), (r"^gpuH200", "h200"), (r"^gpuMI100", "mi100"),
                  (r"^ghx4", "gh200"), (r"^eng-research-gpu", "a10"), (r"GPU", "gpu"), (r".", "cpu")]

PREEMPT_MINUTES = 5       # a preempt copy running this long or less yields to a normal copy that starts
INSURANCE_MINUTES = 30    # a preempt copy with at least this much left keeps the normal copy as insurance


def now():
    return dt.datetime.now().replace(microsecond=0)


def iso(t):
    return t.isoformat() if t else None


def parse_lane(spec):
    """partition:account[:cluster[:hardware[:normal|preempt]]] -> dict."""
    f = spec.split(":")
    if len(f) < 2 or not f[0] or not f[1]:
        raise ValueError("lane '%s' must be partition:account[:cluster[:hardware[:normal|preempt]]]" % spec)
    part, acct = f[0], f[1]
    cluster = f[2] if len(f) > 2 and f[2] else "delta"
    if cluster not in CLUSTERS:
        raise ValueError("unknown cluster '%s' in lane '%s' (known: %s)" % (cluster, spec, ", ".join(CLUSTERS)))
    hw = f[3] if len(f) > 3 and f[3] else next(h for p, h in HARDWARE_GUESS if re.search(p, part))
    if len(f) > 4 and f[4]:
        kind = f[4]
    else:
        kind = "preempt" if ("preempt" in part or (cluster == "icc" and part == "secondary")) else "normal"
    if kind not in ("normal", "preempt"):
        raise ValueError("lane kind must be normal or preempt: %s" % spec)
    if part in FORBIDDEN_PARTITIONS:
        raise ValueError("partition %s is not allowed (owner: too expensive)" % part)
    return {"partition": part, "account": acct, "cluster": cluster, "hardware": hw, "preempt": kind == "preempt"}


def run_on(cluster, argv, stdin=None, timeout=60):
    """Run argv on a cluster (locally for delta, through ssh otherwise). Returns (rc, stdout, stderr)."""
    prefix = CLUSTERS[cluster]
    if prefix is None:
        cmd = argv
    else:
        cmd = prefix + [" ".join(shell_quote(a) for a in argv)]
    try:
        p = subprocess.run(cmd, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                           timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout after %ss" % timeout
    except Exception as e:
        return 125, "", repr(e)


def shell_quote(s):
    if re.match(r"^[A-Za-z0-9_@%+=:,./~-]+$", s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


# ---------------------------------------------------------------- files

def write_json_atomic(path, obj):
    tmp = "%s.tmp.%s.%d" % (path, socket.gethostname(), os.getpid())
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def groups_dir(state_dir):
    d = os.path.join(state_dir, "groups")
    os.makedirs(os.path.join(d, "closed"), exist_ok=True)
    return d


def new_group_id(unit):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", unit)[:60]
    return "%s-%s-%04x" % (safe, now().strftime("%Y%m%d-%H%M%S"), random.randrange(16 ** 4))


def log_line(state_dir, msg):
    os.makedirs(state_dir, exist_ok=True)
    line = "%s %s[%d] %s\n" % (now().isoformat(sep=" "), socket.gethostname().split(".")[0], os.getpid(), msg)
    with open(os.path.join(state_dir, "reaper.log"), "a") as f:
        f.write(line)
    return line


# ---------------------------------------------------------------- job states

def parse_time(s):
    if not s or s in ("Unknown", "None", "N/A"):
        return None
    try:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def query_states(cluster, jobids):
    """Return {jobid: {state, start, end, limit_min, name, exit}} via sacct; None if the cluster cannot be reached."""
    if not jobids:
        return {}
    rc, out, err = run_on(cluster, ["sacct", "-n", "-P", "-X", "-j", ",".join(jobids), "-o",
                                    "JobIDRaw,JobName%200,State,Start,End,TimelimitRaw,ExitCode"])
    if rc != 0:
        return None
    res = {}
    for line in out.splitlines():
        f = line.split("|")
        if len(f) < 7:
            continue
        jid, name, state, start, end, limit, code = f[:7]
        lim = int(limit) if limit.isdigit() else None
        res[jid] = {"state": state.split()[0], "start": parse_time(start), "end": parse_time(end), "limit_min": lim,
                    "name": name, "exit": code}
    return res


LIVE_PENDING = {"PENDING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "CONFIGURING", "SUSPENDED", "RESV_DEL_HOLD"}
LIVE_RUNNING = {"RUNNING", "COMPLETING", "RESIZING", "SIGNALING", "STAGE_OUT"}


def simple_state(s):
    """P (pending), R (running), C (completed with exit 0), X (ended otherwise), ? (unknown)."""
    if s is None:
        return "?"
    st = s["state"]
    if st in LIVE_PENDING:
        return "P"
    if st in LIVE_RUNNING:
        return "R"
    if st == "COMPLETED" and s.get("exit", "0:0") == "0:0":
        return "C"
    return "X"


# ---------------------------------------------------------------- the duplicate rule (pure function, unit-tested)

def decide(group, states, t_now, done):
    """Decide which copies to cancel.

    group:  the group dict (copies with jobid, preempt; optional expected_minutes; optional insurance pair)
    states: {jobid: {"s": P|R|C|X|?, "start": datetime|None, "limit_min": int|None}}
    done:   True when the unit's output exists (or, with no done file, a copy completed with exit 0)
    Returns (cancels, updates): cancels = [(jobid, reason)], updates = dict of group fields to set.
    Copies in an unknown state (cluster unreachable) are never cancelled, and their presence blocks no decision.
    """
    copies = group["copies"]
    st = lambda c: states.get(c["jobid"], {}).get("s", "?")
    start = lambda c: states.get(c["jobid"], {}).get("start") or t_now
    cancels, updates = [], {}
    live = [c for c in copies if st(c) in ("P", "R")]
    if done:
        for c in live:
            cancels.append((c["jobid"], "the unit is finished"))
        if not [c for c in copies if st(c) == "?"]:
            updates["status"] = "done"
        return cancels, updates
    running = sorted([c for c in copies if st(c) == "R"], key=start)
    pending = [c for c in copies if st(c) == "P"]
    normals = [c for c in running if not c["preempt"]]
    preempts = [c for c in running if c["preempt"]]
    if normals:
        keep_n = normals[0]
        for c in normals[1:]:
            cancels.append((c["jobid"], "normal copy %s started first" % keep_n["jobid"]))
        for c in pending:
            cancels.append((c["jobid"], "copy %s is running on a normal lane" % keep_n["jobid"]))
        if preempts:
            keep_p = preempts[0]
            for c in preempts[1:]:
                cancels.append((c["jobid"], "preempt copy %s started first" % keep_p["jobid"]))
            ins = group.get("insurance")
            if ins and ins.get("normal") == keep_n["jobid"] and ins.get("preempt") == keep_p["jobid"]:
                pass  # decided earlier: keep both until the preempt copy finishes
            else:
                age = (start(keep_n) - start(keep_p)).total_seconds() / 60.0
                if age <= PREEMPT_MINUTES:
                    cancels.append((keep_p["jobid"], "preempt copy had run %.1f min (<= %d) when normal copy %s started; "
                                    "the normal copy does the work" % (max(age, 0), PREEMPT_MINUTES, keep_n["jobid"])))
                else:
                    elapsed = (t_now - start(keep_p)).total_seconds() / 60.0
                    total = group.get("expected_minutes") or states.get(keep_p["jobid"], {}).get("limit_min")
                    remaining = (total - elapsed) if total else None
                    if remaining is None or remaining >= INSURANCE_MINUTES:
                        updates["insurance"] = {"normal": keep_n["jobid"], "preempt": keep_p["jobid"],
                                                "decided": iso(t_now), "remaining_min": remaining}
                    else:
                        cancels.append((keep_n["jobid"], "preempt copy %s ran %.1f min and needs about %.1f more (< %d); "
                                        "it keeps the unit" % (keep_p["jobid"], elapsed, remaining, INSURANCE_MINUTES)))
    elif preempts:
        keep_p = preempts[0]
        for c in preempts[1:]:
            cancels.append((c["jobid"], "preempt copy %s started first" % keep_p["jobid"]))
        for c in pending:
            if c["preempt"]:
                cancels.append((c["jobid"], "preempt copy %s is running" % keep_p["jobid"]))
        # pending normal copies stay: when one starts, the rule above decides which copy keeps the unit
    elif not pending and not [c for c in copies if st(c) == "?"]:
        updates["status"] = "ended"  # every copy ended without the unit's output
    return cancels, updates
