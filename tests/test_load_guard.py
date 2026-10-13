#!/usr/bin/env python3
"""Unit tests of bin/load_guard.py on synthetic host snapshots (no ssh). Run: python3 tests/test_load_guard.py"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "bin"))
import load_guard as lg  # noqa: E402

LSTART = "Tue Oct 13 01:00:00 2026"


def ps(pid, args, user="wanyu", stat="R", etimes=3600, cpu="00:00:10", ppid=1, pgid=None, wchan="-"):
    return "%d %d %d %s %s %d %s %s %s %s" % (pid, ppid, pgid or pid, user, stat, etimes, cpu, wchan, LSTART, args)


def snapshot(host, cores, load, procs, dprocs=None):
    lines = ["__HOST", host, "__NPROC", str(cores), "__LOAD", "%s %s %s 3/900 1234" % tuple(load), "__ME", "wanyu", "__PS"]
    lines += procs
    lines.append("__D")
    for pid, paths in (dprocs or {}).items():
        lines.append("__DP %d" % pid)
        lines += paths
    lines.append("__END")
    return "\n".join(lines) + "\n"


SNAP = {}
lg.run_cmd = lambda host, script, timeout: (SNAP[host], None) if host in SNAP else (None, "ssh: connect refused")
FAILS = []


def expect(name, alerts, check, n=1):
    got = [a for a in alerts if a.check == check and a.level == "ALERT"]
    ok = len(got) == n
    if not ok:
        FAILS.append(name)
    print("%s %s: %s" % ("PASS" if ok else "FAIL", name, " || ".join(a.line() for a in got) or "(no %s alert)" % check))


# 1. CloudLab node with a duplicate launcher: 225 copies of one script on 128 threads, load 230
SNAP["cloudlab-x"] = snapshot("amd024.utah.cloudlab.us", 128, (230.0, 228.0, 200.0),
                              [ps(1000 + i, "python3 /users/wanyu/pal/run_unit.py --slot %d" % i) for i in range(225)])
a, _ = lg.check_host("cloudlab-x", dstate_state={})
expect("load above cores", a, "load")
expect("more copies of one script than cores", a, "copies")

# 2. login node: one heavy process (40 CPU-min over 45 min), one light long-lived loop, an ignored editor process
SNAP["dt-login03"] = snapshot("dt-login03.delta.ncsa.illinois.edu", 128, (20, 20, 20), [
    ps(2001, "python fit_all.py --big", cpu="00:40:00", etimes=2700),
    ps(2002, "python3 /u/wanyu/github/delta-helper/bin/dedup-reaper", cpu="00:20:00", etimes=200000, stat="S"),
    ps(2003, "/u/wanyu/.cursor-server/bin/node server.js", cpu="02:00:00", etimes=7000)])
a, _ = lg.check_host("dt-login03", dstate_state={})
expect("heavy process on a login node (light loop and editor not flagged)", a, "login-cpu")

# 3. the same pool run by two launchers on one host
SNAP["node-p"] = snapshot("node-p", 64, (10, 10, 10), [
    ps(3001, "python3 /u/wanyu/github/delta-helper/bin/pool-run --pool /projects/x/pool --units u.txt"),
    ps(3002, "python3 /u/wanyu/github/delta-helper/bin/pool-run --pool /projects/x/pool --units u.txt --force"),
    ps(3003, "bash -c python3 unit.py", ppid=3001),
    ps(3004, "python3 /u/wanyu/github/delta-helper/bin/pool-run --pool /projects/x/pool --status")])
a, _ = lg.check_host("node-p", dstate_state={})
expect("pool lock held twice (two launchers, one pool)", a, "pool-lock")

# 4. a process stuck in D state on /scratch: reported only once it has stayed stuck 2 min without using CPU
state = {}
def dsnap(cpu):  # noqa: E302
    return snapshot("dt-login02.delta.ncsa.illinois.edu", 128, (30, 30, 30), [
        ps(4001, "tar -xf /scratch/bgwo/wanyu/x.tar", stat="D", cpu=cpu),
        ps(4002, "python3 other.py", stat="D", user="xfan5", cpu="00:00:01"),
        ps(4003, "python3 quick.py", stat="D", cpu="00:00:01")], dprocs={4001: ["/scratch/bgwo/wanyu"], 4003: ["/u/wanyu"]})
SNAP["dt-login02"] = dsnap("00:00:05")
a, _ = lg.check_host("dt-login02", dstate_state=state)
expect("D state just seen: no alert yet", a, "scratch-io", n=0)
for k in state:
    state[k][0] -= 150                       # pretend the first sighting was 150 s ago
a, _ = lg.check_host("dt-login02", dstate_state=state)
expect("D state on /scratch for 2.5 min (own tar with /scratch evidence; other user's process on dt-login02, path "
       "unknown; own process with only a home cwd also reported on dt-login02)", a, "scratch-io", n=3)
SNAP["dt-login02"] = dsnap("00:00:30")       # the tar used CPU since: it is moving, not stuck
for k in state:
    state[k][0] -= 150
a, _ = lg.check_host("dt-login02", dstate_state=state)
expect("D state but CPU time advanced: tar not reported", [x for x in a if "4001" in x.msg], "scratch-io", n=0)

# 5. on another host, D state without /scratch evidence is not reported
SNAP["dt-login03"] = snapshot("dt-login03.delta.ncsa.illinois.edu", 128, (5, 5, 5),
                              [ps(5001, "python3 quick.py", stat="D", cpu="00:00:01")], dprocs={5001: ["/u/wanyu"]})
st = {}
lg.check_host("dt-login03", dstate_state=st)
for k in st:
    st[k][0] -= 600
a, _ = lg.check_host("dt-login03", dstate_state=st)
expect("D state off /scratch on dt-login03: not reported", a, "scratch-io", n=0)

# 6. unreachable host
a, _ = lg.check_host("cloudlab-gone", dstate_state={})
expect("unreachable host", a, "unreachable")

# 7. a healthy host and the state file round trip
SNAP["ok-host"] = snapshot("ok-host", 32, (3, 3, 3), [ps(6001, "python3 unit.py")])
with tempfile.TemporaryDirectory() as d:
    al = lg.check_hosts(["ok-host"], state_file=os.path.join(d, "s.json"))
    ok = not al and os.path.exists(os.path.join(d, "s.json"))
    FAILS.extend([] if ok else ["healthy host"])
    print("%s healthy host: no alerts, state file written" % ("PASS" if ok else "FAIL"))

print("%d failures" % len(FAILS))
sys.exit(1 if FAILS else 0)
