#!/usr/bin/env python3
"""Unit tests of the duplicate rule in bin/slurm_lanes.decide. Run: python3 tests/test_dedup_decide.py"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "bin"))
import slurm_lanes as sl  # noqa: E402

T = dt.datetime(2026, 10, 11, 12, 0, 0)


def ago(m):
    return T - dt.timedelta(minutes=m)


def grp(*kinds, expected=None, insurance=None):
    g = {"group": "g", "copies": [{"jobid": str(i + 1), "preempt": k == "preempt"} for i, k in enumerate(kinds)]}
    if expected:
        g["expected_minutes"] = expected
    if insurance:
        g["insurance"] = insurance
    return g


def s(state, start_ago=None, limit=60):
    return {"s": state, "start": ago(start_ago) if start_ago is not None else None, "limit_min": limit}


CASES = []


def case(name, g, states, done, want_cancel, want_updates=None):
    CASES.append((name, g, states, done, set(want_cancel), want_updates or {}))


# copies: 1 = normal (interactive), 2 = preempt, 3 = normal (regular)
case("nothing runs: keep all pending", grp("normal", "preempt", "normal"), {"1": s("P"), "2": s("P"), "3": s("P")}, False, [])
case("normal copy runs: cancel every pending copy", grp("normal", "preempt", "normal"),
     {"1": s("R", 1), "2": s("P"), "3": s("P")}, False, ["2", "3"])
case("two normal copies run: keep the earlier one", grp("normal", "preempt", "normal"),
     {"1": s("R", 1), "2": s("P"), "3": s("R", 3)}, False, ["1", "2"])
case("only preempt runs: pending normal copies stay", grp("normal", "preempt", "normal"),
     {"1": s("P"), "2": s("R", 2), "3": s("P")}, False, [])
case("two preempt copies: keep the earlier", grp("preempt", "preempt", "normal"),
     {"1": s("R", 1), "2": s("R", 4), "3": s("P")}, False, ["1"])
case("pending preempt copy cancelled once a preempt copy runs", grp("preempt", "preempt", "normal"),
     {"1": s("R", 1), "2": s("P"), "3": s("P")}, False, ["2"])
case("preempt ran <= 5 min when normal started: cancel preempt", grp("normal", "preempt"),
     {"1": s("R", 1), "2": s("R", 5.5)}, False, ["2"])
case("normal started first, preempt later: cancel preempt", grp("normal", "preempt"),
     {"1": s("R", 10), "2": s("R", 2)}, False, ["2"])
case("preempt ran > 5 min, >= 30 min remain (time limit 60): keep both as insurance", grp("normal", "preempt"),
     {"1": s("R", 0), "2": s("R", 20)}, False, [], {"insurance": {"normal": "1", "preempt": "2"}})
case("preempt ran > 5 min, < 30 min remain (expected 40): cancel the normal copy", grp("normal", "preempt", expected=40),
     {"1": s("R", 0), "2": s("R", 20)}, False, ["1"])
case("insurance already decided: keep both on later passes", grp("normal", "preempt",
                                                                  insurance={"normal": "1", "preempt": "2"}),
     {"1": s("R", 30), "2": s("R", 50)}, False, [])
case("preempt copy requeued while normal runs: cancel the requeued copy", grp("normal", "preempt"),
     {"1": s("R", 30), "2": s("P")}, False, ["2"])
case("unit finished: cancel every live copy, close", grp("normal", "preempt", "normal"),
     {"1": s("C"), "2": s("R", 3), "3": s("P")}, True, ["2", "3"], {"status": "done"})
case("unit finished but a cluster unreachable: cancel what is known, stay open", grp("normal", "preempt"),
     {"1": s("C"), "2": s("?")}, True, [], {})
case("every copy ended without the output: close as ended", grp("normal", "preempt"),
     {"1": s("X"), "2": s("X")}, False, [], {"status": "ended"})
case("unknown state is never cancelled", grp("normal", "preempt", "normal"),
     {"1": s("R", 1), "2": s("?"), "3": s("P")}, False, ["3"])

fails = 0
for name, g, states, done, want, want_up in CASES:
    cancels, updates = sl.decide(g, states, T, done)
    got = set(j for j, _ in cancels)
    ok = got == want
    for k, v in want_up.items():
        if k == "insurance":
            ok = ok and updates.get(k, {}).get("normal") == v["normal"] and updates[k].get("preempt") == v["preempt"]
        else:
            ok = ok and updates.get(k) == v
    if not want_up:
        ok = ok and "status" not in updates and "insurance" not in updates
    fails += not ok
    print("%-4s %s -> cancel %s %s" % ("ok" if ok else "FAIL", name, sorted(got), {k: v for k, v in updates.items()
                                                                                   if k != "insurance"} or ""))
print("%d cases, %d failures" % (len(CASES), fails))
sys.exit(1 if fails else 0)
