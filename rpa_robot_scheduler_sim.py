
"""
rpa_robot_scheduler_sim.py

Discrete-event simulator for RPA robot scheduling under SLAs and priorities.

Features
--------
- Job arrivals with time windows (release, due/SLA) and priority classes.
- Multiple robots with heterogeneous speeds and context-switch penalties.
- Policies: FIFO, Priority, Shortest-Processing-Time, Earliest-Due-Date, Hybrid.
- Metrics: throughput, average flow time, SLA hit/miss, robot utilization, queue time.

Input (JSON)
------------
{
  "robots": [{"id": "botA", "speed": 1.0, "switch_penalty_s": 5.0}, ...],
  "jobs": [
    {"id": "J1", "release_s": 0, "proc_s": 30, "due_s": 120, "priority": 2},
    ...
  ],
  "policy": "HYBRID"  # FIFO|PRIO|SPT|EDD|HYBRID
}

CLI
---
python rpa_robot_scheduler_sim.py --input scenario.json --out report.json
"""
from __future__ import annotations

import argparse
import json
import heapq
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

@dataclass
class Job:
    id: str
    release_s: float
    proc_s: float
    due_s: float
    priority: int  # larger = higher priority
    start_s: Optional[float] = None
    finish_s: Optional[float] = None
    assigned_robot: Optional[str] = None

@dataclass
class Robot:
    id: str
    speed: float = 1.0
    switch_penalty_s: float = 0.0
    free_at: float = 0.0
    busy_time: float = 0.0
    last_job: Optional[str] = None

def score_job(policy: str, j: Job, now: float) -> Tuple:
    if policy == "FIFO":
        return (j.release_s, )
    if policy == "PRIO":
        return (-j.priority, j.release_s)
    if policy == "SPT":
        return (j.proc_s, j.release_s)
    if policy == "EDD":
        return (j.due_s, j.release_s)
    # HYBRID: priority first, then slack (due - now), then proc time
    slack = max(0.0, j.due_s - now)
    return (-j.priority, slack, j.proc_s, j.release_s)

def simulate(cfg: Dict) -> Dict:
    robots = [Robot(**r) for r in cfg["robots"]]
    jobs = [Job(**j) for j in cfg["jobs"]]
    policy = cfg.get("policy", "HYBRID").upper()

    # event queue: (time, type, payload)
    events: List[Tuple[float, str, Dict]] = []
    for j in jobs:
        heapq.heappush(events, (j.release_s, "ARRIVE", {"job": j}))

    now = 0.0
    ready: List[Job] = []
    completed: List[Job] = []

    def dispatch():
        nonlocal now
        # find free robots and assign best jobs
        free = [r for r in robots if r.free_at <= now]
        if not free or not ready:
            return
        # sort candidates each time deterministically
        ready.sort(key=lambda j: score_job(policy, j, now))
        for r in free:
            # avoid re-sorting for each robot; pick first feasible job
            if not ready:
                break
            j = ready.pop(0)
            # apply speed & switch penalty
            penalty = r.switch_penalty_s if r.last_job and r.last_job != j.id else 0.0
            dur = j.proc_s / max(1e-9, r.speed) + penalty
            j.start_s = now + penalty
            j.finish_s = j.start_s + (j.proc_s / max(1e-9, r.speed))
            j.assigned_robot = r.id
            r.free_at = now + dur
            r.busy_time += (j.proc_s / max(1e-9, r.speed))
            r.last_job = j.id
            heapq.heappush(events, (r.free_at, "FINISH", {"job": j, "robot": r}))

    while events:
        now, etype, payload = heapq.heappop(events)
        if etype == "ARRIVE":
            ready.append(payload["job"])
            dispatch()
        elif etype == "FINISH":
            j = payload["job"]
            completed.append(j)
            dispatch()

    # metrics
    makespan = max((j.finish_s or 0.0) for j in completed) if completed else 0.0
    throughput = len(completed)
    avg_flow = sum((j.finish_s - j.release_s) for j in completed) / throughput if throughput else 0.0
    sla_miss = sum(1 for j in completed if (j.finish_s or 0.0) > j.due_s)
    util = {r.id: (r.busy_time / max(1e-9, makespan)) for r in robots} if makespan else {r.id: 0.0 for r in robots}

    return {
        "policy": policy,
        "makespan_s": makespan,
        "throughput": throughput,
        "avg_flow_time_s": avg_flow,
        "sla_miss": sla_miss,
        "robot_utilization": util,
        "jobs": [asdict(j) for j in completed]
    }

def main():
    ap = argparse.ArgumentParser(description="RPA robot scheduler simulator")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    report = simulate(cfg)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
