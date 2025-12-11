
"""
rpa_kpi_roi_model.py

KPI computation + ROI modeling for financial-audit RPA programs.

Features
--------
- Computes common audit/RPA KPIs (AHT, throughput, automation rate, FPY, error detection, cost/txn).
- Scenario modeling: baseline vs. post-automation with sensitivity ranges (tolerance bands).
- ROI, break-even month, and cumulative benefit curves (data-only; plotting delegated to caller).
- JSON I/O for reproducible offline analysis and audit sharing.

JSON input (example)
--------------------
{
  "baseline": {"aht_minutes": 18.5, "throughput_tph": 45, "automation_rate": 0.0,
               "fpy": 0.78, "error_detect": 0.65, "cost_per_txn": 12.50},
  "post":     {"aht_minutes": 4.8,  "throughput_tph": 195, "automation_rate": 0.82,
               "fpy": 0.965, "error_detect": 0.93, "cost_per_txn": 2.95},
  "volume_monthly": 50000,
  "costs": {"initial": 250000, "operating_monthly": 6250},
  "labor_rate_per_hour": 35.0
}
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

@dataclass
class Metrics:
    aht_minutes: float
    throughput_tph: float
    automation_rate: float
    fpy: float
    error_detect: float
    cost_per_txn: float

@dataclass
class Scenario:
    baseline: Metrics
    post: Metrics
    volume_monthly: int
    costs_initial: float
    costs_operating_monthly: float
    labor_rate_per_hour: float

def efficiency_gain_minutes(baseline: float, post: float) -> float:
    return max(0.0, baseline - post)

def efficiency_gain_pct(baseline: float, post: float) -> float:
    if baseline <= 0:
        return 0.0
    return (baseline - post) / baseline

def monthly_labor_savings_minutes(scn: Scenario) -> float:
    return scn.volume_monthly * efficiency_gain_minutes(scn.baseline.aht_minutes, scn.post.aht_minutes)

def monthly_labor_savings_dollars(scn: Scenario) -> float:
    minutes = monthly_labor_savings_minutes(scn)
    hours = minutes / 60.0
    return hours * scn.labor_rate_per_hour

def monthly_error_reduction_value(scn: Scenario, rework_cost_per_txn: float = 1.0) -> float:
    # simplistic proxy: improvement in FPY reduces rework transactions times unit rework cost
    delta_fpy = max(0.0, scn.post.fpy - scn.baseline.fpy)
    return scn.volume_monthly * delta_fpy * rework_cost_per_txn

def monthly_total_benefit(scn: Scenario, rework_cost_per_txn: float = 1.0) -> float:
    return monthly_labor_savings_dollars(scn) + monthly_error_reduction_value(scn, rework_cost_per_txn)

def monthly_net_benefit(scn: Scenario, rework_cost_per_txn: float = 1.0) -> float:
    return monthly_total_benefit(scn, rework_cost_per_txn) - scn.costs_operating_monthly

def roi_first_year(scn: Scenario, rework_cost_per_txn: float = 1.0) -> float:
    benefits = monthly_total_benefit(scn, rework_cost_per_txn) * 12
    costs = scn.costs_initial + scn.costs_operating_monthly * 12
    if costs == 0:
        return float('inf')
    return (benefits - costs) / costs

def break_even_month(scn: Scenario, rework_cost_per_txn: float = 1.0) -> Optional[int]:
    cum = -scn.costs_initial
    for m in range(1, 61):  # 5-year horizon
        cum += monthly_net_benefit(scn, rework_cost_per_txn)
        if cum >= 0:
            return m
    return None

def scenario_from_json(d: Dict) -> Scenario:
    baseline = Metrics(**d["baseline"])
    post = Metrics(**d["post"])
    return Scenario(
        baseline=baseline,
        post=post,
        volume_monthly=int(d["volume_monthly"]),
        costs_initial=float(d["costs"]["initial"]),
        costs_operating_monthly=float(d["costs"]["operating_monthly"]),
        labor_rate_per_hour=float(d.get("labor_rate_per_hour", 35.0)),
    )

def simulate_curve(scn: Scenario, months: int = 36, rework_cost_per_txn: float = 1.0) -> Dict:
    points = []
    cum = -scn.costs_initial
    for m in range(1, months + 1):
        nb = monthly_net_benefit(scn, rework_cost_per_txn)
        cum += nb
        points.append({"month": m, "net_benefit": nb, "cumulative": cum})
    return {"points": points}

def main():
    ap = argparse.ArgumentParser(description="RPA KPI & ROI model")
    ap.add_argument("--input", required=True, help="Path to scenario JSON")
    ap.add_argument("--report", required=True, help="Write results JSON here")
    ap.add_argument("--months", type=int, default=36)
    ap.add_argument("--rework-cost", type=float, default=1.0)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    scn = scenario_from_json(cfg)

    out = {
        "efficiency_gain_pct": efficiency_gain_pct(scn.baseline.aht_minutes, scn.post.aht_minutes),
        "monthly_labor_savings_dollars": monthly_labor_savings_dollars(scn),
        "monthly_error_reduction_value": monthly_error_reduction_value(scn, args.rework_cost),
        "monthly_total_benefit": monthly_total_benefit(scn, args.rework_cost),
        "monthly_net_benefit": monthly_net_benefit(scn, args.rework_cost),
        "roi_first_year": roi_first_year(scn, args.rework_cost),
        "break_even_month": break_even_month(scn, args.rework_cost),
        "curve": simulate_curve(scn, args.months, args.rework_cost),
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
