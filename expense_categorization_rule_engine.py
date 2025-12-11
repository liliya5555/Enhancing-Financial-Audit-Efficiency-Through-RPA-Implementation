
"""
expense_categorization_rule_engine.py

Policy-driven expense categorization & validation engine for audit automation.

Capabilities
------------
- Deterministic rules with a compact DSL: amount ranges, MCC codes, keyword includes/excludes,
  weekday/weekend checks, and per-policy thresholds.
- Duplicate detection across submitter/date/amount with configurable time windows.
- Risk scoring that weights rule confidence, past behavior, and anomaly triggers.
- Full decision trace for each expense line (auditable explanations).

Input schemas (CSV)
-------------------
expenses.csv: expense_id,employee_id,submit_date,description,amount,currency,merchant,mcc,cost_center
policies.json: see PolicySchema below

CLI
---
python expense_categorization_rule_engine.py --expenses expenses.csv --policies policies.json --report out.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("expense_rules")
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def parse_date(s: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date: {s}")

def norm_amount(x: str | float) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    s = x.replace(",", "").replace("$", "").strip()
    m = re.match(r"([-+]?[0-9]*\.?[0-9]+)", s)
    if not m:
        raise ValueError(f"Cannot parse amount: {x}")
    return float(m.group(1))

@dataclass
class Expense:
    expense_id: str
    employee_id: str
    submit_date: datetime
    description: str
    amount: float
    currency: str
    merchant: str
    mcc: str
    cost_center: str

@dataclass
class RuleDecision:
    category: str
    confidence: float
    reasons: List[str]

@dataclass
class ValidationFinding:
    expense_id: str
    status: str  # OK | POLICY_VIOLATION | DUPLICATE_SUSPECT | OUTLIER_SUSPECT
    category: str
    risk_score: float
    messages: List[str] = field(default_factory=list)

class Rule:
    """
    Small DSL for categorical mapping. Example snippet:
    {
      "category": "Travel:Airfare",
      "amount_range": [50, 5000],
      "mcc_in": ["3000","3001"],
      "include_keywords": ["airline","flight","airfare"],
      "exclude_keywords": ["baggage fee"],
      "weekend_allowed": false
    }
    """
    def __init__(self, cfg: Dict):
        self.category = cfg["category"]
        self.amount_range = cfg.get("amount_range", [0, float("inf")])
        self.mcc_in = set(cfg.get("mcc_in", []))
        self.include_keywords = [k.lower() for k in cfg.get("include_keywords", [])]
        self.exclude_keywords = [k.lower() for k in cfg.get("exclude_keywords", [])]
        self.weekend_allowed = bool(cfg.get("weekend_allowed", True))
        self.weight = float(cfg.get("weight", 1.0))

    def score(self, e: Expense) -> Tuple[float, List[str]]:
        reasons = []
        score = 0.0
        # Amount
        if self.amount_range[0] <= e.amount <= self.amount_range[1]:
            score += 0.3
            reasons.append("amount_in_range")
        # MCC
        if not self.mcc_in or e.mcc in self.mcc_in:
            score += 0.2
            reasons.append("mcc_ok")
        # Keywords
        d = f"{e.description} {e.merchant}".lower()
        inc_hit = any(k in d for k in self.include_keywords) if self.include_keywords else True
        exc_hit = any(k in d for k in self.exclude_keywords)
        if inc_hit and not exc_hit:
            score += 0.4
            reasons.append("keywords_ok")
        # Weekend policy
        if not self.weekend_allowed:
            if e.submit_date.weekday() >= 5:
                reasons.append("weekend_not_allowed")
                return 0.0, reasons
            else:
                score += 0.1
                reasons.append("weekday_ok")
        return self.weight * score, reasons

class Policy:
    """
    Policy JSON schema:
    {
      "rules": [ <Rule>, ... ],
      "limits": {
        "Travel:Hotel": 400,
        "Meals": 80
      },
      "duplicate_window_days": 30
    }
    """
    def __init__(self, cfg: Dict):
        self.rules = [Rule(rcfg) for rcfg in cfg.get("rules", [])]
        self.limits = cfg.get("limits", {})
        self.dup_window_days = int(cfg.get("duplicate_window_days", 30))

    def classify(self, e: Expense) -> RuleDecision:
        best = (None, -1.0, [])  # (category, score, reasons)
        for r in self.rules:
            sc, why = r.score(e)
            if sc > best[1]:
                best = (r.category, sc, why)
        cat = best[0] or "Uncategorized"
        conf = max(0.0, min(1.0, best[1]))
        return RuleDecision(cat, conf, best[2])

    def check_limit(self, category: str, amount: float) -> Optional[str]:
        lim = self.limits.get(category)
        if lim is not None and amount > float(lim):
            return f"Amount {amount:.2f} exceeds policy limit {lim:.2f} for {category}"
        return None

class ExpenseValidator:
    def __init__(self, policy: Policy):
        self.policy = policy
        self.seen: Dict[Tuple[str, float], List[Expense]] = {}

    def _dup_key(self, e: Expense) -> Tuple[str, float]:
        return (e.employee_id, round(e.amount, 2))

    def _dup_hit(self, e: Expense) -> Optional[str]:
        key = self._dup_key(e)
        prev = self.seen.get(key, [])
        for p in prev:
            delta = abs((e.submit_date - p.submit_date).days)
            if delta <= self.policy.dup_window_days and (e.description[:32] == p.description[:32]):
                return f"Duplicate suspect with {p.expense_id} (Δdays={delta})"
        prev.append(e)
        self.seen[key] = prev
        return None

    def risk_score(self, decision: RuleDecision, limit_msg: Optional[str], dup_msg: Optional[str]) -> float:
        score = 0.0
        score += (1.0 - decision.confidence) * 0.5
        score += 0.3 if limit_msg else 0.0
        score += 0.2 if dup_msg else 0.0
        return max(0.0, min(1.0, score))

    def validate(self, e: Expense) -> ValidationFinding:
        dec = self.policy.classify(e)
        msgs = [f"classified={dec.category} conf={dec.confidence:.2f}"] + dec.reasons
        limit_msg = self.policy.check_limit(dec.category, e.amount)
        if limit_msg:
            msgs.append(limit_msg)
        dup_msg = self._dup_hit(e)
        if dup_msg:
            msgs.append(dup_msg)
        risk = self.risk_score(dec, limit_msg, dup_msg)
        status = "OK"
        if limit_msg:
            status = "POLICY_VIOLATION"
        elif dup_msg:
            status = "DUPLICATE_SUSPECT"
        elif dec.category == "Uncategorized":
            status = "OUTLIER_SUSPECT"
        return ValidationFinding(e.expense_id, status, dec.category, risk, msgs)

# --------------------------- IO & CLI ---------------------------
def read_expenses(path: str) -> List[Expense]:
    out: List[Expense] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out.append(Expense(
                expense_id=r["expense_id"],
                employee_id=r["employee_id"],
                submit_date=parse_date(r["submit_date"]),
                description=r.get("description", ""),
                amount=norm_amount(r["amount"]),
                currency=r.get("currency", "USD"),
                merchant=r.get("merchant", ""),
                mcc=r.get("mcc", ""),
                cost_center=r.get("cost_center", ""),
            ))
    return out

def main():
    ap = argparse.ArgumentParser(description="Expense policy categorization & validation")
    ap.add_argument("--expenses", required=True)
    ap.add_argument("--policies", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    with open(args.policies, "r", encoding="utf-8") as f:
        pol_cfg = json.load(f)
    policy = Policy(pol_cfg)
    validator = ExpenseValidator(policy)

    expenses = read_expenses(args.expenses)
    findings = [validator.validate(e) for e in expenses]

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump([f.__dict__ for f in findings], f, indent=2, default=str)
    logger.info("Report written: %s", args.report)

if __name__ == "__main__":
    main()
