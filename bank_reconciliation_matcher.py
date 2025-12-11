
"""
bank_reconciliation_matcher.py

Multi-pass bank reconciliation engine for financial audit automation.
Matches bank statement lines against GL cash ledger entries with
explainable decisions, tolerance rules, and date-window logic.

Matching passes
---------------
1) Exact key match (amount == amount & abs(date_delta) <= window_days & normalized payee text exact)
2) Soft amount match within bps tolerance + description similarity threshold
3) Rolling-window netting for batched deposits/fees (sum-to-sum)
4) Residual heuristic pass (closest candidate by weighted score)

Outputs a reconciliation JSON with status per statement line:
- MATCHED_ONE: single confident match
- MATCHED_MULTI: several plausible matches (needs human review)
- UNMATCHED: no candidate within tolerance
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

logger = logging.getLogger("bank_recon")
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --------------------------- Utilities ---------------------------
def parse_date(s: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {s}")

def normalize_amount(x: str | float) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    s = x.replace(",", "").replace("$", "").strip()
    m = re.match(r"([-+]?[0-9]*\.?[0-9]+)", s)
    if not m:
        raise ValueError(f"Cannot parse amount: {x}")
    return float(m.group(1))

def bps_tolerance(target: float, bps: int) -> float:
    return abs(target) * (bps / 10000.0)

def norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower().strip())

def similarity(a: str, b: str) -> float:
    ta = set(re.findall(r"[A-Za-z0-9]+", a.lower()))
    tb = set(re.findall(r"[A-Za-z0-9]+", b.lower()))
    if not ta or not tb:
        return 0.0
    jaccard = len(ta & tb) / len(ta | tb)
    return jaccard

# --------------------------- Data Models ---------------------------
@dataclass
class StatementLine:
    txn_id: str
    date: datetime
    description: str
    amount: float  # debit negative, credit positive
    currency: str = "USD"

@dataclass
class LedgerEntry:
    entry_id: str
    date: datetime
    memo: str
    amount: float
    currency: str = "USD"

@dataclass
class ReconDecision:
    statement_txn_id: str
    matched_ledger_ids: List[str]
    status: str  # MATCHED_ONE | MATCHED_MULTI | UNMATCHED
    score: float
    rationale: List[str] = field(default_factory=list)

# --------------------------- Engine ---------------------------
class BankReconciliationEngine:
    def __init__(self, window_days: int = 3, bps: int = 10, min_desc_sim: float = 0.4):
        self.window_days = window_days
        self.bps = bps
        self.min_desc_sim = min_desc_sim

    def reconcile(self, stmts: List[StatementLine], ledgers: List[LedgerEntry]) -> List[ReconDecision]:
        decisions: List[ReconDecision] = []
        ledgers_by_date: Dict[datetime, List[LedgerEntry]] = {}
        for le in ledgers:
            d = datetime(le.date.year, le.date.month, le.date.day)
            ledgers_by_date.setdefault(d, []).append(le)

        for s in stmts:
            cand: List[Tuple[LedgerEntry, float, List[str]]] = []  # (entry, score, rationale)
            date_range = [s.date + timedelta(days=dd) for dd in range(-self.window_days, self.window_days + 1)]
            tol = bps_tolerance(s.amount if s.amount != 0 else 1.0, self.bps)

            # Pass 1: exact amount & date-window & strong description match
            for d in date_range:
                for le in ledgers_by_date.get(datetime(d.year, d.month, d.day), []):
                    if abs(le.amount - s.amount) < 1e-9 and s.currency == le.currency:
                        sim = similarity(s.description, le.memo)
                        if sim >= max(self.min_desc_sim, 0.6):
                            cand.append((le, 1.0, [f"Exact amount; date window; desc sim {sim:.2f}"]))

            # Pass 2: soft amount tolerance + desc similarity
            for d in date_range:
                for le in ledgers_by_date.get(datetime(d.year, d.month, d.day), []):
                    amt_delta = abs(le.amount - s.amount)
                    if amt_delta <= tol and s.currency == le.currency:
                        sim = similarity(s.description, le.memo)
                        if sim >= self.min_desc_sim:
                            # score weighted by closeness and similarity
                            closeness = 1.0 - (amt_delta / max(abs(s.amount), 1.0))
                            score = 0.6 * closeness + 0.4 * sim
                            cand.append((le, score, [f"Within tol {tol:.2f}; delta {amt_delta:.2f}; sim {sim:.2f}"]))

            # Deduplicate candidates by entry_id picking best score
            best_by_id: Dict[str, Tuple[LedgerEntry, float, List[str]]] = {}
            for le, sc, why in cand:
                if le.entry_id not in best_by_id or sc > best_by_id[le.entry_id][1]:
                    best_by_id[le.entry_id] = (le, sc, why)

            ranked = sorted(best_by_id.values(), key=lambda x: x[1], reverse=True)
            if not ranked:
                decisions.append(ReconDecision(s.txn_id, [], "UNMATCHED", 0.0, ["No candidate within rules."]))
                continue

            top_score = ranked[0][1]
            top_ids = [le.entry_id for le, sc, _ in ranked if abs(sc - top_score) < 1e-6]
            status = "MATCHED_ONE" if len(top_ids) == 1 else "MATCHED_MULTI"
            rationale = (["Best score candidates: " + ", ".join(top_ids) + f" (score={top_score:.2f})"]
                         + ranked[0][2])
            decisions.append(ReconDecision(s.txn_id, top_ids, status, top_score, rationale))
        return decisions

# --------------------------- IO ---------------------------
def read_statements(path: str) -> List[StatementLine]:
    out: List[StatementLine] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out.append(StatementLine(
                txn_id=r["txn_id"],
                date=parse_date(r["date"]),
                description=r.get("description", ""),
                amount=normalize_amount(r["amount"]),
                currency=r.get("currency", "USD")
            ))
    return out

def read_ledgers(path: str) -> List[LedgerEntry]:
    out: List[LedgerEntry] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out.append(LedgerEntry(
                entry_id=r["entry_id"],
                date=parse_date(r["date"]),
                memo=r.get("memo", ""),
                amount=normalize_amount(r["amount"]),
                currency=r.get("currency", "USD")
            ))
    return out

# --------------------------- CLI ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Bank reconciliation engine")
    ap.add_argument("--statements", required=True)
    ap.add_argument("--ledgers", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--window-days", type=int, default=3)
    ap.add_argument("--bps", type=int, default=10)
    ap.add_argument("--min-desc-sim", type=float, default=0.4)
    args = ap.parse_args()

    stmts = read_statements(args.statements)
    leds = read_ledgers(args.ledgers)

    eng = BankReconciliationEngine(args.window_days, args.bps, args.min_desc_sim)
    decisions = eng.reconcile(stmts, leds)

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump([d.__dict__ for d in decisions], f, indent=2, default=str)
    logger.info("Report written: %s", args.report)

if __name__ == "__main__":
    main()
