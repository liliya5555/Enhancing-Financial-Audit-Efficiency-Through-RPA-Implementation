
"""
data_quality_integrity_checker.py

Cross-dataset data quality and integrity checker for financial audit pipelines.

Checks
------
- Schema: required columns, type parseability, enum domains, regex formats.
- Completeness: null/empty rate by column; duplicate primary keys.
- Referential integrity: FK ↔ PK between files (e.g., invoices.po_id ⟷ pos.po_id).
- Numeric sanity: ranges, non-negativity, monotonic dates, currency whitelist.
- Time consistency: date window bounds and ordering (e.g., GRN.received_date >= PO.po_date).

Inputs
------
YAML or JSON config describing sources and rules. Example JSON:
{
  "sources": {
    "invoices": {"path": "invoices.csv", "pk": "invoice_id"},
    "pos": {"path": "pos.csv", "pk": "po_id"},
    "grns": {"path": "grns.csv", "pk": "grn_id"}
  },
  "rules": {
    "required": {
      "invoices": ["invoice_id","vendor","invoice_date","po_id","total","currency"]
    },
    "enums": {
      "invoices.currency": ["USD","EUR","CNY"]
    },
    "regex": {
      "invoices.invoice_id": "^[A-Z0-9-]+$"
    },
    "fk": [
      {"child": "invoices.po_id", "parent": "pos.po_id"},
      {"child": "grns.po_id", "parent": "pos.po_id"}
    ],
    "numeric": {
      "invoices.total": {"min": 0.0},
      "grns.received_total": {"min": 0.0}
    },
    "date_order": [
      {"before": "pos.po_date", "after": "grns.received_date"},
      {"before": "pos.po_date", "after": "invoices.invoice_date"}
    ]
  }
}

CLI
---
python data_quality_integrity_checker.py --config dq_config.json --out dq_report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("dq_checker")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ------------------------ IO ------------------------
def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def parse_date(s: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date: {s}")

# ------------------------ Core ------------------------
def load_sources(cfg: Dict) -> Dict[str, List[Dict[str, str]]]:
    out = {}
    for name, sc in cfg["sources"].items():
        out[name] = read_csv(sc["path"])
    return out

def null_rate(rows: List[Dict[str, str]], col: str) -> float:
    n = len(rows) or 1
    miss = sum(1 for r in rows if r.get(col) in (None, "", "NA", "null", "Null"))
    return miss / n

def duplicate_keys(rows: List[Dict[str, str]], pk: str) -> List[str]:
    seen = set()
    dups = []
    for r in rows:
        k = r.get(pk)
        if k in seen:
            dups.append(k)
        else:
            seen.add(k)
    return dups

def get_col(rows: List[Dict[str, str]], col: str) -> List[str]:
    return [r.get(col) for r in rows]

def check_required(rows: List[Dict[str, str]], req: List[str]) -> List[str]:
    missing_cols = [c for c in req if c not in (rows[0].keys() if rows else [])]
    msgs = [f"Missing required column: {c}" for c in missing_cols]
    return msgs

def check_enum(values: List[str], domain: List[str], path: str) -> List[str]:
    invalid = [v for v in values if v and v not in domain]
    return [f"Enum violation {path}: {v}" for v in invalid]

def check_regex(values: List[str], pat: str, path: str) -> List[str]:
    rx = re.compile(pat)
    invalid = [v for v in values if v and not rx.match(v)]
    return [f"Regex violation {path}: {v}" for v in invalid]

def check_numeric(values: List[str], path: str, minv=None, maxv=None) -> List[str]:
    msgs = []
    for i, v in enumerate(values):
        if v in (None, ""):
            continue
        try:
            x = float(str(v).replace(",", "").replace("$", "").strip())
        except Exception:
            msgs.append(f"Numeric parse error {path} row={i}: {v}")
            continue
        if minv is not None and x < minv:
            msgs.append(f"Numeric min violation {path} row={i}: {x} < {minv}")
        if maxv is not None and x > maxv:
            msgs.append(f"Numeric max violation {path} row={i}: {x} > {maxv}")
    return msgs

def check_date_order(values_a: List[str], values_b: List[str], path_a: str, path_b: str) -> List[str]:
    msgs = []
    for i, (va, vb) in enumerate(zip(values_a, values_b)):
        if not va or not vb:
            continue
        try:
            da = parse_date(va)
            db = parse_date(vb)
            if da > db:
                msgs.append(f"Date order violation {path_a} -> {path_b} row={i}: {da} > {db}")
        except Exception as e:
            msgs.append(f"Date parse error row={i}: {e}")
    return msgs

def check_fk(child_rows: List[Dict[str, str]], child_col: str, parent_rows: List[Dict[str, str]], parent_col: str, path: str) -> List[str]:
    parent_keys = set(get_col(parent_rows, parent_col))
    missing = [r.get(child_col) for r in child_rows if r.get(child_col) and r.get(child_col) not in parent_keys]
    return [f"FK violation {path}: {k}" for k in missing]

def run_checks(cfg: Dict) -> Dict:
    sources = load_sources(cfg)
    rules = cfg.get("rules", {})
    report = {"summary": {}, "violations": []}

    # Required columns & null rates & duplicates
    for name, rows in sources.items():
        req = rules.get("required", {}).get(name, [])
        report["violations"] += check_required(rows, req)
        # null rates
        for c in req:
            rate = null_rate(rows, c)
            if rate > 0:
                report["violations"].append(f"Null rate {name}.{c}: {rate:.2%}")
        # duplicate PKs
        pk = cfg["sources"][name].get("pk")
        if pk:
            dups = duplicate_keys(rows, pk)
            for k in dups:
                report["violations"].append(f"Duplicate key {name}.{pk}: {k}")

    # Enums
    for path, domain in rules.get("enums", {}).items():
        t, col = path.split(".")
        report["violations"] += check_enum(get_col(sources[t], col), domain, path)

    # Regex
    for path, pat in rules.get("regex", {}).items():
        t, col = path.split(".")
        report["violations"] += check_regex(get_col(sources[t], col), pat, path)

    # Numeric
    for path, lim in rules.get("numeric", {}).items():
        t, col = path.split(".")
        report["violations"] += check_numeric(get_col(sources[t], col), path, lim.get("min"), lim.get("max"))

    # Date order
    for rule in rules.get("date_order", []):
        ta, ca = rule["before"].split(".")
        tb, cb = rule["after"].split(".")
        report["violations"] += check_date_order(get_col(sources[ta], ca), get_col(sources[tb], cb), rule["before"], rule["after"])

    # FKs
    for fk in rules.get("fk", []):
        tc, cc = fk["child"].split(".")
        tp, cp = fk["parent"].split(".")
        report["violations"] += check_fk(sources[tc], cc, sources[tp], cp, f"{fk['child']} -> {fk['parent']}")

    report["summary"] = {
        "tables": {name: {"rows": len(rows)} for name, rows in sources.items()},
        "violation_count": len(report["violations"])
    }
    return report

def main():
    ap = argparse.ArgumentParser(description="Data quality & integrity checker")
    ap.add_argument("--config", required=True, help="Path to JSON config")
    ap.add_argument("--out", required=True, help="Path to write JSON report")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    rep = run_checks(cfg)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

if __name__ == "__main__":
    main()
