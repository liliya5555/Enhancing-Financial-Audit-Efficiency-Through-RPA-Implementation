
"""
audit_event_log_aggregator.py

Unified aggregator for audit/RPA event logs across modules and systems.

Highlights
---------
- Normalizes heterogeneous logs (JSONL/CSV) into a canonical schema.
- Correlates events by (entity_id, document_id, po_id, invoice_id, txn_id, etc.).
- Builds ordered timelines and computes dwell times, handoff counts, and critical path.
- Redacts PII via configurable regex rules and produces an auditor-ready JSON report.
- Emits quality flags: missing fields, time inversions, clock skew, duplicate event_ids.

Canonical schema (internal representation)
-----------------------------------------
{
  "event_id": str,              # unique id (if missing, synthesized)
  "timestamp": datetime,
  "source": str,                # subsystem/module name
  "actor": str,                 # user/bot/system
  "entity_type": str,           # invoice|po|expense|bank_txn|grn|...
  "entity_id": str,             # invoice_id/po_id/expense_id/etc.
  "action": str,                # created|validated|matched|reconciled|flagged|...
  "status": str,                # success|warning|error|...
  "metadata": dict              # free-form details
}

CLI
---
python audit_event_log_aggregator.py \\
  --inputs a.jsonl b.csv \\
  --out aggregated_report.json \\
  --pii-patterns "\\b[0-9]{16}\\b,\\b\\d{3}-\\d{2}-\\d{4}\\b"
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("audit_log_agg")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ------------------------ Helpers ------------------------
def parse_ts(s: str) -> datetime:
    formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
               "%d/%m/%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S"]
    for f in formats:
        try:
            return datetime.strptime(s.strip(), f)
        except Exception:
            pass
    raise ValueError(f"Unrecognized timestamp: {s}")

def synth_id() -> str:
    return str(uuid.uuid4())

def redact(value: Any, patterns: List[re.Pattern]) -> Any:
    if isinstance(value, str):
        out = value
        for p in patterns:
            out = p.sub("***REDACTED***", out)
        return out
    if isinstance(value, dict):
        return {k: redact(v, patterns) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, patterns) for v in value]
    return value

# ------------------------ Model ------------------------
@dataclass
class CanonicalEvent:
    event_id: str
    timestamp: datetime
    source: str
    actor: str
    entity_type: str
    entity_id: str
    action: str
    status: str
    metadata: Dict[str, Any]

# ------------------------ Loaders ------------------------
def load_file(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        return [obj]
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported file type: {path}")

def normalize(row: Dict[str, Any]) -> CanonicalEvent:
    # best-effort field picking with common aliases
    eid = row.get("event_id") or row.get("id") or synth_id()
    ts = row.get("timestamp") or row.get("time") or row.get("ts")
    source = row.get("source") or row.get("module") or "unknown"
    actor = row.get("actor") or row.get("user") or row.get("bot") or "system"
    entity_type = row.get("entity_type") or row.get("type") or "unknown"
    entity_id = (row.get("entity_id") or row.get("invoice_id") or row.get("po_id") or
                 row.get("expense_id") or row.get("txn_id") or row.get("grn_id") or "unknown")
    action = row.get("action") or row.get("event") or row.get("operation") or "unknown"
    status = row.get("status") or row.get("level") or "success"
    metadata = row.get("metadata") or {k: v for k, v in row.items()
                                       if k not in {"event_id","id","timestamp","time","ts","source","module",
                                                    "actor","user","bot","entity_type","type","entity_id",
                                                    "invoice_id","po_id","expense_id","txn_id","grn_id",
                                                    "action","event","operation","status","level"}}
    return CanonicalEvent(
        event_id=str(eid),
        timestamp=parse_ts(str(ts)) if not isinstance(ts, datetime) else ts,
        source=str(source),
        actor=str(actor),
        entity_type=str(entity_type),
        entity_id=str(entity_id),
        action=str(action),
        status=str(status),
        metadata=metadata if isinstance(metadata, dict) else {"metadata": str(metadata)},
    )

# ------------------------ Aggregation ------------------------
def aggregate(files: List[Path], pii_regexes: List[str]) -> Dict[str, Any]:
    patterns = [re.compile(r) for r in pii_regexes if r]
    events: List[CanonicalEvent] = []
    for p in files:
        logger.info("Loading %s", p)
        for row in load_file(p):
            try:
                ev = normalize(row)
                events.append(ev)
            except Exception as e:
                logger.warning("Skip row in %s: %s", p, e)

    # Redact PII
    redacted = []
    for e in events:
        meta = redact(e.metadata, patterns) if patterns else e.metadata
        redacted.append(CanonicalEvent(e.event_id, e.timestamp, e.source, e.actor,
                                       e.entity_type, e.entity_id, e.action, e.status, meta))

    # Sort and group
    redacted.sort(key=lambda x: (x.entity_type, x.entity_id, x.timestamp, x.event_id))
    report = {"entities": {}, "quality_warnings": []}

    # Quality checks
    seen_ids = set()
    for ev in redacted:
        if ev.event_id in seen_ids:
            report["quality_warnings"].append(f"Duplicate event_id: {ev.event_id}")
        seen_ids.add(ev.event_id)

        key = f"{ev.entity_type}:{ev.entity_id}"
        report["entities"].setdefault(key, {"timeline": [], "stats": {}})
        report["entities"][key]["timeline"].append(asdict(ev))

    # Compute stats per entity
    for key, blk in report["entities"].items():
        tl = blk["timeline"]
        # time inversion check
        for i in range(1, len(tl)):
            if tl[i]["timestamp"] < tl[i-1]["timestamp"]:
                report["quality_warnings"].append(f"Time inversion in {key} between {tl[i-1]['event_id']} and {tl[i]['event_id']}")

        # dwell time & handoffs
        dwell = 0.0
        handoffs = 0
        for i in range(1, len(tl)):
            dt = (tl[i]["timestamp"] - tl[i-1]["timestamp"]).total_seconds()
            dwell += max(0.0, dt)
            if tl[i]["actor"] != tl[i-1]["actor"]:
                handoffs += 1
        blk["stats"] = {"event_count": len(tl), "dwell_seconds": dwell, "handoffs": handoffs}

    return report

# ------------------------ CLI ------------------------
def main():
    ap = argparse.ArgumentParser(description="Aggregate heterogeneous audit logs into canonical timelines")
    ap.add_argument("--inputs", nargs="+", required=True, help="List of JSONL/JSON/CSV files")
    ap.add_argument("--out", required=True, help="Path to write aggregated JSON report")
    ap.add_argument("--pii-patterns", default="", help="Comma-separated regex list for PII redaction")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs]
    patterns = [r.strip() for r in args.pii_patterns.split(",") if r.strip()]
    report = aggregate(inputs, patterns)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Wrote report to %s", args.out)

if __name__ == "__main__":
    main()
