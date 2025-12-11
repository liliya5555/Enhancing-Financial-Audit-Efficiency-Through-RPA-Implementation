
"""
anomalous_transaction_detector.py

Unsupervised detector for anomalous financial transactions (expenses, invoices, bank txns).

Design
------
- RobustZ: Median + MAD-based z-score per numeric feature (scale-invariant).
- Density proxy: cosine-kNN density score in a whitened space (no external libs).
- Composite anomaly score = 0.6 * RobustZ_rank + 0.4 * (1 - density_rank).
- Outputs top-K anomalies with per-feature contributions and textual rationales.

Input (CSV)
-----------
Must include an id column (e.g., txn_id/invoice_id/expense_id). Numeric columns are auto-detected.

CLI
---
python anomalous_transaction_detector.py --input data.csv --id-col txn_id --topk 100 --out out.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

logger = logging.getLogger("anomaly_det")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ------------------------ Utilities ------------------------
def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def median(xs: List[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    m = n // 2
    return (ys[m-1] + ys[m]) / 2 if n % 2 == 0 else ys[m]

def mad(xs: List[float], med: float) -> float:
    return median([abs(x - med) for x in xs]) or 1.0

def dot(a: List[float], b: List[float]) -> float:
    return sum(x*y for x, y in zip(a, b))

def norm(a: List[float]) -> float:
    return math.sqrt(dot(a, a)) or 1.0

def cosine(a: List[float], b: List[float]) -> float:
    return dot(a, b) / (norm(a) * norm(b))

# ------------------------ Core ------------------------
def load_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows, list(rows[0].keys()) if rows else []

def select_numeric_columns(rows: List[Dict[str, str]], exclude: List[str]) -> List[str]:
    counts: Dict[str, int] = {}
    for r in rows:
        for k, v in r.items():
            if k in exclude:
                continue
            if v is None or v == "":
                continue
            if is_float(v):
                counts[k] = counts.get(k, 0) + 1
    # keep columns that are numeric in at least 80% of rows
    n = len(rows)
    return [k for k, c in counts.items() if c >= 0.8 * n]

def robust_z_scores(rows: List[Dict[str, str]], num_cols: List[str]) -> Tuple[List[Dict[str, float]], Dict[str, Tuple[float, float]]]:
    stats = {}
    for c in num_cols:
        col = [float(r.get(c) or 0.0) for r in rows]
        med = median(col)
        m = mad(col, med)
        stats[c] = (med, m)
    z = []
    for r in rows:
        zr = {}
        for c in num_cols:
            val = float(r.get(c) or 0.0)
            med, m = stats[c]
            zr[c] = 0.6745 * (val - med) / m  # 0.6745 makes MAD comparable to std
        z.append(zr)
    return z, stats

def whiten(z: List[Dict[str, float]], num_cols: List[str]) -> List[List[float]]:
    # simple variance normalization in z-space (mean ~0 by construction)
    vars = []
    for c in num_cols:
        vals = [zr[c] for zr in z]
        v = sum(vv*vv for vv in vals) / max(1, len(vals) - 1)
        vars.append(math.sqrt(v) or 1.0)
    X = []
    for zr in z:
        X.append([zr[c] / s for c, s in zip(num_cols, vars)])
    return X

def density_score(X: List[List[float]], k: int = 10) -> List[float]:
    # cosine-kNN density: average cosine to top-k neighbors (exclude self)
    n = len(X)
    scores = []
    for i in range(n):
        sims = []
        xi = X[i]
        for j in range(n):
            if i == j:
                continue
            sims.append(cosine(xi, X[j]))
        sims.sort(reverse=True)
        if not sims:
            scores.append(0.0)
        else:
            scores.append(sum(sims[:min(k, len(sims))]) / min(k, len(sims)))
    return scores

def rank(values: List[float], reverse: bool = True) -> List[int]:
    # returns 0-based ranks
    pairs = sorted([(v, i) for i, v in enumerate(values)], reverse=reverse)
    ranks = [0]*len(values)
    for r, (_, i) in enumerate(pairs):
        ranks[i] = r
    return ranks

def composite_anomaly(zmag: List[float], dens: List[float]) -> List[float]:
    # larger zmag is more anomalous; larger density is less anomalous
    max_z = max(zmag) or 1.0
    max_d = max(dens) or 1.0
    z_norm = [v / max_z for v in zmag]
    d_norm = [v / max_d for v in dens]
    return [0.6*z + 0.4*(1.0 - d) for z, d in zip(z_norm, d_norm)]

def explain_row(zr: Dict[str, float], topn: int = 3) -> List[str]:
    parts = sorted([(abs(v), k, v) for k, v in zr.items()], reverse=True)[:topn]
    return [f"{k}: z={v:.2f}" for _, k, v in parts]

def detect(path: str, id_col: str, topk: int) -> Dict:
    rows, _ = load_csv(path)
    if not rows:
        return {"anomalies": []}
    num_cols = select_numeric_columns(rows, exclude=[id_col])
    z, stats = robust_z_scores(rows, num_cols)
    X = whiten(z, num_cols)
    dens = density_score(X, k=max(5, min(15, len(X)//10 or 5)))
    zmag = [sum(abs(zr[c]) for c in num_cols) for zr in z]
    score = composite_anomaly(zmag, dens)
    ranked = sorted(range(len(score)), key=lambda i: score[i], reverse=True)[:min(topk, len(score))]
    out = []
    for i in ranked:
        rid = rows[i].get(id_col, f"row_{i}")
        out.append({
            "id": rid,
            "score": score[i],
            "explanation": explain_row(z[i], topn=4),
            "z_magnitude": zmag[i],
            "density": dens[i]
        })
    return {"id_col": id_col, "numeric_columns": num_cols, "anomalies": out}

def main():
    ap = argparse.ArgumentParser(description="Unsupervised anomalous transaction detector")
    ap.add_argument("--input", required=True)
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res = detect(args.input, args.id_col, args.topk)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    logger.info("Wrote anomalies to %s", args.out)

if __name__ == "__main__":
    main()
