"""Evaluation and decision utilities shared by Person 2 and Person 4."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping


def f05(predicted: set[str], truth: set[str]) -> float:
    if not predicted and not truth:
        return 1.0
    if not predicted or not truth:
        return 0.0
    tp = len(predicted & truth)
    p, r = tp / len(predicted), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r) if r else 0.0


def evaluate_sets(predictions: Mapping[str, set[str]], truth: Mapping[str, set[str]], all_s1: Iterable[str]) -> dict:
    rows = [(eid, predictions.get(eid, set()), truth.get(eid, set())) for eid in all_s1]
    tp = sum(len(p & t) for _, p, t in rows)
    fp = sum(len(p - t) for _, p, t in rows)
    fn = sum(len(t - p) for _, p, t in rows)
    singleton_errors = sum(bool(p) for _, p, t in rows if not t)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "macro_f05": sum(f05(p, t) for _, p, t in rows) / max(1, len(rows)),
        "precision": precision, "recall": recall, "true_positives": tp,
        "false_positives": fp, "false_negatives": fn,
        "singleton_false_matches": singleton_errors,
        "s1_entities": len(rows), "truth_singletons": sum(not t for _, _, t in rows),
    }


def threshold_predictions(scored: Iterable[tuple[str, str, float]], threshold: float, margin: float = 0.0) -> dict[str, set[str]]:
    by_s1 = defaultdict(list)
    for s1, target, score in scored:
        by_s1[s1].append((target, float(score)))
    out = {}
    for s1, rows in by_s1.items():
        rows.sort(key=lambda x: x[1], reverse=True)
        if margin and len(rows) > 1 and rows[0][1] - rows[1][1] < margin:
            out[s1] = set()
        else:
            out[s1] = {target for target, score in rows if score >= threshold}
    return out


def optimize_threshold(scored: Iterable[tuple[str, str, float]], truth: Mapping[str, set[str]], all_s1: Iterable[str], candidates: Iterable[float] | None = None) -> tuple[float, dict]:
    rows = list(scored)
    best_t, best = 0.5, None
    for threshold in candidates or [i / 100 for i in range(50, 100)]:
        metrics = evaluate_sets(threshold_predictions(rows, threshold), truth, all_s1)
        if best is None or metrics["macro_f05"] > best["macro_f05"]:
            best_t, best = threshold, metrics
    return best_t, best


def mine_hard_negatives(scored: Iterable[tuple[str, str, float]], truth: Mapping[str, set[str]], limit: int = 100_000) -> list[tuple[str, str, float]]:
    """Return highest-scoring known negatives for a second training pass."""
    negatives = [(s1, target, score) for s1, target, score in scored if target not in truth.get(s1, set())]
    negatives.sort(key=lambda x: x[2], reverse=True)
    return negatives[:limit]
