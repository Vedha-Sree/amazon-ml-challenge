"""Person 2: precision-first pair matcher.

The script consumes Person 1's normalized parquet files and
``output/candidate_pairs.tsv``.  It deliberately keeps candidate expansion in
DuckDB and computes pair features in bounded batches.  Training candidates are
created from the same eight blocking rules used by Person 1, so validation and
test inference use the same interface.

Typical usage (from ``student_resource``)::

    python code/business_entity_resolution/src/person1_prepare.py
    python code/business_entity_resolution/src/person1_fast_candidate_generator.py
    python code/business_entity_resolution/src/person2_matcher.py

CatBoost is preferred.  If it is not installed, ``--deterministic`` provides a
reproducible conservative baseline, which is useful for smoke tests only.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    import duckdb
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("Install requirements.txt before running person2_matcher.py") from exc

try:
    from rapidfuzz import fuzz
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("Install requirements.txt before running person2_matcher.py") from exc

from person2_features import CORE_FEATURE_NAMES, TfidfFeatures, pair_features
from person2_evaluation import evaluate_sets


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work" / "person1"
TRAIN = ROOT / "dataset" / "train"
TEST = ROOT / "dataset" / "test"
OUTPUT = ROOT / "output"
ARTIFACTS = ROOT / "artifacts" / "person2"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

FIELDS = (
    "entity_id", "country", "business_name", "business_address",
    "name_norm", "name_compact", "name_tokens", "address_norm",
    "address_compact", "address_tokens",
)
BLOCK_RULES = (
    ("name_norm", 50, False), ("name_compact", 50, False),
    ("address_norm", 50, False), ("address_compact", 50, False),
    ("name_tokens", 50, False), ("address_tokens", 50, False),
    ("name_first_word", 20, True), ("name_prefix6", 15, True),
)


def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("SET threads=4")
    c.execute("SET preserve_insertion_order=false")
    c.execute(f"SET temp_directory='{(ARTIFACTS / 'tmp').as_posix()}'")
    return c


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def register_views(c: duckdb.DuckDBPyConnection, split: str) -> None:
    for source in ("source1", "source2", "source3"):
        p = WORK / f"{split}_{source}.parquet"
        c.execute(f"""
            CREATE OR REPLACE VIEW {source} AS
            SELECT *,
              CASE WHEN length(trim(string_split(name_norm, ' ')[1])) >= 5
                   THEN trim(string_split(name_norm, ' ')[1]) END AS name_first_word,
              CASE WHEN length(name_compact) >= 6
                   THEN substr(name_compact, 1, 6) END AS name_prefix6
            FROM read_parquet('{sql_path(p)}')
        """)


def create_block_candidates(c: duckdb.DuckDBPyConnection, split: str, table: str) -> None:
    """Build distinct S1→S2/S3 candidates in a DuckDB table."""
    register_views(c, split)
    c.execute("CREATE OR REPLACE TEMP TABLE p2_candidates(source1_entity_id VARCHAR, target_entity_id VARCHAR)")
    for target in ("source2", "source3"):
        for key, cap, country_guard in BLOCK_RULES:
            country = "AND a.country = b.country" if country_guard else ""
            c.execute(f"""
                INSERT INTO p2_candidates
                SELECT DISTINCT a.entity_id, b.entity_id
                FROM source1 a JOIN {target} b ON a.{key}=b.{key} {country}
                JOIN (SELECT {key} k FROM source1 WHERE {key} <> '' GROUP BY 1 HAVING count(*) <= {cap}) x
                  ON a.{key}=x.k
                JOIN (SELECT {key} k FROM {target} WHERE {key} <> '' GROUP BY 1 HAVING count(*) <= {cap}) y
                  ON b.{key}=y.k
                WHERE a.{key} <> '' AND b.{key} <> ''
            """)
    c.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT DISTINCT * FROM p2_candidates")


def macro_f05(pred: set[str], truth: set[str]) -> float:
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    tp = len(pred & truth)
    precision = tp / len(pred)
    recall = tp / len(truth)
    return (1.25 * precision * recall) / (0.25 * precision + recall) if recall else 0.0


def gt_map() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with (TRAIN / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[row["source1_entity_id"]] = {
                x for x in row["matched_entity_ids"].split(",") if x
            }
    return out


def stable_validation(eid: str) -> bool:
    return int(hashlib.blake2b(eid.encode(), digest_size=2).hexdigest(), 16) % 5 == 0


def tokens(value: str) -> list[str]:
    return value.split() if value else []


def nums(value: str) -> set[str]:
    return set(re.findall(r"\d+", value or ""))


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    aa, bb = set(a), set(b)
    return len(aa & bb) / len(aa | bb) if aa or bb else 0.0


def char_cosine(a: str, b: str, n: int = 3) -> float:
    def grams(s: str) -> dict[str, int]:
        compact = "".join(ch for ch in s if not ch.isspace())
        return {compact[i:i+n]: compact.count(compact[i:i+n]) for i in range(max(0, len(compact)-n+1))}
    x, y = grams(a), grams(b)
    if not x or not y:
        return 0.0
    dot = sum(v * y.get(k, 0) for k, v in x.items())
    den = math.sqrt(sum(v*v for v in x.values()) * sum(v*v for v in y.values()))
    return dot / den if den else 0.0


FEATURE_NAMES = CORE_FEATURE_NAMES
TFIDF = None
PAIR_LIMIT = None
PAIR_SPLIT_FILTER = None


def features(left: dict, right: dict) -> dict[str, float]:
    return pair_features(left, right, tfidf=TFIDF)


def feature_batch(pairs: list[tuple[str, str, dict, dict]]) -> list[dict[str, float]]:
    """Build lexical features row-wise and TF-IDF features batch-wise."""
    base = [pair_features(left, right) for _, _, left, right in pairs]
    if TFIDF is not None and pairs:
        tfidf_values = TFIDF.batch([left for _, _, left, _ in pairs], [right for _, _, _, right in pairs])
        for row, extra in zip(base, tfidf_values):
            row.update(extra)
    return base


def fit_tfidf(c: duckdb.DuckDBPyConnection, per_source: int = 100_000) -> None:
    """Fit reusable TF-IDF vectorizers once, on training records only."""
    global TFIDF
    rows = []
    for source in ("source1", "source2", "source3"):
        rows.extend(c.execute(f"""
            SELECT business_name, business_address, name_norm, address_norm,
                   name_tokens, address_tokens, country
            FROM {source} LIMIT {per_source}
        """).fetchall())
    TFIDF = TfidfFeatures.fit([
        {"business_name": r[0], "business_address": r[1], "name_norm": r[2], "address_norm": r[3], "name_tokens": r[4], "address_tokens": r[5], "country": r[6]}
        for r in rows
    ])
    print(f"Fitted reusable TF-IDF vectorizers on {len(rows):,} training records")


def rows_from_query(c: duckdb.DuckDBPyConnection, query: str, batch: int = 50_000) -> Iterator[list[dict]]:
    cur = c.execute(query)
    cols = [d[0] for d in cur.description]
    while True:
        vals = cur.fetchmany(batch)
        if not vals:
            return
        yield [dict(zip(cols, row)) for row in vals]


def pair_rows(c: duckdb.DuckDBPyConnection, split: str, candidates: str) -> Iterator[list[dict]]:
    register_views(c, split)
    limit_clause = f"LIMIT {PAIR_LIMIT}" if PAIR_LIMIT else ""
    split_clause = ""
    if PAIR_SPLIT_FILTER == "validation":
        split_clause = "JOIN p2_validation_s1 v ON v.entity_id=p.source1_entity_id"
    elif PAIR_SPLIT_FILTER == "training":
        split_clause = "JOIN p2_training_s1 v ON v.entity_id=p.source1_entity_id"
    yield from rows_from_query(c, f"""
        SELECT p.source1_entity_id, p.target_entity_id,
               a.country a_country, a.business_name a_business_name, a.business_address a_business_address, a.name_norm a_name_norm, a.name_compact a_name_compact,
               a.name_tokens a_name_tokens, a.address_norm a_address_norm,
               a.address_compact a_address_compact, a.address_tokens a_address_tokens,
               b.country b_country, b.business_name b_business_name, b.business_address b_business_address, b.name_norm b_name_norm, b.name_compact b_name_compact,
               b.name_tokens b_name_tokens, b.address_norm b_address_norm,
               b.address_compact b_address_compact, b.address_tokens b_address_tokens
        FROM {candidates} p
        {split_clause}
        JOIN source1 a ON a.entity_id=p.source1_entity_id
        JOIN (SELECT * FROM source2 UNION ALL SELECT * FROM source3) b
          ON b.entity_id=p.target_entity_id
        {limit_clause}
    """)


def normalize_pair(row: dict) -> tuple[str, str, dict, dict]:
    l = {"country": row["a_country"], "business_name": row["a_business_name"] or "", "business_address": row["a_business_address"] or "", "name_norm": row["a_name_norm"] or "", "name_compact": row["a_name_compact"] or "",
         "name_tokens": row["a_name_tokens"] or "", "address_norm": row["a_address_norm"] or "", "address_compact": row["a_address_compact"] or "", "address_tokens": row["a_address_tokens"] or ""}
    r = {"country": row["b_country"], "business_name": row["b_business_name"] or "", "business_address": row["b_business_address"] or "", "name_norm": row["b_name_norm"] or "", "name_compact": row["b_name_compact"] or "",
         "name_tokens": row["b_name_tokens"] or "", "address_norm": row["b_address_norm"] or "", "address_compact": row["b_address_compact"] or "", "address_tokens": row["b_address_tokens"] or ""}
    return row["source1_entity_id"], row["target_entity_id"], l, r


def deterministic_score(f: dict[str, float]) -> float:
    # Conservative score: strong name and address agreement are required;
    # name-only similarity is intentionally insufficient.
    return 0.30*f["name_token_set"] + 0.24*f["address_token_set"] + 0.16*f["name_char_cosine"] + 0.16*f["address_char_cosine"] + 0.14*f["name_address_product"]


def threshold_metrics(scores: dict[str, list[tuple[str, float]]], gt: dict[str, set[str]], threshold: float) -> dict[str, float]:
    """Evaluate S1-level F0.5 without materializing prediction sets per threshold."""
    macro = 0.0
    tp_total = fp_total = fn_total = singleton_errors = 0
    for s1, rows in scores.items():
        truth = gt.get(s1, set())
        ordered = sorted(((-score, int(target in truth)) for target, score in rows), key=lambda x: x[0])
        cutoff = bisect_right([score for score, _ in ordered], -threshold)
        tp = sum(label for _, label in ordered[:cutoff])
        predicted = cutoff
        fp = predicted - tp
        fn = len(truth) - tp
        if not predicted and not truth:
            entity_f05 = 1.0
        elif not predicted or not truth:
            entity_f05 = 0.0
        else:
            precision = tp / predicted
            recall = tp / len(truth)
            entity_f05 = 1.25 * precision * recall / (0.25 * precision + recall) if recall else 0.0
        macro += entity_f05
        tp_total += tp
        fp_total += fp
        fn_total += fn
        singleton_errors += int(not truth and predicted > 0)
    n = max(1, len(scores))
    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 1.0
    recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 1.0
    return {"macro_f05": macro / n, "precision": precision, "recall": recall,
            "true_positives": tp_total, "false_positives": fp_total,
            "false_negatives": fn_total, "singleton_false_matches": singleton_errors,
            "s1_entities": len(scores), "truth_singletons": sum(not gt.get(s1, set()) for s1 in scores)}


def train_model(c: duckdb.DuckDBPyConnection, gt: dict[str, set[str]], deterministic: bool):
    global PAIR_SPLIT_FILTER
    PAIR_SPLIT_FILTER = "training"
    if deterministic:
        return None
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("CatBoost is required for model training; use --deterministic only for a baseline") from exc

    X: list[list[float]] = []
    y: list[int] = []
    negatives_per_s1: dict[str, int] = {}
    # Keep all positives and at most three negatives per S1. Validation is
    # entity-level, preventing near-duplicate pairs from leaking across folds.
    for batch in pair_rows(c, "train", "train_candidates"):
        normalized = [normalize_pair(row) for row in batch]
        batch_features = feature_batch(normalized)
        for (s1, target, left, right), f in zip(normalized, batch_features):
            if stable_validation(s1):
                continue
            label = int(target in gt.get(s1, set()))
            if label or negatives_per_s1.get(s1, 0) < 3:
                X.append([f[n] for n in FEATURE_NAMES])
                y.append(label)
                if not label:
                    negatives_per_s1[s1] = negatives_per_s1.get(s1, 0) + 1
    if not X or not any(y):
        raise RuntimeError("No labeled training candidates were produced; run Person 1 preparation first")
    model = CatBoostClassifier(iterations=500, depth=8, learning_rate=0.08, loss_function="Logloss", eval_metric="AUC", verbose=False, random_seed=42, thread_count=4, l2_leaf_reg=6, train_dir=str(ARTIFACTS / "catboost_train"))
    model.fit(X, y)
    return model


def choose_threshold(c: duckdb.DuckDBPyConnection, model, gt: dict[str, set[str]], deterministic: bool):
    global PAIR_SPLIT_FILTER
    PAIR_SPLIT_FILTER = "validation"
    scores: dict[str, list[tuple[str, float]]] = {}
    for batch in pair_rows(c, "train", "train_candidates"):
        normalized = [normalize_pair(row) for row in batch]
        batch_features = feature_batch(normalized)
        probabilities = None if deterministic else model.predict_proba([[f[n] for n in FEATURE_NAMES] for f in batch_features])[:, 1]
        for i, ((s1, target, left, right), f) in enumerate(zip(normalized, batch_features)):
            if not stable_validation(s1):
                continue
            score = deterministic_score(f) if deterministic else float(probabilities[i])
            scores.setdefault(s1, []).append((target, score))
    if deterministic:
        threshold = 0.82
        metrics = threshold_metrics(scores, gt, threshold)
        print(f"Validation macro F0.5={metrics['macro_f05']:.5f} threshold={threshold:.2f}")
        return threshold, metrics
    best_t, best = 0.90, None
    for t in [i / 100 for i in range(5, 96)]:
        metrics = threshold_metrics(scores, gt, t)
        if best is None or metrics["macro_f05"] > best["macro_f05"]:
            best, best_t = metrics, t
    print(f"Validation macro F0.5={best['macro_f05']:.5f} threshold={best_t:.2f}")
    return best_t, best


def write_output(c: duckdb.DuckDBPyConnection, model, threshold: float, deterministic: bool) -> None:
    global PAIR_SPLIT_FILTER
    PAIR_SPLIT_FILTER = None
    out = OUTPUT / "matching_results.tsv"
    predictions: dict[str, list[str]] = {}
    for batch in pair_rows(c, "test", "test_candidates_expanded"):
        normalized = [normalize_pair(row) for row in batch]
        batch_features = feature_batch(normalized)
        probabilities = None if deterministic else model.predict_proba([[f[n] for n in FEATURE_NAMES] for f in batch_features])[:, 1]
        for i, ((s1, target, left, right), f) in enumerate(zip(normalized, batch_features)):
            score = deterministic_score(f) if deterministic else float(probabilities[i])
            if score >= threshold:
                predictions.setdefault(s1, []).append(target)
    with (TEST / "test_source1.tsv").open(encoding="utf-8", newline="") as f, out.open("w", encoding="utf-8", newline="") as g:
        ids = [r["entity_id"] for r in csv.DictReader(f, delimiter="\t")]
        w = csv.writer(g, delimiter="\t", lineterminator="\n")
        w.writerow(["source1_entity_id", "matched_entity_ids"])
        for eid in ids:
            vals = sorted(set(predictions.get(eid, [])), key=lambda x: (x[:3], x))
            w.writerow([eid, ",".join(vals)])
    print(f"Wrote {out}")


def main() -> None:
    global PAIR_LIMIT
    ap = argparse.ArgumentParser()
    ap.add_argument("--deterministic", action="store_true", help="skip CatBoost and use conservative baseline")
    ap.add_argument("--max-train-pairs", type=int, default=500_000, help="bounded training smoke/benchmark size")
    ap.add_argument("--max-validation-pairs", type=int, default=200_000, help="bounded validation scan size")
    ap.add_argument("--max-inference-pairs", type=int, default=0, help="0 means all test candidates")
    ap.add_argument("--inference-only", action="store_true", help="load the saved CatBoost model and skip training/validation")
    ap.add_argument("--threshold", type=float, default=None, help="override the saved threshold in inference-only mode")
    args = ap.parse_args()
    if not (WORK / "train_source1.parquet").exists():
        raise SystemExit("Missing work/person1 parquet files; run person1_prepare.py first")
    if not (OUTPUT / "candidate_pairs.tsv").exists():
        raise SystemExit("Missing output/candidate_pairs.tsv; run Person 1 candidate generation first")
    c = conn()
    if args.inference_only:
        from catboost import CatBoostClassifier
        saved = json.loads((ARTIFACTS / "model_metadata.json").read_text(encoding="utf-8"))
        model = CatBoostClassifier()
        model.load_model(str(ARTIFACTS / "catboost_matcher.cbm"))
        threshold = args.threshold if args.threshold is not None else float(saved["threshold"])
        validation_metrics = saved.get("validation_metrics", {})
        register_views(c, "train")
        fit_tfidf(c)
        register_views(c, "test")
        print(f"Inference-only mode using saved threshold={threshold:.2f}")
    else:
        print("Building labeled training candidates with Person 1's blocking rules...")
        create_block_candidates(c, "train", "train_candidates")
        fit_tfidf(c)
        print("Training Person 2 matcher...")
        gt = gt_map()
        c.execute("CREATE OR REPLACE TEMP TABLE p2_validation_s1(entity_id VARCHAR)")
        c.execute("CREATE OR REPLACE TEMP TABLE p2_training_s1(entity_id VARCHAR)")
        split_path = ARTIFACTS / "fixed_validation_split.tsv"
        with split_path.open("w", encoding="utf-8") as split_file:
            split_file.write("entity_id\tsplit\n")
            for eid in gt:
                split_file.write(f"{eid}\t{'validation' if stable_validation(eid) else 'training'}\n")
        c.execute(f"""
            CREATE OR REPLACE TEMP TABLE p2_validation_s1 AS
            SELECT entity_id FROM read_csv('{sql_path(split_path)}', delim='\\t', header=true)
            WHERE split='validation'
        """)
        c.execute(f"""
            CREATE OR REPLACE TEMP TABLE p2_training_s1 AS
            SELECT entity_id FROM read_csv('{sql_path(split_path)}', delim='\\t', header=true)
            WHERE split='training'
        """)
        PAIR_LIMIT = args.max_train_pairs
        model = train_model(c, gt, args.deterministic)
        PAIR_LIMIT = args.max_validation_pairs
        threshold, validation_metrics = choose_threshold(c, model, gt, args.deterministic)
    started = time.time()
    # Expand the supplied Person 1 handoff into pair rows for inference.
    candidate_path = sql_path(OUTPUT / "candidate_pairs.tsv")
    c.execute(f"""
        CREATE OR REPLACE TEMP TABLE test_candidates AS
        SELECT * FROM read_csv(
            '{candidate_path}', delim='\\t', header=true,
            columns={{'source1_entity_id':'VARCHAR', 'candidate_entity_ids':'VARCHAR'}}
        )
    """)
    c.execute("""
        CREATE OR REPLACE TEMP TABLE test_candidates_expanded AS
        SELECT source1_entity_id, trim(x) AS target_entity_id
        FROM test_candidates, UNNEST(string_split(candidate_entity_ids, ',')) t(x)
        WHERE trim(x) <> ''
    """)
    PAIR_LIMIT = args.max_inference_pairs or None
    write_output(c, model, threshold, args.deterministic)
    metadata = {
        "feature_names": FEATURE_NAMES,
        "model": "deterministic_baseline" if args.deterministic else "catboost",
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "seed": 42,
        "tfidf_fit_records_per_source": 100_000,
        "runtime_seconds": round(time.time() - started, 3),
        "candidate_table": "test_candidates_expanded",
        "person3_extension": "pair_features accepts extra_features; semantic columns can be appended later",
    }
    (ARTIFACTS / "feature_schema.json").write_text(json.dumps({"features": FEATURE_NAMES}, indent=2), encoding="utf-8")
    (ARTIFACTS / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if model is not None and not args.inference_only:
        model.save_model(str(ARTIFACTS / "catboost_matcher.cbm"))
    print(f"Saved Person 2 artifacts under {ARTIFACTS}")


if __name__ == "__main__":
    main()
