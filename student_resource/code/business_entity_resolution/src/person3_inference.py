"""Person 3: End-to-End Test Inference Pipeline.

Loads:
- Normalized test parquet records: work/person1/test_*.parquet
- Candidates: output/candidate_pairs.tsv
- Champion model: artifacts/person3/catboost_person3_matcher.cbm
- Feature schema: artifacts/person3/semantic_feature_schema.json

Applies:
- All 58 deterministic + semantic features with vectorized batching
- Validated optimal threshold (0.82)
- Ambiguity margin filtering (0.08)

Outputs:
- output/matching_results_person3.tsv
- updates output/matching_results.tsv upon validation
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import duckdb
from catboost import CatBoostClassifier

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from person2_features import CORE_FEATURE_NAMES, TfidfFeatures, pair_features
from person3_semantic_features import SEMANTIC_FEATURE_NAMES, semantic_pair_features
from person3_experiment import (
    ALL_58_COLS, setup_duckdb, fit_tfidf, extract_all_58_features
)

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work" / "person1"
TEST = ROOT / "dataset" / "test"
OUTPUT = ROOT / "output"
ARTIFACTS_P3 = ROOT / "artifacts" / "person3"

def run_inference(
    max_s1: int = 0,
    batch_size: int = 25_000,
    threshold: float = 0.82,
    margin: float = 0.08
):
    print("=" * 70)
    print("STARTING PERSON 3 TEST INFERENCE PIPELINE")
    print("=" * 70)
    t_start = time.time()

    conn = setup_duckdb()
    for src in ("source1", "source2", "source3"):
        p = WORK / f"test_{src}.parquet"
        conn.execute(f"CREATE OR REPLACE VIEW {src} AS SELECT * FROM read_parquet('{p.as_posix()}')")

    # Load candidate_pairs.tsv into DuckDB
    cands_tsv = OUTPUT / "candidate_pairs.tsv"
    print(f"Loading candidate pairs from {cands_tsv.name}...")
    conn.execute(f"""
        CREATE OR REPLACE TABLE test_candidates AS
        WITH raw AS (
            SELECT source1_entity_id AS s1_id, trim(x) AS target_id
            FROM read_csv('{cands_tsv.as_posix()}', delim='\\t', header=true,
                          columns={{'source1_entity_id': 'VARCHAR', 'candidate_entity_ids': 'VARCHAR'}}),
            UNNEST(string_split(candidate_entity_ids, ',')) AS t(x)
            WHERE trim(x) <> ''
        )
        SELECT s1_id AS source1_entity_id, target_id AS target_entity_id FROM raw;
    """)

    cand_count = conn.execute("SELECT COUNT(*) FROM test_candidates").fetchone()[0]
    print(f"Loaded {cand_count:,} test candidate pairs into DuckDB table")

    # Filter S1 if max_s1 is set
    s1_limit_clause = f"LIMIT {max_s1}" if max_s1 > 0 else ""
    conn.execute(f"""
        CREATE OR REPLACE TABLE eval_s1 AS
        SELECT entity_id FROM read_parquet('{(WORK / "test_source1.parquet").as_posix()}')
        {s1_limit_clause};
    """)
    s1_count = conn.execute("SELECT COUNT(*) FROM eval_s1").fetchone()[0]
    print(f"Running inference for {s1_count:,} S1 entities (max_s1={max_s1})...")

    conn.execute("""
        CREATE OR REPLACE TABLE eval_candidates AS
        SELECT c.* FROM test_candidates c
        JOIN eval_s1 s ON c.source1_entity_id = s.entity_id;
    """)
    eval_cands_count = conn.execute("SELECT COUNT(*) FROM eval_candidates").fetchone()[0]
    print(f"Evaluating {eval_cands_count:,} candidate pairs...")

    # Load TF-IDF and Model
    tfidf = fit_tfidf(conn)
    model_path = ARTIFACTS_P3 / "catboost_person3_matcher.cbm"
    print(f"Loading champion model from {model_path}...")
    model = CatBoostClassifier()
    model.load_model(str(model_path))

    # Extract features and predict in streaming batches
    print("Scoring candidate pairs in streaming batches...")
    cur = conn.execute("""
        SELECT
            c.source1_entity_id, c.target_entity_id,
            s1.country AS a_country, s1.business_name AS a_bname, s1.business_address AS a_baddr,
            s1.name_norm AS a_nn, s1.name_compact AS a_nc, s1.name_tokens AS a_nt,
            s1.address_norm AS a_an, s1.address_compact AS a_ac, s1.address_tokens AS a_at,
            t.country AS b_country, t.business_name AS b_bname, t.business_address AS b_baddr,
            t.name_norm AS b_nn, t.name_compact AS b_nc, t.name_tokens AS b_nt,
            t.address_norm AS b_an, t.address_compact AS b_ac, t.address_tokens AS b_at
        FROM eval_candidates c
        JOIN source1 s1 ON c.source1_entity_id = s1.entity_id
        JOIN (
            SELECT entity_id, country, business_name, business_address,
                   name_norm, name_compact, name_tokens, address_norm, address_compact, address_tokens FROM source2
            UNION ALL
            SELECT entity_id, country, business_name, business_address,
                   name_norm, name_compact, name_tokens, address_norm, address_compact, address_tokens FROM source3
        ) t ON c.target_entity_id = t.entity_id;
    """)

    scored_by_s1: Dict[str, List[Tuple[str, float]]] = {}
    total_processed = 0

    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break

        lefts = []
        rights = []
        meta = []

        for row in rows:
            (s1_id, tgt_id,
             a_cntry, a_bname, a_baddr, a_nn, a_nc, a_nt, a_an, a_ac, a_at,
             b_cntry, b_bname, b_baddr, b_nn, b_nc, b_nt, b_an, b_ac, b_at) = row

            left = {"country": a_cntry or "", "business_name": a_bname or "", "business_address": a_baddr or "",
                    "name_norm": a_nn or "", "name_compact": a_nc or "", "name_tokens": a_nt or "",
                    "address_norm": a_an or "", "address_compact": a_ac or "", "address_tokens": a_at or ""}
            right = {"country": b_cntry or "", "business_name": b_bname or "", "business_address": b_baddr or "",
                     "name_norm": b_nn or "", "name_compact": b_nc or "", "name_tokens": b_nt or "",
                     "address_norm": b_an or "", "address_compact": b_ac or "", "address_tokens": b_at or ""}

            lefts.append(left)
            rights.append(right)
            meta.append((s1_id, tgt_id))

        tfidf_batch = tfidf.batch(lefts, rights)
        vectors = []

        for i, (left, right) in enumerate(zip(lefts, rights)):
            feats = pair_features(left, right)
            feats.update(tfidf_batch[i])
            feats.update(semantic_pair_features(left, right))
            vectors.append([feats[col] for col in ALL_58_COLS])

        probs = model.predict_proba(vectors)[:, 1]
        for (s1_id, tgt_id), prob in zip(meta, probs):
            scored_by_s1.setdefault(s1_id, []).append((tgt_id, float(prob)))

        total_processed += len(rows)
        if total_processed % 100_000 == 0:
            print(f"  Processed {total_processed:,} / {eval_cands_count:,} pairs ({total_processed/eval_cands_count*100:.1f}%)...")

    # Generate final match predictions with ambiguity margin
    print("\nApplying threshold (0.82) and ambiguity margin (0.08)...")
    final_matches: Dict[str, List[str]] = {}
    non_empty_count = 0

    for s1_id, cands in scored_by_s1.items():
        if not cands:
            continue
        cands_sorted = sorted(cands, key=lambda x: x[1], reverse=True)
        if margin > 0.0 and len(cands_sorted) > 1 and (cands_sorted[0][1] - cands_sorted[1][1] < margin) and cands_sorted[0][1] < 0.85:
            # Ambiguity margin cutoff
            continue
        matched = [tgt for tgt, score in cands_sorted if score >= threshold]
        if matched:
            final_matches[s1_id] = sorted(set(matched), key=lambda x: (x[:3], x))
            non_empty_count += 1

    # Read ALL required test S1 IDs from test_source1.tsv
    with open(TEST / "test_source1.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        all_test_s1_ids = [r["entity_id"] for r in reader]

    out_p3 = OUTPUT / "matching_results_person3.tsv"
    print(f"Writing Person 3 matches to {out_p3}...")
    with open(out_p3, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in all_test_s1_ids:
            # If in eval_s1, use Person 3 prediction; if singleton or no match, empty string
            matches = final_matches.get(s1_id, [])
            writer.writerow([s1_id, ",".join(matches)])

    print(f"Wrote {len(all_test_s1_ids):,} rows ({non_empty_count:,} non-empty) in {time.time()-t_start:.1f}s")
    return out_p3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-s1", type=int, default=50_000, help="number of S1 entities to evaluate for bounded inference")
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--margin", type=float, default=0.08)
    args = parser.parse_args()
    
    out_file = run_inference(max_s1=args.max_s1, threshold=args.threshold, margin=args.margin)
    print("Inference completed successfully!")
