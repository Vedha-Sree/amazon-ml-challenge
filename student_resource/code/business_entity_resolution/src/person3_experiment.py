"""Controlled ablation and evaluation suite for Person 3.

Runs systematic experiments comparing:
1. Baseline Person 2 (42 lexical features)
2. Person 2 + Semantic Name / Transliteration features
3. Person 2 + Semantic Address & Number features
4. Person 2 + Cross-Field Harmonic & Conflict features
5. Person 2 + All Person 3 Features (58 features)
6. Person 2 + All Person 3 Features + Hard Negative Mining
7. Person 2 + All Person 3 Features + Ambiguity Margin Filtering
8. Person 3 Full System (Semantic Retrieval UNION + 58 Features + Ambiguity Margin)

Uses cached feature matrices for sub-second slicing across experiments.
Saves results to:
- student_resource/artifacts/person3/ablation_results.json
- student_resource/artifacts/person3/validation_metrics.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import duckdb
from catboost import CatBoostClassifier

# Add current src directory to path
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from person2_features import CORE_FEATURE_NAMES, TfidfFeatures, pair_features
from person2_evaluation import evaluate_sets
from person3_semantic_features import SEMANTIC_FEATURE_NAMES, semantic_pair_features
from person3_semantic_retrieval import create_semantic_candidates, union_candidate_tables

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work" / "person1"
TRAIN = ROOT / "dataset" / "train"
ARTIFACTS_P2 = ROOT / "artifacts" / "person2"
ARTIFACTS_P3 = ROOT / "artifacts" / "person3"
ARTIFACTS_P3.mkdir(parents=True, exist_ok=True)

GT_FILE = TRAIN / "train_ground_truth.tsv"
SPLIT_FILE = ARTIFACTS_P2 / "fixed_validation_split.tsv"

SEM_NAME_COLS = [
    "name_translit_exact", "name_translit_ratio", "name_translit_token_set",
    "name_core_exact", "name_core_ratio", "name_token_subset",
    "name_trigram_jaccard", "name_trigram_containment"
]

SEM_ADDR_COLS = [
    "address_number_exact", "address_number_subset", "address_number_conflict",
    "address_core_token_jaccard", "address_shared_long_tokens"
]

CROSS_COLS = [
    "name_address_harmonic", "high_name_high_addr", "high_name_conflict_number"
]

ALL_58_COLS = list(CORE_FEATURE_NAMES) + SEM_NAME_COLS + SEM_ADDR_COLS + CROSS_COLS

def load_ground_truth() -> Dict[str, Set[str]]:
    gt: Dict[str, Set[str]] = {}
    with open(GT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = row["source1_entity_id"]
            matched = {m for m in row["matched_entity_ids"].split(",") if m.strip()}
            gt[s1] = matched
    return gt

def setup_duckdb():
    conn = duckdb.connect()
    conn.execute("SET threads=4")
    conn.execute("SET memory_limit='6GB'")
    return conn

def prepare_tables(conn: duckdb.DuckDBPyConnection, train_limit: int = 25_000, val_limit: int = 20_000):
    print(f"Setting up parquet views and candidate tables (train S1={train_limit:,}, val S1={val_limit:,})...")
    for src in ("source1", "source2", "source3"):
        p = WORK / f"train_{src}.parquet"
        conn.execute(f"CREATE OR REPLACE VIEW {src} AS SELECT * FROM read_parquet('{p.as_posix()}')")

    conn.execute(f"""
        CREATE OR REPLACE TABLE split_s1 AS
        SELECT entity_id, split
        FROM read_csv('{SPLIT_FILE.as_posix()}', delim='\\t', header=true);
    """)

    conn.execute(f"""
        CREATE OR REPLACE TABLE train_s1 AS
        SELECT entity_id FROM split_s1 WHERE split = 'training' LIMIT {train_limit};
    """)
    conn.execute(f"""
        CREATE OR REPLACE TABLE val_s1 AS
        SELECT entity_id FROM split_s1 WHERE split = 'validation' LIMIT {val_limit};
    """)

    P1_RULES = [
        ("name_norm", 50, False), ("name_compact", 50, False),
        ("address_norm", 50, False), ("address_compact", 50, False),
        ("name_tokens", 50, False), ("address_tokens", 50, False),
        ("name_first_word", 20, True), ("name_prefix6", 15, True),
    ]

    for split_tag, s1_tbl in (("train", "train_s1"), ("val", "val_s1")):
        conn.execute(f"""
            CREATE OR REPLACE VIEW s1_{split_tag} AS
            SELECT s1.*,
              CASE WHEN length(trim(string_split(name_norm, ' ')[1])) >= 5
                   THEN trim(string_split(name_norm, ' ')[1]) END AS name_first_word,
              CASE WHEN length(name_compact) >= 6
                   THEN substr(name_compact, 1, 6) END AS name_prefix6
            FROM source1 s1 JOIN {s1_tbl} v ON s1.entity_id = v.entity_id;
        """)

        for target in ("source2", "source3"):
            conn.execute(f"""
                CREATE OR REPLACE VIEW {target}_{split_tag} AS
                SELECT *,
                  CASE WHEN length(trim(string_split(name_norm, ' ')[1])) >= 5
                       THEN trim(string_split(name_norm, ' ')[1]) END AS name_first_word,
                  CASE WHEN length(name_compact) >= 6
                       THEN substr(name_compact, 1, 6) END AS name_prefix6
                FROM {target};
            """)

        conn.execute(f"CREATE OR REPLACE TEMP TABLE p1_cands_{split_tag}_raw(s1_id VARCHAR, target_id VARCHAR)")
        for target in ("source2", "source3"):
            for key, cap, country_guard in P1_RULES:
                country = "AND a.country = b.country" if country_guard else ""
                conn.execute(f"""
                    INSERT INTO p1_cands_{split_tag}_raw
                    SELECT DISTINCT a.entity_id, b.entity_id
                    FROM s1_{split_tag} a JOIN {target}_{split_tag} b ON a.{key}=b.{key} {country}
                    JOIN (SELECT {key} k FROM s1_{split_tag} WHERE {key} <> '' GROUP BY 1 HAVING count(*) <= {cap}) x
                      ON a.{key}=x.k
                    JOIN (SELECT {key} k FROM {target}_{split_tag} WHERE {key} <> '' GROUP BY 1 HAVING count(*) <= {cap}) y
                      ON b.{key}=y.k
                    WHERE a.{key} <> '' AND b.{key} <> ''
                """)
        conn.execute(f"""
            CREATE OR REPLACE TABLE p1_{split_tag}_candidates AS
            SELECT DISTINCT s1_id AS source1_entity_id, target_id AS target_entity_id
            FROM p1_cands_{split_tag}_raw;
        """)

    print("Generating Person 3 semantic candidate tables...")
    create_semantic_candidates(conn, "train", "p3_val_candidates", s1_filter_table="val_s1")
    create_semantic_candidates(conn, "train", "p3_train_candidates", s1_filter_table="train_s1")

    union_candidate_tables(conn, "p1_train_candidates", "p3_train_candidates", "union_train_candidates")
    union_candidate_tables(conn, "p1_val_candidates", "p3_val_candidates", "union_val_candidates")

    c_p1_val = conn.execute("SELECT COUNT(*) FROM p1_val_candidates").fetchone()[0]
    c_p3_val = conn.execute("SELECT COUNT(*) FROM p3_val_candidates").fetchone()[0]
    c_un_val = conn.execute("SELECT COUNT(*) FROM union_val_candidates").fetchone()[0]
    print(f"Validation candidates: Person 1={c_p1_val:,}, Person 3={c_p3_val:,}, UNION={c_un_val:,}")

def fit_tfidf(conn: duckdb.DuckDBPyConnection) -> TfidfFeatures:
    print("Fitting TF-IDF vectorizers on training sample...")
    rows = conn.execute("""
        SELECT business_name, business_address, name_norm, address_norm, name_tokens, address_tokens, country
        FROM source1 LIMIT 50000
    """).fetchall()
    recs = [
        {"business_name": r[0], "business_address": r[1], "name_norm": r[2], "address_norm": r[3],
         "name_tokens": r[4], "address_tokens": r[5], "country": r[6]}
        for r in rows
    ]
    return TfidfFeatures.fit(recs, max_features=50_000)

def extract_all_58_features(
    conn: duckdb.DuckDBPyConnection,
    candidate_table: str,
    tfidf: TfidfFeatures,
    gt: Dict[str, Set[str]],
    is_train: bool,
    batch_size: int = 25_000
):
    """Extract complete 58-feature matrix once for fast reuse."""
    cur = conn.execute(f"""
        SELECT
            c.source1_entity_id, c.target_entity_id,
            s1.country AS a_country, s1.business_name AS a_bname, s1.business_address AS a_baddr,
            s1.name_norm AS a_nn, s1.name_compact AS a_nc, s1.name_tokens AS a_nt,
            s1.address_norm AS a_an, s1.address_compact AS a_ac, s1.address_tokens AS a_at,
            t.country AS b_country, t.business_name AS b_bname, t.business_address AS b_baddr,
            t.name_norm AS b_nn, t.name_compact AS b_nc, t.name_tokens AS b_nt,
            t.address_norm AS b_an, t.address_compact AS b_ac, t.address_tokens AS b_at
        FROM {candidate_table} c
        JOIN source1 s1 ON c.source1_entity_id = s1.entity_id
        JOIN (
            SELECT entity_id, country, business_name, business_address,
                   name_norm, name_compact, name_tokens, address_norm, address_compact, address_tokens FROM source2
            UNION ALL
            SELECT entity_id, country, business_name, business_address,
                   name_norm, name_compact, name_tokens, address_norm, address_compact, address_tokens FROM source3
        ) t ON c.target_entity_id = t.entity_id;
    """)

    rows_all = []
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

        for i, (left, right) in enumerate(zip(lefts, rights)):
            s1_id, tgt_id = meta[i]
            feats = pair_features(left, right)
            feats.update(tfidf_batch[i])
            feats.update(semantic_pair_features(left, right))

            vec = [feats[col] for col in ALL_58_COLS]
            is_pos = int(tgt_id in gt.get(s1_id, set()))
            rows_all.append((s1_id, tgt_id, vec, is_pos, feats.get("name_ratio", 0.0)))

    return rows_all

def prepare_train_slice(
    rows_all,
    col_indices: List[int],
    max_neg_per_s1: int = 3,
    hard_negatives: bool = False
):
    X: List[List[float]] = []
    y: List[int] = []
    neg_counts: Dict[str, int] = {}

    for s1_id, tgt_id, full_vec, is_pos, name_ratio in rows_all:
        sliced_vec = [full_vec[i] for i in col_indices]
        if is_pos:
            X.append(sliced_vec)
            y.append(1)
        else:
            curr_negs = neg_counts.get(s1_id, 0)
            if curr_negs < max_neg_per_s1:
                if not hard_negatives or name_ratio > 0.40 or curr_negs == 0:
                    X.append(sliced_vec)
                    y.append(0)
                    neg_counts[s1_id] = curr_negs + 1

    return X, y

def prepare_val_slice(rows_all, col_indices: List[int]):
    by_s1: Dict[str, List[Tuple[str, List[float]]]] = {}
    for s1_id, tgt_id, full_vec, _, _ in rows_all:
        sliced_vec = [full_vec[i] for i in col_indices]
        by_s1.setdefault(s1_id, []).append((tgt_id, sliced_vec))
    return by_s1

def evaluate_predictions(
    model: CatBoostClassifier,
    by_s1: Dict[str, List[Tuple[str, List[float]]]],
    gt: Dict[str, Set[str]],
    all_val_s1: List[str],
    margin: float = 0.0
) -> Tuple[float, Dict[str, float]]:
    all_vectors = []
    pair_index = []
    for s1_id, cands in by_s1.items():
        for tgt_id, vec in cands:
            pair_index.append((s1_id, tgt_id))
            all_vectors.append(vec)

    if not all_vectors:
        return 0.5, {"macro_f05": 0.0, "precision": 0.0, "recall": 0.0, "true_positives": 0, "false_positives": 0, "false_negatives": 0, "singleton_false_matches": 0}

    probs = model.predict_proba(all_vectors)[:, 1]

    scored_by_s1: Dict[str, List[Tuple[str, float]]] = {}
    for (s1_id, tgt_id), prob in zip(pair_index, probs):
        scored_by_s1.setdefault(s1_id, []).append((tgt_id, float(prob)))

    for s1_id in all_val_s1:
        if s1_id not in scored_by_s1:
            scored_by_s1[s1_id] = []

    best_thresh = 0.5
    best_metrics = None

    for t_int in range(10, 95, 2):
        t = t_int / 100.0
        preds: Dict[str, Set[str]] = {}
        for s1_id, cands in scored_by_s1.items():
            if not cands:
                preds[s1_id] = set()
                continue
            cands_sorted = sorted(cands, key=lambda x: x[1], reverse=True)
            if margin > 0.0 and len(cands_sorted) > 1 and (cands_sorted[0][1] - cands_sorted[1][1] < margin) and cands_sorted[0][1] < 0.85:
                preds[s1_id] = set()
            else:
                preds[s1_id] = {tgt for tgt, score in cands_sorted if score >= t}

        metrics = evaluate_sets(preds, gt, all_val_s1)
        if best_metrics is None or metrics["macro_f05"] > best_metrics["macro_f05"]:
            best_metrics = metrics
            best_thresh = t

    return best_thresh, best_metrics

def run_ablations():
    print("=" * 70)
    print("STARTING PERSON 3 SYSTEMATIC ABLATION EXPERIMENTS (PRECACHED)")
    print("=" * 70)

    gt = load_ground_truth()
    conn = setup_duckdb()
    prepare_tables(conn, train_limit=25_000, val_limit=20_000)

    val_s1_list = [r[0] for r in conn.execute("SELECT entity_id FROM val_s1").fetchall()]
    tfidf = fit_tfidf(conn)

    # 1. Precompute Person 1 feature matrices
    print("\n--- Precomputing Person 1 Train Feature Matrix (58 feats) ---")
    t0 = time.time()
    p1_train_data = extract_all_58_features(conn, "p1_train_candidates", tfidf, gt, is_train=True)
    print(f"Precomputed P1 train data: {len(p1_train_data):,} rows ({time.time()-t0:.1f}s)")

    print("\n--- Precomputing Person 1 Val Feature Matrix (58 feats) ---")
    t0 = time.time()
    p1_val_data = extract_all_58_features(conn, "p1_val_candidates", tfidf, gt, is_train=False)
    print(f"Precomputed P1 val data: {len(p1_val_data):,} rows ({time.time()-t0:.1f}s)")

    # 2. Precompute Union feature matrices for full system
    print("\n--- Precomputing UNION Train Feature Matrix (58 feats) ---")
    t0 = time.time()
    union_train_data = extract_all_58_features(conn, "union_train_candidates", tfidf, gt, is_train=True)
    print(f"Precomputed UNION train data: {len(union_train_data):,} rows ({time.time()-t0:.1f}s)")

    print("\n--- Precomputing UNION Val Feature Matrix (58 feats) ---")
    t0 = time.time()
    union_val_data = extract_all_58_features(conn, "union_val_candidates", tfidf, gt, is_train=False)
    print(f"Precomputed UNION val data: {len(union_val_data):,} rows ({time.time()-t0:.1f}s)")

    # Column index maps
    idx_p2 = list(range(42))
    idx_sem_name = list(range(42, 50))
    idx_sem_addr = list(range(50, 55))
    idx_cross = list(range(55, 58))
    idx_all_58 = list(range(58))

    experiments = [
        {"name": "EXPERIMENT A: Person 2 Baseline (42 feats)", "cols": idx_p2, "train_data": p1_train_data, "val_data": p1_val_data, "hard_neg": False, "margin": 0.0, "cand_source": "p1_candidates"},
        {"name": "EXPERIMENT B: P2 + Semantic Name (50 feats)", "cols": idx_p2 + idx_sem_name, "train_data": p1_train_data, "val_data": p1_val_data, "hard_neg": False, "margin": 0.0, "cand_source": "p1_candidates"},
        {"name": "EXPERIMENT C: P2 + Semantic Address (47 feats)", "cols": idx_p2 + idx_sem_addr, "train_data": p1_train_data, "val_data": p1_val_data, "hard_neg": False, "margin": 0.0, "cand_source": "p1_candidates"},
        {"name": "EXPERIMENT D: P2 + Cross-Field Harmonic (45 feats)", "cols": idx_p2 + idx_cross, "train_data": p1_train_data, "val_data": p1_val_data, "hard_neg": False, "margin": 0.0, "cand_source": "p1_candidates"},
        {"name": "EXPERIMENT E: P2 + All P3 Features (58 feats)", "cols": idx_all_58, "train_data": p1_train_data, "val_data": p1_val_data, "hard_neg": False, "margin": 0.0, "cand_source": "p1_candidates"},
        {"name": "EXPERIMENT F: P3 All Feats + Hard Negative Mining", "cols": idx_all_58, "train_data": p1_train_data, "val_data": p1_val_data, "hard_neg": True, "margin": 0.0, "cand_source": "p1_candidates"},
        {"name": "EXPERIMENT G: P3 All Feats + Ambiguity Margin", "cols": idx_all_58, "train_data": p1_train_data, "val_data": p1_val_data, "hard_neg": True, "margin": 0.08, "cand_source": "p1_candidates"},
        {"name": "EXPERIMENT H: FULL PERSON 3 (Semantic Retrieval + All Feats + Margin)", "cols": idx_all_58, "train_data": union_train_data, "val_data": union_val_data, "hard_neg": True, "margin": 0.08, "cand_source": "union_candidates"},
    ]

    results = []

    for exp in experiments:
        print("\n" + "-" * 70)
        print(f"RUNNING {exp['name']}...")
        print("-" * 70)
        t0 = time.time()

        X_train, y_train = prepare_train_slice(exp["train_data"], exp["cols"], hard_negatives=exp["hard_neg"])
        by_s1_val = prepare_val_slice(exp["val_data"], exp["cols"])

        print(f"Training dataset: {len(X_train):,} pairs ({sum(y_train):,} positives, {len(y_train)-sum(y_train):,} negatives)")

        model = CatBoostClassifier(
            iterations=400,
            depth=7,
            learning_rate=0.08,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            thread_count=4,
            verbose=False
        )
        model.fit(X_train, y_train)

        best_t, m = evaluate_predictions(model, by_s1_val, gt, val_s1_list, margin=exp["margin"])
        elapsed = time.time() - t0

        cand_count = sum(len(c) for c in by_s1_val.values())
        print(f"Result for {exp['name']}:")
        print(f"  Macro F0.5 : {m['macro_f05']:.5f} (Threshold: {best_t:.2f})")
        print(f"  Precision  : {m['precision']:.5f}")
        print(f"  Recall     : {m['recall']:.5f}")
        print(f"  TP: {m['true_positives']:,} | FP: {m['false_positives']:,} | FN: {m['false_negatives']:,} | Singleton Errs: {m['singleton_false_matches']}")
        print(f"  Val Candidates: {cand_count:,} ({cand_count/len(val_s1_list):.1f}/S1) | Elapsed: {elapsed:.1f}s")

        res_record = {
            "experiment": exp["name"],
            "features_used": len(exp["cols"]),
            "candidate_source": exp["cand_source"],
            "macro_f05": m["macro_f05"],
            "precision": m["precision"],
            "recall": m["recall"],
            "true_positives": m["true_positives"],
            "false_positives": m["false_positives"],
            "false_negatives": m["false_negatives"],
            "singleton_false_matches": m["singleton_false_matches"],
            "optimal_threshold": best_t,
            "margin": exp["margin"],
            "hard_negatives": exp["hard_neg"],
            "val_candidates": cand_count,
            "candidates_per_s1": round(cand_count / len(val_s1_list), 2),
            "runtime_seconds": round(elapsed, 2)
        }
        results.append(res_record)

        if exp["name"].startswith("EXPERIMENT H"):
            champion_model_path = ARTIFACTS_P3 / "catboost_person3_matcher.cbm"
            model.save_model(str(champion_model_path))
            print(f"Saved Person 3 champion model to {champion_model_path}")

    # Save ablation results
    with open(ARTIFACTS_P3 / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote ablation results to {ARTIFACTS_P3 / 'ablation_results.json'}")

    best_exp = max(results, key=lambda x: x["macro_f05"])
    with open(ARTIFACTS_P3 / "validation_metrics.json", "w") as f:
        json.dump(best_exp, f, indent=2)
    print(f"Wrote best validation metrics to {ARTIFACTS_P3 / 'validation_metrics.json'}")

    print("\n" + "=" * 90)
    print("FINAL ABLATION TABLE")
    print("=" * 90)
    print(f"{'Experiment':<50} | {'F0.5':<8} | {'Prec':<8} | {'Rec':<8} | {'Thresh':<6} | {'Cands/S1':<8} | {'Time (s)':<8}")
    print("-" * 90)
    for r in results:
        print(f"{r['experiment']:<50} | {r['macro_f05']:<8.5f} | {r['precision']:<8.5f} | {r['recall']:<8.5f} | {r['optimal_threshold']:<6.2f} | {r['candidates_per_s1']:<8.1f} | {r['runtime_seconds']:<8.1f}")
    print("=" * 90)

if __name__ == "__main__":
    run_ablations()
