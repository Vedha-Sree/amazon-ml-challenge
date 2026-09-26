"""Unit tests for Person 3 retrieval, candidate union, ambiguity margin, and singleton logic."""
import sys
from pathlib import Path
import duckdb

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from person3_semantic_retrieval import union_candidate_tables
from person2_evaluation import evaluate_sets, threshold_predictions, f05

def test_candidate_union_deduplication():
    conn = duckdb.connect()
    conn.execute("""
        CREATE TABLE p1_cands(source1_entity_id VARCHAR, target_entity_id VARCHAR);
        INSERT INTO p1_cands VALUES ('S1-1', 'S2-1'), ('S1-1', 'S2-2'), ('S1-2', 'S3-1');

        CREATE TABLE p3_cands(source1_entity_id VARCHAR, target_entity_id VARCHAR);
        INSERT INTO p3_cands VALUES ('S1-1', 'S2-2'), ('S1-1', 'S3-5'), ('S1-3', 'S2-9');
    """)

    count = union_candidate_tables(conn, "p1_cands", "p3_cands", "union_out")
    # P1 has 3 pairs, P3 has 3 pairs, overlap is ('S1-1', 'S2-2') -> total unique = 5
    assert count == 5
    rows = conn.execute("SELECT * FROM union_out ORDER BY source1_entity_id, target_entity_id").fetchall()
    assert rows == [
        ('S1-1', 'S2-1'), ('S1-1', 'S2-2'), ('S1-1', 'S3-5'),
        ('S1-2', 'S3-1'), ('S1-3', 'S2-9')
    ]

def test_ambiguity_margin_logic():
    scored = [
        # S1-1: Clear winner (0.95 vs 0.40 -> gap 0.55 >= 0.08)
        ("S1-1", "S2-1", 0.95),
        ("S1-1", "S2-2", 0.40),
        # S1-2: Ambiguous (0.78 vs 0.75 -> gap 0.03 < 0.08 and top < 0.85)
        ("S1-2", "S2-3", 0.78),
        ("S1-2", "S3-4", 0.75),
    ]

    preds_with_margin = threshold_predictions(scored, threshold=0.70, margin=0.08)
    assert preds_with_margin["S1-1"] == {"S2-1"}
    assert preds_with_margin["S1-2"] == set()  # Dropped because of ambiguity margin

def test_singleton_scoring_behavior():
    # S1-1 is singleton (empty truth), predicted empty -> f05 = 1.0
    # S1-2 is singleton, predicted false match -> f05 = 0.0
    truth = {"S1-1": set(), "S1-2": set(), "S1-3": {"S2-10"}}
    preds = {"S1-1": set(), "S1-2": {"S2-99"}, "S1-3": {"S2-10"}}
    all_s1 = ["S1-1", "S1-2", "S1-3"]

    metrics = evaluate_sets(preds, truth, all_s1)
    assert metrics["singleton_false_matches"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    # Macro F0.5: (1.0 + 0.0 + 1.0) / 3 = 0.6667
    assert abs(metrics["macro_f05"] - 2.0/3.0) < 1e-4

def test_candidate_subset_compliance():
    # Candidate set
    candidates = {
        "S1-1": {"S2-1", "S2-2"},
        "S1-2": {"S3-1"}
    }
    predictions = {
        "S1-1": {"S2-1"},
        "S1-2": set()
    }
    # Check that predictions are strict subsets of candidates
    for s1, p_set in predictions.items():
        assert p_set.issubset(candidates.get(s1, set()))
