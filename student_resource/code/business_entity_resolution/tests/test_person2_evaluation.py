import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from person2_evaluation import evaluate_sets, optimize_threshold, threshold_predictions


def test_singleton_and_macro_metric():
    m = evaluate_sets({"a": set(), "b": {"x"}}, {"a": set(), "b": {"x"}}, ["a", "b"])
    assert m["macro_f05"] == 1.0
    assert m["singleton_false_matches"] == 0


def test_threshold_optimization_is_s1_level():
    scored = [("a", "x", .95), ("a", "y", .20), ("b", "z", .40)]
    pred = threshold_predictions(scored, .5)
    assert pred["a"] == {"x"}
    assert pred["b"] == set()
    threshold, metrics = optimize_threshold(scored, {"a": {"x"}, "b": set()}, ["a", "b"])
    assert threshold >= .5
    assert metrics["macro_f05"] > 0
