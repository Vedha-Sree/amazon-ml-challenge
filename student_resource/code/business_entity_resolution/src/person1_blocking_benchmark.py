from pathlib import Path
import csv
import gc

import polars as pl


ROOT = Path(__file__).resolve().parents[3]

WORK = ROOT / "work" / "person1"
TRAIN = ROOT / "dataset" / "train"

S1_FILE = WORK / "train_source1.parquet"
S2_FILE = WORK / "train_source2.parquet"
S3_FILE = WORK / "train_source3.parquet"
GT_FILE = TRAIN / "train_ground_truth.tsv"

REPORT_FILE = WORK / "blocking_report.txt"


# ============================================================
# Ground truth
# ============================================================

def load_ground_truth():
    """
    Ground truth is loaded as a Polars table.

    We keep the matched IDs as a comma-separated string here.
    We do NOT construct a giant Python dictionary.
    """

    print("\nLoading ground truth...")

    gt = pl.read_csv(
        GT_FILE,
        separator="\t",
        columns=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
        infer_schema_length=10000,
    )

    print(f"Ground-truth S1 rows: {gt.height:,}")

    return gt


# ============================================================
# Data loading
# ============================================================

def load_s1():
    print("\nLoading S1...")

    return pl.read_parquet(
        S1_FILE,
        columns=[
            "entity_id",
            "country",
            "name_norm",
            "name_compact",
            "name_tokens",
            "address_norm",
            "address_compact",
            "address_tokens",
        ],
    )


def load_target(path):
    print(f"\nLoading target: {path.name}")

    return pl.read_parquet(
        path,
        columns=[
            "entity_id",
            "country",
            "name_norm",
            "name_compact",
            "name_tokens",
            "address_norm",
            "address_compact",
            "address_tokens",
        ],
    )


# ============================================================
# Ground-truth expansion
# ============================================================

def expand_ground_truth(gt, prefix):
    """
    Convert only the relevant source IDs into rows:

        source1_entity_id | target_entity_id

    This is done with Polars rather than a Python set/dict.
    """

    print(f"Expanding ground truth for {prefix}")

    result = (
        gt
        .with_columns(
            pl.col("matched_entity_ids")
            .fill_null("")
            .str.split(",")
            .alias("target_entity_id")
        )
        .explode("target_entity_id")
        .with_columns(
            pl.col("target_entity_id").str.strip_chars()
        )
        .filter(
            pl.col("target_entity_id").str.starts_with(prefix)
        )
        .select(
            [
                "source1_entity_id",
                "target_entity_id",
            ]
        )
    )

    print(f"Positive {prefix} links: {result.height:,}")

    return result


# ============================================================
# Blocking benchmark
# ============================================================

def benchmark_rule(
    s1,
    target,
    gt_pairs,
    target_name,
    rule_name,
    s1_key,
    target_key,
):
    """
    Exact blocking using a Polars join.

    IMPORTANT:
    We process one rule at a time.

    We do NOT create a Python dictionary.
    """

    print("\n" + "-" * 70)
    print(f"{target_name} / {rule_name}")
    print("-" * 70)

    # Only keep rows that have a usable blocking key.
    left = (
        s1
        .select(
            [
                pl.col("entity_id").alias("source1_entity_id"),
                pl.col(s1_key).alias("block_key"),
            ]
        )
        .filter(
            pl.col("block_key").is_not_null()
            & (pl.col("block_key") != "")
        )
    )

    right = (
        target
        .select(
            [
                pl.col("entity_id").alias("target_entity_id"),
                pl.col(target_key).alias("block_key"),
            ]
        )
        .filter(
            pl.col("block_key").is_not_null()
            & (pl.col("block_key") != "")
        )
    )

    print(f"S1 usable keys:     {left.height:,}")
    print(f"Target usable keys: {right.height:,}")

    # Inner join creates candidate pairs.
    candidates = (
        left
        .join(
            right,
            on="block_key",
            how="inner",
        )
        .select(
            [
                "source1_entity_id",
                "target_entity_id",
            ]
        )
    )

    candidate_count = candidates.height

    print(f"Candidate pairs: {candidate_count:,}")

    # Count S1 records that received at least one candidate.
    s1_with_candidates = (
        candidates
        .select("source1_entity_id")
        .unique()
        .height
    )

    print(
        f"S1 with candidates: {s1_with_candidates:,}"
    )

    # Intersect candidates with ground truth.
    hits = (
        candidates
        .join(
            gt_pairs,
            on=[
                "source1_entity_id",
                "target_entity_id",
            ],
            how="inner",
        )
        .unique()
    )

    true_matches_found = hits.height
    true_matches_total = gt_pairs.height

    recall = (
        true_matches_found / true_matches_total
        if true_matches_total
        else 1.0
    )

    avg_candidates = candidate_count / s1.height

    print(
        f"True matches found: {true_matches_found:,}"
    )

    print(
        f"True matches total: {true_matches_total:,}"
    )

    print(
        f"Blocking recall:    {recall * 100:.4f}%"
    )

    print(
        f"Avg candidates/S1:   {avg_candidates:.2f}"
    )

    # Free the potentially large candidate table.
    del candidates
    del hits
    del left
    del right

    gc.collect()

    return {
        "target": target_name,
        "rule": rule_name,
        "candidate_pairs": candidate_count,
        "s1_with_candidates": s1_with_candidates,
        "true_matches_found": true_matches_found,
        "true_matches_total": true_matches_total,
        "recall": recall,
        "avg_candidates": avg_candidates,
    }


# ============================================================
# Combined rules
# ============================================================

def benchmark_combined(
    s1,
    target,
    gt_pairs,
    target_name,
    rules,
):
    """
    Union several blocking rules.

    Each rule is generated separately and written to disk.
    Then we combine the candidate files.

    This prevents a huge all-in-memory union.
    """

    candidate_files = []

    for rule_name, key in rules:

        print("\n" + "=" * 70)
        print(
            f"Generating candidates: "
            f"{target_name} / {rule_name}"
        )
        print("=" * 70)

        left = (
            s1
            .select(
                [
                    pl.col("entity_id")
                    .alias("source1_entity_id"),

                    pl.col(key)
                    .alias("block_key"),
                ]
            )
            .filter(
                pl.col("block_key").is_not_null()
                & (pl.col("block_key") != "")
            )
        )

        right = (
            target
            .select(
                [
                    pl.col("entity_id")
                    .alias("target_entity_id"),

                    pl.col(key)
                    .alias("block_key"),
                ]
            )
            .filter(
                pl.col("block_key").is_not_null()
                & (pl.col("block_key") != "")
            )
        )

        candidates = (
            left
            .join(
                right,
                on="block_key",
                how="inner",
            )
            .select(
                [
                    "source1_entity_id",
                    "target_entity_id",
                ]
            )
            .unique()
        )

        path = (
            WORK
            / f"tmp_{target_name}_{rule_name}.parquet"
        )

        candidates.write_parquet(
            path,
            compression="zstd",
        )

        print(
            f"Candidates: {candidates.height:,}"
        )

        print(
            f"Saved: {path.name}"
        )

        candidate_files.append(path)

        del candidates
        del left
        del right

        gc.collect()

    # Scan candidate files lazily.
    scans = [
        pl.scan_parquet(path)
        for path in candidate_files
    ]

    union = (
        pl.concat(scans)
        .unique()
    )

    print("\nCombining blocking rules...")

    # Materialize only the final union.
    combined = union.collect()

    candidate_count = combined.height

    print(
        f"Combined candidate pairs: "
        f"{candidate_count:,}"
    )

    s1_with_candidates = (
        combined
        .select("source1_entity_id")
        .unique()
        .height
    )

    hits = (
        combined
        .join(
            gt_pairs,
            on=[
                "source1_entity_id",
                "target_entity_id",
            ],
            how="inner",
        )
        .unique()
    )

    found = hits.height
    total = gt_pairs.height

    recall = found / total if total else 1.0

    avg_candidates = candidate_count / s1.height

    print(
        f"S1 with candidates: {s1_with_candidates:,}"
    )

    print(
        f"True matches found: {found:,}"
    )

    print(
        f"True matches total: {total:,}"
    )

    print(
        f"Combined recall: {recall * 100:.4f}%"
    )

    print(
        f"Average candidates/S1: {avg_candidates:.2f}"
    )

    # Save benchmark union for inspection.
    output = (
        WORK
        / f"train_candidates_{target_name}.parquet"
    )

    combined.write_parquet(
        output,
        compression="zstd",
    )

    print(f"Saved combined candidates: {output}")

    # Remove temporary rule files.
    for path in candidate_files:
        try:
            path.unlink()
        except OSError:
            pass

    del combined
    del hits

    gc.collect()

    return {
        "target": target_name,
        "rule": "COMBINED",
        "candidate_pairs": candidate_count,
        "s1_with_candidates": s1_with_candidates,
        "true_matches_found": found,
        "true_matches_total": total,
        "recall": recall,
        "avg_candidates": avg_candidates,
    }


# ============================================================
# Report
# ============================================================

def write_report(results):
    lines = []

    lines.append("PERSON 1 BLOCKING BENCHMARK")
    lines.append("=" * 70)
    lines.append("")

    for r in results:

        lines.append(
            f"{r['target']} / {r['rule']}"
        )

        lines.append(
            f"  candidate pairs: "
            f"{r['candidate_pairs']:,}"
        )

        lines.append(
            f"  S1 with candidates: "
            f"{r['s1_with_candidates']:,}"
        )

        lines.append(
            f"  true matches found: "
            f"{r['true_matches_found']:,}"
        )

        lines.append(
            f"  true matches total: "
            f"{r['true_matches_total']:,}"
        )

        lines.append(
            f"  recall: "
            f"{r['recall'] * 100:.4f}%"
        )

        lines.append(
            f"  avg candidates/S1: "
            f"{r['avg_candidates']:.2f}"
        )

        lines.append("")

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    print(
        REPORT_FILE.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("PERSON 1 — MEMORY-SAFE BLOCKING BENCHMARK")
    print("=" * 70)

    gt = load_ground_truth()

    gt_s2 = expand_ground_truth(
        gt,
        "S2-",
    )

    gt_s3 = expand_ground_truth(
        gt,
        "S3-",
    )

    # We no longer need the original GT table.
    del gt
    gc.collect()

    s1 = load_s1()

    s2 = load_target(S2_FILE)
    s3 = load_target(S3_FILE)

    results = []

    # --------------------------------------------------------
    # S2
    # --------------------------------------------------------

    s2_rules = [
        ("exact_name", "name_norm"),
        ("compact_name", "name_compact"),
        ("exact_address", "address_norm"),
        ("compact_address", "address_compact"),
    ]

    for rule_name, key in s2_rules:

        result = benchmark_rule(
            s1,
            s2,
            gt_s2,
            "S2",
            rule_name,
            key,
            key,
        )

        results.append(result)

    # --------------------------------------------------------
    # S3
    # --------------------------------------------------------

    s3_rules = [
        ("exact_name", "name_norm"),
        ("compact_name", "name_compact"),
        ("exact_address", "address_norm"),
        ("compact_address", "address_compact"),
    ]

    for rule_name, key in s3_rules:

        result = benchmark_rule(
            s1,
            s3,
            gt_s3,
            "S3",
            rule_name,
            key,
            key,
        )

        results.append(result)

    # --------------------------------------------------------
    # Combined blocking
    # --------------------------------------------------------

    print("\n" + "#" * 70)
    print("COMBINED BLOCKING")
    print("#" * 70)

    combined_s2 = benchmark_combined(
        s1,
        s2,
        gt_s2,
        "S2",
        s2_rules,
    )

    results.append(combined_s2)

    combined_s3 = benchmark_combined(
        s1,
        s3,
        gt_s3,
        "S3",
        s3_rules,
    )

    results.append(combined_s3)

    write_report(results)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()