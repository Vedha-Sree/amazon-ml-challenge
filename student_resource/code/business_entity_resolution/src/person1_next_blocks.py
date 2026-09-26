from pathlib import Path
import gc

import polars as pl


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work" / "person1"
TRAIN = ROOT / "dataset" / "train"

S1_FILE = WORK / "train_source1.parquet"
S2_FILE = WORK / "train_source2.parquet"
S3_FILE = WORK / "train_source3.parquet"
GT_FILE = TRAIN / "train_ground_truth.tsv"


def load_gt():
    return pl.read_csv(
        GT_FILE,
        separator="\t",
        columns=["source1_entity_id", "matched_entity_ids"],
    )


def expand_gt(gt, prefix):
    return (
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
            pl.col("target_entity_id")
            .str.starts_with(prefix)
        )
        .select(
            ["source1_entity_id", "target_entity_id"]
        )
    )


def make_key_expr(key):
    if isinstance(key, tuple):
        return pl.concat_str(
            [
                pl.col(key[0]).fill_null(""),
                pl.lit("|"),
                pl.col(key[1]).fill_null(""),
            ]
        )
    return pl.col(key).fill_null("")


def benchmark(
    s1,
    target,
    gt_pairs,
    target_label,
    rule_name,
    key,
):
    print("\n" + "=" * 70)
    print(f"{target_label} / {rule_name}")
    print("=" * 70)

    left = (
        s1
        .select(
            [
                pl.col("entity_id")
                .alias("source1_entity_id"),
                make_key_expr(key).alias("block_key"),
            ]
        )
        .filter(
            pl.col("block_key") != ""
        )
    )

    right = (
        target
        .select(
            [
                pl.col("entity_id")
                .alias("target_entity_id"),
                make_key_expr(key).alias("block_key"),
            ]
        )
        .filter(
            pl.col("block_key") != ""
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
    )

    candidate_count = candidates.height

    print(
        f"Candidate pairs: {candidate_count:,}"
    )

    s1_with_candidates = (
        candidates
        .select("source1_entity_id")
        .unique()
        .height
    )

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

    found = hits.height
    total = gt_pairs.height

    recall = found / total if total else 1.0

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
        f"Recall: {recall * 100:.4f}%"
    )

    print(
        f"Avg candidates/S1: "
        f"{candidate_count / s1.height:.2f}"
    )

    del candidates
    del hits
    del left
    del right

    gc.collect()


def main():

    print("=" * 70)
    print("PERSON 1 — NEXT BLOCKING EXPERIMENTS")
    print("=" * 70)

    gt = load_gt()

    gt_s2 = expand_gt(gt, "S2-")
    gt_s3 = expand_gt(gt, "S3-")

    del gt
    gc.collect()

    s1 = pl.read_parquet(
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

    s2 = pl.read_parquet(
        S2_FILE,
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

    s3 = pl.read_parquet(
        S3_FILE,
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

    # --------------------------------------------------------
    # S2
    # --------------------------------------------------------

    tests_s2 = [
        ("name_tokens", "name_tokens"),
        ("address_tokens", "address_tokens"),
        ("country_name", ("country", "name_norm")),
        ("country_compact_name", ("country", "name_compact")),
        ("country_address", ("country", "address_norm")),
    ]

    for name, key in tests_s2:
        benchmark(
            s1,
            s2,
            gt_s2,
            "S2",
            name,
            key,
        )

    # --------------------------------------------------------
    # S3
    # --------------------------------------------------------

    tests_s3 = [
        ("name_tokens", "name_tokens"),
        ("address_tokens", "address_tokens"),
        ("country_name", ("country", "name_norm")),
        ("country_compact_name", ("country", "name_compact")),
        ("country_address", ("country", "address_norm")),
    ]

    for name, key in tests_s3:
        benchmark(
            s1,
            s3,
            gt_s3,
            "S3",
            name,
            key,
        )

    print("\nDONE")


if __name__ == "__main__":
    main()