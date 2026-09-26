from pathlib import Path
import gc
import re

import polars as pl
from rapidfuzz import fuzz


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work" / "person1"
TRAIN = ROOT / "dataset" / "train"
TEST = ROOT / "dataset" / "test"
OUTPUT = ROOT / "output"

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

MAX_FUZZY_BUCKET = 80
FUZZY_TOP_K = 8

# We deliberately use several compact signatures.
# These are blocking keys, NOT match decisions.
SIGNATURE_LENGTHS = (3, 4, 5)


# ============================================================
# FILES
# ============================================================

TRAIN_S1 = WORK / "train_source1.parquet"
TRAIN_S2 = WORK / "train_source2.parquet"
TRAIN_S3 = WORK / "train_source3.parquet"

TEST_S1 = WORK / "test_source1.parquet"
TEST_S2 = WORK / "test_source2.parquet"
TEST_S3 = WORK / "test_source3.parquet"

GT_FILE = TRAIN / "train_ground_truth.tsv"


# ============================================================
# BASIC HELPERS
# ============================================================

def add_signatures(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create bounded approximate-name blocking signatures.

    We use first/last character sequences rather than a global
    TF-IDF index. This keeps memory bounded.
    """

    exprs = []

    for n in SIGNATURE_LENGTHS:
        exprs.extend(
            [
                pl.col("name_compact")
                .str.slice(0, n)
                .alias(f"name_prefix_{n}"),

                pl.col("name_compact")
                .str.slice(-n, n)
                .alias(f"name_suffix_{n}"),
            ]
        )

    return df.with_columns(exprs)


def load_entity_file(path: Path) -> pl.DataFrame:
    return pl.read_parquet(
        path,
        columns=[
            "entity_id",
            "country",
            "business_name",
            "business_address",
            "name_norm",
            "name_compact",
            "name_tokens",
            "address_norm",
            "address_compact",
            "address_tokens",
        ],
    )


# ============================================================
# EXACT BLOCK
# ============================================================

def exact_block(
    s1: pl.DataFrame,
    target: pl.DataFrame,
    key: str,
    rule_name: str,
) -> pl.DataFrame:

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

    result = (
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
        .with_columns(
            pl.lit(rule_name).alias("block_rule")
        )
    )

    return result


# ============================================================
# FUZZY NAME BLOCK
# ============================================================

def fuzzy_name_block(
    s1: pl.DataFrame,
    target: pl.DataFrame,
) -> pl.DataFrame:

    print("\nBuilding fuzzy name buckets...")

    s1_sig = add_signatures(
        s1.select(
            [
                "entity_id",
                "name_compact",
                "name_norm",
            ]
        )
    )

    target_sig = add_signatures(
        target.select(
            [
                "entity_id",
                "name_compact",
                "name_norm",
            ]
        )
    )

    candidate_frames = []

    for n in SIGNATURE_LENGTHS:

        for side in ("prefix", "suffix"):

            key = f"name_{side}_{n}"

            print(
                f"  Fuzzy block: {key}"
            )

            left = (
                s1_sig
                .select(
                    [
                        pl.col("entity_id")
                        .alias("source1_entity_id"),

                        pl.col("name_norm")
                        .alias("source1_name"),

                        pl.col("name_compact")
                        .alias("source1_compact"),

                        pl.col(key)
                        .alias("block_key"),
                    ]
                )
                .filter(
                    pl.col("block_key") != ""
                )
            )

            right = (
                target_sig
                .select(
                    [
                        pl.col("entity_id")
                        .alias("target_entity_id"),

                        pl.col("name_norm")
                        .alias("target_name"),

                        pl.col("name_compact")
                        .alias("target_compact"),

                        pl.col(key)
                        .alias("block_key"),
                    ]
                )
                .filter(
                    pl.col("block_key") != ""
                )
            )

            # Count bucket sizes first.
            bucket_sizes = (
                right
                .group_by("block_key")
                .len()
                .filter(
                    pl.col("len")
                    <= MAX_FUZZY_BUCKET
                )
                .select("block_key")
            )

            right = right.join(
                bucket_sizes,
                on="block_key",
                how="inner",
            )

            left = left.join(
                bucket_sizes,
                on="block_key",
                how="inner",
            )

            pairs = (
                left
                .join(
                    right,
                    on="block_key",
                    how="inner",
                )
            )

            print(
                f"    bucket candidates: "
                f"{pairs.height:,}"
            )

            if pairs.height == 0:
                del pairs
                del left
                del right
                del bucket_sizes
                gc.collect()
                continue

            # RapidFuzz is applied only inside bounded blocks.
            #
            # We process the Polars result row-by-row so we don't
            # create a giant Python object collection.
            rows = pairs.iter_rows(named=True)

            output = []

            for row in rows:

                score = fuzz.ratio(
                    row["source1_compact"],
                    row["target_compact"],
                )

                if score >= 70:

                    output.append(
                        (
                            row["source1_entity_id"],
                            row["target_entity_id"],
                            "fuzzy_name",
                        )
                    )

            if output:

                frame = pl.DataFrame(
                    output,
                    schema=[
                        "source1_entity_id",
                        "target_entity_id",
                        "block_rule",
                    ],
                    orient="row",
                )

                candidate_frames.append(frame)

            del pairs
            del left
            del right
            del bucket_sizes

            gc.collect()

    if not candidate_frames:
        return pl.DataFrame(
            schema={
                "source1_entity_id": pl.String,
                "target_entity_id": pl.String,
                "block_rule": pl.String,
            }
        )

    return (
        pl.concat(candidate_frames)
        .unique()
    )


# ============================================================
# TRAINING GROUND TRUTH
# ============================================================

def load_gt():

    gt = pl.read_csv(
        GT_FILE,
        separator="\t",
        columns=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

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
            pl.col("target_entity_id") != ""
        )
        .select(
            [
                "source1_entity_id",
                "target_entity_id",
            ]
        )
    )


# ============================================================
# TRAINING EVALUATION
# ============================================================

def evaluate(
    candidates: pl.DataFrame,
    gt: pl.DataFrame,
    label: str,
):

    print("\n" + "=" * 70)
    print(f"EVALUATION: {label}")
    print("=" * 70)

    candidates = (
        candidates
        .select(
            [
                "source1_entity_id",
                "target_entity_id",
            ]
        )
        .unique()
    )

    hits = (
        candidates
        .join(
            gt,
            on=[
                "source1_entity_id",
                "target_entity_id",
            ],
            how="inner",
        )
        .unique()
    )

    total = gt.height
    found = hits.height

    recall = (
        found / total
        if total
        else 0.0
    )

    print(
        f"Candidate pairs: {candidates.height:,}"
    )

    print(
        f"True links: {total:,}"
    )

    print(
        f"Recovered: {found:,}"
    )

    print(
        f"Recall: {recall * 100:.4f}%"
    )

    return recall


# ============================================================
# BUILD ONE SOURCE PAIR
# ============================================================

def build_candidates(
    s1: pl.DataFrame,
    target: pl.DataFrame,
    label: str,
):

    print("\n" + "#" * 70)
    print(f"BUILDING {label}")
    print("#" * 70)

    frames = []

    # --------------------------------------------------------
    # Cheap high-precision blocks
    # --------------------------------------------------------

    blocks = [
        ("exact_name", "name_norm"),
        ("compact_name", "name_compact"),
        ("name_tokens", "name_tokens"),
        ("exact_address", "address_norm"),
        ("compact_address", "address_compact"),
        ("address_tokens", "address_tokens"),
    ]

    for rule, key in blocks:

        print(
            f"\nRunning {rule}..."
        )

        result = exact_block(
            s1,
            target,
            key,
            rule,
        )

        print(
            f"  candidates: "
            f"{result.height:,}"
        )

        frames.append(result)

        del result
        gc.collect()

    # --------------------------------------------------------
    # Approximate name block
    # --------------------------------------------------------

    print("\nRunning bounded fuzzy-name retrieval...")

    fuzzy = fuzzy_name_block(
        s1,
        target,
    )

    print(
        f"Fuzzy candidates: "
        f"{fuzzy.height:,}"
    )

    frames.append(fuzzy)

    # --------------------------------------------------------
    # Final union
    # --------------------------------------------------------

    print("\nCreating final candidate set...")

    combined = (
        pl.concat(frames)
        .unique(
            subset=[
                "source1_entity_id",
                "target_entity_id",
            ]
        )
    )

    print(
        f"FINAL {label} candidates: "
        f"{combined.height:,}"
    )

    return combined


# ============================================================
# TRAIN MODE
# ============================================================

def training_mode():

    print("\n" + "=" * 70)
    print("TRAINING CANDIDATE BENCHMARK")
    print("=" * 70)

    s1 = load_entity_file(
        TRAIN_S1
    )

    s2 = load_entity_file(
        TRAIN_S2
    )

    s3 = load_entity_file(
        TRAIN_S3
    )

    gt = load_gt()

    gt_s2 = gt.filter(
        pl.col("target_entity_id")
        .str.starts_with("S2-")
    )

    gt_s3 = gt.filter(
        pl.col("target_entity_id")
        .str.starts_with("S3-")
    )

    candidates_s2 = build_candidates(
        s1,
        s2,
        "S1 -> S2",
    )

    evaluate(
        candidates_s2,
        gt_s2,
        "S1 -> S2",
    )

    del candidates_s2
    del s2
    gc.collect()

    candidates_s3 = build_candidates(
        s1,
        s3,
        "S1 -> S3",
    )

    evaluate(
        candidates_s3,
        gt_s3,
        "S1 -> S3",
    )

    del candidates_s3
    del s3
    gc.collect()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PERSON 1 — CANDIDATE GENERATOR")
    print("=" * 70)

    training_mode()

    print("\n" + "=" * 70)
    print("PERSON 1 CANDIDATE GENERATOR FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()