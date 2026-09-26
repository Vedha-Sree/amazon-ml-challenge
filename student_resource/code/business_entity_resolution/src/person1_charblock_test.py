from pathlib import Path
import gc

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work" / "person1"
TRAIN = ROOT / "dataset" / "train"

SAMPLE_S1 = 100_000
TOP_K = 20

S1_FILE = WORK / "train_source1.parquet"
S2_FILE = WORK / "train_source2.parquet"
S3_FILE = WORK / "train_source3.parquet"
GT_FILE = TRAIN / "train_ground_truth.tsv"


def load_gt():

    gt = pl.read_csv(
        GT_FILE,
        separator="\t",
        columns=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    return gt


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
            [
                "source1_entity_id",
                "target_entity_id",
            ]
        )
    )


def run_test(
    s1,
    target,
    gt_pairs,
    target_name,
):

    print("\n" + "=" * 70)
    print(f"CHARACTER BLOCK TEST: S1 -> {target_name}")
    print("=" * 70)

    # ---------------------------------------------------------
    # Sample S1
    # ---------------------------------------------------------

    sample = (
        s1
        .sample(
            n=min(SAMPLE_S1, s1.height),
            seed=42,
        )
        .select(
            [
                "entity_id",
                "name_norm",
            ]
        )
        .filter(
            pl.col("name_norm") != ""
        )
    )

    print(
        f"S1 sample: {sample.height:,}"
    )

    # ---------------------------------------------------------
    # Target names
    # ---------------------------------------------------------

    target_names = (
        target
        .select(
            [
                "entity_id",
                "name_norm",
            ]
        )
        .filter(
            pl.col("name_norm") != ""
        )
    )

    print(
        f"Target names: {target_names.height:,}"
    )

    # ---------------------------------------------------------
    # Build character TF-IDF
    # ---------------------------------------------------------

    print("\nBuilding character TF-IDF...")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=1_500_000,
        dtype=np.float32,
        sublinear_tf=True,
    )

    target_matrix = vectorizer.fit_transform(
        target_names["name_norm"].to_list()
    )

    print(
        f"Target matrix shape: {target_matrix.shape}"
    )

    print(
        f"Target matrix memory: "
        f"{target_matrix.data.nbytes / 1024**2:.1f} MB"
    )

    # ---------------------------------------------------------
    # Nearest-neighbor index
    # ---------------------------------------------------------

    print("\nBuilding sparse nearest-neighbor index...")

    nn = NearestNeighbors(
        n_neighbors=TOP_K,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    nn.fit(target_matrix)

    # ---------------------------------------------------------
    # Query S1 in batches
    # ---------------------------------------------------------

    BATCH = 5_000

    target_ids = target_names["entity_id"].to_list()
    sample_names = sample["name_norm"].to_list()
    sample_ids = sample["entity_id"].to_list()

    hits = 0
    total = 0

    result_rows = []

    print(
        f"\nQuerying {len(sample_names):,} S1 names..."
    )

    for start in range(
        0,
        len(sample_names),
        BATCH,
    ):

        end = min(
            start + BATCH,
            len(sample_names),
        )

        batch_names = sample_names[start:end]

        query_matrix = vectorizer.transform(
            batch_names
        )

        distances, indices = nn.kneighbors(
            query_matrix,
            return_distance=True,
        )

        for local_i in range(
            len(batch_names)
        ):

            s1_id = sample_ids[start + local_i]

            candidates = [
                target_ids[idx]
                for idx in indices[local_i]
            ]

            result_rows.extend(
                (
                    s1_id,
                    candidate,
                )
                for candidate in candidates
            )

        if start % 25_000 == 0:

            print(
                f"  processed {end:,} / "
                f"{len(sample_names):,}"
            )

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    candidates = pl.DataFrame(
        result_rows,
        schema=[
            "source1_entity_id",
            "target_entity_id",
        ],
        orient="row",
    )

    # Only evaluate S1 records included in our sample.
    sample_gt = (
        gt_pairs
        .join(
            sample.select(
                pl.col("entity_id")
                .alias("source1_entity_id")
            ),
            on="source1_entity_id",
            how="inner",
        )
    )

    hits_df = (
        candidates
        .join(
            sample_gt,
            on=[
                "source1_entity_id",
                "target_entity_id",
            ],
            how="inner",
        )
        .unique()
    )

    found = hits_df.height
    total = sample_gt.height

    recall = (
        found / total
        if total
        else 0.0
    )

    print("\n" + "-" * 70)
    print("RESULT")
    print("-" * 70)

    print(
        f"Sample S1 records: {sample.height:,}"
    )

    print(
        f"Ground-truth links in sample: {total:,}"
    )

    print(
        f"True links retrieved: {found:,}"
    )

    print(
        f"Top-{TOP_K} character recall: "
        f"{recall * 100:.4f}%"
    )

    print(
        f"Candidate pairs generated: "
        f"{candidates.height:,}"
    )

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    del vectorizer
    del target_matrix
    del nn
    del candidates
    del hits_df

    gc.collect()


def main():

    print("=" * 70)
    print("PERSON 1 — CHARACTER BLOCK TEST")
    print("=" * 70)

    gt = load_gt()

    gt_s2 = expand_gt(
        gt,
        "S2-",
    )

    gt_s3 = expand_gt(
        gt,
        "S3-",
    )

    del gt
    gc.collect()

    s1 = pl.read_parquet(
        S1_FILE,
        columns=[
            "entity_id",
            "name_norm",
        ],
    )

    s2 = pl.read_parquet(
        S2_FILE,
        columns=[
            "entity_id",
            "name_norm",
        ],
    )

    s3 = pl.read_parquet(
        S3_FILE,
        columns=[
            "entity_id",
            "name_norm",
        ],
    )

    # Test S2 first.
    run_test(
        s1,
        s2,
        gt_s2,
        "S2",
    )

    # Then S3.
    run_test(
        s1,
        s3,
        gt_s3,
        "S3",
    )

    print("\nDONE")


if __name__ == "__main__":
    main()