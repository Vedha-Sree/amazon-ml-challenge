from pathlib import Path
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
NORMALIZED = ROOT / "artifacts" / "normalized"
GROUND_TRUTH = ROOT / "dataset" / "train" / "train_ground_truth.tsv"


def load_source1():
    return pl.read_parquet(
        NORMALIZED / "train_source1.parquet",
        columns=[
            "entity_id",
            "name_norm",
            "name_compact",
            "address_norm",
            "address_compact",
            "country_norm",
        ],
    )


def load_source2():
    return pl.read_parquet(
        NORMALIZED / "train_source2.parquet",
        columns=[
            "entity_id",
            "name_norm",
            "name_compact",
            "address_norm",
            "address_compact",
            "country_norm",
        ],
    )


def load_source3():
    files = sorted(
        NORMALIZED.glob("train_source3_part_*.parquet")
    )

    return pl.concat(
        [
            pl.read_parquet(
                f,
                columns=[
                    "entity_id",
                    "name_norm",
                    "name_compact",
                    "address_norm",
                    "address_compact",
                    "country_norm",
                ],
            )
            for f in files
        ],
        how="vertical",
    )


def build_name_index(df):
    return (
        df
        .filter(pl.col("name_norm") != "")
        .group_by("name_norm")
        .agg(
            pl.col("entity_id").alias("candidate_ids")
        )
    )


def build_address_index(df):
    return (
        df
        .filter(pl.col("address_norm") != "")
        .group_by("address_norm")
        .agg(
            pl.col("entity_id").alias("candidate_ids")
        )
    )


def main():

    print("=" * 100)
    print("PERSON 1 — EXACT BLOCK EVALUATION")
    print("=" * 100)

    print("\nLoading S1...")
    s1 = load_source1()

    print("Loading S2...")
    s2 = load_source2()

    print("Loading S3...")
    s3 = load_source3()

    print("\nLoaded:")
    print("S1:", f"{s1.height:,}")
    print("S2:", f"{s2.height:,}")
    print("S3:", f"{s3.height:,}")

    print("\nBuilding indexes...")

    s2_name = build_name_index(s2)
    s3_name = build_name_index(s3)

    s2_address = build_address_index(s2)
    s3_address = build_address_index(s3)

    print("\nIndex sizes:")
    print("S2 name keys:", f"{s2_name.height:,}")
    print("S3 name keys:", f"{s3_name.height:,}")
    print("S2 address keys:", f"{s2_address.height:,}")
    print("S3 address keys:", f"{s3_address.height:,}")

    print("\nTesting S1 exact-name coverage...")

    name_test = (
        s1
        .select(
            [
                "entity_id",
                "name_norm",
            ]
        )
        .join(
            s2_name,
            on="name_norm",
            how="left",
        )
        .rename(
            {
                "candidate_ids": "s2_name_candidates"
            }
        )
        .join(
            s3_name,
            on="name_norm",
            how="left",
        )
        .rename(
            {
                "candidate_ids": "s3_name_candidates"
            }
        )
    )

    print("\nName blocking candidate statistics:")

    print(
        name_test
        .select(
            [
                pl.col("s2_name_candidates")
                .list.len()
                .fill_null(0)
                .alias("s2_count"),

                pl.col("s3_name_candidates")
                .list.len()
                .fill_null(0)
                .alias("s3_count"),
            ]
        )
        .describe()
    )

    print("\nTesting exact-address coverage...")

    address_test = (
        s1
        .select(
            [
                "entity_id",
                "address_norm",
            ]
        )
        .join(
            s2_address,
            on="address_norm",
            how="left",
        )
        .rename(
            {
                "candidate_ids": "s2_address_candidates"
            }
        )
        .join(
            s3_address,
            on="address_norm",
            how="left",
        )
        .rename(
            {
                "candidate_ids": "s3_address_candidates"
            }
        )
    )

    print("\nAddress blocking candidate statistics:")

    print(
        address_test
        .select(
            [
                pl.col("s2_address_candidates")
                .list.len()
                .fill_null(0)
                .alias("s2_count"),

                pl.col("s3_address_candidates")
                .list.len()
                .fill_null(0)
                .alias("s3_count"),
            ]
        )
        .describe()
    )

    print("\n" + "=" * 100)
    print("EXACT BLOCK EVALUATION FINISHED")
    print("=" * 100)


if __name__ == "__main__":
    main()