from pathlib import Path
import unicodedata

import polars as pl


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "dataset"
WORK = ROOT / "work" / "person1"

TRAIN = DATASET / "train"
TEST = DATASET / "test"

WORK.mkdir(parents=True, exist_ok=True)


FILES = {
    "train_source1": TRAIN / "train_source1.tsv",
    "train_source2": TRAIN / "train_source2.tsv",
    "train_source3": TRAIN / "train_source3.tsv",
    "test_source1": TEST / "test_source1.tsv",
    "test_source2": TEST / "test_source2.tsv",
    "test_source3": TEST / "test_source3.tsv",
}


def normalize_unicode(value):
    if value is None:
        return ""

    return unicodedata.normalize("NFKC", str(value)).casefold()


def normalize_name(value):
    value = normalize_unicode(value)
    value = value.replace("&", " and ")

    value = "".join(
        ch if (ch.isalnum() or ch.isspace()) else " "
        for ch in value
    )

    return " ".join(value.split())


def compact_name(value):
    return normalize_name(value).replace(" ", "")


def name_tokens(value):
    value = normalize_name(value)

    if not value:
        return ""

    return " ".join(sorted(set(value.split())))


def normalize_address(value):
    value = normalize_unicode(value)

    if not value:
        return ""

    value = value.replace("&", " and ")
    value = value.replace("/", " ")
    value = value.replace("\\", " ")

    value = "".join(
        ch if (ch.isalnum() or ch.isspace()) else " "
        for ch in value
    )

    return " ".join(value.split())


def compact_address(value):
    return normalize_address(value).replace(" ", "")


def address_tokens(value):
    value = normalize_address(value)

    if not value:
        return ""

    return " ".join(sorted(set(value.split())))


def normalize_columns(df):
    return df.with_columns(
        [
            pl.col("business_name")
            .map_elements(normalize_name, return_dtype=pl.String)
            .alias("name_norm"),

            pl.col("business_name")
            .map_elements(compact_name, return_dtype=pl.String)
            .alias("name_compact"),

            pl.col("business_name")
            .map_elements(name_tokens, return_dtype=pl.String)
            .alias("name_tokens"),

            pl.col("business_address")
            .map_elements(normalize_address, return_dtype=pl.String)
            .alias("address_norm"),

            pl.col("business_address")
            .map_elements(compact_address, return_dtype=pl.String)
            .alias("address_compact"),

            pl.col("business_address")
            .map_elements(address_tokens, return_dtype=pl.String)
            .alias("address_tokens"),
        ]
    )


def process_file(name, path):
    out = WORK / f"{name}.parquet"

    if out.exists() and out.stat().st_size > 0:
        print(f"\nSKIP — already exists: {out.name}")
        return

    print("\n" + "=" * 70)
    print(f"Processing {name}")
    print("=" * 70)
    print(f"Input: {path}")

    df = pl.read_csv(
        path,
        separator="\t",
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        infer_schema_length=10000,
        ignore_errors=False,
    )

    print(f"Rows: {df.height:,}")

    df = normalize_columns(df)

    source = name.split("_")[-1]

    df = df.with_columns(
        pl.lit(source).alias("source")
    )

    df = df.select(
        [
            "entity_id",
            "source",
            "country",
            "business_name",
            "business_address",
            "name_norm",
            "name_compact",
            "name_tokens",
            "address_norm",
            "address_compact",
            "address_tokens",
        ]
    )

    df.write_parquet(
        out,
        compression="zstd",
        compression_level=3,
    )

    print(f"Output: {out}")
    print(f"Output size: {out.stat().st_size / (1024**2):.1f} MB")


def main():
    print("=" * 70)
    print("PERSON 1 — RESUMABLE PREPARATION")
    print("=" * 70)

    for name, path in FILES.items():
        process_file(name, path)

    print("\n" + "=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()