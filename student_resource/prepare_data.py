from pathlib import Path
import polars as pl


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "normalized"
OUTPUT.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000


def normalize_expr(column: str):
    return (
        pl.col(column)
        .fill_null("")
        .str.normalize("NFKC")
        .str.to_lowercase()
        .str.replace_all(r"[^\p{L}\p{N}]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def transform(df: pl.DataFrame) -> pl.DataFrame:

    return (
        df.with_columns(
            [
                normalize_expr("business_name").alias("name_norm"),
                normalize_expr("business_address").alias("address_norm"),
                normalize_expr("country").alias("country_norm"),
            ]
        )
        .with_columns(
            [
                pl.col("name_norm")
                .str.replace_all(r"\s+", "")
                .alias("name_compact"),

                pl.col("address_norm")
                .str.replace_all(r"\s+", "")
                .alias("address_compact"),
            ]
        )
    )


def prepare_file(input_path: Path):

    output_path = OUTPUT / f"{input_path.stem}.parquet"

    print()
    print("=" * 100)
    print(f"INPUT : {input_path}")
    print(f"OUTPUT: {output_path}")
    print(f"CHUNK : {CHUNK_SIZE:,} rows")
    print("=" * 100)

    # Remove an old/incomplete output.
    if output_path.exists():
        output_path.unlink()

    reader = pl.read_csv_batched(
        input_path,
        separator="\t",
        infer_schema_length=10_000,
        null_values=["", "null", "NULL", "None"],
        batch_size=CHUNK_SIZE,
    )

    batch_number = 0
    total_rows = 0

    while True:

        batches = reader.next_batches(1)

        if not batches:
            break

        df = batches[0]

        if df.height == 0:
            break

        batch_number += 1
        total_rows += df.height

        print(
            f"Batch {batch_number:03d} | "
            f"rows={df.height:,} | "
            f"total={total_rows:,}"
        )

        transformed = transform(df)

        # Each batch gets its own Parquet file.
        batch_path = (
            OUTPUT
            / f"{input_path.stem}_part_{batch_number:04d}.parquet"
        )

        transformed.write_parquet(
            batch_path,
            compression="zstd",
        )

        del df
        del transformed

    print()
    print(f"FINISHED: {input_path.name}")
    print(f"TOTAL ROWS: {total_rows:,}")
    print(f"PARTS: {batch_number}")


def main():

    files = [
        ROOT / "dataset" / "train" / "train_source1.tsv",
        ROOT / "dataset" / "train" / "train_source2.tsv",
        ROOT / "dataset" / "train" / "train_source3.tsv",
    ]

    for path in files:

        if not path.exists():
            print(f"MISSING: {path}")
            continue

        # Only create chunked output for files that don't
        # already have a successful monolithic Parquet file.
        normal_output = OUTPUT / f"{path.stem}.parquet"

        if normal_output.exists():
            print(f"SKIPPING EXISTING: {normal_output.name}")
            continue

        prepare_file(path)

    print()
    print("=" * 100)
    print("NORMALIZATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()