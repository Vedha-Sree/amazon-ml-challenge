from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parent

FILES = [
    ROOT / "dataset" / "train" / "train_source1.tsv",
    ROOT / "dataset" / "train" / "train_source2.tsv",
    ROOT / "dataset" / "train" / "train_source3.tsv",
    ROOT / "dataset" / "train" / "train_ground_truth.tsv",
    ROOT / "dataset" / "test" / "test_source1.tsv",
    ROOT / "dataset" / "test" / "test_source2.tsv",
    ROOT / "dataset" / "test" / "test_source3.tsv",
]


def inspect_file(path: Path):
    print("\n" + "=" * 100)
    print(f"FILE: {path}")
    print("=" * 100)

    df = pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=10000,
    )

    print(f"ROWS: {df.height:,}")
    print(f"COLUMNS: {df.width}")
    print(f"COLUMN NAMES: {df.columns}")

    print("\nDATA TYPES:")
    print(df.schema)

    print("\nNULL COUNTS:")
    print(df.null_count())

    print("\nFIRST 3 ROWS:")
    print(df.head(3))


def main():
    print("=" * 100)
    print("ML CHALLENGE 2026 - PERSON 1 DATA PROFILER")
    print("=" * 100)

    for path in FILES:
        if not path.exists():
            print(f"\nMISSING FILE: {path}")
            continue

        inspect_file(path)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()