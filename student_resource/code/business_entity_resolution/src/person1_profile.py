from pathlib import Path
import polars as pl


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "dataset"

FILES = {
    "train_source1": DATASET / "train" / "train_source1.tsv",
    "train_source2": DATASET / "train" / "train_source2.tsv",
    "train_source3": DATASET / "train" / "train_source3.tsv",
    "ground_truth": DATASET / "train" / "train_ground_truth.tsv",
    "test_source1": DATASET / "test" / "test_source1.tsv",
    "test_source2": DATASET / "test" / "test_source2.tsv",
    "test_source3": DATASET / "test" / "test_source3.tsv",
}


def inspect_file(name: str, path: Path) -> None:
    print("\n" + "=" * 70)
    print(name)
    print(path)
    print("=" * 70)

    if not path.exists():
        print("MISSING")
        return

    print(f"File size: {path.stat().st_size / (1024**3):.3f} GB")

    df = pl.read_csv(
        path,
        separator="\t",
        n_rows=5,
        infer_schema_length=1000,
        ignore_errors=False,
    )

    print("Columns:")
    for col, dtype in zip(df.columns, df.dtypes):
        print(f"  {col}: {dtype}")

    print("\nFirst 5 rows:")
    print(df)


def main():
    print("Person-1 dataset profiling")
    print(f"Root: {ROOT}")

    for name, path in FILES.items():
        inspect_file(name, path)


if __name__ == "__main__":
    main()