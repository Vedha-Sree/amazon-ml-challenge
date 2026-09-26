from pathlib import Path
import duckdb


ROOT = Path(__file__).resolve().parents[1]

DB_PATH = ROOT / "artifacts" / "duckdb" / "person1.duckdb"
OUTPUT_DIR = ROOT / "artifacts" / "candidates"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# MEMORY CONFIGURATION
# ---------------------------------------------------------------------

MEMORY_LIMIT = "4GB"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def source3_glob():
    return str(
        ROOT
        / "artifacts"
        / "normalized"
        / "train_source3_part_*.parquet"
    )


def connect():

    con = duckdb.connect(str(DB_PATH))

    con.execute(
        f"SET memory_limit='{MEMORY_LIMIT}'"
    )

    # Keep temporary spilling on disk.
    con.execute(
        "SET temp_directory='artifacts/duckdb/tmp'"
    )

    con.execute(
        "SET preserve_insertion_order=false"
    )

    return con


# ---------------------------------------------------------------------
# DATABASE SETUP
# ---------------------------------------------------------------------

def create_views(con):

    print("\nCreating Parquet views...")

    s1 = ROOT / "artifacts" / "normalized" / "train_source1.parquet"
    s2 = ROOT / "artifacts" / "normalized" / "train_source2.parquet"

    con.execute(f"""
        CREATE OR REPLACE VIEW s1 AS
        SELECT *
        FROM read_parquet('{s1.as_posix()}');
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW s2 AS
        SELECT *
        FROM read_parquet('{s2.as_posix()}');
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW s3 AS
        SELECT *
        FROM read_parquet('{source3_glob()}');
    """)

    print("Views created.")


# ---------------------------------------------------------------------
# BASIC STATISTICS
# ---------------------------------------------------------------------

def show_counts(con):

    print("\nCounting records...")

    for table in ["s1", "s2", "s3"]:

        result = con.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

        print(f"{table.upper()}: {result:,}")


# ---------------------------------------------------------------------
# GROUND TRUTH
# ---------------------------------------------------------------------

def create_ground_truth(con):

    gt = ROOT / "dataset" / "train" / "train_ground_truth.tsv"

    print("\nLoading ground truth...")

    con.execute(f"""
        CREATE OR REPLACE TABLE ground_truth AS

        SELECT
            source1_entity_id,
            trim(matched_entity_ids) AS matched_entity_ids
        FROM read_csv(
            '{gt.as_posix()}',
            delim='\\t',
            header=true,
            columns={{
                'source1_entity_id': 'VARCHAR',
                'matched_entity_ids': 'VARCHAR'
            }}
        );
    """)

    print(
        "Ground truth rows:",
        con.execute(
            "SELECT COUNT(*) FROM ground_truth"
        ).fetchone()[0],
    )


# ---------------------------------------------------------------------
# EXACT NAME BLOCK
# ---------------------------------------------------------------------

def exact_name_block(con):

    print("\n" + "=" * 80)
    print("BLOCK 1: EXACT NORMALIZED NAME")
    print("=" * 80)

    output = OUTPUT_DIR / "exact_name_candidates.parquet"

    if output.exists():
        output.unlink()

    query = f"""
        COPY (

            SELECT
                s1.entity_id AS source1_entity_id,
                s2.entity_id AS candidate_entity_id,
                'exact_name' AS block_type

            FROM s1

            INNER JOIN s2
                ON s1.name_norm = s2.name_norm

            WHERE
                s1.name_norm <> ''

        )

        TO '{output.as_posix()}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
    """

    con.execute(query)

    count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{output.as_posix()}')"
    ).fetchone()[0]

    print(f"Candidates: {count:,}")


# ---------------------------------------------------------------------
# EXACT ADDRESS BLOCK
# ---------------------------------------------------------------------

def exact_address_block(con):

    print("\n" + "=" * 80)
    print("BLOCK 2: EXACT NORMALIZED ADDRESS")
    print("=" * 80)

    output = OUTPUT_DIR / "exact_address_candidates.parquet"

    if output.exists():
        output.unlink()

    query = f"""
        COPY (

            SELECT
                s1.entity_id AS source1_entity_id,
                s2.entity_id AS candidate_entity_id,
                'exact_address' AS block_type

            FROM s1

            INNER JOIN s2
                ON s1.address_norm = s2.address_norm

            WHERE
                s1.address_norm <> ''

        )

        TO '{output.as_posix()}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
    """

    con.execute(query)

    count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{output.as_posix()}')"
    ).fetchone()[0]

    print(f"Candidates: {count:,}")


# ---------------------------------------------------------------------
# GROUND TRUTH COVERAGE
# ---------------------------------------------------------------------

def evaluate_block(con, candidate_file, block_name):

    print("\n" + "=" * 80)
    print(f"EVALUATING: {block_name}")
    print("=" * 80)

    # Turn comma-separated ground truth into one row per
    # expected candidate.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE truth_pairs AS

        SELECT
            source1_entity_id,
            trim(
                unnest(
                    string_split(matched_entity_ids, ',')
                )
            ) AS candidate_entity_id

        FROM ground_truth

        WHERE
            matched_entity_ids IS NOT NULL
            AND matched_entity_ids <> '';
    """)

    total_truth = con.execute(
        """
        SELECT COUNT(*)
        FROM truth_pairs
        """
    ).fetchone()[0]

    recovered = con.execute(
        f"""
        SELECT COUNT(*)

        FROM truth_pairs t

        INNER JOIN read_parquet(
            '{candidate_file.as_posix()}'
        ) c

        ON
            t.source1_entity_id = c.source1_entity_id
            AND
            t.candidate_entity_id = c.candidate_entity_id
        """
    ).fetchone()[0]

    recall = (
        recovered / total_truth
        if total_truth
        else 0.0
    )

    print(f"True pairs:       {total_truth:,}")
    print(f"Recovered pairs:  {recovered:,}")
    print(f"Blocking recall:  {recall:.6%}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 100)
    print("PERSON 1 — DISK-BACKED BLOCKING")
    print("=" * 100)

    con = connect()

    try:

        create_views(con)

        show_counts(con)

        create_ground_truth(con)

        exact_name_block(con)

        evaluate_block(
            con,
            OUTPUT_DIR / "exact_name_candidates.parquet",
            "EXACT NAME",
        )

        exact_address_block(con)

        evaluate_block(
            con,
            OUTPUT_DIR / "exact_address_candidates.parquet",
            "EXACT ADDRESS",
        )

    finally:

        con.close()

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()