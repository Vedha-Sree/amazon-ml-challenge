from pathlib import Path
import sqlite3
import csv
import gc

import polars as pl


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work" / "person1"
TRAIN = ROOT / "dataset" / "train"

WORK.mkdir(parents=True, exist_ok=True)

DB_DIR = WORK / "sqlite"
DB_DIR.mkdir(parents=True, exist_ok=True)

BATCH = 25_000


TRAIN_FILES = {
    "S2": WORK / "train_source2.parquet",
    "S3": WORK / "train_source3.parquet",
}


RULES = [
    ("exact_name", "name_norm"),
    ("compact_name", "name_compact"),
    ("name_tokens", "name_tokens"),
    ("exact_address", "address_norm"),
    ("compact_address", "address_compact"),
    ("address_tokens", "address_tokens"),
]


def open_db(source):

    path = DB_DIR / f"{source.lower()}_index.sqlite"

    conn = sqlite3.connect(
        path,
        timeout=120,
    )

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-131072")

    return conn, path


def build_index(source, parquet_path):

    conn, db_path = open_db(source)

    # --------------------------------------------------------
    # If a complete index already exists, reuse it.
    # --------------------------------------------------------

    exists = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = 'entities'
        LIMIT 1
        """
    ).fetchone()

    if exists is not None:

        count = conn.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]

        if count > 0:

            print(
                f"{source}: existing index found "
                f"({count:,} rows)."
            )

            return conn, db_path

    # --------------------------------------------------------
    # Incomplete database: rebuild cleanly.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(f"BUILDING DISK INDEX: {source}")
    print("=" * 70)

    conn.execute("DROP TABLE IF EXISTS entities")
    conn.commit()

    conn.execute(
        """
        CREATE TABLE entities (
            entity_id TEXT PRIMARY KEY,
            name_norm TEXT,
            name_compact TEXT,
            name_tokens TEXT,
            address_norm TEXT,
            address_compact TEXT,
            address_tokens TEXT
        )
        """
    )

    conn.commit()

    total = 0

    lf = (
        pl.scan_parquet(parquet_path)
        .select([
            "entity_id",
            "name_norm",
            "name_compact",
            "name_tokens",
            "address_norm",
            "address_compact",
            "address_tokens",
        ])
    )

    for df in lf.collect_batches(
        chunk_size=BATCH
    ):

        conn.executemany(
            """
            INSERT INTO entities (
                entity_id,
                name_norm,
                name_compact,
                name_tokens,
                address_norm,
                address_compact,
                address_tokens
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            df.iter_rows(),
        )

        conn.commit()

        total += df.height

        print(
            f"  indexed {total:,} rows",
            flush=True,
        )

        del df
        gc.collect()

    print()
    print(f"Finished loading {total:,} rows.")

    # --------------------------------------------------------
    # Create indexes one at a time.
    # --------------------------------------------------------

    for column in [
        "name_norm",
        "name_compact",
        "name_tokens",
        "address_norm",
        "address_compact",
        "address_tokens",
    ]:

        print(
            f"  creating index: {column}",
            flush=True,
        )

        conn.execute(
            f"""
            CREATE INDEX idx_{column}
            ON entities({column})
            """
        )

        conn.commit()

    conn.execute("ANALYZE")
    conn.commit()

    print()
    print(f"Index ready: {db_path}")

    return conn, db_path


def stream_s1():

    return (
        pl.scan_parquet(
            WORK / "train_source1.parquet"
        )
        .select([
            "entity_id",
            "name_norm",
            "name_compact",
            "name_tokens",
            "address_norm",
            "address_compact",
            "address_tokens",
        ])
        .collect_batches(
            chunk_size=BATCH
        )
    )


def generate_candidates(
    target_source,
    target_db,
    output_path,
):

    print()
    print("=" * 70)
    print(
        f"GENERATING CANDIDATES: "
        f"S1 -> {target_source}"
    )
    print("=" * 70)

    candidate_db_path = (
        DB_DIR /
        f"candidates_{target_source.lower()}.sqlite"
    )

    cdb = sqlite3.connect(
        candidate_db_path,
        timeout=120,
    )

    cdb.execute(
        "PRAGMA journal_mode=WAL"
    )

    cdb.execute(
        "PRAGMA synchronous=NORMAL"
    )

    cdb.execute(
        "PRAGMA temp_store=FILE"
    )

    cdb.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            source1_entity_id TEXT NOT NULL,
            target_entity_id TEXT NOT NULL,
            PRIMARY KEY (
                source1_entity_id,
                target_entity_id
            )
        )
        """
    )

    cdb.commit()

    processed = 0

    for batch in stream_s1():

        # ----------------------------------------------------
        # Process each blocking rule separately.
        # ----------------------------------------------------

        for rule, column in RULES:

            rows_to_insert = []

            for s1_id, key in batch.select(
                ["entity_id", column]
            ).iter_rows():

                if key is None or key == "":
                    continue

                rows = target_db.execute(
                    f"""
                    SELECT entity_id
                    FROM entities
                    WHERE {column} = ?
                    """,
                    (key,),
                )

                for target_id, in rows:

                    rows_to_insert.append(
                        (
                            s1_id,
                            target_id,
                        )
                    )

            if rows_to_insert:

                cdb.executemany(
                    """
                    INSERT OR IGNORE INTO candidates
                    VALUES (?, ?)
                    """,
                    rows_to_insert,
                )

                cdb.commit()

            del rows_to_insert

        processed += batch.height

        print(
            f"  processed S1: {processed:,}",
            flush=True,
        )

        del batch
        gc.collect()

    # --------------------------------------------------------
    # Export final candidate file.
    # --------------------------------------------------------

    print()
    print("Exporting candidate TSV...")

    cursor = cdb.execute(
        """
        SELECT
            source1_entity_id,
            GROUP_CONCAT(target_entity_id)
        FROM candidates
        GROUP BY source1_entity_id
        ORDER BY source1_entity_id
        """
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(
            f,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writerow([
            "source1_entity_id",
            "candidate_entity_ids",
        ])

        count = 0

        for s1_id, target_ids in cursor:

            writer.writerow([
                s1_id,
                target_ids or "",
            ])

            count += 1

    print()
    print(
        f"Wrote {count:,} S1 rows."
    )

    print(
        f"Output: {output_path}"
    )

    cdb.close()


def main():

    print("=" * 70)
    print("PERSON 1 — DISK-BACKED CANDIDATE GENERATOR")
    print("=" * 70)

    # --------------------------------------------------------
    # S2
    # --------------------------------------------------------

    s2_conn, _ = build_index(
        "S2",
        TRAIN_FILES["S2"],
    )

    # --------------------------------------------------------
    # S3
    # --------------------------------------------------------

    s3_conn, _ = build_index(
        "S3",
        TRAIN_FILES["S3"],
    )

    # --------------------------------------------------------
    # Generate S1 -> S2 candidates
    # --------------------------------------------------------

    generate_candidates(
        "S2",
        s2_conn,
        WORK / "train_candidate_pairs_S2.tsv",
    )

    # --------------------------------------------------------
    # Generate S1 -> S3 candidates
    # --------------------------------------------------------

    generate_candidates(
        "S3",
        s3_conn,
        WORK / "train_candidate_pairs_S3.tsv",
    )

    s2_conn.close()
    s3_conn.close()

    print()
    print("=" * 70)
    print("PERSON 1 CANDIDATE GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()