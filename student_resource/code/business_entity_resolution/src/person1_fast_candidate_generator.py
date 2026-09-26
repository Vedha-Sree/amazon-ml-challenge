"""
Person 1 — Enhanced High-Recall Candidate Generator
====================================================
Blocking strategy (8 rules, both-side frequency caps):
  1. exact_name        — name_norm equality          (cap 200)
  2. compact_name      — name_compact equality        (cap 200)
  3. exact_address     — address_norm equality        (cap 200)
  4. compact_address   — address_compact equality     (cap 200)
  5. name_tokens       — sorted-token-set equality    (cap 200)
  6. address_tokens    — sorted-token-set equality    (cap 200)
  7. first_word+country— first word of name + country (cap 50, min_len 5)
  8. prefix6+country   — first 6 chars + country      (cap 30, min_len 6)

Both-side frequency cap: a key is eligible only when it appears
<= cap times in BOTH s1 AND the target source.  This prevents
a single common key (e.g. "the") from exploding into millions
of pairs and dominating the candidate table.

Candidates are written to a persistent disk table so dedup is
performed in a single streaming pass rather than on 400 M rows.
"""

from pathlib import Path
import time
import duckdb

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work" / "person1"
TRAIN = ROOT / "dataset" / "train"
TEST = ROOT / "dataset" / "test"
OUTPUT = ROOT / "output"
TMP_DIR = ROOT / "artifacts" / "duckdb" / "tmp"

OUTPUT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

GT_FILE = TRAIN / "train_ground_truth.tsv"


def log(msg: str) -> None:
    print(msg, flush=True)


def setup_duckdb() -> duckdb.DuckDBPyConnection:
    log("Initializing DuckDB engine...")
    conn = duckdb.connect()
    conn.execute(f"""
        SET memory_limit='6GB';
        SET temp_directory='{TMP_DIR.as_posix()}';
        SET preserve_insertion_order=false;
        SET threads=4;
    """)
    return conn


# ------------------------------------------------------------------ #
#  Core blocking function                                              #
# ------------------------------------------------------------------ #

def run_candidate_generation(
    conn: duckdb.DuckDBPyConnection,
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    is_train: bool,
) -> None:

    label = "TRAIN" if is_train else "TEST"
    log(f"\n{'='*60}")
    log(f"  CANDIDATE GENERATION  —  {label}")
    log(f"{'='*60}")

    # ---------------------------------------------------------------- #
    # 1. Register Parquet views                                         #
    # ---------------------------------------------------------------- #
    for view_name, path in [("s1", s1_path), ("s2", s2_path), ("s3", s3_path)]:
        conn.execute(f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT
                entity_id, country,
                name_norm, name_compact, name_tokens,
                address_norm, address_compact, address_tokens,
                -- derived blocking keys
                CASE
                    WHEN length(trim(string_split(name_norm, ' ')[1])) >= 5
                    THEN trim(string_split(name_norm, ' ')[1])
                    ELSE NULL
                END AS name_first_word,
                CASE
                    WHEN length(name_compact) >= 6
                    THEN substr(name_compact, 1, 6)
                    ELSE NULL
                END AS name_prefix6
            FROM read_parquet('{path.as_posix()}');
        """)
        count = conn.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0]
        log(f"  View '{view_name}': {count:,} rows")

    # ---------------------------------------------------------------- #
    # 2. Blocking rules — (name, key_col, min_len, both_side_cap)       #
    # ---------------------------------------------------------------- #
    #
    # both_side_cap: key must appear <= cap times in s1 AND in target.
    # We dedup per source-pair (S2, S3) to keep peak memory bounded.
    #
    rules = [
        # rule_name          key_col          min_len  cap
        ("exact_name",       "name_norm",          1,   50),
        ("compact_name",     "name_compact",        1,   50),
        ("exact_address",    "address_norm",        1,   50),
        ("compact_address",  "address_compact",     1,   50),
        ("name_tokens",      "name_tokens",         1,   50),
        ("address_tokens",   "address_tokens",      1,   50),
        ("first_word+cntry", "name_first_word",     5,   20),
        ("prefix6+cntry",    "name_prefix6",        6,   15),
    ]

    # Rules that additionally require country equality
    country_rules = {"first_word+cntry", "prefix6+cntry"}

    total_inserted = 0

    # Dedup per source pair so peak memory stays bounded
    for target_name, target_view in [("S2", "s2"), ("S3", "s3")]:
        log(f"\n  --- S1 → {target_name} ---")

        # Staging table for this source pair only
        conn.execute("""
            CREATE OR REPLACE TABLE staging (
                source1_entity_id VARCHAR,
                target_entity_id  VARCHAR
            );
        """)
        staged = 0

        for rule_name, key_col, min_len, cap in rules:
            t0 = time.time()

            country_clause = (
                f"AND s1.country = {target_view}.country"
                if rule_name in country_rules
                else ""
            )

            conn.execute(f"""
                INSERT INTO staging
                WITH
                s1_freq AS (
                    SELECT {key_col} AS key
                    FROM s1
                    WHERE {key_col} IS NOT NULL
                      AND {key_col} <> ''
                      AND length({key_col}) >= {min_len}
                    GROUP BY {key_col}
                    HAVING COUNT(*) <= {cap}
                ),
                tgt_freq AS (
                    SELECT {key_col} AS key
                    FROM {target_view}
                    WHERE {key_col} IS NOT NULL
                      AND {key_col} <> ''
                      AND length({key_col}) >= {min_len}
                    GROUP BY {key_col}
                    HAVING COUNT(*) <= {cap}
                ),
                eligible_keys AS (
                    SELECT s1_freq.key FROM s1_freq JOIN tgt_freq USING (key)
                )
                SELECT DISTINCT
                    s1.entity_id            AS source1_entity_id,
                    {target_view}.entity_id AS target_entity_id
                FROM s1
                JOIN eligible_keys      ON s1.{key_col} = eligible_keys.key
                JOIN {target_view}      ON s1.{key_col} = {target_view}.{key_col}
                                          {country_clause}
                ;
            """)

            new_staged = conn.execute("SELECT COUNT(*) FROM staging").fetchone()[0]
            added = new_staged - staged
            staged = new_staged
            elapsed = time.time() - t0
            log(
                f"    [{target_name}] {rule_name:<22} "
                f"+{added:>8,} raw  |  staging {staged:>10,}  "
                f"({elapsed:.1f}s)"
            )

        # Dedup this source pair and append to main table
        log(f"    [{target_name}] deduplicating {staged:,} rows...")
        t_dd = time.time()
        conn.execute("""
            CREATE OR REPLACE TABLE staging AS
            SELECT DISTINCT source1_entity_id, target_entity_id
            FROM staging;
        """)
        deduped = conn.execute("SELECT COUNT(*) FROM staging").fetchone()[0]
        log(f"    [{target_name}] {deduped:,} unique pairs after dedup ({time.time()-t_dd:.1f}s)")

        # First pair: create main table; subsequent: append
        if target_name == "S2":
            conn.execute("""
                CREATE OR REPLACE TABLE distinct_candidates AS
                SELECT * FROM staging;
            """)
        else:
            conn.execute("INSERT INTO distinct_candidates SELECT * FROM staging;")

        total_inserted += deduped

    log(f"\n  TOTAL DISTINCT CANDIDATES: {total_inserted:,}")

    # ---------------------------------------------------------------- #
    # 4. Evaluate recall (train only)                                   #
    # ---------------------------------------------------------------- #
    if is_train:
        log("\n  Evaluating candidate recall against ground truth...")

        conn.execute(f"""
            CREATE OR REPLACE TABLE gt AS
            SELECT
                source1_entity_id AS s1_id,
                trim(x)           AS matched_id
            FROM read_csv(
                '{GT_FILE.as_posix()}',
                delim='\\t',
                header=true,
                columns={{
                    'source1_entity_id': 'VARCHAR',
                    'matched_entity_ids': 'VARCHAR'
                }}
            ),
            UNNEST(string_split(matched_entity_ids, ',')) AS t(x)
            WHERE trim(x) <> '';
        """)

        gt_total = conn.execute("SELECT COUNT(*) FROM gt").fetchone()[0]

        recovered = conn.execute("""
            SELECT COUNT(*)
            FROM gt
            JOIN distinct_candidates c
              ON gt.s1_id    = c.source1_entity_id
             AND gt.matched_id = c.target_entity_id
        """).fetchone()[0]

        recall = (recovered / gt_total * 100.0) if gt_total > 0 else 0.0

        s1_count = conn.execute(
            "SELECT COUNT(DISTINCT entity_id) FROM s1"
        ).fetchone()[0]
        avg_cands = total_inserted / s1_count if s1_count > 0 else 0.0

        log(f"\n{'='*60}")
        log(f"  TRAIN GROUND TRUTH CANDIDATE RECALL:")
        log(f"    Recovered : {recovered:,} / {gt_total:,}  ({recall:.2f}%)")
        log(f"    Avg cands / S1 entity : {avg_cands:.1f}")
        log(f"{'='*60}")

    # ---------------------------------------------------------------- #
    # 5. Export candidate_pairs.tsv (test only)                        #
    # ---------------------------------------------------------------- #
    else:
        out_tsv = OUTPUT / "candidate_pairs.tsv"
        log(f"\n  Exporting {out_tsv.name} ...")
        t0 = time.time()

        conn.execute("""
            CREATE OR REPLACE TABLE final_export AS
            SELECT
                s1.entity_id AS source1_entity_id,
                COALESCE(c.candidate_ids, '') AS candidate_entity_ids
            FROM s1
            LEFT JOIN (
                SELECT
                    source1_entity_id,
                    string_agg(target_entity_id, ',') AS candidate_ids
                FROM distinct_candidates
                GROUP BY source1_entity_id
            ) c ON s1.entity_id = c.source1_entity_id
            ORDER BY s1.entity_id
        """)

        conn.execute(f"""
            COPY final_export TO '{out_tsv.as_posix()}' (HEADER, DELIMITER '\t')
        """)

        rows = conn.execute("SELECT COUNT(*) FROM final_export").fetchone()[0]
        log(f"  Exported {rows:,} S1 rows  ({time.time()-t0:.1f}s)")


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def main() -> None:
    log("=" * 60)
    log("  PERSON 1 — HIGH-RECALL CANDIDATE GENERATOR")
    log("=" * 60)

    conn = setup_duckdb()

    log("\n>>> STEP 1: TRAIN (recall evaluation)")
    run_candidate_generation(
        conn,
        WORK / "train_source1.parquet",
        WORK / "train_source2.parquet",
        WORK / "train_source3.parquet",
        is_train=True,
    )

    log("\n>>> STEP 2: TEST (output candidate_pairs.tsv)")
    run_candidate_generation(
        conn,
        WORK / "test_source1.parquet",
        WORK / "test_source2.parquet",
        WORK / "test_source3.parquet",
        is_train=False,
    )

    conn.close()
    log("\nCANDIDATE GENERATION COMPLETE.")


if __name__ == "__main__":
    main()
