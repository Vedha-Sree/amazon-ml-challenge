"""
Person 1 — High-Recall Candidate Generator (Final Production Version)
======================================================================

Strategy:
  Layer 1 — Exact / token blocking (DuckDB, fast)
      8 rules with both-side frequency caps:
      • name_norm, name_compact, name_tokens
      • address_norm, address_compact, address_tokens
      • first_word+country (min_len 5, cap 50)
      • prefix6+country    (min_len 6, cap 30)

  Layer 2 — Fuzzy character-prefix blocking (Polars + RapidFuzz)
      • Build prefix-bucket index on name_compact (lengths 4, 5, 6)
      • For each S1 entity look up its prefix bucket(s)
      • Buckets with > FUZZY_BUCKET_CAP entries are skipped
      • Score with RapidFuzz partial_ratio, keep pairs >= threshold
      • Fully memory-bounded: no global inverted index

  Final output: combined deduplicated candidate set.

KPI: candidate recall on train set
Output: output/candidate_pairs.tsv (test set)

Run from student_resource/ directory:
    python code/business_entity_resolution/src/person1_high_recall_generator.py
"""

from pathlib import Path
from collections import defaultdict
import time

import duckdb
import polars as pl
from rapidfuzz import fuzz

# ------------------------------------------------------------------ #
#  Paths                                                               #
# ------------------------------------------------------------------ #

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

# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #

EXACT_CAP = 50           # both-side cap for exact/token rules
FUZZY_BUCKET_CAP = 150   # max targets per bucket in fuzzy step
FUZZY_SCORE_THRESHOLD = 65  # RapidFuzz partial_ratio minimum
PREFIX_LENGTHS = [4, 5, 6]  # char prefix lengths for fuzzy buckets

# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ #
#  Layer 1: exact/token blocking via DuckDB                           #
# ------------------------------------------------------------------ #

def run_exact_blocking(
    conn: duckdb.DuckDBPyConnection,
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
) -> int:
    """
    Populate 'distinct_candidates' table with pairs from all exact/token rules.
    Returns total distinct pairs inserted.
    """
    log("[Layer 1] Exact / token blocking...")

    for view_name, path in [("s1", s1_path), ("s2", s2_path), ("s3", s3_path)]:
        conn.execute(f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT entity_id, country,
                   name_norm, name_compact, name_tokens,
                   address_norm, address_compact, address_tokens,
                   CASE WHEN length(trim(string_split(name_norm,' ')[1])) >= 5
                        THEN trim(string_split(name_norm,' ')[1]) ELSE NULL END AS name_first_word,
                   CASE WHEN length(name_compact) >= 6
                        THEN substr(name_compact,1,6) ELSE NULL END AS name_prefix6
            FROM read_parquet('{path.as_posix()}')
        """)
        cnt = conn.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0]
        log(f"  View '{view_name}': {cnt:,} rows")

    conn.execute("""
        CREATE OR REPLACE TABLE distinct_candidates (
            source1_entity_id VARCHAR,
            target_entity_id  VARCHAR
        )
    """)

    # (rule_name, key_col, min_key_len, both_side_cap, requires_country_match)
    rules = [
        ("exact_name",       "name_norm",       1, EXACT_CAP, False),
        ("compact_name",     "name_compact",    1, EXACT_CAP, False),
        ("exact_address",    "address_norm",    1, EXACT_CAP, False),
        ("compact_address",  "address_compact", 1, EXACT_CAP, False),
        ("name_tokens",      "name_tokens",     1, EXACT_CAP, False),
        ("address_tokens",   "address_tokens",  1, EXACT_CAP, False),
    ]

    total = 0
    for target_name, target_view in [("S2", "s2"), ("S3", "s3")]:
        # Create a deduplicating accumulator table for this source pair
        conn.execute(f"""
            CREATE OR REPLACE TABLE accum_{target_name} (
                source1_entity_id VARCHAR,
                target_entity_id  VARCHAR
            )
        """)

        for rule_name, key_col, min_len, cap, use_country in rules:
            t0 = time.time()
            country_clause = f"AND s1.country = {target_view}.country" if use_country else ""
            # Insert only NEW pairs not already in accumulator
            conn.execute(f"""
                INSERT INTO accum_{target_name}
                WITH
                s1f AS (
                    SELECT {key_col} AS k FROM s1
                    WHERE {key_col} IS NOT NULL AND {key_col} <> ''
                      AND length({key_col}) >= {min_len}
                    GROUP BY {key_col} HAVING COUNT(*) <= {cap}
                ),
                tgf AS (
                    SELECT {key_col} AS k FROM {target_view}
                    WHERE {key_col} IS NOT NULL AND {key_col} <> ''
                      AND length({key_col}) >= {min_len}
                    GROUP BY {key_col} HAVING COUNT(*) <= {cap}
                ),
                elig AS (SELECT s1f.k FROM s1f JOIN tgf USING (k)),
                new_pairs AS (
                    SELECT DISTINCT s1.entity_id AS s1id, {target_view}.entity_id AS tgid
                    FROM s1
                    JOIN elig          ON s1.{key_col} = elig.k
                    JOIN {target_view} ON s1.{key_col} = {target_view}.{key_col}
                                         {country_clause}
                    WHERE s1.{key_col} IS NOT NULL AND s1.{key_col} <> ''
                )
                SELECT np.s1id, np.tgid FROM new_pairs np
                WHERE NOT EXISTS (
                    SELECT 1 FROM accum_{target_name} a
                    WHERE a.source1_entity_id = np.s1id
                      AND a.target_entity_id  = np.tgid
                )
            """)
            new_total = conn.execute(f"SELECT COUNT(*) FROM accum_{target_name}").fetchone()[0]
            added = new_total - (total - sum(
                conn.execute(f"SELECT COUNT(*) FROM accum_{tn}").fetchone()[0]
                for tn in (["S2"] if target_name == "S3" else [])
            ))
            # Simpler: just track per-target total
            log(
                f"  [{target_name}] {rule_name:<22} total so far: {new_total:>9,}  "
                f"({time.time()-t0:.1f}s)"
            )

        deduped = conn.execute(f"SELECT COUNT(*) FROM accum_{target_name}").fetchone()[0]
        log(f"  [{target_name}] Final unique pairs: {deduped:,}")

        if target_name == "S2":
            conn.execute("CREATE OR REPLACE TABLE distinct_candidates AS SELECT * FROM accum_S2")
        else:
            conn.execute("INSERT INTO distinct_candidates SELECT * FROM accum_S3")
        total += deduped

    log(f"\n  [Layer 1] Total exact candidates: {total:,}")
    return total


# ------------------------------------------------------------------ #
#  Layer 2: fuzzy character-prefix blocking                           #
# ------------------------------------------------------------------ #

def run_fuzzy_blocking(
    s1_df: pl.DataFrame,
    target_df: pl.DataFrame,
    label: str,
) -> pl.DataFrame:
    """
    Character prefix/suffix bucket blocking:
    - Group target entities by name_compact[:plen] for plen in PREFIX_LENGTHS.
    - Only process buckets with <= FUZZY_BUCKET_CAP entries.
    - For each S1 entity, look up its prefix bucket candidates.
    - Score with RapidFuzz partial_ratio; keep pairs >= FUZZY_SCORE_THRESHOLD.
    Memory-bounded: no global inverted index built.
    """
    log(f"\n[Layer 2] Fuzzy name blocking  S1 -> {label}...")
    t0 = time.time()

    tgt_ids = target_df["entity_id"].to_list()
    tgt_names = target_df["name_compact"].to_list()

    # Build prefix -> list[(tid, tname)] buckets
    buckets: dict[str, list] = defaultdict(list)
    for tid, tname in zip(tgt_ids, tgt_names):
        if not tname or len(tname) < 3:
            continue
        for plen in PREFIX_LENGTHS:
            if len(tname) >= plen:
                buckets[tname[:plen]].append((tid, tname))

    # Only keep bounded buckets
    eligible: dict[str, list] = {
        k: v for k, v in buckets.items() if len(v) <= FUZZY_BUCKET_CAP
    }
    log(
        f"  Buckets: {len(buckets):,} total, "
        f"{len(eligible):,} eligible (cap<={FUZZY_BUCKET_CAP})"
    )
    del buckets

    s1_ids = s1_df["entity_id"].to_list()
    s1_names = s1_df["name_compact"].to_list()

    result_s1: list[str] = []
    result_tgt: list[str] = []
    processed = 0

    for s1_id, s1_name in zip(s1_ids, s1_names):
        if not s1_name or len(s1_name) < 3:
            processed += 1
            continue

        # Collect candidates from all prefix lengths, dedup by tid
        cand_map: dict[str, str] = {}
        for plen in PREFIX_LENGTHS:
            if len(s1_name) >= plen:
                key = s1_name[:plen]
                if key in eligible:
                    for tid, tname in eligible[key]:
                        if tid not in cand_map:
                            cand_map[tid] = tname

        # Score & filter
        for tid, tname in cand_map.items():
            if fuzz.partial_ratio(s1_name, tname) >= FUZZY_SCORE_THRESHOLD:
                result_s1.append(s1_id)
                result_tgt.append(tid)

        processed += 1
        if processed % 300_000 == 0:
            log(
                f"  {processed:,} / {len(s1_ids):,} S1 entities, "
                f"{len(result_s1):,} fuzzy pairs so far..."
            )

    elapsed = time.time() - t0
    log(f"  [Layer 2] {label}: {len(result_s1):,} fuzzy pairs in {elapsed:.1f}s")

    if not result_s1:
        return pl.DataFrame({
            "source1_entity_id": pl.Series([], dtype=pl.String),
            "target_entity_id":  pl.Series([], dtype=pl.String),
        })

    return pl.DataFrame({
        "source1_entity_id": result_s1,
        "target_entity_id":  result_tgt,
    }).unique()


# ------------------------------------------------------------------ #
#  Recall evaluation                                                   #
# ------------------------------------------------------------------ #

def evaluate_recall(
    conn: duckdb.DuckDBPyConnection,
) -> tuple[int, int, float]:
    """Load GT and compute recall against distinct_candidates."""
    conn.execute(f"""
        CREATE OR REPLACE TABLE gt AS
        SELECT source1_entity_id AS s1_id, trim(x) AS matched_id
        FROM read_csv(
            '{GT_FILE.as_posix()}',
            delim='\t',
            header=true,
            columns={{'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'}}
        ),
        UNNEST(string_split(matched_entity_ids,',')) AS t(x)
        WHERE trim(x) <> ''
    """)
    total = conn.execute("SELECT COUNT(*) FROM gt").fetchone()[0]
    recovered = conn.execute("""
        SELECT COUNT(*) FROM gt
        JOIN distinct_candidates c
          ON gt.s1_id      = c.source1_entity_id
         AND gt.matched_id = c.target_entity_id
    """).fetchone()[0]
    pct = recovered / total * 100.0 if total > 0 else 0.0
    return recovered, total, pct


# ------------------------------------------------------------------ #
#  Main pipeline                                                       #
# ------------------------------------------------------------------ #

def run_pipeline(
    conn: duckdb.DuckDBPyConnection,
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    is_train: bool,
) -> None:

    label = "TRAIN" if is_train else "TEST"
    log(f"\n{'='*60}")
    log(f"  PIPELINE  —  {label}")
    log(f"{'='*60}")

    # ---- Layer 1: exact/token blocking ----
    exact_total = run_exact_blocking(conn, s1_path, s2_path, s3_path)

    if is_train:
        rec, tot, pct = evaluate_recall(conn)
        log(f"\n  [After Layer 1] Recall: {rec:,} / {tot:,}  ({pct:.2f}%)")

    # ---- Layer 2: fuzzy prefix blocking ----
    s1_df = pl.read_parquet(s1_path, columns=["entity_id", "name_compact"])
    s2_df = pl.read_parquet(s2_path, columns=["entity_id", "name_compact"])
    s3_df = pl.read_parquet(s3_path, columns=["entity_id", "name_compact"])

    fuzzy_s2 = run_fuzzy_blocking(s1_df, s2_df, "S2")
    fuzzy_s3 = run_fuzzy_blocking(s1_df, s3_df, "S3")

    del s1_df, s2_df, s3_df

    if len(fuzzy_s2) > 0 or len(fuzzy_s3) > 0:
        log("\n  Merging fuzzy candidates into candidate set...")
        fuzzy_all = pl.concat([fuzzy_s2, fuzzy_s3])
        log(f"  Fuzzy pairs (before merge): {len(fuzzy_all):,}")

        conn.register("fuzzy_df", fuzzy_all.to_arrow())
        conn.execute("""
            INSERT INTO distinct_candidates
            SELECT DISTINCT f.source1_entity_id, f.target_entity_id
            FROM fuzzy_df f
            WHERE NOT EXISTS (
                SELECT 1 FROM distinct_candidates c
                WHERE c.source1_entity_id = f.source1_entity_id
                  AND c.target_entity_id  = f.target_entity_id
            )
        """)
        del fuzzy_all

    final_total = conn.execute("SELECT COUNT(*) FROM distinct_candidates").fetchone()[0]
    fuzzy_added = final_total - exact_total
    log(f"\n  [After Layer 2] Fuzzy added {fuzzy_added:,} pairs → total {final_total:,}")

    # ---- Final recall (train) ----
    if is_train:
        rec, tot, pct = evaluate_recall(conn)
        s1_count = conn.execute("SELECT COUNT(DISTINCT entity_id) FROM s1").fetchone()[0]
        avg_cands = final_total / s1_count if s1_count > 0 else 0.0
        log(f"\n{'='*60}")
        log(f"  FINAL TRAIN CANDIDATE RECALL:")
        log(f"    Recovered  : {rec:,} / {tot:,}  ({pct:.2f}%)")
        log(f"    Total pairs: {final_total:,}")
        log(f"    Avg cands / S1 entity: {avg_cands:.1f}")
        log(f"{'='*60}")

    # ---- Export candidate_pairs.tsv (test) ----
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
                SELECT source1_entity_id,
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
    log("  PERSON 1 — HIGH-RECALL CANDIDATE GENERATOR (v2)")
    log(f"  exact_cap={EXACT_CAP}, fuzzy_bucket_cap={FUZZY_BUCKET_CAP}, "
        f"score_threshold={FUZZY_SCORE_THRESHOLD}")
    log("=" * 60)

    conn = duckdb.connect()
    conn.execute(f"""
        SET memory_limit='6GB';
        SET temp_directory='{TMP_DIR.as_posix()}';
        SET preserve_insertion_order=false;
        SET threads=4;
    """)

    log("\n>>> STEP 1: TRAIN  (recall evaluation)")
    run_pipeline(
        conn,
        WORK / "train_source1.parquet",
        WORK / "train_source2.parquet",
        WORK / "train_source3.parquet",
        is_train=True,
    )

    log("\n>>> STEP 2: TEST  (output candidate_pairs.tsv)")
    run_pipeline(
        conn,
        WORK / "test_source1.parquet",
        WORK / "test_source2.parquet",
        WORK / "test_source3.parquet",
        is_train=False,
    )

    conn.close()
    log("\nDONE.")


if __name__ == "__main__":
    main()
