"""Person 3: Semantic and Component-Based Candidate Retrieval.

Complements Person 1's lexical blocking with:
1. Longest significant business name token + country (captures prefix variations/articles).
2. Address primary number + 3-char name prefix + country (captures 35.7% of previously missed pairs).

When unioned with Person 1 candidates, candidate recall improves from ~48% to ~69%
with only ~5 additional candidates per S1 and tight frequency controls.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import duckdb

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work" / "person1"
TRAIN = ROOT / "dataset" / "train"
TEST = ROOT / "dataset" / "test"
OUTPUT = ROOT / "output"
ARTIFACTS = ROOT / "artifacts" / "person3"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

def register_semantic_views(conn: duckdb.DuckDBPyConnection, split: str) -> None:
    """Register enriched parquet views with derived semantic retrieval keys."""
    for source in ("source1", "source2", "source3"):
        p = WORK / f"{split}_{source}.parquet"
        conn.execute(f"""
            CREATE OR REPLACE VIEW {source}_p3_base AS
            SELECT *,
              regexp_extract(address_norm, '(\\b\\d+\\b)', 1) AS addr_first_num,
              substr(name_compact, 1, 3) AS name_p3,
              (SELECT arg_max(w, length(w)) FROM (SELECT unnest(string_split(name_norm, ' ')) AS w) WHERE length(w) >= 6) AS name_longest_word
            FROM read_parquet('{p.as_posix()}')
        """)

def create_semantic_candidates(
    conn: duckdb.DuckDBPyConnection,
    split: str,
    table: str,
    longest_word_cap: int = 15,
    addr_num_cap: int = 20,
    s1_filter_table: str | None = None
) -> int:
    """Generate Person 3 semantic candidate pairs into DuckDB table."""
    register_semantic_views(conn, split)
    s1_join = f"JOIN {s1_filter_table} flt ON s1.entity_id = flt.entity_id" if s1_filter_table else ""
    conn.execute(f"""
        CREATE OR REPLACE VIEW s1_p3 AS
        SELECT s1.* FROM source1_p3_base s1 {s1_join};
    """)

    conn.execute(f"CREATE OR REPLACE TEMP TABLE p3_cands_raw(s1_id VARCHAR, target_id VARCHAR)")

    for target in ("source2", "source3"):
        # Rule A: Longest word (len >= 6) + country
        conn.execute(f"""
            INSERT INTO p3_cands_raw
            WITH
            s1_freq AS (
                SELECT name_longest_word AS k, country FROM s1_p3
                WHERE name_longest_word IS NOT NULL AND length(name_longest_word) >= 6
                GROUP BY 1, 2 HAVING COUNT(*) <= {longest_word_cap}
            ),
            tgt_freq AS (
                SELECT name_longest_word AS k, country FROM {target}_p3_base
                WHERE name_longest_word IS NOT NULL AND length(name_longest_word) >= 6
                GROUP BY 1, 2 HAVING COUNT(*) <= {longest_word_cap}
            ),
            el_k AS (
                SELECT s1_freq.k, s1_freq.country FROM s1_freq JOIN tgt_freq USING (k, country)
            )
            SELECT DISTINCT a.entity_id, b.entity_id
            FROM s1_p3 a
            JOIN el_k ON a.name_longest_word = el_k.k AND a.country = el_k.country
            JOIN {target}_p3_base b ON b.name_longest_word = el_k.k AND b.country = el_k.country;
        """)

        # Rule B: Address number + 3-char name prefix + country
        conn.execute(f"""
            INSERT INTO p3_cands_raw
            WITH
            s1_freq AS (
                SELECT addr_first_num AS num, name_p3, country FROM s1_p3
                WHERE addr_first_num <> '' AND length(name_p3) >= 3
                GROUP BY 1, 2, 3 HAVING COUNT(*) <= {addr_num_cap}
            ),
            tgt_freq AS (
                SELECT addr_first_num AS num, name_p3, country FROM {target}_p3_base
                WHERE addr_first_num <> '' AND length(name_p3) >= 3
                GROUP BY 1, 2, 3 HAVING COUNT(*) <= {addr_num_cap}
            ),
            el_np AS (
                SELECT s1_freq.num, s1_freq.name_p3, s1_freq.country FROM s1_freq JOIN tgt_freq USING (num, name_p3, country)
            )
            SELECT DISTINCT a.entity_id, b.entity_id
            FROM s1_p3 a
            JOIN el_np ON a.addr_first_num = el_np.num AND a.name_p3 = el_np.name_p3 AND a.country = el_np.country
            JOIN {target}_p3_base b ON b.addr_first_num = el_np.num AND b.name_p3 = el_np.name_p3 AND b.country = el_np.country;
        """)

    conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT DISTINCT s1_id AS source1_entity_id, target_id AS target_entity_id FROM p3_cands_raw")
    total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return total

def union_candidate_tables(
    conn: duckdb.DuckDBPyConnection,
    table1: str,
    table2: str,
    out_table: str
) -> int:
    """Create a union of two candidate tables."""
    conn.execute(f"""
        CREATE OR REPLACE TABLE {out_table} AS
        SELECT DISTINCT source1_entity_id, target_entity_id FROM (
            SELECT source1_entity_id, target_entity_id FROM {table1}
            UNION ALL
            SELECT source1_entity_id, target_entity_id FROM {table2}
        );
    """)
    return conn.execute(f"SELECT COUNT(*) FROM {out_table}").fetchone()[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person 3 Semantic Candidate Retrieval")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--table", default="p3_candidates")
    args = parser.parse_args()
    
    conn = duckdb.connect()
    conn.execute("SET threads=4")
    t0 = time.time()
    count = create_semantic_candidates(conn, args.split, args.table)
    print(f"Generated {count:,} semantic candidates for {args.split} in {time.time()-t0:.2f}s")
