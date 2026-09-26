from pathlib import Path
import time
import duckdb

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work" / "person1"
TRAIN = ROOT / "dataset" / "train"
TEST = ROOT / "dataset" / "test"
OUTPUT = ROOT / "output"

GT_FILE = TRAIN / "train_ground_truth.tsv"

def log(msg):
    print(msg, flush=True)

def main():
    log("==================================================")
    log("PERSON 1 — HIGH-RECALL BLOCKING EVALUATOR")
    log("==================================================")

    conn = duckdb.connect()
    conn.execute("""
        SET memory_limit='4GB';
        SET temp_directory='artifacts/duckdb/tmp';
        SET preserve_insertion_order=false;
        SET threads=4;
    """)

    conn.execute(f"CREATE OR REPLACE VIEW s1 AS SELECT entity_id, country, business_name, business_address, name_norm, name_compact, address_norm, address_compact FROM read_parquet('{WORK / 'train_source1.parquet'}');")
    conn.execute(f"CREATE OR REPLACE VIEW s2 AS SELECT entity_id, country, business_name, business_address, name_norm, name_compact, address_norm, address_compact FROM read_parquet('{WORK / 'train_source2.parquet'}');")
    conn.execute(f"CREATE OR REPLACE VIEW s3 AS SELECT entity_id, country, business_name, business_address, name_norm, name_compact, address_norm, address_compact FROM read_parquet('{WORK / 'train_source3.parquet'}');")

    conn.execute(f"""
        CREATE OR REPLACE TABLE gt AS
        SELECT
            source1_entity_id AS s1_id,
            trim(x) AS matched_id
        FROM read_csv(
            '{GT_FILE}',
            delim='\\t',
            header=true,
            columns={{'source1_entity_id': 'VARCHAR', 'matched_entity_ids': 'VARCHAR'}}
        ),
        UNNEST(string_split(matched_entity_ids, ',')) AS t(x)
        WHERE trim(x) <> '';
    """)

    gt_total = conn.execute("SELECT COUNT(*) FROM gt;").fetchone()[0]
    log(f"Total Ground Truth Matches: {gt_total:,}")

    # Test adding prefix/suffix & token blocking keys
    conn.execute("CREATE OR REPLACE TABLE candidates (source1_entity_id VARCHAR, target_entity_id VARCHAR);")

    # Blocking Rules with appropriate frequency limits
    rules = [
        ("exact_name", "name_norm", 500),
        ("compact_name", "name_compact", 500),
        ("exact_address", "address_norm", 500),
        ("compact_address", "address_compact", 500),
        ("prefix_6_name", "substr(name_compact, 1, 6)", 250),
        ("prefix_8_name", "substr(name_compact, 1, 8)", 400),
        ("suffix_6_name", "substr(name_compact, -6)", 250),
    ]

    for rule_name, expr, freq_cap in rules:
        t0 = time.time()
        conn.execute(f"""
            INSERT INTO candidates
            WITH 
            s1_k AS (SELECT entity_id AS s1_id, {expr} AS k FROM s1 WHERE {expr} IS NOT NULL AND length({expr}) >= 3),
            s2_k AS (SELECT entity_id AS target_id, {expr} AS k FROM s2 WHERE {expr} IS NOT NULL AND length({expr}) >= 3),
            s3_k AS (SELECT entity_id AS target_id, {expr} AS k FROM s3 WHERE {expr} IS NOT NULL AND length({expr}) >= 3),
            target_k AS (SELECT * FROM s2_k UNION ALL SELECT * FROM s3_k),
            freq AS (
                SELECT k, COUNT(*) AS cnt
                FROM target_k
                GROUP BY k
                HAVING COUNT(*) <= {freq_cap}
            )
            SELECT s1_k.s1_id, target_k.target_id
            FROM s1_k
            JOIN target_k USING (k)
            JOIN freq USING (k);
        """)
        c_count = conn.execute("SELECT COUNT(*) FROM candidates;").fetchone()[0]
        log(f"Rule '{rule_name}' done in {time.time()-t0:.2f}s | Candidates: {c_count:,}")

    log("\nDeduplicating candidates...")
    conn.execute("CREATE OR REPLACE TABLE distinct_candidates AS SELECT DISTINCT source1_entity_id, target_entity_id FROM candidates;")
    total_distinct = conn.execute("SELECT COUNT(*) FROM distinct_candidates;").fetchone()[0]

    recovered = conn.execute("""
        SELECT COUNT(*)
        FROM gt
        JOIN distinct_candidates c ON gt.s1_id = c.source1_entity_id AND gt.matched_id = c.target_entity_id;
    """).fetchone()[0]

    recall = (recovered / gt_total) * 100.0 if gt_total > 0 else 0
    log(f"\n==================================================")
    log(f"EVALUATED CANDIDATE RECALL: {recovered:,} / {gt_total:,} ({recall:.2f}%)")
    log(f"Total Unique Candidates: {total_distinct:,}")
    log(f"==================================================")

if __name__ == "__main__":
    main()
