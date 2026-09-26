from pathlib import Path
import time
import duckdb

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work" / "person1"
TRAIN = ROOT / "dataset" / "train"
GT_FILE = TRAIN / "train_ground_truth.tsv"

def log(msg):
    print(msg, flush=True)

def evaluate_blocking_rules():
    log("==================================================")
    log("PERSON 1 — RECALL OPTIMIZATION EXPERIMENT")
    log("==================================================")
    
    conn = duckdb.connect()
    conn.execute("""
        SET memory_limit='4GB';
        SET temp_directory='artifacts/duckdb/tmp';
        SET preserve_insertion_order=false;
        SET threads=4;
    """)

    log("Loading Parquet data...")
    conn.execute(f"""
        CREATE OR REPLACE VIEW s1 AS
        SELECT entity_id, country, business_name, business_address,
               name_norm, name_compact, name_tokens,
               address_norm, address_compact, address_tokens,
               string_split(name_norm, ' ')[1] AS name_first_word,
               substr(name_compact, 1, 4) AS name_prefix4,
               substr(name_compact, 1, 6) AS name_prefix6
        FROM read_parquet('{WORK / "train_source1.parquet"}');
    """)

    conn.execute(f"""
        CREATE OR REPLACE VIEW target AS
        SELECT entity_id, country, business_name, business_address,
               name_norm, name_compact, name_tokens,
               address_norm, address_compact, address_tokens,
               string_split(name_norm, ' ')[1] AS name_first_word,
               substr(name_compact, 1, 4) AS name_prefix4,
               substr(name_compact, 1, 6) AS name_prefix6
        FROM (
            SELECT * FROM read_parquet('{WORK / "train_source2.parquet"}')
            UNION ALL
            SELECT * FROM read_parquet('{WORK / "train_source3.parquet"}')
        );
    """)

    log("Loading Ground Truth...")
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
    log(f"Total Ground Truth Positive Pairs: {gt_total:,}\n")

    conn.execute("CREATE OR REPLACE TABLE candidates (s1_id VARCHAR, target_id VARCHAR);")

    # Blocking strategies to test with safe cap per block
    strategies = [
        ("exact_name", "s1.name_norm = target.name_norm AND s1.name_norm <> ''", 1000),
        ("compact_name", "s1.name_compact = target.name_compact AND s1.name_compact <> ''", 1000),
        ("exact_address", "s1.address_norm = target.address_norm AND s1.address_norm <> ''", 1000),
        ("name_tokens", "s1.name_tokens = target.name_tokens AND s1.name_tokens <> ''", 1000),
        ("first_word_and_country", "s1.name_first_word = target.name_first_word AND s1.country = target.country AND length(s1.name_first_word) >= 4", 500),
        ("prefix6_and_country", "s1.name_prefix6 = target.name_prefix6 AND s1.country = target.country AND length(s1.name_prefix6) >= 5", 500),
    ]

    for rule_name, cond, freq_cap in strategies:
        t0 = time.time()
        
        # Extract blocking key column name for frequency cap
        key_expr = cond.split(" = ")[0]
        
        conn.execute(f"""
            INSERT INTO candidates
            WITH target_freq AS (
                SELECT {key_expr.replace('s1.', 'target.')} AS key, COUNT(*) AS cnt
                FROM target
                WHERE {key_expr.replace('s1.', 'target.')} IS NOT NULL
                GROUP BY key
                HAVING COUNT(*) <= {freq_cap}
            )
            SELECT s1.entity_id AS s1_id, target.entity_id AS target_id
            FROM s1
            JOIN target ON {cond}
            JOIN target_freq ON {key_expr} = target_freq.key;
        """)

        c_count = conn.execute("SELECT COUNT(*) FROM candidates;").fetchone()[0]
        
        # Evaluate current cumulative recall
        conn.execute("""
            CREATE OR REPLACE TABLE cur_distinct AS
            SELECT DISTINCT s1_id, target_id FROM candidates;
        """)
        
        cur_unique = conn.execute("SELECT COUNT(*) FROM cur_distinct;").fetchone()[0]
        recovered = conn.execute("""
            SELECT COUNT(*)
            FROM gt
            JOIN cur_distinct c ON gt.s1_id = c.s1_id AND gt.matched_id = c.target_id;
        """).fetchone()[0]
        
        recall = (recovered / gt_total) * 100.0
        log(f"Rule '{rule_name}' (+{time.time()-t0:.2f}s) | Unique Candidates: {cur_unique:,} | Recall: {recovered:,}/{gt_total:,} ({recall:.2f}%)")

    conn.close()

if __name__ == "__main__":
    evaluate_blocking_rules()
