"""Benchmark Person 3 semantic retrieval vs Person 1 on validation data."""
import duckdb
from pathlib import Path
import time
import json

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work" / "person1"
TRAIN = ROOT / "dataset" / "train"
ARTIFACTS_P2 = ROOT / "artifacts" / "person2"
ARTIFACTS_P3 = ROOT / "artifacts" / "person3"

conn = duckdb.connect()
conn.execute("SET threads=4")
conn.execute("SET memory_limit='6GB'")

print("Initializing views...")
for src in ("source1", "source2", "source3"):
    conn.execute(f"CREATE VIEW {src} AS SELECT * FROM read_parquet('{(WORK / f'train_{src}.parquet').as_posix()}')")

# Fixed validation split S1 entities (sample 50,000 S1 entities for benchmarking)
conn.execute(f"""
    CREATE TABLE val_s1_sample AS
    SELECT entity_id
    FROM read_csv('{(ARTIFACTS_P2 / 'fixed_validation_split.tsv').as_posix()}', delim='\\t', header=true)
    WHERE split = 'validation'
    LIMIT 50000;
""")
num_val_s1 = conn.execute("SELECT COUNT(*) FROM val_s1_sample").fetchone()[0]
print(f"Benchmarking with {num_val_s1:,} validation S1 entities...")

# Validation ground truth for this sample
conn.execute(f"""
    CREATE TABLE val_gt AS
    WITH parsed_gt AS (
        SELECT source1_entity_id AS s1_id, trim(x) AS target_id
        FROM read_csv('{(TRAIN / 'train_ground_truth.tsv').as_posix()}', delim='\\t', header=true,
                      columns={{'source1_entity_id': 'VARCHAR', 'matched_entity_ids': 'VARCHAR'}}),
        UNNEST(string_split(matched_entity_ids, ',')) AS t(x)
        WHERE trim(x) <> ''
    )
    SELECT p.s1_id, p.target_id
    FROM parsed_gt p
    JOIN val_s1_sample v ON p.s1_id = v.entity_id;
""")
total_gt_pairs = conn.execute("SELECT COUNT(*) FROM val_gt").fetchone()[0]
print(f"Total true positive pairs in sample: {total_gt_pairs:,}")

# 1. Person 1 Baseline Blocking Candidates for this sample
print("\n--- Generating Person 1 Candidates ---")
t0 = time.time()
conn.execute("""
    CREATE OR REPLACE VIEW s1_p1 AS
    SELECT s1.*,
      CASE WHEN length(trim(string_split(name_norm, ' ')[1])) >= 5
           THEN trim(string_split(name_norm, ' ')[1]) END AS name_first_word,
      CASE WHEN length(name_compact) >= 6
           THEN substr(name_compact, 1, 6) END AS name_prefix6
    FROM source1 s1
    JOIN val_s1_sample v ON s1.entity_id = v.entity_id;
""")

for target in ("source2", "source3"):
    conn.execute(f"""
        CREATE OR REPLACE VIEW {target}_p1 AS
        SELECT *,
          CASE WHEN length(trim(string_split(name_norm, ' ')[1])) >= 5
               THEN trim(string_split(name_norm, ' ')[1]) END AS name_first_word,
          CASE WHEN length(name_compact) >= 6
               THEN substr(name_compact, 1, 6) END AS name_prefix6
        FROM {target};
    """)

# Person 1 blocking rules
P1_RULES = [
    ("name_norm", 50, False), ("name_compact", 50, False),
    ("address_norm", 50, False), ("address_compact", 50, False),
    ("name_tokens", 50, False), ("address_tokens", 50, False),
    ("name_first_word", 20, True), ("name_prefix6", 15, True),
]

conn.execute("CREATE OR REPLACE TABLE p1_candidates(s1_id VARCHAR, target_id VARCHAR)")
for target in ("source2", "source3"):
    for key, cap, country_guard in P1_RULES:
        country = f"AND a.country = b.country" if country_guard else ""
        conn.execute(f"""
            INSERT INTO p1_candidates
            SELECT DISTINCT a.entity_id, b.entity_id
            FROM s1_p1 a JOIN {target}_p1 b ON a.{key}=b.{key} {country}
            JOIN (SELECT {key} k FROM s1_p1 WHERE {key} <> '' GROUP BY 1 HAVING count(*) <= {cap}) x
              ON a.{key}=x.k
            JOIN (SELECT {key} k FROM {target}_p1 WHERE {key} <> '' GROUP BY 1 HAVING count(*) <= {cap}) y
              ON b.{key}=y.k
            WHERE a.{key} <> '' AND b.{key} <> ''
        """)

conn.execute("CREATE OR REPLACE TABLE p1_candidates_distinct AS SELECT DISTINCT * FROM p1_candidates")
p1_cand_count = conn.execute("SELECT COUNT(*) FROM p1_candidates_distinct").fetchone()[0]
p1_recovered = conn.execute("""
    SELECT COUNT(*) FROM val_gt g
    JOIN p1_candidates_distinct c ON g.s1_id = c.s1_id AND g.target_id = c.target_id
""").fetchone()[0]
p1_recall = p1_recovered / total_gt_pairs * 100
t_p1 = time.time() - t0
print(f"Person 1 candidates: {p1_cand_count:,} ({p1_cand_count/num_val_s1:.1f} per S1) in {t_p1:.2f}s")
print(f"Person 1 Candidate Recall: {p1_recovered:,} / {total_gt_pairs:,} ({p1_recall:.2f}%)")

# 2. Person 3 Semantic / Additive Blocking Rules
print("\n--- Generating Person 3 Semantic Candidates ---")
t0 = time.time()

# Enriched views with transliterated prefix, core name, and street number
# In SQL, we can define transliteration mapping and regex street numbers:
conn.execute("""
    CREATE OR REPLACE VIEW s1_p3 AS
    SELECT s1.*,
      -- First street number from address
      regexp_extract(address_norm, '(\\b\\d+\\b)', 1) AS addr_first_num,
      -- First 3 chars of name
      substr(name_compact, 1, 3) AS name_p3,
      -- Longest word (len >= 6)
      (SELECT arg_max(w, length(w)) FROM (SELECT unnest(string_split(name_norm, ' ')) AS w) WHERE length(w) >= 6) AS name_longest_word
    FROM s1_p1 s1;
""")

for target in ("source2", "source3"):
    conn.execute(f"""
        CREATE OR REPLACE VIEW {target}_p3 AS
        SELECT t.*,
          regexp_extract(address_norm, '(\\b\\d+\\b)', 1) AS addr_first_num,
          substr(name_compact, 1, 3) AS name_p3,
          (SELECT arg_max(w, length(w)) FROM (SELECT unnest(string_split(name_norm, ' ')) AS w) WHERE length(w) >= 6) AS name_longest_word
        FROM {target}_p1 t;
    """)

# Person 3 candidate rules:
# Rule A: Longest word of business name (cap 15, len >= 6) + country
# Rule B: First address number + first 3 letters of name + country (cap 20)
conn.execute("CREATE OR REPLACE TABLE p3_candidates(s1_id VARCHAR, target_id VARCHAR)")

for target in ("source2", "source3"):
    # Rule A: Longest word
    conn.execute(f"""
        INSERT INTO p3_candidates
        WITH
        s1_freq AS (
            SELECT name_longest_word AS k, country FROM s1_p3
            WHERE name_longest_word IS NOT NULL AND length(name_longest_word) >= 6
            GROUP BY 1, 2 HAVING COUNT(*) <= 15
        ),
        tgt_freq AS (
            SELECT name_longest_word AS k, country FROM {target}_p3
            WHERE name_longest_word IS NOT NULL AND length(name_longest_word) >= 6
            GROUP BY 1, 2 HAVING COUNT(*) <= 15
        ),
        el_k AS (
            SELECT s1_freq.k, s1_freq.country FROM s1_freq JOIN tgt_freq USING (k, country)
        )
        SELECT DISTINCT a.entity_id, b.entity_id
        FROM s1_p3 a
        JOIN el_k ON a.name_longest_word = el_k.k AND a.country = el_k.country
        JOIN {target}_p3 b ON b.name_longest_word = el_k.k AND b.country = el_k.country;
    """)

    # Rule B: Address number + name prefix 3
    conn.execute(f"""
        INSERT INTO p3_candidates
        WITH
        s1_freq AS (
            SELECT addr_first_num AS num, name_p3, country FROM s1_p3
            WHERE addr_first_num <> '' AND length(name_p3) >= 3
            GROUP BY 1, 2, 3 HAVING COUNT(*) <= 20
        ),
        tgt_freq AS (
            SELECT addr_first_num AS num, name_p3, country FROM {target}_p3
            WHERE addr_first_num <> '' AND length(name_p3) >= 3
            GROUP BY 1, 2, 3 HAVING COUNT(*) <= 20
        ),
        el_np AS (
            SELECT s1_freq.num, s1_freq.name_p3, s1_freq.country FROM s1_freq JOIN tgt_freq USING (num, name_p3, country)
        )
        SELECT DISTINCT a.entity_id, b.entity_id
        FROM s1_p3 a
        JOIN el_np ON a.addr_first_num = el_np.num AND a.name_p3 = el_np.name_p3 AND a.country = el_np.country
        JOIN {target}_p3 b ON b.addr_first_num = el_np.num AND b.name_p3 = el_np.name_p3 AND b.country = el_np.country;
    """)

conn.execute("CREATE OR REPLACE TABLE p3_candidates_distinct AS SELECT DISTINCT * FROM p3_candidates")
p3_cand_count = conn.execute("SELECT COUNT(*) FROM p3_candidates_distinct").fetchone()[0]
p3_recovered = conn.execute("""
    SELECT COUNT(*) FROM val_gt g
    JOIN p3_candidates_distinct c ON g.s1_id = c.s1_id AND g.target_id = c.target_id
""").fetchone()[0]
p3_recall = p3_recovered / total_gt_pairs * 100
t_p3 = time.time() - t0
print(f"Person 3 candidates alone: {p3_cand_count:,} in {t_p3:.2f}s")
print(f"Person 3 Candidate Recall alone: {p3_recovered:,} / {total_gt_pairs:,} ({p3_recall:.2f}%)")

# 3. UNION Candidates (Person 1 UNION Person 3)
print("\n--- Evaluating Candidate UNION ---")
conn.execute("""
    CREATE OR REPLACE TABLE union_candidates AS
    SELECT * FROM p1_candidates_distinct
    UNION
    SELECT * FROM p3_candidates_distinct;
""")
union_cand_count = conn.execute("SELECT COUNT(*) FROM union_candidates").fetchone()[0]
union_recovered = conn.execute("""
    SELECT COUNT(*) FROM val_gt g
    JOIN union_candidates c ON g.s1_id = c.s1_id AND g.target_id = c.target_id
""").fetchone()[0]
union_recall = union_recovered / total_gt_pairs * 100
overlap_count = conn.execute("""
    SELECT COUNT(*) FROM p1_candidates_distinct p1
    JOIN p3_candidates_distinct p3 ON p1.s1_id = p3.s1_id AND p1.target_id = p3.target_id
""").fetchone()[0]

added_candidates = union_cand_count - p1_cand_count
recall_gain = union_recall - p1_recall

# Candidate percentiles per S1
cands_per_s1 = conn.execute("""
    SELECT count(*) as cnt FROM union_candidates GROUP BY s1_id
""").fetchall()
counts = sorted([c[0] for c in cands_per_s1])
median_cands = counts[len(counts)//2] if counts else 0
p95_cands = counts[int(len(counts)*0.95)] if counts else 0
max_cands = counts[-1] if counts else 0

print(f"UNION Candidates Total: {union_cand_count:,}")
print(f"New Candidates Added: +{added_candidates:,} (+{added_candidates/num_val_s1:.2f} per S1)")
print(f"Overlap: {overlap_count:,}")
print(f"Candidate Recall (Person 1): {p1_recall:.2f}%")
print(f"Candidate Recall (UNION):    {union_recall:.2f}% (GAIN: +{recall_gain:.2f}%!)")
print(f"Candidate distribution per S1: median={median_cands}, 95th-pct={p95_cands}, max={max_cands}")

