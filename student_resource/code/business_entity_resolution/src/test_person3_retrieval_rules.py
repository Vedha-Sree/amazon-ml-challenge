"""Test semantic and targeted blocking rules for Person 3 candidate retrieval."""
import duckdb
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work" / "person1"
TRAIN = ROOT / "dataset" / "train"
ARTIFACTS_P2 = ROOT / "artifacts" / "person2"

conn = duckdb.connect()
conn.execute("SET threads=4")
conn.execute("SET memory_limit='6GB'")

for src in ("source1", "source2", "source3"):
    conn.execute(f"CREATE VIEW {src} AS SELECT * FROM read_parquet('{(WORK / f'train_{src}.parquet').as_posix()}')")

# Ground truth sample for validation (50k pairs)
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
    JOIN read_csv('{(ARTIFACTS_P2 / 'fixed_validation_split.tsv').as_posix()}', delim='\\t', header=true) v
      ON p.s1_id = v.entity_id
    WHERE v.split = 'validation'
    LIMIT 50000;
""")

total_gt = conn.execute("SELECT COUNT(*) FROM val_gt").fetchone()[0]
print(f"Validation GT sample: {total_gt:,}")

# Let's inspect potential semantic / phonetic / postal blocking keys
conn.execute("""
    CREATE OR REPLACE VIEW s1_enriched AS
    SELECT *,
        -- Extract 5 or 6 digit postal code from address
        regexp_extract(address_norm, '(\\b\\d{5,6}\\b)', 1) AS postal_code,
        -- Extract longest word (len >= 6) from name_norm
        (SELECT arg_max(w, length(w)) FROM (SELECT unnest(string_split(name_norm, ' ')) AS w) WHERE length(w) >= 6) AS longest_name_word,
        -- First 3 chars of name
        substr(name_compact, 1, 3) AS name_p3
    FROM source1;
""")

for target in ("source2", "source3"):
    conn.execute(f"""
        CREATE OR REPLACE VIEW {target}_enriched AS
        SELECT *,
            regexp_extract(address_norm, '(\\b\\d{5,6}\\b)', 1) AS postal_code,
            (SELECT arg_max(w, length(w)) FROM (SELECT unnest(string_split(name_norm, ' ')) AS w) WHERE length(w) >= 6) AS longest_name_word,
            substr(name_compact, 1, 3) AS name_p3
        FROM {target};
    """)

# Test Rule 1: postal_code + name_p3 (cap 30)
for target in ("source2", "source3"):
    conn.execute(f"""
        CREATE OR REPLACE TABLE test_postal_cand_{target} AS
        SELECT DISTINCT s1.entity_id as s1_id, t.entity_id as target_id
        FROM s1_enriched s1
        JOIN {target}_enriched t
          ON s1.postal_code = t.postal_code
         AND s1.name_p3 = t.name_p3
         AND s1.country = t.country
        WHERE s1.postal_code <> '' AND length(s1.name_p3) >= 3;
    """)

# Test Rule 2: longest_name_word + country (cap 15)
for target in ("source2", "source3"):
    conn.execute(f"""
        CREATE OR REPLACE TABLE test_longest_cand_{target} AS
        WITH
        s1_freq AS (
            SELECT longest_name_word AS key, country FROM s1_enriched
            WHERE longest_name_word IS NOT NULL AND length(longest_name_word) >= 6
            GROUP BY 1, 2 HAVING COUNT(*) <= 15
        ),
        tgt_freq AS (
            SELECT longest_name_word AS key, country FROM {target}_enriched
            WHERE longest_name_word IS NOT NULL AND length(longest_name_word) >= 6
            GROUP BY 1, 2 HAVING COUNT(*) <= 15
        ),
        el_keys AS (
            SELECT s1_freq.key, s1_freq.country FROM s1_freq JOIN tgt_freq USING (key, country)
        )
        SELECT DISTINCT s1.entity_id as s1_id, t.entity_id as target_id
        FROM s1_enriched s1
        JOIN el_keys k ON s1.longest_name_word = k.key AND s1.country = k.country
        JOIN {target}_enriched t ON t.longest_name_word = k.key AND t.country = k.country;
    """)

# Measure recovery of val_gt
postal_rec = conn.execute("""
    SELECT COUNT(*) FROM val_gt g
    WHERE EXISTS (SELECT 1 FROM test_postal_cand_source2 c WHERE c.s1_id=g.s1_id AND c.target_id=g.target_id)
       OR EXISTS (SELECT 1 FROM test_postal_cand_source3 c WHERE c.s1_id=g.s1_id AND c.target_id=g.target_id);
""").fetchone()[0]

longest_rec = conn.execute("""
    SELECT COUNT(*) FROM val_gt g
    WHERE EXISTS (SELECT 1 FROM test_longest_cand_source2 c WHERE c.s1_id=g.s1_id AND c.target_id=g.target_id)
       OR EXISTS (SELECT 1 FROM test_longest_cand_source3 c WHERE c.s1_id=g.s1_id AND c.target_id=g.target_id);
""").fetchone()[0]

print(f"Postal code + name_p3 recovered: {postal_rec:,} / {total_gt:,} ({postal_rec/total_gt*100:.2f}%)")
print(f"Longest name word (cap 15) recovered: {longest_rec:,} / {total_gt:,} ({longest_rec/total_gt*100:.2f}%)")

