"""
Analyze how both-side frequency cap affects candidate recall and pair count.
Run from student_resource/ directory.
"""
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work" / "person1"
TRAIN = ROOT / "dataset" / "train"
GT_FILE = TRAIN / "train_ground_truth.tsv"

conn = duckdb.connect()
conn.execute("SET memory_limit='6GB'")

# Load ground truth
gt_path = GT_FILE.as_posix()
conn.execute(f"""
    CREATE TABLE gt AS
    SELECT source1_entity_id AS s1_id, trim(x) AS matched_id
    FROM read_csv(
        '{gt_path}',
        delim='\t',
        header=true,
        columns={{'source1_entity_id': 'VARCHAR', 'matched_entity_ids': 'VARCHAR'}}
    ),
    UNNEST(string_split(matched_entity_ids, ',')) AS t(x)
    WHERE trim(x) <> '';
""")

s1_path = (WORK / "train_source1.parquet").as_posix()
s2_path = (WORK / "train_source2.parquet").as_posix()
s3_path = (WORK / "train_source3.parquet").as_posix()

conn.execute(f"CREATE VIEW s1 AS SELECT * FROM read_parquet('{s1_path}')")
conn.execute(f"CREATE VIEW s2 AS SELECT * FROM read_parquet('{s2_path}')")
conn.execute(f"CREATE VIEW s3 AS SELECT * FROM read_parquet('{s3_path}')")

total_gt = conn.execute("SELECT COUNT(*) FROM gt").fetchone()[0]
total_s2_gt = conn.execute("SELECT COUNT(*) FROM gt WHERE matched_id LIKE 'S2-%'").fetchone()[0]
total_s3_gt = conn.execute("SELECT COUNT(*) FROM gt WHERE matched_id LIKE 'S3-%'").fetchone()[0]
print(f"\nTotal GT pairs: {total_gt:,}  (S2: {total_s2_gt:,}, S3: {total_s3_gt:,})")

rules = [
    ("exact_name",    "name_norm"),
    ("compact_name",  "name_compact"),
    ("name_tokens",   "name_tokens"),
    ("exact_address", "address_norm"),
]

print("\n--- Cap sensitivity per rule (S2) ---")
print(f"{'Rule':<18} {'Cap':>6} | {'Pairs':>10} | {'Recovered':>10} | {'Recall%':>8}")
print("-" * 65)

for rule, key in rules:
    for cap in [5, 10, 20, 50, 100, 500]:
        conn.execute(f"""
            CREATE OR REPLACE TABLE cands AS
            WITH
            s1f AS (
                SELECT {key} AS k FROM s1
                WHERE {key} IS NOT NULL AND {key} <> ''
                GROUP BY {key} HAVING COUNT(*) <= {cap}
            ),
            s2f AS (
                SELECT {key} AS k FROM s2
                WHERE {key} IS NOT NULL AND {key} <> ''
                GROUP BY {key} HAVING COUNT(*) <= {cap}
            ),
            elig AS (SELECT s1f.k FROM s1f JOIN s2f USING (k))
            SELECT DISTINCT s1.entity_id AS s1id, s2.entity_id AS tgid
            FROM s1 JOIN elig ON s1.{key} = elig.k
                    JOIN s2  ON s1.{key} = s2.{key}
            WHERE s1.{key} <> ''
        """)
        pairs = conn.execute("SELECT COUNT(*) FROM cands").fetchone()[0]
        rec = conn.execute("""
            SELECT COUNT(*) FROM gt
            JOIN cands ON gt.s1_id = cands.s1id AND gt.matched_id = cands.tgid
            WHERE gt.matched_id LIKE 'S2-%'
        """).fetchone()[0]
        print(f"  {rule:<16} {cap:>6} | {pairs:>10,} | {rec:>10,} / {total_s2_gt:,} | {rec/total_s2_gt*100:>7.1f}%")
    print()

conn.close()
