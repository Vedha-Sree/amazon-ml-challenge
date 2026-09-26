"""
Quick check: what is the THEORETICAL maximum recall from exact/compact/token rules
at any cap?  This tells us if we need fuzzy matching.
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

total_gt_s2 = conn.execute("SELECT COUNT(*) FROM gt WHERE matched_id LIKE 'S2-%'").fetchone()[0]
total_gt_s3 = conn.execute("SELECT COUNT(*) FROM gt WHERE matched_id LIKE 'S3-%'").fetchone()[0]
print(f"Total GT pairs  S2: {total_gt_s2:,}  S3: {total_gt_s3:,}")

# Union ALL rules at very high cap (unlimited = 9999999) to see max possible recall
print("\n--- Max possible recall from all exact/token rules (no cap) vs S2 ---")
for src, tgt, gt_total in [("s2", "s2", total_gt_s2), ("s3", "s3", total_gt_s3)]:
    conn.execute(f"""
        CREATE OR REPLACE TABLE all_exact AS
        SELECT DISTINCT s1.entity_id s1id, {tgt}.entity_id tgid
        FROM s1 JOIN {tgt}
        ON (s1.name_norm = {tgt}.name_norm AND s1.name_norm <> '')
        OR (s1.name_compact = {tgt}.name_compact AND s1.name_compact <> '')
        OR (s1.name_tokens = {tgt}.name_tokens AND s1.name_tokens <> '')
        OR (s1.address_norm = {tgt}.address_norm AND s1.address_norm <> '')
        OR (s1.address_compact = {tgt}.address_compact AND s1.address_compact <> '')
        OR (s1.address_tokens = {tgt}.address_tokens AND s1.address_tokens <> '');
    """)
    pairs = conn.execute("SELECT COUNT(*) FROM all_exact").fetchone()[0]
    rec = conn.execute(f"""
        SELECT COUNT(*) FROM gt
        JOIN all_exact ON gt.s1_id = all_exact.s1id AND gt.matched_id = all_exact.tgid
        WHERE gt.matched_id LIKE '{tgt.upper()[0]}%-'
    """).fetchone()[0]
    # fix prefix
    prefix = "S2" if tgt == "s2" else "S3"
    rec2 = conn.execute(f"""
        SELECT COUNT(*) FROM gt
        JOIN all_exact ON gt.s1_id = all_exact.s1id AND gt.matched_id = all_exact.tgid
        WHERE gt.matched_id LIKE '{prefix}-%'
    """).fetchone()[0]
    print(f"  {prefix}: {pairs:>12,} pairs  |  GT recovered: {rec2:>10,} / {gt_total:,}  |  Recall: {rec2/gt_total*100:.2f}%")

conn.close()
