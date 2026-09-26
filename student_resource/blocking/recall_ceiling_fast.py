"""
Fast recall ceiling check - no cap on exact/token rules.
Shows the theoretical max recall achievable without fuzzy matching.
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
conn.execute("SET threads=4")

s1p = (WORK / "train_source1.parquet").as_posix()
s2p = (WORK / "train_source2.parquet").as_posix()
s3p = (WORK / "train_source3.parquet").as_posix()
gt_p = GT_FILE.as_posix()

# Load as tables so joins are fast
conn.execute(f"""
    CREATE TABLE s1 AS
    SELECT entity_id, name_norm, name_compact, name_tokens,
           address_norm, address_compact, address_tokens
    FROM read_parquet('{s1p}')
""")
conn.execute(f"""
    CREATE TABLE s2 AS
    SELECT entity_id, name_norm, name_compact, name_tokens,
           address_norm, address_compact, address_tokens
    FROM read_parquet('{s2p}')
""")
conn.execute(f"""
    CREATE TABLE s3 AS
    SELECT entity_id, name_norm, name_compact, name_tokens,
           address_norm, address_compact, address_tokens
    FROM read_parquet('{s3p}')
""")

conn.execute(f"""
    CREATE TABLE gt AS
    SELECT source1_entity_id AS s1_id, trim(x) AS matched_id
    FROM read_csv(
        '{gt_p}',
        delim='\t',
        header=true,
        columns={{'source1_entity_id': 'VARCHAR', 'matched_entity_ids': 'VARCHAR'}}
    ),
    UNNEST(string_split(matched_entity_ids, ',')) AS t(x)
    WHERE trim(x) <> ''
""")

total_s2 = conn.execute("SELECT COUNT(*) FROM gt WHERE matched_id LIKE 'S2-%'").fetchone()[0]
total_s3 = conn.execute("SELECT COUNT(*) FROM gt WHERE matched_id LIKE 'S3-%'").fetchone()[0]
total_gt = total_s2 + total_s3
print(f"\nTotal GT pairs: {total_gt:,}  (S2: {total_s2:,}, S3: {total_s3:,})")

rules = [
    ("exact_name",    "name_norm"),
    ("compact_name",  "name_compact"),
    ("name_tokens",   "name_tokens"),
    ("exact_address", "address_norm"),
    ("addr_compact",  "address_compact"),
    ("addr_tokens",   "address_tokens"),
]

print(f"\n{'Rule':<18} | {'S2 pairs':>12} | {'S2 recall%':>10} | {'S3 pairs':>12} | {'S3 recall%':>10}")
print("-" * 80)

combined_s2 = set()
combined_s3 = set()

for rule, key in rules:
    # S2
    conn.execute(f"""
        CREATE OR REPLACE TABLE cands_s2 AS
        SELECT DISTINCT s1.entity_id AS s1id, s2.entity_id AS tgid
        FROM s1 JOIN s2 ON s1.{key} = s2.{key}
        WHERE s1.{key} IS NOT NULL AND s1.{key} <> ''
    """)
    pairs_s2 = conn.execute("SELECT COUNT(*) FROM cands_s2").fetchone()[0]
    rec_s2 = conn.execute("""
        SELECT COUNT(*) FROM gt
        JOIN cands_s2 ON gt.s1_id = cands_s2.s1id AND gt.matched_id = cands_s2.tgid
        WHERE gt.matched_id LIKE 'S2-%'
    """).fetchone()[0]

    # S3
    conn.execute(f"""
        CREATE OR REPLACE TABLE cands_s3 AS
        SELECT DISTINCT s1.entity_id AS s1id, s3.entity_id AS tgid
        FROM s1 JOIN s3 ON s1.{key} = s3.{key}
        WHERE s1.{key} IS NOT NULL AND s1.{key} <> ''
    """)
    pairs_s3 = conn.execute("SELECT COUNT(*) FROM cands_s3").fetchone()[0]
    rec_s3 = conn.execute("""
        SELECT COUNT(*) FROM gt
        JOIN cands_s3 ON gt.s1_id = cands_s3.s1id AND gt.matched_id = cands_s3.tgid
        WHERE gt.matched_id LIKE 'S3-%'
    """).fetchone()[0]

    print(
        f"  {rule:<16} | {pairs_s2:>12,} | {rec_s2/total_s2*100:>9.1f}% "
        f"| {pairs_s3:>12,} | {rec_s3/total_s3*100:>9.1f}%"
    )

# Cumulative union recall
print("\n--- Cumulative union recall (all rules combined, no cap) ---")
rules_union = [
    ("name_norm",),
    ("name_compact",),
    ("name_tokens",),
    ("address_norm",),
    ("address_compact",),
    ("address_tokens",),
]
for i in range(1, len(rules_union) + 1):
    parts_s2 = " OR ".join(
        [f"(s1.{k[0]} = s2.{k[0]} AND s1.{k[0]} <> '')" for k in rules_union[:i]]
    )
    conn.execute(f"""
        CREATE OR REPLACE TABLE cum_s2 AS
        SELECT DISTINCT s1.entity_id AS s1id, s2.entity_id AS tgid
        FROM s1 JOIN s2 ON {parts_s2}
    """)
    p2 = conn.execute("SELECT COUNT(*) FROM cum_s2").fetchone()[0]
    r2 = conn.execute("""
        SELECT COUNT(*) FROM gt
        JOIN cum_s2 ON gt.s1_id = cum_s2.s1id AND gt.matched_id = cum_s2.tgid
        WHERE gt.matched_id LIKE 'S2-%'
    """).fetchone()[0]

    parts_s3 = parts_s2.replace("s2", "s3")
    conn.execute(f"""
        CREATE OR REPLACE TABLE cum_s3 AS
        SELECT DISTINCT s1.entity_id AS s1id, s3.entity_id AS tgid
        FROM s1 JOIN s3 ON {parts_s3}
    """)
    p3 = conn.execute("SELECT COUNT(*) FROM cum_s3").fetchone()[0]
    r3 = conn.execute("""
        SELECT COUNT(*) FROM gt
        JOIN cum_s3 ON gt.s1_id = cum_s3.s1id AND gt.matched_id = cum_s3.tgid
        WHERE gt.matched_id LIKE 'S3-%'
    """).fetchone()[0]

    rule_names = [r[0] for r in rules_union[:i]]
    total_combined = r2 + r3
    print(
        f"  {i} rules ({', '.join(rule_names[-1:]):<18}): "
        f"S2={r2/total_s2*100:.1f}%  S3={r3/total_s3*100:.1f}%  "
        f"Combined={total_combined/total_gt*100:.1f}%"
    )

conn.close()
