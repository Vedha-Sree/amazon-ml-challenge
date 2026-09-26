import duckdb

DB = duckdb.connect()

DB.execute("""
SET memory_limit='4GB';
SET temp_directory='artifacts/duckdb/tmp';
SET preserve_insertion_order=false;
""")

S1 = "artifacts/normalized/train_source1.parquet"
S2 = "artifacts/normalized/train_source2.parquet"
S3 = "artifacts/normalized/train_source3_part_*.parquet"


DB.execute(f"""
CREATE OR REPLACE VIEW s1 AS
SELECT entity_id, name_norm
FROM read_parquet('{S1}');
""")

DB.execute(f"""
CREATE OR REPLACE VIEW s2 AS
SELECT entity_id, name_norm
FROM read_parquet('{S2}');
""")

DB.execute(f"""
CREATE OR REPLACE VIEW s3 AS
SELECT entity_id, name_norm
FROM read_parquet('{S3}');
""")


# Remove ONLY a trailing "group".
# This avoids accidentally deleting "group" when it is part
# of the actual business name.
def core_sql(col):
    return f"""
        CASE
            WHEN regexp_matches({col}, ' group$')
            THEN substr({col}, 1, length({col}) - 6)
            ELSE {col}
        END
    """


print("=" * 100)
print("VERIFYING NAME CORE TRANSFORMATION")
print("=" * 100)

for source, table in [("S1", "s1"), ("S2", "s2"), ("S3", "s3")]:
    print(f"\n{source}")

    rows = DB.execute(f"""
        SELECT
            name_norm,
            {core_sql("name_norm")} AS name_core,
            COUNT(*) AS freq
        FROM {table}
        WHERE name_norm <> ''
        GROUP BY name_norm
        ORDER BY freq DESC
        LIMIT 20
    """).fetchall()

    for name, core, freq in rows:
        print(f"{freq:>6,} | {name:<45} -> {core}")


print("\n" + "=" * 100)
print("S1 -> S2 NAME CORE CANDIDATES")
print("=" * 100)

result = DB.execute(f"""
    WITH a AS (
        SELECT
            entity_id,
            {core_sql("name_norm")} AS name_core
        FROM s1
        WHERE name_norm <> ''
    ),
    b AS (
        SELECT
            entity_id,
            {core_sql("name_norm")} AS name_core
        FROM s2
        WHERE name_norm <> ''
    ),
    fa AS (
        SELECT name_core, COUNT(*) AS freq
        FROM a
        WHERE name_core <> ''
        GROUP BY name_core
    ),
    fb AS (
        SELECT name_core, COUNT(*) AS freq
        FROM b
        WHERE name_core <> ''
        GROUP BY name_core
    )
    SELECT
        COUNT(*) AS candidate_pairs
    FROM a
    JOIN b USING (name_core)
    JOIN fa USING (name_core)
    JOIN fb USING (name_core)
    WHERE fa.freq <= 100
      AND fb.freq <= 100
""").fetchone()

print(f"Candidate pairs, both sides <=100: {result[0]:,}")


print("\n" + "=" * 100)
print("NAME CORE BLOCK SIZE DISTRIBUTION")
print("=" * 100)

rows = DB.execute(f"""
    WITH a AS (
        SELECT {core_sql("name_norm")} AS name_core
        FROM s1
        WHERE name_norm <> ''
    ),
    b AS (
        SELECT {core_sql("name_norm")} AS name_core
        FROM s2
        WHERE name_norm <> ''
    ),
    fa AS (
        SELECT name_core, COUNT(*) AS n
        FROM a
        WHERE name_core <> ''
        GROUP BY name_core
    ),
    fb AS (
        SELECT name_core, COUNT(*) AS n
        FROM b
        WHERE name_core <> ''
        GROUP BY name_core
    )
    SELECT
        COUNT(*) AS distinct_keys,
        SUM(fa.n * fb.n) AS candidate_pairs
    FROM fa
    JOIN fb USING (name_core)
    WHERE fa.n <= 100
      AND fb.n <= 100
""").fetchone()

print(f"Distinct keys <=100 on both sides: {rows[0]:,}")
print(f"Candidate pairs <=100 on both sides: {rows[1]:,}")


print("\n" + "=" * 100)
print("LARGEST ELIGIBLE BLOCKS")
print("=" * 100)

rows = DB.execute(f"""
    WITH a AS (
        SELECT {core_sql("name_norm")} AS name_core
        FROM s1
        WHERE name_norm <> ''
    ),
    b AS (
        SELECT {core_sql("name_norm")} AS name_core
        FROM s2
        WHERE name_norm <> ''
    ),
    fa AS (
        SELECT name_core, COUNT(*) AS n
        FROM a
        WHERE name_core <> ''
        GROUP BY name_core
    ),
    fb AS (
        SELECT name_core, COUNT(*) AS n
        FROM b
        WHERE name_core <> ''
        GROUP BY name_core
    )
    SELECT
        fa.name_core,
        fa.n AS s1_count,
        fb.n AS s2_count,
        fa.n * fb.n AS candidate_pairs
    FROM fa
    JOIN fb USING (name_core)
    WHERE fa.n <= 100
      AND fb.n <= 100
    ORDER BY candidate_pairs DESC
    LIMIT 30
""").fetchall()

for name, n1, n2, pairs in rows:
    print(f"{pairs:>10,} | S1={n1:>3,} S2={n2:>3,} | {name}")


DB.close()