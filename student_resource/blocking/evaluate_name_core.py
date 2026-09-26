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
GT = "dataset/train/train_ground_truth.tsv"


def core_sql(col):
    return f"""
        CASE
            WHEN regexp_matches({col}, ' group$')
            THEN substr({col}, 1, length({col}) - 6)
            ELSE {col}
        END
    """


print("=" * 100)
print("NAME CORE — GROUND TRUTH EVALUATION")
print("=" * 100)

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


# Ground truth: one row per true S1 -> matched S2/S3 relationship
DB.execute(f"""
CREATE OR REPLACE TABLE gt AS
SELECT
    source1_entity_id AS s1_id,
    trim(x) AS matched_id
FROM read_csv(
    '{GT}',
    delim='\\t',
    header=true,
    columns={{
        'source1_entity_id': 'VARCHAR',
        'matched_entity_ids': 'VARCHAR'
    }}
),
UNNEST(string_split(matched_entity_ids, ',')) AS t(x)
WHERE trim(x) <> '';
""")

gt_total = DB.execute("""
SELECT COUNT(*) FROM gt
""").fetchone()[0]

print(f"\nTotal ground-truth pairs: {gt_total:,}")


# Materialize the transformed names once.
DB.execute(f"""
CREATE OR REPLACE TEMP TABLE a AS
SELECT
    entity_id,
    {core_sql("name_norm")} AS name_core
FROM s1
WHERE name_norm <> '';
""")

DB.execute(f"""
CREATE OR REPLACE TEMP TABLE b2 AS
SELECT
    entity_id,
    {core_sql("name_norm")} AS name_core
FROM s2
WHERE name_norm <> '';
""")

DB.execute(f"""
CREATE OR REPLACE TEMP TABLE b3 AS
SELECT
    entity_id,
    {core_sql("name_norm")} AS name_core
FROM s3
WHERE name_norm <> '';
""")


for threshold in [20, 50, 100, 200]:

    print("\n" + "=" * 100)
    print(f"FREQUENCY THRESHOLD: {threshold}")
    print("=" * 100)

    # ------------------------------------------------------------
    # S1 -> S2
    # ------------------------------------------------------------
    DB.execute(f"""
    CREATE OR REPLACE TEMP TABLE cand2 AS
    WITH fa AS (
        SELECT name_core, COUNT(*) AS n
        FROM a
        WHERE name_core <> ''
        GROUP BY name_core
    ),
    fb AS (
        SELECT name_core, COUNT(*) AS n
        FROM b2
        WHERE name_core <> ''
        GROUP BY name_core
    )
    SELECT
        a.entity_id AS s1_id,
        b2.entity_id AS candidate_id
    FROM a
    JOIN b2 USING (name_core)
    JOIN fa USING (name_core)
    JOIN fb USING (name_core)
    WHERE fa.n <= {threshold}
      AND fb.n <= {threshold};
    """)

    cand2_count = DB.execute("""
        SELECT COUNT(*) FROM cand2
    """).fetchone()[0]

    recovered2 = DB.execute("""
        SELECT COUNT(*)
        FROM gt
        JOIN cand2
          ON gt.s1_id = cand2.s1_id
         AND gt.matched_id = cand2.candidate_id
        WHERE gt.matched_id LIKE 'S2-%'
    """).fetchone()[0]

    gt2 = DB.execute("""
        SELECT COUNT(*)
        FROM gt
        WHERE matched_id LIKE 'S2-%'
    """).fetchone()[0]

    # ------------------------------------------------------------
    # S1 -> S3
    # ------------------------------------------------------------
    DB.execute(f"""
    CREATE OR REPLACE TEMP TABLE cand3 AS
    WITH fa AS (
        SELECT name_core, COUNT(*) AS n
        FROM a
        WHERE name_core <> ''
        GROUP BY name_core
    ),
    fb AS (
        SELECT name_core, COUNT(*) AS n
        FROM b3
        WHERE name_core <> ''
        GROUP BY name_core
    )
    SELECT
        a.entity_id AS s1_id,
        b3.entity_id AS candidate_id
    FROM a
    JOIN b3 USING (name_core)
    JOIN fa USING (name_core)
    JOIN fb USING (name_core)
    WHERE fa.n <= {threshold}
      AND fb.n <= {threshold};
    """)

    cand3_count = DB.execute("""
        SELECT COUNT(*) FROM cand3
    """).fetchone()[0]

    recovered3 = DB.execute("""
        SELECT COUNT(*)
        FROM gt
        JOIN cand3
          ON gt.s1_id = cand3.s1_id
         AND gt.matched_id = cand3.candidate_id
        WHERE gt.matched_id LIKE 'S3-%'
    """).fetchone()[0]

    gt3 = DB.execute("""
        SELECT COUNT(*)
        FROM gt
        WHERE matched_id LIKE 'S3-%'
    """).fetchone()[0]

    # ------------------------------------------------------------
    # Union recall
    # ------------------------------------------------------------
    recovered_union = DB.execute("""
        SELECT COUNT(*)
        FROM gt
        WHERE EXISTS (
            SELECT 1
            FROM cand2
            WHERE cand2.s1_id = gt.s1_id
              AND cand2.candidate_id = gt.matched_id
        )
        OR EXISTS (
            SELECT 1
            FROM cand3
            WHERE cand3.s1_id = gt.s1_id
              AND cand3.candidate_id = gt.matched_id
        )
    """).fetchone()[0]

    print(f"S2 candidates:       {cand2_count:,}")
    print(f"S2 recovered:         {recovered2:,}")
    print(f"S2 ground truth:      {gt2:,}")
    print(f"S2 recall:            {100 * recovered2 / gt2:.3f}%")

    print()

    print(f"S3 candidates:       {cand3_count:,}")
    print(f"S3 recovered:         {recovered3:,}")
    print(f"S3 ground truth:      {gt3:,}")
    print(f"S3 recall:            {100 * recovered3 / gt3:.3f}%")

    print()

    print(f"Union candidates:     {cand2_count + cand3_count:,}")
    print(f"Union recovered:      {recovered_union:,}")
    print(f"Union recall:         {100 * recovered_union / gt_total:.3f}%")


DB.close()