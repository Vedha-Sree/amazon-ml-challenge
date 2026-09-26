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
SELECT entity_id, name_norm, address_norm, country_norm
FROM read_parquet('{S1}');
""")

DB.execute(f"""
CREATE OR REPLACE VIEW s2 AS
SELECT entity_id, name_norm, address_norm, country_norm
FROM read_parquet('{S2}');
""")

DB.execute(f"""
CREATE OR REPLACE VIEW s3 AS
SELECT entity_id, name_norm, address_norm, country_norm
FROM read_parquet('{S3}');
""")

print("=" * 100)
print("NAME FREQUENCY PROFILE")
print("=" * 100)

for source, table in [("S1", "s1"), ("S2", "s2"), ("S3", "s3")]:
    print(f"\n{source}")

    result = DB.execute(f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT name_norm) AS distinct_names,
            COUNT(*) FILTER (WHERE name_norm = '') AS empty_names,
            COUNT(*) FILTER (WHERE name_norm <> '') AS nonempty_names
        FROM {table}
    """).fetchone()

    print(f"Rows:             {result[0]:,}")
    print(f"Distinct names:   {result[1]:,}")
    print(f"Empty names:      {result[2]:,}")
    print(f"Non-empty names:  {result[3]:,}")

    print("\nMost frequent normalized names:")
    rows = DB.execute(f"""
        SELECT name_norm, COUNT(*) AS freq
        FROM {table}
        WHERE name_norm <> ''
        GROUP BY name_norm
        ORDER BY freq DESC
        LIMIT 20
    """).fetchall()

    for name, freq in rows:
        print(f"{freq:>10,}  {name[:100]}")

print("\n" + "=" * 100)
print("ADDRESS FREQUENCY PROFILE")
print("=" * 100)

for source, table in [("S1", "s1"), ("S2", "s2"), ("S3", "s3")]:
    print(f"\n{source}")

    rows = DB.execute(f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT address_norm) AS distinct_addresses,
            COUNT(*) FILTER (WHERE address_norm = '') AS empty_addresses
        FROM {table}
    """).fetchone()

    print(f"Rows:              {rows[0]:,}")
    print(f"Distinct addresses:{rows[1]:,}")
    print(f"Empty addresses:   {rows[2]:,}")

print("\n" + "=" * 100)
print("COUNTRY DISTRIBUTION")
print("=" * 100)

for source, table in [("S1", "s1"), ("S2", "s2"), ("S3", "s3")]:
    print(f"\n{source}")
    rows = DB.execute(f"""
        SELECT country_norm, COUNT(*) AS freq
        FROM {table}
        GROUP BY country_norm
        ORDER BY freq DESC
    """).fetchall()

    for country, freq in rows:
        print(f"{str(country):<20} {freq:>12,}")

DB.close()