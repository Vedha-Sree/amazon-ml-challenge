from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[3]
TEST_S1 = ROOT / "dataset" / "test" / "test_source1.tsv"
OUT_MATCHING = ROOT / "output" / "matching_results.tsv"

conn = duckdb.connect()
conn.execute(f"""
    CREATE OR REPLACE TABLE s1 AS
    SELECT source1_entity_id, '' AS matched_entity_ids
    FROM read_csv('{TEST_S1}', delim='\t', header=true, columns={{'entity_id': 'VARCHAR'}})
    RENAME entity_id TO source1_entity_id;
""")

conn.execute(f"""
    COPY s1 TO '{OUT_MATCHING}' (HEADER, DELIMITER '\t');
""")
print(f"Written matching_results.tsv template with {conn.execute('SELECT COUNT(*) FROM s1').fetchone()[0]:,} rows.")
