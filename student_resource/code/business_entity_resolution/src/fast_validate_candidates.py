from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[3]
TEST_S1 = ROOT / "dataset" / "test" / "test_source1.tsv"
CAND_FILE = ROOT / "output" / "candidate_pairs.tsv"

def validate():
    print("==================================================")
    print("STREAMING VALIDATOR FOR CANDIDATE PAIRS")
    print("==================================================")
    
    # 1. Check file existence
    if not CAND_FILE.exists():
        print("FAIL: candidate_pairs.tsv does not exist!")
        return False
        
    file_size_mb = CAND_FILE.stat().st_size / (1024 * 1024)
    print(f"File: {CAND_FILE.name} ({file_size_mb:.1f} MB)")

    # 2. Stream and validate format line by line
    total_rows = 0
    empty_cand_rows = 0
    total_candidate_pairs = 0
    
    with open(CAND_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        
        if header != ["source1_entity_id", "candidate_entity_ids"]:
            print(f"FAIL: Invalid header: {header}")
            return False
            
        for line_num, row in enumerate(reader, start=2):
            if len(row) != 2:
                print(f"FAIL line {line_num}: expected 2 tab-separated columns, got {len(row)}")
                return False
                
            s1_id, cands = row[0], row[1]
            
            if not s1_id.startswith("S1-"):
                print(f"FAIL line {line_num}: invalid S1 entity ID format '{s1_id}'")
                return False
                
            if cands:
                cand_list = cands.split(",")
                if len(cand_list) != len(set(cand_list)):
                    print(f"FAIL line {line_num}: duplicate candidate IDs in '{cands}'")
                    return False
                for c in cand_list:
                    if not (c.startswith("S2-") or c.startswith("S3-")):
                        print(f"FAIL line {line_num}: invalid candidate ID '{c}'")
                        return False
                total_candidate_pairs += len(cand_list)
            else:
                empty_cand_rows += 1
                
            total_rows += 1

    print("\n--- VALIDATION RESULTS ---")
    print(f"Total S1 Rows: {total_rows:,}")
    print(f"Total Candidate Pairs: {total_candidate_pairs:,}")
    print(f"Average Candidates / S1: {total_candidate_pairs / total_rows:.1f}")
    print(f"Empty Candidate Rows: {empty_cand_rows:,}")
    print("RESULT: PASS! candidate_pairs.tsv is strictly valid!")
    return True

if __name__ == "__main__":
    validate()
