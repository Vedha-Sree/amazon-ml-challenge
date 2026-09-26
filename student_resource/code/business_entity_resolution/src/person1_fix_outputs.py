"""
Fix output files so they pass the submission validator:
1. candidate_pairs.tsv — replace quoted-empty-string "" with truly empty field
2. matching_results.tsv — generate one row per S1 test entity (all empty matches)

Run from student_resource/ directory:
    python code/business_entity_resolution/src/person1_fix_outputs.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "output"
TEST_S1 = ROOT / "dataset" / "test" / "test_source1.tsv"

OUTPUT.mkdir(parents=True, exist_ok=True)


def fix_candidate_pairs() -> None:
    in_path  = OUTPUT / "candidate_pairs.tsv"
    out_path = OUTPUT / "candidate_pairs.tsv"
    tmp_path = OUTPUT / "candidate_pairs_tmp.tsv"

    print(f"Fixing {in_path.name}...")
    fixed = 0
    total = 0

    with open(in_path, "r", encoding="utf-8") as f_in, \
         open(tmp_path, "w", encoding="utf-8", newline="") as f_out:

        for line in f_in:
            total += 1
            # Replace tab + quoted empty string ("") with tab + empty
            if '\t""' in line:
                line = line.replace('\t""', "\t")
                fixed += 1
            f_out.write(line)

    tmp_path.replace(out_path)
    print(f"  Fixed {fixed:,} / {total:,} rows  ->  {out_path}")


def generate_matching_results() -> None:
    out_path = OUTPUT / "matching_results.tsv"
    print(f"\nGenerating {out_path.name} from {TEST_S1.name}...")

    # Read all S1 entity IDs
    s1_ids = []
    with open(TEST_S1, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts:
                s1_ids.append(parts[0])

    print(f"  S1 entities: {len(s1_ids):,}")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in s1_ids:
            f.write(f"{eid}\t\n")

    print(f"  Written: {out_path}  ({len(s1_ids):,} rows, all empty matches)")
    print("  NOTE: Person 2 (Matching Model) will populate matched_entity_ids.")


if __name__ == "__main__":
    fix_candidate_pairs()
    generate_matching_results()
    print("\nDone.")
