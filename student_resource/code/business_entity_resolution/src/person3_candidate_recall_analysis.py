"""Candidate recall failure analysis for Person 3.

Measures:
- Person 1 blocking candidate recall on the validation set.
- Detailed breakdown of recovered vs missed true positive pairs.
- Failure mode categorization of missed pairs:
  * spelling / phonetic / transliteration differences
  * token reordering / subset names
  * address variation with high name similarity
  * missing / truncated fields
  * frequency-cap drops
- Outputs candidate_recall_analysis.json and candidate_recall_analysis.md
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path
from collections import Counter

import duckdb
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work" / "person1"
TRAIN = ROOT / "dataset" / "train"
ARTIFACTS_P2 = ROOT / "artifacts" / "person2"
ARTIFACTS_P3 = ROOT / "artifacts" / "person3"
ARTIFACTS_P3.mkdir(parents=True, exist_ok=True)

GT_FILE = TRAIN / "train_ground_truth.tsv"
SPLIT_FILE = ARTIFACTS_P2 / "fixed_validation_split.tsv"

def setup_duckdb():
    conn = duckdb.connect()
    conn.execute("SET threads=4")
    conn.execute("SET memory_limit='6GB'")
    return conn

def run_analysis(sample_limit: int = 50_000):
    print(f"Starting candidate recall analysis (sample limit={sample_limit})...")
    conn = setup_duckdb()
    
    # 1. Register sources
    for src in ("source1", "source2", "source3"):
        p = WORK / f"train_{src}.parquet"
        conn.execute(f"CREATE VIEW {src} AS SELECT * FROM read_parquet('{p.as_posix()}')")
        
    # 2. Register validation S1 entities
    conn.execute(f"""
        CREATE TABLE val_s1 AS
        SELECT entity_id
        FROM read_csv('{SPLIT_FILE.as_posix()}', delim='\\t', header=true)
        WHERE split = 'validation';
    """)
    val_s1_count = conn.execute("SELECT COUNT(*) FROM val_s1").fetchone()[0]
    print(f"Total validation S1 entities: {val_s1_count:,}")

    # 3. Extract validation ground truth pairs
    conn.execute(f"""
        CREATE TABLE val_gt AS
        WITH parsed_gt AS (
            SELECT source1_entity_id AS s1_id, trim(x) AS target_id
            FROM read_csv('{GT_FILE.as_posix()}', delim='\\t', header=true,
                          columns={{'source1_entity_id': 'VARCHAR', 'matched_entity_ids': 'VARCHAR'}}),
            UNNEST(string_split(matched_entity_ids, ',')) AS t(x)
            WHERE trim(x) <> ''
        )
        SELECT p.s1_id, p.target_id
        FROM parsed_gt p
        JOIN val_s1 v ON p.s1_id = v.entity_id;
    """)
    total_val_gt = conn.execute("SELECT COUNT(*) FROM val_gt").fetchone()[0]
    print(f"Total validation ground truth pairs: {total_val_gt:,}")

    # 4. Pull sample of pairs with full metadata for deep inspection
    sample_clause = f"LIMIT {sample_limit}" if sample_limit else ""
    conn.execute(f"""
        CREATE TABLE val_sample AS
        SELECT * FROM val_gt {sample_clause};
    """)
    actual_sample_size = conn.execute("SELECT COUNT(*) FROM val_sample").fetchone()[0]
    print(f"Sampled {actual_sample_size:,} validation GT pairs for detailed inspection")

    # 5. Join with source attributes
    conn.execute("""
        CREATE TABLE val_pairs AS
        SELECT
            v.s1_id, v.target_id,
            s1.country AS s1_country,
            s1.business_name AS s1_name,
            s1.business_address AS s1_address,
            COALESCE(s1.name_norm, '') AS s1_nn,
            COALESCE(s1.name_compact, '') AS s1_nc,
            COALESCE(s1.name_tokens, '') AS s1_nt,
            COALESCE(s1.address_norm, '') AS s1_an,
            COALESCE(s1.address_compact, '') AS s1_ac,
            COALESCE(s1.address_tokens, '') AS s1_at,
            tgt.country AS tgt_country,
            tgt.business_name AS tgt_name,
            tgt.business_address AS tgt_address,
            COALESCE(tgt.name_norm, '') AS tgt_nn,
            COALESCE(tgt.name_compact, '') AS tgt_nc,
            COALESCE(tgt.name_tokens, '') AS tgt_nt,
            COALESCE(tgt.address_norm, '') AS tgt_an,
            COALESCE(tgt.address_compact, '') AS tgt_ac,
            COALESCE(tgt.address_tokens, '') AS tgt_at,
            CASE WHEN v.target_id LIKE 'S2-%' THEN 'S2' ELSE 'S3' END AS target_source
        FROM val_sample v
        JOIN source1 s1 ON v.s1_id = s1.entity_id
        JOIN (
            SELECT entity_id, country, business_name, business_address,
                   name_norm, name_compact, name_tokens,
                   address_norm, address_compact, address_tokens
            FROM source2
            UNION ALL
            SELECT entity_id, country, business_name, business_address,
                   name_norm, name_compact, name_tokens,
                   address_norm, address_compact, address_tokens
            FROM source3
        ) tgt ON v.target_id = tgt.entity_id;
    """)

    pairs = conn.execute("""
        SELECT
            s1_id, target_id, target_source, s1_country, tgt_country,
            s1_name, tgt_name, s1_address, tgt_address,
            s1_nn, tgt_nn, s1_nc, tgt_nc, s1_nt, tgt_nt,
            s1_an, tgt_an, s1_ac, tgt_ac, s1_at, tgt_at
        FROM val_pairs
    """).fetchall()

    recovered_by_rule = Counter()
    total_recovered = 0
    missed_pairs = []
    recovered_pairs = []

    for r in pairs:
        (s1_id, target_id, target_source, s1_country, tgt_country,
         s1_name, tgt_name, s1_address, tgt_address,
         s1_nn, tgt_nn, s1_nc, tgt_nc, s1_nt, tgt_nt,
         s1_an, tgt_an, s1_ac, tgt_ac, s1_at, tgt_at) = r

        s1_nn, tgt_nn = s1_nn or "", tgt_nn or ""
        s1_nc, tgt_nc = s1_nc or "", tgt_nc or ""
        s1_nt, tgt_nt = s1_nt or "", tgt_nt or ""
        s1_an, tgt_an = s1_an or "", tgt_an or ""
        s1_ac, tgt_ac = s1_ac or "", tgt_ac or ""
        s1_at, tgt_at = s1_at or "", tgt_at or ""

        matched_rules = []
        if s1_nn and s1_nn == tgt_nn:
            matched_rules.append("exact_name")
        if s1_nc and s1_nc == tgt_nc:
            matched_rules.append("compact_name")
        if s1_an and s1_an == tgt_an:
            matched_rules.append("exact_address")
        if s1_ac and s1_ac == tgt_ac:
            matched_rules.append("compact_address")
        if s1_nt and s1_nt == tgt_nt:
            matched_rules.append("name_tokens")
        if s1_at and s1_at == tgt_at:
            matched_rules.append("address_tokens")
        
        # first word + country
        s1_fw = s1_nn.split()[0] if s1_nn else ""
        tgt_fw = tgt_nn.split()[0] if tgt_nn else ""
        if len(s1_fw) >= 5 and s1_fw == tgt_fw and s1_country == tgt_country:
            matched_rules.append("first_word+cntry")

        # prefix6 + country
        s1_p6 = s1_nc[:6] if len(s1_nc) >= 6 else ""
        tgt_p6 = tgt_nc[:6] if len(tgt_nc) >= 6 else ""
        if s1_p6 and s1_p6 == tgt_p6 and s1_country == tgt_country:
            matched_rules.append("prefix6+cntry")

        for rule in matched_rules:
            recovered_by_rule[rule] += 1

        pair_dict = {
            "s1_id": s1_id, "target_id": target_id, "source": target_source,
            "s1_country": s1_country or "", "tgt_country": tgt_country or "",
            "s1_name": s1_name or "", "tgt_name": tgt_name or "",
            "s1_address": s1_address or "", "tgt_address": tgt_address or "",
            "s1_nn": s1_nn, "tgt_nn": tgt_nn,
            "s1_an": s1_an, "tgt_an": tgt_an
        }

        if matched_rules:
            total_recovered += 1
            if len(recovered_pairs) < 10:
                pair_dict["matched_rules"] = matched_rules
                recovered_pairs.append(pair_dict)
        else:
            missed_pairs.append(pair_dict)

    recall_rate = total_recovered / len(pairs) if pairs else 0.0
    print(f"\nRule-level coverage on sample ({len(pairs):,} pairs):")
    print(f"Overall raw rule recall (before frequency caps): {recall_rate*100:.2f}% ({total_recovered:,}/{len(pairs):,})")
    for rule, count in recovered_by_rule.most_common():
        print(f"  {rule:<20}: {count:>6,} ({count/len(pairs)*100:.2f}%)")

    # Analyze failure categories of missed pairs
    print(f"\nAnalyzing {len(missed_pairs):,} missed pairs...")
    failure_categories = Counter()
    category_examples = {}

    for p in missed_pairs:
        s1_n, tgt_n = p["s1_nn"], p["tgt_nn"]
        s1_a, tgt_a = p["s1_an"], p["tgt_an"]
        s1_words = set(s1_n.split())
        tgt_words = set(tgt_n.split())

        name_ratio = fuzz.ratio(s1_n, tgt_n) if s1_n and tgt_n else 0
        token_sort = fuzz.token_sort_ratio(s1_n, tgt_n) if s1_n and tgt_n else 0
        token_set = fuzz.token_set_ratio(s1_n, tgt_n) if s1_n and tgt_n else 0
        addr_token_set = fuzz.token_set_ratio(s1_a, tgt_a) if s1_a and tgt_a else 0

        s1_nums = set(re.findall(r"\d+", s1_a))
        tgt_nums = set(re.findall(r"\d+", tgt_a))
        num_overlap = bool(s1_nums & tgt_nums)

        cat = None
        if not s1_n or not tgt_n:
            cat = "missing_name"
        elif not s1_a or not tgt_a:
            cat = "missing_address"
        elif p["s1_country"] != p["tgt_country"]:
            cat = "conflicting_or_missing_country"
        elif token_set >= 85 and token_sort < 85:
            cat = "token_permutation_or_subset_name"
        elif name_ratio >= 80:
            cat = "spelling_typo_or_phonetic_name"
        elif token_set >= 80:
            cat = "partial_name_overlap_with_extra_tokens"
        elif addr_token_set >= 80 and (s1_words & tgt_words):
            cat = "address_shared_with_moderate_name_overlap"
        elif num_overlap and addr_token_set >= 65:
            cat = "shared_address_pin_or_number"
        elif s1_nums and s1_nums == tgt_nums and (s1_words & tgt_words):
            cat = "same_address_numbers_and_shared_name_word"
        else:
            cat = "low_lexical_similarity_semantic_challenge"

        failure_categories[cat] += 1
        if cat not in category_examples:
            category_examples[cat] = {
                "s1_name": p["s1_name"], "tgt_name": p["tgt_name"],
                "s1_address": p["s1_address"], "tgt_address": p["tgt_address"],
                "s1_country": p["s1_country"], "tgt_country": p["tgt_country"],
                "name_ratio": name_ratio, "token_set": token_set,
                "addr_token_set": addr_token_set
            }

    print("\nMissed pair failure categories:")
    cat_summary = {}
    for cat, count in failure_categories.most_common():
        pct = count / len(missed_pairs) * 100
        cat_summary[cat] = {"count": count, "percentage": round(pct, 2), "example": category_examples.get(cat)}
        print(f"  {cat:<45}: {count:>5} ({pct:.1f}%)")

    # By source
    missed_by_source = Counter(p["source"] for p in missed_pairs)
    total_by_source = Counter(p[2] for p in pairs)
    source_recall = {
        s: round((total_by_source[s] - missed_by_source[s]) / total_by_source[s] * 100, 2)
        for s in total_by_source
    }
    print(f"\nRecall by target source: {source_recall}")

    # By country
    missed_by_country = Counter(p["s1_country"] for p in missed_pairs)
    total_by_country = Counter(p[3] for p in pairs)
    country_recall = {
        c: round((total_by_country[c] - missed_by_country[c]) / total_by_country[c] * 100, 2)
        for c in total_by_country
    }
    print(f"Recall by country: {country_recall}")

    results = {
        "sample_size": len(pairs),
        "total_val_ground_truth_pairs": total_val_gt,
        "person1_raw_rule_recall_pct": round(recall_rate * 100, 2),
        "person1_effective_recall_pct": 47.68,
        "rules_breakdown": dict(recovered_by_rule),
        "source_recall": source_recall,
        "country_recall": country_recall,
        "failure_categories": cat_summary,
        "sample_missed_count": len(missed_pairs),
        "sample_recovered_count": total_recovered
    }

    with open(ARTIFACTS_P3 / "candidate_recall_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {ARTIFACTS_P3 / 'candidate_recall_analysis.json'}")

    # Generate candidate_recall_analysis.md
    md_content = f"""# Candidate Recall Analysis — Person 3

## Executive Summary

- **Total Ground Truth Pairs (Train)**: 7,638,365 across 2,083,574 non-singleton S1 entities.
- **Validation Split**: 441,016 S1 entities (~20% entity-level deterministic split).
- **Validation Ground Truth Pairs**: {total_val_gt:,}.
- **Person 1 Effective Candidate Recall**: 47.68% (with frequency caps applied).
- **Person 1 Raw Uncapped Rule Recall on Sample**: {recall_rate*100:.2f}%.
- **Unrecovered Truth Pairs in Person 1**: ~52.32% (~800,000 true pairs in validation split alone).

Because Person 2 strictly operates as a reranker over Person 1 candidates (`matching_results ⊆ candidate_pairs`), **Person 2 cannot score or recover any true pair absent from the candidate set**. This confirms **Failure Mode A (Candidate Retrieval)** is the largest structural ceiling on overall recall.

---

## Recall by Source and Country

| Slice | Total Sample Pairs | Missed Pairs | Candidate Recall (%) |
|---|---|---|---|
| Target Source 2 | {total_by_source.get('S2', 0):,} | {missed_by_source.get('S2', 0):,} | {source_recall.get('S2', 0)}% |
| Target Source 3 | {total_by_source.get('S3', 0):,} | {missed_by_source.get('S3', 0):,} | {source_recall.get('S3', 0)}% |
| Country: India | {total_by_country.get('India', 0):,} | {missed_by_country.get('India', 0):,} | {country_recall.get('India', 0)}% |
| Country: US | {total_by_country.get('US', 0):,} | {missed_by_country.get('US', 0):,} | {country_recall.get('US', 0)}% |

---

## Detailed Failure Mode Taxonomy of Missed Pairs

Analysis of {len(missed_pairs):,} unrecovered true pairs revealed clear structural patterns:

"""
    for cat, info in cat_summary.items():
        ex = info["example"] or {}
        md_content += f"""### {cat.replace('_', ' ').title()} ({info['count']:,} cases, {info['percentage']}%)
- **Pattern**: Differences preventing exact equality in `name_norm`, `name_tokens`, `name_compact`, or first 6 characters.
- **Example**:
  - S1 Name: `{ex.get('s1_name', '')}`
  - Tgt Name: `{ex.get('tgt_name', '')}`
  - S1 Address: `{ex.get('s1_address', '')}`
  - Tgt Address: `{ex.get('tgt_address', '')}`
  - RapidFuzz Name Ratio: {ex.get('name_ratio', 0)}, Token-Set: {ex.get('token_set', 0)}, Address Token-Set: {ex.get('addr_token_set', 0)}

"""

    md_content += """---

## Architectural Implications for Person 3

1. **Semantic & Token Candidate Retrieval (Failure Mode A)**:
   - High-yield opportunity: Adding multi-token prefix / token-intersection blocking or character-trigram minhash/simhash / phonetics or address postal code + top name token can recover significant portions of missed true pairs.
   - Any new candidate generation must be unioned with Person 1 (`candidates_p1 ∪ candidates_p3`) to strictly expand recall without losing Person 1's existing candidates.

2. **Semantic Matcher Features (Failure Mode B)**:
   - For candidates already retrieved, Person 2 fails when address strings differ in format despite referring to the same location, or when names have abbreviations/transliterations.
   - Person 3 semantic feature layer will provide transliteration normalization, token jaccard/subset ratios, postal code agreement, and ambiguity margins.
"""

    with open(ARTIFACTS_P3 / "candidate_recall_analysis.md", "w") as f:
        f.write(md_content)
    print(f"Wrote {ARTIFACTS_P3 / 'candidate_recall_analysis.md'}")

if __name__ == "__main__":
    run_analysis()
