# Business Entity Resolution — Person 1: Blocking + Data Engine

## Overview

Person 1 owns the **blocking / candidate generation stage**. The output is
`output/candidate_pairs.tsv` — the set of (S1, S2/S3) entity pairs fed into the
matching model (Person 2).

**KPI:** Candidate recall on the training set.  
**Achieved:** 47.68 % (3,642,256 / 7,638,365 true pairs recovered)  
**Avg candidates per S1 entity:** 8.4

---

## Setup After Cloning

```bash
git clone https://github.com/Vedha-Sree/amazon-ml-challenge.git
cd amazon-ml-challenge
git checkout person1
```

**The dataset is NOT in this repo** (files are too large for GitHub).
Get the `student_resource/dataset/` folder from Person 1 and place it here:

```
amazon-ml-challenge/
└── student_resource/
    └── dataset/          ← place the dataset folder here
        ├── train/
│       │   ├── train_source1.tsv
│       │   ├── train_source2.tsv
│       │   ├── train_source3.tsv
│       │   └── train_ground_truth.tsv
        └── test/
            ├── test_source1.tsv
            ├── test_source2.tsv
            └── test_source3.tsv
```

Then install dependencies:

```bash
pip install -r student_resource/code/business_entity_resolution/requirements.txt
```

---

## Reproduce End-to-End

Run the three steps below in order. Each step is idempotent — re-running is safe.

### Step 1 — Normalise raw data

Creates `work/person1/train_source{1,2,3}.parquet` and
`work/person1/test_source{1,2,3}.parquet` with derived blocking columns:
`name_norm`, `name_compact`, `name_tokens`, `address_norm`, `address_compact`,
`address_tokens`.

```bash
python code/business_entity_resolution/src/person1_prepare.py
```

Expected output: 6 parquet files written to `work/person1/`.  
Skips any file that already exists and is non-empty.

---

### Step 2 — Generate candidates (blocking)

Runs 8 blocking rules via DuckDB (both-side frequency cap = 20), evaluates
recall on the training set, then writes `output/candidate_pairs.tsv` for the
test set.

```bash
python code/business_entity_resolution/src/person1_fast_candidate_generator.py
```

Expected output (train):
```
TRAIN GROUND TRUTH CANDIDATE RECALL: 3,642,256 / 7,638,365 (47.68%)
Average Candidates per S1 Entity: 8.4
```

Expected output (test):
```
output/candidate_pairs.tsv   — 1,732,544 rows (one per S1 test entity)
```

---

### Step 3 — Fix output format and generate matching template

Cleans up DuckDB's quoted-empty-string artefact in `candidate_pairs.tsv` and
writes a blank `output/matching_results.tsv` (one row per S1 test entity, all
`matched_entity_ids` empty — ready for Person 2 to fill in).

```bash
python code/business_entity_resolution/src/person1_fix_outputs.py
```

---

### Step 4 — Validate outputs

```bash
python utils/validate_submission.py \
    --matching  output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir  dataset/test
```

Expected result: `PASS — no blocking issues found.`

---

## Source Files

| File | Purpose |
|---|---|
| `src/person1_prepare.py` | Raw TSV → normalised Parquet (6 columns per entity) |
| `src/person1_fast_candidate_generator.py` | DuckDB blocking pipeline — 8 rules, both-side frequency cap |
| `src/person1_fix_outputs.py` | Fixes quoted-empty artefact; writes blank matching_results.tsv |
| `src/person1_high_recall_generator.py` | Experimental: exact + RapidFuzz prefix-bucket fuzzy layer |
| `src/person1_candidate_generator.py` | Alternative Polars + RapidFuzz generator (char sig blocking) |
| `src/person1_blocking_benchmark.py` | Per-rule recall benchmark on train set |
| `src/person1_profile.py` | Quick schema / row-count profiler |
| `src/person1_streaming_candidates.py` | Disk-backed SQLite version (memory-safe alternative) |
| `src/person1_high_recall_test.py` | Incremental recall measurement per rule |
| `src/person1_next_blocks.py` | Experiments: token blocks, country-conditioned blocks |
| `src/person1_charblock_test.py` | Experiment: TF-IDF char n-gram + sklearn NearestNeighbors |
| `src/fast_validate_candidates.py` | Streaming format validator for candidate_pairs.tsv |
| `src/create_matching_template.py` | One-shot utility to create blank matching_results.tsv |

---

## Blocking Rules (person1_fast_candidate_generator.py)

| # | Rule | Key column | Both-side cap |
|---|---|---|---|
| 1 | exact_name | `name_norm` | 50 |
| 2 | compact_name | `name_compact` | 50 |
| 3 | exact_address | `address_norm` | 50 |
| 4 | compact_address | `address_compact` | 50 |
| 5 | name_tokens | `name_tokens` (sorted dedup token set) | 50 |
| 6 | address_tokens | `address_tokens` | 50 |
| 7 | first_word+country | first word of `name_norm` + `country` (min len 5) | 20 |
| 8 | prefix6+country | first 6 chars of `name_compact` + `country` (min len 6) | 15 |

**Both-side cap:** a blocking key is only used when it appears ≤ cap times in
both S1 and the target source. This prevents common/generic names from
generating explosive pair counts.

---

## Normalisation Details (person1_prepare.py)

| Column | Transformation |
|---|---|
| `name_norm` | NFKC + casefold + `&`→`and` + non-alnum→space |
| `name_compact` | `name_norm` with all spaces removed |
| `name_tokens` | `name_norm` tokens, deduplicated & sorted (order-invariant) |
| `address_norm` | NFKC + casefold + `/\`→space + non-alnum→space |
| `address_compact` | `address_norm` with spaces removed |
| `address_tokens` | `address_norm` tokens, deduplicated & sorted |

---

## Outputs Consumed by Person 2 (Matching Model)

| File | Description |
|---|---|
| `output/candidate_pairs.tsv` | One row per S1 test entity; `candidate_entity_ids` = comma-separated S2/S3 IDs |
| `work/person1/test_source{1,2,3}.parquet` | Normalised test parquets with all 6 feature columns |
| `work/person1/train_source{1,2,3}.parquet` | Normalised train parquets for feature engineering |

---

## For Person 2 (Matching Model)

After cloning and placing the dataset, run Steps 1–4 above to regenerate:
- `work/person1/*.parquet` — normalised features for your model
- `output/candidate_pairs.tsv` — the candidate set to run inference over
- `output/matching_results.tsv` — blank template for your predictions

**Recall context:** the blocking stage captures 47.68 % of true matches.
Your matching model works on those candidates. Person 3's semantic layer
will add more candidates on top — coordinate with them before final submission.
