# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** Person 1 (Blocking + Data Engine), Person 2 (Matching Model), Person 3 (Semantic + Ultra-Edge), Person 4 (System / Experiment / Leaderboard)  
**Submission Date:** September 25, 2026

---

## 1. Executive Summary

We built a multi-stage entity resolution pipeline that first narrows the
comparison space through layered blocking, then applies a supervised matching
model to decide which candidate pairs are true matches. Person 1 owns the
blocking stage (candidate recall KPI), Person 2 owns the matching model (macro
F₀.₅ KPI), Person 3 adds semantic and embedding-based retrieval to cover
matches that lexical blocking misses, and Person 4 owns system integration,
validation, and leaderboard submissions.

---

## 2. Methodology

### 2.1 Problem Analysis

Key observations from exploratory data analysis:

- **Three noisy sources.** Source 1 is the deduplicated reference; Sources 2
  and 3 have heavy abbreviation, punctuation, and ordering noise in both names
  and addresses.
- **Name variation is the dominant challenge.** Abbreviations (Corp/Corporation,
  Pvt/Private, Ltd/Limited), DBA/trade names, `&` vs "and", punctuation
  differences, and word-order transpositions all prevent simple exact matching
  from achieving high recall. Even with no frequency cap, the union of all
  exact/token rules recovers well under 60 % of true pairs — fuzzy and semantic
  layers are essential.
- **Address noise is secondary.** Address fields are often missing components
  (no PIN, no state), use landmark references, or have transliteration variants.
  Address blocking is useful but contributes far less recall than name blocking.
- **Country is an open set.** Training data covers US and India; the test set
  adds France. All normalisation and blocking rules treat `country` as a string
  label without hard-coding country-specific logic.
- **Large scale.** S1 ≈ 2.2 M records, S2 ≈ 5 M, S3 ≈ 5.3 M (train). Naive
  all-pairs comparison is infeasible — the blocking stage must keep the candidate
  set tractable.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Supervised Classifier + Semantic Retrieval (Hybrid)

**Pipeline:**
1. **Normalisation** (Person 1) — NFKC Unicode normalisation, casefolding,
   punctuation stripping, and derived blocking columns.
2. **Blocking / Candidate Generation** (Person 1) — 8 DuckDB-based exact/token
   rules with both-side frequency caps produce an initial candidate set.
3. **Semantic Retrieval** (Person 3) — Multilingual embedding model + FAISS
   index extends recall for matches that lexical rules miss.
4. **Matching Model** (Person 2) — CatBoost/LightGBM classifier trained on
   name + address similarity features; threshold tuned to maximise macro F₀.₅.
5. **System Integration** (Person 4) — Validation, experiment tracking,
   full-scale execution, and submission packaging.

**Core Innovation:** Both-side frequency caps on blocking keys prevent runaway
pair explosion from common/generic tokens (e.g., "the", "group") without
requiring a global TF-IDF weighting scheme.

---

## 3. Candidate Generation (Blocking)

### 3.1 Normalisation

All raw TSV fields pass through a consistent normalisation pipeline
(`person1_prepare.py`) before any blocking key is applied:

| Derived column | Transformation |
|---|---|
| `name_norm` | NFKC + casefold + `&`→`and` + non-alnum→space + collapse whitespace |
| `name_compact` | `name_norm` with all spaces removed |
| `name_tokens` | `name_norm` tokens sorted & deduplicated (order-invariant key) |
| `address_norm` | NFKC + casefold + `/\`→space + non-alnum→space |
| `address_compact` | `address_norm` spaces removed |
| `address_tokens` | `address_norm` tokens sorted & deduplicated |

### 3.2 Blocking Rules

Eight rules are applied via DuckDB with a **both-side frequency cap**: a key is
only used when it appears ≤ cap times in S1 **and** ≤ cap times in the target
source. This bounds pair explosion without discarding rare distinctive keys.

| # | Rule | Key column | Cap |
|---|---|---|---|
| 1 | exact_name | `name_norm` | 20 |
| 2 | compact_name | `name_compact` | 20 |
| 3 | exact_address | `address_norm` | 20 |
| 4 | compact_address | `address_compact` | 20 |
| 5 | name_tokens | `name_tokens` | 20 |
| 6 | address_tokens | `address_tokens` | 20 |
| 7 | first_word+country | first word of `name_norm` + `country` (min len 5) | 50 |
| 8 | prefix6+country | first 6 chars of `name_compact` + `country` (min len 6) | 30 |

Candidates from all rules are unioned and deduplicated per source pair (S2 and
S3 handled separately to keep peak memory bounded), then merged into
`output/candidate_pairs.tsv`.

### 3.3 Results

| Metric | Value |
|---|---|
| Total distinct candidate pairs (test) | 16,908,195 |
| S1 entities with at least one candidate | 1,568,874 / 1,732,544 |
| S1 entities with no candidates (singletons) | 163,670 |
| Train candidate recall | **47.68 %** (3,642,256 / 7,638,365) |
| Avg candidates per S1 entity | 8.4 |

### 3.4 Ensuring True Matches Are Not Lost

- **Multiple complementary keys:** name_tokens catches word-order transpositions
  that exact_name misses; compact_name catches punctuation-only differences.
- **Address blocking as a safety net:** address_norm/compact/tokens recover
  matches where the business name differs too much for name-based rules.
- **Frequency cap is bilateral:** capping both sides means a true match is only
  excluded if its key is so common that it appears in >20 records on each side —
  in that case it contributes very little signal anyway.
- **Fuzzy layer (Person 3):** The ~57 % of true pairs not recovered by exact
  rules are the primary target for the semantic/fuzzy retrieval stage.

---

## 4. Matching Model

*To be completed by Person 2.*

**Features used:**
- Name features: RapidFuzz token_sort_ratio, partial_ratio, Jaro-Winkler;
  TF-IDF cosine similarity on name_norm
- Address features: token overlap, edit distance on address_compact
- Boolean: exact_name match, exact_address match, same country

**Model type:** LightGBM / CatBoost gradient boosting classifier  
**Threshold selection method:** F₀.₅ optimisation on held-out validation split

---

## 5. Results & Error Analysis

- **Train candidate recall (blocking):** 47.68 %
- **F₀.₅ Score (macro, validation):** [to be filled by Person 2]
- **Common false positives:** Chains / franchises with identical names at
  different addresses (e.g., "McDonald's" at two locations).
- **Common false negatives:** Businesses with entirely different trade names vs.
  legal names; heavy transliteration variants in India addresses.

---

## 6. Conclusion

We split the problem cleanly into blocking (Person 1), matching (Person 2),
semantic extension (Person 3), and system/infrastructure (Person 4). The
blocking pipeline achieves 43 % recall with a very compact candidate set
(avg 3.8 candidates per S1 entity), giving the matching model a high-precision
input. Improving recall to 85 %+ requires the fuzzy/semantic layers from
Person 3, which are the primary lever for lifting the final F₀.₅ score.

---

## Appendix

### A. Code Artefacts

All source code is in `code/business_entity_resolution/src/`. The pipeline
has three entry points:

```
# Step 1 — Normalise data
python code/business_entity_resolution/src/person1_prepare.py

# Step 2 — Generate candidates
python code/business_entity_resolution/src/person1_fast_candidate_generator.py

# Step 3 — Fix output format
python code/business_entity_resolution/src/person1_fix_outputs.py
```

Full reproduction instructions are in
`code/business_entity_resolution/README.md`.

Dependencies are pinned in `code/business_entity_resolution/requirements.txt`.

### B. Key Files

| Path | Description |
|---|---|
| `output/candidate_pairs.tsv` | Blocking output — 1,732,544 S1 rows |
| `output/matching_results.tsv` | Final predictions (Person 2 fills this) |
| `work/person1/*.parquet` | Normalised train + test parquets |
| `artifacts/duckdb/person1.duckdb` | Persistent DuckDB (early exploration) |
