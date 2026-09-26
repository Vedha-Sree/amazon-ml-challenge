"""Person 3 Semantic, Multilingual, Transliteration, and Address Features.

Provides additive semantic features designed to resolve:
1. Transliterated / phonetic variations (especially Indic & European scripts).
2. Token subset / permutation and legal entity suffix differences.
3. Component-level address matching (street numbers, building numbers, road abbreviations).
4. Address contradiction detection (same name, conflicting street numbers).
5. Cross-field harmonic interactions.
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Mapping, Sequence

from rapidfuzz import fuzz

SEMANTIC_FEATURE_NAMES = [
    # Transliteration & Phonetic
    "name_translit_exact",
    "name_translit_ratio",
    "name_translit_token_set",
    # Core Name / Legal Suffix Removal
    "name_core_exact",
    "name_core_ratio",
    "name_token_subset",
    "name_trigram_jaccard",
    "name_trigram_containment",
    # Address Component & Number Semantics
    "address_number_exact",
    "address_number_subset",
    "address_number_conflict",
    "address_core_token_jaccard",
    "address_shared_long_tokens",
    # Cross-Field Interaction
    "name_address_harmonic",
    "high_name_high_addr",
    "high_name_conflict_number",
]

# Business stopwords and legal suffixes across US, India, and France
LEGAL_SUFFIXES = {
    "pvt", "ltd", "private", "limited", "corp", "corporation", "inc", "incorporated",
    "llc", "llp", "gmbh", "sarl", "sa", "co", "company", "enterprises", "associates",
    "industries", "holdings", "group", "traders", "agency", "services", "mrs", "mr",
    "shri", "sri", "the", "and", "de", "la", "le", "des"
}

# Address abbreviation mappings
ADDR_ABBREVIATIONS = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "ln": "lane", "dr": "drive", "blvd": "boulevard", "hwy": "highway",
    "opp": "opposite", "nr": "near", "flr": "floor", "bldg": "building",
    "apt": "apartment", "ste": "suite", "pl": "place", "sq": "square",
    "ct": "court", "pk": "park", "pkwy": "parkway", "cir": "circle"
}

def _safe(val) -> str:
    return "" if val is None else str(val)

def strip_accents(text: str) -> str:
    """Normalize accents/diacritics for multilingual matching (French, transliterated Indic, etc.)."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )

def transliterate_norm(text: str) -> str:
    """Phonetic / transliteration normalization for common English/Indic/Multilingual variants."""
    s = strip_accents(text).lower()
    # Normalize common phonetically interchangeable sequences
    s = re.sub(r"ph", "f", s)
    s = re.sub(r"ee", "i", s)
    s = re.sub(r"oo", "u", s)
    s = re.sub(r"ou", "u", s)
    s = re.sub(r"ow", "u", s)
    s = re.sub(r"aa", "a", s)
    s = re.sub(r"w", "v", s)
    s = re.sub(r"ksh", "x", s)
    s = re.sub(r"kh", "k", s)
    s = re.sub(r"gh", "g", s)
    s = re.sub(r"dh", "d", s)
    s = re.sub(r"th", "t", s)
    s = re.sub(r"bh", "b", s)
    s = re.sub(r"sh", "s", s)
    s = re.sub(r"ch", "c", s)
    s = re.sub(r"z", "j", s)
    # Collapse double characters (e.g. 'mittal' -> 'mital', 'sharma' -> 'sarma')
    s = re.sub(r"(.)\1+", r"\1", s)
    # Retain only alphanumeric and space
    s = "".join(c if (c.isalnum() or c.isspace()) else " " for c in s)
    return " ".join(s.split())

def core_name_tokens(text: str) -> list[str]:
    """Extract significant name tokens removing legal suffixes and common noise."""
    tokens = text.lower().split()
    core = [t for t in tokens if t not in LEGAL_SUFFIXES and len(t) >= 2]
    return core if core else tokens

def trigrams(text: str) -> set[str]:
    s = "".join(text.split())
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i+3] for i in range(len(s) - 2)}

def extract_address_numbers(text: str) -> list[str]:
    """Extract normalized numeric tokens from address in order of appearance."""
    return re.findall(r"\b\d+\b", text)

def canonical_address_tokens(text: str) -> list[str]:
    """Standardize road and landmark abbreviations in address."""
    tokens = text.lower().split()
    return [ADDR_ABBREVIATIONS.get(t, t) for t in tokens if len(t) > 1]

def semantic_pair_features(left: Mapping[str, str], right: Mapping[str, str]) -> dict[str, float]:
    """Compute Person 3 semantic features for a pair of entities."""
    ln = _safe(left.get("name_norm"))
    rn = _safe(right.get("name_norm"))
    la = _safe(left.get("address_norm"))
    ra = _safe(right.get("address_norm"))

    # 1. Transliteration / Phonetic
    t_ln = transliterate_norm(ln)
    t_rn = transliterate_norm(rn)
    t_exact = float(t_ln == t_rn and bool(t_ln))
    t_ratio = fuzz.ratio(t_ln, t_rn) / 100.0 if t_ln and t_rn else 0.0
    t_token_set = fuzz.token_set_ratio(t_ln, t_rn) / 100.0 if t_ln and t_rn else 0.0

    # 2. Core Name & Token Subset (Legal Suffixes Removed)
    c_lt = core_name_tokens(ln)
    c_rt = core_name_tokens(rn)
    c_ls = " ".join(c_lt)
    c_rs = " ".join(c_rt)
    core_exact = float(c_ls == c_rs and bool(c_ls))
    core_ratio = fuzz.ratio(c_ls, c_rs) / 100.0 if c_ls and c_rs else 0.0

    set_lt, set_rt = set(c_lt), set(c_rt)
    token_subset = 0.0
    if set_lt and set_rt:
        if set_lt.issubset(set_rt) or set_rt.issubset(set_lt):
            token_subset = 1.0
        else:
            token_subset = len(set_lt & set_rt) / min(len(set_lt), len(set_rt))

    # Character Trigrams
    tri_l = trigrams(ln)
    tri_r = trigrams(rn)
    if tri_l and tri_r:
        tri_jaccard = len(tri_l & tri_r) / len(tri_l | tri_r)
        tri_contain = len(tri_l & tri_r) / min(len(tri_l), len(tri_r))
    else:
        tri_jaccard = 0.0
        tri_contain = 0.0

    # 3. Address Numbers & Components
    nums_l = extract_address_numbers(la)
    nums_r = extract_address_numbers(ra)
    set_num_l = set(nums_l)
    set_num_r = set(nums_r)

    if set_num_l and set_num_r:
        addr_num_exact = float(nums_l == nums_r)
        addr_num_subset = float(set_num_l.issubset(set_num_r) or set_num_r.issubset(set_num_l))
        addr_num_conflict = float(not (set_num_l & set_num_r))
    else:
        addr_num_exact = 0.0
        addr_num_subset = 0.0
        addr_num_conflict = 0.0

    # Canonical address tokens
    addr_tok_l = canonical_address_tokens(la)
    addr_tok_r = canonical_address_tokens(ra)
    set_at_l = set(addr_tok_l)
    set_at_r = set(addr_tok_r)
    addr_core_jaccard = len(set_at_l & set_at_r) / len(set_at_l | set_at_r) if (set_at_l or set_at_r) else 0.0
    shared_long = float(len([t for t in (set_at_l & set_at_r) if len(t) >= 5]))

    # 4. Cross-Field Interactions
    base_name_ratio = fuzz.ratio(ln, rn) / 100.0 if ln and rn else 0.0
    base_addr_ratio = fuzz.ratio(la, ra) / 100.0 if la and ra else 0.0

    if base_name_ratio + base_addr_ratio > 0:
        harmonic = (2.0 * base_name_ratio * base_addr_ratio) / (base_name_ratio + base_addr_ratio)
    else:
        harmonic = 0.0

    high_high = float(t_token_set >= 0.85 and (base_addr_ratio >= 0.65 or addr_core_jaccard >= 0.50))
    # Hard negative flag: identical/similar name but contradictory street numbers
    high_conflict = float(base_name_ratio >= 0.85 and addr_num_conflict == 1.0 and len(set_num_l) > 0 and len(set_num_r) > 0)

    return {
        "name_translit_exact": t_exact,
        "name_translit_ratio": t_ratio,
        "name_translit_token_set": t_token_set,
        "name_core_exact": core_exact,
        "name_core_ratio": core_ratio,
        "name_token_subset": token_subset,
        "name_trigram_jaccard": tri_jaccard,
        "name_trigram_containment": tri_contain,
        "address_number_exact": addr_num_exact,
        "address_number_subset": addr_num_subset,
        "address_number_conflict": addr_num_conflict,
        "address_core_token_jaccard": addr_core_jaccard,
        "address_shared_long_tokens": shared_long,
        "name_address_harmonic": harmonic,
        "high_name_high_addr": high_high,
        "high_name_conflict_number": high_conflict,
    }

if __name__ == "__main__":
    # Smoke test
    rec1 = {"name_norm": "lakshmi stores pvt ltd", "address_norm": "12 mg road bangalore 560001"}
    rec2 = {"name_norm": "laxmi stores private limited", "address_norm": "12 mahatma gandhi rd bengaluru 560001"}
    feats = semantic_pair_features(rec1, rec2)
    print("Semantic features smoke test:")
    for k, v in feats.items():
        print(f"  {k:<30}: {v:.4f}")
