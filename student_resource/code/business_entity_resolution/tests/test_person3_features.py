"""Unit tests for Person 3 semantic, transliteration, and address features."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from person3_semantic_features import (
    SEMANTIC_FEATURE_NAMES,
    semantic_pair_features,
    transliterate_norm,
    strip_accents,
    core_name_tokens,
    extract_address_numbers,
    canonical_address_tokens,
)
from person2_features import CORE_FEATURE_NAMES, pair_features

def test_transliteration_phonetic_normalization():
    # Test Indic transliteration pairs
    assert transliterate_norm("Laxmi") == transliterate_norm("Lakshmi")
    assert transliterate_norm("Choudhury") == transliterate_norm("Chowdhury")
    from rapidfuzz import fuzz
    assert fuzz.ratio(transliterate_norm("Mohamed"), transliterate_norm("Muhammed")) >= 85
    assert transliterate_norm("Bhandup") == transliterate_norm("Bandup")
    assert transliterate_norm("Kiran") == transliterate_norm("Keeran")

def test_unicode_accents_and_multilingual():
    # Test European/French accents and multilingual strings
    s1 = "Société Générale"
    s2 = "Societe Generale"
    assert strip_accents(s1).lower() == s2.lower()
    
    # Feature extraction with diacritics
    feats = semantic_pair_features(
        {"name_norm": "café de la paix", "address_norm": "12 boulevard des capucines paris"},
        {"name_norm": "cafe de la paix", "address_norm": "12 blvd des capucines paris"}
    )
    assert feats["name_translit_ratio"] >= 0.90
    assert feats["address_number_exact"] == 1.0
    assert feats["address_core_token_jaccard"] >= 0.60

def test_core_name_and_legal_suffixes():
    tokens1 = core_name_tokens("Tata Consultancy Services Private Limited")
    tokens2 = core_name_tokens("Tata Consultancy Services Pvt Ltd")
    assert tokens1 == tokens2 == ["tata", "consultancy"]

    feats = semantic_pair_features(
        {"name_norm": "acme technologies pvt ltd", "address_norm": "100 main st"},
        {"name_norm": "acme technologies inc", "address_norm": "100 main street"}
    )
    assert feats["name_core_exact"] == 1.0
    assert feats["name_token_subset"] == 1.0

def test_address_number_semantics():
    # Same numbers
    la = "Flat 204, Tower 3, High Point, Bangalore 560001"
    ra = "204, Tower 3, High Point, Bengaluru 560001"
    assert extract_address_numbers(la) == extract_address_numbers(ra) == ["204", "3", "560001"]

    feats = semantic_pair_features({"address_norm": la}, {"address_norm": ra})
    assert feats["address_number_exact"] == 1.0
    assert feats["address_number_conflict"] == 0.0

    # Conflicting numbers (hard negative signal)
    la_diff = "Flat 204, Tower 3, Bangalore"
    ra_diff = "Flat 508, Tower 9, Bangalore"
    feats_diff = semantic_pair_features(
        {"name_norm": "reliance fresh", "address_norm": la_diff},
        {"name_norm": "reliance fresh", "address_norm": ra_diff}
    )
    assert feats_diff["address_number_conflict"] == 1.0
    assert feats_diff["high_name_conflict_number"] == 1.0

def test_missing_values_and_empty_inputs():
    feats = semantic_pair_features({}, {})
    assert len(feats) == len(SEMANTIC_FEATURE_NAMES)
    assert all(isinstance(v, float) for v in feats.values())
    assert feats["name_translit_exact"] == 0.0
    assert feats["address_number_exact"] == 0.0

def test_combined_feature_registry_stability():
    left = {"name_norm": "apollo pharmacy", "address_norm": "10 mg road", "country": "India"}
    right = {"name_norm": "apollo pharmacy pvt ltd", "address_norm": "10 mahatma gandhi rd", "country": "India"}
    
    base = pair_features(left, right)
    sem = semantic_pair_features(left, right)
    base.update(sem)

    total_cols = list(CORE_FEATURE_NAMES) + SEMANTIC_FEATURE_NAMES
    assert len(total_cols) == 42 + 16 == 58
    for col in total_cols:
        assert col in base
        assert isinstance(base[col], float)
