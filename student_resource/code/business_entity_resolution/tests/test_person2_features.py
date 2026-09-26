import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from person2_features import CORE_FEATURE_NAMES, pair_features


def record(name, address, country="France"):
    return {
        "business_name": name, "business_address": address, "country": country,
        "name_norm": name.casefold(), "name_tokens": name.casefold(),
        "address_norm": address.casefold(), "address_tokens": address.casefold(),
    }


def test_feature_registry_is_stable_and_numeric():
    f = pair_features(record("Acme Ltd", "12 Main Road"), record("Acme Limited", "12 Main Rd"))
    assert list(f) == CORE_FEATURE_NAMES
    assert all(isinstance(v, float) for v in f.values())
    assert f["country_exact"] == 1.0
    assert f["name_address_product"] > 0


def test_missing_values_do_not_raise():
    f = pair_features(record("", ""), record("Other", "Somewhere", "India"))
    assert f["name_missing"] == 1.0
    assert f["address_missing"] == 1.0
    assert 0.0 <= f["address_length_ratio"] <= 1.0
