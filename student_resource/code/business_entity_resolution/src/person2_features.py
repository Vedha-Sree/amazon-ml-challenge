"""Reusable Person-2 pair feature registry.

The feature names are stable and the optional TF-IDF bundle is fitted once on
training records. Person 3 can add numeric columns through ``extra_features``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from rapidfuzz import fuzz

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover
    TfidfVectorizer = None


CORE_FEATURE_NAMES = [
    "name_raw_exact", "name_normalized_exact", "name_ratio", "name_wratio", "name_token_set", "name_token_sort", "name_partial", "name_char_cosine", "name_tfidf_char", "name_tfidf_word", "name_token_jaccard", "name_common_tokens", "name_length_diff", "name_length_ratio", "name_token_count_diff", "name_missing",
    "address_raw_exact", "address_normalized_exact", "address_ratio", "address_wratio", "address_token_set", "address_token_sort", "address_partial", "address_char_cosine", "address_tfidf_char", "address_tfidf_word", "address_token_jaccard", "address_common_tokens", "address_length_diff", "address_length_ratio", "address_token_count_diff", "address_number_jaccard", "address_postal_exact", "address_missing", "country_exact", "country_missing",
    "name_address_product", "name_address_min", "name_address_mean", "name_address_gap", "strong_name_weak_address", "weak_name_strong_address",
]


def _safe(value) -> str:
    return "" if value is None else str(value)


def _tokens(value: str) -> list[str]:
    return _safe(value).split()


def _numbers(value: str) -> set[str]:
    return set(re.findall(r"\d+", _safe(value)))


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    aa, bb = set(a), set(b)
    return len(aa & bb) / len(aa | bb) if aa or bb else 0.0


def _char_cosine(a: str, b: str, n: int = 3) -> float:
    def grams(s: str) -> set[str]:
        s = "".join(s.split())
        return {s[i:i+n] for i in range(max(0, len(s) - n + 1))}
    x, y = grams(a), grams(b)
    return len(x & y) / math.sqrt(len(x) * len(y)) if x and y else 0.0


def _length_ratio(a: str, b: str) -> float:
    x, y = len(a), len(b)
    return min(x, y) / max(x, y) if max(x, y) else 1.0


@dataclass
class TfidfFeatures:
    name_char: object | None = None
    name_word: object | None = None
    address_char: object | None = None
    address_word: object | None = None

    @classmethod
    def fit(cls, records: Sequence[Mapping[str, str]], max_features: int = 150_000):
        if TfidfVectorizer is None:
            raise RuntimeError("scikit-learn is required for TF-IDF features")
        names = [_safe(r.get("name_norm")) for r in records]
        addresses = [_safe(r.get("address_norm")) for r in records]
        def make(values, analyzer, ngram):
            v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, max_features=max_features, dtype="float32")
            v.fit(values or [""])
            return v
        return cls(make(names, "char", (2, 5)), make(names, "word", (1, 2)), make(addresses, "char", (3, 6)), make(addresses, "word", (1, 2)))

    @staticmethod
    def _cosine(vectorizer, a: str, b: str) -> float:
        if vectorizer is None or not a or not b:
            return 0.0
        x, y = vectorizer.transform([a, b])
        return float(x.multiply(y).sum())

    def pair(self, left: Mapping[str, str], right: Mapping[str, str]) -> dict[str, float]:
        ln, rn = _safe(left.get("name_norm")), _safe(right.get("name_norm"))
        la, ra = _safe(left.get("address_norm")), _safe(right.get("address_norm"))
        return {
            "name_tfidf_char": self._cosine(self.name_char, ln, rn), "name_tfidf_word": self._cosine(self.name_word, ln, rn),
            "address_tfidf_char": self._cosine(self.address_char, la, ra), "address_tfidf_word": self._cosine(self.address_word, la, ra),
        }

    def batch(self, lefts: Sequence[Mapping[str, str]], rights: Sequence[Mapping[str, str]]) -> list[dict[str, float]]:
        """Compute the four TF-IDF similarities with one transform per field."""
        n = len(lefts)
        if n != len(rights):
            raise ValueError("lefts and rights must have the same length")
        out = [{"name_tfidf_char": 0.0, "name_tfidf_word": 0.0,
                "address_tfidf_char": 0.0, "address_tfidf_word": 0.0} for _ in range(n)]
        for attr, key, field in (
            ("name_char", "name_tfidf_char", "name_norm"),
            ("name_word", "name_tfidf_word", "name_norm"),
            ("address_char", "address_tfidf_char", "address_norm"),
            ("address_word", "address_tfidf_word", "address_norm"),
        ):
            vectorizer = getattr(self, attr)
            if vectorizer is None:
                continue
            values = [_safe(x.get(field)) for x in lefts] + [_safe(x.get(field)) for x in rights]
            matrix = vectorizer.transform(values)
            left_matrix, right_matrix = matrix[:n], matrix[n:]
            similarities = left_matrix.multiply(right_matrix).sum(axis=1)
            for i, value in enumerate(similarities):
                out[i][key] = float(value)
        return out


def pair_features(left: Mapping[str, str], right: Mapping[str, str], tfidf: TfidfFeatures | None = None, extra_features: Mapping[str, float] | None = None) -> dict[str, float]:
    ln, rn = _safe(left.get("name_norm")), _safe(right.get("name_norm"))
    la, ra = _safe(left.get("address_norm")), _safe(right.get("address_norm"))
    raw_ln, raw_rn = _safe(left.get("business_name")), _safe(right.get("business_name"))
    raw_la, raw_ra = _safe(left.get("business_address")), _safe(right.get("business_address"))
    nt1, nt2 = _tokens(left.get("name_tokens") or ln), _tokens(right.get("name_tokens") or rn)
    at1, at2 = _tokens(left.get("address_tokens") or la), _tokens(right.get("address_tokens") or ra)
    ns, ads = fuzz.ratio(ln, rn) / 100.0, fuzz.ratio(la, ra) / 100.0
    f = {
        "name_raw_exact": float(raw_ln == raw_rn and bool(raw_ln)), "name_normalized_exact": float(ln == rn and bool(ln)), "name_ratio": ns, "name_wratio": fuzz.WRatio(ln, rn) / 100.0, "name_token_set": fuzz.token_set_ratio(ln, rn) / 100.0, "name_token_sort": fuzz.token_sort_ratio(ln, rn) / 100.0, "name_partial": fuzz.partial_ratio(ln, rn) / 100.0, "name_char_cosine": _char_cosine(ln, rn), "name_tfidf_char": 0.0, "name_tfidf_word": 0.0, "name_token_jaccard": _jaccard(nt1, nt2), "name_common_tokens": float(len(set(nt1) & set(nt2))), "name_length_diff": float(abs(len(ln) - len(rn))), "name_length_ratio": _length_ratio(ln, rn), "name_token_count_diff": float(abs(len(nt1) - len(nt2))), "name_missing": float(not ln),
        "address_raw_exact": float(raw_la == raw_ra and bool(raw_la)), "address_normalized_exact": float(la == ra and bool(la)), "address_ratio": ads, "address_wratio": fuzz.WRatio(la, ra) / 100.0, "address_token_set": fuzz.token_set_ratio(la, ra) / 100.0, "address_token_sort": fuzz.token_sort_ratio(la, ra) / 100.0, "address_partial": fuzz.partial_ratio(la, ra) / 100.0, "address_char_cosine": _char_cosine(la, ra), "address_tfidf_char": 0.0, "address_tfidf_word": 0.0, "address_token_jaccard": _jaccard(at1, at2), "address_common_tokens": float(len(set(at1) & set(at2))), "address_length_diff": float(abs(len(la) - len(ra))), "address_length_ratio": _length_ratio(la, ra), "address_token_count_diff": float(abs(len(at1) - len(at2))), "address_number_jaccard": _jaccard(sorted(_numbers(la)), sorted(_numbers(ra))), "address_postal_exact": float(bool(_numbers(la) & _numbers(ra))), "address_missing": float(not la),
        "country_exact": float(_safe(left.get("country")) == _safe(right.get("country"))), "country_missing": float(not _safe(left.get("country")) or not _safe(right.get("country"))), "name_address_product": ns * ads, "name_address_min": min(ns, ads), "name_address_mean": (ns + ads) / 2.0, "name_address_gap": abs(ns - ads), "strong_name_weak_address": float(ns >= .90 and ads < .50), "weak_name_strong_address": float(ns < .50 and ads >= .90),
    }
    if tfidf is not None:
        f.update(tfidf.pair(left, right))
    if extra_features:
        f.update({str(k): float(v) for k, v in extra_features.items()})
    return f
