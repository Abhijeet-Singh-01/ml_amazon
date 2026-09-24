"""
Entity Name Similarity Features.

Calculates multi-metric string and token similarity features across entity names:
- Exact match indicators (raw, clean, base name)
- Edit distance & alignment metrics (Levenshtein, Jaro-Winkler)
- Token-set & token-sort set similarities
- Prefix / Suffix length ratios
- Length disparity and character count differences
"""

import difflib
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Rapidfuzz acceleration with standard library fallback
try:
    from rapidfuzz import fuzz, distance
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


def _safe_str(val: Optional[str]) -> str:
    if val is None or not isinstance(val, str):
        return ""
    return val.strip()


def compute_levenshtein_sim(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return fuzz.ratio(s1, s2) / 100.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def compute_jaro_winkler_sim(s1: str, s2: str) -> float:
    """Compute Jaro-Winkler similarity in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return distance.JaroWinkler.similarity(s1, s2)
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def compute_token_sort_sim(s1: str, s2: str) -> float:
    """Compute token sort similarity in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return fuzz.token_sort_ratio(s1, s2) / 100.0
    t1 = " ".join(sorted(s1.split()))
    t2 = " ".join(sorted(s2.split()))
    return difflib.SequenceMatcher(None, t1, t2).ratio()


def compute_token_set_sim(s1: str, s2: str) -> float:
    """Compute token set similarity in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return fuzz.token_set_ratio(s1, s2) / 100.0
    set1 = set(s1.split())
    set2 = set(s2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 1.0


def compute_lcp_ratio(s1: str, s2: str) -> float:
    """Compute Longest Common Prefix (LCP) ratio relative to max length."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    min_len = min(len(s1), len(s2))
    lcp = 0
    while lcp < min_len and s1[lcp] == s2[lcp]:
        lcp += 1
    return lcp / max(len(s1), len(s2))


def extract_name_features_pair(
    src_name_clean: str,
    tgt_name_clean: str,
    src_name_base: str = "",
    tgt_name_base: str = "",
    src_suffix: str = "",
    tgt_suffix: str = "",
) -> Dict[str, float]:
    """Extract full suite of name similarity features for a single pair."""
    s1 = _safe_str(src_name_clean)
    s2 = _safe_str(tgt_name_clean)
    b1 = _safe_str(src_name_base)
    b2 = _safe_str(tgt_name_base)

    exact_clean = 1.0 if s1 and s2 and s1 == s2 else 0.0
    exact_base = 1.0 if b1 and b2 and b1 == b2 else 0.0

    # Suffix agreement: 1 if both match, 0 if both present but differ, 0.5 if either missing
    if src_suffix and tgt_suffix:
        suffix_match = 1.0 if src_suffix.lower() == tgt_suffix.lower() else 0.0
    else:
        suffix_match = 0.5

    len1, len2 = len(s1), len(s2)
    len_diff = abs(len1 - len2)
    len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0

    tokens1, tokens2 = s1.split(), s2.split()
    tok_diff = abs(len(tokens1) - len(tokens2))

    return {
        "name_exact_clean": exact_clean,
        "name_exact_base": exact_base,
        "name_levenshtein": compute_levenshtein_sim(s1, s2),
        "name_jaro_winkler": compute_jaro_winkler_sim(s1, s2),
        "name_token_sort": compute_token_sort_sim(s1, s2),
        "name_token_set": compute_token_set_sim(s1, s2),
        "name_lcp_ratio": compute_lcp_ratio(s1, s2),
        "name_base_levenshtein": compute_levenshtein_sim(b1, b2),
        "name_len_diff": float(len_diff),
        "name_len_ratio": float(len_ratio),
        "name_tok_diff": float(tok_diff),
        "name_suffix_match": float(suffix_match),
    }


def compute_name_similarity_features(
    pairs_df: pd.DataFrame,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute vectorized name similarity features across all candidate pairs.
    """
    src_map = source_df.set_index("id")[
        ["name_clean", "name_base", "legal_suffix"]
    ].to_dict("index")
    tgt_map = target_df.set_index("id")[
        ["name_clean", "name_base", "legal_suffix"]
    ].to_dict("index")

    rows = []
    for _, row in pairs_df.iterrows():
        s_id = str(row["source_id"])
        t_id = str(row["target_id"])

        s_meta = src_map.get(s_id, {})
        t_meta = tgt_map.get(t_id, {})

        feats = extract_name_features_pair(
            src_name_clean=s_meta.get("name_clean", ""),
            tgt_name_clean=t_meta.get("name_clean", ""),
            src_name_base=s_meta.get("name_base", ""),
            tgt_name_base=t_meta.get("name_base", ""),
            src_suffix=s_meta.get("legal_suffix", ""),
            tgt_suffix=t_meta.get("legal_suffix", ""),
        )
        rows.append(feats)

    return pd.DataFrame(rows, index=pairs_df.index)
