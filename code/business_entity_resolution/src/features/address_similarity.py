"""
Address and Geographic Similarity Features.

Calculates multi-granular address similarity features:
- Cleaned address string edit distances & token intersections
- Postal code exact matches and common prefix ratios
- Street number identity indicators
- Missingness indicators for geographic metadata
"""

from typing import Dict, Optional
import pandas as pd

from .name_similarity import compute_levenshtein_sim, compute_token_set_sim


def _safe_str(val: Optional[str]) -> str:
    if val is None or not isinstance(val, str):
        return ""
    return val.strip()


def extract_address_features_pair(
    src_addr: str,
    tgt_addr: str,
    src_postal: str = "",
    tgt_postal: str = "",
) -> Dict[str, float]:
    """Extract address similarity features for a single candidate pair."""
    a1 = _safe_str(src_addr)
    a2 = _safe_str(tgt_addr)
    p1 = _safe_str(src_postal).upper()
    p2 = _safe_str(tgt_postal).upper()

    has_both_addr = 1.0 if a1 and a2 else 0.0
    exact_addr = 1.0 if has_both_addr and a1 == a2 else 0.0

    addr_lev = compute_levenshtein_sim(a1, a2) if has_both_addr else 0.0
    addr_tok_set = compute_token_set_sim(a1, a2) if has_both_addr else 0.0

    has_both_postal = 1.0 if p1 and p2 else 0.0
    exact_postal = 1.0 if has_both_postal and p1 == p2 else 0.0

    # Postal code prefix match (e.g. first 3 characters)
    postal_prefix_match = 0.0
    if has_both_postal:
        min_p_len = min(len(p1), len(p2))
        if min_p_len >= 3 and p1[:3] == p2[:3]:
            postal_prefix_match = 1.0

    return {
        "addr_has_both": has_both_addr,
        "addr_exact_match": exact_addr,
        "addr_levenshtein": addr_lev,
        "addr_token_set": addr_tok_set,
        "postal_has_both": has_both_postal,
        "postal_exact_match": exact_postal,
        "postal_prefix_match": postal_prefix_match,
    }


def compute_address_similarity_features(
    pairs_df: pd.DataFrame,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute vectorized address similarity features across candidate pairs."""
    src_map = source_df.set_index("id")[
        ["address_clean", "postal_code"]
    ].to_dict("index")
    tgt_map = target_df.set_index("id")[
        ["address_clean", "postal_code"]
    ].to_dict("index")

    rows = []
    for _, row in pairs_df.iterrows():
        s_id = str(row["source_id"])
        t_id = str(row["target_id"])

        s_meta = src_map.get(s_id, {})
        t_meta = tgt_map.get(t_id, {})

        feats = extract_address_features_pair(
            src_addr=s_meta.get("address_clean", ""),
            tgt_addr=t_meta.get("address_clean", ""),
            src_postal=s_meta.get("postal_code", ""),
            tgt_postal=t_meta.get("postal_code", ""),
        )
        rows.append(feats)

    return pd.DataFrame(rows, index=pairs_df.index)
