"""
Country-Related Matching Features.

Generates indicators and compatibility signals for country attributes:
- Exact country code match
- Explicit country conflict indicator (strong negative predictor)
- Country missingness flags for source and target records
"""

from typing import Dict, Optional
import pandas as pd


def _safe_country(val: Optional[str]) -> str:
    if val is None or not isinstance(val, str):
        return ""
    return val.strip().upper()


def extract_country_features_pair(
    src_country: str,
    tgt_country: str,
) -> Dict[str, float]:
    """Extract country-level features for a single candidate pair."""
    c1 = _safe_country(src_country)
    c2 = _safe_country(tgt_country)

    has_both = 1.0 if c1 and c2 else 0.0
    country_match = 1.0 if has_both and c1 == c2 else 0.0
    country_mismatch = 1.0 if has_both and c1 != c2 else 0.0
    src_missing = 1.0 if not c1 else 0.0
    tgt_missing = 1.0 if not c2 else 0.0

    return {
        "country_has_both": has_both,
        "country_match": country_match,
        "country_mismatch": country_mismatch,
        "country_src_missing": src_missing,
        "country_tgt_missing": tgt_missing,
    }


def compute_country_features(
    pairs_df: pd.DataFrame,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute vectorized country features across candidate pairs."""
    src_map = source_df.set_index("id")["country_code"].to_dict()
    tgt_map = target_df.set_index("id")["country_code"].to_dict()

    rows = []
    for _, row in pairs_df.iterrows():
        s_id = str(row["source_id"])
        t_id = str(row["target_id"])

        s_c = src_map.get(s_id, "")
        t_c = tgt_map.get(t_id, "")

        rows.append(extract_country_features_pair(s_c, t_c))

    return pd.DataFrame(rows, index=pairs_df.index)
