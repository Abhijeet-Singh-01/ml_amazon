"""
Address-Based Candidate Pair Blocker.

Generates candidate entity pairs by grouping records sharing identical
or near-identical geographic identifiers:
1. Postal Code + Country Code
2. Cleaned Address key (exact or prefix match)
"""

from typing import List, Optional
import pandas as pd


def generate_address_candidates(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_id_col: str = "id",
    target_id_col: str = "id",
    postal_col: str = "postal_code",
    country_col: str = "country_code",
    address_col: str = "address_clean",
    max_candidates_per_key: int = 20,
) -> pd.DataFrame:
    """
    Generate candidate pairs by indexing on postal code, country, and address keys.

    Parameters
    ----------
    source_df : pd.DataFrame
        Normalized source entities table.
    target_df : pd.DataFrame
        Normalized target entities table.
    max_candidates_per_key : int
        Caps excessive cross-products when a postal code contains too many entities.

    Returns
    -------
    pd.DataFrame
        Candidate pairs DataFrame with columns:
        ['source_id', 'target_id', 'address_blocked']
    """
    candidates = []

    # 1. Block on Postal Code + Country
    has_geo = (
        postal_col in source_df.columns
        and country_col in source_df.columns
        and postal_col in target_df.columns
        and country_col in target_df.columns
    )

    if has_geo:
        src_geo = source_df[[source_id_col, postal_col, country_col]].copy()
        tgt_geo = target_df[[target_id_col, postal_col, country_col]].copy()

        # Filter out empty postal codes
        src_geo = src_geo[src_geo[postal_col].str.len() >= 3]
        tgt_geo = tgt_geo[tgt_geo[postal_col].str.len() >= 3]

        merged_geo = pd.merge(
            src_geo,
            tgt_geo,
            on=[postal_col, country_col],
            suffixes=("_src", "_tgt"),
        )

        if not merged_geo.empty:
            # Cap candidates per source to prevent combinatoric explosion
            capped_geo = (
                merged_geo.groupby(f"{source_id_col}_src")
                .head(max_candidates_per_key)
                .reset_index(drop=True)
            )
            pairs_geo = pd.DataFrame({
                "source_id": capped_geo[f"{source_id_col}_src"].astype(str),
                "target_id": capped_geo[f"{target_id_col}_tgt"].astype(str),
                "address_blocked": 1,
            })
            candidates.append(pairs_geo)

    # 2. Block on exact Cleaned Address (if non-empty)
    if address_col in source_df.columns and address_col in target_df.columns:
        src_addr = source_df[[source_id_col, address_col]].copy()
        tgt_addr = target_df[[target_id_col, address_col]].copy()

        src_addr = src_addr[src_addr[address_col].str.len() >= 5]
        tgt_addr = tgt_addr[tgt_addr[address_col].str.len() >= 5]

        merged_addr = pd.merge(
            src_addr,
            tgt_addr,
            on=address_col,
            suffixes=("_src", "_tgt"),
        )

        if not merged_addr.empty:
            capped_addr = (
                merged_addr.groupby(f"{source_id_col}_src")
                .head(max_candidates_per_key)
                .reset_index(drop=True)
            )
            pairs_addr = pd.DataFrame({
                "source_id": capped_addr[f"{source_id_col}_src"].astype(str),
                "target_id": capped_addr[f"{target_id_col}_tgt"].astype(str),
                "address_blocked": 1,
            })
            candidates.append(pairs_addr)

    if not candidates:
        return pd.DataFrame(columns=["source_id", "target_id", "address_blocked"])

    combined = pd.concat(candidates, ignore_index=True)
    return combined.drop_duplicates(subset=["source_id", "target_id"]).reset_index(drop=True)
