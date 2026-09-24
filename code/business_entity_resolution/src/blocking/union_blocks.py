"""
Union of Multi-Strategy Candidate Blocks.

Combines candidate pairs produced by heterogeneous blocking strategies
(TF-IDF, address-based, dense embeddings) into a unified candidate pool.
Guarantees deduplication without loss of candidate pairs, preserving origin
indicator flags and initial similarity scores.
"""

from typing import List, Optional
import pandas as pd


def union_candidate_blocks(
    candidate_dfs: List[pd.DataFrame],
    max_candidates_per_source: Optional[int] = None,
) -> pd.DataFrame:
    """
    Union multiple candidate pairs DataFrames on (source_id, target_id).

    Parameters
    ----------
    candidate_dfs : List[pd.DataFrame]
        List of candidate DataFrames from different blocking strategies.
    max_candidates_per_source : Optional[int]
        Optional upper bound on total candidates per source entity to maintain
        tractable feature engineering.

    Returns
    -------
    pd.DataFrame
        Unified candidate pairs DataFrame with standardized schema:
        - `source_id`
        - `target_id`
        - Preserved origin flags and similarity scores
    """
    non_empty_dfs = [
        df.copy() for df in candidate_dfs
        if df is not None and not df.empty and "source_id" in df.columns and "target_id" in df.columns
    ]

    if not non_empty_dfs:
        return pd.DataFrame(columns=["source_id", "target_id"])

    # Ensure IDs are string-typed in all DataFrames
    for df in non_empty_dfs:
        df["source_id"] = df["source_id"].astype(str)
        df["target_id"] = df["target_id"].astype(str)

    # Iterative full outer merge to retain all blocker features
    unified = non_empty_dfs[0]
    for df in non_empty_dfs[1:]:
        unified = pd.merge(
            unified,
            df,
            on=["source_id", "target_id"],
            how="outer",
            suffixes=("", "_dup"),
        )
        # Drop duplicate columns if any were suffixed
        dup_cols = [c for c in unified.columns if c.endswith("_dup")]
        if dup_cols:
            unified.drop(columns=dup_cols, inplace=True)

    # Ensure deduplication on pair key
    unified = unified.drop_duplicates(subset=["source_id", "target_id"]).reset_index(drop=True)

    # Fill default values for blocker indicators and scores
    if "tfidf_similarity" in unified.columns:
        unified["tfidf_similarity"] = unified["tfidf_similarity"].fillna(0.0)
    if "embedding_similarity" in unified.columns:
        unified["embedding_similarity"] = unified["embedding_similarity"].fillna(0.0)
    if "address_blocked" in unified.columns:
        unified["address_blocked"] = unified["address_blocked"].fillna(0).astype(int)

    # Cap candidates per source if specified
    if max_candidates_per_source is not None and max_candidates_per_source > 0:
        # Prioritize by highest available similarity score
        sort_col = "tfidf_similarity" if "tfidf_similarity" in unified.columns else None
        if sort_col:
            unified = (
                unified.sort_values(sort_col, ascending=False)
                .groupby("source_id")
                .head(max_candidates_per_source)
                .reset_index(drop=True)
            )

    return unified
