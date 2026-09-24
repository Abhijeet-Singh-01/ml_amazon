"""
Candidate Ranking and Relative-Position Features.

Generates group-relative features for each candidate pair within the source entity's
candidate set:
- Relative rank by preliminary blocking similarity (TF-IDF / composite)
- Score margin to the top candidate (delta from top-1)
- Score margin between top-1 and top-2 candidate
- Total candidate pool size for the source entity
"""

import numpy as np
import pandas as pd


def compute_rank_features(
    pairs_df: pd.DataFrame,
    score_col: str = "tfidf_similarity",
) -> pd.DataFrame:
    """
    Compute group-wise ranking features per source entity.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        Candidate pairs containing 'source_id', 'target_id', and a score column.
    score_col : str
        Column to use for ranking candidates within each source group.

    Returns
    -------
    pd.DataFrame
        DataFrame with ranking features:
        - `rank_within_source`
        - `score_margin_from_top1`
        - `candidates_per_source`
        - `is_top1_candidate`
    """
    if pairs_df.empty:
        return pd.DataFrame(
            columns=[
                "rank_within_source",
                "score_margin_from_top1",
                "candidates_per_source",
                "is_top1_candidate",
            ],
            index=pairs_df.index,
        )

    df = pairs_df.copy()
    if score_col not in df.columns:
        df[score_col] = 0.0

    # Ensure numeric score
    scores = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
    df["_rank_score"] = scores

    # Rank descending within each source_id (1 is highest score)
    df["rank_within_source"] = (
        df.groupby("source_id")["_rank_score"]
        .rank(ascending=False, method="min")
        .astype(float)
    )

    # Max score per source
    max_scores = df.groupby("source_id")["_rank_score"].transform("max")
    df["score_margin_from_top1"] = (max_scores - df["_rank_score"]).astype(float)

    # Total candidates per source
    df["candidates_per_source"] = (
        df.groupby("source_id")["target_id"].transform("count").astype(float)
    )

    # Top-1 binary indicator
    df["is_top1_candidate"] = (df["rank_within_source"] == 1.0).astype(float)

    feature_cols = [
        "rank_within_source",
        "score_margin_from_top1",
        "candidates_per_source",
        "is_top1_candidate",
    ]
    return df[feature_cols].copy()
