"""
Feature Matrix Builder.

Combines name similarities, address similarities, country indicators,
blocker metrics, and ranking features into the final tabular feature matrix
consumed by the LightGBM matching model.
"""

from typing import List, Optional, Tuple
import pandas as pd

from .name_similarity import compute_name_similarity_features
from .address_similarity import compute_address_similarity_features
from .country_features import compute_country_features
from .rank_features import compute_rank_features


def build_feature_matrix(
    candidate_pairs_df: pd.DataFrame,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    ground_truth_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    """
    Construct the master feature matrix for candidate pairs.

    Parameters
    ----------
    candidate_pairs_df : pd.DataFrame
        Candidate pairs containing ['source_id', 'target_id'] and any blocker scores.
    source_df : pd.DataFrame
        Normalized source entities table.
    target_df : pd.DataFrame
        Normalized target entities table.
    ground_truth_df : Optional[pd.DataFrame]
        Ground truth matching pairs (if building training set).

    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.Series], List[str]]
        - X : DataFrame of engineered features.
        - y : Binary Series of target labels (1 for match, 0 for non-match) or None.
        - feature_names : List of feature column names.
    """
    if candidate_pairs_df.empty:
        return pd.DataFrame(), None, []

    pairs = candidate_pairs_df.copy().reset_index(drop=True)
    pairs["source_id"] = pairs["source_id"].astype(str)
    pairs["target_id"] = pairs["target_id"].astype(str)

    # 1. Name features
    name_feats = compute_name_similarity_features(pairs, source_df, target_df)

    # 2. Address features
    addr_feats = compute_address_similarity_features(pairs, source_df, target_df)

    # 3. Country features
    country_feats = compute_country_features(pairs, source_df, target_df)

    # 4. Rank features
    rank_feats = compute_rank_features(pairs, score_col="tfidf_similarity")

    # 5. Blocker signals if present
    blocker_cols = []
    blocker_df = pd.DataFrame(index=pairs.index)
    for col in ["tfidf_similarity", "embedding_similarity", "address_blocked"]:
        if col in pairs.columns:
            blocker_df[col] = pd.to_numeric(pairs[col], errors="coerce").fillna(0.0)
            blocker_cols.append(col)

    # Concatenate all feature blocks
    feature_blocks = [name_feats, addr_feats, country_feats, rank_feats]
    if blocker_cols:
        feature_blocks.append(blocker_df)

    X = pd.concat(feature_blocks, axis=1)
    feature_names = list(X.columns)

    # Construct target label y if ground truth is supplied
    y = None
    if ground_truth_df is not None and not ground_truth_df.empty:
        gt_pairs = set(
            zip(
                ground_truth_df["source_id"].astype(str),
                ground_truth_df["target_id"].astype(str),
            )
        )
        cand_pairs = list(zip(pairs["source_id"], pairs["target_id"]))
        y = pd.Series([1 if p in gt_pairs else 0 for p in cand_pairs], name="is_match")

    return X, y, feature_names
