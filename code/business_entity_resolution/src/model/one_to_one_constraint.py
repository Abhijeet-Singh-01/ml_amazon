"""
One-to-One Matching Constraint Solver.

Enforces strict 1-to-1 matching constraints over candidate entity pairs:
- Resolves conflicts where a source entity maps to multiple target candidates.
- Resolves conflicts where multiple source entities contend for the same target record.
- Implements greedy priority assignment in descending order of predicted match confidence.
"""

from typing import Optional, Set
import pandas as pd


def apply_one_to_one_constraint(
    predictions_df: pd.DataFrame,
    source_id_col: str = "source_id",
    target_id_col: str = "target_id",
    score_col: str = "match_probability",
) -> pd.DataFrame:
    """
    Enforce global one-to-one matching constraint using greedy priority assignment.

    Pairs with higher predicted confidence are matched first. Once a source_id
    or target_id is committed, any competing pairs involving either entity
    are eliminated.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Candidate pairs with match probability scores.
    source_id_col : str
        Source entity identifier column name.
    target_id_col : str
        Target entity identifier column name.
    score_col : str
        Predicted confidence / probability column name used for ranking.

    Returns
    -------
    pd.DataFrame
        Resolved pairs strictly satisfying the 1-to-1 constraint.
    """
    if predictions_df.empty:
        return predictions_df.copy()

    df = predictions_df.copy()
    df[source_id_col] = df[source_id_col].astype(str)
    df[target_id_col] = df[target_id_col].astype(str)
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)

    # Sort pairs by descending match probability
    df_sorted = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)

    matched_sources: Set[str] = set()
    matched_targets: Set[str] = set()
    selected_indices = []

    for idx, row in df_sorted.iterrows():
        s_id = row[source_id_col]
        t_id = row[target_id_col]

        # Check if either entity has already been committed to another match
        if s_id not in matched_sources and t_id not in matched_targets:
            matched_sources.add(s_id)
            matched_targets.add(t_id)
            selected_indices.append(idx)

    resolved_df = df_sorted.iloc[selected_indices].reset_index(drop=True)
    return resolved_df
