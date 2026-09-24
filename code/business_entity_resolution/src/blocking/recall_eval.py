"""
Blocking Recall and Completeness Evaluator.

Assesses the efficacy of the candidate generation stage by measuring:
1. Pair Completeness (Blocking Recall): Fraction of true matching pairs retained in candidate set.
2. Reduction Ratio (RR): Fraction of all possible pairs filtered out by blocking.
3. Candidate Cardinality: Total and average candidates evaluated per source entity.
"""

from typing import Dict, Optional
import pandas as pd


def evaluate_blocking_recall(
    candidates_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    total_source_count: Optional[int] = None,
    total_target_count: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate candidate pool recall against ground truth matching pairs.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Candidate pairs with ['source_id', 'target_id'].
    ground_truth_df : pd.DataFrame
        Ground truth matching pairs with ['source_id', 'target_id'].
    total_source_count : Optional[int]
        Total number of source entities (for Reduction Ratio calculation).
    total_target_count : Optional[int]
        Total number of target entities (for Reduction Ratio calculation).

    Returns
    -------
    Dict[str, float]
        Metrics dictionary containing:
        - `pair_completeness` (recall)
        - `retained_true_pairs`
        - `total_ground_truth_pairs`
        - `total_candidates`
        - `reduction_ratio`
        - `avg_candidates_per_source`
    """
    if ground_truth_df.empty:
        return {
            "pair_completeness": 0.0,
            "retained_true_pairs": 0,
            "total_ground_truth_pairs": 0,
            "total_candidates": len(candidates_df),
            "reduction_ratio": 1.0,
            "avg_candidates_per_source": 0.0,
        }

    cand_set = set(
        zip(
            candidates_df["source_id"].astype(str),
            candidates_df["target_id"].astype(str),
        )
    )
    gt_set = set(
        zip(
            ground_truth_df["source_id"].astype(str),
            ground_truth_df["target_id"].astype(str),
        )
    )

    true_retained = len(gt_set.intersection(cand_set))
    pair_completeness = true_retained / len(gt_set) if len(gt_set) > 0 else 0.0

    # Calculate Reduction Ratio if universe size known
    total_possible_pairs = None
    reduction_ratio = 0.0
    if total_source_count and total_target_count:
        total_possible_pairs = total_source_count * total_target_count
        if total_possible_pairs > 0:
            reduction_ratio = 1.0 - (len(cand_set) / total_possible_pairs)

    avg_cand_per_source = 0.0
    if not candidates_df.empty and "source_id" in candidates_df.columns:
        avg_cand_per_source = len(candidates_df) / candidates_df["source_id"].nunique()

    metrics = {
        "pair_completeness": float(pair_completeness),
        "retained_true_pairs": int(true_retained),
        "total_ground_truth_pairs": int(len(gt_set)),
        "total_candidates": int(len(cand_set)),
        "reduction_ratio": float(reduction_ratio),
        "avg_candidates_per_source": float(avg_cand_per_source),
    }

    return metrics
