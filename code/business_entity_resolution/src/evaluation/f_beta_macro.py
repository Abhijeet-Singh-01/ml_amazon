"""
Macro F-beta Evaluation Metric Implementation.

Computes Macro F-beta evaluation metrics for Business Entity Resolution:
1. Entity-Level Macro F-beta: Averages F-beta scores computed per source entity record.
2. Global Pair-Level F-beta: Computes global precision, recall, and harmonic F-beta over all pairs.

Formula:
    F_beta = (1 + beta^2) * (Precision * Recall) / ((beta^2 * Precision) + Recall)
"""

from typing import Dict, Set, Tuple
import numpy as np
import pandas as pd


def compute_f_beta_score(precision: float, recall: float, beta: float = 1.0) -> float:
    """Calculate F-beta given precision and recall."""
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0
    return (1.0 + beta_sq) * (precision * recall) / denom


def evaluate_pair_f_beta(
    predicted_pairs_df: pd.DataFrame,
    ground_truth_pairs_df: pd.DataFrame,
    beta: float = 1.0,
) -> Dict[str, float]:
    """
    Compute global pair-level precision, recall, and F-beta.

    Parameters
    ----------
    predicted_pairs_df : pd.DataFrame
        Predicted matching pairs with ['source_id', 'target_id'].
    ground_truth_pairs_df : pd.DataFrame
        Ground truth matching pairs with ['source_id', 'target_id'].
    beta : float
        F-beta parameter (default 1.0 for standard F1).

    Returns
    -------
    Dict[str, float]
        Dictionary with precision, recall, f_beta, tp, fp, fn.
    """
    pred_set: Set[Tuple[str, str]] = set(
        zip(
            predicted_pairs_df["source_id"].astype(str),
            predicted_pairs_df["target_id"].astype(str),
        )
    )
    gt_set: Set[Tuple[str, str]] = set(
        zip(
            ground_truth_pairs_df["source_id"].astype(str),
            ground_truth_pairs_df["target_id"].astype(str),
        )
    )

    tp = len(pred_set.intersection(gt_set))
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f_beta = compute_f_beta_score(precision, recall, beta=beta)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f_beta": float(f_beta),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "beta": float(beta),
    }


def evaluate_macro_f_beta_per_entity(
    predicted_pairs_df: pd.DataFrame,
    ground_truth_pairs_df: pd.DataFrame,
    beta: float = 1.0,
) -> Dict[str, float]:
    """
    Compute Macro F-beta averaged across individual source entities.
    """
    # Group target IDs by source entity
    pred_grouped = (
        predicted_pairs_df.groupby("source_id")["target_id"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )
    gt_grouped = (
        ground_truth_pairs_df.groupby("source_id")["target_id"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )

    all_source_ids = set(pred_grouped.keys()).union(set(gt_grouped.keys()))
    if not all_source_ids:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f_beta": 0.0}

    precisions = []
    recalls = []
    f_betas = []

    for src_id in all_source_ids:
        preds = pred_grouped.get(src_id, set())
        actuals = gt_grouped.get(src_id, set())

        tp = len(preds.intersection(actuals))
        p = tp / len(preds) if len(preds) > 0 else (1.0 if len(actuals) == 0 else 0.0)
        r = tp / len(actuals) if len(actuals) > 0 else (1.0 if len(preds) == 0 else 0.0)
        fb = compute_f_beta_score(p, r, beta=beta)

        precisions.append(p)
        recalls.append(r)
        f_betas.append(fb)

    return {
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f_beta": float(np.mean(f_betas)),
        "num_entities": int(len(all_source_ids)),
        "beta": float(beta),
    }
