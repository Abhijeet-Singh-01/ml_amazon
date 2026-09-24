"""
Decision Threshold Tuning.

Determines the optimal decision threshold on validation data to maximize
the macro F-beta evaluation metric. Avoids arbitrary hard-coding of classification
cutoffs and serializes the selected threshold for consistent inference.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def compute_f_beta(precision: float, recall: float, beta: float = 1.0) -> float:
    """Calculate F-beta score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = 1.0,
    grid_min: float = 0.05,
    grid_max: float = 0.95,
    grid_steps: int = 91,
) -> Dict[str, float]:
    """
    Search grid of candidate thresholds to maximize F-beta score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_prob : np.ndarray
        Model predicted probabilities in [0.0, 1.0].
    beta : float
        F-beta weighting parameter (1.0 for F1, 0.5 for precision-weighted, 2.0 for recall-weighted).
    grid_min : float
        Minimum threshold boundary.
    grid_max : float
        Maximum threshold boundary.
    grid_steps : int
        Number of search increments across grid.

    Returns
    -------
    Dict[str, float]
        Dictionary with optimal threshold and corresponding evaluation metrics:
        - `optimal_threshold`
        - `best_f_beta`
        - `best_precision`
        - `best_recall`
    """
    thresholds = np.linspace(grid_min, grid_max, grid_steps)
    best_f_beta = -1.0
    best_threshold = 0.50
    best_prec = 0.0
    best_rec = 0.0

    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob).astype(float)
    total_positives = np.sum(y_t == 1)

    if total_positives == 0:
        return {
            "optimal_threshold": 0.50,
            "best_f_beta": 0.0,
            "best_precision": 0.0,
            "best_recall": 0.0,
        }

    for tau in thresholds:
        y_pred = (y_p >= tau).astype(int)
        tp = np.sum((y_pred == 1) & (y_t == 1))
        fp = np.sum((y_pred == 1) & (y_t == 0))
        fn = np.sum((y_pred == 0) & (y_t == 1))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_beta = compute_f_beta(prec, rec, beta=beta)

        if f_beta > best_f_beta:
            best_f_beta = f_beta
            best_threshold = float(tau)
            best_prec = float(prec)
            best_rec = float(rec)

    return {
        "optimal_threshold": round(best_threshold, 4),
        "best_f_beta": round(best_f_beta, 4),
        "best_precision": round(best_prec, 4),
        "best_recall": round(best_rec, 4),
        "beta": beta,
    }


def save_threshold_config(threshold_info: Dict[str, float], output_path: Path) -> None:
    """Save tuned threshold results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(threshold_info, f, indent=2)


def load_threshold_config(config_path: Path, default_threshold: float = 0.50) -> float:
    """Load tuned threshold from JSON file, falling back to default if unavailable."""
    if not config_path.exists():
        return default_threshold
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return float(data.get("optimal_threshold", default_threshold))
    except Exception:
        return default_threshold
