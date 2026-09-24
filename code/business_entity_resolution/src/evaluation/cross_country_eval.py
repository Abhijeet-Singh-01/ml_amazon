"""
Cross-Country Evaluation Module.

Computes disaggregated performance metrics across distinct geographic regions
and countries to detect regional disparities, underperforming markets, or
jurisdiction-specific entity resolution weaknesses.
"""

from typing import Dict, Optional, Tuple
import pandas as pd

from .f_beta_macro import evaluate_pair_f_beta


def evaluate_cross_country_performance(
    predicted_pairs_df: pd.DataFrame,
    ground_truth_pairs_df: pd.DataFrame,
    source_df: pd.DataFrame,
    country_col: str = "country_code",
    beta: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Evaluate precision, recall, and F-beta broken down by country.

    Parameters
    ----------
    predicted_pairs_df : pd.DataFrame
        Predicted matching pairs with ['source_id', 'target_id'].
    ground_truth_pairs_df : pd.DataFrame
        Ground truth matching pairs with ['source_id', 'target_id'].
    source_df : pd.DataFrame
        Source entities table containing country assignments.
    country_col : str
        Column name for country code.
    beta : float
        F-beta parameter.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        - country_metrics_df: Table of metrics per country.
        - summary: Aggregate metrics across countries (mean, min, max F-beta).
    """
    if source_df.empty or country_col not in source_df.columns:
        return pd.DataFrame(), {}

    # Map source_id to country
    src_country_map = source_df.set_index("id")[country_col].fillna("UNKNOWN").astype(str).to_dict()

    pred_with_country = predicted_pairs_df.copy()
    pred_with_country["country"] = pred_with_country["source_id"].map(src_country_map).fillna("UNKNOWN")

    gt_with_country = ground_truth_pairs_df.copy()
    gt_with_country["country"] = gt_with_country["source_id"].map(src_country_map).fillna("UNKNOWN")

    all_countries = sorted(list(set(pred_with_country["country"]).union(set(gt_with_country["country"]))))

    records = []
    for c in all_countries:
        c_preds = pred_with_country[pred_with_country["country"] == c]
        c_gts = gt_with_country[gt_with_country["country"] == c]

        res = evaluate_pair_f_beta(c_preds, c_gts, beta=beta)
        records.append({
            "country": c,
            "f_beta": res["f_beta"],
            "precision": res["precision"],
            "recall": res["recall"],
            "true_positives": res["true_positives"],
            "false_positives": res["false_positives"],
            "false_negatives": res["false_negatives"],
            "total_ground_truth": res["true_positives"] + res["false_negatives"],
            "total_predicted": res["true_positives"] + res["false_positives"],
        })

    country_df = pd.DataFrame(records)

    summary = {
        "mean_country_f_beta": float(country_df["f_beta"].mean()) if not country_df.empty else 0.0,
        "min_country_f_beta": float(country_df["f_beta"].min()) if not country_df.empty else 0.0,
        "max_country_f_beta": float(country_df["f_beta"].max()) if not country_df.empty else 0.0,
        "total_countries_evaluated": len(country_df),
    }

    return country_df, summary
