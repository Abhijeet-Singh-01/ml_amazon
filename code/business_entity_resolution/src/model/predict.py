"""
Candidate Pair Scoring and Inference Pipeline.

Loads serialized model artifacts and tuned decision thresholds to score
all candidate pairs, outputting calibrated match probabilities and positive
match assignments.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .train_lgbm import load_model_artifacts
from .threshold_tuning import load_threshold_config


def predict_candidate_scores(
    candidate_pairs_df: pd.DataFrame,
    X: pd.DataFrame,
    model_artifact_path: Path,
    threshold_config_path: Optional[Path] = None,
    default_threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Score candidate pairs using trained model artifact and calibrated threshold.

    Parameters
    ----------
    candidate_pairs_df : pd.DataFrame
        Candidate pairs DataFrame containing ['source_id', 'target_id'].
    X : pd.DataFrame
        Constructed feature matrix matching the candidate pairs rows.
    model_artifact_path : Path
        Path to persisted LightGBM model artifact.
    threshold_config_path : Optional[Path]
        Path to optimal threshold JSON configuration.
    default_threshold : float
        Threshold fallback if threshold config is not found.

    Returns
    -------
    pd.DataFrame
        Copy of candidate_pairs_df augmented with:
        - `match_probability`: Predicted probability [0.0, 1.0].
        - `is_match`: Boolean flag indicating if probability >= threshold.
    """
    if candidate_pairs_df.empty or X.empty:
        df_empty = candidate_pairs_df.copy()
        df_empty["match_probability"] = 0.0
        df_empty["is_match"] = False
        return df_empty

    # Load model and expected feature names
    model, expected_features = load_model_artifacts(model_artifact_path)

    # Align feature columns (fill missing with 0.0)
    for feat in expected_features:
        if feat not in X.columns:
            X[feat] = 0.0
    X_aligned = X[expected_features]

    # Predict probabilities for class 1
    probabilities = model.predict_proba(X_aligned)[:, 1]

    # Load calibrated threshold
    threshold = default_threshold
    if threshold_config_path and threshold_config_path.exists():
        threshold = load_threshold_config(threshold_config_path, default_threshold=default_threshold)

    scored_df = candidate_pairs_df.copy().reset_index(drop=True)
    scored_df["match_probability"] = probabilities.astype(float)
    scored_df["is_match"] = scored_df["match_probability"] >= threshold

    return scored_df
