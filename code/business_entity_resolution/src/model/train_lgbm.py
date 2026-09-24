"""
LightGBM Matching Model Training.

Modular workflow separating:
1. Training data preparation and label alignment
2. Feature matrix generation
3. Model training with early stopping
4. Validation and metric logging
5. Model persistence to artifacts directory
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier


def prepare_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split feature matrix and labels into train and validation partitions.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=random_seed,
        stratify=y,
    )
    return X_train, X_val, y_train, y_val


def train_matching_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = 6,
    scale_pos_weight: float = 5.0,
    random_seed: int = 42,
) -> LGBMClassifier:
    """
    Train a LightGBM binary classifier for entity pair matching.
    """
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
        random_state=random_seed,
        verbose=-1,
    )

    if X_val is not None and y_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
        )
    else:
        model.fit(X_train, y_train)

    return model


def save_model_artifacts(
    model: LGBMClassifier,
    feature_names: List[str],
    artifact_path: Path,
) -> None:
    """
    Persist trained model and feature metadata to disk.
    """
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_names": feature_names,
    }
    joblib.dump(payload, artifact_path)


def load_model_artifacts(artifact_path: Path) -> Tuple[LGBMClassifier, List[str]]:
    """
    Load persisted model and feature names from disk.
    """
    payload = joblib.load(artifact_path)
    return payload["model"], payload["feature_names"]
