"""
Central Configuration for Business Entity Resolution Pipeline.

Defines all file paths, blocking parameters, feature generation settings,
model hyperparameters, threshold search grids, and reproducible random seeds.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


# Base Directory Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
SUBMISSION_ROOT = PROJECT_DIR.parent.parent
OUTPUT_DIR = SUBMISSION_ROOT / "output"
ARTIFACTS_DIR = SRC_DIR / "artifacts"


@dataclass
class PathConfig:
    """Paths for input datasets, artifacts, and output files."""
    # Input data paths (configurable via CLI or environment)
    source_data_path: Path = PROJECT_DIR / "data" / "source_entities.csv"
    target_data_path: Path = PROJECT_DIR / "data" / "target_entities.csv"
    ground_truth_path: Path = PROJECT_DIR / "data" / "ground_truth.tsv"

    # Serialized model & pipeline artifacts
    model_artifact_path: Path = ARTIFACTS_DIR / "lgbm_model.joblib"
    tfidf_vectorizer_path: Path = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
    threshold_config_path: Path = ARTIFACTS_DIR / "optimal_threshold.json"

    # Submission output paths
    output_matching_results_path: Path = OUTPUT_DIR / "matching_results.tsv"
    output_candidate_pairs_path: Path = OUTPUT_DIR / "candidate_pairs.tsv"


@dataclass
class BlockingConfig:
    """Parameters controlling candidate blocking methods."""
    # TF-IDF candidate generation
    tfidf_ngram_range: Tuple[int, int] = (2, 4)
    tfidf_analyzer: str = "char_wb"
    tfidf_min_df: int = 1
    tfidf_top_k: int = 15
    tfidf_min_similarity: float = 0.25

    # Address-based blocking
    address_exact_fields: List[str] = field(
        default_factory=lambda: ["postal_code", "country_code"]
    )
    address_top_k: int = 10

    # Embedding / Dense similarity blocking
    embedding_top_k: int = 15
    embedding_min_similarity: float = 0.40
    embedding_dimension: int = 64

    # Combined candidate budget per source entity
    max_candidates_per_source: int = 50


@dataclass
class FeatureConfig:
    """Parameters for feature extraction and similarity metrics."""
    # Text similarity metrics
    include_levenshtein: bool = True
    include_jaro_winkler: bool = True
    include_token_sort: bool = True
    include_token_set: bool = True
    include_lcs: bool = True
    include_lcp: bool = True

    # Address & location features
    include_address_similarity: bool = True
    include_country_match: bool = True
    include_postal_match: bool = True

    # Candidate ranking features
    include_rank_features: bool = True
    include_score_margins: bool = True


@dataclass
class ModelConfig:
    """Hyperparameters for LightGBM matching model."""
    random_seed: int = 42
    objective: str = "binary"
    metric: str = "binary_logloss"
    boosting_type: str = "gbdt"
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = 6
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    scale_pos_weight: float = 5.0
    early_stopping_rounds: int = 30
    validation_split_ratio: float = 0.2


@dataclass
class ThresholdConfig:
    """Settings for decision threshold tuning."""
    beta: float = 1.0  # Macro F-beta (F1 by default)
    grid_min: float = 0.05
    grid_max: float = 0.95
    grid_steps: int = 91
    default_threshold: float = 0.50
    enforce_one_to_one: bool = True


@dataclass
class PipelineConfig:
    """Top-level master configuration combining all sub-configs."""
    paths: PathConfig = field(default_factory=PathConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    random_seed: int = 42
    verbose: bool = True


# Default pipeline configuration singleton
config = PipelineConfig()
