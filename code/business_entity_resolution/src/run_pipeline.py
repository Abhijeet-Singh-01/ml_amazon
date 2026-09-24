"""
Master Pipeline Entry Point for Business Entity Resolution.

Executes the end-to-end resolution pipeline:
1. Data Ingestion & Preprocessing
2. Normalization (Text, Corporate Suffixes, Addresses)
3. Multi-Strategy Blocking (TF-IDF, Address, Embedding)
4. Candidate Pair Union (Deduplication without candidate loss)
5. Feature Engineering (String similarities, Address, Country, Rank)
6. Model Inference / Training (LightGBM)
7. Threshold Calibration & Application
8. One-to-One Constraint Enforcement (Greedy bipartite matching)
9. Rigorous Sanity Checks & Invariant Verification
10. Atomic Output Assembly:
    - output/candidate_pairs.tsv
    - output/matching_results.tsv
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure src directory is on sys.path for direct script execution
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from config import config
from normalize.run_normalize import run_normalization
from blocking.tfidf_blocker import generate_tfidf_candidates
from blocking.address_blocker import generate_address_candidates
from blocking.embedding_blocker import generate_embedding_candidates
from blocking.union_blocks import union_candidate_blocks
from blocking.recall_eval import evaluate_blocking_recall
from features.build_feature_matrix import build_feature_matrix
from model.train_lgbm import (
    prepare_training_data,
    train_matching_model,
    save_model_artifacts,
)
from model.threshold_tuning import find_optimal_threshold, save_threshold_config, load_threshold_config
from model.one_to_one_constraint import apply_one_to_one_constraint
from model.predict import predict_candidate_scores
from postprocess.assemble_outputs import assemble_and_export_outputs
from postprocess.sanity_checks import run_full_sanity_checks
from evaluation.f_beta_macro import evaluate_pair_f_beta, evaluate_macro_f_beta_per_entity
from evaluation.cross_country_eval import evaluate_cross_country_performance


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("entity_resolution")


def load_input_table(path: Path) -> pd.DataFrame:
    """Load CSV or TSV tabular file safely."""
    if not path.exists():
        logger.error("Input file not found at: %s", path)
        raise FileNotFoundError(f"Input file not found at: {path}")
    sep = "\t" if path.suffix in [".tsv", ".tab"] else ","
    return pd.read_csv(path, sep=sep, dtype=str)


def run_pipeline(
    source_path: Path,
    target_path: Path,
    ground_truth_path: Optional[Path] = None,
    output_candidate_path: Optional[Path] = None,
    output_matching_path: Optional[Path] = None,
    is_training: bool = False,
) -> None:
    """Execute complete end-to-end resolution pipeline."""
    start_time = time.time()
    logger.info("=== STARTING BUSINESS ENTITY RESOLUTION PIPELINE ===")

    # Resolve output paths
    out_cand_path = output_candidate_path or config.paths.output_candidate_pairs_path
    out_match_path = output_matching_path or config.paths.output_matching_results_path

    # Step 1: Load Data
    logger.info("Step 1/10: Loading raw input tables...")
    source_raw = load_input_table(source_path)
    target_raw = load_input_table(target_path)
    logger.info("Loaded %d source entities and %d target entities.", len(source_raw), len(target_raw))

    gt_df = None
    if ground_truth_path and ground_truth_path.exists():
        gt_df = load_input_table(ground_truth_path)
        logger.info("Loaded %d ground truth matching pairs.", len(gt_df))

    # Step 2: Normalization
    logger.info("Step 2/10: Running text, suffix, and address normalization...")
    norm_source, norm_target = run_normalization(
        source_raw,
        target_raw,
        id_col="id" if "id" in source_raw.columns else source_raw.columns[0],
        name_col="name" if "name" in source_raw.columns else source_raw.columns[1],
        address_col="address" if "address" in source_raw.columns else None,
        country_col="country" if "country" in source_raw.columns else None,
    )
    logger.info("Normalization complete for source and target tables.")

    # Step 3: Multi-Strategy Blocking
    logger.info("Step 3/10: Executing multi-strategy blocking...")
    tfidf_cands = generate_tfidf_candidates(
        norm_source,
        norm_target,
        ngram_range=config.blocking.tfidf_ngram_range,
        top_k=config.blocking.tfidf_top_k,
        min_similarity=config.blocking.tfidf_min_similarity,
    )
    logger.info("Generated %d TF-IDF candidate pairs.", len(tfidf_cands))

    addr_cands = generate_address_candidates(
        norm_source,
        norm_target,
    )
    logger.info("Generated %d address/postal candidate pairs.", len(addr_cands))

    emb_cands = generate_embedding_candidates(
        norm_source,
        norm_target,
        embedding_dim=config.blocking.embedding_dimension,
        top_k=config.blocking.embedding_top_k,
        min_similarity=config.blocking.embedding_min_similarity,
        random_state=config.random_seed,
    )
    logger.info("Generated %d dense embedding candidate pairs.", len(emb_cands))

    # Step 4: Candidate Pair Union
    logger.info("Step 4/10: Merging candidate blocks (Lossless Union)...")
    candidate_pairs = union_candidate_blocks(
        [tfidf_cands, addr_cands, emb_cands],
        max_candidates_per_source=config.blocking.max_candidates_per_source,
    )
    logger.info("Total unique candidates after union: %d", len(candidate_pairs))

    # Evaluate blocking recall if ground truth provided
    if gt_df is not None:
        recall_metrics = evaluate_blocking_recall(
            candidate_pairs,
            gt_df,
            total_source_count=len(norm_source),
            total_target_count=len(norm_target),
        )
        logger.info(
            "Blocking Recall: %.4f (%d / %d true pairs captured)",
            recall_metrics["pair_completeness"],
            recall_metrics["retained_true_pairs"],
            recall_metrics["total_ground_truth_pairs"],
        )

    # Step 5: Feature Engineering
    logger.info("Step 5/10: Building master feature matrix...")
    X, y, feature_names = build_feature_matrix(
        candidate_pairs,
        norm_source,
        norm_target,
        ground_truth_df=gt_df,
    )
    logger.info("Constructed feature matrix: shape %s across %d features.", X.shape, len(feature_names))

    # Step 6: Model Training or Inference
    if is_training and y is not None:
        logger.info("Step 6/10: Training LightGBM matching model...")
        X_tr, X_val, y_tr, y_val = prepare_training_data(
            X, y, val_size=config.model.validation_split_ratio, random_seed=config.random_seed
        )
        model = train_matching_model(
            X_tr,
            y_tr,
            X_val,
            y_val,
            n_estimators=config.model.n_estimators,
            learning_rate=config.model.learning_rate,
            num_leaves=config.model.num_leaves,
            max_depth=config.model.max_depth,
            scale_pos_weight=config.model.scale_pos_weight,
            random_seed=config.random_seed,
        )
        save_model_artifacts(model, feature_names, config.paths.model_artifact_path)
        logger.info("Saved trained model artifact to %s", config.paths.model_artifact_path)

        # Step 7: Threshold Calibration
        logger.info("Step 7/10: Tuning decision threshold on validation data...")
        val_probs = model.predict_proba(X_val)[:, 1]
        tuning_res = find_optimal_threshold(
            y_val.values,
            val_probs,
            beta=config.threshold.beta,
            grid_min=config.threshold.grid_min,
            grid_max=config.threshold.grid_max,
            grid_steps=config.threshold.grid_steps,
        )
        save_threshold_config(tuning_res, config.paths.threshold_config_path)
        logger.info(
            "Optimal threshold: %.4f (Validation F-beta: %.4f, Prec: %.4f, Rec: %.4f)",
            tuning_res["optimal_threshold"],
            tuning_res["best_f_beta"],
            tuning_res["best_precision"],
            tuning_res["best_recall"],
        )
        threshold = tuning_res["optimal_threshold"]
    else:
        logger.info("Step 6/10: Scoring candidate pairs using existing model artifact...")
        threshold = load_threshold_config(
            config.paths.threshold_config_path,
            default_threshold=config.threshold.default_threshold,
        )

    # Score all candidates
    scored_candidates = predict_candidate_scores(
        candidate_pairs,
        X,
        model_artifact_path=config.paths.model_artifact_path,
        threshold_config_path=config.paths.threshold_config_path,
        default_threshold=threshold,
    )
    logger.info("Scored %d candidate pairs.", len(scored_candidates))

    # Filter positive matches above threshold
    positives = scored_candidates[scored_candidates["is_match"]].copy()
    logger.info("Candidate pairs exceeding threshold (%.4f): %d", threshold, len(positives))

    # Step 8: One-to-One Constraint Enforcement
    logger.info("Step 8/10: Enforcing one-to-one matching constraint...")
    resolved_matches = apply_one_to_one_constraint(
        positives,
        source_id_col="source_id",
        target_id_col="target_id",
        score_col="match_probability",
    )
    logger.info("Final resolved matches after 1-to-1 constraint: %d", len(resolved_matches))

    # Step 9: Postprocess & Sanity Checks
    logger.info("Step 9/10: Running integrity and sanity checks...")
    run_full_sanity_checks(
        candidates_df=scored_candidates,
        matches_df=resolved_matches,
        enforce_one_to_one=config.threshold.enforce_one_to_one,
    )
    logger.info("All sanity checks passed successfully.")

    # Step 10: Atomic Output Assembly
    logger.info("Step 10/10: Assembling and exporting output TSVs...")
    assemble_and_export_outputs(
        scored_candidates_df=scored_candidates,
        final_matches_df=resolved_matches,
        output_candidate_path=out_cand_path,
        output_matching_path=out_match_path,
    )
    logger.info("Successfully exported outputs:")
    logger.info("  -> Candidate pairs:  %s", out_cand_path)
    logger.info("  -> Matching results: %s", out_match_path)

    # Optional Evaluation Reporting
    if gt_df is not None:
        eval_metrics = evaluate_pair_f_beta(resolved_matches, gt_df, beta=config.threshold.beta)
        logger.info("=== EVALUATION RESULTS ===")
        logger.info("  Precision: %.4f", eval_metrics["precision"])
        logger.info("  Recall:    %.4f", eval_metrics["recall"])
        logger.info("  F-beta:    %.4f", eval_metrics["f_beta"])

    elapsed = time.time() - start_time
    logger.info("Pipeline executed successfully in %.2f seconds.", elapsed)


def main():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument("--source", type=str, default=str(config.paths.source_data_path), help="Path to source entities file")
    parser.add_argument("--target", type=str, default=str(config.paths.target_data_path), help="Path to target entities file")
    parser.add_argument("--ground-truth", type=str, default=str(config.paths.ground_truth_path), help="Path to ground truth matches")
    parser.add_argument("--output-candidates", type=str, default=str(config.paths.output_candidate_pairs_path), help="Output path for candidate_pairs.tsv")
    parser.add_argument("--output-matching", type=str, default=str(config.paths.output_matching_results_path), help="Output path for matching_results.tsv")
    parser.add_argument("--train", action="store_true", help="Flag to train model on input datasets")

    args = parser.parse_args()

    run_pipeline(
        source_path=Path(args.source),
        target_path=Path(args.target),
        ground_truth_path=Path(args.ground_truth) if args.ground_truth else None,
        output_candidate_path=Path(args.output_candidates),
        output_matching_path=Path(args.output_matching),
        is_training=args.train,
    )


if __name__ == "__main__":
    main()
