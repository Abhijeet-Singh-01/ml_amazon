"""
Final Output Assembly Stage.

Responsible for generating the mandatory submission files:
- output/matching_results.tsv
- output/candidate_pairs.tsv

CRITICAL INVARIANT:
Both output files are generated from the SAME pipeline run and the EXACT SAME
underlying candidate/prediction DataFrame. There is zero possibility of drift
between the candidate set evaluated by the model and candidate_pairs.tsv.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd


def assemble_and_export_outputs(
    scored_candidates_df: pd.DataFrame,
    final_matches_df: pd.DataFrame,
    output_candidate_path: Path,
    output_matching_path: Path,
    source_id_col: str = "source_id",
    target_id_col: str = "target_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble and export submission TSV files atomically from the identical candidate data.

    Parameters
    ----------
    scored_candidates_df : pd.DataFrame
        The complete, identical DataFrame of candidate pairs scored by the model.
    final_matches_df : pd.DataFrame
        The subset of candidate pairs resolved as true matches after thresholding
        and one-to-one constraint enforcement.
    output_candidate_path : Path
        Target path for `output/candidate_pairs.tsv`.
    output_matching_path : Path
        Target path for `output/matching_results.tsv`.
    source_id_col : str
        Column name for source entity IDs.
    target_id_col : str
        Column name for target entity IDs.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (clean_candidate_pairs, clean_matching_results).
    """
    output_candidate_path.parent.mkdir(parents=True, exist_ok=True)
    output_matching_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Prepare candidate_pairs.tsv
    export_candidates = pd.DataFrame({
        "source_id": scored_candidates_df[source_id_col].astype(str),
        "target_id": scored_candidates_df[target_id_col].astype(str),
    }).drop_duplicates(subset=["source_id", "target_id"]).reset_index(drop=True)

    # 2. Prepare matching_results.tsv
    export_matches = pd.DataFrame({
        "source_id": final_matches_df[source_id_col].astype(str),
        "target_id": final_matches_df[target_id_col].astype(str),
    }).drop_duplicates(subset=["source_id", "target_id"]).reset_index(drop=True)

    # Invariant Verification: All matching results must exist within candidate pairs
    cand_pair_set = set(zip(export_candidates["source_id"], export_candidates["target_id"]))
    match_pair_set = set(zip(export_matches["source_id"], export_matches["target_id"]))
    unaccounted_matches = match_pair_set - cand_pair_set
    if unaccounted_matches:
        raise ValueError(
            f"Candidate drift invariant violated! Found {len(unaccounted_matches)} "
            f"matches in matching_results.tsv that were not present in candidate_pairs.tsv."
        )

    # Export to standard tab-delimited format
    export_candidates.to_csv(
        output_candidate_path,
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    export_matches.to_csv(
        output_matching_path,
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    return export_candidates, export_matches
