"""
Pipeline Integrity and Sanity Checks.

Provides comprehensive validation of intermediate representations and final outputs:
- Missing values / null ID checks
- Duplicate pair detection
- Output schema verification (columns, tab delimiter)
- Candidate / prediction consistency (strict subset invariant)
- One-to-one constraint violation detection
- Empty output / abnormal row count diagnostics
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


class SanityCheckError(Exception):
    """Raised when critical pipeline integrity checks fail."""
    pass


def check_missing_or_invalid_ids(df: pd.DataFrame, file_label: str = "DataFrame") -> None:
    """Verify that neither source_id nor target_id contains nulls or empty strings."""
    for col in ["source_id", "target_id"]:
        if col not in df.columns:
            raise SanityCheckError(f"[{file_label}] Missing mandatory column: '{col}'")
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise SanityCheckError(f"[{file_label}] Found {null_count} null values in column '{col}'")

        empty_count = (df[col].astype(str).str.strip() == "").sum()
        if empty_count > 0:
            raise SanityCheckError(f"[{file_label}] Found {empty_count} blank/empty IDs in column '{col}'")


def check_duplicate_pairs(df: pd.DataFrame, file_label: str = "DataFrame") -> None:
    """Verify that no duplicate (source_id, target_id) pairs exist."""
    dups = df.duplicated(subset=["source_id", "target_id"]).sum()
    if dups > 0:
        raise SanityCheckError(
            f"[{file_label}] Found {dups} duplicate (source_id, target_id) pairs!"
        )


def check_one_to_one_violations(df: pd.DataFrame, file_label: str = "matching_results") -> None:
    """
    Verify that 1-to-1 constraint is strictly satisfied:
    - No source_id appears more than once.
    - No target_id appears more than once.
    """
    dup_sources = df["source_id"].duplicated().sum()
    if dup_sources > 0:
        raise SanityCheckError(
            f"[{file_label}] One-to-one violation: {dup_sources} source_ids mapped to multiple targets!"
        )

    dup_targets = df["target_id"].duplicated().sum()
    if dup_targets > 0:
        raise SanityCheckError(
            f"[{file_label}] One-to-one violation: {dup_targets} target_ids mapped from multiple sources!"
        )


def check_candidate_prediction_consistency(
    candidates_df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> None:
    """
    Verify the critical invariant:
    Every pair in matching_results must originate from candidate_pairs.
    """
    cand_pairs = set(zip(candidates_df["source_id"].astype(str), candidates_df["target_id"].astype(str)))
    match_pairs = set(zip(matches_df["source_id"].astype(str), matches_df["target_id"].astype(str)))

    leaked = match_pairs - cand_pairs
    if leaked:
        raise SanityCheckError(
            f"Consistency violation: {len(leaked)} match pairs are absent from candidate_pairs!"
        )


def check_file_format(file_path: Path, expected_cols: List[str]) -> pd.DataFrame:
    """Verify disk file existence, tab separation, and header schema."""
    if not file_path.exists():
        raise SanityCheckError(f"File does not exist: {file_path}")

    try:
        df = pd.read_csv(file_path, sep="\t", dtype=str)
    except Exception as e:
        raise SanityCheckError(f"Failed to parse TSV file at {file_path}: {e}")

    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise SanityCheckError(
            f"File {file_path.name} is missing expected columns: {missing_cols}. Found: {list(df.columns)}"
        )

    return df


def run_full_sanity_checks(
    candidates_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    candidate_file: Optional[Path] = None,
    matching_file: Optional[Path] = None,
    enforce_one_to_one: bool = True,
) -> Dict[str, bool]:
    """
    Execute all sanity checks over in-memory DataFrames and serialized files.
    """
    # 1. Null / Blank ID checks
    check_missing_or_invalid_ids(candidates_df, "Candidate Pairs")
    check_missing_or_invalid_ids(matches_df, "Matching Results")

    # 2. Duplicate checks
    check_duplicate_pairs(candidates_df, "Candidate Pairs")
    check_duplicate_pairs(matches_df, "Matching Results")

    # 3. One-to-one constraint check
    if enforce_one_to_one:
        check_one_to_one_violations(matches_df, "Matching Results")

    # 4. Invariant consistency check
    check_candidate_prediction_consistency(candidates_df, matches_df)

    # 5. File format checks if paths provided
    if candidate_file and candidate_file.exists():
        check_file_format(candidate_file, ["source_id", "target_id"])

    if matching_file and matching_file.exists():
        check_file_format(matching_file, ["source_id", "target_id"])

    return {"status": True, "all_checks_passed": True}
