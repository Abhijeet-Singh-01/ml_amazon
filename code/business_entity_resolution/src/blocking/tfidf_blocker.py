"""
TF-IDF Candidate Pair Blocker.

Generates candidate entity pairs using character/word n-gram TF-IDF vectorization
and cosine similarity retrieval. Fast sparse matrix multiplication retrieves
top-k candidates per source entity above a configurable similarity threshold.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


def generate_tfidf_candidates(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_text_col: str = "name_clean",
    target_text_col: str = "name_clean",
    source_id_col: str = "id",
    target_id_col: str = "id",
    ngram_range: Tuple[int, int] = (2, 4),
    analyzer: str = "char_wb",
    min_df: int = 1,
    top_k: int = 15,
    min_similarity: float = 0.20,
    batch_size: int = 2000,
) -> pd.DataFrame:
    """
    Generate candidate pairs between source and target entities via TF-IDF cosine similarity.

    Parameters
    ----------
    source_df : pd.DataFrame
        Source entities table.
    target_df : pd.DataFrame
        Target reference entities table.
    source_text_col : str
        Column containing normalized text for source entities.
    target_text_col : str
        Column containing normalized text for target entities.
    top_k : int
        Maximum candidates to retrieve per source record.
    min_similarity : float
        Minimum cosine similarity threshold to qualify as a candidate.
    batch_size : int
        Chunk size for source matrix multiplication to prevent memory spikes.

    Returns
    -------
    pd.DataFrame
        Candidate pairs DataFrame with columns:
        ['source_id', 'target_id', 'tfidf_similarity']
    """
    if source_df.empty or target_df.empty:
        return pd.DataFrame(columns=["source_id", "target_id", "tfidf_similarity"])

    # Fit vectorizer on target texts (and source to build vocabulary)
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        analyzer=analyzer,
        min_df=min_df,
        sublinear_tf=True,
    )

    corpus = pd.concat([source_df[source_text_col], target_df[target_text_col]])
    vectorizer.fit(corpus.fillna(""))

    X_source = vectorizer.transform(source_df[source_text_col].fillna(""))
    X_target = vectorizer.transform(target_df[target_text_col].fillna(""))

    source_ids = source_df[source_id_col].values
    target_ids = target_df[target_id_col].values

    n_sources = X_source.shape[0]
    candidate_records = []

    # Process in batches to control peak memory consumption
    for start_idx in range(0, n_sources, batch_size):
        end_idx = min(start_idx + batch_size, n_sources)
        batch_source = X_source[start_idx:end_idx]

        # Cosine similarity between batch and all targets: shape (batch_len, n_targets)
        sim_matrix = batch_source.dot(X_target.T).toarray()

        for local_i in range(sim_matrix.shape[0]):
            global_src_idx = start_idx + local_i
            src_id = str(source_ids[global_src_idx])
            sim_scores = sim_matrix[local_i]

            # Indices with score >= min_similarity
            valid_indices = np.where(sim_scores >= min_similarity)[0]
            if len(valid_indices) == 0:
                continue

            # Sort valid indices by score descending and keep top_k
            valid_scores = sim_scores[valid_indices]
            if len(valid_scores) > top_k:
                top_part_idx = np.argpartition(-valid_scores, top_k)[:top_k]
                valid_indices = valid_indices[top_part_idx]
                valid_scores = valid_scores[top_part_idx]

            sorted_order = np.argsort(-valid_scores)
            selected_indices = valid_indices[sorted_order]
            selected_scores = valid_scores[sorted_order]

            for tgt_idx, score in zip(selected_indices, selected_scores):
                candidate_records.append({
                    "source_id": src_id,
                    "target_id": str(target_ids[tgt_idx]),
                    "tfidf_similarity": float(score),
                })

    if not candidate_records:
        return pd.DataFrame(columns=["source_id", "target_id", "tfidf_similarity"])

    return pd.DataFrame(candidate_records)
