"""
Embedding-Based Candidate Pair Blocker.

Generates candidate pairs using dense vector representations (LSA / Dense Subspace
Projection over character & subword n-grams) and nearest-neighbor cosine similarity.
Captures semantic and latent lexical similarities beyond exact term matching.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def generate_embedding_candidates(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_text_col: str = "name_clean",
    target_text_col: str = "name_clean",
    source_id_col: str = "id",
    target_id_col: str = "id",
    embedding_dim: int = 64,
    top_k: int = 15,
    min_similarity: float = 0.40,
    random_state: int = 42,
    batch_size: int = 2000,
) -> pd.DataFrame:
    """
    Generate candidate pairs using dense LSA embeddings and cosine similarity.

    Parameters
    ----------
    source_df : pd.DataFrame
        Source entities table.
    target_df : pd.DataFrame
        Target reference entities table.
    embedding_dim : int
        Dimensionality of dense projected embeddings.
    top_k : int
        Maximum candidates to retrieve per source record.
    min_similarity : float
        Minimum cosine similarity threshold.
    random_state : int
        Seed for reproducible randomized SVD.

    Returns
    -------
    pd.DataFrame
        Candidate pairs DataFrame with columns:
        ['source_id', 'target_id', 'embedding_similarity']
    """
    if source_df.empty or target_df.empty:
        return pd.DataFrame(columns=["source_id", "target_id", "embedding_similarity"])

    corpus = pd.concat([source_df[source_text_col], target_df[target_text_col]]).fillna("")

    # Build character n-gram TF-IDF representations
    vectorizer = TfidfVectorizer(
        ngram_range=(2, 4),
        analyzer="char_wb",
        min_df=1,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Project to dense semantic space via TruncatedSVD
    n_components = min(embedding_dim, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    if n_components < 2:
        return pd.DataFrame(columns=["source_id", "target_id", "embedding_similarity"])

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    svd.fit(tfidf_matrix)

    emb_source = svd.transform(vectorizer.transform(source_df[source_text_col].fillna("")))
    emb_target = svd.transform(vectorizer.transform(target_df[target_text_col].fillna("")))

    # L2 normalize embeddings for fast cosine similarity dot product
    norm_src = np.linalg.norm(emb_source, axis=1, keepdims=True)
    norm_tgt = np.linalg.norm(emb_target, axis=1, keepdims=True)
    norm_src[norm_src == 0] = 1.0
    norm_tgt[norm_tgt == 0] = 1.0
    emb_source = emb_source / norm_src
    emb_target = emb_target / norm_tgt

    source_ids = source_df[source_id_col].values
    target_ids = target_df[target_id_col].values

    n_sources = emb_source.shape[0]
    candidate_records = []

    for start_idx in range(0, n_sources, batch_size):
        end_idx = min(start_idx + batch_size, n_sources)
        batch_source = emb_source[start_idx:end_idx]

        sim_matrix = np.dot(batch_source, emb_target.T)

        for local_i in range(sim_matrix.shape[0]):
            global_src_idx = start_idx + local_i
            src_id = str(source_ids[global_src_idx])
            sim_scores = sim_matrix[local_i]

            valid_indices = np.where(sim_scores >= min_similarity)[0]
            if len(valid_indices) == 0:
                continue

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
                    "embedding_similarity": float(score),
                })

    if not candidate_records:
        return pd.DataFrame(columns=["source_id", "target_id", "embedding_similarity"])

    return pd.DataFrame(candidate_records)
