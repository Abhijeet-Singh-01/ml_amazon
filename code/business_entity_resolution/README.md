# Business Entity Resolution Pipeline

A high-performance, reproducible machine learning pipeline designed to perform large-scale entity resolution and record linkage between noisy business entities and reference enterprise registries.

---

## 1. Project Overview
This repository contains the complete end-to-end solution for the Business Entity Resolution ML hackathon challenge. The pipeline scales to large entity databases by employing a multi-strategy blocking architecture, robust feature engineering across lexical and geographic modalities, a gradient-boosted decision tree ranker (LightGBM), data-driven decision threshold calibration, and greedy bipartite one-to-one matching constraints.

## 2. Problem Statement
The challenge requires linking records from an uncurated source entity table to a canonical reference target catalog. Real-world business data suffers from significant noise:
- Corporate designation variations (`LLC`, `Limited`, `GmbH`, `Inc.`, `Corp.`)
- Multilingual diacritics, character encoding artifacts, and casing disparities
- Fragmented and abbreviated street addresses
- Quadratic comparison complexity ($O(N \times M)$) preventing brute-force pair scoring

## 3. Approach
Our approach balances recall and precision through a phased funnel architecture:
1. **Normalization**: Canonical text cleaning, legal suffix extraction/stripping, and street address standardization.
2. **Multi-Strategy Blocking**: Tri-part candidate generation via subword TF-IDF, postal/geographic grouping, and dense semantic subspace projection.
3. **Lossless Candidate Union**: Merging candidate streams while preserving all candidate pairs and candidate origins.
4. **Rich Feature Engineering**: Multi-modal similarity vectors spanning string edit distances, token sets, geographic markers, and candidate group rankings.
5. **LightGBM Classification & Scoring**: Gradient-boosted decision trees trained to estimate match probabilities.
6. **Threshold Optimization**: Calibrating decision thresholds on validation sets to optimize macro F-beta.
7. **One-to-One Matching Constraint**: Global greedy assignment preventing duplicate record linkage.
8. **Invariant Output Assembly & Verification**: Atomic generation of `candidate_pairs.tsv` and `matching_results.tsv` from the exact same scored candidate pool.

## 4. Repository Structure
```text
team_name_submission/
│
├── output/
│   ├── matching_results.tsv             # Final 1-to-1 resolved entity matches
│   └── candidate_pairs.tsv              # Identical candidate pairs scored by model
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── config.py                # Centralized pipeline configuration
│       │
│       │   ├── normalize/               # Text, legal suffix & address normalization
│       │   │   ├── text_clean.py
│       │   │   ├── legal_suffixes.py
│       │   │   ├── address_parse.py
│       │   │   └── run_normalize.py
│       │
│       │   ├── blocking/                # Multi-strategy candidate generation
│       │   │   ├── tfidf_blocker.py
│       │   │   ├── address_blocker.py
│       │   │   ├── embedding_blocker.py
│       │   │   ├── union_blocks.py
│       │   │   └── recall_eval.py
│       │
│       │   ├── features/                # Feature extraction & matrix assembly
│       │   │   ├── name_similarity.py
│       │   │   ├── address_similarity.py
│       │   │   ├── country_features.py
│       │   │   ├── rank_features.py
│       │   │   └── build_feature_matrix.py
│       │
│       │   ├── model/                   # LightGBM training, thresholding & inference
│       │   │   ├── train_lgbm.py
│       │   │   ├── threshold_tuning.py
│       │   │   ├── one_to_one_constraint.py
│       │   │   └── predict.py
│       │
│       │   ├── postprocess/             # Output assembly & integrity verification
│       │   │   ├── assemble_outputs.py
│       │   │   └── sanity_checks.py
│       │
│       │   ├── evaluation/              # Macro F-beta & country-level diagnostics
│       │   │   ├── f_beta_macro.py
│       │   │   └── cross_country_eval.py
│       │
│       │   ├── artifacts/               # Serialized weights, vectorizers, thresholds
│       │   │   └── .gitkeep
│       │
│       │   └── run_pipeline.py          # Master single entry point
│       │
│       ├── README.md                    # Solution guide & operational manual
│       └── requirements.txt             # Pinned project dependencies
│
└── Documentation_template.md            # Hackathon methodology documentation
```

## 5. Environment Setup
- Python version: 3.10+ recommended (compatible with 3.9+)
- Operating system: Linux, macOS, or Windows
- Memory: Minimum 8 GB RAM (16 GB recommended for large catalogs)

Create and activate an isolated virtual environment:
```bash
# On Linux / macOS
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## 6. Installation Instructions
Install the pinned dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 7. Dataset / Input Expectations
The pipeline expects tabular data (`.csv` or `.tsv`) with the following fields:
- **Source Entities (`source_entities.csv`)**:
  - `id`: Unique identifier for the source record
  - `name`: Business entity legal/trading name
  - `address` (optional): Street address
  - `postal_code` (optional): Postal or ZIP code
  - `country` (optional): Two-letter ISO country code
- **Target Entities (`target_entities.csv`)**:
  - `id`: Unique identifier for the target reference record
  - `name`: Canonical business name
  - `address` (optional): Physical street address
  - `postal_code` (optional): Postal or ZIP code
  - `country` (optional): Two-letter ISO country code
- **Ground Truth Matches (`ground_truth.tsv`, optional for training/eval)**:
  - `source_id`: Source record identifier
  - `target_id`: True matching target record identifier

## 8. Configuration
All hyperparameters, file paths, and execution flags are centrally defined in `src/config.py`:
- `paths`: Input/output directories and artifact locations
- `blocking`: Subword n-gram range, top-$k$ candidates, and similarity thresholds
- `features`: Feature toggles for name, address, country, and ranking signals
- `model`: LightGBM boosting parameters, learning rate, tree depth, and random seeds
- `threshold`: Search grid bounds and F-beta objective parameter

## 9. How to Train
To train the LightGBM model on labeled data and calibrate the decision threshold:
```bash
cd code/business_entity_resolution
python src/run_pipeline.py --train \
  --source path/to/source_entities.csv \
  --target path/to/target_entities.csv \
  --ground-truth path/to/ground_truth.tsv
```
This trains the model, saves `artifacts/lgbm_model.joblib`, and records the optimal threshold in `artifacts/optimal_threshold.json`.

## 10. How to Run Inference
To run scoring on new entity datasets using previously trained artifacts:
```bash
cd code/business_entity_resolution
python src/run_pipeline.py \
  --source path/to/source_entities.csv \
  --target path/to/target_entities.csv
```

## 11. How to Run the Complete Pipeline
Execute the full end-to-end pipeline (train, tune, predict, enforce 1-to-1 constraints, verify sanity, and export outputs):
```bash
cd code/business_entity_resolution
python src/run_pipeline.py --train \
  --source path/to/source_entities.csv \
  --target path/to/target_entities.csv \
  --ground-truth path/to/ground_truth.tsv \
  --output-candidates ../../output/candidate_pairs.tsv \
  --output-matching ../../output/matching_results.tsv
```

## 12. Blocking Methodology
To scale beyond full Cartesian products:
- **Subword TF-IDF Blocker**: Generates character 2-to-4-grams within word boundaries. Sparse matrix cosine similarity retrieves the top-$k$ nearest neighbors per source entity.
- **Geographic Blocker**: Groups records sharing identical Postal Codes + Country Codes, or clean address strings.
- **Dense Embedding Blocker**: Applies Truncated SVD over subword space to capture semantic latent similarities.
- **Lossless Union**: Merges candidate sets into a consolidated table without dropping candidates.

## 13. Feature Engineering
The feature matrix $X$ integrates four core feature families:
- **Name Signals**: Normalized Levenshtein ratio, Jaro-Winkler similarity, Token-Sort ratio, Token-Set intersection, Longest Common Prefix (LCP) ratio, base name similarities, and legal suffix match indicator.
- **Address Signals**: Cleaned address string edit distances, token set overlap, postal code exact and prefix match indicators.
- **Country Signals**: Country code agreement, cross-border mismatch flag, and missingness indicators.
- **Ranking Signals**: Candidate rank within the source entity group, score delta from the top-1 candidate, and candidate pool cardinality.

## 14. ML Model
- **Algorithm**: LightGBM Classifier (`gbdt`)
- **Objective**: Binary logloss (`binary`)
- **Imbalance Handling**: Configurable `scale_pos_weight` to account for high negative-to-positive candidate ratios.
- **Regularization**: Early stopping on out-of-fold validation data to prevent overfitting.

## 15. Threshold Selection
Instead of hard-coding an arbitrary 0.50 cutoff:
- The validation probability distribution is searched over a fine-grained grid $\tau \in [0.05, 0.95]$ with step size 0.01.
- The threshold that maximizes the macro F-beta score is automatically identified and saved into `artifacts/optimal_threshold.json`.

## 16. One-to-One Matching
When target catalogs represent distinct organizational entities, multiple source records cannot link to the same target record, nor can one source record resolve to multiple targets:
- Greedy bipartite matching sorts candidate pairs in descending order of predicted probability.
- Mutual exclusive assignment commits the highest-confidence pair and eliminates competing candidates.

## 17. Evaluation Methodology
- **Macro F-beta Metric**: Implemented in `src/evaluation/f_beta_macro.py`, supporting entity-averaged macro metrics and global pair-level calculations.
- **Cross-Country Diagnostics**: Evaluated in `src/evaluation/cross_country_eval.py` to identify performance disparities across distinct national jurisdictions.

## 18. Output Format
Two standard tab-delimited files are emitted:
1. `output/candidate_pairs.tsv`:
   ```text
   source_id	target_id
   src_001	tgt_014
   src_001	tgt_089
   ```
2. `output/matching_results.tsv`:
   ```text
   source_id	target_id
   src_001	tgt_014
   ```

## 19. Reproducibility
- Random seed fixed across NumPy, Scikit-learn, and LightGBM (`random_seed = 42`).
- Pinned dependency requirements in `requirements.txt`.
- Deterministic feature calculation order and tie-breaking in one-to-one constraint solvers.

## 20. Sanity Checks
Automated pre-export checks enforced in `src/postprocess/sanity_checks.py`:
- Zero null or empty source/target identifiers.
- Zero duplicate candidate or matching pairs.
- Invariant verification: Matching pairs are a strict subset of candidate pairs.
- Strict one-to-one constraint compliance (no duplicate `source_id` or `target_id`).
- Correct TSV delimiter, UTF-8 encoding, and header validation.

## 21. Important Assumptions
- Entities in the reference target table represent unique business records.
- Records without addresses or postal codes rely primarily on name similarity and subword blocking.
- Missing country codes are treated as neutral signals rather than hard negative matches.
