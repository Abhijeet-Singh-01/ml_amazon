# Business Entity Resolution — Solution Methodology & Documentation

## 1. Problem Understanding
Business Entity Resolution (BER) is the task of identifying and linking records across disparate datasets that refer to the same real-world business organization. In commercial and supply-chain contexts, entity records are often noisy, featuring inconsistencies in organizational naming conventions (e.g., abbreviations, legal structures), address formatting, postal codes, and multilingual variations.

The primary objective is to reliably align source entities against reference target entities with high precision and recall while managing quadratic comparison complexity ($O(N \times M)$) over large-scale catalogs.

## 2. Objective
- **Core Goal**: Given a set of source business records and a reference target catalog, identify the true matching pairs `(source_id, target_id)`.
- **Constraint**: Strict 1-to-1 matching when an entity in the source corresponds to at most one target record.
- **Optimization Target**: Macro F-beta score across entity evaluations and geographic regions, penalizing false positives and false negatives appropriately according to hackathon evaluation guidelines.

## 3. Data Understanding
- **Input Data Format**: Tabular entity records containing company names, street addresses, cities, postal codes, and country identifiers.
- **Data Modalities & Challenges**:
  - Variations in legal suffixes (`Ltd`, `Limited`, `LLC`, `GmbH`, `S.A.`, etc.).
  - Punctuation, casing, and Unicode/diacritic inconsistencies.
  - Abbreviated and missing address components (e.g., suite numbers, directional prefixes).
  - Geographic disparity across domestic and cross-border entities.

## 4. Data Preprocessing
- Text normalization: Canonical Unicode conversion (NFKD/NFC), lowercasing, whitespace trimming, and punctuation sanitization.
- Null value imputation: Standardizing missing values, placeholder tokens, and non-informative strings (`N/A`, `unknown`, `null`).
- Structural standardization: Uniform schema mapping across source and target entities.

## 5. Normalization
- **Text Cleaning**: Strip redundant punctuation, normalize special characters (`&` vs `and`), remove spurious whitespace.
- **Legal Suffixes**: Modular parsing and stripping of corporate designations across jurisdictions (US, UK, Germany, France, etc.) to extract base trading names while retaining legal forms as separate matching signals.
- **Address Standardization**: Standardize street descriptors (`St`, `Street`, `Ave`, `Avenue`, `Rd`, `Road`, `Blvd`), parse postal codes, and normalize administrative divisions.

## 6. Blocking Strategy
To avoid exhaustive $O(N \times M)$ pair comparisons, multi-index blocking is employed:
- **TF-IDF Blocking**: Substring and character n-gram TF-IDF vectorization to identify top-$k$ nearest neighbors above a similarity threshold.
- **Address Blocking**: Blocking on standardized geographic attributes (e.g., postal code + country or city + country).
- **Embedding Blocking**: Dense vector representations capturing semantic/syntactic proximity.

## 7. Candidate Generation
- **Union of Candidate Blocks**: Candidate pairs from TF-IDF, address, and embedding blockers are merged via set union.
- **Candidate Set Invariant**: The merged candidate set is preserved without loss and forms the exact dataset scored by the ranking model and emitted to `candidate_pairs.tsv`.
- **Pair Completeness vs. Reduction Ratio**: Blocking recall is verified to ensure true matches are captured while reducing pair cardinality by orders of magnitude.

## 8. Feature Engineering
The feature matrix $X$ is composed of complementary feature families:
- **Name Similarities**:
  - Exact match, lowercase match, cleaned base name match.
  - Levenshtein ratio, Jaro-Winkler similarity, Token Sort ratio, Token Set ratio.
  - Longest common prefix (LCP) and longest common substring (LCS) ratios.
  - Legal suffix compatibility indicator.
- **Address & Location Similarities**:
  - Address token similarity, edit distance on street numbers and street names.
  - Postal code exact match and prefix distance.
  - City token match and phonetic similarity.
- **Country Features**:
  - Exact country match indicator, missing country flags, cross-border compatibility.
- **Ranking & Contextual Features**:
  - Candidate rank within source entity's candidate pool based on preliminary blocking similarity.
  - Score margin between top-1 and top-2 candidate per source entity.

## 9. Model Architecture
- **Classifier / Ranker**: Gradient Boosted Decision Trees (LightGBM) optimized for binary classification or pairwise ranking.
- **Tree Hyperparameters**: Tuned tree depth, number of leaves, min child weight, and sub-sampling to mitigate overfitting on sparse candidate distributions.
- **Class Imbalance Mitigation**: Utilizing scale-pos-weight / focal objective to address candidate imbalance where non-matches vastly outnumber true matches.

## 10. Training Methodology
- **Training Pair Construction**: Positive pairs from ground truth matches combined with hard negative candidates generated through the blocking layer.
- **Data Splitting**: Entity-grouped cross-validation or disjoint source-target entity splitting to prevent data leakage between folds.
- **Early Stopping**: Monitored on validation loss/macro metric to prevent over-fitting.

## 11. Validation Methodology
- Grouped $K$-fold cross-validation grouped by source/parent organization or country.
- Validation checks assessing out-of-fold generalization on unseen entity names.

## 12. Threshold Selection
- Post-training threshold calibration over the validation prediction distribution.
- Grid search over decision thresholds $\tau \in [0.01, 0.99]$ to find the global threshold maximizing Macro F-beta score.
- Validation curve analysis to verify stability and avoid over-tuning to sample anomalies.

## 13. One-to-One Matching
- Enforcing global bipartite matching or greedy conflict resolution on predicted candidate pairs above threshold.
- Ensures each source entity is assigned to at most one target entity and conflicts are resolved in favor of the highest confidence match.

## 14. Evaluation Metrics
- **Macro F-beta**: Calculated globally and macro-averaged according to challenge specifications.
- **Precision and Recall**: Precision prioritizing reduction of erroneous entity merges; recall ensuring coverage of true matches.

## 15. Cross-Country Evaluation
- Segmented performance metrics computed per country.
- Diagnostics verifying that models do not disproportionately degrade in low-resource or non-English address formats.

## 16. Error Analysis
- Diagnostics on false positive and false negative predictions:
  - Common legal suffix misalignments.
  - Agglomeration errors in multi-branch corporations.
  - Transliteration and spelling noise in multilingual records.

## 17. Reproducibility
- Centralized configuration via `src/config.py`.
- Deterministic random seeds across numpy, scikit-learn, and LightGBM.
- Containerized or pinned environment via `requirements.txt`.
- Single-command pipeline execution via `src/run_pipeline.py`.

## 18. Output Generation
- Dual output files generated in a single atomic pass from the exact scored candidates:
  - `output/matching_results.tsv`: Final resolved pairs satisfying threshold and 1-to-1 constraints.
  - `output/candidate_pairs.tsv`: Complete set of candidates evaluated by the model.
- Rigorous sanity checks verifying absence of null values, duplicates, ID mismatches, and schema violations.

## 19. Limitations
- Coverage bounded by blocking recall: True matches omitted during blocking cannot be recovered by downstream models.
- Sensitivity of address parsing on unformatted free-text address strings.
- Computational throughput constraints when scaling to tens of millions of records.

## 20. Future Improvements
- Pretrained multilingual Transformer representations (e.g., multilingual RoBERTa, DeBERTa) fine-tuned with contrastive triplet loss.
- Graph-based entity clustering and transitive closure resolution for multi-catalog consolidation.
- Active learning loops for high-uncertainty candidate verification.
