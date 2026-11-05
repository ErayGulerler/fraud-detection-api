# Rare-event classification under distribution shift

Protocol fixed before the first real-data experiment on 5 November 2026.

## Question

How do imbalance handling and held-out probability calibration affect ranking,
decision quality and probability estimates on later transactions and under
controlled changes in fraud prevalence?

## Data and split

Use the public ULB/Worldline credit-card fraud dataset: 284,807 transactions,
492 frauds, two days of anonymized European card transactions. Download source:
https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
This mirror is linked by the official TensorFlow tutorial:
https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
Original dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Do not redistribute the CSV in this repository; consult the original dataset terms.

Validate all 31 columns, reject missing/non-finite values, then remove exact
duplicate rows using all columns. This fixed deduplication rule is not learned
from outcomes. Preserve original row IDs for hashed split audits. Split in
chronological order into approximately 55% training, 15% calibration, 15%
validation, 15% untouched test. Keep equal timestamps in the same partition.
Fail if any partition lacks both classes or has fewer than six frauds.

Use V1-V28 and log1p(Amount). Exclude Time and the legacy full-hour transaction
count: the latter includes future events. Fit StandardScaler on training data
only. V1-V28 are supplied PCA features; their upstream fitting provenance is
unavailable, so this cannot establish absence of leakage in the source dataset.

## Fixed comparisons and selection

Logistic Regression (C=1, max_iter=2000, tol=1e-6) and Random Forest
(100 trees, max_depth=12, min_samples_leaf=2, max_features=sqrt), each with:
no imbalance treatment; balanced class weights; or SMOTE at minority/majority
ratio 0.1 with five neighbours. SMOTE operates on scaled training data only.
No hyperparameter search. Seeds 42,43,44 repeat training randomness on the SAME
chronological split. They do not estimate variability across data splits.

For each fitted model, compare raw probabilities and a sigmoid of clipped
log-odds (epsilon=1e-6), fitted by LogisticRegression(C=1000, tol=1e-8) using
only the calibration partition at its natural prevalence. This is a specified
regularized sigmoid calibration, not cross-validated calibration.

Choose each operating threshold by maximum validation F1, ties resolved with
the highest threshold. F1 is an exploratory criterion, not a financial cost
function. Select the model/imbalance variant by mean raw validation average
precision across the three seeds; select raw versus calibrated probabilities
for that variant by mean validation log loss. Break selection ties lexically.
Write the frozen selection to disk BEFORE predicting on the test set.
Report all predeclared comparisons, not just the winner. Do not retune after
seeing test results. Seed42 is the predeclared representative, never chosen by
test performance. This is a prospective evaluation within this extension,
not an external preregistration or an independent replication of the source data.

## Test conditions and metrics

1. Natural chronological holdout. This measures later-period generalization;
   temporal splitting alone does not prove a distribution change exists.
2. Controlled label-prior shifts to fraud rates 0.05%, 0.5%, 1% using exact class
   reweighting of the SAME holdout: weight positives by target/source prevalence
   and negatives by (1-target)/(1-source). This preserves empirical P(X|Y).
   It is sensitivity analysis, not new observations, concept drift, or an
   identified real-world target distribution. Keep models, calibration and
   thresholds fixed. No target-prior probability correction is applied.

Report average precision (sklearn AP, not trapezoidal PR-AUC), ROC-AUC,
precision, recall, F1, Brier score, log loss, and 10 equal-width-bin ECE.
Include reliability-bin counts. Brier/log loss combine discrimination and
calibration; low global Brier/ECE can conceal poor minority-class performance.
Weighted confusion counts in prior-shift scenarios are expected mass rather
than additional transactions. ROC-AUC and recall should be invariant to pure
class reweighting; precision, AP and probability losses need not be.

For the selected variant/mapping at seed42, compute 200 class-stratified
bootstrap resamples and percentile 95% intervals for AP, precision, recall and
Brier on the natural holdout. These are conditional on the fitted model,
selected threshold and fixed class counts. They exclude model selection,
training uncertainty, prevalence uncertainty, and temporal/card dependence.
No significance or universal superiority claim from these intervals.

## Reproducibility and limitations

Record dataset, source and protocol SHA256 hashes, versions, split sizes,
fraud counts, timestamp ranges, row-index hashes, seeds and Git state. Keep
aggregate results, source and a pinned dependency file in Git. Tests use only
synthetic fixtures; substantive results must use the public dataset.

Only two days and hundreds of frauds are available. No card IDs, raw feature
semantics, delayed labels, deployment costs, or external population are known.
Calibration may overfit a small number of frauds. No real bank deployment or
general distribution-shift robustness claim is justified. The Flask demo and
legacy train.py remain separate and retain known evaluation/serving issues;
the research results must not be described as API performance.
