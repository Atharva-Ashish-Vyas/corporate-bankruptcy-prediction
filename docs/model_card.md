# Model card

## Intended use

Academic comparison of binary classifiers on the UCI Polish company statements. The local demo illustrates scoring and evaluation. It is not validated for loan approval, investment, or real company risk assessment.

## Input and output

Exactly 64 numeric fields, X1-X64, as defined by UCI. Empty cells represent missing values. The web app rejects missing columns, unknown columns, nonnumeric strings, infinite values, and completely empty rows. A risk score and a thresholded class are returned. Class 1 means bankruptcy during the three-year horizon.

## Training

10,416 unique statements after removing 87 exact duplicates. Development set: 8,332 rows with 394 positive labels. Test set: 2,084 rows with 99 positive labels. Stratified split and CV use seed 42. Each candidate uses one prespecified configuration; this is a controlled comparison, not an exhaustive hyperparameter search. Selection uses mean three-fold development average precision.

Selected estimator: histogram gradient boosting, 250 iterations, learning rate 0.07, up to 15 leaves, L2 regularization 1, balanced class weights, native missing-value handling, no early stopping. The decision threshold is 0.19795170, selected by maximum development OOF F2.

## Evaluation

| Measure | Estimate | Conditional 95% bootstrap interval |
|---|---:|---:|
| Recall | 0.848 | 0.778-0.919 |
| Precision | 0.343 | 0.311-0.387 |
| Average precision | 0.750 | 0.683-0.820 |
| ROC AUC | 0.968 | 0.956-0.980 |

Intervals use 500 stratified bootstrap resamples of the fixed test predictions. They are approximate, conditional on the model and split, and do not address repeated company observations, temporal drift, or training variability.

## Threshold sensitivity

| Rule | Recall | Precision | F2 | False positives | False negatives |
|---|---:|---:|---:|---:|---:|
| Development-selected threshold 0.1980 | 84.8% | 34.3% | 0.655 | 161 | 15 |
| Prespecified default threshold 0.5 | 68.7% | 65.4% | 0.680 | 36 | 31 |

The lower threshold catches 16 more bankruptcies and flags 125 additional solvent statements. The default threshold actually has higher test F2 in this split. We do not switch thresholds after seeing that result; it demonstrates that a development optimum may not transfer exactly to test data. F2 is a project evaluation choice, not a measured monetary cost ratio.

## Interpretability and limitations

Permutation importance shuffles one test feature at a time and measures the decrease in average precision, with three repeats. It describes the selected model's reliance on these variables. It is not causal, and correlated ratios can make individual importances misleading. No model or threshold changes followed this analysis.

The historical sample is geographically restricted. Company-level separation cannot be verified because IDs are absent. A random split is not proof of forecasting future years. Class weighting and the chosen threshold do not produce calibrated probabilities. Missingness may reflect collection processes that change over time.

## Reproduction

Run `python run.py train`. Full configuration, raw data hash, row assignments, out-of-fold predictions, test predictions, versions, and metrics are committed under `reports/`. The model hash is in `models/manifest.json`. Load only model files you trained yourself or trust, because joblib uses pickle serialization.
