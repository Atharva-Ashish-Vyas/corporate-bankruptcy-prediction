## Slide 1

Introduce the problem as a three-year binary prediction task. Explain that the presentation reports a fresh, independently implemented experiment. Team: Atharva Ashish Vyas, PES1UG24CS093, and Atharva Rakesh Bhidre, PES1UG24CS095.

## Slide 2

Source: Tomczak (2016), UCI Polish Companies Bankruptcy, https://doi.org/10.24432/C5F600. The third-year ARFF contains 64 financial predictors and a target for bankruptcy within three years. The raw file has 10,503 rows and 495 positives. Counts displayed here are after deterministic exact duplicate removal, documented in reports/data_audit.json.

## Slide 3

Source: reports/data_audit.json and the raw UCI file. 87 duplicates include two positive rows. The resulting set has 10,416 rows and 493 positives. X37 is missing in roughly 45% of the raw data. No conflicting identical feature vectors were found. Missing values are handled inside training pipelines or directly by the histogram model. Company identities are unavailable, so exact deduplication is only a partial leakage control.

## Slide 4

Source: src/bankruptcy/train.py, reports/experiment_config.json and reports/split_manifest.csv. A stratified 80/20 split uses seed 42. Development has 8,332 rows and 394 positives, test has 2,084 rows and 99 positives. All six models use the same 3 folds. Selection uses mean fold average precision. The selected modelâ€™s OOF scores set the F2 threshold. The model is refitted on all development rows and evaluated on the fixed test set. No test-driven changes are made.

## Slide 5

Source: src/bankruptcy/modeling.py. One configuration per algorithm is declared before test evaluation. Scaled pipelines use median imputation plus missing indicators, arcsinh transformation and standard scaling. Forest pipelines use median imputation plus indicators. Histogram boosting accepts missing values directly. Balanced class weights are used where specified. This is a controlled comparison, not an exhaustive tuning study.

## Slide 6

Source: reports/model_comparison.csv. Mean AP across the same 3 development folds: histogram boosting 0.619, random forest 0.434, RBF SVM 0.422, extra trees 0.383, logistic regression 0.297, Gaussian NB 0.087. Histogram boosting fold SD is 0.037. CV estimates are used for selection and should not be presented as an independent test result.

## Slide 7

Source: reports/threshold_curve.csv and reports/metrics.json. OOF means out-of-fold predictions: each development row is scored by a model not fitted on that row. The threshold maximizes F2 on the selected modelâ€™s OOF scores. F2 = 5PR/(4P+R). The chosen threshold is 0.1979516978. This is a declared academic screening objective, not a monetary cost model. Because the same CV is used for selection, development scores can be optimistic; test results remain separate.

## Slide 8

Source: reports/metrics.json and reports/test_predictions.csv. Independent test counts are TN 1824, FP 161, FN 15, TP 84. Recall 84.85%, precision 34.29%, AP 0.7495, ROC AUC 0.9677. The all-negative baseline has 95.25% accuracy and zero recall. Approximate stratified bootstrap 95% recall interval: 77.8â€“91.9%; precision interval: 31.1â€“38.7%. These intervals condition on this fitted model and split.

## Slide 9

Source: reports/pr_curve.csv, reports/roc_curve.csv, and reports/metrics.json. Curves are drawn from the fixed test predictions. The native charts use sampled curve points for display; exact curve values are committed as CSVs. Average precision is 0.750 and ROC AUC 0.968. The test prevalence baseline for precision is 0.0475. AP and ROC AUC evaluate ranking and do not require changing the decision threshold.

## Slide 10

Source: reports/metrics.json. Comparing two prespecified rules on the same fixed test scores: the development-selected threshold catches 84 positives and flags 161 negatives; the default threshold 0.5 catches 68 positives and flags 36 negatives. The default threshold has test F2 0.680 versus 0.655. Do not claim the chosen threshold is optimal on the test set. It remains unchanged to respect the protocol.

## Slide 11

Source: reports/feature_importance.csv and UCI variable definitions. Permutation importance is the decrease in average precision after shuffling one feature, using 3 test-set repeats. Top variables are X27 operating profit / financial expenses, X34 operating expenses / total liabilities, X46 current assets minus inventory / current liabilities, X58 total costs / total sales, and X21 sales growth ratio. These are descriptive model dependencies, not causal claims. Correlated financial ratios complicate interpretation. No test-driven model changes followed this analysis.

## Slide 12

Source: the local Flask application in app.py and templates/index.html. Screenshot captured from the running local app. Start python app.py, open http://127.0.0.1:5000, run a detected-bankruptcy case and an error case, then upload examples/sample_companies.csv. The saved model and checksum are included, so retraining is not required for the demo. Backend tests cover valid and invalid CSV inputs.

## Slide 13

Source: the experiment and docs/model_card.md. The chosen model detects 84 of 99 bankruptcies but yields 161 false alarms. The dataset is historical and company identifiers are missing; random holdout is not verified group or temporal validation. Scores are uncalibrated. Future work should prioritize company/date information, threshold stability, and independent calibration. Do not promise performance on present-day firms.

## Slide 14

Sources: [1] Tomczak, S. (2016). Polish Companies Bankruptcy. UCI. https://doi.org/10.24432/C5F600, CC BY 4.0. [2] Scikit-learn documentation: https://scikit-learn.org/stable/modules/model_evaluation.html and https://scikit-learn.org/stable/modules/cross_validation.html. [3] Supplied Corporate Bankruptcy Prediction reference report by Neha Abajirao Patil and https://github.com/codeatri/Bankruptcy_Prediction. Background only; no results or source code copied into this project. Software versions and all row-level evidence are recorded in the repository.
