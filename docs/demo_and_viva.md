# Live demo and viva guide

## Five-minute demonstration

1. Start `python app.py` and open http://127.0.0.1:8000. Explain that 1 means bankruptcy within three years.
2. Show the held-out results and confusion matrix. State 84 detected, 15 missed, and 161 false alarms. Explain why the majority baseline's 95.25% accuracy is insufficient.
3. In the case selector, run “Detected bankruptcy” and “Correct non-bankruptcy.” Then show “Missed bankruptcy” or “False alarm” so the demonstration includes a real error.
4. Download the sample CSV, upload it, and download predictions. Explain that all 64 columns are required but individual missing values are accepted.
5. Show `reports/model_comparison.csv`, `reports/metrics.json`, and `reports/split_manifest.csv`. Explain that the model and threshold were chosen without the test labels.
6. Optional terminal demo: `python run.py predict --input examples/sample_companies.csv --output predictions.csv`.

Keep the saved model for the review. A complete rerun is `python run.py train`; it need not be run during a short presentation.

## Likely questions

**Why this dataset?** It is an attributed standard bankruptcy dataset with financial indicators and a defined forecasting horizon, and it matches the supplied reference project.

**Why average precision for model selection?** Only about 4.7% of statements are positive. Average precision summarizes how effectively a model ranks positive cases across recall levels. Accuracy can be high for an all-negative classifier.

**Why three folds?** It provides repeated development evaluation while keeping a one-week project's runtime modest. More folds or repeated CV would strengthen stability analysis but were not used here.

**Why F2 for the threshold?** F2 emphasizes recall more than F1. We use it as an explicit educational screening objective. No actual financial costs were estimated.

**Why is the threshold below 0.5?** It was selected on development out-of-fold predictions. It detects more bankruptcies and also creates more false alarms. We report both consequences.

**Why does threshold 0.5 have better test F2?** A development optimum need not be the test optimum. Selecting a new threshold from the test set would bias the final evaluation, so we retain the original choice.

**How is leakage controlled?** Exact duplicates are removed before splitting. Test rows are reserved before model selection. Pipeline preprocessing is fitted inside each CV training fold. Threshold selection uses development OOF predictions. Company-level leakage cannot be fully ruled out without IDs.

**What does histogram boosting do?** It learns a sequence of trees that improve the loss, using binned numeric feature values to speed up split evaluation. This implementation handles missing values directly.

**Are these calibrated bankruptcy probabilities?** No. We call them risk scores, especially because class weighting changes the fitting objective. Calibration would require an additional suitable validation design.

**What is the difference between AP and ROC AUC?** AP summarizes precision and recall while ROC AUC measures ranking over true-positive and false-positive rates. Both are reported, but AP is the selection metric because the positive class is rare.

**What is original here?** The modular implementation, duplicate audit, six-model experiment, OOF threshold selection, stored split/prediction evidence, uncertainty estimates, local app, tests, and documentation were built for this submission. The source dataset and reference report are credited.

**What would you improve next?** Obtain company IDs and dates for group/temporal validation; evaluate threshold stability on repeated splits; calibrate scores on independent data; and choose an operating threshold from actual review costs.
