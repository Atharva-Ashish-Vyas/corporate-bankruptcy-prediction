"""Fixed candidate models and evaluation rules, declared before test evaluation."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, fbeta_score,
                             accuracy_score, balanced_accuracy_score, confusion_matrix)

def candidates(seed=42):
    def scaled(model):
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             FunctionTransformer(np.arcsinh, feature_names_out="one-to-one"),
                             StandardScaler(), model)
    def trees(model):
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), model)
    return {
        "Logistic regression": scaled(LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=seed)),
        "Gaussian naive Bayes": scaled(GaussianNB(var_smoothing=1e-8)),
        "RBF support vector machine": scaled(SVC(C=3, class_weight="balanced", probability=True, random_state=seed)),
        "Random forest": trees(RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=2, random_state=seed)),
        "Extra trees": trees(ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", n_jobs=2, random_state=seed)),
        "Histogram gradient boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=.07, max_leaf_nodes=15, l2_regularization=1, class_weight="balanced", early_stopping=False, random_state=seed),
    }

def choose_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f2 = np.divide(5 * precision[:-1] * recall[:-1], 4 * precision[:-1] + recall[:-1],
                   out=np.zeros_like(precision[:-1]), where=(4 * precision[:-1] + recall[:-1]) > 0)
    # Prefer the larger threshold if the score ties exactly.
    idx = np.flatnonzero(np.isclose(f2, f2.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[idx])

def metrics(y, scores, threshold):
    predicted = (np.asarray(scores) >= threshold).astype(int)
    return {
        "threshold": float(threshold), "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "f2": float(fbeta_score(y, predicted, beta=2, zero_division=0)),
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }
