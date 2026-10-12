"""Shared prediction path for the command line and local web application."""
from pathlib import Path
import json
import joblib
import pandas as pd
from .data import validate_features, sha256

def load_bundle(root):
    root = Path(root)
    path = root / "models" / "bankruptcy_model.joblib"
    manifest = json.loads((root / "models" / "manifest.json").read_text())
    if sha256(path) != manifest["sha256"]:
        raise ValueError("Model checksum does not match the project manifest. Re-run training.")
    # Only load the model shipped with this repository or one you trained yourself.
    return joblib.load(path)

def predict_frame(bundle, frame):
    validated = validate_features(frame)
    scores = bundle["model"].predict_proba(validated)[:, 1]
    return pd.DataFrame({"row": range(1, len(frame) + 1), "risk_score": scores,
                         "threshold": bundle["threshold"],
                         "predicted_bankrupt": (scores >= bundle["threshold"]).astype(int),
                         "missing_features": validated.isna().sum(axis=1).to_numpy()})
