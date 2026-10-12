"""Dataset loading, deterministic cleaning, and prediction schema checks."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

FEATURES = [f"X{i}" for i in range(1, 65)]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_dataset(path):
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data")
    frame = pd.read_csv(path, skiprows=start + 1, header=None, na_values=["?"])
    if frame.shape[1] != 65:
        raise ValueError("Expected exactly 64 features plus the bankruptcy target.")
    frame.columns = FEATURES + ["bankrupt"]
    if not frame.bankrupt.isin([0, 1]).all():
        raise ValueError("Bankruptcy labels must be 0 or 1.")
    frame[FEATURES] = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    raw_rows = len(frame)
    raw_positive = int(frame.bankrupt.sum())
    duplicate_count = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()
    # Conflicting labels for identical feature vectors cannot be assigned reliably.
    conflict = frame.duplicated(subset=FEATURES, keep=False)
    conflict_count = int(conflict.sum())
    frame = frame.loc[~conflict].copy()
    frame.index.name = "source_row"
    audit = dict(raw_rows=raw_rows, raw_positive=raw_positive,
                 exact_duplicates_removed=duplicate_count,
                 conflicting_rows_removed=conflict_count,
                 clean_rows=len(frame), clean_positive=int(frame.bankrupt.sum()),
                 features=64, missing_cells=int(frame[FEATURES].isna().sum().sum()),
                 sha256=sha256(path))
    return frame, audit

def validate_features(frame):
    if len(frame) == 0:
        raise ValueError("The CSV contains no rows.")
    missing = sorted(set(FEATURES) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(FEATURES))
    if missing or extra:
        raise ValueError(f"Use exactly X1 through X64. Missing: {missing}; unexpected: {extra}")
    ordered = frame.loc[:, FEATURES].copy()
    for column in FEATURES:
        converted = pd.to_numeric(ordered[column], errors="coerce")
        invalid = ordered[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"{column} contains nonnumeric values. Use empty cells for missing values.")
        ordered[column] = converted
    if np.isinf(ordered.to_numpy(dtype=float)).any():
        raise ValueError("Infinite values are not supported.")
    if ordered.isna().all(axis=1).any():
        raise ValueError("At least one row has all 64 features missing.")
    return ordered
