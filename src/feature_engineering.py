"""
Phase 2A: Feature engineering for traditional ML models.
Loads pre-extracted CSV features, applies track-level split,
and scales features using StandardScaler (fit on train only).
"""
import os
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import get_logger, ensure_dirs

logger = get_logger(__name__)


def _parse_track_id(filename: str) -> str:
    """
    Extract track ID from filename.
    'blues.00000.wav' -> 'blues.00000'
    'blues.00000.3.wav' (3-sec) -> 'blues.00000'
    """
    parts = filename.replace(".wav", "").split(".")
    # 30-sec: ['blues', '00000']
    # 3-sec:  ['blues', '00000', '3']
    return f"{parts[0]}.{parts[1]}"


def load_and_split_features(csv_path: str, splits: dict):
    """
    Load a feature CSV, assign rows to partitions based on track-level splits,
    and return (X_train, y_train, X_val, y_val, X_test, y_test).

    Works for both features_30_sec.csv and features_3_sec.csv.
    """
    df = pd.read_csv(csv_path)
    logger.info("Loaded %s: %d rows, %d columns", csv_path, len(df), len(df.columns))

    # Map each row to a track ID
    df["track_id"] = df["filename"].apply(_parse_track_id)

    # Build a lookup: track_id -> partition
    partition_map = {}
    for partition, track_ids in splits.items():
        for tid in track_ids:
            partition_map[tid] = partition

    # Assign partitions (skip rows with unknown track IDs, e.g. corrupted tracks)
    df["partition"] = df["track_id"].map(partition_map)
    unknown = df["partition"].isna().sum()
    if unknown > 0:
        logger.info("Dropping %d rows with no split assignment (likely corrupted)", unknown)
        df = df.dropna(subset=["partition"])

    # Separate features and labels
    drop_cols = ["filename", "length", "track_id", "partition"]
    feature_cols = [c for c in df.columns if c not in drop_cols and c != "label"]
    label_col = "label"

    logger.info("Using %d features: %s ... %s", len(feature_cols), feature_cols[:3], feature_cols[-3:])

    result = {}
    for partition in ["train", "val", "test"]:
        mask = df["partition"] == partition
        result[f"X_{partition}"] = df.loc[mask, feature_cols].values.astype(np.float32)
        result[f"y_{partition}"] = df.loc[mask, label_col].values
        logger.info("  %s: %d samples", partition, mask.sum())

    return result, feature_cols


def scale_features(data: dict):
    """
    Fit StandardScaler on training set, transform all partitions.
    Saves the fitted scaler for later use in inference.
    """
    scaler = StandardScaler()
    data["X_train"] = scaler.fit_transform(data["X_train"])
    data["X_val"] = scaler.transform(data["X_val"])
    data["X_test"] = scaler.transform(data["X_test"])

    ensure_dirs(config.MODELS_DIR)
    scaler_path = os.path.join(config.MODELS_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved to %s", scaler_path)

    return data, scaler


def encode_labels(data: dict):
    """Encode string labels to integer indices."""
    le = LabelEncoder()
    le.fit(config.GENRES)  # consistent ordering

    for partition in ["train", "val", "test"]:
        key = f"y_{partition}"
        data[key] = le.transform(data[key])

    le_path = os.path.join(config.MODELS_DIR, "label_encoder.joblib")
    joblib.dump(le, le_path)
    logger.info("Label encoder saved (classes: %s)", list(le.classes_))
    return data, le


def prepare_features(use_3sec: bool = False):
    """
    Full feature preparation pipeline for traditional ML.
    Returns scaled and encoded (X_train, y_train, ...) arrays.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2A: Feature Engineering (%s)", "3-sec" if use_3sec else "30-sec")
    logger.info("=" * 60)

    # Load splits
    with open(config.SPLITS_FILE, "r") as f:
        splits = json.load(f)

    csv_path = config.CSV_3S if use_3sec else config.CSV_30S
    data, feature_cols = load_and_split_features(csv_path, splits)
    data, scaler = scale_features(data)
    data, le = encode_labels(data)

    # Quick sanity check
    assert not np.isnan(data["X_train"]).any(), "NaN in training features"
    assert not np.isnan(data["X_val"]).any(), "NaN in validation features"
    assert not np.isnan(data["X_test"]).any(), "NaN in test features"

    return data, feature_cols, le


if __name__ == "__main__":
    data, feature_cols, le = prepare_features(use_3sec=False)
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Train shape: {data['X_train'].shape}")
    print(f"Val shape:   {data['X_val'].shape}")
    print(f"Test shape:  {data['X_test'].shape}")
