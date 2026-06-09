"""Canonical preprocessing pipeline with strict leakage prevention."""

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.features.feature_alignment import align_and_document
from src.features.lrs import append_lrs, compute_lrs
from src.utils.seed import set_seed

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "artifacts" / "reports"
DATASET_CONFIG_PATH = ROOT_DIR / "configs" / "dataset_config.yaml"


def _load_dataset_config() -> dict:
    with open(DATASET_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Framingham and Cleveland raw datasets."""
    framingham_path = RAW_DIR / "framingham.csv"
    cleveland_path = RAW_DIR / "cleveland.csv"

    missing = []
    if not framingham_path.exists():
        missing.append(str(framingham_path))
    if not cleveland_path.exists():
        missing.append(str(cleveland_path))
    if missing:
        raise FileNotFoundError(
            "Missing required raw dataset file(s): " + ", ".join(missing)
        )

    framingham_df = pd.read_csv(framingham_path)
    cleveland_df = pd.read_csv(cleveland_path, na_values=["?"])
    return framingham_df, cleveland_df


def preprocess_datasets(random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """Run canonical preprocessing and persist leakage-safe artifacts."""
    # A. deterministic seed
    set_seed(random_state)

    # B. load raw datasets
    framingham_raw, cleveland_raw = load_raw_datasets()

    # C. align features and document alignment decisions
    framingham_aligned, cleveland_aligned = align_and_document(
        framingham_raw, cleveland_raw
    )

    cfg = _load_dataset_config()
    framingham_target_col = cfg["framingham_target_column"]
    lifestyle_columns = cfg["lrs_components"]

    # D/E. compute and append LRS to Framingham only
    missing_lrs_columns = [c for c in lifestyle_columns if c not in framingham_aligned.columns]
    if missing_lrs_columns:
        raise ValueError(
            "Cannot compute LRS: missing lifestyle columns in aligned Framingham dataset: "
            f"{missing_lrs_columns}. Please provide these columns before preprocessing."
        )
    lrs_series = compute_lrs(framingham_aligned, {c: c for c in lifestyle_columns})
    framingham_aligned = append_lrs(framingham_aligned, lrs_series)

    # F. explicit listwise deletion
    framingham_aligned = framingham_aligned.dropna().reset_index(drop=True)
    cleveland_aligned = cleveland_aligned.dropna().reset_index(drop=True)

    # G. separate features and target (training universe = Framingham)
    X = framingham_aligned.drop(columns=[framingham_target_col])
    y = framingham_aligned[framingham_target_col].astype(int)

    # H. strict three-way split: 60/20/20 with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=random_state,
    )
    X_calib, X_holdout, y_calib, y_holdout = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )

    # explicit non-overlap guard
    train_idx = set(X_train.index)
    calib_idx = set(X_calib.index)
    holdout_idx = set(X_holdout.index)
    if train_idx & calib_idx or train_idx & holdout_idx or calib_idx & holdout_idx:
        raise RuntimeError("Leakage detected: split overlap found among train/calib/holdout.")

    # I. fit scaler ONLY on training data, transform others
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_calib_scaled = pd.DataFrame(
        scaler.transform(X_calib), columns=X_calib.columns, index=X_calib.index
    )
    X_holdout_scaled = pd.DataFrame(
        scaler.transform(X_holdout), columns=X_holdout.columns, index=X_holdout.index
    )

    # J. save processed outputs
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train_scaled.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_calib_scaled.to_csv(PROCESSED_DIR / "X_calib.csv", index=False)
    X_holdout_scaled.to_csv(PROCESSED_DIR / "X_holdout.csv", index=False)
    y_train.to_frame(name=framingham_target_col).to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_calib.to_frame(name=framingham_target_col).to_csv(PROCESSED_DIR / "y_calib.csv", index=False)
    y_holdout.to_frame(name=framingham_target_col).to_csv(PROCESSED_DIR / "y_holdout.csv", index=False)

    # K. save scaler object
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # L. write preprocessing metadata report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    class_dist = {
        "train": y_train.value_counts(normalize=True).sort_index().to_dict(),
        "calib": y_calib.value_counts(normalize=True).sort_index().to_dict(),
        "holdout": y_holdout.value_counts(normalize=True).sort_index().to_dict(),
    }
    report = (
        "Preprocessing Report\n"
        "====================\n"
        f"Random seed: {random_state}\n"
        f"Raw Framingham shape: {framingham_raw.shape}\n"
        f"Raw Cleveland shape: {cleveland_raw.shape}\n"
        f"Aligned Framingham shape (after dropna): {framingham_aligned.shape}\n"
        f"Aligned Cleveland shape (after dropna): {cleveland_aligned.shape}\n"
        f"Split sizes: train={X_train_scaled.shape}, calib={X_calib_scaled.shape}, holdout={X_holdout_scaled.shape}\n"
        f"Class distribution (proportion): {class_dist}\n"
        f"Feature count: {X_train_scaled.shape[1]}\n"
        "LRS appended: yes (Framingham only)\n"
        "Scaler fit scope: training set only\n"
    )
    (REPORTS_DIR / "preprocessing_report.txt").write_text(report, encoding="utf-8")

    return {
        "X_train": X_train_scaled,
        "X_calib": X_calib_scaled,
        "X_holdout": X_holdout_scaled,
        "y_train": y_train.to_frame(name=framingham_target_col),
        "y_calib": y_calib.to_frame(name=framingham_target_col),
        "y_holdout": y_holdout.to_frame(name=framingham_target_col),
    }


def load_processed_data() -> Dict[str, pd.DataFrame]:
    """Load saved processed CSV artifacts."""
    required = [
        "X_train.csv",
        "X_calib.csv",
        "X_holdout.csv",
        "y_train.csv",
        "y_calib.csv",
        "y_holdout.csv",
    ]
    missing = [f for f in required if not (PROCESSED_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed file(s): " + ", ".join(str(PROCESSED_DIR / f) for f in missing)
        )

    return {
        "X_train": pd.read_csv(PROCESSED_DIR / "X_train.csv"),
        "X_calib": pd.read_csv(PROCESSED_DIR / "X_calib.csv"),
        "X_holdout": pd.read_csv(PROCESSED_DIR / "X_holdout.csv"),
        "y_train": pd.read_csv(PROCESSED_DIR / "y_train.csv"),
        "y_calib": pd.read_csv(PROCESSED_DIR / "y_calib.csv"),
        "y_holdout": pd.read_csv(PROCESSED_DIR / "y_holdout.csv"),
    }


if __name__ == "__main__":
    processed = preprocess_datasets()
    print("Final split shapes:")
    print(f"X_train: {processed['X_train'].shape}, y_train: {processed['y_train'].shape}")
    print(f"X_calib: {processed['X_calib'].shape}, y_calib: {processed['y_calib'].shape}")
    print(f"X_holdout: {processed['X_holdout'].shape}, y_holdout: {processed['y_holdout'].shape}")
    print("Saved: models/scaler.pkl and preprocessing artifacts in data/processed + artifacts/reports.")
