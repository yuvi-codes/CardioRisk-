"""
Preprocessing utilities — CVD Risk Predictor.

Handles target binarisation, cross-dataset feature alignment,
stratified three-way splitting, and StandardScaler management.

AGENTS.md §4: 60 / 20 / 20 split is FIXED and NON-NEGOTIABLE.
AGENTS.md §14: Scaler MUST be fitted on training data only.
"""

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.seed import set_seed

# ---------------------------------------------------------------------------
# Seed — set at module load for deterministic imports
# ---------------------------------------------------------------------------
set_seed()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

_MODEL_CFG_PATH = os.path.join(_ROOT, "configs", "model_config.yaml")
with open(_MODEL_CFG_PATH, "r") as _f:
    _MODEL_CFG = yaml.safe_load(_f)

_DATA_CFG_PATH = os.path.join(_ROOT, "configs", "dataset_config.yaml")
with open(_DATA_CFG_PATH, "r") as _f:
    _DATA_CFG = yaml.safe_load(_f)

_SEED: int = int(_MODEL_CFG["random_seed"])
_TRAIN_SIZE: float = float(_MODEL_CFG["train_size"])
_CAL_SIZE: float = float(_MODEL_CFG["calibration_size"])
_TEST_SIZE: float = float(_MODEL_CFG["test_size"])

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature-alignment mapping
# ---------------------------------------------------------------------------
# Maps a unified (aligned) column name → source column in each dataset.
# ``None`` means the dataset does not contain that feature.

_FEATURE_MAP = {
    # aligned_name        : (framingham_col,    cleveland_col)
    "age":                  ("age",             "age"),
    "sex":                  ("male",            "sex"),
    "cholesterol":          ("totChol",         "chol"),
    "systolic_bp":          ("sysBP",           "trestbps"),
    "fasting_blood_sugar":  ("diabetes",        "fbs"),
    "chest_pain_type":      (None,              "cp"),
    "max_heart_rate":       ("heartRate",       "thalach"),
    "exercise_angina":      (None,              "exang"),
    "st_depression":        (None,              "oldpeak"),
    "st_slope":             (None,              "slope"),
    "num_major_vessels":    (None,              "ca"),
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def binarise_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Map target values > 0 to 1 and 0 to 0.

    This is essential for the Cleveland dataset where the target ranges 0-4,
    but we need a binary classification (disease present / absent).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the target column.
    target_col : str
        Name of the target column to binarise.

    Returns
    -------
    pd.DataFrame
        DataFrame with binarised target column.
    """
    df = df.copy()
    df[target_col] = (df[target_col] > 0).astype(int)
    logger.info(
        "Binarised '%s' — class distribution: %s",
        target_col,
        df[target_col].value_counts().to_dict(),
    )
    return df


def align_features(
    framingham_df: pd.DataFrame,
    cleveland_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identify common clinical features and return aligned DataFrames.

    Columns that exist in only one dataset are dropped.  A report of every
    dropped column and the reason is printed to stdout **and** saved to
    ``artifacts/reports/feature_alignment.csv``.

    Parameters
    ----------
    framingham_df : pd.DataFrame
        Cleaned Framingham dataset (including target).
    cleveland_df : pd.DataFrame
        Cleaned Cleveland dataset (including target).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(framingham_aligned, cleveland_aligned)`` with only the shared
        columns, renamed to unified names.
    """
    # --- Build rename maps and determine shared features -------------------
    fram_rename: dict[str, str] = {}
    clev_rename: dict[str, str] = {}
    shared_features: list[str] = []

    for aligned_name, (fram_col, clev_col) in _FEATURE_MAP.items():
        if fram_col is not None and clev_col is not None:
            fram_rename[fram_col] = aligned_name
            clev_rename[clev_col] = aligned_name
            shared_features.append(aligned_name)

    # Also keep target columns (un-renamed) — they'll be handled separately
    fram_target = _DATA_CFG["framingham_target_column"]
    clev_target = _DATA_CFG["cleveland_target_column"]

    # --- Identify and document dropped columns -----------------------------
    drop_records: list[dict[str, str]] = []

    # Collect columns that are in _FEATURE_MAP but have None on one side.
    # We process these first so we can skip them in the raw-column scan.
    _map_only_fram: set[str] = set()  # Framingham cols with no Cleveland match
    _map_only_clev: set[str] = set()  # Cleveland cols with no Framingham match

    for aligned_name, (fram_col, clev_col) in _FEATURE_MAP.items():
        if fram_col is None and clev_col is not None:
            _map_only_clev.add(clev_col)
            drop_records.append(
                {
                    "dataset": "cleveland",
                    "column": clev_col,
                    "reason": f"No Framingham equivalent for '{aligned_name}'",
                }
            )
        elif clev_col is None and fram_col is not None:
            _map_only_fram.add(fram_col)
            drop_records.append(
                {
                    "dataset": "framingham",
                    "column": fram_col,
                    "reason": f"No Cleveland equivalent for '{aligned_name}'",
                }
            )

    # Framingham columns not in rename map AND not already handled above
    for col in framingham_df.columns:
        if col == fram_target or col in fram_rename or col in _map_only_fram:
            continue
        reason = _drop_reason_framingham(col)
        drop_records.append(
            {"dataset": "framingham", "column": col, "reason": reason}
        )

    # Cleveland columns not in rename map AND not already handled above
    for col in cleveland_df.columns:
        if col == clev_target or col in clev_rename or col in _map_only_clev:
            continue
        reason = _drop_reason_cleveland(col)
        drop_records.append(
            {"dataset": "cleveland", "column": col, "reason": reason}
        )

    drop_df = pd.DataFrame(drop_records)

    # Print the table (ASCII-safe for Windows cp1252 console)
    print("\n+--------------------------------------------------------------+")
    print("|          FEATURE ALIGNMENT -- DROPPED COLUMNS                |")
    print("+--------------------------------------------------------------+")
    if not drop_df.empty:
        print(drop_df.to_string(index=False))
    else:
        print("  No columns dropped.")
    print("+--------------------------------------------------------------+\n")

    # Save report
    report_dir = os.path.join(_ROOT, "artifacts", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "feature_alignment.csv")
    drop_df.to_csv(report_path, index=False)
    logger.info("Feature alignment report saved to %s", report_path)

    # --- Build aligned DataFrames ------------------------------------------
    fram_keep = list(fram_rename.keys()) + [fram_target]
    clev_keep = list(clev_rename.keys()) + [clev_target]

    framingham_aligned = framingham_df[fram_keep].rename(columns=fram_rename)
    cleveland_aligned = cleveland_df[clev_keep].rename(columns=clev_rename)

    logger.info(
        "Aligned features (%d shared): %s", len(shared_features), shared_features
    )
    logger.info("Framingham aligned shape: %s", framingham_aligned.shape)
    logger.info("Cleveland aligned shape:  %s", cleveland_aligned.shape)

    return framingham_aligned, cleveland_aligned


def three_way_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
]:
    """Stratified 60 / 20 / 20 split into train, calibration, and holdout.

    Split ratios and random seed are read from ``configs/model_config.yaml``.

    AGENTS.md §4 — the holdout set is NEVER used for any intermediate
    decision.  Calibrating and evaluating on the same split silently
    invalidates uncertainty results.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.

    Returns
    -------
    X_train, X_cal, X_holdout, y_train, y_cal, y_holdout
    """
    # First split: 60 % train vs 40 % remainder
    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y,
        test_size=(_CAL_SIZE + _TEST_SIZE),
        random_state=_SEED,
        stratify=y,
    )

    # Second split: 50/50 of remainder → 20 % calibration, 20 % holdout
    cal_ratio = _CAL_SIZE / (_CAL_SIZE + _TEST_SIZE)
    X_cal, X_holdout, y_cal, y_holdout = train_test_split(
        X_rem, y_rem,
        test_size=(1 - cal_ratio),
        random_state=_SEED,
        stratify=y_rem,
    )

    logger.info(
        "Three-way split (seed=%d): train=%s, cal=%s, holdout=%s",
        _SEED, X_train.shape, X_cal.shape, X_holdout.shape,
    )

    return X_train, X_cal, X_holdout, y_train, y_cal, y_holdout


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on training data ONLY.

    AGENTS.md §14: fitting the scaler on the full dataset would leak
    information from the calibration / holdout sets into the training
    pipeline.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.

    Returns
    -------
    StandardScaler
        Fitted scaler instance.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    logger.info("StandardScaler fitted on training data (%s)", X_train.shape)
    return scaler


def scale(
    X: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Apply a fitted StandardScaler to a feature matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix to transform.
    scaler : StandardScaler
        A scaler previously fitted via :func:`fit_scaler`.

    Returns
    -------
    pd.DataFrame
        Scaled feature matrix with original column names preserved.
    """
    return pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )


def save_processed(df: pd.DataFrame, name: str) -> None:
    """Write a DataFrame to ``data/processed/<name>.csv``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    name : str
        Filename stem (e.g. ``"framingham_clean"``).
    """
    out_dir = os.path.join(_ROOT, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    logger.info("Saved processed dataset to %s (%s)", path, df.shape)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _drop_reason_framingham(col: str) -> str:
    """Return a human-readable reason for dropping a Framingham column."""
    reasons = {
        "education":       "Socio-demographic variable, not a clinical risk factor; no Cleveland equivalent",
        "currentSmoker":   "Binary smoking flag; Cleveland has no equivalent. cigsPerDay also dropped",
        "cigsPerDay":      "Continuous smoking variable; no Cleveland equivalent. Will be captured by LRS",
        "BPMeds":          "Blood-pressure medication flag; no Cleveland equivalent",
        "prevalentStroke":  "Medical history flag; no Cleveland equivalent",
        "prevalentHyp":    "Medical history flag; no Cleveland equivalent",
        "diaBP":           "Diastolic BP; Cleveland only records resting (systolic-equivalent) trestbps",
        "BMI":             "Body mass index; no Cleveland equivalent",
        "glucose":         "Fasting glucose; Cleveland uses binary fbs instead",
    }
    return reasons.get(col, "No equivalent column in Cleveland dataset")


def _drop_reason_cleveland(col: str) -> str:
    """Return a human-readable reason for dropping a Cleveland column."""
    reasons = {
        "restecg": "Resting ECG result; no Framingham equivalent",
        "thal":    "Thalassemia type; no Framingham equivalent",
    }
    return reasons.get(col, "No equivalent column in Framingham dataset")
