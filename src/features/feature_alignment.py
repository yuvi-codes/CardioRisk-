"""
Feature alignment utilities -- CVD Risk Predictor.

Provides the canonical column-name mappings for Framingham and Cleveland
datasets, and an ``align_and_document`` function that applies them,
computes the intersection, drops non-shared columns, and writes a
detailed alignment report to ``artifacts/reports/feature_alignment.csv``.

The mappings here are the single source of truth for column renaming.
They are intentionally duplicated from ``src/data/preprocessing._FEATURE_MAP``
to keep the features package self-contained; any change to the mapping
must be reflected in both places.

Clinical notes on selected mappings:
- ``male`` -> ``sex``:        Framingham uses ``male`` (1/0), Cleveland uses ``sex`` (1/0 same encoding)
- ``totChol`` -> ``cholesterol``: both measure total serum cholesterol in mg/dL
- ``sysBP`` -> ``systolic_bp``:   Framingham systolic BP; Cleveland ``trestbps`` is resting BP (systolic)
- ``diabetes`` -> ``fasting_blood_sugar``: Framingham ``diabetes`` is binary; Cleveland ``fbs`` is fbs > 120 mg/dL
- ``heartRate`` -> ``max_heart_rate``:  See data/README.md for the semantic caveat (resting vs exercise max)
"""

import logging
import os
from typing import Dict, List, Tuple

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

_DATA_CFG_PATH = os.path.join(_ROOT, "configs", "dataset_config.yaml")
with open(_DATA_CFG_PATH, "r") as _f:
    _DATA_CFG = yaml.safe_load(_f)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column maps
# ---------------------------------------------------------------------------

# Framingham raw column -> standardised project column name
_FRAMINGHAM_MAP: Dict[str, str] = {
    "age":       "age",
    "male":      "sex",
    "totChol":   "cholesterol",
    "sysBP":     "systolic_bp",
    "diabetes":  "fasting_blood_sugar",
    "heartRate": "max_heart_rate",
}

# Cleveland raw column -> standardised project column name
_CLEVELAND_MAP: Dict[str, str] = {
    "age":      "age",
    "sex":      "sex",
    "chol":     "cholesterol",
    "trestbps": "systolic_bp",
    "fbs":      "fasting_blood_sugar",
    "thalach":  "max_heart_rate",
}

# Reasons for dropping columns that don't appear in the intersection
_DROP_REASONS: Dict[str, str] = {
    # Framingham-only
    "education":       "Socio-demographic variable, not a clinical risk factor; no Cleveland equivalent",
    "currentSmoker":   "Binary smoking flag; no Cleveland equivalent. Captured by LRS",
    "cigsPerDay":      "Continuous smoking measure; no Cleveland equivalent. Captured by LRS",
    "BPMeds":          "Blood-pressure medication flag; no Cleveland equivalent",
    "prevalentStroke": "Medical history flag; no Cleveland equivalent",
    "prevalentHyp":    "Medical history flag; no Cleveland equivalent",
    "diaBP":           "Diastolic BP; Cleveland only records systolic-equivalent trestbps",
    "BMI":             "Body mass index; no Cleveland equivalent",
    "glucose":         "Continuous fasting glucose; Cleveland uses binary fbs instead",
    # Cleveland-only
    "cp":              "Chest pain type (4 categories); no Framingham equivalent",
    "restecg":         "Resting ECG result; no Framingham equivalent",
    "exang":           "Exercise-induced angina; no Framingham equivalent",
    "oldpeak":         "ST depression induced by exercise; no Framingham equivalent",
    "slope":           "Slope of peak exercise ST segment; no Framingham equivalent",
    "ca":              "Number of major vessels coloured by fluoroscopy; no Framingham equivalent",
    "thal":            "Thalassemia type; no Framingham equivalent",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_framingham_column_map() -> Dict[str, str]:
    """Return a dictionary mapping Framingham column names to standardised names.

    Returns
    -------
    Dict[str, str]
        ``{raw_column: standardised_column}`` for columns that have a
        cross-dataset equivalent.
    """
    return dict(_FRAMINGHAM_MAP)


def get_cleveland_column_map() -> Dict[str, str]:
    """Return a dictionary mapping Cleveland column names to standardised names.

    Returns
    -------
    Dict[str, str]
        ``{raw_column: standardised_column}`` for columns that have a
        cross-dataset equivalent.
    """
    return dict(_CLEVELAND_MAP)


def align_and_document(
    framingham_df: pd.DataFrame,
    cleveland_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align both datasets to shared standardised columns and write a report.

    Workflow:
    1. Rename columns in each dataset using the canonical maps.
    2. Compute the intersection of standardised column names.
    3. Drop columns not in the intersection.
    4. Write a detailed CSV report to ``artifacts/reports/feature_alignment.csv``
       with columns: ``original_name``, ``dataset``, ``standardised_name``,
       ``dropped`` (boolean), ``reason``.

    Parameters
    ----------
    framingham_df : pd.DataFrame
        Cleaned Framingham dataset (including target column).
    cleveland_df : pd.DataFrame
        Cleaned Cleveland dataset (including target column).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(framingham_aligned, cleveland_aligned)`` containing only the
        shared columns (renamed) plus their respective target columns.
    """
    fram_target = _DATA_CFG["framingham_target_column"]
    clev_target = _DATA_CFG["cleveland_target_column"]

    # --- Compute shared standardised names ---------------------------------
    fram_std_names = set(_FRAMINGHAM_MAP.values())
    clev_std_names = set(_CLEVELAND_MAP.values())
    shared: set[str] = fram_std_names & clev_std_names

    logger.info("Shared standardised features (%d): %s", len(shared), sorted(shared))

    # --- Build the alignment report ----------------------------------------
    records: List[Dict] = []

    # Framingham columns
    for raw_col in framingham_df.columns:
        if raw_col == fram_target:
            records.append({
                "original_name": raw_col,
                "dataset": "framingham",
                "standardised_name": raw_col,
                "dropped": False,
                "reason": "Target column (retained as-is)",
            })
            continue

        std_name = _FRAMINGHAM_MAP.get(raw_col)
        if std_name is not None and std_name in shared:
            records.append({
                "original_name": raw_col,
                "dataset": "framingham",
                "standardised_name": std_name,
                "dropped": False,
                "reason": "",
            })
        else:
            reason = _DROP_REASONS.get(
                raw_col, "No equivalent column in Cleveland dataset"
            )
            records.append({
                "original_name": raw_col,
                "dataset": "framingham",
                "standardised_name": std_name or "",
                "dropped": True,
                "reason": reason,
            })

    # Cleveland columns
    for raw_col in cleveland_df.columns:
        if raw_col == clev_target:
            records.append({
                "original_name": raw_col,
                "dataset": "cleveland",
                "standardised_name": raw_col,
                "dropped": False,
                "reason": "Target column (retained as-is)",
            })
            continue

        std_name = _CLEVELAND_MAP.get(raw_col)
        if std_name is not None and std_name in shared:
            records.append({
                "original_name": raw_col,
                "dataset": "cleveland",
                "standardised_name": std_name,
                "dropped": False,
                "reason": "",
            })
        else:
            reason = _DROP_REASONS.get(
                raw_col, "No equivalent column in Framingham dataset"
            )
            records.append({
                "original_name": raw_col,
                "dataset": "cleveland",
                "standardised_name": std_name or "",
                "dropped": True,
                "reason": reason,
            })

    report_df = pd.DataFrame(records)

    # --- Save report -------------------------------------------------------
    report_dir = os.path.join(_ROOT, "artifacts", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "feature_alignment.csv")
    report_df.to_csv(report_path, index=False)
    logger.info("Feature alignment report saved to %s", report_path)

    # --- Print summary (ASCII-safe) ----------------------------------------
    dropped = report_df[report_df["dropped"] == True]  # noqa: E712
    print("\n+--------------------------------------------------------------+")
    print("|          FEATURE ALIGNMENT REPORT                            |")
    print("+--------------------------------------------------------------+")
    if not dropped.empty:
        print(dropped[["dataset", "original_name", "reason"]].to_string(index=False))
    else:
        print("  No columns dropped.")
    print("+--------------------------------------------------------------+\n")

    # --- Build aligned DataFrames ------------------------------------------
    # Keep only raw columns that map to a shared standardised name + target
    fram_keep = [c for c in framingham_df.columns if _FRAMINGHAM_MAP.get(c) in shared]
    fram_keep.append(fram_target)

    clev_keep = [c for c in cleveland_df.columns if _CLEVELAND_MAP.get(c) in shared]
    clev_keep.append(clev_target)

    framingham_aligned = framingham_df[fram_keep].rename(columns=_FRAMINGHAM_MAP)
    cleveland_aligned = cleveland_df[clev_keep].rename(columns=_CLEVELAND_MAP)

    logger.info("Framingham aligned: %s", framingham_aligned.shape)
    logger.info("Cleveland aligned:  %s", cleveland_aligned.shape)

    return framingham_aligned, cleveland_aligned
