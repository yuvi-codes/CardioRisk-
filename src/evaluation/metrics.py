"""
Holdout evaluation utilities — CVD Risk Predictor.

Called after training and calibration to produce final metrics on the
held-out test set. These numbers are reported in the paper.
"""

import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def _to_1d(y) -> np.ndarray:
    if hasattr(y, "iloc"):
        return y.iloc[:, 0].to_numpy() if getattr(y, "ndim", 1) == 2 else y.to_numpy()
    return np.asarray(y).ravel()


def evaluate_model(
    model,
    X: pd.DataFrame,
    y,
    threshold: float = 0.5,
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Compute a full evaluation suite on a given dataset split.

    Parameters
    ----------
    model : fitted classifier with predict_proba
    X : pd.DataFrame — feature matrix (already scaled)
    y : array-like — true labels
    threshold : float — classification threshold
    model_name : str — label for logging

    Returns
    -------
    dict with keys: auc_roc, auc_pr, brier, accuracy, sensitivity,
                    specificity, ppv, npv, f1, confusion_matrix
    """
    y_true = _to_1d(y)
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr  = average_precision_score(y_true, y_prob)
    brier   = brier_score_loss(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0.0

    metrics = {
        "model": model_name,
        "auc_roc":     round(auc_roc, 4),
        "auc_pr":      round(auc_pr, 4),
        "brier":       round(brier, 4),
        "accuracy":    round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "ppv":         round(ppv, 4),
        "npv":         round(npv, 4),
        "f1":          round(f1, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }

    logger.info("[%s] AUC-ROC=%.4f | AUC-PR=%.4f | Brier=%.4f | "
                "Sens=%.4f | Spec=%.4f | F1=%.4f",
                model_name, auc_roc, auc_pr, brier,
                sensitivity, specificity, f1)

    return metrics


def print_metrics_table(metrics_list: list) -> None:
    """Pretty-print a comparison table of multiple model metrics."""
    keys = ["model", "auc_roc", "auc_pr", "brier", "sensitivity", "specificity", "f1"]
    df = pd.DataFrame(metrics_list)[keys]
    print("\n" + "=" * 70)
    print("HOLDOUT EVALUATION RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70 + "\n")


def save_metrics(metrics_list: list, out_path: str = "artifacts/metrics/holdout_metrics.csv") -> None:
    """Save metrics dict list to CSV."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_list).to_csv(path, index=False)
    logger.info("Metrics saved to %s", path)