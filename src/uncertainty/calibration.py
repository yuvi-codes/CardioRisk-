"""Calibration utilities for probability reliability and Brier evaluation."""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from src.utils.seed import set_seed

set_seed()


def _to_1d(y):
    if hasattr(y, "iloc"):
        if getattr(y, "ndim", 1) == 2:
            return y.iloc[:, 0].to_numpy()
        return y.to_numpy()
    return np.asarray(y).ravel()


def _positive_probs(model, X):
    probs = model.predict_proba(X)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError("Model predict_proba must return class probabilities with 2 columns.")
    return probs[:, 1]


def calibrate_platt(model, X_cal, y_cal):
    """Apply Platt scaling (sigmoid) on a prefit model using calibration set."""
    calibrator = CalibratedClassifierCV(estimator=model, method="sigmoid", cv=None)
    calibrator.fit(X_cal, _to_1d(y_cal))
    return calibrator


def calibrate_isotonic(model, X_cal, y_cal):
    """Apply isotonic calibration on a prefit model using calibration set."""
    calibrator = CalibratedClassifierCV(estimator=model, method="isotonic", cv=None)
    calibrator.fit(X_cal, _to_1d(y_cal))
    return calibrator


def compute_brier_score(model, X, y) -> float:
    """Compute Brier score for binary class probabilities."""
    y_true = _to_1d(y)
    y_prob = _positive_probs(model, X)
    return float(brier_score_loss(y_true, y_prob))


def plot_calibration_curve(models_dict: Dict[str, object], X, y, save_path):
    """Plot reliability diagrams for all required models and save under artifacts/plots/."""
    required = {
        "Uncalibrated XGBoost",
        "Platt-calibrated XGBoost",
        "Isotonic-calibrated XGBoost",
        "Logistic Regression",
    }
    missing = required - set(models_dict.keys())
    if missing:
        raise ValueError(f"models_dict is missing required keys: {sorted(missing)}")

    y_true = _to_1d(y)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    for name, model in models_dict.items():
        y_prob = _positive_probs(model, X)
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, label=name)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    requested_path = Path(save_path)
    plots_dir = Path("artifacts") / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if "artifacts" not in requested_path.parts or "plots" not in requested_path.parts:
        final_path = plots_dir / requested_path.name
    else:
        final_path = requested_path
        final_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(final_path, dpi=300)
    return fig


def select_best_calibrator(platt_model, isotonic_model, X_cal, y_cal) -> Tuple[object, str, float, float]:
    """Select calibrator with lower calibration-set Brier score."""
    platt_brier = compute_brier_score(platt_model, X_cal, y_cal)
    isotonic_brier = compute_brier_score(isotonic_model, X_cal, y_cal)

    if platt_brier <= isotonic_brier:
        return platt_model, "platt", platt_brier, isotonic_brier
    return isotonic_model, "isotonic", platt_brier, isotonic_brier
