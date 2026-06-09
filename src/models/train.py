"""Model training utilities with W&B logging."""

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.utils.seed import set_seed
from src.utils.wandb_logger import finish_run, init_run, log_artifact, log_metrics

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "configs"
MODELS_DIR = ROOT_DIR / "models"

set_seed()


def load_all_configs() -> Dict[str, Dict[str, Any]]:
    """Load dataset/model/training configs as a dictionary."""
    configs: Dict[str, Dict[str, Any]] = {}
    for name in ["dataset_config.yaml", "model_config.yaml", "training_config.yaml"]:
        path = CONFIG_DIR / name
        with open(path, "r", encoding="utf-8") as f:
            configs[name.replace(".yaml", "")] = yaml.safe_load(f)
    return configs


def _as_1d(y: Any) -> np.ndarray:
    if isinstance(y, pd.DataFrame):
        return y.iloc[:, 0].to_numpy()
    if isinstance(y, pd.Series):
        return y.to_numpy()
    return np.asarray(y).ravel()


def train_xgboost(X_train, y_train, X_cal, y_cal):
    """Train XGBoost with 5-fold stratified GridSearchCV and log to W&B."""
    configs = load_all_configs()
    model_cfg = configs["model_config"]
    seed = int(model_cfg["random_seed"])
    set_seed(seed)

    y_train_1d = _as_1d(y_train)
    y_cal_1d = _as_1d(y_cal)

    run = init_run(
        run_name="xgboost_training",
        config_dict={
            "random_seed": seed,
            "dataset_name": "framingham",
            "split_sizes": {"train": 0.60, "calibration": 0.20, "holdout": 0.20},
            "model_type": "xgboost",
        },
    )

    try:
        base = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=seed,
            n_jobs=-1,
        )
        grid = model_cfg["xgboost"]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        search = GridSearchCV(
            estimator=base,
            param_grid=grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train_1d)

        best_model = search.best_estimator_
        best_model.fit(X_train, y_train_1d)

        cal_probs = best_model.predict_proba(X_cal)[:, 1]
        cal_auc = float(roc_auc_score(y_cal_1d, cal_probs))

        log_metrics(
            {
                "xgboost/best_cv_auc": float(search.best_score_),
                "xgboost/calibration_auc": cal_auc,
            }
        )

        for key, value in search.best_params_.items():
            log_metrics({f"xgboost/best_param/{key}": value})

        run.summary["xgboost_best_params"] = search.best_params_
        run.summary["xgboost_calibration_auc"] = cal_auc
        return best_model
    finally:
        finish_run()


def train_logistic(X_train, y_train, X_cal, y_cal):
    """Train Logistic Regression with 5-fold stratified GridSearchCV and log to W&B."""
    configs = load_all_configs()
    model_cfg = configs["model_config"]
    seed = int(model_cfg["random_seed"])
    set_seed(seed)

    y_train_1d = _as_1d(y_train)
    y_cal_1d = _as_1d(y_cal)

    run = init_run(
        run_name="logistic_training",
        config_dict={
            "random_seed": seed,
            "dataset_name": "framingham",
            "split_sizes": {"train": 0.60, "calibration": 0.20, "holdout": 0.20},
            "model_type": "logistic_regression",
        },
    )

    try:
        max_iter = int(model_cfg["logistic_regression"].get("max_iter", 1000))
        base = LogisticRegression(max_iter=max_iter, random_state=seed)

        param_grid = {
            "C": model_cfg["logistic_regression"]["C"],
            "solver": model_cfg["logistic_regression"]["solver"],
            "penalty": model_cfg["logistic_regression"]["penalty"],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        search = GridSearchCV(
            estimator=base,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train_1d)

        best_model = search.best_estimator_
        best_model.fit(X_train, y_train_1d)

        cal_probs = best_model.predict_proba(X_cal)[:, 1]
        cal_auc = float(roc_auc_score(y_cal_1d, cal_probs))

        log_metrics(
            {
                "logistic/best_cv_auc": float(search.best_score_),
                "logistic/calibration_auc": cal_auc,
            }
        )

        for key, value in search.best_params_.items():
            log_metrics({f"logistic/best_param/{key}": value})

        run.summary["logistic_best_params"] = search.best_params_
        run.summary["logistic_calibration_auc"] = cal_auc
        return best_model
    finally:
        finish_run()


def save_model(model, filename: str) -> Path:
    """Serialize a model to models/ and log it as a W&B artifact."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / filename
    joblib.dump(model, output_path)

    run = init_run(
        run_name=f"artifact_{output_path.stem}",
        config_dict={
            "random_seed": "n/a",
            "dataset_name": "framingham",
            "split_sizes": {"train": 0.60, "calibration": 0.20, "holdout": 0.20},
            "model_type": output_path.stem,
        },
    )
    try:
        log_artifact(
            file_path=str(output_path),
            artifact_type="model",
            artifact_name=output_path.stem,
        )
        run.summary["artifact_path"] = str(output_path)
    finally:
        finish_run()

    return output_path
