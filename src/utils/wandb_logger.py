"""Weights & Biases logging utilities for training runs."""

from pathlib import Path
from typing import Any, Dict, Optional

import wandb
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
TRAINING_CONFIG_PATH = ROOT_DIR / "configs" / "training_config.yaml"

with open(TRAINING_CONFIG_PATH, "r", encoding="utf-8") as f:
    TRAINING_CONFIG = yaml.safe_load(f)


def init_run(run_name: str, config_dict: Dict[str, Any]) -> wandb.sdk.wandb_run.Run:
    """Initialize a W&B run with required metadata in config."""
    merged_config = dict(config_dict or {})

    # Required tracked parameters per project policy
    required_keys = ["random_seed", "dataset_name", "split_sizes", "model_type"]
    for key in required_keys:
        if key not in merged_config:
            merged_config[key] = "missing"

    run = wandb.init(
        project=TRAINING_CONFIG["wandb_project"],
        name=run_name,
        tags=TRAINING_CONFIG.get("wandb_tags", []),
        config=merged_config,
    )
    return run


def log_metrics(metrics_dict: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log a dictionary of metrics to active W&B run."""
    if step is None:
        wandb.log(metrics_dict)
    else:
        wandb.log(metrics_dict, step=step)


def log_artifact(file_path: str, artifact_type: str, artifact_name: str) -> None:
    """Log a file as a W&B artifact."""
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    artifact.add_file(file_path)
    wandb.log_artifact(artifact)


def finish_run() -> None:
    """Close the active W&B run."""
    wandb.finish()
