"""
Reproducibility seed utility — CVD Risk Predictor.

Sets random seeds for Python, NumPy, and PYTHONHASHSEED to ensure
deterministic behaviour across all pipeline stages.

Seed value is read from configs/model_config.yaml by default.
"""

import os
import random

import numpy as np
import yaml


def _load_seed_from_config() -> int:
    """Read the random_seed value from configs/model_config.yaml."""
    config_path = os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, "configs", "model_config.yaml"
    )
    with open(os.path.abspath(config_path), "r") as f:
        config = yaml.safe_load(f)
    return int(config["random_seed"])


def set_seed(seed: int | None = None) -> None:
    """Set random seed for Python, NumPy, and os.environ PYTHONHASHSEED.

    Parameters
    ----------
    seed : int, optional
        Seed value. If ``None``, the value is read from
        ``configs/model_config.yaml``.
    """
    if seed is None:
        seed = _load_seed_from_config()

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
