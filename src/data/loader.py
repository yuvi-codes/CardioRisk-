"""
Data loading utilities — CVD Risk Predictor.

Reads raw and processed CSV files for Framingham and Cleveland datasets.
All paths are resolved from configs/dataset_config.yaml.
"""

import logging
import os

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "configs", "dataset_config.yaml"
)
with open(os.path.abspath(_CONFIG_PATH), "r") as _f:
    _CFG = yaml.safe_load(_f)

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_framingham() -> pd.DataFrame:
    """Load the raw Framingham CSV, drop rows with any missing values.

    Returns
    -------
    pd.DataFrame
        Cleaned Framingham dataset (listwise deletion applied).
    """
    path = os.path.join(_PROJECT_ROOT, _CFG["framingham_raw_path"])
    df = pd.read_csv(path)
    initial_shape = df.shape
    df = df.dropna()
    logger.info(
        "Loaded Framingham: %s → %s after listwise deletion (%d rows dropped)",
        initial_shape,
        df.shape,
        initial_shape[0] - df.shape[0],
    )
    return df.reset_index(drop=True)


def load_cleveland() -> pd.DataFrame:
    """Load the raw Cleveland CSV, replace ``?`` with NaN, drop missing rows.

    The Cleveland dataset uses ``?`` as the missing-value marker.

    Returns
    -------
    pd.DataFrame
        Cleaned Cleveland dataset.
    """
    path = os.path.join(_PROJECT_ROOT, _CFG["cleveland_raw_path"])
    df = pd.read_csv(path, na_values="?")
    initial_shape = df.shape
    df = df.dropna()
    logger.info(
        "Loaded Cleveland: %s → %s after dropping missing values (%d rows dropped)",
        initial_shape,
        df.shape,
        initial_shape[0] - df.shape[0],
    )
    return df.reset_index(drop=True)


def load_processed(dataset_name: str) -> pd.DataFrame:
    """Load a processed CSV from ``data/processed/``.

    Parameters
    ----------
    dataset_name : str
        Either ``"framingham"`` or ``"cleveland"``.

    Returns
    -------
    pd.DataFrame
        The processed dataset.

    Raises
    ------
    ValueError
        If *dataset_name* is not ``"framingham"`` or ``"cleveland"``.
    """
    key = f"{dataset_name}_processed_path"
    if key not in _CFG:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Expected 'framingham' or 'cleveland'."
        )
    path = os.path.join(_PROJECT_ROOT, _CFG[key])
    df = pd.read_csv(path)
    logger.info("Loaded processed %s: %s", dataset_name, df.shape)
    return df
