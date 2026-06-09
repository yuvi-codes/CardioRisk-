"""
Lifestyle Risk Score (LRS) — CVD Risk Predictor.

Composite engineered feature built from five lifestyle components, each
normalised independently to [0, 1] and combined with equal weights (0.2).

AGENTS.md SS7 specifies:
- Components: sleep regularity, physical activity (METs), smoking (pack-years),
  sedentary hours, alcohol (units/week)
- Each normalised to 0-1 before weighting
- Weights must be justified: we use equal weights (0.2 each) as the default,
  documented explicitly so reviewers can assess the choice
- LRS is fed into the model as ONE additional feature alongside clinical features
- LRS contribution is evaluated via SHAP (Contribution 4)

Clinical rationale for normalisation directions:
- Higher smoking pack-years   -> higher CVD risk (dose-response, WHO/AHA)
- Higher sedentary hours/day  -> higher CVD risk (metabolic syndrome pathway)
- Higher alcohol units/week   -> higher CVD risk (J-curve, but linear for heavy use)
- Higher sleep irregularity   -> higher CVD risk (circadian disruption, Framingham literature)
- LOWER physical activity     -> higher CVD risk (inverse relationship, cardioprotective effect)
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.utils.seed import set_seed

# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
set_seed()

# ---------------------------------------------------------------------------
# Configuration — load LRS component names and weights from dataset_config
# ---------------------------------------------------------------------------

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

_DATA_CFG_PATH = os.path.join(_ROOT, "configs", "dataset_config.yaml")
with open(_DATA_CFG_PATH, "r") as _f:
    _DATA_CFG = yaml.safe_load(_f)

LRS_COMPONENTS: List[str] = _DATA_CFG["lrs_components"]
LRS_WEIGHTS: List[float] = _DATA_CFG["lrs_weights"]

logger = logging.getLogger(__name__)

# Components where HIGHER raw value = HIGHER risk  (direct scaling)
_DIRECT_RISK_COMPONENTS = frozenset({
    "smoking_packyears",
    "sedentary_hours",
    "alcohol_units",
    "sleep_regularity",   # measured as variance/irregularity — higher = worse
})

# Components where LOWER raw value = HIGHER risk   (inverted scaling)
_INVERSE_RISK_COMPONENTS = frozenset({
    "activity_mets",      # physical inactivity is the risk factor
})

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalise_component(series: pd.Series, component_name: str) -> pd.Series:
    """Min-max normalise a single LRS component to [0, 1].

    The normalisation direction depends on the clinical relationship between
    the raw measurement and cardiovascular risk:

    **Direct-risk components** (smoking_packyears, sedentary_hours,
    alcohol_units, sleep_regularity):
        Higher raw values indicate higher risk, so they map to scores
        closer to 1.  Formula: ``(x - min) / (max - min)``.
        Clinical basis: dose-response relationships established in
        Framingham, INTERHEART, and WHO guidelines show monotonically
        increasing CVD risk with higher exposure.

    **Inverse-risk components** (activity_mets):
        Lower raw values indicate higher risk (inactivity), so they map
        to scores closer to 1.  Formula: ``1 - (x - min) / (max - min)``.
        Clinical basis: the cardioprotective effect of exercise is
        well-established (AHA 2018 Physical Activity Guidelines); the
        risk factor is *lack* of activity, not presence of it.

    Parameters
    ----------
    series : pd.Series
        Raw values for one lifestyle component.
    component_name : str
        Name of the component (must be in ``LRS_COMPONENTS``).

    Returns
    -------
    pd.Series
        Normalised values in [0, 1].

    Raises
    ------
    ValueError
        If *component_name* is not a recognised LRS component.
    """
    if component_name not in _DIRECT_RISK_COMPONENTS | _INVERSE_RISK_COMPONENTS:
        raise ValueError(
            f"Unknown LRS component '{component_name}'. "
            f"Expected one of: {sorted(_DIRECT_RISK_COMPONENTS | _INVERSE_RISK_COMPONENTS)}"
        )

    s_min = series.min()
    s_max = series.max()

    # Guard against constant columns (max == min)
    if s_max == s_min:
        logger.warning(
            "Component '%s' has zero variance (min == max == %s). "
            "Returning 0.0 for all rows.",
            component_name,
            s_min,
        )
        return pd.Series(0.0, index=series.index, name=component_name)

    normalised = (series - s_min) / (s_max - s_min)

    if component_name in _INVERSE_RISK_COMPONENTS:
        # Invert: low activity -> high risk score
        normalised = 1.0 - normalised

    normalised.name = component_name
    return normalised


def compute_lrs(
    df: pd.DataFrame,
    component_columns: Optional[Dict[str, str]] = None,
) -> pd.Series:
    """Compute the Lifestyle Risk Score from five lifestyle components.

    Each component is normalised to [0, 1] via :func:`normalise_component`,
    weighted equally at 0.2, and summed to produce a composite score in [0, 1].

    Clinical justification (AGENTS.md SS7):
        The LRS aggregates modifiable lifestyle factors into a single
        feature that captures behavioural risk beyond what standard
        clinical measurements (cholesterol, BP, etc.) provide.  Equal
        weighting is used because no single lifestyle factor has been
        shown to dominate the others in the Framingham cohort after
        adjusting for clinical covariates.  This choice is documented
        explicitly so that reviewers can assess it.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the five lifestyle component columns.
    component_columns : dict, optional
        Mapping from canonical component names (as in ``dataset_config.yaml``)
        to actual column names in *df*.  If ``None``, assumes column names
        match the canonical names exactly.

    Returns
    -------
    pd.Series
        Series named ``"LRS"`` with values in [0, 1].
    """
    if component_columns is None:
        component_columns = {c: c for c in LRS_COMPONENTS}

    weighted_sum = pd.Series(0.0, index=df.index)

    for component_name, weight in zip(LRS_COMPONENTS, LRS_WEIGHTS):
        col_name = component_columns[component_name]
        if col_name not in df.columns:
            raise KeyError(
                f"Column '{col_name}' (for component '{component_name}') "
                f"not found in DataFrame. Available: {list(df.columns)}"
            )

        normalised = normalise_component(df[col_name], component_name)
        weighted_sum += weight * normalised

        logger.debug(
            "LRS component '%s' (col='%s'): weight=%.2f, "
            "normalised range=[%.3f, %.3f]",
            component_name,
            col_name,
            weight,
            normalised.min(),
            normalised.max(),
        )

    weighted_sum.name = "LRS"

    logger.info(
        "Computed LRS: mean=%.4f, std=%.4f, range=[%.4f, %.4f]",
        weighted_sum.mean(),
        weighted_sum.std(),
        weighted_sum.min(),
        weighted_sum.max(),
    )

    return weighted_sum


def append_lrs(df: pd.DataFrame, lrs_series: pd.Series) -> pd.DataFrame:
    """Append the LRS column to an existing DataFrame.

    Clinical justification:
        The LRS is added as a single composite feature rather than five
        separate lifestyle columns to avoid multicollinearity among
        correlated lifestyle behaviours (e.g. smoking and sedentary
        hours often co-occur).  A composite score also reduces
        dimensionality, which is important given the relatively small
        sample sizes of both Framingham (~3,600 after cleaning) and
        Cleveland (~300).

    Parameters
    ----------
    df : pd.DataFrame
        The feature DataFrame to augment.
    lrs_series : pd.Series
        The LRS values, as returned by :func:`compute_lrs`.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an ``LRS`` column appended.
    """
    df = df.copy()
    df["LRS"] = lrs_series.values
    logger.info("Appended LRS column. New shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Create dummy data for all five components
    np.random.seed(42)
    n = 20
    dummy = pd.DataFrame(
        {
            "sleep_regularity": np.random.uniform(0.5, 3.0, n),   # hours variance
            "activity_mets":    np.random.uniform(5.0, 50.0, n),   # MET-hours/week
            "smoking_packyears": np.random.uniform(0.0, 40.0, n),  # pack-years
            "sedentary_hours":  np.random.uniform(2.0, 14.0, n),   # hours/day
            "alcohol_units":    np.random.uniform(0.0, 30.0, n),   # units/week
        }
    )

    print("=== Raw dummy data (first 5 rows) ===")
    print(dummy.head().to_string())

    lrs = compute_lrs(dummy)

    print("\n=== Computed LRS (first 5 rows) ===")
    print(lrs.head().to_string())

    result = append_lrs(dummy, lrs)

    print("\n=== DataFrame with LRS appended (first 5 rows) ===")
    print(result.head().to_string())

    # Sanity checks
    assert lrs.min() >= 0.0, f"LRS min below 0: {lrs.min()}"
    assert lrs.max() <= 1.0, f"LRS max above 1: {lrs.max()}"
    assert "LRS" in result.columns, "LRS column not found in result"
    print("\nAll assertions passed.")
