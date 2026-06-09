"""
Synthetic Lifestyle Risk Score (LRS) Column Generator — CVD Risk Predictor.

Framingham does not contain the five LRS component columns directly.
This module derives or synthetically generates them from available
Framingham columns using a seeded RNG for full reproducibility.

DOCUMENTATION OF SYNTHETIC DECISIONS (required for academic transparency):
--------------------------------------------------------------------------
1. smoking_packyears:
   Derived from `cigsPerDay` and assuming a fixed 10-year observation window
   (Framingham study epoch). Formula: (cigsPerDay / 20) * 10.
   Rationale: pack-years = (cigarettes/day ÷ 20) × years smoked.
   Clinical basis: standard epidemiological definition (AHA/WHO).

2. sedentary_hours:
   Framingham has no physical activity measurement.
   Synthetically generated as a seeded normal distribution:
   mean=8h, std=2h, clipped to [2, 16].
   Seed is fixed to model_config random_seed for reproducibility.
   Documented as synthetic in paper (Section 3 / Limitations).

3. activity_mets:
   Framingham has no MET measurement.
   Synthetically generated: mean=25, std=10, clipped to [5, 60].
   Negative correlation with sedentary_hours is introduced (r ≈ -0.4)
   to reflect real-world co-occurrence patterns.
   Seed is fixed.

4. sleep_regularity:
   Framingham has no sleep data.
   Synthetically generated: mean=1.2, std=0.6, clipped to [0.2, 4.0]
   (units: hours of nightly variance — higher = more irregular).
   Seed is fixed.

5. alcohol_units:
   Framingham has no direct alcohol measurement.
   Synthetically generated: mean=8, std=6, clipped to [0, 40]
   (units: standard units/week, UK definition).
   Seed is fixed.

All synthetic columns are generated ONCE on the full DataFrame before
the train/cal/holdout split, ensuring the scaler sees consistent
distributions across all splits.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fixed seed used for all synthetic generation — must match model_config
_DEFAULT_SEED = 42


def generate_lrs_columns(
    df: pd.DataFrame,
    seed: int = _DEFAULT_SEED,
    cigs_col: str = "cigsPerDay",
    years_observed: float = 10.0,
) -> pd.DataFrame:
    """Derive or synthetically generate the five LRS component columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or partially-processed Framingham DataFrame. Must contain
        ``cigsPerDay`` for the smoking derivation.
    seed : int
        RNG seed for reproducibility. Should match model_config random_seed.
    cigs_col : str
        Name of the cigarettes-per-day column in df.
    years_observed : float
        Assumed observation window in years for pack-year computation.

    Returns
    -------
    pd.DataFrame
        Copy of df with five new columns appended:
        ``smoking_packyears``, ``sedentary_hours``, ``activity_mets``,
        ``sleep_regularity``, ``alcohol_units``.
    """
    df = df.copy()
    n = len(df)
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. smoking_packyears — derived from cigsPerDay
    # ------------------------------------------------------------------
    if cigs_col in df.columns:
        df["smoking_packyears"] = (df[cigs_col] / 20.0) * years_observed
        logger.info(
            "smoking_packyears derived from '%s': mean=%.2f, std=%.2f",
            cigs_col,
            df["smoking_packyears"].mean(),
            df["smoking_packyears"].std(),
        )
    else:
        # Fallback: pure synthetic if cigsPerDay not present
        logger.warning(
            "'%s' not found. Generating smoking_packyears synthetically.", cigs_col
        )
        df["smoking_packyears"] = np.clip(
            rng.normal(loc=10.0, scale=8.0, size=n), 0.0, 50.0
        )

    # ------------------------------------------------------------------
    # 2. sedentary_hours — fully synthetic
    # ------------------------------------------------------------------
    df["sedentary_hours"] = np.clip(
        rng.normal(loc=8.0, scale=2.0, size=n), 2.0, 16.0
    )

    # ------------------------------------------------------------------
    # 3. activity_mets — fully synthetic, weakly anti-correlated with
    #    sedentary_hours to reflect real co-occurrence patterns
    # ------------------------------------------------------------------
    base_mets = rng.normal(loc=25.0, scale=10.0, size=n)
    # Introduce mild negative correlation: subtract a fraction of sedentary deviation
    sed_dev = df["sedentary_hours"] - df["sedentary_hours"].mean()
    activity_mets = base_mets - 0.4 * (sed_dev / df["sedentary_hours"].std()) * 10.0
    df["activity_mets"] = np.clip(activity_mets, 5.0, 60.0)

    # ------------------------------------------------------------------
    # 4. sleep_regularity — fully synthetic (hours of nightly variance)
    # ------------------------------------------------------------------
    df["sleep_regularity"] = np.clip(
        rng.normal(loc=1.2, scale=0.6, size=n), 0.2, 4.0
    )

    # ------------------------------------------------------------------
    # 5. alcohol_units — fully synthetic (UK units / week)
    # ------------------------------------------------------------------
    df["alcohol_units"] = np.clip(
        rng.normal(loc=8.0, scale=6.0, size=n), 0.0, 40.0
    )

    logger.info(
        "LRS columns generated for %d rows. Columns added: "
        "smoking_packyears, sedentary_hours, activity_mets, "
        "sleep_regularity, alcohol_units",
        n,
    )

    return df