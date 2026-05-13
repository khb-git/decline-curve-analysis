"""
Synthetic production data generator.

Generates realistic-looking well production histories for testing the
fitting and forecasting modules without requiring external data downloads.
Based on typical Bakken horizontal well behavior.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from src.models.arps import DeclineParameters, rate as arps_rate


@dataclass
class SyntheticWellConfig:
    """Configuration for generating a synthetic well."""
    well_name: str = "SYNTHETIC-001"
    formation: str = "BAKKEN"
    lateral_length_ft: float = 10000
    qi_bbl_day: float = 800.0
    Di_per_year: float = 0.75
    b: float = 0.95
    months_of_history: int = 36
    noise_level: float = 0.10        # Fractional noise on rates
    shut_in_probability: float = 0.02 # Probability of zero production in a month
    decline_param_drift: float = 0.0  # Optional drift in b over time


def generate_well(
    config: SyntheticWellConfig,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic monthly production history.

    Returns:
        DataFrame with columns:
            month_index, days_on_production, oil_bbl_month, oil_bbl_day
    """
    rng = np.random.default_rng(seed)

    # Decline parameters
    params = DeclineParameters(
        qi=config.qi_bbl_day,
        Di=config.Di_per_year,
        b=config.b,
    )

    # Monthly samples — use the midpoint of each month for the rate
    days_per_month = 30.4
    midpoints = np.arange(config.months_of_history) * days_per_month + days_per_month / 2

    rates = arps_rate(midpoints, params)

    # Add multiplicative noise
    noise = rng.normal(loc=1.0, scale=config.noise_level, size=len(rates))
    rates_noisy = np.maximum(rates * noise, 0.0)

    # Random shut-ins
    shut_ins = rng.random(len(rates)) < config.shut_in_probability
    rates_noisy[shut_ins] = 0.0

    # Monthly volumes
    monthly_volumes = rates_noisy * days_per_month

    df = pd.DataFrame({
        "month_index": np.arange(config.months_of_history),
        "days_on_production": (np.arange(config.months_of_history) + 1) * days_per_month,
        "oil_bbl_month": monthly_volumes,
        "oil_bbl_day": rates_noisy,
        "well_name": config.well_name,
        "formation": config.formation,
        "lateral_length_ft": config.lateral_length_ft,
    })

    return df


def generate_field(
    n_wells: int = 50,
    base_config: Optional[SyntheticWellConfig] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic field of wells with parameter variation.

    Useful for testing type curve generation and field-level forecasting.
    """
    rng = np.random.default_rng(seed)
    base = base_config or SyntheticWellConfig()

    all_wells = []
    for i in range(n_wells):
        # Vary qi and Di across wells
        config = SyntheticWellConfig(
            well_name=f"{base.formation}-{i+1:03d}",
            formation=base.formation,
            lateral_length_ft=base.lateral_length_ft * rng.uniform(0.85, 1.15),
            qi_bbl_day=base.qi_bbl_day * rng.uniform(0.5, 1.8),
            Di_per_year=base.Di_per_year * rng.uniform(0.7, 1.3),
            b=np.clip(base.b * rng.uniform(0.8, 1.1), 0.1, 1.4),
            months_of_history=base.months_of_history,
            noise_level=base.noise_level,
        )
        df = generate_well(config, seed=int(rng.integers(0, 1_000_000)))
        all_wells.append(df)

    return pd.concat(all_wells, ignore_index=True)
