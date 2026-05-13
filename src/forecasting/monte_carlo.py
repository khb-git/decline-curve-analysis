"""
Monte Carlo probabilistic forecasting for reserves estimation.

Generates probability distributions for EUR by sampling from the
parameter covariance matrix returned by the fitter. Reports P10/P50/P90
in the petroleum engineering convention (P90 = 90% probability of
recovering AT LEAST this volume).

References:
    SPE-PRMS (2018). Petroleum Resources Management System.
    Society of Petroleum Engineers.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from src.models.arps import DeclineParameters, eur
from src.fitting.nonlinear_regression import FitResult


@dataclass
class ProbabilisticEUR:
    """Results of probabilistic EUR analysis."""
    P90: float          # Proved reserves (90% probability of >= this)
    P50: float          # Proved + probable
    P10: float          # Proved + probable + possible
    mean: float
    median: float
    std: float
    n_samples_valid: int
    n_samples_total: int
    samples: np.ndarray
    deterministic_eur: float  # EUR from best-fit parameters


def monte_carlo_eur(
    fit: FitResult,
    q_econ: float,
    n_samples: int = 10000,
    max_years: float = 50.0,
    seed: Optional[int] = None,
) -> ProbabilisticEUR:
    """
    Generate probabilistic EUR distribution via Monte Carlo sampling.

    Samples parameters from a multivariate normal distribution defined by
    the fitter's covariance matrix. Rejects unphysical samples (negative
    rates, decline rates outside reasonable bounds, hyperbolic exponent
    out of range).

    Args:
        fit: FitResult from fitting routine
        q_econ: Economic limit rate
        n_samples: Number of Monte Carlo realizations
        max_years: Maximum forecast horizon
        seed: Random seed for reproducibility

    Returns:
        ProbabilisticEUR with P10/P50/P90 and full distribution
    """
    rng = np.random.default_rng(seed)
    params = fit.params
    pcov = fit.covariance

    # Build mean parameter vector
    if params.decline_type == "hyperbolic":
        mean_vec = np.array([params.qi, params.Di, params.b])
    else:
        mean_vec = np.array([params.qi, params.Di])

    # Sample from multivariate normal
    samples = rng.multivariate_normal(mean_vec, pcov, size=n_samples)

    # Filter physically valid samples
    if params.decline_type == "hyperbolic":
        qi_s, Di_s, b_s = samples[:, 0], samples[:, 1], samples[:, 2]
        valid = (
            (qi_s > 0)
            & (Di_s > 0.001) & (Di_s < 5.0)
            & (b_s > 0.0) & (b_s < 1.5)
        )
    else:
        qi_s, Di_s = samples[:, 0], samples[:, 1]
        valid = (qi_s > 0) & (Di_s > 0.001) & (Di_s < 5.0)

    valid_samples = samples[valid]

    # Compute EUR for each valid sample
    eurs = np.empty(len(valid_samples))
    b_fixed = params.b if params.decline_type != "hyperbolic" else None

    for i, s in enumerate(valid_samples):
        if b_fixed is not None:
            p = DeclineParameters(qi=s[0], Di=s[1], b=b_fixed)
        else:
            p = DeclineParameters(qi=s[0], Di=s[1], b=s[2])
        try:
            eurs[i] = eur(p, q_econ, max_years=max_years)
        except (ValueError, OverflowError):
            eurs[i] = np.nan

    eurs = eurs[~np.isnan(eurs)]

    # Deterministic EUR from best-fit
    det_eur = eur(params, q_econ, max_years=max_years)

    # PRMS percentile convention:
    # P90 = value at 10th percentile (90% probability of >= this)
    # P50 = median
    # P10 = value at 90th percentile (10% probability of >= this)
    return ProbabilisticEUR(
        P90=float(np.percentile(eurs, 10)),
        P50=float(np.percentile(eurs, 50)),
        P10=float(np.percentile(eurs, 90)),
        mean=float(np.mean(eurs)),
        median=float(np.median(eurs)),
        std=float(np.std(eurs)),
        n_samples_valid=len(eurs),
        n_samples_total=n_samples,
        samples=eurs,
        deterministic_eur=det_eur,
    )


def monte_carlo_production_forecast(
    fit: FitResult,
    t_forecast_days: np.ndarray,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> dict:
    """
    Generate probabilistic production forecasts at specified future times.

    Args:
        fit: FitResult from fitting routine
        t_forecast_days: Array of future times to forecast at
        n_samples: Number of Monte Carlo realizations
        seed: Random seed

    Returns:
        Dict with 'P90', 'P50', 'P10', 'mean' arrays of length len(t_forecast_days)
    """
    from src.models.arps import rate as arps_rate

    rng = np.random.default_rng(seed)
    params = fit.params

    if params.decline_type == "hyperbolic":
        mean_vec = np.array([params.qi, params.Di, params.b])
    else:
        mean_vec = np.array([params.qi, params.Di])

    samples = rng.multivariate_normal(mean_vec, fit.covariance, size=n_samples)

    rates_matrix = np.zeros((n_samples, len(t_forecast_days)))
    valid_count = 0
    b_fixed = params.b if params.decline_type != "hyperbolic" else None

    for i, s in enumerate(samples):
        if s[0] <= 0 or s[1] <= 0.001 or s[1] > 5.0:
            continue
        if b_fixed is None and (s[2] <= 0 or s[2] >= 1.5):
            continue

        if b_fixed is not None:
            p = DeclineParameters(qi=s[0], Di=s[1], b=b_fixed)
        else:
            p = DeclineParameters(qi=s[0], Di=s[1], b=s[2])
        rates_matrix[valid_count] = arps_rate(t_forecast_days, p)
        valid_count += 1

    rates_matrix = rates_matrix[:valid_count]

    return {
        "P90": np.percentile(rates_matrix, 10, axis=0),
        "P50": np.percentile(rates_matrix, 50, axis=0),
        "P10": np.percentile(rates_matrix, 90, axis=0),
        "mean": np.mean(rates_matrix, axis=0),
        "n_valid": valid_count,
    }
