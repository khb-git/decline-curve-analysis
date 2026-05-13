"""
Nonlinear regression fitting for decline curve models.

Uses scipy.optimize.curve_fit (Levenberg-Marquardt with bounds) to fit
Arps and modern decline models to historical production data. Returns
both best-fit parameters and the parameter covariance matrix for use
in probabilistic forecasting.
"""

from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np
from scipy.optimize import curve_fit

from src.models.arps import DeclineParameters, rate as arps_rate


@dataclass
class FitResult:
    """Result of fitting a decline curve to production data."""
    params: DeclineParameters
    covariance: np.ndarray  # Parameter covariance matrix
    rmse: float             # Root mean squared error
    r_squared: float        # Coefficient of determination
    n_observations: int     # Number of data points used
    fit_window_days: tuple  # (start, end) of data window used


def _initial_guess(t_days: np.ndarray, q_observed: np.ndarray) -> dict:
    """Reasonable starting parameter values based on observed data."""
    # qi: peak rate over first 90 days (or first 3 points if shorter)
    n_init = min(len(q_observed), 90)
    qi_guess = float(np.max(q_observed[:n_init]))

    # Di: rough estimate from first vs last available rates over the window
    if len(q_observed) > 30 and t_days[-1] > t_days[0]:
        # Estimate using exponential assumption as starting point
        years = (t_days[-1] - t_days[0]) / 365.25
        if q_observed[-1] > 0 and years > 0:
            Di_guess = float(np.log(qi_guess / max(q_observed[-1], 0.01)) / years)
            Di_guess = max(0.05, min(Di_guess, 2.0))
        else:
            Di_guess = 0.5
    else:
        Di_guess = 0.5

    return {"qi": qi_guess, "Di": Di_guess}


def fit_arps(
    t_days: np.ndarray,
    q_observed: np.ndarray,
    model_type: str = "hyperbolic",
    b_bounds: tuple = (0.001, 1.5),
    fit_window: Optional[tuple] = None,
    weight_by_rate: bool = False,
) -> FitResult:
    """
    Fit an Arps decline curve to observed production data.

    Args:
        t_days: Time array in days from production start
        q_observed: Observed rates (bbl/day)
        model_type: One of 'exponential', 'hyperbolic', 'harmonic'
        b_bounds: Bounds on hyperbolic exponent (only used for hyperbolic)
        fit_window: Optional (start_day, end_day) to restrict fitting window.
            Useful for fitting only the boundary-dominated flow regime.
        weight_by_rate: If True, weight observations by rate magnitude
            (gives more importance to early high-rate data).

    Returns:
        FitResult with parameters, covariance, and goodness-of-fit metrics.
    """
    t = np.asarray(t_days, dtype=float)
    q = np.asarray(q_observed, dtype=float)

    # Validate inputs
    if len(t) != len(q):
        raise ValueError(f"Length mismatch: t={len(t)}, q={len(q)}")
    if len(t) < 3:
        raise ValueError(f"Need at least 3 data points, got {len(t)}")

    # Apply fit window
    if fit_window is not None:
        mask = (t >= fit_window[0]) & (t <= fit_window[1])
        t = t[mask]
        q = q[mask]
        if len(t) < 3:
            raise ValueError(f"Fit window contains only {len(t)} points")

    # Drop zero or negative rates (well shut-in periods)
    valid = q > 0
    t = t[valid]
    q = q[valid]

    init = _initial_guess(t, q)
    qi_g, Di_g = init["qi"], init["Di"]

    # Weights
    sigma = None
    if weight_by_rate:
        sigma = 1.0 / np.sqrt(q + 1.0)

    # Define model and bounds per type
    if model_type == "exponential":
        def model(t_arr, qi, Di):
            return arps_rate(t_arr, DeclineParameters(qi=qi, Di=Di, b=0.0))
        p0 = [qi_g, Di_g]
        bounds = ([1.0, 0.001], [qi_g * 10, 5.0])

    elif model_type == "harmonic":
        def model(t_arr, qi, Di):
            return arps_rate(t_arr, DeclineParameters(qi=qi, Di=Di, b=1.0))
        p0 = [qi_g, Di_g]
        bounds = ([1.0, 0.001], [qi_g * 10, 5.0])

    elif model_type == "hyperbolic":
        def model(t_arr, qi, Di, b):
            return arps_rate(t_arr, DeclineParameters(qi=qi, Di=Di, b=b))
        p0 = [qi_g, Di_g, 0.8]
        bounds = (
            [1.0, 0.001, b_bounds[0]],
            [qi_g * 10, 5.0, b_bounds[1]],
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, pcov = curve_fit(
            model, t, q, p0=p0, bounds=bounds, sigma=sigma, maxfev=20000
        )

    # Build parameter object
    if model_type == "exponential":
        params = DeclineParameters(qi=popt[0], Di=popt[1], b=0.0)
    elif model_type == "harmonic":
        params = DeclineParameters(qi=popt[0], Di=popt[1], b=1.0)
    else:
        params = DeclineParameters(qi=popt[0], Di=popt[1], b=popt[2])

    # Goodness of fit
    q_pred = arps_rate(t, params)
    residuals = q - q_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((q - np.mean(q)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return FitResult(
        params=params,
        covariance=pcov,
        rmse=rmse,
        r_squared=r_squared,
        n_observations=len(t),
        fit_window_days=(float(t[0]), float(t[-1])),
    )


def fit_all_arps(
    t_days: np.ndarray,
    q_observed: np.ndarray,
    **kwargs,
) -> dict:
    """
    Fit all three Arps models and return results for comparison.

    Returns a dict keyed by model type with FitResult values.
    Useful for model selection — usually choose the model with the
    lowest RMSE while remaining physically reasonable.
    """
    results = {}
    for mt in ["exponential", "hyperbolic", "harmonic"]:
        try:
            results[mt] = fit_arps(t_days, q_observed, model_type=mt, **kwargs)
        except Exception as e:
            results[mt] = e
    return results
