"""
Reusable plotting utilities for decline curve analysis.

These functions produce publication-quality figures used in the demo
notebook and (optionally) any downstream Streamlit dashboard.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.models.arps import DeclineParameters, rate as arps_rate, cumulative
from src.fitting.nonlinear_regression import FitResult
from src.forecasting.monte_carlo import ProbabilisticEUR


# Consistent styling for all plots
PLOT_STYLE = {
    "data_color": "#2c3e50",
    "fit_color": "#e74c3c",
    "forecast_color": "#3498db",
    "p90_color": "#27ae60",
    "p50_color": "#f39c12",
    "p10_color": "#c0392b",
    "fan_alpha": 0.20,
    "data_marker_size": 30,
}


def plot_rate_time(
    t_days: np.ndarray,
    q_observed: np.ndarray,
    fit: Optional[FitResult] = None,
    forecast_days: float = 365.25 * 10,
    log_scale: bool = False,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> Figure:
    """
    Plot production rate vs time with optional fitted curve and forecast.

    Args:
        t_days: Historical production times (days)
        q_observed: Observed rates (bbl/day)
        fit: Optional FitResult to overlay fitted curve
        forecast_days: Total forecast horizon from t=0
        log_scale: Use log y-axis (standard for decline curves)
        ax: Optional matplotlib axes
        title: Optional title

    Returns:
        Matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    t_years = t_days / 365.25

    # Historical data
    ax.scatter(
        t_years, q_observed,
        s=PLOT_STYLE["data_marker_size"],
        color=PLOT_STYLE["data_color"],
        label="Observed",
        zorder=3,
        edgecolors="white",
        linewidths=0.5,
    )

    if fit is not None:
        # Fit window (within historical data)
        t_fit = np.linspace(0, t_days[-1], 300)
        q_fit = arps_rate(t_fit, fit.params)
        ax.plot(
            t_fit / 365.25, q_fit,
            color=PLOT_STYLE["fit_color"],
            lw=2,
            label=f"Fitted ({fit.params.decline_type})",
        )

        # Forecast beyond history
        if forecast_days > t_days[-1]:
            t_forecast = np.linspace(t_days[-1], forecast_days, 300)
            q_forecast = arps_rate(t_forecast, fit.params)
            ax.plot(
                t_forecast / 365.25, q_forecast,
                color=PLOT_STYLE["forecast_color"],
                lw=2,
                ls="--",
                label="Forecast",
            )

    ax.set_xlabel("Years on production", fontsize=11)
    ax.set_ylabel("Oil rate (bbl/day)", fontsize=11)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    return fig


def plot_cumulative(
    t_days: np.ndarray,
    q_observed: np.ndarray,
    fit: Optional[FitResult] = None,
    forecast_days: float = 365.25 * 10,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> Figure:
    """Plot cumulative production vs time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Cumulative from observations (approximate, via trapezoidal rule)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    cum_obs = np.array([
        trapz(q_observed[:i+1], t_days[:i+1]) if i > 0 else 0
        for i in range(len(t_days))
    ])

    t_years = t_days / 365.25
    ax.plot(
        t_years, cum_obs / 1000,
        "o-",
        color=PLOT_STYLE["data_color"],
        markersize=4,
        label="Observed cumulative",
    )

    if fit is not None:
        t_forecast = np.linspace(0, forecast_days, 500)
        cum_fit = cumulative(t_forecast, fit.params)
        ax.plot(
            t_forecast / 365.25, cum_fit / 1000,
            color=PLOT_STYLE["forecast_color"],
            lw=2,
            ls="--",
            label="Forecast cumulative",
        )

    ax.set_xlabel("Years on production", fontsize=11)
    ax.set_ylabel("Cumulative oil (Mbbl)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    return fig


def plot_fan_chart(
    fit: FitResult,
    t_days_hist: np.ndarray,
    q_hist: np.ndarray,
    forecast_years: float = 20,
    n_samples: int = 1000,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    seed: Optional[int] = 42,
) -> Figure:
    """
    Plot a fan chart showing probabilistic production forecast.

    The fan visualizes uncertainty bands (P10-P90 and P25-P75) around
    the median forecast, computed via Monte Carlo from the fit covariance.
    """
    from src.forecasting.monte_carlo import monte_carlo_production_forecast

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Forecast times: start of history through forecast_years
    t_forecast = np.linspace(0, forecast_years * 365.25, 400)
    forecast = monte_carlo_production_forecast(
        fit, t_forecast, n_samples=n_samples, seed=seed,
    )

    # Calculate inner band (P25/P75) for narrower band
    from src.models.arps import DeclineParameters as DP, rate as arps_rate
    rng = np.random.default_rng(seed)
    if fit.params.decline_type == "hyperbolic":
        mean_vec = np.array([fit.params.qi, fit.params.Di, fit.params.b])
    else:
        mean_vec = np.array([fit.params.qi, fit.params.Di])
    samples = rng.multivariate_normal(mean_vec, fit.covariance, size=n_samples)

    rates_matrix = np.zeros((n_samples, len(t_forecast)))
    valid = 0
    b_fixed = fit.params.b if fit.params.decline_type != "hyperbolic" else None
    for s in samples:
        if s[0] <= 0 or s[1] <= 0.001 or s[1] > 5.0:
            continue
        if b_fixed is None and (s[2] <= 0 or s[2] >= 1.5):
            continue
        if b_fixed is not None:
            p = DP(qi=s[0], Di=s[1], b=b_fixed)
        else:
            p = DP(qi=s[0], Di=s[1], b=s[2])
        rates_matrix[valid] = arps_rate(t_forecast, p)
        valid += 1
    rates_matrix = rates_matrix[:valid]

    p25 = np.percentile(rates_matrix, 25, axis=0)
    p75 = np.percentile(rates_matrix, 75, axis=0)

    t_years = t_forecast / 365.25

    # Outer band: P10-P90
    ax.fill_between(
        t_years, forecast["P90"], forecast["P10"],
        color=PLOT_STYLE["forecast_color"],
        alpha=PLOT_STYLE["fan_alpha"],
        label="P10–P90 range",
    )
    # Inner band: P25-P75
    ax.fill_between(
        t_years, p25, p75,
        color=PLOT_STYLE["forecast_color"],
        alpha=PLOT_STYLE["fan_alpha"] + 0.15,
        label="P25–P75 range",
    )
    # Median
    ax.plot(
        t_years, forecast["P50"],
        color=PLOT_STYLE["p50_color"],
        lw=2,
        label="P50 (median forecast)",
    )

    # Historical observations on top
    ax.scatter(
        t_days_hist / 365.25, q_hist,
        s=20,
        color=PLOT_STYLE["data_color"],
        label="Historical data",
        zorder=5,
        edgecolors="white",
        linewidths=0.4,
    )

    # Mark end of history
    ax.axvline(
        t_days_hist[-1] / 365.25,
        color="gray", ls=":", lw=1, alpha=0.7,
    )
    ax.text(
        t_days_hist[-1] / 365.25, ax.get_ylim()[1] * 0.95,
        " End of history",
        color="gray", fontsize=9, va="top",
    )

    ax.set_xlabel("Years on production", fontsize=11)
    ax.set_ylabel("Oil rate (bbl/day)", fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", framealpha=0.95)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    return fig


def plot_eur_distribution(
    result: ProbabilisticEUR,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    clip_at_percentile: float = 99.0,
) -> Figure:
    """
    Plot Monte Carlo EUR distribution as histogram with P10/P50/P90 markers.

    Args:
        result: Probabilistic EUR results
        ax: Optional matplotlib axes
        title: Optional title
        clip_at_percentile: Clip x-axis at this percentile of the EUR
            distribution to avoid wasted space from long upper tails.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Histogram of EUR samples (in thousands of barrels)
    eurs_mbbl = result.samples / 1000
    upper_clip = np.percentile(eurs_mbbl, clip_at_percentile)

    ax.hist(
        eurs_mbbl,
        bins=60,
        range=(eurs_mbbl.min(), upper_clip),
        color=PLOT_STYLE["forecast_color"],
        alpha=0.65,
        edgecolor="white",
        linewidth=0.5,
    )

    # Percentile markers
    percentile_lines = [
        (result.P90 / 1000, "P90 (Proved)", PLOT_STYLE["p90_color"]),
        (result.P50 / 1000, "P50 (2P)", PLOT_STYLE["p50_color"]),
        (result.P10 / 1000, "P10 (3P)", PLOT_STYLE["p10_color"]),
    ]
    for val, label, color in percentile_lines:
        ax.axvline(val, color=color, lw=2.5, label=f"{label}: {val:,.0f} Mbbl")

    # Deterministic EUR marker
    ax.axvline(
        result.deterministic_eur / 1000,
        color="black", lw=1.5, ls="--",
        label=f"Deterministic: {result.deterministic_eur/1000:,.0f} Mbbl",
    )

    ax.set_xlabel("EUR (thousand barrels)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    return fig


def plot_model_comparison(
    t_days: np.ndarray,
    q_observed: np.ndarray,
    results: dict,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> Figure:
    """Plot all three Arps models fitted to the same data for comparison."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    t_years = t_days / 365.25
    ax.scatter(
        t_years, q_observed,
        s=PLOT_STYLE["data_marker_size"],
        color=PLOT_STYLE["data_color"],
        label="Observed",
        zorder=5,
        edgecolors="white",
        linewidths=0.5,
    )

    colors = {
        "exponential": "#3498db",
        "hyperbolic": "#e74c3c",
        "harmonic": "#27ae60",
    }
    t_fit = np.linspace(0, t_days[-1], 300)
    for name, fit_result in results.items():
        if not hasattr(fit_result, "params"):
            continue
        q_fit = arps_rate(t_fit, fit_result.params)
        ax.plot(
            t_fit / 365.25, q_fit,
            color=colors[name], lw=2,
            label=f"{name.capitalize()} (R²={fit_result.r_squared:.3f}, "
                  f"RMSE={fit_result.rmse:.1f})",
        )

    ax.set_xlabel("Years on production", fontsize=11)
    ax.set_ylabel("Oil rate (bbl/day)", fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    return fig
