"""
Bayesian fitting for Arps decline curves using PyMC.

Standard nonlinear regression fitting (in `nonlinear_regression.py`)
produces a point estimate plus a covariance matrix — uncertainty is
quantified via the local quadratic approximation around the MLE.

The Bayesian approach replaces this with explicit posterior distributions
over parameters. Two practical advantages for decline curve analysis:

1. **Informative priors from analog wells.** If neighboring wells in
   the same play have b values clustered around 0.85, we can use a
   Beta(8, 2) prior centered there. This dramatically tightens the
   posterior for short-history wells.

2. **Physical constraints by construction.** Priors on b can be bounded
   to [0, 1], preventing the b > 1 pathology that inflates EURs in
   shale wells fit by ordinary least squares.

This module uses PyMC's NUTS sampler (No-U-Turn Sampler — the modern
default Hamiltonian Monte Carlo algorithm). For a typical well with
36 months of monthly data, 2000 posterior samples per chain across
4 chains takes 30-60 seconds.

References:
    Gong, X. et al. (2014). "Bayesian Probabilistic Decline-Curve Analysis
        Reliably Quantifies Uncertainty in Shale-Well-Production Forecasts."
        SPE Journal 19(6).
"""

from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np

# PyMC imports are heavy; do them at module level but guard against import errors
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None

from src.models.arps import DeclineParameters


@dataclass
class ArpsPriors:
    """
    Prior distributions for Arps decline parameters.

    Defaults are reasonable for unconventional (shale) wells. For
    conventional wells, you might shift `b_alpha`/`b_beta` toward lower
    values and tighten the qi prior.

    Attributes:
        qi_observed: Observed initial rate from data (used to set qi prior scale)
        qi_log_sd: Log-normal standard deviation on qi prior (0.3 = ~30% uncertainty)
        Di_mu: Mean of Di prior (1/year)
        Di_sigma: Standard deviation of Di prior
        b_alpha, b_beta: Beta distribution parameters for b ∈ [0, 1].
            Default Beta(8, 2) is centered at ~0.8 with std ~0.12,
            appropriate for typical unconventional wells.
        sigma_observed: Observed scatter in production data (sets noise prior)
    """
    qi_observed: float
    qi_log_sd: float = 0.3
    Di_mu: float = 0.7
    Di_sigma: float = 0.4
    b_alpha: float = 8.0
    b_beta: float = 2.0
    sigma_observed: float = 50.0


@dataclass
class BayesianFit:
    """Result of Bayesian fitting."""
    posterior_qi: np.ndarray         # 1D array of posterior samples
    posterior_Di: np.ndarray
    posterior_b: np.ndarray
    posterior_sigma: np.ndarray      # Observation noise
    posterior_median_params: DeclineParameters
    n_samples: int
    r_hat: dict                      # Convergence diagnostic per parameter
    ess: dict                        # Effective sample size per parameter
    idata: object                    # ArviZ InferenceData for full diagnostics


def fit_arps_bayesian(
    t_days: np.ndarray,
    q_observed: np.ndarray,
    priors: Optional[ArpsPriors] = None,
    n_samples: int = 2000,
    n_tune: int = 1000,
    n_chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
    progressbar: bool = True,
    cores: Optional[int] = None,
) -> BayesianFit:
    """
    Fit a hyperbolic Arps decline curve using Bayesian inference.

    The model assumes lognormal observation noise (production rates are
    strictly positive and typically have multiplicative measurement error).

    Args:
        t_days: Time array (days from production start)
        q_observed: Observed production rates (bbl/day, must be positive)
        priors: ArpsPriors instance, or None for defaults inferred from data
        n_samples: Posterior samples per chain
        n_tune: NUTS tuning steps per chain
        n_chains: Number of parallel MCMC chains
        target_accept: NUTS target acceptance probability (0.8-0.95 typical;
            higher for difficult posteriors)
        random_seed: Random seed
        progressbar: Show progress bar during sampling
        cores: Number of CPU cores for parallel chains. None lets PyMC
            decide; pass 1 if you hit BLAS/process issues.

    Returns:
        BayesianFit with posterior samples and convergence diagnostics
    """
    if not PYMC_AVAILABLE:
        raise ImportError(
            "PyMC is required for Bayesian fitting. "
            "Install with: pip install pymc arviz"
        )

    t = np.asarray(t_days, dtype=float)
    q = np.asarray(q_observed, dtype=float)

    # Filter out zero/negative observations
    valid = q > 0
    t = t[valid]
    q = q[valid]

    if len(t) < 5:
        raise ValueError(f"Need at least 5 positive observations, got {len(t)}")

    # Auto-construct priors from data if not provided
    if priors is None:
        qi_observed = float(np.max(q[:min(len(q), 90)]))
        priors = ArpsPriors(
            qi_observed=qi_observed,
            sigma_observed=float(np.std(q) * 0.5),
        )

    with pm.Model() as model:
        # Priors
        # qi: lognormal because production rates are strictly positive
        log_qi_mu = np.log(priors.qi_observed)
        qi = pm.LogNormal("qi", mu=log_qi_mu, sigma=priors.qi_log_sd)

        # Di: truncated normal (positive)
        Di = pm.TruncatedNormal(
            "Di",
            mu=priors.Di_mu,
            sigma=priors.Di_sigma,
            lower=0.01,
            upper=5.0,
        )

        # b: Beta(alpha, beta) restricts to [0, 1] — physically meaningful range
        b = pm.Beta("b", alpha=priors.b_alpha, beta=priors.b_beta)

        # Observation noise (lognormal scatter)
        sigma = pm.HalfNormal("sigma", sigma=priors.sigma_observed)

        # Decline curve: vectorized hyperbolic Arps in PyMC tensors
        Di_daily = Di / 365.25
        q_predicted = qi / (1 + b * Di_daily * t) ** (1 / b)

        # Likelihood: lognormal observations
        # Using log-transformed rates avoids issues with the bound at zero
        log_q_predicted = pm.math.log(q_predicted)
        log_sigma_normalized = sigma / qi  # Scale-normalize the noise
        pm.LogNormal("q_obs", mu=log_q_predicted, sigma=log_sigma_normalized, observed=q)

        # Sample using NUTS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sample_kwargs = dict(
                draws=n_samples,
                tune=n_tune,
                chains=n_chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=progressbar,
                return_inferencedata=True,
            )
            if cores is not None:
                sample_kwargs["cores"] = cores
            idata = pm.sample(**sample_kwargs)

    # Extract posterior samples
    posterior = idata.posterior
    qi_samples = posterior["qi"].values.flatten()
    Di_samples = posterior["Di"].values.flatten()
    b_samples = posterior["b"].values.flatten()
    sigma_samples = posterior["sigma"].values.flatten()

    # Posterior median (best point estimate from a Bayesian fit)
    median_params = DeclineParameters(
        qi=float(np.median(qi_samples)),
        Di=float(np.median(Di_samples)),
        b=float(np.median(b_samples)),
    )

    # Convergence diagnostics
    summary = az.summary(idata, var_names=["qi", "Di", "b", "sigma"])
    r_hat = summary["r_hat"].to_dict()
    ess = summary["ess_bulk"].to_dict()

    return BayesianFit(
        posterior_qi=qi_samples,
        posterior_Di=Di_samples,
        posterior_b=b_samples,
        posterior_sigma=sigma_samples,
        posterior_median_params=median_params,
        n_samples=len(qi_samples),
        r_hat=r_hat,
        ess=ess,
        idata=idata,
    )


def posterior_eur(
    bfit: BayesianFit,
    q_econ: float,
    max_years: float = 50.0,
) -> dict:
    """
    Compute EUR distribution from Bayesian posterior samples.

    Unlike the frequentist Monte Carlo (which samples parameters from
    a multivariate normal around the MLE), this iterates directly over
    the posterior samples, giving the *true* posterior EUR distribution
    consistent with the prior + data.

    Args:
        bfit: BayesianFit result
        q_econ: Economic limit rate (bbl/day)
        max_years: Forecast horizon cap

    Returns:
        Dict with P10/P50/P90, mean, std, samples (in PRMS convention).
    """
    from src.models.arps import eur

    eurs = np.empty(bfit.n_samples)
    for i in range(bfit.n_samples):
        p = DeclineParameters(
            qi=bfit.posterior_qi[i],
            Di=bfit.posterior_Di[i],
            b=bfit.posterior_b[i],
        )
        try:
            eurs[i] = eur(p, q_econ, max_years=max_years)
        except (ValueError, OverflowError):
            eurs[i] = np.nan

    eurs = eurs[~np.isnan(eurs)]

    return {
        # PRMS convention: P90 = 10th percentile (90% chance of >=)
        "P90": float(np.percentile(eurs, 10)),
        "P50": float(np.percentile(eurs, 50)),
        "P10": float(np.percentile(eurs, 90)),
        "mean": float(np.mean(eurs)),
        "median": float(np.median(eurs)),
        "std": float(np.std(eurs)),
        "samples": eurs,
        "n_samples": len(eurs),
    }


def posterior_production_forecast(
    bfit: BayesianFit,
    t_forecast_days: np.ndarray,
    n_samples_to_use: Optional[int] = None,
) -> dict:
    """
    Compute probabilistic production forecast from posterior samples.

    Args:
        bfit: BayesianFit result
        t_forecast_days: Times to forecast at (days from t=0)
        n_samples_to_use: Optionally subsample the posterior for speed.
            None uses all samples.

    Returns:
        Dict with 'P90', 'P50', 'P10', 'mean' arrays.
    """
    from src.models.arps import rate as arps_rate

    n_avail = bfit.n_samples
    if n_samples_to_use is None or n_samples_to_use > n_avail:
        idx = np.arange(n_avail)
    else:
        idx = np.random.default_rng(42).choice(n_avail, n_samples_to_use, replace=False)

    rates = np.zeros((len(idx), len(t_forecast_days)))
    for i, j in enumerate(idx):
        p = DeclineParameters(
            qi=bfit.posterior_qi[j],
            Di=bfit.posterior_Di[j],
            b=bfit.posterior_b[j],
        )
        rates[i] = arps_rate(t_forecast_days, p)

    return {
        "P90": np.percentile(rates, 10, axis=0),
        "P50": np.percentile(rates, 50, axis=0),
        "P10": np.percentile(rates, 90, axis=0),
        "mean": np.mean(rates, axis=0),
    }
