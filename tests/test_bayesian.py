"""
Tests for Bayesian decline curve fitting.

These tests verify that the Bayesian inference:
1. Imports correctly when PyMC is available
2. Recovers known parameters on synthetic data
3. Reports convergence diagnostics correctly
4. Respects physical constraints (b in [0, 1])

Bayesian inference is harder to unit-test than deterministic code — we
use coarse tolerances and check qualitative properties rather than exact
values. Tests are marked `slow` because each fit takes ~20 seconds.
"""

import numpy as np
import pytest

# Skip the entire file if PyMC isn't installed
pytest_pymc = pytest.importorskip("pymc")

from src.models.arps import DeclineParameters, rate as arps_rate
from src.fitting.bayesian import (
    fit_arps_bayesian,
    posterior_eur,
    posterior_production_forecast,
    ArpsPriors,
    BayesianFit,
)


@pytest.fixture(scope="module")
def synthetic_well_data():
    """
    Generate a clean synthetic well for repeated use across tests.
    Module-scoped so we only build the data once.
    """
    rng = np.random.default_rng(42)
    true_params = DeclineParameters(qi=800, Di=0.75, b=0.85)
    t = np.arange(30, 365.25 * 3, 30.4)
    q_clean = arps_rate(t, true_params)
    q_noisy = q_clean * rng.normal(1.0, 0.10, size=len(t))
    q_noisy = np.maximum(q_noisy, 1.0)  # Ensure positive
    return t, q_noisy, true_params


@pytest.fixture(scope="module")
def fitted(synthetic_well_data):
    """Fit Bayesian model once for use across multiple tests."""
    t, q, _ = synthetic_well_data
    return fit_arps_bayesian(
        t, q,
        n_samples=1000, n_tune=500, n_chains=2,
        progressbar=False, random_seed=42, cores=1,
    )


@pytest.mark.slow
class TestBayesianFit:
    def test_returns_bayesian_fit_object(self, fitted):
        assert isinstance(fitted, BayesianFit)

    def test_correct_number_of_samples(self, fitted):
        # 2 chains × 1000 samples = 2000
        assert fitted.n_samples == 2000

    def test_recovers_true_qi(self, fitted, synthetic_well_data):
        _, _, true_params = synthetic_well_data
        qi_median = float(np.median(fitted.posterior_qi))
        # Should be within ~15% of true value
        assert 0.85 * true_params.qi < qi_median < 1.15 * true_params.qi

    def test_recovers_true_b_within_credible_interval(self, fitted, synthetic_well_data):
        _, _, true_params = synthetic_well_data
        # True b should fall within the 90% credible interval
        b_lo = np.percentile(fitted.posterior_b, 5)
        b_hi = np.percentile(fitted.posterior_b, 95)
        assert b_lo < true_params.b < b_hi

    def test_b_constrained_to_unit_interval(self, fitted):
        """Beta prior must constrain b in [0, 1] by construction."""
        assert fitted.posterior_b.min() >= 0
        assert fitted.posterior_b.max() <= 1

    def test_chains_converged(self, fitted):
        """R-hat should be very close to 1.0 for converged chains."""
        for param in ["qi", "Di", "b"]:
            assert fitted.r_hat[param] < 1.05, f"{param} did not converge"

    def test_effective_sample_size_adequate(self, fitted):
        """ESS should be at least a few hundred for reasonable inference."""
        for param in ["qi", "Di", "b"]:
            assert fitted.ess[param] > 200, f"{param} has poor mixing"


@pytest.mark.slow
class TestPosteriorEUR:
    def test_posterior_eur_returns_valid_distribution(self, fitted):
        result = posterior_eur(fitted, q_econ=20)
        assert result["P90"] <= result["P50"] <= result["P10"]
        assert result["mean"] > 0
        assert result["std"] > 0

    def test_lower_econ_limit_gives_higher_eur(self, fitted):
        result_low = posterior_eur(fitted, q_econ=5)
        result_high = posterior_eur(fitted, q_econ=100)
        assert result_low["P50"] > result_high["P50"]


@pytest.mark.slow
class TestProductionForecast:
    def test_forecast_returns_correct_shape(self, fitted):
        t_forecast = np.linspace(0, 365.25 * 5, 50)
        result = posterior_production_forecast(fitted, t_forecast)
        assert len(result["P50"]) == 50
        assert len(result["P10"]) == 50
        assert len(result["P90"]) == 50

    def test_forecast_bands_ordered_correctly(self, fitted):
        t_forecast = np.linspace(0, 365.25 * 5, 50)
        result = posterior_production_forecast(fitted, t_forecast)
        assert np.all(result["P90"] <= result["P50"])
        assert np.all(result["P50"] <= result["P10"])

    def test_forecast_monotonically_decreasing(self, fitted):
        t_forecast = np.linspace(0, 365.25 * 10, 100)
        result = posterior_production_forecast(fitted, t_forecast)
        # P50 should be monotonically decreasing
        assert np.all(np.diff(result["P50"]) <= 0)


@pytest.mark.slow
class TestPriors:
    def test_custom_priors_constrain_inference(self, synthetic_well_data):
        """A tight Beta prior near 0.5 should pull the b estimate toward 0.5."""
        t, q, _ = synthetic_well_data

        # Tight prior centered at 0.5
        priors = ArpsPriors(
            qi_observed=q[0],
            b_alpha=10.0,  # Beta(10, 10) is centered at 0.5
            b_beta=10.0,
        )
        fitted_biased = fit_arps_bayesian(
            t, q, priors=priors,
            n_samples=500, n_tune=500, n_chains=2,
            progressbar=False, random_seed=42, cores=1,
        )
        b_median_biased = float(np.median(fitted_biased.posterior_b))
        # Should be pulled below the data-implied value (~0.85)
        assert b_median_biased < 0.8


@pytest.mark.slow
class TestInputValidation:
    def test_too_few_points_raises(self):
        t = np.array([0, 30, 60])
        q = np.array([1000, 900, 800])
        with pytest.raises(ValueError):
            fit_arps_bayesian(t, q, cores=1)
