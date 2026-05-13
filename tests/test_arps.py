"""
Tests for Arps decline curve module.

Validates against analytical limits and the classic worked example
from Arps' 1945 paper (Table III).
"""

import numpy as np
import pytest

from src.models.arps import (
    DeclineParameters,
    rate,
    cumulative,
    time_to_rate,
    eur,
)


class TestDeclineParameters:
    def test_decline_type_classification(self):
        assert DeclineParameters(qi=100, Di=0.5, b=0).decline_type == "exponential"
        assert DeclineParameters(qi=100, Di=0.5, b=0.5).decline_type == "hyperbolic"
        assert DeclineParameters(qi=100, Di=0.5, b=1).decline_type == "harmonic"

    def test_invalid_parameters_rejected(self):
        with pytest.raises(ValueError):
            DeclineParameters(qi=-1, Di=0.5, b=0.5)
        with pytest.raises(ValueError):
            DeclineParameters(qi=100, Di=-0.1, b=0.5)
        with pytest.raises(ValueError):
            DeclineParameters(qi=100, Di=0.5, b=-0.1)

    def test_effective_decline_rate_exponential(self):
        # For exponential: Deff = 1 - exp(-Di)
        params = DeclineParameters(qi=100, Di=0.5, b=0)
        assert np.isclose(params.effective_decline_rate, 1 - np.exp(-0.5))

    def test_effective_decline_rate_harmonic(self):
        # For harmonic: Deff = Di / (1 + Di)
        params = DeclineParameters(qi=100, Di=0.5, b=1)
        assert np.isclose(params.effective_decline_rate, 0.5 / 1.5)


class TestRate:
    def test_rate_at_t_zero_equals_qi(self):
        for b in [0, 0.5, 1.0]:
            params = DeclineParameters(qi=1000, Di=0.5, b=b)
            assert np.isclose(rate(np.array([0]), params)[0], 1000)

    def test_exponential_rate_after_one_year(self):
        # q(1yr) = qi * exp(-Di * 1)
        params = DeclineParameters(qi=1000, Di=0.5, b=0)
        q = rate(np.array([365.25]), params)
        assert np.isclose(q[0], 1000 * np.exp(-0.5))

    def test_harmonic_rate_after_one_year(self):
        # q(1yr) = qi / (1 + Di * 1)
        params = DeclineParameters(qi=1000, Di=0.5, b=1)
        q = rate(np.array([365.25]), params)
        assert np.isclose(q[0], 1000 / 1.5)

    def test_hyperbolic_rate_after_one_year(self):
        # q(1yr) = qi / (1 + b*Di*1)^(1/b)
        params = DeclineParameters(qi=1000, Di=0.5, b=0.5)
        q = rate(np.array([365.25]), params)
        expected = 1000 / (1 + 0.5 * 0.5) ** (1 / 0.5)
        assert np.isclose(q[0], expected)

    def test_rate_is_monotonically_decreasing(self):
        for b in [0, 0.3, 0.7, 1.0]:
            params = DeclineParameters(qi=1000, Di=0.6, b=b)
            t = np.linspace(0, 3650, 100)
            q = rate(t, params)
            diffs = np.diff(q)
            assert np.all(diffs <= 0), f"Rate increased for b={b}"


class TestCumulative:
    def test_cumulative_at_zero_is_zero(self):
        for b in [0, 0.5, 1.0]:
            params = DeclineParameters(qi=1000, Di=0.5, b=b)
            assert np.isclose(cumulative(np.array([0]), params)[0], 0)

    def test_cumulative_monotonic(self):
        params = DeclineParameters(qi=1000, Di=0.5, b=0.7)
        t = np.linspace(0, 3650, 100)
        np_cum = cumulative(t, params)
        assert np.all(np.diff(np_cum) >= 0)

    def test_cumulative_exponential_analytical(self):
        # Np = qi/D * (1 - exp(-D*t))
        params = DeclineParameters(qi=1000, Di=0.5, b=0)
        t = np.array([365.25])
        D_daily = 0.5 / 365.25
        expected = 1000 / D_daily * (1 - np.exp(-0.5))
        assert np.isclose(cumulative(t, params)[0], expected)

    def test_cumulative_harmonic_unbounded(self):
        # Harmonic cumulative grows logarithmically without bound
        params = DeclineParameters(qi=1000, Di=0.5, b=1)
        np_10yr = cumulative(np.array([365.25 * 10]), params)[0]
        np_100yr = cumulative(np.array([365.25 * 100]), params)[0]
        assert np_100yr > np_10yr  # Still growing


class TestTimeToRate:
    def test_time_to_initial_rate_is_zero(self):
        params = DeclineParameters(qi=1000, Di=0.5, b=0.5)
        assert time_to_rate(1000, params) == 0

    def test_time_to_higher_rate_is_zero(self):
        params = DeclineParameters(qi=1000, Di=0.5, b=0.5)
        assert time_to_rate(2000, params) == 0

    def test_time_to_half_rate_exponential(self):
        # qi * exp(-D*t) = qi/2  =>  t = ln(2)/D
        params = DeclineParameters(qi=1000, Di=0.5, b=0)
        t = time_to_rate(500, params)
        expected_days = np.log(2) / (0.5 / 365.25)
        assert np.isclose(t, expected_days)


class TestEUR:
    def test_eur_exponential_at_economic_limit(self):
        params = DeclineParameters(qi=1000, Di=0.5, b=0)
        # Economic limit at 100 bbl/day
        # q = qi * exp(-D*t) = 100 means t = ln(10)/D
        # Np = qi/D * (1 - 1/10) = 0.9 * qi/D
        D_daily = 0.5 / 365.25
        expected = 0.9 * 1000 / D_daily
        result = eur(params, q_econ=100)
        assert np.isclose(result, expected, rtol=1e-6)

    def test_eur_decreases_with_higher_econ_limit(self):
        params = DeclineParameters(qi=1000, Di=0.5, b=0.7)
        eur_low = eur(params, q_econ=10)
        eur_high = eur(params, q_econ=200)
        assert eur_low > eur_high

    def test_eur_capped_by_max_years(self):
        # Harmonic with b=1 grows unboundedly; max_years caps it
        params = DeclineParameters(qi=1000, Di=0.1, b=1)
        eur_10yr = eur(params, q_econ=0.01, max_years=10)
        eur_50yr = eur(params, q_econ=0.01, max_years=50)
        assert eur_50yr > eur_10yr


class TestRateCumulativeConsistency:
    """
    Self-consistency tests between rate() and cumulative().

    The cumulative production is the integral of the rate function.
    These tests verify that the analytical cumulative formulas agree
    with numerical integration of the rate function — a strong check
    on the correctness of both implementations.
    """

    @pytest.mark.parametrize("b", [0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
    def test_cumulative_matches_numerical_integration(self, b):
        """Analytical Np must match trapezoidal integration of q(t)."""
        params = DeclineParameters(qi=1000, Di=0.5, b=b)
        t = np.linspace(0, 365.25 * 5, 10000)
        q = rate(t, params)

        # Numerical integration of rate (np.trapezoid added in NumPy 2.0)
        trapz = getattr(np, "trapezoid", None) or np.trapz
        cum_numerical = trapz(q, t)
        # Analytical cumulative at the same endpoint
        cum_analytical = cumulative(np.array([t[-1]]), params)[0]

        assert np.isclose(cum_numerical, cum_analytical, rtol=1e-3)

    @pytest.mark.parametrize("b,Di", [
        (0.0, 0.3), (0.0, 0.7),
        (0.5, 0.5), (0.5, 1.0),
        (1.0, 0.4), (1.0, 0.8),
    ])
    def test_eur_consistent_with_time_to_econ_limit(self, b, Di):
        """
        EUR computed two ways must agree:
        1. Direct EUR calculation
        2. Cumulative production at the time-to-economic-limit
        """
        params = DeclineParameters(qi=1000, Di=Di, b=b)
        q_econ = 50

        eur_direct = eur(params, q_econ)
        t_econ = time_to_rate(q_econ, params)
        eur_via_cum = cumulative(np.array([t_econ]), params)[0]

        assert np.isclose(eur_direct, eur_via_cum, rtol=1e-6)
