"""
Arps decline curve models (1945).

Implements exponential (b=0), hyperbolic (0<b<1), and harmonic (b=1) decline
curves used to forecast production from oil and gas wells.

References:
    Arps, J.J. (1945). "Analysis of Decline Curves". Transactions of the AIME,
    160(1), 228-247.
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np


DeclineType = Literal["exponential", "hyperbolic", "harmonic"]


@dataclass
class DeclineParameters:
    """
    Parameters for an Arps decline curve.

    Attributes:
        qi: Initial production rate (bbl/day for oil, mcf/day for gas)
        Di: Initial nominal decline rate (1/year). For exponential decline,
            this equals the effective decline rate. For hyperbolic/harmonic,
            nominal and effective differ.
        b: Decline exponent. 0 = exponential, 0<b<1 = hyperbolic, 1 = harmonic.
            Values >1 are unphysical for bounded reservoirs but sometimes
            fit early shale data.
        t_start: Time offset (days) from production start. Default 0.
    """
    qi: float
    Di: float
    b: float
    t_start: float = 0.0

    def __post_init__(self):
        if self.qi <= 0:
            raise ValueError(f"qi must be positive, got {self.qi}")
        if self.Di <= 0:
            raise ValueError(f"Di must be positive, got {self.Di}")
        if self.b < 0:
            raise ValueError(f"b must be non-negative, got {self.b}")

    @property
    def decline_type(self) -> DeclineType:
        if np.isclose(self.b, 0):
            return "exponential"
        elif np.isclose(self.b, 1):
            return "harmonic"
        else:
            return "hyperbolic"

    @property
    def Di_daily(self) -> float:
        """Initial decline rate in 1/day."""
        return self.Di / 365.25

    @property
    def effective_decline_rate(self) -> float:
        """
        Effective annual decline rate (fraction per year).

        For exponential: D_eff = 1 - exp(-Di)
        For hyperbolic: D_eff = 1 - (1 + b*Di)^(-1/b)
        For harmonic: D_eff = Di / (1 + Di)
        """
        if self.decline_type == "exponential":
            return 1 - np.exp(-self.Di)
        elif self.decline_type == "harmonic":
            return self.Di / (1 + self.Di)
        else:
            return 1 - (1 + self.b * self.Di) ** (-1 / self.b)


def rate(t_days: np.ndarray, params: DeclineParameters) -> np.ndarray:
    """
    Calculate production rate at time t using Arps equation.

    Args:
        t_days: Time array (days from production start)
        params: Decline parameters

    Returns:
        Rate array in same units as params.qi
    """
    t = np.atleast_1d(np.asarray(t_days, dtype=float))
    t_eff = np.maximum(t - params.t_start, 0)

    if params.decline_type == "exponential":
        q = params.qi * np.exp(-params.Di_daily * t_eff)
    elif params.decline_type == "harmonic":
        q = params.qi / (1 + params.Di_daily * t_eff)
    else:  # hyperbolic
        q = params.qi / (1 + params.b * params.Di_daily * t_eff) ** (1 / params.b)

    return q


def cumulative(t_days: np.ndarray, params: DeclineParameters) -> np.ndarray:
    """
    Calculate cumulative production from t=0 to t_days.

    Args:
        t_days: Time array (days from production start)
        params: Decline parameters

    Returns:
        Cumulative production array in same units as params.qi * days
    """
    t = np.atleast_1d(np.asarray(t_days, dtype=float))
    t_eff = np.maximum(t - params.t_start, 0)
    D = params.Di_daily

    if params.decline_type == "exponential":
        Np = (params.qi / D) * (1 - np.exp(-D * t_eff))
    elif params.decline_type == "harmonic":
        Np = (params.qi / D) * np.log(1 + D * t_eff)
    else:  # hyperbolic
        q_t = rate(t_days, params)
        Np = (params.qi ** params.b / ((1 - params.b) * D)) * \
             (params.qi ** (1 - params.b) - q_t ** (1 - params.b))

    return Np


def time_to_rate(q_target: float, params: DeclineParameters) -> float:
    """
    Calculate time required to decline from qi to q_target.

    Args:
        q_target: Target rate (must be less than qi)
        params: Decline parameters

    Returns:
        Time in days. Returns np.inf if target cannot be reached.
    """
    if q_target >= params.qi:
        return 0.0
    if q_target <= 0:
        return np.inf

    D = params.Di_daily

    if params.decline_type == "exponential":
        return float(np.log(params.qi / q_target) / D)
    elif params.decline_type == "harmonic":
        return float((params.qi / q_target - 1) / D)
    else:  # hyperbolic
        return float(((params.qi / q_target) ** params.b - 1) / (params.b * D))


def eur(params: DeclineParameters, q_econ: float,
        max_years: float = 50.0) -> float:
    """
    Calculate Estimated Ultimate Recovery to economic limit.

    EUR is the cumulative production from t=0 to the time at which the
    production rate reaches the economic limit q_econ.

    Args:
        params: Decline parameters
        q_econ: Economic limit rate (rate below which the well is uneconomic)
        max_years: Maximum forecast horizon to prevent unbounded forecasts
            from hyperbolic models with b>=1.

    Returns:
        EUR in same units as qi * days (typically barrels)
    """
    t_econ = time_to_rate(q_econ, params)
    t_max = max_years * 365.25
    t_eur = min(t_econ, t_max)
    return float(cumulative(np.array([t_eur]), params)[0])
