"""
Modern decline curve models for unconventional (shale) wells.

Arps hyperbolic with b > 1 gives unbounded EUR, which is unphysical and
overstates reserves. These models handle the early-time transient flow
behavior of fractured horizontal wells more appropriately.

References:
    Duong, A.N. (2011). "Rate-Decline Analysis for Fracture-Dominated Shale
        Reservoirs." SPE 137748.
    Valko, P.P. (2009). "Assigning value to stimulation in the Barnett Shale:
        a simultaneous analysis of 7000 plus production histories and well
        completion records." SPE 119369.
    Robertson, S. (1988). "Generalized Hyperbolic Equation." SPE 18731.
"""

from dataclasses import dataclass
import numpy as np

from src.models.arps import DeclineParameters, rate as arps_rate


@dataclass
class ModifiedHyperbolicParameters:
    """
    Modified hyperbolic decline (Robertson 1988).

    Uses hyperbolic decline until the decline rate drops to a terminal
    value, then switches to exponential. Prevents unphysical EUR.

    Attributes:
        qi: Initial rate
        Di: Initial nominal decline rate (1/year)
        b: Decline exponent
        D_terminal: Terminal effective decline rate (1/year), typically 0.05-0.10
    """
    qi: float
    Di: float
    b: float
    D_terminal: float

    def __post_init__(self):
        if self.D_terminal >= self.Di:
            raise ValueError(
                "D_terminal must be less than Di; otherwise switch occurs immediately"
            )

    def switch_time_days(self) -> float:
        """Time at which decline transitions from hyperbolic to exponential."""
        # Nominal terminal decline
        D_term_nominal = -np.log(1 - self.D_terminal)
        # Time at which instantaneous hyperbolic decline equals terminal
        # D(t) = Di / (1 + b*Di*t) = D_term_nominal
        # Solve for t in years, convert to days
        t_switch_yr = (self.Di / D_term_nominal - 1) / (self.b * self.Di)
        return t_switch_yr * 365.25

    def rate_switch(self) -> float:
        """Rate at the switch time."""
        t_switch = self.switch_time_days()
        hyp_params = DeclineParameters(qi=self.qi, Di=self.Di, b=self.b)
        return float(arps_rate(np.array([t_switch]), hyp_params)[0])


def modified_hyperbolic_rate(
    t_days: np.ndarray,
    params: ModifiedHyperbolicParameters
) -> np.ndarray:
    """Calculate rate using modified hyperbolic (Robertson) decline."""
    t = np.atleast_1d(np.asarray(t_days, dtype=float))
    t_switch = params.switch_time_days()
    q_switch = params.rate_switch()

    # Hyperbolic phase
    hyp = DeclineParameters(qi=params.qi, Di=params.Di, b=params.b)
    q_hyp = arps_rate(t, hyp)

    # Exponential phase from t_switch onward
    D_term_daily = -np.log(1 - params.D_terminal) / 365.25
    t_after_switch = np.maximum(t - t_switch, 0)
    q_exp = q_switch * np.exp(-D_term_daily * t_after_switch)

    return np.where(t < t_switch, q_hyp, q_exp)


@dataclass
class DuongParameters:
    """
    Duong (2011) decline model for fracture-dominated shale wells.

    Based on the observation that q/Gp (or q/Np) plotted against time on
    log-log scales gives a straight line for transient linear flow in
    fractured reservoirs.

    Rate equation:
        q(t) = q1 * t^(-m) * exp[(a/(1-m)) * (t^(1-m) - 1)]

    Where t is in days from start of production, and q1 is the rate at t=1 day.

    Attributes:
        q1: Rate at t=1 day (bbl/day)
        a: Intercept parameter (dimensionless)
        m: Slope parameter (dimensionless, typically 1.0-1.4)
        q_inf: Optional rate at infinite time (asymptotic, often 0)
    """
    q1: float
    a: float
    m: float
    q_inf: float = 0.0


def duong_rate(t_days: np.ndarray, params: DuongParameters) -> np.ndarray:
    """Calculate rate using Duong's method."""
    t = np.atleast_1d(np.asarray(t_days, dtype=float))
    t = np.maximum(t, 1.0)  # Duong is undefined at t=0; convention is t>=1 day

    if np.isclose(params.m, 1.0):
        # Special case: log behavior
        t_func = np.log(t)
    else:
        t_func = (t ** (1 - params.m) - 1) / (1 - params.m)

    return params.q_inf + params.q1 * t ** (-params.m) * np.exp(params.a * t_func)


@dataclass
class StretchedExponentialParameters:
    """
    Stretched Exponential (Valko 2009) decline model.

    Rate equation:
        q(t) = qi * exp[-(t/tau)^n]

    Has finite EUR analytically. Two parameters (tau, n).

    Attributes:
        qi: Initial rate at t=0
        tau: Characteristic time (days)
        n: Stretching exponent (0 < n <= 1)
    """
    qi: float
    tau: float
    n: float


def stretched_exponential_rate(
    t_days: np.ndarray,
    params: StretchedExponentialParameters
) -> np.ndarray:
    """Calculate rate using stretched exponential decline."""
    t = np.atleast_1d(np.asarray(t_days, dtype=float))
    return params.qi * np.exp(-(t / params.tau) ** params.n)
