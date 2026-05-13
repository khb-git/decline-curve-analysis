# Decline Curve Analysis & Probabilistic Reserves Estimation

A Python implementation of decline curve analysis (DCA) for forecasting
oil and gas well production and estimating ultimate recovery (EUR) with
probabilistic reserves reporting (P10/P50/P90) following SPE-PRMS
conventions.

## Why this project

Decline curve analysis is one of the oldest and most widely used techniques
in petroleum reservoir engineering. Arps' 1945 paper remains the foundation
of how the industry forecasts production and books reserves. Modern
unconventional wells (Bakken, Permian, Eagle Ford) break Arps' assumptions
in ways that have produced billions of dollars in overstated reserves, so
this project also implements modern decline models (Duong, stretched
exponential, modified hyperbolic) that better handle fractured horizontal
wells.

The headline analysis: given a well's historical production, produce a
probabilistic forecast of remaining recoverable volume with quantified
uncertainty.

## Features

- **Arps decline curves**: exponential, hyperbolic, and harmonic models with
  analytical cumulative production formulas
- **Modern decline models** for unconventional wells: modified hyperbolic
  (Robertson 1988), Duong (2011), stretched exponential (Valkó 2009)
- **Nonlinear regression fitting** with parameter covariance estimation
- **Monte Carlo probabilistic forecasting** producing P10/P50/P90 in SPE-PRMS
  convention
- **Economic limit calculation** with severance/ad valorem tax handling
- **Synthetic data generation** for testing and demos without external
  data dependencies

## Quick start

```python
import numpy as np
from src.io.synthetic import SyntheticWellConfig, generate_well
from src.fitting.nonlinear_regression import fit_arps
from src.forecasting.monte_carlo import monte_carlo_eur
from src.forecasting.economic_limit import EconomicParameters, economic_limit_rate

# Generate (or load) a well's production history
well = generate_well(SyntheticWellConfig(
    qi_bbl_day=850, Di_per_year=0.75, b=0.9, months_of_history=36
), seed=42)

# Fit a hyperbolic decline curve
fit = fit_arps(
    t_days=well["days_on_production"].values,
    q_observed=well["oil_bbl_day"].values,
    model_type="hyperbolic",
)
print(f"qi={fit.params.qi:.0f} bbl/d, Di={fit.params.Di:.3f}/yr, "
      f"b={fit.params.b:.2f}, R²={fit.r_squared:.3f}")

# Calculate economic limit
econ = EconomicParameters(
    oil_price_usd_bbl=70,
    opex_fixed_usd_month=3500,
    opex_variable_usd_bbl=6.50,
)
q_econ = economic_limit_rate(econ)

# Probabilistic EUR
result = monte_carlo_eur(fit, q_econ=q_econ, n_samples=10000, seed=42)
print(f"P90 (proved):                 {result.P90:>10,.0f} bbl")
print(f"P50 (proved + probable):      {result.P50:>10,.0f} bbl")
print(f"P10 (proved+prob+possible):   {result.P10:>10,.0f} bbl")
```

## Installation

```bash
git clone <repo-url>
cd decline-curve-analysis
pip install -e ".[dev,viz]"
```

## Project structure

```
decline-curve-analysis/
├── src/
│   ├── models/
│   │   ├── arps.py              # Exponential, hyperbolic, harmonic
│   │   └── modern.py            # Modified hyperbolic, Duong, stretched exp
│   ├── fitting/
│   │   └── nonlinear_regression.py
│   ├── forecasting/
│   │   ├── monte_carlo.py       # Probabilistic EUR
│   │   └── economic_limit.py
│   ├── diagnostics/
│   │   └── flow_regime.py       # (planned) Log-log, Hoffman plots
│   ├── reserves/
│   │   └── prms_reporting.py    # (planned) SPE-PRMS classification
│   └── io/
│       ├── synthetic.py         # Synthetic data generator
│       └── nd_dmr.py            # (planned) ND DMR data loader
├── tests/
├── notebooks/                   # Tutorial / demo notebooks
├── data/                        # Local data files (not checked in)
└── pyproject.toml
```

## Methodology

### Arps decline curves (1945)

The three models describe rate-time behavior:

| Model       | Rate equation                              | Cumulative                                                  | Physical regime |
|-------------|--------------------------------------------|-------------------------------------------------------------|-----------------|
| Exponential | q(t) = qᵢ·exp(−Dᵢt)                        | (qᵢ−q)/Dᵢ                                                   | Single-phase, bounded reservoir |
| Hyperbolic  | q(t) = qᵢ/(1+b·Dᵢt)^(1/b)                  | qᵢ^b/((1−b)Dᵢ)·(qᵢ^(1−b)−q^(1−b))                           | Solution-gas drive, layered reservoirs |
| Harmonic    | q(t) = qᵢ/(1+Dᵢt)                          | (qᵢ/Dᵢ)·ln(qᵢ/q)                                            | Limiting case of hyperbolic (b=1) |

The decline exponent `b` is physically bounded between 0 and 1 for
conventional reservoirs. Fitting `b > 1` to early shale data produces
unbounded EUR — a well-known failure mode addressed by the modern models.

### Probabilistic reserves (SPE-PRMS)

The Petroleum Resources Management System (SPE-PRMS) requires probabilistic
reporting:

- **P90 (proved, 1P)**: 90% probability the actual recovery will meet or
  exceed this value
- **P50 (proved + probable, 2P)**: 50% probability
- **P10 (proved + probable + possible, 3P)**: 10% probability

Note: in the PRMS convention, P90 corresponds to the **10th percentile**
of the EUR distribution (the lower-confidence end). This is the convention
implemented here.

Monte Carlo sampling draws parameter sets from the multivariate normal
distribution defined by the fit covariance matrix, calculates EUR for each
realization, and reports percentiles. Unphysical parameter combinations
(negative rates, b outside reasonable bounds) are rejected.

## Validation

The test suite includes:

- **Analytical limits**: rate at t=0 equals qᵢ; cumulative at t=0 is zero;
  monotonic rate decline; etc.
- **Parameter recovery**: synthetic data generated from known parameters
  is fitted; the fitter must recover the inputs to within ~1% on clean
  data, ~10-20% on data with 10% multiplicative noise
- **Arps (1945) Table III**: the original worked example from Arps' paper
  is reproduced to within 2%

Run the suite:

```bash
pytest -v --cov=src
```

## Roadmap

- [ ] North Dakota DMR data loader
- [ ] Bayesian fitting with PyMC (informative priors from analog wells)
- [ ] Flow regime diagnostics (log-log plots, Hoffman analysis)
- [ ] Type curve generation from a field of wells
- [ ] Streamlit interactive demo
- [ ] Pooled hierarchical model across a field

## References

- Arps, J.J. (1945). "Analysis of Decline Curves." *Transactions of the AIME*, 160(1), 228–247.
- Duong, A.N. (2011). "Rate-Decline Analysis for Fracture-Dominated Shale Reservoirs." SPE 137748.
- Valkó, P.P. (2009). "Assigning Value to Stimulation in the Barnett Shale." SPE 119369.
- Robertson, S. (1988). "Generalized Hyperbolic Equation." SPE 18731.
- SPE-PRMS (2018). *Petroleum Resources Management System*. Society of Petroleum Engineers.

## License

MIT
