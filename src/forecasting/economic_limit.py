"""
Economic limit calculation.

The economic limit is the production rate at which operating cash flow
equals zero. Below this rate, the well loses money and is typically
shut in. EUR is calculated to this limit, not to zero rate.
"""

from dataclasses import dataclass


@dataclass
class EconomicParameters:
    """
    Economic parameters for a producing well.

    Attributes:
        oil_price_usd_bbl: Wellhead oil price ($/bbl)
        gas_price_usd_mcf: Wellhead gas price ($/mcf)
        opex_fixed_usd_month: Fixed operating cost ($/month) — pumper,
            chemicals, base maintenance
        opex_variable_usd_bbl: Variable operating cost ($/bbl) —
            lifting, water disposal
        net_revenue_interest: NRI fraction (typical 0.75-0.85 for working
            interest owner; lower for royalty interests)
        severance_tax: Severance tax rate (typical 0.045-0.075)
        ad_valorem_tax: Ad valorem property tax rate (typical 0.025)
        gas_oil_ratio_scf_bbl: Producing GOR for combined revenue calc
    """
    oil_price_usd_bbl: float
    opex_fixed_usd_month: float
    opex_variable_usd_bbl: float
    net_revenue_interest: float = 0.80
    severance_tax: float = 0.046
    ad_valorem_tax: float = 0.025
    gas_price_usd_mcf: float = 0.0
    gas_oil_ratio_scf_bbl: float = 0.0


def economic_limit_rate(econ: EconomicParameters, days_per_month: float = 30.4) -> float:
    """
    Calculate economic limit rate in bbl/day.

    Net revenue per barrel = price * NRI * (1 - severance - ad_valorem)
    Variable cost per barrel = opex_variable
    Net margin per barrel = net revenue per bbl - variable cost per bbl

    Economic limit (bbl/day) =
        opex_fixed_per_month / (net_margin_per_bbl * days_per_month)
    """
    # Effective oil price after royalties and taxes
    net_oil = (
        econ.oil_price_usd_bbl
        * econ.net_revenue_interest
        * (1 - econ.severance_tax - econ.ad_valorem_tax)
    )

    # Add gas revenue per barrel of oil (if applicable)
    net_gas_per_bbl = 0.0
    if econ.gas_price_usd_mcf > 0 and econ.gas_oil_ratio_scf_bbl > 0:
        net_gas_per_bbl = (
            econ.gas_oil_ratio_scf_bbl / 1000  # mcf per bbl of oil
            * econ.gas_price_usd_mcf
            * econ.net_revenue_interest
            * (1 - econ.severance_tax - econ.ad_valorem_tax)
        )

    net_margin_per_bbl = net_oil + net_gas_per_bbl - econ.opex_variable_usd_bbl

    if net_margin_per_bbl <= 0:
        # Well is uneconomic at any rate
        return float("inf")

    return econ.opex_fixed_usd_month / (net_margin_per_bbl * days_per_month)
