from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import numpy_financial as npf


@dataclass
class FinanceInputs:
    years: int
    discount_rate: float

    capex_total: float

    price_sell_eur_per_kwh: float
    price_buy_eur_per_kwh: float
    kwh_sold_year1: float
    kwh_growth_yoy: float

    fixed_opex_year1: float
    fixed_opex_growth_yoy: float

    variable_opex_per_kwh: float  # e.g., roaming %, payment fees


@dataclass
class FinanceResult:
    cashflows: List[float]
    npv: float
    irr: float
    payback_year: float
    revenue_year1: float
    ebitda_year1: float


def build_cashflows(inp: FinanceInputs) -> Tuple[List[float], Dict[str, np.ndarray]]:
    years = int(inp.years)
    r = float(inp.discount_rate)

    kwh = np.array([inp.kwh_sold_year1 * ((1 + inp.kwh_growth_yoy) ** i) for i in range(years)])

    revenue = kwh * inp.price_sell_eur_per_kwh
    energy_cost = kwh * inp.price_buy_eur_per_kwh
    var_cost = kwh * inp.variable_opex_per_kwh

    fixed = np.array([inp.fixed_opex_year1 * ((1 + inp.fixed_opex_growth_yoy) ** i) for i in range(years)])

    ebitda = revenue - energy_cost - var_cost - fixed

    # cashflows: year0 capex outflow, then annual EBITDA (simplified, no taxes/depr.)
    cfs = [-inp.capex_total] + ebitda.tolist()
    details = {
        "kwh": kwh,
        "revenue": revenue,
        "energy_cost": energy_cost,
        "var_cost": var_cost,
        "fixed_opex": fixed,
        "ebitda": ebitda,
    }
    return cfs, details


def npv_irr_payback(cashflows: List[float], discount_rate: float) -> Tuple[float, float, float]:
    npv = float(npf.npv(discount_rate, cashflows))
    try:
        irr = float(npf.irr(cashflows))
    except Exception:
        irr = float('nan')

    # discounted payback
    cum = 0.0
    payback = float('inf')
    for t, cf in enumerate(cashflows):
        disc = cf / ((1 + discount_rate) ** t)
        prev = cum
        cum += disc
        if t > 0 and prev < 0 <= cum:
            # linear interpolation
            frac = (0 - prev) / max(disc, 1e-9)
            payback = (t - 1) + frac
            break
    return npv, irr, payback


def evaluate_finance(inp: FinanceInputs) -> Tuple[FinanceResult, Dict[str, np.ndarray]]:
    cfs, details = build_cashflows(inp)
    npv, irr, payback = npv_irr_payback(cfs, inp.discount_rate)

    revenue_y1 = float(details["revenue"][0])
    ebitda_y1 = float(details["ebitda"][0])

    return (
        FinanceResult(
            cashflows=cfs,
            npv=npv,
            irr=irr,
            payback_year=payback,
            revenue_year1=revenue_y1,
            ebitda_year1=ebitda_y1,
        ),
        details,
    )
