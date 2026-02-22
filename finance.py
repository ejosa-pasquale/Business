# finance.py - flat layout finance helpers
from __future__ import annotations

import numpy as np
import pandas as pd


def build_cashflows(
    years: int,
    demand_kwh_year: float,
    annual_growth_kwh: float,
    sell_price_kwh: float,
    energy_cost_kwh: float,
    roaming_fee_pct: float,
    n_ac: int,
    n_dc: int,
    opex_ac_year: float,
    opex_dc_year: float,
    fixed_opex_year: float,
    capex_total: float,
) -> dict:
    years = int(years)
    demand_kwh_year = float(demand_kwh_year)
    annual_growth_kwh = float(annual_growth_kwh)

    kwh = []
    rev = []
    c_energy = []
    c_roam = []
    c_maint = []
    c_fixed = []
    net = []

    for y in range(years + 1):
        if y == 0:
            k = 0.0
            r = 0.0
            ce = 0.0
            cr = 0.0
            cm = 0.0
            cf = 0.0
            n = -float(capex_total)
        else:
            k = demand_kwh_year * ((1.0 + annual_growth_kwh) ** (y - 1))
            r = k * float(sell_price_kwh)
            ce = k * float(energy_cost_kwh)
            cr = r * float(roaming_fee_pct)
            cm = int(n_ac) * float(opex_ac_year) + int(n_dc) * float(opex_dc_year)
            cf = float(fixed_opex_year)
            n = r - ce - cr - cm - cf

        kwh.append(k)
        rev.append(r)
        c_energy.append(ce)
        c_roam.append(cr)
        c_maint.append(cm)
        c_fixed.append(cf)
        net.append(n)

    table = pd.DataFrame({
        "Year": list(range(years + 1)),
        "kWh": kwh,
        "Revenue": rev,
        "EnergyCost": c_energy,
        "RoamingFees": c_roam,
        "Maintenance": c_maint,
        "FixedOpex": c_fixed,
        "NetCashflow": net,
    })

    return {"table": table, "net_cashflow": net}


def npv(cashflows: list, discount_rate: float) -> float:
    dr = float(discount_rate)
    total = 0.0
    for t, cf in enumerate(cashflows):
        total += float(cf) / ((1.0 + dr) ** t)
    return total


def irr(cashflows: list):
    try:
        return float(np.irr(cashflows))
    except Exception:
        return None


def payback_year(cashflows: list):
    cum = 0.0
    for t, cf in enumerate(cashflows):
        cum += float(cf)
        if t > 0 and cum >= 0:
            prev = cum - float(cf)
            if float(cf) == 0:
                return float(t)
            frac = (0 - prev) / float(cf)
            return float(t - 1) + float(frac)
    return None
