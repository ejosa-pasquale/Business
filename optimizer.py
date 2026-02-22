from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math

from finance import FinanceInputs, evaluate_finance


@dataclass
class TechCost:
    capex_per_charger: float
    fixed_opex_per_charger_year: float
    connectors: int
    power_kw: float


@dataclass
class OptimizationInputs:
    # Demand split
    kwh_ac_year1: float
    kwh_dc_year1: float

    # Constraints
    power_available_kw: float
    capex_budget: float

    # Financial common
    years: int
    discount_rate: float
    price_sell_eur_per_kwh: float
    price_buy_eur_per_kwh: float
    kwh_growth_yoy: float
    variable_opex_per_kwh: float
    fixed_opex_overhead_year1: float
    fixed_opex_overhead_growth_yoy: float

    # Candidate ranges
    max_ac: int
    max_dc: int


@dataclass
class OptimizationResult:
    n_ac: int
    n_dc: int
    capex: float
    power_installed_kw: float
    npv: float
    irr: float
    payback: float
    notes: str


def optimize_mix(
    inp: OptimizationInputs,
    ac: TechCost,
    dc: TechCost,
) -> Tuple[OptimizationResult, List[OptimizationResult]]:
    results: List[OptimizationResult] = []

    best: OptimizationResult | None = None

    for n_ac in range(0, inp.max_ac + 1):
        for n_dc in range(0, inp.max_dc + 1):
            if n_ac == 0 and n_dc == 0:
                continue

            power = n_ac * ac.power_kw + n_dc * dc.power_kw
            if power > inp.power_available_kw + 1e-9:
                continue

            capex = n_ac * ac.capex_per_charger + n_dc * dc.capex_per_charger
            if capex > inp.capex_budget + 1e-9:
                continue

            # If you have demand split, assume AC serves kwh_ac and DC serves kwh_dc.
            # If you install zero of one tech, re-route that demand to the other (with a penalty).
            kwh_ac = inp.kwh_ac_year1
            kwh_dc = inp.kwh_dc_year1
            notes = ""

            if n_ac == 0 and kwh_ac > 0:
                # reroute to DC but assume 15% drop due to mismatch
                kwh_dc += 0.85 * kwh_ac
                notes += "AC=0: domanda AC riversata su DC (−15%). "
                kwh_ac = 0
            if n_dc == 0 and kwh_dc > 0:
                kwh_ac += 0.85 * kwh_dc
                notes += "DC=0: domanda DC riversata su AC (−15%). "
                kwh_dc = 0

            kwh_total = kwh_ac + kwh_dc

            fixed_opex_year1 = (
                inp.fixed_opex_overhead_year1
                + n_ac * ac.fixed_opex_per_charger_year
                + n_dc * dc.fixed_opex_per_charger_year
            )

            fin_inp = FinanceInputs(
                years=inp.years,
                discount_rate=inp.discount_rate,
                capex_total=capex,
                price_sell_eur_per_kwh=inp.price_sell_eur_per_kwh,
                price_buy_eur_per_kwh=inp.price_buy_eur_per_kwh,
                kwh_sold_year1=kwh_total,
                kwh_growth_yoy=inp.kwh_growth_yoy,
                fixed_opex_year1=fixed_opex_year1,
                fixed_opex_growth_yoy=inp.fixed_opex_overhead_growth_yoy,
                variable_opex_per_kwh=inp.variable_opex_per_kwh,
            )

            fin_res, _ = evaluate_finance(fin_inp)

            res = OptimizationResult(
                n_ac=n_ac,
                n_dc=n_dc,
                capex=capex,
                power_installed_kw=power,
                npv=fin_res.npv,
                irr=fin_res.irr,
                payback=fin_res.payback_year,
                notes=notes.strip(),
            )
            results.append(res)

            if best is None or res.npv > best.npv:
                best = res

    if best is None:
        best = OptimizationResult(
            n_ac=0,
            n_dc=0,
            capex=0.0,
            power_installed_kw=0.0,
            npv=float('-inf'),
            irr=float('nan'),
            payback=float('inf'),
            notes="Nessuna combinazione rispetta i vincoli (potenza/budget).",
        )

    # Sort results by NPV desc
    results_sorted = sorted(results, key=lambda x: x.npv, reverse=True)
    return best, results_sorted
