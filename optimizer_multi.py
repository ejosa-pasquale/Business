from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import math

from finance import FinanceInputs, evaluate_finance


@dataclass
class TechCost:
    """Tech parameters used by the optimizer."""

    name: str
    capex_per_charger: float
    fixed_opex_per_charger_year: float
    connectors: int
    power_kw: float


@dataclass
class OptimizationInputs:
    # Demand split (year 1) — expressed as *annual* kWh
    kwh_ac_year1: float
    kwh_dc_year1: float

    # Queue/throughput knobs
    uptime: float
    target_utilization: float  # (0-1) comfort utilization; above this → queue risk

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
    max_dc30: int
    max_dc60: int
    max_dc90: int


@dataclass
class OptimizationResult:
    n_ac: int
    n_dc30: int
    n_dc60: int
    n_dc90: int
    capex: float
    power_installed_kw: float
    kwh_sold_year1: float
    npv: float
    irr: float
    payback: float
    notes: str


def _annual_capacity_kwh(n: int, tech: TechCost, uptime: float, util: float) -> float:
    # Connector-level energy capacity
    connectors = max(int(tech.connectors), 1)
    return float(n) * connectors * float(tech.power_kw) * 24.0 * float(uptime) * float(util) * 365.0


def optimize_mix_4tech(
    inp: OptimizationInputs,
    ac22: TechCost,
    dc30: TechCost,
    dc60: TechCost,
    dc90: TechCost,
    mismatch_penalty: float = 0.15,
) -> Tuple[OptimizationResult, List[OptimizationResult]]:
    """Brute-force optimization across AC22 + DC30 + DC60 + DC90.

    Improvements vs v0:
      1) multiple DC technologies
      2) sold kWh is *capped by installed capacity* (so budget/power trade-offs are visible)
      3) flag queue risk when demand forces utilization above target
    """

    results: List[OptimizationResult] = []
    best: OptimizationResult | None = None

    for n_ac in range(0, inp.max_ac + 1):
        for n30 in range(0, inp.max_dc30 + 1):
            for n60 in range(0, inp.max_dc60 + 1):
                for n90 in range(0, inp.max_dc90 + 1):
                    if n_ac == 0 and n30 == 0 and n60 == 0 and n90 == 0:
                        continue

                    power = (
                        n_ac * ac22.power_kw
                        + n30 * dc30.power_kw
                        + n60 * dc60.power_kw
                        + n90 * dc90.power_kw
                    )
                    if power > inp.power_available_kw + 1e-9:
                        continue

                    capex = (
                        n_ac * ac22.capex_per_charger
                        + n30 * dc30.capex_per_charger
                        + n60 * dc60.capex_per_charger
                        + n90 * dc90.capex_per_charger
                    )
                    if capex > inp.capex_budget + 1e-9:
                        continue

                    # Demand split; if one family is missing, re-route with a penalty.
                    kwh_ac = float(inp.kwh_ac_year1)
                    kwh_dc = float(inp.kwh_dc_year1)
                    notes = ""

                    has_ac = n_ac > 0
                    has_dc = (n30 + n60 + n90) > 0
                    if (not has_ac) and kwh_ac > 0:
                        kwh_dc += (1.0 - mismatch_penalty) * kwh_ac
                        notes += f"AC=0: domanda AC → DC (−{int(mismatch_penalty*100)}%). "
                        kwh_ac = 0.0
                    if (not has_dc) and kwh_dc > 0:
                        kwh_ac += (1.0 - mismatch_penalty) * kwh_dc
                        notes += f"DC=0: domanda DC → AC (−{int(mismatch_penalty*100)}%). "
                        kwh_dc = 0.0

                    # Capacity constraints (annual kWh)
                    cap_ac_target = _annual_capacity_kwh(n_ac, ac22, inp.uptime, inp.target_utilization)
                    cap_ac_max = _annual_capacity_kwh(n_ac, ac22, inp.uptime, 1.0)

                    cap_dc_target = (
                        _annual_capacity_kwh(n30, dc30, inp.uptime, inp.target_utilization)
                        + _annual_capacity_kwh(n60, dc60, inp.uptime, inp.target_utilization)
                        + _annual_capacity_kwh(n90, dc90, inp.uptime, inp.target_utilization)
                    )
                    cap_dc_max = (
                        _annual_capacity_kwh(n30, dc30, inp.uptime, 1.0)
                        + _annual_capacity_kwh(n60, dc60, inp.uptime, 1.0)
                        + _annual_capacity_kwh(n90, dc90, inp.uptime, 1.0)
                    )

                    sold_ac = min(kwh_ac, cap_ac_max)
                    sold_dc = min(kwh_dc, cap_dc_max)

                    if kwh_ac > cap_ac_max + 1e-6:
                        notes += "AC: domanda > capacità (vendite perse). "
                    elif kwh_ac > cap_ac_target + 1e-6:
                        notes += "AC: oltre target util (rischio code). "

                    if kwh_dc > cap_dc_max + 1e-6:
                        notes += "DC: domanda > capacità (vendite perse). "
                    elif kwh_dc > cap_dc_target + 1e-6:
                        notes += "DC: oltre target util (rischio code). "

                    kwh_total_sold = sold_ac + sold_dc

                    fixed_opex_year1 = (
                        inp.fixed_opex_overhead_year1
                        + n_ac * ac22.fixed_opex_per_charger_year
                        + n30 * dc30.fixed_opex_per_charger_year
                        + n60 * dc60.fixed_opex_per_charger_year
                        + n90 * dc90.fixed_opex_per_charger_year
                    )

                    fin_inp = FinanceInputs(
                        years=inp.years,
                        discount_rate=inp.discount_rate,
                        capex_total=capex,
                        price_sell_eur_per_kwh=inp.price_sell_eur_per_kwh,
                        price_buy_eur_per_kwh=inp.price_buy_eur_per_kwh,
                        kwh_sold_year1=kwh_total_sold,
                        kwh_growth_yoy=inp.kwh_growth_yoy,
                        fixed_opex_year1=fixed_opex_year1,
                        fixed_opex_growth_yoy=inp.fixed_opex_overhead_growth_yoy,
                        variable_opex_per_kwh=inp.variable_opex_per_kwh,
                    )
                    fin_res, _ = evaluate_finance(fin_inp)

                    res = OptimizationResult(
                        n_ac=n_ac,
                        n_dc30=n30,
                        n_dc60=n60,
                        n_dc90=n90,
                        capex=float(capex),
                        power_installed_kw=float(power),
                        kwh_sold_year1=float(kwh_total_sold),
                        npv=float(fin_res.npv),
                        irr=float(fin_res.irr),
                        payback=float(fin_res.payback_year),
                        notes=notes.strip(),
                    )
                    results.append(res)

                    if best is None or res.npv > best.npv:
                        best = res

    if best is None:
        best = OptimizationResult(
            n_ac=0,
            n_dc30=0,
            n_dc60=0,
            n_dc90=0,
            capex=0.0,
            power_installed_kw=0.0,
            kwh_sold_year1=0.0,
            npv=float("-inf"),
            irr=float("nan"),
            payback=float("inf"),
            notes="Nessuna combinazione rispetta i vincoli (potenza/budget).",
        )

    results_sorted = sorted(results, key=lambda x: x.npv, reverse=True)
    return best, results_sorted
