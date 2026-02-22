# optimizer.py - flat layout brute-force optimizer
from __future__ import annotations

import pandas as pd

from sizing import suggest_mix_from_targets
from finance import build_cashflows, npv, irr, payback_year


def optimize_mix_bruteforce(
    years: int,
    discount_rate: float,
    demand_kwh_year: float,
    annual_growth_kwh: float,
    sell_price_kwh: float,
    energy_cost_kwh: float,
    roaming_fee_pct: float,
    ac_power_kw: float,
    dc_power_kw: float,
    ac_capex: float,
    dc_capex: float,
    ac_opex_year: float,
    dc_opex_year: float,
    fixed_capex: float,
    fixed_opex_year: float,
    uptime: float,
    target_util: float,
    share_dc: float,
    site_power_kw: float,
    capex_budget,
    max_ac: int,
    max_dc: int,
):
    rows = []
    best = None
    best_npv = None

    for n_ac in range(0, int(max_ac) + 1):
        for n_dc in range(0, int(max_dc) + 1):
            if n_ac == 0 and n_dc == 0:
                continue

            power_req = n_ac * float(ac_power_kw) + n_dc * float(dc_power_kw)
            if power_req > float(site_power_kw):
                continue

            # Coverage check using target utilization capacity (approx)
            sizing = suggest_mix_from_targets(
                demand_kwh_year=demand_kwh_year,
                share_dc=share_dc,
                ac_power_kw=ac_power_kw,
                dc_power_kw=dc_power_kw,
                uptime=uptime,
                target_util=target_util,
                site_power_kw=site_power_kw,
            )
            # Require candidate to be at least as big as suggested (simple feasibility)
            if n_ac < sizing["n_ac"] or n_dc < sizing["n_dc"]:
                continue

            capex_total = n_ac * float(ac_capex) + n_dc * float(dc_capex) + float(fixed_capex)
            if capex_budget is not None and float(capex_total) > float(capex_budget):
                continue

            cash = build_cashflows(
                years=years,
                demand_kwh_year=demand_kwh_year,
                annual_growth_kwh=annual_growth_kwh,
                sell_price_kwh=sell_price_kwh,
                energy_cost_kwh=energy_cost_kwh,
                roaming_fee_pct=roaming_fee_pct,
                n_ac=n_ac,
                n_dc=n_dc,
                opex_ac_year=ac_opex_year,
                opex_dc_year=dc_opex_year,
                fixed_opex_year=fixed_opex_year,
                capex_total=capex_total,
            )
            npv_val = npv(cash["net_cashflow"], discount_rate)
            irr_val = irr(cash["net_cashflow"])
            pb = payback_year(cash["net_cashflow"])

            row = {
                "n_ac": n_ac,
                "n_dc": n_dc,
                "power_kw": power_req,
                "capex_total": capex_total,
                "npv": npv_val,
                "irr": irr_val,
                "payback": pb,
            }
            rows.append(row)

            if best_npv is None or npv_val > best_npv:
                best_npv = npv_val
                best = row

    if not rows:
        return None

    top = pd.DataFrame(rows).sort_values("npv", ascending=False).head(30).reset_index(drop=True)
    return {"best": best, "top_table": top}
