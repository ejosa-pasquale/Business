# sizing.py - flat layout sizing helpers
from __future__ import annotations

import math


def compute_capacity_kwh_per_year(power_kw: float, uptime: float, utilization: float) -> float:
    power_kw = max(0.0, float(power_kw))
    uptime = min(max(0.0, float(uptime)), 1.0)
    utilization = min(max(0.0, float(utilization)), 1.0)
    return power_kw * 8760.0 * uptime * utilization


def suggest_mix_from_targets(
    demand_kwh_year: float,
    share_dc: float,
    ac_power_kw: float,
    dc_power_kw: float,
    uptime: float,
    target_util: float,
    site_power_kw: float,
) -> dict:
    demand_kwh_year = max(0.0, float(demand_kwh_year))
    share_dc = min(max(0.0, float(share_dc)), 1.0)

    ac_kwh_need = demand_kwh_year * (1.0 - share_dc)
    dc_kwh_need = demand_kwh_year * share_dc

    cap_ac_one = compute_capacity_kwh_per_year(ac_power_kw, uptime, target_util)
    cap_dc_one = compute_capacity_kwh_per_year(dc_power_kw, uptime, target_util)

    n_ac = int(math.ceil(ac_kwh_need / cap_ac_one)) if cap_ac_one > 0 else 0
    n_dc = int(math.ceil(dc_kwh_need / cap_dc_one)) if cap_dc_one > 0 else 0

    # Power constraint (simple peak assumption: all simultaneous at nameplate)
    power_required = n_ac * float(ac_power_kw) + n_dc * float(dc_power_kw)
    if power_required > float(site_power_kw) and float(site_power_kw) > 0:
        # try reduce DC first, then AC (heuristic)
        while n_dc > 0 and (n_ac * ac_power_kw + n_dc * dc_power_kw) > site_power_kw:
            n_dc -= 1
        while n_ac > 0 and (n_ac * ac_power_kw + n_dc * dc_power_kw) > site_power_kw:
            n_ac -= 1

    power_required = n_ac * float(ac_power_kw) + n_dc * float(dc_power_kw)

    return {
        "n_ac": n_ac,
        "n_dc": n_dc,
        "power_required_kw": power_required,
        "ac_kwh_need": ac_kwh_need,
        "dc_kwh_need": dc_kwh_need,
        "site_power_kw": float(site_power_kw),
    }
