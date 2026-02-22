from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math


@dataclass
class ChargerTech:
    name: str
    power_kw: float
    connectors: int


@dataclass
class SizingInputs:
    demand_kwh_per_day: float
    demand_sessions_per_day: float
    uptime: float
    target_utilization: float  # average utilization over the day (0-1)
    avg_session_hours: float


@dataclass
class SizingResult:
    required_connectors: int
    required_chargers: int
    achieved_utilization: float
    capacity_kwh_per_day: float
    sessions_capacity_per_day: float


def size_for_tech(tech: ChargerTech, inp: SizingInputs) -> SizingResult:
    """Size number of chargers for a single technology.

    Uses two constraints:
      - energy: kWh/day capacity
      - throughput: sessions/day capacity based on avg session duration

    We pick the max required.
    """
    # Energy capacity per connector
    cap_kwh_per_day_per_connector = tech.power_kw * 24.0 * inp.uptime * inp.target_utilization

    # Sessions capacity per connector
    sessions_cap_per_day_per_connector = (24.0 * inp.uptime * inp.target_utilization) / max(inp.avg_session_hours, 0.1)

    req_by_energy = inp.demand_kwh_per_day / max(cap_kwh_per_day_per_connector, 1e-9)
    req_by_sessions = inp.demand_sessions_per_day / max(sessions_cap_per_day_per_connector, 1e-9)

    required_connectors = int(math.ceil(max(req_by_energy, req_by_sessions)))
    required_chargers = int(math.ceil(required_connectors / tech.connectors))

    # Achieved utilization given resulting count
    total_connectors = required_chargers * tech.connectors
    capacity_kwh_per_day = total_connectors * tech.power_kw * 24.0 * inp.uptime * inp.target_utilization
    sessions_capacity = total_connectors * sessions_cap_per_day_per_connector

    # utilization needed to meet energy (approx)
    needed_util = inp.demand_kwh_per_day / max(total_connectors * tech.power_kw * 24.0 * inp.uptime, 1e-9)
    achieved_util = float(min(max(needed_util, 0.0), 1.0))

    return SizingResult(
        required_connectors=required_connectors,
        required_chargers=required_chargers,
        achieved_utilization=achieved_util,
        capacity_kwh_per_day=float(capacity_kwh_per_day),
        sessions_capacity_per_day=float(sessions_capacity),
    )
