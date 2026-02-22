from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd


@dataclass
class DemandInputs:
    # Parking flow side
    vehicles_per_day: float
    bev_share: float
    share_bev_that_charge: float
    kwh_per_session_ac: float
    kwh_per_session_dc: float
    share_sessions_dc: float  # fraction of charging sessions that choose DC


@dataclass
class DemandResult:
    sessions_per_day: float
    sessions_ac_per_day: float
    sessions_dc_per_day: float
    kwh_per_day: float
    kwh_ac_per_day: float
    kwh_dc_per_day: float


def demand_from_parking(inp: DemandInputs) -> DemandResult:
    bev_vehicles = inp.vehicles_per_day * inp.bev_share
    sessions = bev_vehicles * inp.share_bev_that_charge

    sessions_dc = sessions * inp.share_sessions_dc
    sessions_ac = sessions - sessions_dc

    kwh_dc = sessions_dc * inp.kwh_per_session_dc
    kwh_ac = sessions_ac * inp.kwh_per_session_ac

    return DemandResult(
        sessions_per_day=float(sessions),
        sessions_ac_per_day=float(sessions_ac),
        sessions_dc_per_day=float(sessions_dc),
        kwh_per_day=float(kwh_ac + kwh_dc),
        kwh_ac_per_day=float(kwh_ac),
        kwh_dc_per_day=float(kwh_dc),
    )


@dataclass
class FunnelInputs:
    bev_2030: int
    kwh_per_bev_year: float
    public_share: float
    capture_share: float


def demand_from_funnel(inp: FunnelInputs) -> float:
    """Annual kWh captured by the site."""
    return float(inp.bev_2030) * float(inp.kwh_per_bev_year) * float(inp.public_share) * float(inp.capture_share)


def cagr(v0: float, v1: float, years: int) -> float:
    v0 = max(float(v0), 1e-9)
    v1 = max(float(v1), 1e-9)
    years = max(int(years), 1)
    return (v1 / v0) ** (1 / years) - 1


def forecast_path(start_year: int, start_value: float, end_year: int, end_value: float) -> pd.DataFrame:
    """Simple exponential path."""
    years = end_year - start_year
    r = cagr(start_value, end_value, years)
    vals = [start_value * ((1 + r) ** i) for i in range(0, years + 1)]
    return pd.DataFrame({"year": list(range(start_year, end_year + 1)), "value": vals})
