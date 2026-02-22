# demand.py - flat layout demand models
from __future__ import annotations


def demand_from_parking_model(
    daily_traffic: float,
    bev_share: float,
    charge_take_rate: float,
    kwh_per_session_ac: float,
    kwh_per_session_dc: float,
    share_dc: float = 0.25,
    days_per_year: int = 365,
) -> dict:
    '''
    Estimate annual demand (kWh/year) from parking traffic.

    Inputs:
      - daily_traffic: vehicles/day
      - bev_share: fraction of vehicles that are BEV (0..1)
      - charge_take_rate: fraction of BEV that choose to charge on site (0..1)
      - kwh_per_session_ac/dc: average energy per charging session
      - share_dc: fraction of charging sessions that are DC (0..1)

    Output dict includes kWh/year, sessions/year split, etc.
    '''
    daily_traffic = max(0.0, float(daily_traffic))
    bev_share = min(max(0.0, float(bev_share)), 1.0)
    charge_take_rate = min(max(0.0, float(charge_take_rate)), 1.0)
    share_dc = min(max(0.0, float(share_dc)), 1.0)
    days_per_year = int(days_per_year)

    bev_daily = daily_traffic * bev_share
    sessions_daily = bev_daily * charge_take_rate

    sessions_dc_daily = sessions_daily * share_dc
    sessions_ac_daily = sessions_daily * (1.0 - share_dc)

    kwh_daily = sessions_ac_daily * float(kwh_per_session_ac) + sessions_dc_daily * float(kwh_per_session_dc)
    kwh_year = kwh_daily * days_per_year

    return {
        "daily_traffic": daily_traffic,
        "bev_daily": bev_daily,
        "sessions_daily": sessions_daily,
        "sessions_ac_daily": sessions_ac_daily,
        "sessions_dc_daily": sessions_dc_daily,
        "kwh_daily": kwh_daily,
        "kwh_year": kwh_year,
        "share_dc": share_dc,
    }


def demand_from_funnel_model(
    bev_count: int,
    kwh_per_bev_year: float,
    public_share: float,
    capture: float,
) -> dict:
    '''
    Macro funnel:
      total_kwh = bev_count * kwh_per_bev_year
      public_kwh = total_kwh * public_share
      site_kwh = public_kwh * capture
    '''
    bev_count = max(0, int(bev_count))
    kwh_per_bev_year = max(0.0, float(kwh_per_bev_year))
    public_share = min(max(0.0, float(public_share)), 1.0)
    capture = min(max(0.0, float(capture)), 1.0)

    total_kwh = bev_count * kwh_per_bev_year
    public_kwh = total_kwh * public_share
    site_kwh = public_kwh * capture

    return {
        "bev_count": bev_count,
        "total_kwh": total_kwh,
        "public_kwh": public_kwh,
        "kwh_year": site_kwh,
        "public_share": public_share,
        "capture": capture,
    }
