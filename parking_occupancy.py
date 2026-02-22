# parking_occupancy.py - flat layout
from __future__ import annotations

import pandas as pd


def estimate_daily_traffic_from_parking(parking_df: pd.DataFrame, total_stalls: int, avg_dwell_h: float) -> float:
    if parking_df is None or parking_df.empty:
        raise ValueError("parking_df vuoto")
    if total_stalls <= 0:
        raise ValueError("total_stalls deve essere > 0")
    if avg_dwell_h <= 0:
        raise ValueError("avg_dwell_h deve essere > 0")

    df = parking_df.copy()
    if not set(["timestamp", "value", "metric_type"]).issubset(set(df.columns)):
        raise ValueError("parking_df deve avere colonne: timestamp, value, metric_type")

    metric_type = str(df["metric_type"].iloc[0]).lower()
    v = pd.to_numeric(df["value"], errors="coerce")

    if metric_type == "free":
        occupied = total_stalls - v
    else:
        vmax = float(pd.to_numeric(v, errors="coerce").max())
        if vmax <= 1.2:
            occupied = v * total_stalls
        elif vmax <= 120:
            occupied = (v / 100.0) * total_stalls
        else:
            occupied = v

    occupied = occupied.clip(lower=0, upper=total_stalls)
    avg_occupied = float(occupied.mean(skipna=True))
    daily_traffic = (avg_occupied * 24.0) / float(avg_dwell_h)
    return max(0.0, daily_traffic)
