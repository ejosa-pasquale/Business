# parking_occupancy.py - flat layout
from __future__ import annotations

import pandas as pd


def estimate_daily_traffic_from_parking(
    parking_df: pd.DataFrame,
    total_stalls: int,
    avg_dwell_h: float,
) -> float:
    '''
    Estimate daily vehicle entries ("traffic") from parking occupancy time series.

    Expected parking_df columns (from common.parse_parking_csv):
      - timestamp: datetime
      - value: numeric
      - metric_type: 'occupancy' or 'free'

    Logic (robust, MVP):
      1) Convert to estimated occupied stalls over time.
      2) Compute average occupied stalls during the day.
      3) Estimate daily vehicle count ≈ (avg_occupied_stalls * 24) / avg_dwell_h

    Notes:
      - If metric_type == 'free', occupied = total_stalls - free
      - If occupancy looks like a ratio (0..1) or percent (0..100), convert to stalls.
      - Clamp to [0, total_stalls].
    '''
    if parking_df is None or parking_df.empty:
        raise ValueError("parking_df vuoto")
    if total_stalls <= 0:
        raise ValueError("total_stalls deve essere > 0")
    if avg_dwell_h <= 0:
        raise ValueError("avg_dwell_h deve essere > 0")

    df = parking_df.copy()
    if "timestamp" not in df.columns or "value" not in df.columns or "metric_type" not in df.columns:
        raise ValueError("parking_df non ha le colonne attese: timestamp, value, metric_type")

    metric_type = str(df["metric_type"].iloc[0]).lower()

    series = pd.to_numeric(df["value"], errors="coerce").dropna()
    if series.empty:
        raise ValueError("Nessun valore numerico valido in parking_df")

    # Convert to occupied stalls time series
    v = pd.to_numeric(df["value"], errors="coerce")

    if metric_type == "free":
        occupied = total_stalls - v
    else:
        # occupancy: could be stalls, ratio, or percent
        # Heuristic: if max <= 1.2 -> ratio; if max <= 120 -> percent; else stalls
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

    # Guardrail: at least 0
    return max(0.0, daily_traffic)
