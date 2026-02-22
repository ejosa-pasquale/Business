from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import pandas as pd


@dataclass
class ParkingSeries:
    df: pd.DataFrame
    time_col: str
    metric_col: str
    metric_kind: Literal["occupancy", "free_spots"]


def parse_parking_csv(df: pd.DataFrame) -> ParkingSeries:
    """Attempt to parse a parking time series.

    Expected columns (any of these):
      - time: timestamp, datetime, date, ora
      - metric: occupancy, occupazione, occupied, posti_occupati, free, liberi

    Returns normalized dataframe with columns: ts, value
    """
    cols = {c.lower(): c for c in df.columns}
    time_candidates = [
        k for k in cols
        if any(x in k for x in ["timestamp", "datetime", "data", "date", "ora", "time", "ts"]) 
    ]
    if not time_candidates:
        raise ValueError("Nessuna colonna tempo trovata. Attese: timestamp/datetime/data/ora")

    time_col = cols[time_candidates[0]]

    metric_candidates_occ = [k for k in cols if any(x in k for x in ["occup", "occupied", "posti_occupati"]) ]
    metric_candidates_free = [k for k in cols if any(x in k for x in ["free", "liber", "posti_liberi"]) ]

    if metric_candidates_occ:
        metric_col = cols[metric_candidates_occ[0]]
        kind = "occupancy"
    elif metric_candidates_free:
        metric_col = cols[metric_candidates_free[0]]
        kind = "free_spots"
    else:
        raise ValueError("Nessuna colonna metrica trovata. Attese: occupazione/occupied o liberi/free")

    out = df[[time_col, metric_col]].copy()
    out.columns = ["ts", "value"]
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce", utc=False)
    out = out.dropna(subset=["ts"]).sort_values("ts")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])  # drop non numeric

    return ParkingSeries(df=out, time_col=time_col, metric_col=metric_col, metric_kind=kind)


def estimate_daily_arrivals(
    series: ParkingSeries,
    total_spots: int,
    avg_stay_hours: float,
) -> pd.DataFrame:
    """Estimate arrivals/day from occupancy time series.

    Very simple estimator:
      vehicles_per_day ≈ (avg_occupied_spots * 24) / avg_stay_hours

    If series is free_spots, occupancy = total_spots - free.
    """
    df = series.df.copy()
    if series.metric_kind == "free_spots":
        df["occ"] = total_spots - df["value"]
    else:
        df["occ"] = df["value"]

    df["date"] = df["ts"].dt.date
    g = df.groupby("date")["occ"].mean().reset_index()
    g["vehicles_per_day"] = (g["occ"] * 24.0) / max(avg_stay_hours, 0.25)
    return g
