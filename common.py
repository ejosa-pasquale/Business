# common.py - shared utilities (flat layout)

from __future__ import annotations

import pandas as pd


def fetch_csv(url: str, **read_csv_kwargs) -> pd.DataFrame:
    '''
    Download a CSV from a URL and return as DataFrame.
    Uses pandas' built-in URL handling; works for most https links.
    '''
    if not url or not isinstance(url, str):
        raise ValueError("URL non valido")
    return pd.read_csv(url, **read_csv_kwargs)


def _find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    # partial match
    for c in df.columns:
        lc = str(c).lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c
    return None


def parse_parking_csv(raw: pd.DataFrame) -> pd.DataFrame:
    '''
    Normalize a parking time series CSV to:
      - datetime column: 'timestamp'
      - metric column: 'value'
      - metric_type in {'occupancy', 'free'}

    Accepted time column names include: timestamp, datetime, date, ora, time
    Accepted metric names include:
      occupancy/occupazione/occupied (0..1 or 0..100 or count)
      free/liberi/posti_liberi (count)
    '''
    if raw is None or raw.empty:
        raise ValueError("CSV parcheggio vuoto")

    df = raw.copy()

    time_col = _find_first_col(df, ["timestamp", "datetime", "date", "data", "ora", "time"])
    if time_col is None:
        raise ValueError("Non trovo una colonna tempo (timestamp/datetime/date/ora).")

    occ_col = _find_first_col(df, ["occupancy", "occupazione", "occupied", "occ"])
    free_col = _find_first_col(df, ["free", "liberi", "posti_liberi", "available", "disponibili"])

    if occ_col is not None:
        metric_type = "occupancy"
        metric_col = occ_col
    elif free_col is not None:
        metric_type = "free"
        metric_col = free_col
    else:
        raise ValueError("Non trovo colonna metrica (occupancy/occupied oppure free/liberi).")

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    out["value"] = pd.to_numeric(df[metric_col], errors="coerce")
    out["metric_type"] = metric_type
    out = out.dropna(subset=["timestamp", "value"]).sort_values("timestamp")

    return out


def format_eur(x: float) -> str:
    try:
        v = float(x)
    except Exception:
        return "€0"
    s = f"{v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"€{s}"


def format_pct(x: float) -> str:
    try:
        v = float(x) * 100.0
    except Exception:
        return "0%"
    return f"{v:.0f}%"
