# trento_chargers.py - flat layout helper for chargers dataset
from __future__ import annotations

import pandas as pd
from typing import Optional, List, Dict


def _col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    cols = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    for c in df.columns:
        lc = str(c).lower()
        for n in names:
            if n.lower() in lc:
                return c
    return None


def summarize_trento_chargers(df: pd.DataFrame) -> Dict:
    if df is None or df.empty:
        return {"rows": 0}

    out: Dict = {"rows": int(len(df)), "columns": list(df.columns)}
    power_col = _col(df, ["power", "potenza", "kw", "max_power_kw"])
    operator_col = _col(df, ["operator", "gestore", "provider", "brand"])
    status_col = _col(df, ["status", "stato", "available", "attiva"])
    type_col = _col(df, ["type", "tipo", "connector", "presa", "ac_dc", "current_type"])

    if power_col:
        p = pd.to_numeric(df[power_col], errors="coerce")
        out["power_kw"] = {
            "min": float(p.min()) if p.notna().any() else None,
            "median": float(p.median()) if p.notna().any() else None,
            "max": float(p.max()) if p.notna().any() else None,
        }
    if operator_col:
        out["top_operators"] = df[operator_col].astype(str).value_counts().head(10).to_dict()
    if status_col:
        out["status_counts"] = df[status_col].astype(str).value_counts().head(10).to_dict()
    if type_col:
        out["type_counts"] = df[type_col].astype(str).value_counts().head(10).to_dict()

    lat_col = _col(df, ["lat", "latitude", "y"])
    lon_col = _col(df, ["lon", "lng", "longitude", "x"])
    out["has_coordinates"] = bool(lat_col and lon_col)
    return out
