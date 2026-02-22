from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from common import fetch_csv, read_local_csv, FetchResult


DEFAULT_TRENTO_DATASET_PAGE = (
    "https://www.comune.trento.it/Amministrazione/Documenti-e-dati/Dataset/"
    "Colonnine-di-ricarica-per-auto-elettriche"
)


@dataclass
class ChargersSummary:
    n_points: int
    n_locations: int
    power_cols: list[str]


def summarize_chargers(df: pd.DataFrame) -> ChargersSummary:
    cols = [c.lower() for c in df.columns]
    # Find likely identifiers
    loc_cols = [c for c in df.columns if c.lower() in {"indirizzo", "address", "via", "luogo", "location"}]
    power_cols = [c for c in df.columns if "kw" in c.lower() or "potenza" in c.lower() or "power" in c.lower()]
    n_points = len(df)
    n_locations = df[loc_cols[0]].nunique() if loc_cols else n_points
    return ChargersSummary(n_points=n_points, n_locations=int(n_locations), power_cols=power_cols)


def load_chargers_from_url(url: str) -> FetchResult:
    return fetch_csv(url)


def load_chargers_from_path(path: str) -> FetchResult:
    return read_local_csv(path)
