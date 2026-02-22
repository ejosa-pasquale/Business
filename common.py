from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import requests


@dataclass
class FetchResult:
    df: pd.DataFrame
    source: str
    note: str = ""


def fetch_csv(url: str, timeout: int = 20) -> FetchResult:
    """Fetch a CSV (or TSV) from a URL with basic heuristics.

    - Tries to infer separator.
    - Returns a dataframe + metadata.

    This function is meant to run at app time (user machine / deployed server).
    """
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    content_type = (r.headers.get("content-type") or "").lower()

    raw = r.content
    text = raw.decode("utf-8", errors="replace")

    # Heuristic delimiter detection
    sample = "\n".join(text.splitlines()[:20])
    sep = ";" if sample.count(";") > sample.count(",") else ","
    if sample.count("\t") > max(sample.count(";"), sample.count(",")):
        sep = "\t"

    df = pd.read_csv(io.StringIO(text), sep=sep)
    note = f"content-type={content_type}, sep='{sep}'"
    return FetchResult(df=df, source=url, note=note)


def read_local_csv(path: str, sep: Optional[str] = None) -> FetchResult:
    df = pd.read_csv(path, sep=sep)
    return FetchResult(df=df, source=path, note="local file")
