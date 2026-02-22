from __future__ import annotations

import math
from typing import Optional


def eur(x: Optional[float], digits: int = 0) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "n/a"
    fmt = f"{{:,.{digits}f}}".format(float(x))
    # 12,345.67 -> 12.345,67
    fmt = fmt.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"€ {fmt}"


def num(x: Optional[float], digits: int = 0) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "n/a"
    fmt = f"{{:,.{digits}f}}".format(float(x))
    return fmt.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: Optional[float], digits: int = 1) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "n/a"
    return f"{float(x)*100:.{digits}f}%".replace(".", ",")
