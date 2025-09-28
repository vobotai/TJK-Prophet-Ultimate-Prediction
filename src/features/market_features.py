"""Market related enrichments."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .parsers import parse_float, sigmoid


def compute_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "ganyan" in data.columns:
        clean = pd.to_numeric(data["ganyan"], errors="coerce")
        clean = clean.where(clean > 0)
        data["implied_prob"] = 1.0 / clean
    if "implied_prob" not in data.columns:
        data["implied_prob"] = np.nan

    data["implied_prob"] = data["implied_prob"].replace([np.inf, -np.inf], np.nan)
    totals = data.groupby("race_uid")["implied_prob"].transform("sum", min_count=1)
    data["market_overround"] = totals
    data["p_market"] = data["implied_prob"] / totals
    data.loc[data["implied_prob"].isna(), "p_market"] = np.nan

    if "agf_01" not in data.columns:
        data["agf_01"] = np.nan
    diff = data["agf_01"] - data["p_market"].fillna(0.0)
    data["mdi"] = diff.apply(lambda x: sigmoid(10 * x) if not np.isnan(x) else np.nan)

    for col in ["drift_dp60", "drift_dp30", "drift_dp15", "drift_dp5", "dagf15", "dagf30"]:
        data[col] = np.nan
    return data
