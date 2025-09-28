"""Set/field-wise statistics for each race."""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FIELDS = [
    "yas",
    "siklet",
    "handikap_puani",
    "kgs",
    "s20",
    "en_iyi_derece_s",
    "agf_01",
    "implied_prob",
    "start_no",
]


def compute_set_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["field_size"] = data.groupby("race_uid")[["race_uid"]].transform("count")
    for col in NUMERIC_FIELDS:
        if col not in data.columns:
            data[col] = np.nan
        grouped = data.groupby("race_uid")[col]
        data[f"{col}_mean"] = grouped.transform("mean")
        data[f"{col}_std"] = grouped.transform("std").fillna(0.0)
        data[f"{col}_median"] = grouped.transform("median")
        data[f"{col}_min"] = grouped.transform("min")
        data[f"{col}_max"] = grouped.transform("max")
        denom = data[f"{col}_std"].replace(0.0, 1e-6)
        data[f"{col}_rel_z"] = (data[col] - data[f"{col}_mean"]) / denom
        data[f"{col}_rank_pct"] = grouped.rank(pct=True, method="average")
        data[f"{col}_delta_med"] = data[col] - data[f"{col}_median"]
    return data
