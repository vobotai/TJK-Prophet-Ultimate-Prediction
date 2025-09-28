"""Gate and context level enrichments."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .parsers import slugify


DISTANCE_BUCKETS = [(0, 1400, "<=1400"), (1400, 2000, "1400-2000"), (2000, float("inf"), ">2000")]


def distance_bucket(distance: float | int | None) -> str | None:
    if distance is None or (isinstance(distance, float) and np.isnan(distance)):
        return None
    for low, high, label in DISTANCE_BUCKETS:
        if distance <= high:
            if low == 1400 and distance <= low:
                return "<=1400"
            return label
    return None


def compute_gate_and_context(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "start_no" not in data.columns:
        data["start_no"] = np.nan
    data["gate_rank_pct"] = data.groupby("race_uid")["start_no"].transform(
        lambda s: s.rank(pct=True, method="average")
    )
    gate_keys = []
    race_buckets = []
    for _, row in data.iterrows():
        bucket = distance_bucket(row.get("mesafe"))
        race_buckets.append(bucket)
        pist = row.get("pist_tipi") or "unknown"
        pist_durumu = row.get("pist_durumu") or "none"
        hip_slug = slugify(row.get("hipodrom") or "")
        key = f"{pist}|{pist_durumu}|{bucket}|{hip_slug}" if bucket else None
        gate_keys.append(key)
    data["mesafe_bucket"] = race_buckets
    data["gate_context_key"] = gate_keys

    if "field_size" not in data.columns:
        data["field_size"] = data.groupby("race_uid").size().values

    summary = (
        data.groupby("race_uid")
        .agg(
            field_size=("field_size", "first"),
            mesafe=("mesafe", "first"),
            pist_tipi=("pist_tipi", "first"),
            pist_durumu=("pist_durumu", "first"),
            hipodrom=("hipodrom", "first"),
            kosu_sinifi=("kosu_sinifi", "first"),
            hava_durumu=("hava_durumu", "first"),
            median_handikap=("handikap_puani_median", "first"),
            median_en_iyi=("en_iyi_derece_s_median", "first"),
            median_agf=("agf_01_median", "first"),
        )
        .to_dict(orient="index")
    )

    race_context = []
    for _, row in data.iterrows():
        ctx = summary.get(row["race_uid"], {})
        race_context.append(
            {
                "field_size": ctx.get("field_size"),
                "mesafe": ctx.get("mesafe"),
                "pist_tipi": ctx.get("pist_tipi"),
                "pist_durumu": ctx.get("pist_durumu"),
                "hipodrom": ctx.get("hipodrom"),
                "kosu_sinifi": ctx.get("kosu_sinifi"),
                "hava_durumu": ctx.get("hava_durumu"),
                "median_handikap": ctx.get("median_handikap"),
                "median_en_iyi_derece_s": ctx.get("median_en_iyi"),
                "median_agf_01": ctx.get("median_agf"),
            }
        )
    data["race_context"] = race_context
    return data
