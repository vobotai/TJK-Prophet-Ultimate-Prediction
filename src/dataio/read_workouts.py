"""Reading the optional workouts CSV."""
from __future__ import annotations

import difflib
from typing import Dict, List

import pandas as pd

from features.parsers import normalize_distance, parse_date, parse_float


def _clean_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()

EXPECTED_COLUMNS = [
    "Tarih",
    "Hipodrom",
    "Koşu ID",
    "At İsmi",
    "At ID",
    "Workout Tarih",
    "W_Hip",
    "W_Pist",
    "W_Type",
    "W_Status",
    "W_Jokey",
    "w1200_s",
    "w1000_s",
    "w800_s",
    "w600_s",
    "w400_s",
    "w200_s",
    "matched_name",
    "matched_url",
    "match_method",
    "match_score",
]


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(df.columns)
    mapped = {}
    lower_map = {c.lower(): c for c in columns}
    for expected in EXPECTED_COLUMNS:
        if expected in columns:
            mapped[expected] = expected
            continue
        low = expected.lower()
        if low in lower_map:
            mapped[expected] = lower_map[low]
            continue
        choices = difflib.get_close_matches(expected, columns, n=1, cutoff=0.86)
        if choices:
            mapped[expected] = choices[0]
    rename = {mapped[k]: k for k in mapped if mapped[k] != k}
    if rename:
        df = df.rename(columns=rename)
    return df


def read_workouts_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    df = _map_columns(df)
    if "Tarih" not in df.columns:
        raise ValueError("Workout CSV için 'Tarih' kolonu bulunamadı")

    records: List[Dict[str, object]] = []
    for idx, row in df.iterrows():
        tarih_iso = parse_date(row.get("Tarih"))
        if not tarih_iso:
            continue
        hipodrom = _clean_str(row.get("Hipodrom"))
        if not hipodrom:
            continue
        kosu_id = _clean_str(row.get("Koşu ID")) or None
        at_ismi = _clean_str(row.get("At İsmi"))
        if not at_ismi:
            continue
        workout_date = parse_date(row.get("Workout Tarih"))
        match_score = parse_float(row.get("match_score"))

        record = {
            "tarih": tarih_iso,
            "hipodrom": hipodrom,
            "kosu_id": kosu_id,
            "at_ismi": at_ismi,
            "workout_date": workout_date,
            "w_hip": _clean_str(row.get("W_Hip")) or None,
            "w_pist": _clean_str(row.get("W_Pist")) or None,
            "w_type": _clean_str(row.get("W_Type")) or None,
            "w_status": _clean_str(row.get("W_Status")) or None,
            "w_jokey": _clean_str(row.get("W_Jokey")) or None,
            "w1200_s": parse_float(row.get("w1200_s")),
            "w1000_s": parse_float(row.get("w1000_s")),
            "w800_s": parse_float(row.get("w800_s")),
            "w600_s": parse_float(row.get("w600_s")),
            "w400_s": parse_float(row.get("w400_s")),
            "w200_s": parse_float(row.get("w200_s")),
            "matched_name": _clean_str(row.get("matched_name")) or None,
            "matched_url": _clean_str(row.get("matched_url")) or None,
            "match_method": _clean_str(row.get("match_method")) or None,
            "match_score": match_score,
        }
        records.append(record)
    return pd.DataFrame.from_records(records)
