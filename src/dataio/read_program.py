"""Reading and validating the program CSV according to spec."""
from __future__ import annotations

import difflib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

from features.parsers import (
    genealogy_token,
    normalize_distance,
    parse_agf,
    parse_best_time,
    parse_date,
    parse_float,
    parse_int,
    parse_time,
    slugify,
)

EXPECTED_COLUMNS = [
    "Program Başlık",
    "Tarih",
    "Gün",
    "Hipodrom Başlık",
    "Hipodrom",
    "Yarış Günü Numarası",
    "Program Şehir ID",
    "Çim Durumu",
    "Kum Durumu",
    "Hava Bilgisi",
    "Hava Sıcaklığı",
    "Hava Durumu",
    "Nem",
    "Koşu ID",
    "Koşu Başlık",
    "Koşu Numarası",
    "Koşu Saati",
    "Özel Koşu Adı",
    "Koşu Sınıfı",
    "Baz Sıklet",
    "Mesafe",
    "Pist Tipi",
    "En İyi Derece (Tarihçe)",
    "Koşu Koşulları",
    "Muhtemel Linki",
    "Koşu PDF",
    "Pist Şeması Görseli",
    "Program Sırası",
    "At İsmi",
    "At Detay Linki",
    "Donanım Kodları",
    "Yaş Bilgisi",
    "Orijin (Baba - Anne)",
    "Baba",
    "Baba Linki",
    "Anne",
    "Anne Linki",
    "Kısrak Babası",
    "Kısrak Babası Linki",
    "Sıklet",
    "Jokey",
    "Jokey Linki",
    "Sahip",
    "Sahip Linki",
    "Antrenör",
    "Antrenör Linki",
    "Start No",
    "Handikap Puanı",
    "Son 6 Yarış",
    "KGS",
    "s20",
    "En İyi Derece",
    "En İyi Derece Linki",
    "En İyi Derece Açıklaması",
    "Ganyan",
    "AGF",
    "AGF Açıklaması",
    "İdman Linki",
    "W_Workout_Count",
    "W_Latest_Date",
    "W_Latest_Hip",
    "W_800m_Best",
    "W_600m_Best",
    "W_Match_Score",
    "W_Match_Method",
    "W_Data_Quality",
]

REQUIRED_COLUMNS = ["Tarih", "Hipodrom", "Koşu Saati", "Mesafe", "Pist Tipi", "At İsmi"]


class ProgramData:
    def __init__(self, frame: pd.DataFrame, errors: List[Dict[str, object]]):
        self.frame = frame
        self.errors = errors


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


def read_program_csv(path: str) -> ProgramData:
    df = pd.read_csv(path, dtype=str, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    df = _map_columns(df)

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Eksik zorunlu kolonlar: {missing_required}")

    errors: List[Dict[str, object]] = []
    records: List[Dict[str, object]] = []
    invalid_race_keys: set = set()
    start_counters: defaultdict[str, int] = defaultdict(int)

    for idx, row in df.iterrows():
        row_idx = int(idx)
        raw_date = row.get("Tarih")
        iso_date = parse_date(raw_date)
        if not iso_date:
            errors.append({"row": row_idx, "reason": "invalid_date", "value": raw_date})
            continue

        hipodrom = _clean_str(row.get("Hipodrom"))
        if not hipodrom:
            errors.append({"row": row_idx, "reason": "missing_hipodrom", "value": row.get("Hipodrom")})
            continue

        raw_race_id = _clean_str(row.get("Koşu ID"))
        raw_race_no = _clean_str(row.get("Koşu Numarası"))
        race_key = raw_race_id or raw_race_no
        if not race_key:
            race_key = f"race{row_idx}"
        base_race_key = (iso_date, hipodrom, race_key)
        if base_race_key in invalid_race_keys:
            continue

        raw_time = row.get("Koşu Saati")
        time_value = parse_time(raw_time)
        if not time_value:
            errors.append({"row": row_idx, "reason": "invalid_time", "value": raw_time})
            invalid_race_keys.add(base_race_key)
            continue

        distance = normalize_distance(row.get("Mesafe"))
        if distance is None:
            errors.append({"row": row_idx, "reason": "invalid_distance", "value": row.get("Mesafe")})
            invalid_race_keys.add(base_race_key)
            continue

        pist_tipi_raw = _clean_str(row.get("Pist Tipi")).lower()
        pist_tipi_map = {"çim": "cim", "cim": "cim", "kum": "kum", "sentetik": "sentetik"}
        pist_tipi = pist_tipi_map.get(pist_tipi_raw, None)
        if pist_tipi is None:
            errors.append({"row": row_idx, "reason": "invalid_track", "value": row.get("Pist Tipi")})
            invalid_race_keys.add(base_race_key)
            continue

        pist_durumu = None
        if pist_tipi == "cim":
            pist_durumu = _clean_str(row.get("Çim Durumu")) or None
        else:
            pist_durumu = _clean_str(row.get("Kum Durumu")) or None

        horse_name = _clean_str(row.get("At İsmi"))
        if not horse_name:
            errors.append({"row": row_idx, "reason": "missing_horse", "value": row.get("At İsmi")})
            continue

        race_uid_component = raw_race_id if raw_race_id else raw_race_no
        if not race_uid_component:
            race_uid_component = slugify(_clean_str(row.get("Koşu Başlık")) or race_key)
        race_uid = f"{iso_date}_{slugify(hipodrom)}_{race_uid_component}"

        start_no_raw = row.get("Start No")
        start_no = parse_int(start_no_raw)
        if start_no is None:
            start_counters[race_uid] += 1
            start_no = None
            start_tag = f"x{start_counters[race_uid]}"
        else:
            start_tag = str(start_no)

        horse_slug = slugify(horse_name)
        horse_uid = f"{race_uid}-{start_tag}-{horse_slug}"

        ganyan = parse_float(row.get("Ganyan"))
        if ganyan is not None and ganyan > 250:
            ganyan = None
        implied_prob = None
        if ganyan and ganyan > 0:
            implied_prob = 1.0 / ganyan

        agf_01 = parse_agf(row.get("AGF"))

        en_iyi_derece_s = parse_best_time(row.get("En İyi Derece"))
        en_iyi_derece_hist = parse_best_time(row.get("En İyi Derece (Tarihçe)"))

        yas = parse_int(row.get("Yaş Bilgisi"))
        siklet = parse_float(row.get("Sıklet"))
        handikap = parse_float(row.get("Handikap Puanı"))
        kgs = parse_float(row.get("KGS"))
        s20 = parse_float(row.get("s20"))

        donanim = _clean_str(row.get("Donanım Kodları"))
        has_kg = 1 if "KG" in donanim.upper() else 0

        genealogy_tokens = [
            token
            for token in [
                genealogy_token(row.get("Baba")),
                genealogy_token(row.get("Anne")),
                genealogy_token(row.get("Kısrak Babası")),
            ]
            if token
        ]

        record = {
            "row_index": row_idx,
            "race_uid": race_uid,
            "race_key": base_race_key,
            "race_date": iso_date,
            "hipodrom": hipodrom,
            "kosu_id": raw_race_id or None,
            "kosu_no": raw_race_no or None,
            "kosu_saati": time_value,
            "kosu_sinifi": _clean_str(row.get("Koşu Sınıfı")) or None,
            "mesafe": distance,
            "pist_tipi": pist_tipi,
            "pist_durumu": pist_durumu,
            "hava_durumu": _clean_str(row.get("Hava Durumu")) or None,
            "hava_sicakligi": parse_float(row.get("Hava Sıcaklığı")),
            "hava_nem": parse_float(row.get("Nem")),
            "baz_siklet": parse_float(row.get("Baz Sıklet")),
            "kosu_kosullari": _clean_str(row.get("Koşu Koşulları")) or None,
            "program_sirasi": parse_int(row.get("Program Sırası")),
            "at_ismi": horse_name,
            "horse_uid": horse_uid,
            "start_no": start_no,
            "start_tag": start_tag,
            "ganyan": ganyan,
            "implied_prob": implied_prob,
            "agf_01": agf_01,
            "en_iyi_derece_s": en_iyi_derece_s,
            "en_iyi_derece_hist_s": en_iyi_derece_hist,
            "yas": yas,
            "siklet": siklet,
            "handikap_puani": handikap,
            "kgs": kgs,
            "s20": s20,
            "son6": _clean_str(row.get("Son 6 Yarış")) or None,
            "donanim": donanim or None,
            "has_KG": has_kg,
            "jokey": _clean_str(row.get("Jokey")) or None,
            "sahip": _clean_str(row.get("Sahip")) or None,
            "antrenor": _clean_str(row.get("Antrenör")) or None,
            "genealogy_tokens": genealogy_tokens,
            "w_workout_count": parse_int(row.get("W_Workout_Count")),
            "w_latest_date": parse_date(row.get("W_Latest_Date")),
            "w_latest_hip": _clean_str(row.get("W_Latest_Hip")) or None,
            "w_800m_best": parse_float(row.get("W_800m_Best")),
            "w_600m_best": parse_float(row.get("W_600m_Best")),
            "w_match_score": parse_float(row.get("W_Match_Score")),
            "w_match_method": _clean_str(row.get("W_Match_Method")) or None,
            "w_data_quality": _clean_str(row.get("W_Data_Quality")) or None,
        }
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    if not frame.empty and invalid_race_keys:
        mask = frame["race_key"].apply(lambda x: tuple(x) in invalid_race_keys)
        frame = frame[~mask]
        frame = frame.drop(columns=["race_key"])
    elif "race_key" in frame.columns:
        frame = frame.drop(columns=["race_key"])
    return ProgramData(frame=frame, errors=errors)
