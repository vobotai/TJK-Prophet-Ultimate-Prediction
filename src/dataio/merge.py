"""Merging program and workout information."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .read_program import ProgramData


def merge_program_and_workouts(program: ProgramData, workouts: pd.DataFrame | None) -> ProgramData:
    frame = program.frame.copy()
    if workouts is None or workouts.empty:
        frame["workout_sequence"] = [[] for _ in range(len(frame))]
        return ProgramData(frame=frame, errors=program.errors)

    workout_groups: Dict[Tuple[str, str, str | None], List[Dict[str, object]]] = defaultdict(list)
    for record in workouts.to_dict("records"):
        tarih = record.get("tarih")
        hipodrom = record.get("hipodrom")
        kosu_id = record.get("kosu_id") or None
        if not tarih or not hipodrom:
            continue
        key = (tarih, hipodrom, kosu_id)
        workout_groups[key].append(record)

    sequences: List[List[Dict[str, object]]] = []
    for _, row in frame.iterrows():
        tarih = row.get("race_date")
        hipodrom = row.get("hipodrom")
        kosu_id = row.get("kosu_id") or None
        race_key = (tarih, hipodrom, kosu_id)
        candidates = workout_groups.get(race_key, [])
        if not candidates and kosu_id is None:
            candidates = workout_groups.get((tarih, hipodrom, None), [])

        name = (row.get("at_ismi") or "").strip().lower()
        start_no = row.get("start_no")
        matches: List[Dict[str, object]] = []
        for cand in candidates:
            cand_name = (cand.get("at_ismi") or "").strip().lower()
            score = cand.get("match_score")
            if cand_name == name or (score is not None and score >= 85):
                matches.append({
                    "workout_date": cand.get("workout_date"),
                    "match_score": score,
                    "match_method": cand.get("match_method"),
                    "w1200_s": cand.get("w1200_s"),
                    "w1000_s": cand.get("w1000_s"),
                    "w800_s": cand.get("w800_s"),
                    "w600_s": cand.get("w600_s"),
                    "w400_s": cand.get("w400_s"),
                    "w200_s": cand.get("w200_s"),
                    "w_type": cand.get("w_type"),
                    "w_status": cand.get("w_status"),
                })

        matches = sorted(
            [m for m in matches if m.get("workout_date")],
            key=lambda m: m["workout_date"],
            reverse=True,
        ) or matches

        if (row.get("w_data_quality") or "").lower() != "matched":
            matches = []
        sequences.append(matches)

    frame["workout_sequence"] = sequences
    return ProgramData(frame=frame, errors=program.errors)
