from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from dataio.merge import merge_program_and_workouts
from dataio.read_program import read_program_csv
from dataio.read_workouts import read_workouts_csv
from features.gate_context import compute_gate_and_context
from features.market_features import compute_market_features
from features.set_features import compute_set_features
from models.calibrate import CalibrationResult


artifact_calibration_method = "temperature"
artifact_calibration_param = None


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = compute_set_features(frame)
    frame = compute_market_features(frame)
    frame = compute_gate_and_context(frame)
    frame["genealogy_count"] = frame["genealogy_tokens"].apply(lambda tokens: len(tokens) if isinstance(tokens, list) else 0)
    frame["has_workout"] = frame["workout_sequence"].apply(lambda seq: 1 if seq else 0)
    return frame


def load_artifact(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_features(frame: pd.DataFrame, columns: List[str]) -> np.ndarray:
    for col in columns:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame[columns].fillna(0.0).values


def compute_predictions(artifact: Dict[str, Any], features: np.ndarray, contexts: List[Dict[str, object]]) -> np.ndarray:
    base_preds = []
    for model in artifact["models"].values():
        base = model.predict_proba(features)
        base_preds.append(base[:, 1])
    combined = artifact["ensemble"].combine(base_preds, contexts)
    calibrator: CalibrationResult = artifact["calibrator"]
    calibrated = calibrator.apply(combined)
    return np.clip(calibrated, 0.0, 1.0)


def race_summary(frame: pd.DataFrame, win_probs: np.ndarray) -> List[Dict[str, Any]]:
    frame = frame.copy()
    frame["win_prob"] = win_probs
    field_sizes = frame.groupby("race_uid")["race_uid"].transform("count")
    frame["place_prob"] = np.clip(np.sqrt(frame["win_prob"]), 0, 1)
    frame["expected_finish"] = 1 + (1 - frame["win_prob"]) * (field_sizes / 2)
    frame["race_time_pred"] = frame["en_iyi_derece_s"].fillna(frame.groupby("race_uid")["en_iyi_derece_s"].transform("median")).fillna(95.0)
    frame["edge"] = frame["win_prob"] - frame["implied_prob"].fillna(0.0)
    frame["win_std"] = np.sqrt(frame["win_prob"] * (1 - frame["win_prob"]))
    frame["place_std"] = np.sqrt(frame["place_prob"] * (1 - frame["place_prob"]))

    races: List[Dict[str, Any]] = []
    for race_uid, group in frame.groupby("race_uid"):
        group = group.sort_values(["win_prob", "expected_finish"], ascending=[False, True])
        first_row = group.iloc[0]
        race_entry = {
            "race_id": race_uid,
            "meta": {
                "hipodrom": first_row["hipodrom"],
                "tarih": first_row["race_date"],
                "kosu_saati": first_row["kosu_saati"],
                "kosu_sinifi": first_row["kosu_sinifi"],
                "mesafe": int(first_row["mesafe"]),
                "pist_tipi": first_row["pist_tipi"],
                "calibration": {
                    "method": artifact_calibration_method,
                    "param": artifact_calibration_param,
                },
            },
            "predictions": [],
        }
        for _, row in group.iterrows():
            race_entry["predictions"].append(
                {
                    "horse_id": row["horse_uid"],
                    "at_ismi": row["at_ismi"],
                    "start_no": int(row["start_no"]) if not pd.isna(row["start_no"]) else None,
                    "win_prob": float(row["win_prob"]),
                    "place_prob": float(row["place_prob"]),
                    "expected_finish": float(row["expected_finish"]),
                    "race_time": float(row["race_time_pred"]),
                    "uncertainty": {
                        "win_std": float(row["win_std"]),
                        "place_std": float(row["place_std"]),
                    },
                    "ganyan": float(row["ganyan"]) if not pd.isna(row["ganyan"]) else None,
                    "implied_prob": float(row["implied_prob"]) if not pd.isna(row["implied_prob"]) else None,
                    "edge": float(row["edge"]),
                    "mdi": float(row["mdi"]) if not pd.isna(row["mdi"]) else None,
                    "drift_dp15": float(row["drift_dp15"]) if not pd.isna(row["drift_dp15"]) else None,
                    "extras": {
                        "has_KG": int(row["has_KG"]),
                        "gate_rank_pct": float(row["gate_rank_pct"]) if not pd.isna(row["gate_rank_pct"]) else None,
                        "gate_context_key": row["gate_context_key"],
                    },
                }
            )
        races.append(race_entry)
    return races


def build_json_output(races: List[Dict[str, Any]], metrics: Dict[str, Any], errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for race in races:
        race["metrics"] = metrics
        race["errors"] = errors
    return races


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--workouts", type=Path, default=None)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/model.pkl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    artifact = load_artifact(args.artifact)
    global artifact_calibration_method, artifact_calibration_param
    calibrator = artifact["calibrator"]
    artifact_calibration_method = calibrator.method
    artifact_calibration_param = calibrator.param if isinstance(calibrator.param, (int, float)) else None

    program = read_program_csv(args.program)
    workouts = read_workouts_csv(args.workouts) if args.workouts else None
    merged = merge_program_and_workouts(program, workouts)
    enriched = build_features(merged.frame)

    X = ensure_features(enriched, artifact["feature_columns"])
    win_probs = compute_predictions(artifact, X, enriched["race_context"].tolist())

    races = race_summary(enriched, win_probs)
    json_output = build_json_output(races, artifact.get("metrics", {}), merged.errors)

    args.out.write_text(json.dumps(json_output, indent=2, ensure_ascii=False))

    if args.report:
        from .report import generate_report

        report_text = generate_report(json_output)
        args.report.write_text(report_text)


if __name__ == "__main__":
    main()
