from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from dataio.merge import merge_program_and_workouts
from dataio.read_program import read_program_csv
from dataio.read_workouts import read_workouts_csv
from eval.backtest import Split, time_based_split
from eval.metrics import (
    auc_score,
    brier_score,
    edge_statistics,
    expected_calibration_error,
    log_loss_score,
    ndcg_at_k,
    pr_auc_score,
)
from features.gate_context import compute_gate_and_context
from features.market_features import compute_market_features
from features.set_features import compute_set_features
from models.calibrate import CalibrationResult, choose_best_calibrator
from models.catb import CatBoostWrapper
from models.ensemble import ContextGatedEnsemble
from models.lgbm import LGBMWrapper
from models.set_mlp import SetMLPWrapper
from models.xgb import XGBWrapper


NUMERIC_FILL = 0.0


def build_targets(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    targets: Dict[str, np.ndarray] = {}
    if "Result_Win" in frame.columns:
        targets["win"] = frame["Result_Win"].fillna(0).astype(float).values
    else:
        win_guess = frame.groupby("race_uid")["ganyan"].transform("min")
        targets["win"] = (frame["ganyan"] == win_guess).astype(float).fillna(0.0).values

    if "Result_Place" in frame.columns:
        targets["place"] = frame["Result_Place"].fillna(0).astype(float).values
    else:
        ranks = frame.groupby("race_uid")["ganyan"].rank(method="dense")
        targets["place"] = (ranks <= 3).astype(float).values

    if "Finish_Position" in frame.columns:
        targets["finish"] = frame["Finish_Position"].fillna(frame.groupby("race_uid")["Finish_Position"].transform("median")).values
    else:
        targets["finish"] = frame.groupby("race_uid")["ganyan"].rank(method="dense").values

    if "Race_Time" in frame.columns:
        targets["race_time"] = frame["Race_Time"].fillna(frame.groupby("race_uid")["Race_Time"].transform("median")).values
    else:
        targets["race_time"] = frame["en_iyi_derece_s"].fillna(frame.groupby("race_uid")["en_iyi_derece_s"].transform("median")).fillna(90.0).values
    return targets


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = compute_set_features(frame)
    frame = compute_market_features(frame)
    frame = compute_gate_and_context(frame)
    frame["genealogy_count"] = frame["genealogy_tokens"].apply(lambda tokens: len(tokens) if isinstance(tokens, list) else 0)
    frame["has_workout"] = frame["workout_sequence"].apply(lambda seq: 1 if seq else 0)
    return frame


def select_feature_matrix(frame: pd.DataFrame) -> (np.ndarray, List[str]):
    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"Result_Win", "Result_Place", "Finish_Position", "Race_Time"}
    numeric_cols = [c for c in numeric_cols if c not in exclude]
    X = frame[numeric_cols].fillna(NUMERIC_FILL).values
    return X, numeric_cols


def train_models(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None, input_dim: int = 10):
    models = {}
    xgb_model = XGBWrapper()
    xgb_model.fit(X_train, y_train, X_val, y_val)
    models["xgb"] = xgb_model

    lgbm_model = LGBMWrapper()
    lgbm_model.fit(X_train, y_train, X_val, y_val)
    models["lgbm"] = lgbm_model

    cat_model = CatBoostWrapper()
    cat_model.fit(X_train, y_train, X_val, y_val)
    models["catboost"] = cat_model

    mlp_model = SetMLPWrapper(input_dim=input_dim)
    mlp_model.fit(X_train, y_train, X_val, y_val)
    models["set_mlp"] = mlp_model

    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--workouts", type=Path, default=None)
    parser.add_argument("--val-date", type=str, required=True)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/model.pkl"))
    parser.add_argument("--meta-out", type=Path, default=Path("artifacts/train_meta.json"))
    args = parser.parse_args()

    program = read_program_csv(args.program)
    workouts = read_workouts_csv(args.workouts) if args.workouts else None
    merged = merge_program_and_workouts(program, workouts)
    enriched = build_features(merged.frame)

    targets = build_targets(enriched)
    X, feature_columns = select_feature_matrix(enriched)

    split = time_based_split(enriched["race_date"].tolist(), args.val_date)
    total_indices = np.arange(len(X))
    if len(split.train_idx) == 0:
        if len(total_indices) > 1:
            train_idx = total_indices[:-1]
            val_idx = total_indices[-1:]
        else:
            train_idx = total_indices
            val_idx = total_indices
        split = Split(train_idx=train_idx, val_idx=val_idx, cutoff=split.cutoff)
    elif len(split.val_idx) == 0:
        val_size = max(1, int(0.2 * len(total_indices)))
        val_idx = total_indices[-val_size:]
        train_idx = np.setdiff1d(total_indices, val_idx)
        split = Split(train_idx=train_idx, val_idx=val_idx, cutoff=split.cutoff)

    X_train = X[split.train_idx]
    y_train = targets["win"][split.train_idx]
    X_val = X[split.val_idx] if len(split.val_idx) else None
    y_val = targets["win"][split.val_idx] if len(split.val_idx) else None

    models = train_models(X_train, y_train, X_val, y_val, input_dim=X.shape[1])

    base_preds = []
    for model in models.values():
        proba = model.predict_proba(X)
        base_preds.append(proba[:, 1])
    base_matrix = np.stack(base_preds, axis=1)

    race_contexts = enriched["race_context"].tolist()
    ensemble = ContextGatedEnsemble()
    ensemble.fit(base_matrix, race_contexts, np.vstack([1 - targets["win"], targets["win"]]).T)
    combined = ensemble.combine(base_preds, race_contexts)

    calibrator = choose_best_calibrator(combined[split.val_idx], targets["win"][split.val_idx]) if len(split.val_idx) else CalibrationResult("temperature", 1.0)

    calibrated = calibrator.apply(combined)

    metrics = {
        "auc": auc_score(targets["win"], calibrated),
        "pr_auc": pr_auc_score(targets["win"], calibrated),
        "brier": brier_score(targets["win"], calibrated),
        "logloss": log_loss_score(targets["win"], calibrated),
        "ndcg@3": ndcg_at_k(targets["win"], calibrated, k=3),
        "ece": expected_calibration_error(targets["win"], calibrated),
    }
    edges = edge_statistics(calibrated, enriched["implied_prob"].values)
    metrics.update(edges)

    artifact = {
        "feature_columns": feature_columns,
        "models": models,
        "ensemble": ensemble,
        "calibrator": calibrator,
        "metrics": metrics,
        "meta": {
            "val_date": args.val_date,
            "train_size": int(len(split.train_idx)),
            "val_size": int(len(split.val_idx)),
            "random_seed": 42,
        },
    }

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    with open(args.artifact, "wb") as f:
        pickle.dump(artifact, f)

    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.write_text(json.dumps(metrics, indent=2))

    print(json.dumps({"status": "ok", "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
