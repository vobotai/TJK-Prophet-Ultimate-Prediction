"""CatBoost wrapper with CPU fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:  # pragma: no cover
    from catboost import CatBoostClassifier  # type: ignore
except ImportError:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore

try:  # pragma: no cover
    from sklearn.ensemble import ExtraTreesClassifier
except ImportError:  # pragma: no cover
    ExtraTreesClassifier = None  # type: ignore


@dataclass
class CatBoostWrapper:
    params: Optional[Dict[str, Any]] = None
    model: Any = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> None:
        defaults = {
            "depth": 8,
            "l2_leaf_reg": 3,
            "iterations": 2000,
            "learning_rate": 0.05,
        }
        if self.params:
            defaults.update(self.params)
        if CatBoostClassifier is not None:
            booster = CatBoostClassifier(
                task_type="CPU",
                loss_function="Logloss",
                random_seed=42,
                verbose=False,
                **defaults,
            )
            if X_val is not None and y_val is not None:
                booster.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
            else:
                booster.fit(X_train, y_train)
            self.model = booster
        else:
            if ExtraTreesClassifier is None:
                raise ImportError("CatBoost ve ExtraTrees bulunamadı")
            forest = ExtraTreesClassifier(n_estimators=600, random_state=42)
            forest.fit(X_train, y_train)
            self.model = forest

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained")
        proba = self.model.predict_proba(X)
        if isinstance(proba, list):
            proba = np.array(proba)
        if proba.ndim == 1:
            proba = np.vstack([1 - proba, proba]).T
        return proba
