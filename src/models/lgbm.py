"""LightGBM wrapper with CPU fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:  # pragma: no cover
    import lightgbm as lgb  # type: ignore
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore

try:  # pragma: no cover
    from sklearn.ensemble import RandomForestClassifier
except ImportError:  # pragma: no cover
    RandomForestClassifier = None  # type: ignore


@dataclass
class LGBMWrapper:
    params: Optional[Dict[str, Any]] = None
    model: Any = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> None:
        defaults = {
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "learning_rate": 0.05,
            "n_estimators": 600,
        }
        if self.params:
            defaults.update(self.params)
        if lgb is not None:
            booster = lgb.LGBMClassifier(
                objective="binary",
                random_state=42,
                **defaults,
            )
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
            booster.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            self.model = booster
        else:
            if RandomForestClassifier is None:
                raise ImportError("Neither lightgbm nor sklearn RandomForest available")
            forest = RandomForestClassifier(n_estimators=400, random_state=42)
            forest.fit(X_train, y_train)
            self.model = forest

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained")
        proba = self.model.predict_proba(X)
        if proba.ndim == 1:
            proba = np.vstack([1 - proba, proba]).T
        return proba
