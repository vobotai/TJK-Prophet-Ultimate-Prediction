"""CPU friendly XGBoost wrapper with graceful fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import xgboost as xgb  # type: ignore
except ImportError:  # pragma: no cover
    xgb = None  # type: ignore

try:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingClassifier
except ImportError:  # pragma: no cover
    GradientBoostingClassifier = None  # type: ignore


@dataclass
class XGBWrapper:
    params: Optional[Dict[str, Any]] = None
    model: Any = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> None:
        params = {
            "tree_method": "hist",
            "max_depth": 7,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 600,
            "eval_metric": "logloss",
        }
        if self.params:
            params.update(self.params)
        if xgb is not None:
            booster_params = params.copy()
            n_estimators = booster_params.pop("n_estimators", 600)
            booster = xgb.XGBClassifier(
                n_estimators=n_estimators,
                tree_method=booster_params.pop("tree_method", "hist"),
                max_depth=booster_params.pop("max_depth", 7),
                eta=booster_params.pop("eta", 0.05),
                subsample=booster_params.pop("subsample", 0.8),
                colsample_bytree=booster_params.pop("colsample_bytree", 0.8),
                eval_metric="logloss",
                objective="binary:logistic",
                random_state=42,
                **booster_params,
            )
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
            booster.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            self.model = booster
        else:
            if GradientBoostingClassifier is None:
                raise ImportError("Neither xgboost nor sklearn is available")
            booster = GradientBoostingClassifier(random_state=42)
            booster.fit(X_train, y_train)
            self.model = booster

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained")
        proba = self.model.predict_proba(X)
        if proba.ndim == 1:
            proba = np.vstack([1 - proba, proba]).T
        return proba
