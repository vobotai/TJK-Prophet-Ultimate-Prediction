"""Calibration utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np

try:  # pragma: no cover
    from sklearn.isotonic import IsotonicRegression
except ImportError:  # pragma: no cover
    IsotonicRegression = None  # type: ignore


@dataclass
class CalibrationResult:
    method: Literal["temperature", "isotonic"]
    param: Optional[float | object]

    def apply(self, probs: np.ndarray) -> np.ndarray:
        if self.method == "temperature":
            temp = float(self.param or 1.0)
            logits = np.log(np.clip(probs, 1e-6, 1 - 1e-6) / np.clip(1 - probs, 1e-6, 1))
            logits = logits / max(temp, 1e-6)
            calibrated = 1 / (1 + np.exp(-logits))
            return calibrated
        elif self.method == "isotonic" and isinstance(self.param, IsotonicRegression):
            model = self.param
            flat = probs.reshape(-1)
            calibrated = model.predict(flat)
            return calibrated.reshape(probs.shape)
        return probs


def fit_temperature_scaling(probs: np.ndarray, targets: np.ndarray) -> CalibrationResult:
    temps = np.linspace(0.5, 3.0, 26)
    best_temp = 1.0
    best_loss = float("inf")
    for temp in temps:
        logits = np.log(np.clip(probs, 1e-6, 1 - 1e-6) / np.clip(1 - probs, 1e-6, 1))
        scaled = 1 / (1 + np.exp(-logits / temp))
        loss = -np.mean(targets * np.log(np.clip(scaled, 1e-6, 1)) + (1 - targets) * np.log(np.clip(1 - scaled, 1e-6, 1)))
        if loss < best_loss:
            best_loss = loss
            best_temp = temp
    return CalibrationResult(method="temperature", param=best_temp)


def fit_isotonic(probs: np.ndarray, targets: np.ndarray) -> CalibrationResult:
    if IsotonicRegression is None:
        return CalibrationResult(method="temperature", param=1.0)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, targets)
    return CalibrationResult(method="isotonic", param=iso)


def choose_best_calibrator(probs: np.ndarray, targets: np.ndarray) -> CalibrationResult:
    temp = fit_temperature_scaling(probs, targets)
    isotonic = fit_isotonic(probs, targets)
    temp_loss = _brier(temp.apply(probs), targets)
    iso_loss = _brier(isotonic.apply(probs), targets)
    return temp if temp_loss <= iso_loss else isotonic


def _brier(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((preds - targets) ** 2))
