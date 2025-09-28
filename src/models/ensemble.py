"""Context gated ensemble combiner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

try:  # pragma: no cover
    from sklearn.linear_model import LogisticRegression
except ImportError:  # pragma: no cover
    LogisticRegression = None  # type: ignore


def _context_to_vector(context: Dict[str, object]) -> List[float]:
    return [
        float(context.get("field_size") or 0.0),
        float(context.get("mesafe") or 0.0),
        1.0 if (context.get("pist_tipi") or "") == "cim" else 0.0,
        1.0 if (context.get("pist_tipi") or "") == "kum" else 0.0,
        1.0 if (context.get("pist_tipi") or "") == "sentetik" else 0.0,
        float(context.get("median_handikap") or 0.0),
        float(context.get("median_en_iyi_derece_s") or 0.0),
        float(context.get("median_agf_01") or 0.0),
    ]


@dataclass
class ContextGatedEnsemble:
    model: object | None = None

    def fit(self, base_outputs: np.ndarray, race_contexts: List[Dict[str, object]], targets: np.ndarray) -> None:
        context_vectors = np.array([_context_to_vector(ctx) for ctx in race_contexts])
        intercept = np.ones((base_outputs.shape[0], 1))
        design = np.hstack([base_outputs, context_vectors, intercept])
        y = targets[:, 1]
        if LogisticRegression is not None:
            clf = LogisticRegression(max_iter=200)
            clf.fit(design, y)
            self.model = ("logreg", clf)
        else:
            theta, *_ = np.linalg.lstsq(design, y, rcond=None)
            self.model = ("linear", theta)

    def combine(self, base_predictions: List[np.ndarray], race_contexts: List[Dict[str, object]]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Ensemble not trained")
        base = np.column_stack(base_predictions)
        context_vectors = np.array([_context_to_vector(ctx) for ctx in race_contexts])
        intercept = np.ones((base.shape[0], 1))
        design = np.hstack([base, context_vectors, intercept])
        kind, model = self.model
        if kind == "logreg":
            probs = model.predict_proba(design)[:, 1]
        else:
            theta = model
            preds = design @ theta
            probs = 1 / (1 + np.exp(-preds))
        return probs
