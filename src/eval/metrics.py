"""Evaluation metrics utilities."""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
except ImportError:  # pragma: no cover
    average_precision_score = None  # type: ignore
    brier_score_loss = None  # type: ignore
    log_loss = None  # type: ignore
    roc_auc_score = None  # type: ignore


def auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if roc_auc_score is None:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def pr_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if average_precision_score is None:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if brier_score_loss is None:
        return float(np.mean((y_true - y_prob) ** 2))
    return float(brier_score_loss(y_true, y_prob))


def log_loss_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if log_loss is None:
        eps = 1e-6
        prob = np.clip(y_prob, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(prob) + (1 - y_true) * np.log(1 - prob)))
    try:
        return float(log_loss(y_true, y_prob))
    except ValueError:
        return float("nan")


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 3) -> float:
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order][:k]
    gains = (2 ** y_true_sorted - 1) / np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains)
    ideal_order = np.argsort(-y_true)[:k]
    ideal = y_true[ideal_order]
    ideal_gains = (2 ** ideal - 1) / np.log2(np.arange(2, k + 2))
    idcg = np.sum(ideal_gains)
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        ece += np.abs(acc - conf) * np.mean(mask)
    return float(ece)


def topk_lift(y_true: np.ndarray, y_prob: np.ndarray, k: int = 3) -> float:
    if len(y_true) == 0:
        return float("nan")
    baseline = float(np.mean(y_true))
    if baseline == 0:
        return float("nan")
    top_idx = np.argsort(-y_prob)[:k]
    top_rate = float(np.mean(y_true[top_idx]))
    return top_rate / baseline


def edge_statistics(win_prob: np.ndarray, implied_prob: np.ndarray) -> Dict[str, float]:
    diff = win_prob - np.nan_to_num(implied_prob, nan=0.0)
    return {
        "edge_mean": float(np.nanmean(diff)),
        "edge_median": float(np.nanmedian(diff)),
        "edge_positive_rate": float(np.mean(diff > 0)),
    }
