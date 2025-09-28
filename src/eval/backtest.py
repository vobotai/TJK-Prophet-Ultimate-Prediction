"""Walk-forward evaluation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    cutoff: str


def time_based_split(dates: Iterable[str], val_date: str) -> Split:
    parsed = pd.to_datetime(list(dates))
    cutoff = pd.to_datetime(val_date)
    train_idx = np.where(parsed < cutoff)[0]
    val_idx = np.where(parsed >= cutoff)[0]
    return Split(train_idx=train_idx, val_idx=val_idx, cutoff=val_date)


def walk_forward_splits(dates: Iterable[str], n_splits: int = 3) -> List[Split]:
    sorted_dates = sorted(set(dates))
    if len(sorted_dates) < n_splits + 1:
        n_splits = max(1, len(sorted_dates) - 1)
    splits: List[Split] = []
    for i in range(1, n_splits + 1):
        cutoff = sorted_dates[-i]
        splits.append(time_based_split(dates, cutoff))
    return splits
