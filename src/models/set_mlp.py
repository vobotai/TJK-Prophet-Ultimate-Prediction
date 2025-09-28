"""Set-style MLP encoder for CPU with optional PyTorch backend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:  # pragma: no cover
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore

try:  # pragma: no cover
    from sklearn.neural_network import MLPClassifier
except ImportError:  # pragma: no cover
    MLPClassifier = None  # type: ignore


if torch is not None:
    class _SetEncoder(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            layers = []
            last = input_dim
            for _ in range(num_layers):
                layers.append(nn.Linear(last, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                last = hidden_dim
            layers.append(nn.Linear(last, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
            return self.net(x).squeeze(-1)


@dataclass
class SetMLPWrapper:
    input_dim: int
    device: str = "cpu"
    model: Any = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> None:
        if torch is None:
            if MLPClassifier is None:
                raise ImportError("PyTorch ve sklearn MLP mevcut değil")
            mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4, learning_rate_init=3e-4, max_iter=500, random_state=42)
            mlp.fit(X_train, y_train)
            self.model = mlp
            return

        torch.manual_seed(42)
        model = _SetEncoder(self.input_dim, hidden_dim=128, num_layers=3, dropout=0.1)
        model.to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=3e-4)

        X_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        val_tensor = None
        val_target = None
        if X_val is not None and y_val is not None:
            val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            val_target = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        best_loss = float("inf")
        best_state = None
        for epoch in range(1, 151):
            model.train()
            optimizer.zero_grad()
            logits = model(X_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()

            if val_tensor is not None:
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(val_tensor), val_target)
                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    best_state = model.state_dict()
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained")
        if torch is None or not isinstance(self.model, nn.Module):
            proba = self.model.predict_proba(X)
            if proba.ndim == 1:
                proba = np.vstack([1 - proba, proba]).T
            return proba
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.vstack([1 - probs, probs]).T

    # TODO: TRT-FP16 export path for RTX 4060 deployment.
