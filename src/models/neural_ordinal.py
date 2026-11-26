"""Linear, neural (MLP), and ordinal regression model wrappers."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.models.base import align_proba, to_onehot_arrays, to_ordinal_arrays


class OrdinalLGB:
    """Continuous age-group regressor converted to Gaussian class probabilities."""

    def __init__(self, params: dict[str, Any], seed: int, num_classes: int) -> None:
        import lightgbm as lgb

        cfg = dict(params)
        self.sigma = float(cfg.pop("sigma", 1.3))
        self.num_classes = num_classes
        self.model = lgb.LGBMRegressor(**cfg, random_state=seed, verbose=-1, n_jobs=-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinalLGB":
        """Fit regressor on integer age-group indices."""
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Project continuous age prediction onto discrete class probabilities."""
        mu = np.clip(self.model.predict(X), 0, self.num_classes - 1)[:, None]
        grid = np.arange(self.num_classes, dtype=np.float32)[None, :]
        probs = np.exp(-((grid - mu) ** 2) / (2.0 * (self.sigma ** 2)))
        return (probs / probs.sum(axis=1, keepdims=True)).astype(np.float32)


def fit_predict_lr(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Train LogisticRegression with fold-local one-hot + scaled numeric features."""
    M_tr, M_va, M_te = to_onehot_arrays(X_tr, X_va, X_te, meta["cat_cols"], meta["num_cols"])
    model = LogisticRegression(**params, random_state=seed)
    model.fit(M_tr, y_tr)
    num_classes = int(meta["num_classes"])
    return (
        align_proba(model.predict_proba(M_va), model.classes_, num_classes),
        align_proba(model.predict_proba(M_te), model.classes_, num_classes),
    )


def fit_predict_mlp(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Train MLPClassifier with fold-local one-hot + scaled numeric features."""
    M_tr, M_va, M_te = to_onehot_arrays(X_tr, X_va, X_te, meta["cat_cols"], meta["num_cols"])
    cfg = dict(params)
    if "hidden_layer_sizes" in cfg:
        cfg["hidden_layer_sizes"] = tuple(cfg["hidden_layer_sizes"])
    model = MLPClassifier(**cfg, random_state=seed)
    model.fit(M_tr, y_tr)
    num_classes = int(meta["num_classes"])
    return (
        align_proba(model.predict_proba(M_va), model.classes_, num_classes),
        align_proba(model.predict_proba(M_te), model.classes_, num_classes),
    )


def fit_predict_ord(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Train OrdinalLGB regressor with fold-local median imputation."""
    M_tr, M_va, M_te = to_ordinal_arrays(X_tr, X_va, X_te, meta["cat_cols"], meta["num_cols"])
    num_classes = int(meta["num_classes"])
    model = OrdinalLGB(params=params, seed=seed, num_classes=num_classes)
    model.fit(M_tr, y_tr)
    return model.predict_proba(M_va), model.predict_proba(M_te)
