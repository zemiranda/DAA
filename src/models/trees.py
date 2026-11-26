"""Bagging tree model wrappers (Random Forest, ExtraTrees + hierarchical RF)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from src.models.base import align_proba, run_hierarchical, to_ordinal_arrays


def fit_predict_rf(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Train RandomForestClassifier with fold-local imputation."""
    M_tr, M_va, M_te = to_ordinal_arrays(X_tr, X_va, X_te, meta["cat_cols"], meta["num_cols"])
    model = RandomForestClassifier(**params, random_state=seed, n_jobs=-1)
    model.fit(M_tr, y_tr)
    num_classes = int(meta["num_classes"])
    return (
        align_proba(model.predict_proba(M_va), model.classes_, num_classes),
        align_proba(model.predict_proba(M_te), model.classes_, num_classes),
    )


def fit_predict_et(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Train ExtraTreesClassifier with fold-local imputation."""
    M_tr, M_va, M_te = to_ordinal_arrays(X_tr, X_va, X_te, meta["cat_cols"], meta["num_cols"])
    model = ExtraTreesClassifier(**params, random_state=seed, n_jobs=-1)
    model.fit(M_tr, y_tr)
    num_classes = int(meta["num_classes"])
    return (
        align_proba(model.predict_proba(M_va), model.classes_, num_classes),
        align_proba(model.predict_proba(M_te), model.classes_, num_classes),
    )


def fit_predict_hrf(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Train hierarchical (<55 vs 55+) RandomForest classifiers."""
    return run_hierarchical(
        X_tr,
        y_tr,
        X_va,
        X_te,
        make_estimator=lambda: RandomForestClassifier(**params, random_state=seed, n_jobs=-1),
        prepare_fn=to_ordinal_arrays,
        meta=meta,
        use_sample_weight=False,
    )
