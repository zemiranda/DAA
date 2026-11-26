"""Gradient boosting model wrappers (LightGBM, XGBoost, CatBoost + hierarchical)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from src.models.base import align_proba, run_hierarchical, to_ordinal_arrays


def _identity_df(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pass DataFrames directly for LightGBM native categorical + NaN support."""
    return X_tr, X_va, X_te


def fit_predict_lgb(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Train LightGBM classifier and return (val_probs, test_probs)."""
    import lightgbm as lgb

    model_params = {**params, "random_state": seed, "verbose": -1, "n_jobs": -1}
    model = lgb.LGBMClassifier(**model_params)
    model.fit(X_tr, y_tr)
    num_classes = int(meta["num_classes"])
    return (
        align_proba(model.predict_proba(X_va), model.classes_, num_classes),
        align_proba(model.predict_proba(X_te), model.classes_, num_classes),
    )


def fit_predict_xgb(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Train XGBoost classifier with fold-local median imputation and balanced weights."""
    import xgboost as xgb

    M_tr, M_va, M_te = to_ordinal_arrays(X_tr, X_va, X_te, meta["cat_cols"], meta["num_cols"])
    model_params = {**params, "random_state": seed, "n_jobs": -1, "verbosity": 0}
    model = xgb.XGBClassifier(**model_params)
    sw = compute_sample_weight("balanced", y_tr)
    model.fit(M_tr, y_tr, sample_weight=sw)
    num_classes = int(meta["num_classes"])
    return (
        align_proba(model.predict_proba(M_va), model.classes_, num_classes),
        align_proba(model.predict_proba(M_te), model.classes_, num_classes),
    )


def fit_predict_cat(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Train CatBoost classifier with fold-local ordinal/median preprocessing."""
    from catboost import CatBoostClassifier

    M_tr, M_va, M_te = to_ordinal_arrays(X_tr, X_va, X_te, meta["cat_cols"], meta["num_cols"])
    model_params = {**params, "random_seed": seed, "verbose": 0, "thread_count": -1}
    model = CatBoostClassifier(**model_params)
    model.fit(M_tr, y_tr)
    num_classes = int(meta["num_classes"])
    return (
        align_proba(model.predict_proba(M_va), model.classes_, num_classes),
        align_proba(model.predict_proba(M_te), model.classes_, num_classes),
    )


def fit_predict_hlgb(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Train hierarchical (<55 vs 55+) LightGBM classifiers."""
    import lightgbm as lgb

    model_params = {**params, "random_state": seed, "verbose": -1, "n_jobs": -1}
    return run_hierarchical(
        X_tr,
        y_tr,
        X_va,
        X_te,
        make_estimator=lambda: lgb.LGBMClassifier(**model_params),
        prepare_fn=_identity_df,
        meta=meta,
        use_sample_weight=False,
    )


def fit_predict_hxgb(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Train hierarchical (<55 vs 55+) XGBoost classifiers."""
    import xgboost as xgb

    model_params = {**params, "random_state": seed, "n_jobs": -1, "verbosity": 0}
    return run_hierarchical(
        X_tr,
        y_tr,
        X_va,
        X_te,
        make_estimator=lambda: xgb.XGBClassifier(**model_params),
        prepare_fn=to_ordinal_arrays,
        meta=meta,
        use_sample_weight=True,
    )


def fit_predict_hcat(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Train hierarchical (<55 vs 55+) CatBoost classifiers."""
    from catboost import CatBoostClassifier

    model_params = {**params, "random_seed": seed, "verbose": 0, "thread_count": -1}
    return run_hierarchical(
        X_tr,
        y_tr,
        X_va,
        X_te,
        make_estimator=lambda: CatBoostClassifier(**model_params),
        prepare_fn=to_ordinal_arrays,
        meta=meta,
        use_sample_weight=False,
    )
