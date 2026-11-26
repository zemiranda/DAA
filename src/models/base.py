"""Shared fold-local preprocessing helpers and unified model execution contracts."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


def to_ordinal_arrays(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit median imputer on X_tr only and convert categories to integer codes."""
    imputer = SimpleImputer(strategy="median")
    num_tr = imputer.fit_transform(X_tr[num_cols])
    num_va = imputer.transform(X_va[num_cols])
    num_te = imputer.transform(X_te[num_cols])

    def _combine(num_arr: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        if not cat_cols:
            return num_arr.astype(np.float32)
        codes = np.column_stack([df[c].cat.codes.to_numpy(dtype=np.float32) for c in cat_cols])
        return np.hstack([num_arr, codes]).astype(np.float32)

    return _combine(num_tr, X_tr), _combine(num_va, X_va), _combine(num_te, X_te)


def to_onehot_arrays(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """Fit imputer, scaler, and OneHotEncoder strictly on X_tr."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    num_tr = scaler.fit_transform(imputer.fit_transform(X_tr[num_cols]))
    num_va = scaler.transform(imputer.transform(X_va[num_cols]))
    num_te = scaler.transform(imputer.transform(X_te[num_cols]))

    if not cat_cols:
        return (
            sparse.csr_matrix(num_tr, dtype=np.float32),
            sparse.csr_matrix(num_va, dtype=np.float32),
            sparse.csr_matrix(num_te, dtype=np.float32),
        )

    ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)
    cat_tr = ohe.fit_transform(X_tr[cat_cols].astype(str))
    cat_va = ohe.transform(X_va[cat_cols].astype(str))
    cat_te = ohe.transform(X_te[cat_cols].astype(str))

    return (
        sparse.hstack([cat_tr, sparse.csr_matrix(num_tr)]).tocsr(),
        sparse.hstack([cat_va, sparse.csr_matrix(num_va)]).tocsr(),
        sparse.hstack([cat_te, sparse.csr_matrix(num_te)]).tocsr(),
    )


def align_proba(probs: np.ndarray, classes: np.ndarray | None, num_classes: int) -> np.ndarray:
    """Map predicted class columns into a full (N, num_classes) float32 array."""
    out = np.zeros((probs.shape[0], num_classes), dtype=np.float32)
    cls = classes if classes is not None else np.arange(probs.shape[1])
    out[:, cls] = probs.astype(np.float32)
    return out


def route_55_plus(df: pd.DataFrame) -> np.ndarray:
    """Return boolean mask identifying respondents routed to the 55+ module."""
    if "PCHA_APLICAVEL" in df.columns:
        return df["PCHA_APLICAVEL"].to_numpy() == 1
    return df["PC1A"].astype(str).to_numpy() != "Não aplicável"


def run_hierarchical(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    make_estimator: Callable[[], Any],
    prepare_fn: Callable[
        [pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]],
        tuple[Any, Any, Any],
    ],
    meta: dict[str, Any],
    use_sample_weight: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Train separate models for <55 and 55+ survey halves using fold-local transforms."""
    num_classes = int(meta["num_classes"])
    cat_cols = meta["cat_cols"]
    num_cols = meta["num_cols"]

    r_tr = route_55_plus(X_tr)
    r_va = route_55_plus(X_va)
    r_te = route_55_plus(X_te)

    val_probs = np.zeros((len(X_va), num_classes), dtype=np.float32)
    test_probs = np.zeros((len(X_te), num_classes), dtype=np.float32)

    for side in (False, True):
        tr_idx = np.where(r_tr == side)[0]
        va_idx = np.where(r_va == side)[0]
        te_idx = np.where(r_te == side)[0]
        if len(tr_idx) == 0:
            continue

        X_tr_sub, X_va_sub, X_te_sub = prepare_fn(
            X_tr.iloc[tr_idx],
            X_va.iloc[va_idx] if len(va_idx) else X_tr.iloc[:0],
            X_te.iloc[te_idx] if len(te_idx) else X_tr.iloc[:0],
            cat_cols,
            num_cols,
        )
        cls = np.unique(y_tr[tr_idx])
        encoded_y = np.searchsorted(cls, y_tr[tr_idx])

        est = make_estimator()
        if use_sample_weight:
            sw = compute_sample_weight("balanced", encoded_y)
            est.fit(X_tr_sub, encoded_y, sample_weight=sw)
        else:
            est.fit(X_tr_sub, encoded_y)

        if len(va_idx):
            val_probs[np.ix_(va_idx, cls)] = est.predict_proba(X_va_sub).astype(np.float32)
        if len(te_idx):
            test_probs[np.ix_(te_idx, cls)] = est.predict_proba(X_te_sub).astype(np.float32)

    return val_probs, test_probs
