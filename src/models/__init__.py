"""Unified model registry and hyperparameter resolution."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.models.boosting import (
    fit_predict_cat,
    fit_predict_hcat,
    fit_predict_hlgb,
    fit_predict_hxgb,
    fit_predict_lgb,
    fit_predict_xgb,
)
from src.models.neural_ordinal import fit_predict_lr, fit_predict_mlp, fit_predict_ord
from src.models.trees import fit_predict_et, fit_predict_hrf, fit_predict_rf


FitPredictFn = Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, Any], int, dict[str, Any]],
    tuple[np.ndarray, np.ndarray],
]

MODEL_REGISTRY: dict[str, FitPredictFn] = {
    "lgb_fe": fit_predict_lgb,
    "lgb_tuned": fit_predict_lgb,
    "xgb": fit_predict_xgb,
    "cat": fit_predict_cat,
    "rf": fit_predict_rf,
    "et": fit_predict_et,
    "lr": fit_predict_lr,
    "mlp": fit_predict_mlp,
    "ord": fit_predict_ord,
    "hlgb": fit_predict_hlgb,
    "hxgb": fit_predict_hxgb,
    "hcat": fit_predict_hcat,
    "hrf": fit_predict_hrf,
}

_HIER_BASE_MAP: dict[str, str] = {
    "hlgb": "lgb_tuned",
    "hxgb": "xgb",
    "hcat": "cat",
    "hrf": "rf",
}


def get_model_params(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve model hyperparameters from trials.json (for lgb_tuned/hlgb) or config.yaml."""
    base_name = _HIER_BASE_MAP.get(name, name)
    params = dict(cfg["models"][base_name])

    if base_name == "lgb_tuned":
        trials_path = Path(cfg["paths"]["trials_file"])
        if trials_path.exists():
            with trials_path.open("r", encoding="utf-8") as fh:
                trials = json.load(fh)
            if trials:
                best_score, best_params = max(trials, key=lambda item: item[0])
                clean = {
                    k: v
                    for k, v in best_params.items()
                    if k not in ("random_state", "verbose", "n_jobs")
                }
                params.update(clean)
                params.update(cfg["tune"]["lgb_final_override"])
                print(f"[{name}] Loaded best trial (F1={best_score:.4f}) from {trials_path}")

    return params
