"""Hyperparameter random search for LightGBM writing results to out/trials.json."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.config import load_config


def run_tune(
    cfg: dict[str, Any],
    n_trials: int | None = None,
    append: bool = False,
) -> None:
    """Run randomized hyperparameter search for LightGBM and persist out/trials.json."""
    proc_dir = Path(cfg["paths"]["processed_dir"])
    X_train = pd.read_parquet(proc_dir / "X_train.parquet")
    y_train = np.load(proc_dir / "y_train.npy")
    folds = np.load(proc_dir / "folds.npy")

    tune_cfg = cfg["tune"]
    total_trials = int(n_trials if n_trials is not None else tune_cfg["n_trials"])
    n_folds = int(tune_cfg.get("n_folds", 3))
    num_classes = int(cfg["num_classes"])
    seed = int(cfg["seed"])

    if n_folds == int(cfg["n_splits"]):
        split_list = [
            (np.where(folds != f)[0], np.where(folds == f)[0])
            for f in range(n_folds)
        ]
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
        split_list = list(skf.split(X_train, y_train))

    rng = np.random.RandomState(int(tune_cfg.get("search_seed", 7)))
    fixed = dict(tune_cfg["lgb_search_fixed"])
    trials: list[tuple[float, dict[str, Any]]] = []

    for i in range(total_trials):
        params = {
            **fixed,
            "num_leaves": int(rng.choice([7, 15, 31])),
            "colsample_bytree": float(rng.choice([0.1, 0.2, 0.3, 0.5])),
            "subsample": float(rng.choice([0.6, 0.8, 1.0])),
            "min_child_samples": int(rng.choice([10, 20, 40, 80])),
            "reg_lambda": float(rng.choice([0.1, 1.0, 5.0, 20.0])),
            "cat_smooth": float(rng.choice([10.0, 30.0])),
            "random_state": seed,
            "verbose": -1,
            "n_jobs": -1,
        }
        t0 = time.time()
        oof = np.zeros((len(X_train), num_classes), dtype=np.float32)
        for tr_idx, va_idx in split_list:
            est = lgb.LGBMClassifier(**params)
            est.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            oof[va_idx] = est.predict_proba(X_train.iloc[va_idx]).astype(np.float32)
        score = float(f1_score(y_train, oof.argmax(axis=1), average="macro"))
        trials.append((score, params))
        print(
            f"trial {i}: F1={score:.4f} "
            f"(leaves={params['num_leaves']}, colsample={params['colsample_bytree']}, "
            f"min_child={params['min_child_samples']}, reg={params['reg_lambda']}) "
            f"[{time.time() - t0:.1f}s]",
            flush=True,
        )

    trials_path = Path(cfg["paths"]["trials_file"])
    existing: list[Any] = []
    if append and trials_path.exists():
        with trials_path.open("r", encoding="utf-8") as fh:
            existing = json.load(fh)
    combined = existing + [[s, p] for s, p in trials]
    with trials_path.open("w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2)
    best_score = max(item[0] for item in combined)
    print(f"Saved {len(combined)} trials to {trials_path} (best F1={best_score:.4f})")


if __name__ == "__main__":
    run_tune(load_config())
