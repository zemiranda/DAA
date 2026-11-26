"""Level-1 stacking meta-learner and CV-validated Macro F1 threshold optimization."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.config import load_config


def _log_clip(probs: np.ndarray) -> np.ndarray:
    """Convert class probabilities into clipped log-probabilities."""
    return np.log(np.clip(probs, 1e-6, 1.0)).astype(np.float32)


def _cv_meta_oof(
    names: list[str],
    c_val: float,
    oof_map: dict[str, np.ndarray],
    y_train: np.ndarray,
    folds: np.ndarray,
    num_classes: int,
    max_iter: int,
) -> tuple[float, np.ndarray]:
    """Evaluate meta-learner strictly across the frozen CV folds."""
    X_meta = np.hstack([oof_map[n] for n in names])
    meta_oof = np.zeros((len(y_train), num_classes), dtype=np.float32)
    unique_folds = np.sort(np.unique(folds))

    for f in unique_folds:
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]
        clf = LogisticRegression(C=c_val, max_iter=max_iter, class_weight="balanced")
        clf.fit(X_meta[tr_idx], y_train[tr_idx])
        meta_oof[va_idx] = clf.predict_proba(X_meta[va_idx]).astype(np.float32)

    score = float(f1_score(y_train, meta_oof.argmax(axis=1), average="macro"))
    return score, meta_oof


def _fit_class_multipliers(
    probs: np.ndarray,
    y_true: np.ndarray,
    reg_strength: float = 0.01,
) -> np.ndarray:
    """Optimize per-class probability multipliers via regularized Powell search."""
    num_classes = probs.shape[1]

    def _objective(log_w: np.ndarray) -> float:
        w = np.exp(np.clip(log_w, -0.7, 0.7))
        preds = (probs * w[None, :]).argmax(axis=1)
        macro_f1 = float(f1_score(y_true, preds, average="macro"))
        penalty = reg_strength * float(np.mean(log_w ** 2))
        return -macro_f1 + penalty

    res = minimize(
        _objective,
        x0=np.zeros(num_classes, dtype=np.float64),
        method="Powell",
        options={"maxiter": 600, "ftol": 1e-4},
    )
    return np.exp(np.clip(res.x, -0.7, 0.7)).astype(np.float32)


def _evaluate_oof_multipliers(
    meta_oof: np.ndarray,
    y_train: np.ndarray,
    folds: np.ndarray,
) -> tuple[float, np.ndarray | None]:
    """Validate per-class multipliers out-of-fold; return multipliers only if they generalize."""
    raw_f1 = float(f1_score(y_train, meta_oof.argmax(axis=1), average="macro"))
    tuned_preds = np.empty(len(y_train), dtype=np.int64)
    for f in np.sort(np.unique(folds)):
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]
        w_fold = _fit_class_multipliers(meta_oof[tr_idx], y_train[tr_idx])
        tuned_preds[va_idx] = (meta_oof[va_idx] * w_fold[None, :]).argmax(axis=1)

    tuned_f1 = float(f1_score(y_train, tuned_preds, average="macro"))
    print(f"Threshold tuning OOF check: raw F1={raw_f1:.4f} | tuned F1={tuned_f1:.4f}")
    if tuned_f1 > raw_f1 + 1e-4:
        full_w = _fit_class_multipliers(meta_oof, y_train)
        return tuned_f1, full_w
    print("Multipliers did not improve out-of-fold Macro F1; keeping calibrated argmax.")
    return raw_f1, None


def _next_submission_path(sub_dir: Path) -> Path:
    """Return the next incremented submission path (submission_v<N>.csv)."""
    versions = []
    for p in sub_dir.glob("submission_v*.csv"):
        m = re.fullmatch(r"submission_v(\d+)\.csv", p.name)
        if m:
            versions.append(int(m.group(1)))
    next_version = max(versions, default=0) + 1
    return sub_dir / f"submission_v{next_version}.csv"


def run_stack(
    cfg: dict[str, Any],
    models: list[str] | None = None,
    optimize_thresholds: bool | None = None,
) -> None:
    """Train Level-1 meta-learner on OOF probabilities and export submission_stack.csv."""
    proc_dir = Path(cfg["paths"]["processed_dir"])
    oof_dir = Path(cfg["paths"]["oof_dir"])
    preds_dir = Path(cfg["paths"]["preds_dir"])
    sub_dir = Path(cfg["paths"]["submissions_dir"])

    y_train = np.load(proc_dir / "y_train.npy")
    folds = np.load(proc_dir / "folds.npy")
    num_classes = int(cfg["num_classes"])
    stack_cfg = cfg["stack"]
    c_grid = [float(c) for c in stack_cfg["c_grid"]]
    max_iter = int(stack_cfg.get("max_iter", 1000))
    use_opt = bool(stack_cfg.get("optimize_thresholds", True) if optimize_thresholds is None else optimize_thresholds)

    available = {
        p.stem
        for p in oof_dir.glob("*.npy")
        if p.stem != "stack_best" and (preds_dir / p.name).exists()
    }
    if not available:
        raise FileNotFoundError(f"No matching OOF/test .npy files found in {oof_dir} and {preds_dir}.")

    if models:
        missing = [m for m in models if m not in available]
        if missing:
            raise FileNotFoundError(f"Missing OOF/test predictions for models: {missing}")
        candidate_sets = {"custom": models}
    else:
        candidate_sets = {
            label: m_list
            for label, m_list in stack_cfg["sets"].items()
            if all(m in available for m in m_list)
        }
        if not candidate_sets:
            candidate_sets = {"all_available": sorted(available)}

    needed = sorted({m for m_list in candidate_sets.values() for m in m_list})
    oof_map = {m: _log_clip(np.load(oof_dir / f"{m}.npy")) for m in needed}
    tst_map = {m: _log_clip(np.load(preds_dir / f"{m}.npy")) for m in needed}

    best: tuple[float, str, list[str], float, np.ndarray] | None = None
    for label, names in candidate_sets.items():
        for c_val in c_grid:
            score, meta_oof = _cv_meta_oof(names, c_val, oof_map, y_train, folds, num_classes, max_iter)
            print(f"{label:24s} C={c_val:<6} F1={score:.4f}")
            if best is None or score > best[0]:
                best = (score, label, names, c_val, meta_oof)

    assert best is not None
    best_score, best_label, best_names, best_c, best_oof = best
    mae = float(np.abs(best_oof.argmax(axis=1) - y_train).mean())
    print(f"\nBEST STACK [{best_label}]: F1={best_score:.4f} | models={best_names} | C={best_c} | MAE={mae:.2f}")

    multipliers: np.ndarray | None = None
    final_oof = best_oof
    if use_opt:
        _, multipliers = _evaluate_oof_multipliers(best_oof, y_train, folds)
        if multipliers is not None:
            final_oof = (best_oof * multipliers[None, :]).astype(np.float32)

    X_all = np.hstack([oof_map[n] for n in best_names])
    T_all = np.hstack([tst_map[n] for n in best_names])
    meta_clf = LogisticRegression(C=best_c, max_iter=max_iter, class_weight="balanced")
    meta_clf.fit(X_all, y_train)
    test_probs = meta_clf.predict_proba(T_all).astype(np.float32)
    if multipliers is not None:
        test_probs = (test_probs * multipliers[None, :]).astype(np.float32)

    np.save(oof_dir / "stack_best.npy", final_oof)
    np.save(preds_dir / "stack_best.npy", test_probs)

    example_sub = pd.read_csv(cfg["paths"]["example_submission"])
    sub = example_sub.copy()
    sub[cfg["target_col"]] = test_probs.argmax(axis=1)
    assert len(sub) == len(example_sub)
    assert list(sub.columns) == [cfg["submission_id_col"], cfg["target_col"]]

    versioned_path = _next_submission_path(sub_dir)
    latest_path = sub_dir / "submission_stack.csv"
    sub.to_csv(versioned_path, index=False)
    sub.to_csv(latest_path, index=False)
    print(f"Saved submission to {versioned_path} (and updated {latest_path})")
    print("Predicted class counts:", sub[cfg["target_col"]].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    run_stack(load_config())
