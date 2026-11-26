"""Unified CV training runner for all registered models."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import load_config
from src.models import MODEL_REGISTRY, get_model_params


def _load_processed(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load preprocessed features, target, frozen folds, and schema metadata."""
    proc_dir = Path(cfg["paths"]["processed_dir"])
    X_train = pd.read_parquet(proc_dir / "X_train.parquet")
    X_test = pd.read_parquet(proc_dir / "X_test.parquet")
    y_train = np.load(proc_dir / "y_train.npy")
    folds = np.load(proc_dir / "folds.npy")
    with (proc_dir / "meta.json").open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return X_train, X_test, y_train, folds, meta


def _train_single_seed(
    model_name: str,
    cfg: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    folds: np.ndarray,
    meta: dict[str, Any],
    seed: int,
    fold_ids: list[int] | None = None,
    output_tag: str = "",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Train a model across specified (or all) frozen CV folds for one seed."""
    fit_predict = MODEL_REGISTRY[model_name]
    params = get_model_params(model_name, cfg)
    n_splits = int(cfg["n_splits"])
    num_classes = int(cfg["num_classes"])
    selected_folds = fold_ids if fold_ids is not None else list(range(n_splits))

    parts_dir = Path(cfg["paths"]["parts_dir"])
    oof = np.zeros((len(X_train), num_classes), dtype=np.float32)
    test_preds = np.zeros((len(X_test), num_classes), dtype=np.float32)

    t0 = time.time()
    for f in selected_folds:
        ft0 = time.time()
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]
        val_p, te_p = fit_predict(
            X_train.iloc[tr_idx],
            y_train[tr_idx],
            X_train.iloc[va_idx],
            X_test,
            params,
            seed,
            meta,
        )
        fold_f1 = f1_score(y_train[va_idx], val_p.argmax(axis=1), average="macro")
        print(f"  [{model_name}{output_tag}] fold {f}: F1={fold_f1:.4f} ({time.time() - ft0:.1f}s)", flush=True)

        if fold_ids is not None:
            part_path = parts_dir / f"part_{model_name}{output_tag}_{f}.npz"
            np.savez(part_path, oof=val_p, te=te_p)
        else:
            oof[va_idx] = val_p
            test_preds += te_p / n_splits

    if fold_ids is not None:
        return None

    macro_f1 = f1_score(y_train, oof.argmax(axis=1), average="macro")
    acc = float((oof.argmax(axis=1) == y_train).mean())
    print(
        f"{model_name}{output_tag}: macroF1={macro_f1:.4f} acc={acc:.4f} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    out_name = f"{model_name}{output_tag}"
    np.save(Path(cfg["paths"]["oof_dir"]) / f"{out_name}.npy", oof)
    np.save(Path(cfg["paths"]["preds_dir"]) / f"{out_name}.npy", test_preds)
    return oof, test_preds


def merge_folds(model_name: str, cfg: dict[str, Any], output_tag: str = "") -> None:
    """Merge per-fold partial .npz files into out/oof/<model>.npy and out/preds/<model>.npy."""
    _, X_test, y_train, folds, _ = _load_processed(cfg)
    n_splits = int(cfg["n_splits"])
    num_classes = int(cfg["num_classes"])
    parts_dir = Path(cfg["paths"]["parts_dir"])

    oof = np.zeros((len(y_train), num_classes), dtype=np.float32)
    test_preds = np.zeros((len(X_test), num_classes), dtype=np.float32)

    for f in range(n_splits):
        part_path = parts_dir / f"part_{model_name}{output_tag}_{f}.npz"
        data = np.load(part_path)
        va_idx = np.where(folds == f)[0]
        oof[va_idx] = data["oof"]
        test_preds += data["te"] / n_splits

    macro_f1 = f1_score(y_train, oof.argmax(axis=1), average="macro")
    acc = float((oof.argmax(axis=1) == y_train).mean())
    out_name = f"{model_name}{output_tag}"
    np.save(Path(cfg["paths"]["oof_dir"]) / f"{out_name}.npy", oof)
    np.save(Path(cfg["paths"]["preds_dir"]) / f"{out_name}.npy", test_preds)
    print(f"{out_name} (merged): macroF1={macro_f1:.4f} acc={acc:.4f}")


def run_train(
    cfg: dict[str, Any],
    models: list[str] | None = None,
    run_all: bool = False,
    folds: list[int] | None = None,
    merge: bool = False,
    seeds: list[int] | None = None,
) -> None:
    """Train one or more registered models using frozen CV folds."""
    target_models = list(cfg["core_models"]) if run_all else (models or [])
    if not target_models:
        raise ValueError("Specify --model <name> or --all.")
    for m in target_models:
        if m not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{m}'. Available: {sorted(MODEL_REGISTRY)}")

    if merge:
        for m in target_models:
            merge_folds(m, cfg)
        return

    X_train, X_test, y_train, fold_arr, meta = _load_processed(cfg)
    base_seed = int(cfg["seed"])

    for m in target_models:
        if seeds and len(seeds) > 1:
            oof_list, pred_list = [], []
            for s in seeds:
                res = _train_single_seed(
                    m, cfg, X_train, X_test, y_train, fold_arr, meta,
                    seed=s, fold_ids=None, output_tag=f"_s{s}",
                )
                assert res is not None
                oof_list.append(res[0])
                pred_list.append(res[1])
            oof_ms = np.mean(oof_list, axis=0).astype(np.float32)
            pred_ms = np.mean(pred_list, axis=0).astype(np.float32)
            ms_f1 = f1_score(y_train, oof_ms.argmax(axis=1), average="macro")
            print(f"{m}_ms ({len(seeds)} seeds): macroF1={ms_f1:.4f}")
            np.save(Path(cfg["paths"]["oof_dir"]) / f"{m}_ms.npy", oof_ms)
            np.save(Path(cfg["paths"]["preds_dir"]) / f"{m}_ms.npy", pred_ms)
        else:
            use_seed = seeds[0] if seeds else base_seed
            _train_single_seed(
                m, cfg, X_train, X_test, y_train, fold_arr, meta,
                seed=use_seed, fold_ids=folds,
            )


if __name__ == "__main__":
    run_train(load_config(), run_all=True)
