"""Configuration loading and path management utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load YAML pipeline configuration and ensure output directories exist."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    ensure_dirs(cfg)
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create all working directories referenced in config.yaml if missing."""
    paths = cfg["paths"]
    for key in (
        "processed_dir",
        "out_dir",
        "oof_dir",
        "preds_dir",
        "parts_dir",
        "report_dir",
        "figs_dir",
        "submissions_dir",
    ):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)
