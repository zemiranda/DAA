"""Unified CLI entrypoint for the DAA 2026/27 age-group Macro F1 pipeline."""
from __future__ import annotations

import argparse

from src.config import load_config


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for all pipeline stages."""
    parser = argparse.ArgumentParser(description="DAA 2026/27 Age Group Prediction Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preprocess", help="Clean features, engineer domain signals, freeze CV folds")

    p_tune = sub.add_parser("tune", help="Run LightGBM hyperparameter random search")
    p_tune.add_argument("--trials", type=int, default=None, help="Number of random search trials")
    p_tune.add_argument("--append", action="store_true", help="Append to existing out/trials.json")

    p_train = sub.add_parser("train", help="Train one or all registered models across frozen folds")
    p_train.add_argument("--model", nargs="+", help="Model name(s) to train (e.g. lgb_tuned cat)")
    p_train.add_argument("--all", action="store_true", help="Train all 9 core models")
    p_train.add_argument("--folds", nargs="+", type=int, help="Optional subset of fold indices to run")
    p_train.add_argument("--merge", action="store_true", help="Merge partial fold outputs in out/parts/")
    p_train.add_argument("--seeds", nargs="+", type=int, help="Optional list of random seeds to average")

    p_stack = sub.add_parser("stack", help="Train Level-1 meta-learner and generate submission")
    p_stack.add_argument("--models", nargs="+", help="Specific base model names to stack")
    p_stack.add_argument("--no-threshold-opt", action="store_true", help="Disable OOF multiplier tuning")

    p_report = sub.add_parser("report", help="Generate diagnostics, clustering, and report figures")
    p_report.add_argument("--importance", action="store_true", help="Feature importance & SHAP contributions")
    p_report.add_argument("--anomalies", action="store_true", help="Supervised discrepancy diagnostics")
    p_report.add_argument("--unsupervised", action="store_true", help="Autoencoder, K-Means, DBSCAN")
    p_report.add_argument("--characterization", action="store_true", help="Age group profile summary")
    p_report.add_argument("--figures", action="store_true", help="Generate PNG charts in out/report/figs/")
    p_report.add_argument("--all", action="store_true", help="Run all report stages")

    return parser


def main() -> None:
    """Dispatch CLI command to the corresponding pipeline stage."""
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    if args.command == "preprocess":
        from src.preprocess import run_preprocess

        run_preprocess(cfg)
    elif args.command == "tune":
        from src.tune import run_tune

        run_tune(cfg, n_trials=args.trials, append=args.append)
    elif args.command == "train":
        from src.train import run_train

        run_train(
            cfg,
            models=args.model,
            run_all=args.all,
            folds=args.folds,
            merge=args.merge,
            seeds=args.seeds,
        )
    elif args.command == "stack":
        from src.stack import run_stack

        run_stack(
            cfg,
            models=args.models,
            optimize_thresholds=False if args.no_threshold_opt else None,
        )
    elif args.command == "report":
        from src.report import run_report

        run_report(
            cfg,
            importance=args.importance,
            anomalies=args.anomalies,
            unsupervised=args.unsupervised,
            characterization=args.characterization,
            figures=args.figures,
            run_all=args.all,
        )


if __name__ == "__main__":
    main()
