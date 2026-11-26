# DAA 2026/27 – Execution Order

Place raw data in `data/raw/`: `dataset_train.csv`, `dataset_test_X.csv`, `example_submission_10.csv`.
Install dependencies: `pip install -r requirements.txt` (or use `.venv/bin/python`).

## Main Commands (`main.py` / `Makefile`)

| # | Command | What It Does |
|---|---|---|
| 1 | `python main.py preprocess` | Leakage-free cleaning, feature engineering (`fe2` + MNAR flags + life-stage features), generates `data/processed/{X_train,X_test}.parquet`, `y_train.npy`, and frozen `folds.npy` |
| 2 | `python main.py tune --trials 8` | *(Optional)* LightGBM random search updating `out/trials.json` |
| 3 | `python main.py train --all` | Trains the 9 base models (`lgb_fe`, `lgb_tuned`, `xgb`, `cat`, `rf`, `et`, `lr`, `mlp`, `ord`) on the same folds, saving to `out/oof/` and `out/preds/` |
| 4 | `python main.py stack` | Logistic meta-model over OOF log-probabilities + OOF validation of Macro F1 multipliers, generates `submissions/submission_stack.csv` |
| 5 | `python main.py report --all` | Feature importance, supervised anomalies, autoencoder, K-Means, DBSCAN, characterization, and figures in `out/report/` |

### Advanced Training Options
- Train a single model only: `python main.py train --model lgb_tuned`
- Hierarchical training (`<55` vs `55+`): `python main.py train --model hlgb hxgb hcat hrf`
- Per-fold training (resumable): `python main.py train --model xgb --folds 0 1 2` followed by `python main.py train --model xgb --folds 3 4` and `python main.py train --model xgb --merge`
- Multi-seed averaging: `python main.py train --model xgb --seeds 1 2 3`
- Fully automated via `Makefile`: `make all`
