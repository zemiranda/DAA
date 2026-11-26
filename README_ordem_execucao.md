# DAA 2026/27 – Ordem de Execução

Colocar os dados brutos em `data/raw/`: `dataset_train.csv`, `dataset_test_X.csv`, `example_submission_10.csv`.
Instalar dependências: `pip install -r requirements.txt` (ou usar `.venv/bin/python`).

## Comandos Principais (`main.py` / `Makefile`)

| # | Comando | O que faz |
|---|---|---|
| 1 | `python main.py preprocess` | Limpeza sem *data leakage*, engenharia de variáveis (`fe2` + flags MNAR + ciclo de vida), gera `data/processed/{X_train,X_test}.parquet`, `y_train.npy` e *folds* congelados `folds.npy` |
| 2 | `python main.py tune --trials 8` | *(Opcional)* Random search do LightGBM com atualização de `out/trials.json` |
| 3 | `python main.py train --all` | Treina os 9 modelos base (`lgb_fe`, `lgb_tuned`, `xgb`, `cat`, `rf`, `et`, `lr`, `mlp`, `ord`) nos mesmos *folds*, gravando em `out/oof/` e `out/preds/` |
| 4 | `python main.py stack` | Meta-modelo logístico sobre log-probabilidades OOF + validação OOF de multiplicadores Macro F1, gera `submissions/submission_stack.csv` |
| 5 | `python main.py report --all` | Importância de variáveis, anomalias supervisionadas, autoencoder, K-Means, DBSCAN, caracterização e figuras em `out/report/` |

### Opções avançadas de treino
- Treinar apenas um modelo: `python main.py train --model lgb_tuned`
- Treino hierárquico (`<55` vs `55+`): `python main.py train --model hlgb hxgb hcat hrf`
- Treino por *folds* (retomável): `python main.py train --model xgb --folds 0 1 2` seguido de `python main.py train --model xgb --folds 3 4` e `python main.py train --model xgb --merge`
- Média multi-semente: `python main.py train --model xgb --seeds 1 2 3`
- Tudo automatizado via `Makefile`: `make all`
