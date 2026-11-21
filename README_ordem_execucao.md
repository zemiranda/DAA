# DAA 2026/27 – ordem de execução

Colocar na mesma pasta: dataset_train.csv, dataset_test_X.csv, example_submission_10.csv.
Dependências: pandas numpy scipy scikit-learn lightgbm xgboost catboost
Os resultados intermédios ficam em ./out/ (cada passo lê o que o anterior gravou).

| # | Comando | O que faz | Tempo (1 CPU) |
|---|---------|-----------|---------------|
| 1 | python pipeline.py | limpeza, features, LightGBM base (lgb_fe) | ~3 min |

--- Poderá faltar o fe2.py ---------------------------------------------
| 2 | python models2.py | regressão logística (lr) + random forest (rf) | ~2,5 min |
| 3 | python tune_stack.py search 0 8 | random search do LightGBM (grava out/trials.json) | ~4 min |
| 4 | python tune_stack.py lgb | LightGBM afinado (lgb_tuned) | ~2,5 min |
| 5 | python tune_stack.py et | ExtraTrees (et) | ~3 min |
| 6 | python new_models.py xgb 0 5 ; python new_models.py xgb merge | XGBoost | ~1 min |
| 7 | python new_models.py cat 0 5 ; python new_models.py cat merge | CatBoost | ~2 min |
| 8 | python new_models.py mlp 0 5 ; python new_models.py mlp merge | MLP | ~1 min |
| 9 | python new_models.py ord 0 5 ; python new_models.py ord merge | regressão ordinal | segundos |
| 10 | python stack_final.py | stacking, compara conjuntos, grava out/submission_stack.csv | ~1 min |
| 11 | python analysis.py importance ; python analysis.py anomalies | importância das variáveis e anomalias supervisionadas | ~3 min |
| 12 | python unsupervised.py ae ; ... kmeans ; ... dbscan | autoencoder, K-Means, DBSCAN | ~3 min |

Notas:
- blend.py é opcional (blend simples por pesos); o stacking de stack_final.py substitui-o.
- tune_stack.py stack (etapa antiga de stacking) também é opcional.
- A ordem importa: 1 e 2 antes de tudo; 3→4; 10 precisa dos passos 1–9; 11 e 12 precisam de 1 e 10.
