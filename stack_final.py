"""Stacking (meta-modelo logistico sobre log-probabilidades OOF) e geracao da submissao.
Uso: python stack_final.py            -> compara conjuntos de modelos e escreve out/submission_stack.csv (melhor conjunto)
     python stack_final.py m1,m2,...  -> usa exatamente esse conjunto
"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")

OUT = "out"
y = pd.read_csv("dataset_train.csv", usecols=["AGE_RANGE"]).AGE_RANGE.values; ntr = len(y)
L = lambda a: np.log(np.clip(a, 1e-6, 1))
oof = lambda n: L(np.load(f"{OUT}/oof_{n}.npy")); tst = lambda n: L(np.load(f"{OUT}/test_{n}.npy"))
mf1 = lambda p: f1_score(y, p.argmax(1), average="macro")

def cv_stack(names, C):
    O = np.hstack([oof(n) for n in names]); P = np.zeros((ntr, 16))
    for tri, vai in StratifiedKFold(5, shuffle=True, random_state=99).split(O, y):
        m = LogisticRegression(C=C, max_iter=1000, class_weight="balanced").fit(O[tri], y[tri]); P[vai] = m.predict_proba(O[vai])
    return mf1(P), P

SETS = {
    "v2 (lgb_tuned,lgb_fe,lr,rf,et)": ["lgb_tuned", "lgb_fe", "lr", "rf", "et"],
    "+xgb,cat": ["lgb_tuned", "lgb_fe", "lr", "rf", "et", "xgb", "cat"],
    "+xgb,cat,mlp": ["lgb_tuned", "lgb_fe", "lr", "rf", "et", "xgb", "cat", "mlp"],
    "+xgb,cat,mlp,ord (todos)": ["lgb_tuned", "lgb_fe", "lr", "rf", "et", "xgb", "cat", "mlp", "ord"],
}
if len(sys.argv) > 1:
    SETS = {"escolhido": sys.argv[1].split(",")}
best = (0, None, None, None)
for label, names in SETS.items():
    for C in [0.001, 0.003, 0.01]:
        s, P = cv_stack(names, C); print(f"{label:34s} C={C:<6} F1={s:.4f}")
        if s > best[0]: best = (s, names, C, P)
s, names, C, P = best
print(f"\nMELHOR: F1={s:.4f} | modelos={names} | C={C} | MAE grupos={np.abs(P.argmax(1)-y).mean():.2f}")
np.save(f"{OUT}/oof_stack_best.npy", P)
O = np.hstack([oof(n) for n in names]); T = np.hstack([tst(n) for n in names])
m = LogisticRegression(C=C, max_iter=1000, class_weight="balanced").fit(O, y)
sub = pd.read_csv("example_submission_10.csv"); sub["AGE_RANGE"] = m.predict_proba(T).argmax(1)
assert len(sub) == 3655 and list(sub.columns) == ["RowId", "AGE_RANGE"]
sub.to_csv(f"{OUT}/submission_stack.csv", index=False); print(sub.AGE_RANGE.value_counts().sort_index().to_dict())
