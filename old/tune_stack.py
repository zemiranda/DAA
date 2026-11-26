"""Afinacao (random search) do LightGBM + ExtraTrees + stacking com meta-modelo logistico.
Corre em background: python tune_stack.py > out/log_tune.txt
"""
import json, time, warnings, sys
STAGE=sys.argv[1]
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

OUT, SEED, K = "out", 42, 16
feat = pd.read_pickle(f"{OUT}/features.pkl")
y = pd.read_csv("dataset_train.csv", usecols=["AGE_RANGE"]).AGE_RANGE.values; ntr = len(y)
Xtr, Xte = feat.iloc[:ntr].reset_index(drop=True), feat.iloc[ntr:].reset_index(drop=True)
folds5 = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, y))
folds3 = list(StratifiedKFold(3, shuffle=True, random_state=1).split(Xtr, y))
mf1 = lambda p, yy=y: f1_score(yy, p.argmax(1), average="macro")
log = lambda *a: print(*a, flush=True)

# ------------------------------------------------ 1) random search LightGBM (3 folds, lr alto)
rng = np.random.RandomState(7)
import os
trials = []
for i in range(8):
    p = dict(n_estimators=110, learning_rate=0.09,
             num_leaves=int(rng.choice([7, 15, 31])),
             colsample_bytree=float(rng.choice([0.1, 0.2, 0.3, 0.5])),
             subsample=float(rng.choice([0.6, 0.8, 1.0])), subsample_freq=1,
             min_child_samples=int(rng.choice([10, 20, 40, 80])),
             reg_lambda=float(rng.choice([0.1, 1, 5, 20])),
             max_bin=63, cat_smooth=float(rng.choice([10, 30])), min_data_per_group=50,
             class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=-1)
    lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 8)
    if STAGE != "search" or not (lo <= i < hi): continue
    t = time.time(); oof = np.zeros((ntr, K))
    for tri, vai in folds3:
        m = lgb.LGBMClassifier(**p).fit(Xtr.iloc[tri], y[tri]); oof[vai] = m.predict_proba(Xtr.iloc[vai])
    s = mf1(oof); trials.append((s, p)); log(f"trial {i}: F1={s:.4f} {({k: p[k] for k in ['num_leaves','colsample_bytree','subsample','min_child_samples','reg_lambda']})} ({time.time()-t:.0f}s)")
if STAGE=="search":
    prev = json.load(open(f"{OUT}/trials.json")) if os.path.exists(f"{OUT}/trials.json") else []
    json.dump(prev + [(s_, p_) for s_, p_ in trials], open(f"{OUT}/trials.json", "w"))
    log("search done")
    sys.exit()

# ------------------------------------------------ 2) modelos finais com 5 folds
def cv_model(make, name, cat_native=True, Xt=Xtr, Xe=Xte):
    t = time.time(); oof = np.zeros((ntr, K)); pte = np.zeros((len(Xe), K))
    for tri, vai in folds5:
        m = make().fit(Xt.iloc[tri], y[tri]); oof[vai] = m.predict_proba(Xt.iloc[vai]); pte += m.predict_proba(Xe) / 5
    log(f"{name}: F1={mf1(oof):.4f} ({time.time()-t:.0f}s)")
    np.save(f"{OUT}/oof_{name}.npy", oof); np.save(f"{OUT}/test_{name}.npy", pte)

if STAGE=="lgb":
    tj=json.load(open(f"{OUT}/trials.json"))
    b=max(tj,key=lambda x:x[0]); log("melhor trial",round(b[0],4),b[1]); bp=dict(b[1]); bp.update(n_estimators=300,learning_rate=0.04)
    cv_model(lambda: lgb.LGBMClassifier(**bp), "lgb_tuned"); sys.exit()

num_df = feat.copy()
for c in num_df.columns:
    if str(num_df[c].dtype) == "category": num_df[c] = num_df[c].cat.codes
num_df = num_df.fillna(num_df.median())
if STAGE=="et": cv_model(lambda: ExtraTreesClassifier(500, min_samples_leaf=2, max_features=0.25, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=SEED), "et",
         Xt=num_df.iloc[:ntr].reset_index(drop=True), Xe=num_df.iloc[ntr:].reset_index(drop=True))

if STAGE!="stack": sys.exit()
# ------------------------------------------------ 3) stacking (meta-modelo LR nas probabilidades OOF)
names = ["lgb_tuned", "lgb_fe", "lr", "rf", "et"]
O = np.hstack([np.log(np.clip(np.load(f"{OUT}/oof_{n}.npy"), 1e-6, 1)) for n in names])
T = np.hstack([np.log(np.clip(np.load(f"{OUT}/test_{n}.npy"), 1e-6, 1)) for n in names])
skf2 = StratifiedKFold(5, shuffle=True, random_state=99)
for C in [0.003, 0.01, 0.03]:
    mo = np.zeros((ntr, K))
    for tri, vai in skf2.split(O, y):
        m = LogisticRegression(C=C, max_iter=500, class_weight="balanced").fit(O[tri], y[tri]); mo[vai] = m.predict_proba(O[vai])
    log(f"stacking LR C={C}: F1={mf1(mo):.4f}")
    np.save(f"{OUT}/oof_stack_C{C}.npy", mo)

# media simples das probabilidades como referencia
avg = sum(np.load(f"{OUT}/oof_{n}.npy") for n in names) / len(names); log(f"media simples: F1={mf1(avg):.4f}")
log("FIM")
