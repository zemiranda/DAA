"""Abordagem hierarquica: o encaminhamento do inquerito separa EXATAMENTE <55 anos (classes 0-8) de 55+ (classes 9-15)
(PC1A 'Nao aplicavel' <=> idade < 55, sem excecoes no treino; no teste 55,5% tem PC1A aplicavel, igual ao treino).
Treina um modelo por metade, com os MESMOS folds dos restantes scripts.
Uso: python hier.py <modelo>      modelos: xgb | lgb | cat | rf     -> out/oof_h<modelo>.npy, out/test_h<modelo>.npy
Variaveis opcionais: FEATS (por omissao features.pkl)
Depois: python stack_final.py hlgb,hxgb,hcat,lgb_tuned,lgb_fe,lr,rf,et   (ou o conjunto que preferires)
"""
import sys, os, json, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
warnings.filterwarnings("ignore")

OUT, K, SEED = "out", 16, 42
MODEL = sys.argv[1]
feat = pd.read_pickle(f"{OUT}/{os.environ.get('FEATS', 'features.pkl')}")
tr = pd.read_csv("dataset_train.csv"); te = pd.read_csv("dataset_test_X.csv")
y = tr.AGE_RANGE.values; ntr = len(y)
route = lambda d: (d.PC1A != "Não aplicável").values            # True = 55+
r_tr, r_te = route(tr), route(te)
assert (r_tr == (y >= 9)).all()
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(ntr), y))
cat = [c for c in feat.columns if str(feat[c].dtype) == "category"]; num = [c for c in feat.columns if c not in cat]

def ordinal_matrix():
    nd = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(feat[num]), columns=num)
    return pd.concat([nd, feat[cat].apply(lambda s: s.cat.codes)], axis=1).values.astype(np.float32)

if MODEL == "xgb":
    import xgboost as xgb
    X, sw = ordinal_matrix(), True
    make = lambda: xgb.XGBClassifier(n_estimators=160, learning_rate=0.08, max_depth=4, subsample=0.8, colsample_bytree=0.3,
                                     min_child_weight=5, reg_lambda=5, tree_method="hist", max_bin=64, n_jobs=-1,
                                     objective="multi:softprob", random_state=SEED, verbosity=0)
elif MODEL == "lgb":
    import lightgbm as lgb
    best = max(json.load(open(f"{OUT}/trials.json")), key=lambda t: t[0])[1]; best.update(n_estimators=300, learning_rate=0.04)
    X, sw = feat, False; make = lambda: lgb.LGBMClassifier(**best)
elif MODEL == "cat":
    from catboost import CatBoostClassifier
    X, sw = ordinal_matrix(), False
    make = lambda: CatBoostClassifier(iterations=250, learning_rate=0.12, depth=5, loss_function="MultiClass",
                                      auto_class_weights="Balanced", rsm=0.3, l2_leaf_reg=5, random_seed=SEED,
                                      verbose=0, thread_count=-1, border_count=64)
elif MODEL == "rf":
    from sklearn.ensemble import RandomForestClassifier
    X, sw = ordinal_matrix(), False
    make = lambda: RandomForestClassifier(400, min_samples_leaf=3, max_features=0.2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
else:
    raise SystemExit("modelo desconhecido: xgb | lgb | cat | rf")

df = isinstance(X, pd.DataFrame)
sub = lambda M, idx: M.iloc[idx] if df else M[idx]
Xtr, Xte = sub(X, np.arange(ntr)), sub(X, np.arange(ntr, len(X)))
oof = np.zeros((ntr, K)); pte = np.zeros((len(te), K)); t0 = time.time()
for tri, vai in folds:
    for side in (False, True):                                   # False: <55 ; True: 55+
        a = tri[r_tr[tri] == side]; b = vai[r_tr[vai] == side]; e = np.where(r_te == side)[0]
        m = make(); cls = np.unique(y[a]); code = np.searchsorted(cls, y[a])   # rotulos 0..n-1 (exigido pelo XGBoost)
        if sw: m.fit(sub(Xtr, a), code, sample_weight=compute_sample_weight("balanced", code))
        else: m.fit(sub(Xtr, a), code)
        oof[np.ix_(b, cls)] = m.predict_proba(sub(Xtr, b))
        pte[np.ix_(e, cls)] += m.predict_proba(sub(Xte, e)) / len(folds)
np.save(f"{OUT}/oof_h{MODEL}.npy", oof); np.save(f"{OUT}/test_h{MODEL}.npy", pte)
print(f"h{MODEL}: macroF1={f1_score(y, oof.argmax(1), average='macro'):.4f} acc={(oof.argmax(1)==y).mean():.4f} ({time.time()-t0:.0f}s)")
