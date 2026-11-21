"""Media de varias sementes (menos variancia) para lgb_tuned, xgb e cat, mantendo os MESMOS folds.
Uso:   python multiseed.py <modelo> <sementes>        ex.: python multiseed.py xgb 1,2,3
       modelos: lgb_tuned | xgb | cat
Cada semente e guardada em out/oof_<modelo>_s<semente>.npy (retomavel: sementes ja feitas sao saltadas);
no fim grava a media de TODAS as sementes existentes em out/oof_<modelo>_ms.npy e out/test_<modelo>_ms.npy.
Depois:  python stack_final.py lgb_tuned_ms,xgb_ms,cat_ms,lgb_fe,lr,rf,et
Variaveis opcionais: FEATS (ficheiro de features, por omissao features.pkl)
"""
import sys, os, glob, json, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
warnings.filterwarnings("ignore")

OUT, K, FOLD_SEED = "out", 16, 42      # FOLD_SEED igual ao dos restantes scripts (OOF comparaveis)
MODEL, SEEDS = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
feat = pd.read_pickle(f"{OUT}/{os.environ.get('FEATS', 'features.pkl')}")
y = pd.read_csv("dataset_train.csv", usecols=["AGE_RANGE"]).AGE_RANGE.values; ntr = len(y)
folds = list(StratifiedKFold(5, shuffle=True, random_state=FOLD_SEED).split(np.zeros(ntr), y))
cat = [c for c in feat.columns if str(feat[c].dtype) == "category"]; num = [c for c in feat.columns if c not in cat]

def ordinal_matrix():
    nd = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(feat[num]), columns=num)
    return pd.concat([nd, feat[cat].apply(lambda s: s.cat.codes)], axis=1).values.astype(np.float32)

if MODEL == "lgb_tuned":
    import lightgbm as lgb
    best = max(json.load(open(f"{OUT}/trials.json")), key=lambda t: t[0])[1]; best.update(n_estimators=300, learning_rate=0.04)
    X, use_sw = feat, False
    make = lambda s: lgb.LGBMClassifier(**{**best, "random_state": s})
elif MODEL == "xgb":
    import xgboost as xgb
    X, use_sw = ordinal_matrix(), True
    make = lambda s: xgb.XGBClassifier(n_estimators=160, learning_rate=0.08, max_depth=4, subsample=0.8, colsample_bytree=0.3,
                                       min_child_weight=5, reg_lambda=5, tree_method="hist", max_bin=64, n_jobs=-1,
                                       objective="multi:softprob", random_state=s, verbosity=0)
elif MODEL == "cat":
    from catboost import CatBoostClassifier
    X, use_sw = ordinal_matrix(), False
    make = lambda s: CatBoostClassifier(iterations=250, learning_rate=0.12, depth=5, loss_function="MultiClass",
                                        auto_class_weights="Balanced", rsm=0.3, l2_leaf_reg=5, random_seed=s,
                                        verbose=0, thread_count=-1, border_count=64)
else:
    raise SystemExit("modelo desconhecido: lgb_tuned | xgb | cat")

iloc = (lambda M, i: M.iloc[i]) if isinstance(X, pd.DataFrame) else (lambda M, i: M[i])
Xtr_all, Xte = (X.iloc[:ntr], X.iloc[ntr:]) if isinstance(X, pd.DataFrame) else (X[:ntr], X[ntr:])

for s in SEEDS:
    fo, ft = f"{OUT}/oof_{MODEL}_s{s}.npy", f"{OUT}/test_{MODEL}_s{s}.npy"
    if os.path.exists(fo) and os.path.exists(ft):
        print(f"{MODEL} semente {s}: ja existe, salto"); continue
    t = time.time(); oof = np.zeros((ntr, K)); pte = np.zeros((len(y) * 0 + Xte.shape[0], K))
    for tri, vai in folds:
        m = make(s)
        if use_sw: m.fit(iloc(Xtr_all, tri), y[tri], sample_weight=compute_sample_weight("balanced", y[tri]))
        else: m.fit(iloc(Xtr_all, tri), y[tri])
        oof[vai] = m.predict_proba(iloc(Xtr_all, vai)); pte += m.predict_proba(Xte) / len(folds)
    np.save(fo, oof); np.save(ft, pte)
    print(f"{MODEL} semente {s}: F1={f1_score(y, oof.argmax(1), average='macro'):.4f} ({time.time()-t:.0f}s)", flush=True)

files = sorted(glob.glob(f"{OUT}/oof_{MODEL}_s*.npy"))
O = np.mean([np.load(f) for f in files], axis=0); T = np.mean([np.load(f.replace("oof_", "test_")) for f in files], axis=0)
np.save(f"{OUT}/oof_{MODEL}_ms.npy", O); np.save(f"{OUT}/test_{MODEL}_ms.npy", T)
print(f"{MODEL}_ms: media de {len(files)} semente(s): F1={f1_score(y, O.argmax(1), average='macro'):.4f}")
