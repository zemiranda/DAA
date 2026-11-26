"""Modelos adicionais para o blend/stacking, executados por etapas e por folds (retomaveis).
Uso:   python new_models.py <modelo> <fold_inicio> <fold_fim>     modelos: xgb | cat | mlp | ord
       python new_models.py <modelo> merge                         junta os folds e guarda oof_/test_<modelo>.npy
Ex.:   python new_models.py xgb 0 3 ; python new_models.py xgb 3 5 ; python new_models.py xgb merge
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
warnings.filterwarnings("ignore")

OUT, SEED, K = "out", 42, 16
NAME = sys.argv[1]
FEATS = os.environ.get("FEATS", "features.pkl"); TAG = os.environ.get("TAG", ""); ON = NAME + TAG   # FEATS/TAG: variantes de features
feat = pd.read_pickle(f"{OUT}/{FEATS}")
y = pd.read_csv("dataset_train.csv", usecols=["AGE_RANGE"]).AGE_RANGE.values; ntr = len(y)
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(ntr), y))

cat = [c for c in feat.columns if str(feat[c].dtype) == "category"]
num = [c for c in feat.columns if c not in cat]

def ordinal_matrix():          # numericas + codigos das categoricas (para arvores)
    nd = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(feat[num]), columns=num)
    return pd.concat([nd, feat[cat].apply(lambda s: s.cat.codes)], axis=1).values.astype(np.float32)

def onehot_matrix():           # one-hot + numericas padronizadas (para MLP)
    nd = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(feat[num]), columns=num)
    oh = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True).fit_transform(feat[cat].astype(str))
    return sparse.hstack([oh, sparse.csr_matrix(StandardScaler().fit_transform(nd))]).tocsr()

class OrdinalLGB:              # regressao da idade -> pseudo-probabilidades gaussianas em torno da previsao
    def __init__(self, sigma=1.3):
        import lightgbm as lgb
        self.m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=15, colsample_bytree=0.2,
                                   subsample=0.8, subsample_freq=1, min_child_samples=40, reg_lambda=1.0,
                                   random_state=SEED, verbose=-1, n_jobs=-1)
        self.sigma = sigma
    def fit(self, X, yy): self.m.fit(X, yy); return self
    def predict_proba(self, X):
        mu = np.clip(self.m.predict(X), 0, K - 1)[:, None]
        p = np.exp(-((np.arange(K)[None, :] - mu) ** 2) / (2 * self.sigma ** 2)); return p / p.sum(1, keepdims=True)

def build(name):
    if name == "xgb":
        import xgboost as xgb
        X = ordinal_matrix()
        mk = lambda: xgb.XGBClassifier(n_estimators=160, learning_rate=0.08, max_depth=4, subsample=0.8, colsample_bytree=0.3,
                                       min_child_weight=5, reg_lambda=5, tree_method="hist", max_bin=64, n_jobs=-1,
                                       objective="multi:softprob", random_state=SEED, verbosity=0)
        return mk, X, True
    if name == "cat":
        from catboost import CatBoostClassifier
        X = ordinal_matrix()
        mk = lambda: CatBoostClassifier(iterations=250, learning_rate=0.12, depth=5, loss_function="MultiClass",
                                        auto_class_weights="Balanced", rsm=0.3, l2_leaf_reg=5, random_seed=SEED,
                                        verbose=0, thread_count=-1, border_count=64)
        return mk, X, False
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        X = onehot_matrix()
        mk = lambda: MLPClassifier((256, 64), alpha=3e-2, early_stopping=True, validation_fraction=0.1, n_iter_no_change=6,
                                   max_iter=60, learning_rate_init=1e-3, random_state=SEED)
        return mk, X, False
    if name == "ord":
        return (lambda: OrdinalLGB()), ordinal_matrix(), False
    raise SystemExit("modelo desconhecido")

mk, X, use_sw = build(NAME)
Xt, Xe = X[:ntr], X[ntr:]
sl = lambda M, i: M[i] if not sparse.issparse(M) else M[i]

if sys.argv[2] == "merge":
    oof = np.zeros((ntr, K)); pte = np.zeros((Xe.shape[0], K))
    for f, (tri, vai) in enumerate(folds):
        d = np.load(f"{OUT}/part_{ON}_{f}.npz"); oof[vai] = d["oof"]; pte += d["te"] / 5
    print(f"{ON}: macroF1={f1_score(y, oof.argmax(1), average='macro'):.4f} acc={(oof.argmax(1)==y).mean():.4f}")
    np.save(f"{OUT}/oof_{ON}.npy", oof); np.save(f"{OUT}/test_{ON}.npy", pte)
else:
    for f in range(int(sys.argv[2]), int(sys.argv[3])):
        tri, vai = folds[f]; t = time.time(); m = mk()
        if use_sw: m.fit(Xt[tri], y[tri], sample_weight=compute_sample_weight("balanced", y[tri]))
        else: m.fit(Xt[tri], y[tri])
        P = m.predict_proba(Xt[vai])
        full = np.zeros((len(vai), K)); full[:, m.classes_ if hasattr(m, "classes_") else np.arange(K)] = P
        T = m.predict_proba(Xe); ft = np.zeros((Xe.shape[0], K)); ft[:, m.classes_ if hasattr(m, "classes_") else np.arange(K)] = T
        np.savez(f"{OUT}/part_{ON}_{f}.npz", oof=full, te=ft)
        print(f"{ON} fold {f}: F1={f1_score(y[vai], full.argmax(1), average='macro'):.4f} ({time.time()-t:.0f}s)", flush=True)
