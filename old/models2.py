"""Segunda fase: LR (one-hot), Random Forest (ordinal), blend, ajuste de pesos p/ Macro F1, submissao."""
import numpy as np, pandas as pd, time, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")

OUT, SEED, K = "out", 42, 16
feat = pd.read_pickle(f"{OUT}/features.pkl")
tr = pd.read_csv("dataset_train.csv", usecols=["AGE_RANGE"]); y = tr.AGE_RANGE.values; ntr = len(y)
cat = [c for c in feat.columns if str(feat[c].dtype) == "category"]
num = [c for c in feat.columns if c not in cat]
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(ntr), y))

# --- matrizes
num_df = feat[num].copy()
num_df = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(num_df), columns=num)
oh = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)
Xoh = oh.fit_transform(feat[cat].astype(str))
from scipy import sparse
Xlr = sparse.hstack([Xoh, sparse.csr_matrix(StandardScaler().fit_transform(num_df))]).tocsr()
Xrf = pd.concat([num_df, feat[cat].apply(lambda s: s.cat.codes)], axis=1).values  # codigos ordinais
Xlr_tr, Xlr_te = Xlr[:ntr], Xlr[ntr:]; Xrf_tr, Xrf_te = Xrf[:ntr], Xrf[ntr:]

def cv(make, Xt, Xe, name):
    t = time.time(); oof = np.zeros((ntr, K)); pte = np.zeros((Xe.shape[0], K))
    for tri, vai in folds:
        m = make().fit(Xt[tri], y[tri])
        oof[vai] = m.predict_proba(Xt[vai]); pte += m.predict_proba(Xe) / len(folds)
    print(f"{name}: macroF1={f1_score(y, oof.argmax(1), average='macro'):.4f} ({time.time()-t:.0f}s)", flush=True)
    np.save(f"{OUT}/oof_{name}.npy", oof); np.save(f"{OUT}/test_{name}.npy", pte)

cv(lambda: LogisticRegression(C=0.02, max_iter=300, class_weight="balanced"), Xlr_tr, Xlr_te, "lr")
cv(lambda: RandomForestClassifier(400, min_samples_leaf=3, max_features=0.2, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=SEED), Xrf_tr, Xrf_te, "rf")
