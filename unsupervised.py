"""Metodos nao supervisionados (anomalias e caracterizacao). Requer out/features.pkl e out/oof_stack_best.npy
Uso: python unsupervised.py ae | kmeans | dbscan
"""
import sys, warnings
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, silhouette_score
warnings.filterwarnings("ignore")
OUT, K, TH = "out", 16, 3
STAGE = sys.argv[1]
LAB = ["15-17","18-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84","85+"]
feat = pd.read_pickle(f"{OUT}/features.pkl")
y = pd.read_csv("dataset_train.csv", usecols=["AGE_RANGE"]).AGE_RANGE.values; ntr = len(y)
cat = [c for c in feat.columns if str(feat[c].dtype) == "category"]; num = [c for c in feat.columns if c not in cat]
nd = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(feat[num]), columns=num)
oh = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False).fit_transform(feat[cat].astype(str))
X = np.hstack([oh, StandardScaler().fit_transform(nd)]).astype(np.float32)   # SEM a idade (nao supervisionado)
Xtr = X[:ntr]
P = np.load(f"{OUT}/oof_stack_best.npy"); d = P.argmax(1) - y
anom = np.abs(d) >= TH            # anomalias supervisionadas (modelo vs idade real)
print("matriz:", X.shape, "| anomalias supervisionadas:", anom.sum())

if STAGE == "ae":
    from sklearn.neural_network import MLPRegressor
    ae = MLPRegressor(hidden_layer_sizes=(128, 24, 128), alpha=1e-3, max_iter=40, early_stopping=False, random_state=42, learning_rate_init=1e-3)
    ae.fit(X, X)   # autoencoder: reconstroi a propria entrada (gargalo de 24 neuronios)
    err = ((ae.predict(Xtr) - Xtr) ** 2).mean(1)
    z = pd.Series(err).groupby(y).transform(lambda s: (s - s.mean()) / s.std()).values   # z-score dentro do grupo etario real
    print(f"erro de reconstrucao: media={err.mean():.4f} p95={np.percentile(err,95):.4f}")
    print(f"AUC (erro AE -> anomalia supervisionada): {roc_auc_score(anom, err):.3f} | com z-score por grupo: {roc_auc_score(anom, z):.3f}")
    print(f"AUC so 'perfil mais velho': {roc_auc_score(d>=TH, z):.3f} | so 'perfil mais novo': {roc_auc_score(d<=-TH, z):.3f}")
    top = np.argsort(-err)[: int(0.03 * ntr)]
    print("top 3% de erro AE: % que sao anomalias supervisionadas:", f"{anom[top].mean():.1%}", "(base:", f"{anom.mean():.1%})")
    print("grupos reais mais frequentes no top 3%:", pd.Series([LAB[k] for k in y[top]]).value_counts().head(5).to_dict())
    pd.DataFrame({"idx": np.arange(ntr), "AGE_RANGE": y, "erro_ae": err.round(5), "z_grupo": z.round(3), "anomalia_sup": anom}).to_csv(f"{OUT}/autoencoder_erros.csv", index=False)

if STAGE in ("kmeans", "dbscan"):
    Z = PCA(n_components=20, random_state=0).fit_transform(Xtr)
    print("variancia explicada PCA20: n/a (usado para distancias)")

if STAGE == "kmeans":
    from sklearn.cluster import KMeans
    sil = {}
    for k in range(3, 9):
        lab = KMeans(k, n_init=5, random_state=0).fit_predict(Z); sil[k] = silhouette_score(Z, lab, sample_size=4000, random_state=0)
    print("silhouette por k:", {k: round(v, 3) for k, v in sil.items()})
    k = max(sil, key=sil.get); lab = KMeans(k, n_init=10, random_state=0).fit_predict(Z); print("k escolhido:", k)
    Xf = feat.iloc[:ntr].reset_index(drop=True)
    rows = []
    for c in range(k):
        s = lab == c
        r = {"cluster": c, "n": int(s.sum()), "idade_media_idx": round(y[s].mean(), 2), "idade_mediana": LAB[int(np.median(y[s]))],
             "p10-p90": f"{LAB[int(np.percentile(y[s],10))]}..{LAB[int(np.percentile(y[s],90))]}",
             "N_DOENCAS": round(Xf.loc[s, "N_DOENCAS"].mean(), 2), "N_LIMITACOES": round(Xf.loc[s, "N_LIMITACOES"].mean(), 2),
             "HS1_ord": round(Xf.loc[s, "HS1_ord"].mean(), 2), "HATLEVEL_ord": round(Xf.loc[s, "HATLEVEL_ord"].mean(), 2),
             "MAINSTAT": Xf.loc[s, "MAINSTAT"].astype(str).value_counts().index[0][:40], "%anomalia_sup": f"{anom[s].mean():.1%}"}
        rows.append(r)
    R = pd.DataFrame(rows).sort_values("idade_media_idx"); print(R.to_string(index=False))
    R.to_csv(f"{OUT}/kmeans_perfis.csv", index=False)

if STAGE == "dbscan":
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    Z10 = Z[:, :10]
    dist = np.sort(NearestNeighbors(n_neighbors=10).fit(Z10).kneighbors(Z10)[0][:, -1])
    for q in [0.90, 0.95]:
        eps = float(np.quantile(dist, q)); lab = DBSCAN(eps=eps, min_samples=10).fit_predict(Z10)
        out = lab == -1
        print(f"eps(q{q})={eps:.2f}: outliers={out.sum()} ({out.mean():.1%}) clusters={len(set(lab))-1} | % outliers que sao anomalias sup.: {anom[out].mean():.1%} (base {anom.mean():.1%}) | AUC={roc_auc_score(anom, out):.3f}")
    print("grupos reais dos outliers (q0.95):", pd.Series([LAB[k] for k in y[out]]).value_counts().head(5).to_dict())
    pd.DataFrame({"idx": np.arange(ntr), "AGE_RANGE": y, "outlier_dbscan": out, "anomalia_sup": anom}).to_csv(f"{OUT}/dbscan_outliers.csv", index=False)
