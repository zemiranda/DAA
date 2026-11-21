"""Interpretabilidade + anomalias. Uso: python analysis.py importance | anomalies"""
import sys, os, json, warnings
import numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore")
OUT, SEED, K = "out", 42, 16
STAGE = sys.argv[1]
FEATS = os.environ.get("FEATS", "features.pkl"); TAG = os.environ.get("TAG", "")
feat = pd.read_pickle(f"{OUT}/{FEATS}")
tr = pd.read_csv("dataset_train.csv")
y = tr.AGE_RANGE.values; ntr = len(y)
Xtr = feat.iloc[:ntr].reset_index(drop=True)
LAB = ["15-17","18-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84","85+"]

if STAGE == "importance":
    p = dict(num_leaves=7, colsample_bytree=0.2, subsample=1.0, subsample_freq=1, min_child_samples=80,
             reg_lambda=0.1, max_bin=63, cat_smooth=30, min_data_per_group=50, class_weight="balanced",
             random_state=SEED, verbose=-1, n_jobs=-1, n_estimators=300, learning_rate=0.04)
    m = lgb.LGBMClassifier(**p).fit(Xtr, y)
    gain = pd.Series(m.booster_.feature_importance("gain"), index=Xtr.columns)
    gain = (gain / gain.sum()).sort_values(ascending=False)
    idx = np.random.RandomState(0).choice(ntr, 1200, replace=False)
    C = m.booster_.predict(Xtr.iloc[idx], pred_contrib=True).reshape(len(idx), K, -1)   # (n, K, F+1)
    shap_abs = pd.Series(np.abs(C[:, :, :-1]).mean(axis=(0, 1)), index=Xtr.columns).sort_values(ascending=False)
    # contribuicao media por grupo etario para as top variaveis
    top = shap_abs.index[:15]
    by_class = pd.DataFrame({LAB[k]: pd.Series(np.abs(C[:, k, :-1]).mean(0), index=Xtr.columns)[top] for k in range(K)})
    pd.DataFrame({"gain_norm": gain, "mean_abs_contrib": shap_abs}).sort_values("mean_abs_contrib", ascending=False)\
        .to_csv(f"{OUT}/feature_importance{TAG}.csv")
    by_class.round(4).to_csv(f"{OUT}/importance_by_age_group{TAG}.csv")
    print("TOP 25 (|contribuicao| media, 16 classes):")
    print(shap_abs.head(25).round(4).to_string())
    print("\nTOP 10 por ganho:"); print(gain.head(10).round(4).to_string())

if STAGE == "anomalies":
    P = np.load(f"{OUT}/oof_stack_best.npy")
    pred = P.argmax(1); exp_age = P @ np.arange(K)
    d = pred - y; de = exp_age - y
    df = tr[["PID", "AGE_RANGE"]].copy()
    df["pred"] = pred; df["exp_idx"] = exp_age.round(2); df["diff"] = d; df["diff_exp"] = de.round(2)
    df["conf_pred"] = P.max(1).round(3)
    print("distribuicao de |diff|:", {int(k): int(v) for k, v in pd.Series(np.abs(d)).value_counts().sort_index().items()})
    for t in [2, 3, 4]:
        print(f"|diff|>={t}: {(np.abs(d)>=t).sum()} casos ({(np.abs(d)>=t).mean():.1%})")
    TH = 3
    older = df[d >= TH]; younger = df[d <= -TH]
    print(f"\nanomalias com limiar {TH} grupos: {len(older)} 'perfil mais velho' + {len(younger)} 'perfil mais novo'")
    print("real -> n (mais velho):", older.AGE_RANGE.map(lambda k: LAB[k]).value_counts().head(6).to_dict())
    print("real -> n (mais novo):", younger.AGE_RANGE.map(lambda k: LAB[k]).value_counts().head(6).to_dict())
    # caracterizacao: comparar anomalias com pares do mesmo grupo etario real (nao anomalos)
    X = Xtr.copy(); num = [c for c in X.columns if str(X[c].dtype) != "category"]
    ok = np.abs(d) < 1
    rows = []
    for name, sel in [("mais_velho", d >= TH), ("mais_novo", d <= -TH)]:
        for g in np.unique(y[sel]):
            a = (y == g) & sel; b = (y == g) & ok
            if a.sum() < 8 or b.sum() < 8: continue
            for c in ["N_DOENCAS", "DIF_SEV_TOT", "N_LIMITACOES", "HS1_ord", "HATLEVEL_ord", "HHINCOME_ord", "IMC", "IN43_ord", "SS1_ord", "N_SEM_RESPOSTA"]:
                if c in X.columns:
                    rows.append((name, LAB[g], c, int(a.sum()), X.loc[a, c].mean(), X.loc[b, c].mean()))
    R = pd.DataFrame(rows, columns=["tipo", "grupo_real", "variavel", "n_anom", "media_anom", "media_pares_ok"])
    R["dif"] = R.media_anom - R.media_pares_ok
    agg = R.groupby(["tipo", "variavel"]).apply(lambda g: np.average(g.dif, weights=g.n_anom)).unstack(0).round(3)
    print("\ndiferenca media (anomalos - pares bem previstos do mesmo grupo real):"); print(agg.to_string())
    for c in ["MAINSTAT", "HATLEVEL"]:
        s1 = Xtr.loc[d >= TH, c].astype(str).value_counts(normalize=True).head(3).round(2).to_dict()
        s2 = Xtr.loc[d <= -TH, c].astype(str).value_counts(normalize=True).head(3).round(2).to_dict()
        print(f"{c} (mais velho):", s1, "\n{c} (mais novo):".replace("{c}", c), s2)
    df.assign(tipo=np.where(d >= TH, "perfil_mais_velho", np.where(d <= -TH, "perfil_mais_novo", "")))\
      .query("tipo != ''").to_csv(f"{OUT}/anomalias{TAG}.csv", index=False)
    R.round(3).to_csv(f"{OUT}/anomalias_caracterizacao{TAG}.csv", index=False)
