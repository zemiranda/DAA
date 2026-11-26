"""Feature engineering ronda 2 -> out/features2.pkl
Corrige/complementa N_LIMITACOES (que media sobretudo a aplicabilidade das perguntas PC/HA, so feitas a 55+):
 - dificuldades REAIS (Alguma / Muita / Nao consegue) por dominio e com gravidade
 - contagem de 'Nao aplicavel' e 'Sem resposta' por bloco do questionario (prefixo do nome da coluna)
"""
import re, numpy as np, pandas as pd
tr = pd.read_csv("dataset_train.csv"); te = pd.read_csv("dataset_test_X.csv")
raw = pd.concat([tr.drop(columns=["AGE_RANGE"]), te], ignore_index=True)
feat = pd.read_pickle("out/features.pkl")
assert len(raw) == len(feat)
SEV = {"Alguma dificuldade": 1, "Muita dificuldade": 2, "Não consegue de todo": 3}
dom = {"PL": [c for c in raw.columns if re.fullmatch(r"PL[1-8]", c)],
       "PC": [c for c in raw.columns if re.fullmatch(r"PC1[A-E]", c)],
       "HA": [c for c in raw.columns if re.fullmatch(r"HA1[A-G]", c)]}
new = pd.DataFrame(index=raw.index)
for d, cols in dom.items():
    s = pd.concat([raw[c].map(SEV).fillna(0) for c in cols], axis=1)
    new[f"DIF_N_{d}"] = (s > 0).sum(1); new[f"DIF_SEV_{d}"] = s.sum(1)
new["DIF_N_TOT"] = new[[f"DIF_N_{d}" for d in dom]].sum(1); new["DIF_SEV_TOT"] = new[[f"DIF_SEV_{d}" for d in dom]].sum(1)
new["DIF_SEV_PCHA"] = new["DIF_SEV_PC"] + new["DIF_SEV_HA"]
new["DIF_SEV_MEDIA"] = new["DIF_SEV_TOT"] / (new["DIF_N_TOT"] + 1)
new["PCHA_APLICAVEL"] = (raw["PC1A"] != "Não aplicável").astype(int)    # regra de encaminhamento (55+)
# blocos por prefixo
pref = {}
for c in raw.columns:
    if c in ("PID", "HHID", "PROXY", "REFYEAR", "COUNTRY"): continue
    pref.setdefault(re.match(r"[A-Za-z]+", c).group(0), []).append(c)
for p, cols in pref.items():
    if len(cols) >= 3:
        new[f"NA_{p}"] = sum((raw[c] == "Não aplicável").astype(int) for c in cols)
        new[f"SR_{p}"] = sum((raw[c] == "Sem resposta").astype(int) for c in cols)
# razoes simples
new["CD_POR_LIM"] = feat["N_DOENCAS"] / (new["DIF_N_TOT"] + 1)
new = new.loc[:, new.nunique() > 1]
out = pd.concat([feat, new], axis=1); out.to_pickle("out/features2.pkl")
print("features2:", out.shape, "| novas:", new.shape[1])
yy = tr.AGE_RANGE; n = len(tr)
print("corr(idade):", {c: round(new[c].iloc[:n].corr(yy, method='spearman'), 3) for c in ["DIF_N_TOT", "DIF_SEV_TOT", "DIF_SEV_PCHA", "PCHA_APLICAVEL"]})
