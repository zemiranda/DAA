"""DAA 2026/27 - pipeline base: pre-processamento, features, CV (OOF) e submissao.
Uso: python pipeline.py  (dados em ./ ; resultados em ./out/)
"""
import os, re, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")

SEED = 42
OUT = "out"; os.makedirs(OUT, exist_ok=True)
DROP = ["PID", "HHID", "COUNTRY", "REFYEAR", "PROXY"]

# ---------------------------------------------------------------- dados
tr = pd.read_csv("dataset_train.csv")
te = pd.read_csv("dataset_test_X.csv")
y = tr["AGE_RANGE"].values
full = pd.concat([tr.drop(columns=["AGE_RANGE"]), te], ignore_index=True)
ntr = len(tr)
full = full.drop(columns=[c for c in DROP if c in full.columns])

# ---------------------------------------------------------------- limpeza
def to_num(s):
    """converte colunas numericas guardadas como texto (>=7, HHMM, ...)"""
    t = s.astype(str).str.replace(r"^>=\s*", "", regex=True)
    return pd.to_numeric(t, errors="coerce")

feat = pd.DataFrame(index=full.index)
n_sr = pd.Series(0, index=full.index)   # "Sem resposta"
n_na = pd.Series(0, index=full.index)   # "Nao aplicavel"
cat_cols, num_cols = [], []

for c in full.columns:
    s = full[c]
    if s.dtype == object or str(s.dtype) == "str":
        n_sr += (s == "Sem resposta").astype(int)
        n_na += (s == "Não aplicável").astype(int)
        num = to_num(s)
        if num.notna().mean() > 0.5:           # coluna numerica em texto
            if s.astype(str).str.fullmatch(r"\d{4}").sum() > 0:   # HHMM -> minutos
                h = s.astype(str).where(s.astype(str).str.fullmatch(r"\d{4}"))
                num = h.str[:2].astype(float) * 60 + h.str[2:].astype(float)
            feat[c] = num
            feat[c + "_NA"] = (s == "Não aplicável").astype(int)
            feat[c + "_SR"] = (s == "Sem resposta").astype(int)
            num_cols.append(c)
        else:
            feat[c] = s.fillna("MISSING").astype("category")
            cat_cols.append(c)
    else:
        feat[c] = s
        num_cols.append(c)

# ordinais conhecidas -> codigo numerico com ordem correta
ORD = {
    "HATLEVEL": ["Nenhum nível completado", "Básico 1º e 2º ciclo", "Básico 3º ciclo", "Secundário",
                 "Pós secundário", "C.T.Superior Profissional", "Bacharelato ou licenciatura 1º ciclo",
                 "Licenciatura pré-bolonha ou mestrado 2º ciclo", "Doutoramento"],
    "HS1": ["Muito bom", "Bom", "Razoável", "Mau", "Muito mau"],
    "HS3": ["Nada limitado", "Limitado mas não severamente", "Severamente limitado"],
    "HHINCOME": ["1º quintil", "2º quintil", "3º quintil", "4º quintil", "5º quintil"],
    "SS1": ["Nenhuma", "1 ou 2", "3 a 5", "6 ou mais"],
    "IN43": ["Não fuma", "Já fumou, mas atualmente não fuma", "Fuma ocasionalmente", "Fuma diariamente"],
}
for c, order in ORD.items():
    if c in full.columns:
        feat[c + "_ord"] = full[c].map({v: i for i, v in enumerate(order)})

# agregados
cd = [c for c in full.columns if re.fullmatch(r"CD1[A-P]", c)]
feat["N_DOENCAS"] = sum((full[c] == "Sim").astype(int) for c in cd)
lim = [c for c in full.columns if re.match(r"(PL[1-8]|PC1[A-E]|HA1[A-G])$", c)]
def difficulty(s):  # tudo o que nao e 'Sim/Não/Sem resposta/Não aplicável' conta como grau de dificuldade
    return (~s.isin(["Não", "Sem resposta", "Não aplicável", "Nenhuma"])).astype(int)
feat["N_LIMITACOES"] = sum(difficulty(full[c]) for c in lim)
feat["N_SEM_RESPOSTA"] = n_sr
feat["N_NAO_APLICAVEL"] = n_na
if {"BM1", "BM2"} <= set(feat.columns):
    h = feat["BM1"] / 100
    feat["IMC"] = feat["BM2"] / (h ** 2)

print("features:", feat.shape, "| categoricas:", len(cat_cols), "| numericas:", len(num_cols))
Xtr, Xte = feat.iloc[:ntr].reset_index(drop=True), feat.iloc[ntr:].reset_index(drop=True)
feat.to_pickle(f"{OUT}/features.pkl")

# ---------------------------------------------------------------- CV
import lightgbm as lgb
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
folds = list(skf.split(Xtr, y))
K = 16

def run_lgb(params, name):
    t = time.time()
    oof = np.zeros((ntr, K)); pte = np.zeros((len(Xte), K))
    for tri, vai in folds:
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr.iloc[tri], y[tri])
        oof[vai] = m.predict_proba(Xtr.iloc[vai])
        pte += m.predict_proba(Xte) / len(folds)
    f1 = f1_score(y, oof.argmax(1), average="macro")
    print(f"{name}: macroF1={f1:.4f} acc={(oof.argmax(1)==y).mean():.4f} ({time.time()-t:.0f}s)")
    np.save(f"{OUT}/oof_{name}.npy", oof); np.save(f"{OUT}/test_{name}.npy", pte)
    return oof, pte

base = dict(n_estimators=250, learning_rate=0.04, num_leaves=15, colsample_bytree=0.3,
            subsample=0.8, subsample_freq=1, min_child_samples=20, reg_lambda=1.0,
            class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=-1)
if __name__ == "__main__":
    run_lgb(base, "lgb_fe")
