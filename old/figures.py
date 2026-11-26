"""Gera as figuras do relatorio/checkpoint em out/figs/ (PNG, 150 dpi).
Requer: dataset_train.csv e, em out/: oof_*.npy (modelos), oof_stack_best.npy (stack_final.py),
feature_importance_f2.csv (analysis.py importance com FEATS=features2.pkl TAG=_f2) e caracterizacao_faixas.csv (characterization.py).
Figuras em falta por falta de ficheiros sao saltadas com um aviso.
Uso: python figures.py
"""
import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
warnings.filterwarnings("ignore")

OUT = "out"; FIG = f"{OUT}/figs"; os.makedirs(FIG, exist_ok=True)
LAB = ["15-17","18-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84","85+"]
y = pd.read_csv("dataset_train.csv", usecols=["AGE_RANGE"]).AGE_RANGE.values
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
AZ, LJ, CZ = "#2a6f97", "#e07a1f", "#8a8f98"
def save(name): plt.tight_layout(); plt.savefig(f"{FIG}/{name}.png", dpi=150); plt.close(); print("ok:", name)
have = lambda p: os.path.exists(f"{OUT}/{p}")

# 1 - distribuicao das classes
n = np.bincount(y, minlength=16)
plt.figure(figsize=(8, 3.4)); plt.bar(LAB, n, color=AZ); plt.xticks(rotation=45)
plt.ylabel("Pessoas no treino"); plt.title("Distribuição do grupo etário no treino (classes desequilibradas)")
for i, v in enumerate(n): plt.text(i, v + 10, str(v), ha="center", fontsize=7)
save("01_distribuicao_classes")

# 2 - comparacao de modelos (Macro F1 em validacao cruzada)
nomes = {"lgb_fe": "LightGBM (features)", "lgb_tuned": "LightGBM afinado", "xgb": "XGBoost", "cat": "CatBoost", "rf": "Random Forest",
         "lr": "Regressão logística", "et": "ExtraTrees", "mlp": "MLP", "ord": "Regressão ordinal", "stack_best": "Stacking"}
res = [(v, f1_score(y, np.load(f"{OUT}/oof_{k}.npy").argmax(1), average="macro")) for k, v in nomes.items() if have(f"oof_{k}.npy")]
res.sort(key=lambda t: t[1])
plt.figure(figsize=(7.5, 3.8)); cores = [LJ if r[0] == "Stacking" else AZ for r in res]
plt.barh([r[0] for r in res], [r[1] for r in res], color=cores)
for i, r in enumerate(res): plt.text(r[1] + 0.003, i, f"{r[1]:.3f}", va="center", fontsize=8)
plt.xlim(0.3, max(r[1] for r in res) + 0.04); plt.xlabel("Macro F1 (validação cruzada, 5 folds)"); plt.title("Comparação de modelos")
save("02_comparacao_modelos")

if have("oof_stack_best.npy"):
    P = np.load(f"{OUT}/oof_stack_best.npy"); p = P.argmax(1); d = p - y
    # 3 - matriz de confusao normalizada
    cm = confusion_matrix(y, p, labels=range(16)); cmn = cm / cm.sum(1, keepdims=True)
    plt.figure(figsize=(6.6, 5.8)); plt.imshow(cmn, cmap="Blues", vmin=0, vmax=0.9)
    plt.xticks(range(16), LAB, rotation=60, fontsize=7); plt.yticks(range(16), LAB, fontsize=7)
    plt.xlabel("Faixa prevista"); plt.ylabel("Faixa real"); plt.title("Matriz de confusão normalizada (stacking)"); plt.colorbar(label="% da faixa real")
    save("03_matriz_confusao")
    # 4 - F1 por faixa
    f1 = f1_score(y, p, average=None, labels=range(16))
    plt.figure(figsize=(8, 3.4)); plt.bar(LAB, f1, color=[LJ if v < 0.35 else AZ for v in f1]); plt.xticks(rotation=45)
    plt.axhline(f1.mean(), color=CZ, ls="--", lw=1); plt.text(15.4, f1.mean() + 0.01, f"média {f1.mean():.2f}", ha="right", fontsize=8)
    plt.ylabel("F1"); plt.title("F1 por faixa etária (stacking; a laranja, F1 < 0,35)")
    save("04_f1_por_faixa")
    # 5 - distribuicao do erro em faixas
    ad = np.abs(d); vc = pd.Series(ad).value_counts().sort_index()
    plt.figure(figsize=(6, 3.4)); plt.bar(vc.index.astype(str), vc.values / len(y) * 100, color=AZ)
    for i, v in enumerate(vc.values): plt.text(i, v / len(y) * 100 + 0.5, f"{v}", ha="center", fontsize=8)
    plt.xlabel("|faixa prevista − faixa real|"); plt.ylabel("% das pessoas"); plt.title("Erro em número de faixas (n por barra)")
    save("05_erro_em_faixas")
    # 6 - anomalias: quem sao (limiar 3) por faixa real
    TH = 3; mv = [((y == k) & (d >= TH)).sum() for k in range(16)]; mn = [((y == k) & (d <= -TH)).sum() for k in range(16)]
    x = np.arange(16); plt.figure(figsize=(8.5, 3.6))
    plt.bar(x, mv, color=LJ, label=f"perfil mais velho (n={sum(mv)})"); plt.bar(x, [-v for v in mn], color=AZ, label=f"perfil mais novo (n={sum(mn)})")
    plt.axhline(0, color="k", lw=0.6); plt.xticks(x, LAB, rotation=45); plt.ylabel("Casos"); plt.legend(frameon=False, fontsize=8)
    plt.title(f"Anomalias (discrepância ≥ {TH} faixas) por faixa real")
    save("06_anomalias_por_faixa")

# 7 - importancia de variaveis
if have("feature_importance_f2.csv"):
    fi = pd.read_csv(f"{OUT}/feature_importance_f2.csv", index_col=0).sort_values("mean_abs_contrib", ascending=False).head(20)[::-1]
    rota = {"N_LIMITACOES", "PCHA_APLICAVEL", "NA_PC", "NA_HA"}   # sinais de encaminhamento (55+)
    plt.figure(figsize=(7.5, 5)); plt.barh(fi.index, fi.mean_abs_contrib, color=[LJ if i in rota else AZ for i in fi.index])
    plt.xlabel("Contribuição absoluta média (16 classes)"); plt.title("Variáveis mais importantes (laranja: encaminhamento 55+)")
    save("07_importancia_variaveis")

# 8 - caracterizacao por faixa
if have("caracterizacao_faixas.csv"):
    D = pd.read_csv(f"{OUT}/caracterizacao_faixas.csv", index_col=0)
    paineis = [("Saúde", ["%saude_ma_muito_ma", "%3+_doencas", "%dif_sensorial_motora"], ["Saúde má/muito má", "3+ doenças crónicas", "Dificuldade sensorial/motora"]),
               ("Trabalho e reforma", ["%empregado", "%reformado", "%ate_ensino_basico"], ["Empregado", "Reformado", "Até ensino básico"]),
               ("Vida familiar", ["%casado", "%viuvo", "%vive_so"], ["Casado", "Viúvo", "Vive só"]),
               ("Hábitos e corpo", ["%fuma_diariamente", "%alcool_diario", "%obesidade(IMC>=30)"], ["Fuma diariamente", "Álcool diário", "Obesidade"])]
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for a, (tit, cols, nomes_) in zip(ax.ravel(), paineis):
        for c, nm, cor in zip(cols, nomes_, [AZ, LJ, "#5a9e6f"]): a.plot(range(16), D[c].values, marker="o", ms=3, label=nm, color=cor)
        a.set_title(tit); a.set_ylabel("%"); a.legend(frameon=False, fontsize=7)
        a.set_xticks(range(16)); a.set_xticklabels(LAB, rotation=60, fontsize=7)
    plt.suptitle("Caracterização das faixas etárias (idade real, treino)"); save("08_caracterizacao_faixas")
print("figuras em", FIG)
