"""Diagnostic reporting: feature importance, anomaly analysis, unsupervised profiling, and plots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_config


def _load_report_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Load processed X_train, y_train, raw train DataFrame, and metadata."""
    proc_dir = Path(cfg["paths"]["processed_dir"])
    X_train = pd.read_parquet(proc_dir / "X_train.parquet")
    y_train = np.load(proc_dir / "y_train.npy")
    raw_train = pd.read_csv(cfg["paths"]["train_csv"])
    with (proc_dir / "meta.json").open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return X_train, y_train, raw_train, meta


def run_importance(cfg: dict[str, Any]) -> None:
    """Compute LightGBM gain and SHAP-style feature contributions on X_train."""
    X_train, y_train, _, _ = _load_report_inputs(cfg)
    labels = cfg["age_labels"]
    num_classes = int(cfg["num_classes"])
    seed = int(cfg["seed"])
    report_dir = Path(cfg["paths"]["report_dir"])

    params = dict(cfg["models"]["lgb_tuned"])
    params.update({"random_state": seed, "verbose": -1, "n_jobs": -1})
    model = lgb.LGBMClassifier(**params).fit(X_train, y_train)

    gain = pd.Series(model.booster_.feature_importance("gain"), index=X_train.columns)
    gain = (gain / gain.sum()).sort_values(ascending=False)

    sample_size = min(1200, len(X_train))
    idx = np.random.RandomState(0).choice(len(X_train), sample_size, replace=False)
    contrib = model.booster_.predict(X_train.iloc[idx], pred_contrib=True).reshape(sample_size, num_classes, -1)
    shap_abs = pd.Series(np.abs(contrib[:, :, :-1]).mean(axis=(0, 1)), index=X_train.columns).sort_values(ascending=False)

    top_cols = shap_abs.index[:15]
    by_class = pd.DataFrame(
        {labels[k]: pd.Series(np.abs(contrib[:, k, :-1]).mean(0), index=X_train.columns)[top_cols] for k in range(num_classes)}
    )
    pd.DataFrame({"gain_norm": gain, "mean_abs_contrib": shap_abs}).sort_values(
        "mean_abs_contrib", ascending=False
    ).to_csv(report_dir / "feature_importance.csv")
    by_class.round(4).to_csv(report_dir / "importance_by_age_group.csv")
    print("TOP 20 features (mean |SHAP contrib| across classes):")
    print(shap_abs.head(20).round(4).to_string())


def run_anomalies(cfg: dict[str, Any]) -> None:
    """Characterize supervised discrepancies where predicted age differs by >= threshold."""
    X_train, y_train, raw_train, _ = _load_report_inputs(cfg)
    labels = cfg["age_labels"]
    num_classes = int(cfg["num_classes"])
    th = int(cfg.get("anomaly_threshold", 3))
    oof_dir = Path(cfg["paths"]["oof_dir"])
    report_dir = Path(cfg["paths"]["report_dir"])

    probs = np.load(oof_dir / "stack_best.npy")
    pred = probs.argmax(axis=1)
    exp_age = probs @ np.arange(num_classes)
    diff = pred - y_train

    df = raw_train[[cfg["id_col"], cfg["target_col"]]].copy()
    df["pred"] = pred
    df["exp_idx"] = exp_age.round(2)
    df["diff"] = diff
    df["diff_exp"] = (exp_age - y_train).round(2)
    df["conf_pred"] = probs.max(axis=1).round(3)

    for t in (2, 3, 4):
        print(f"|diff| >= {t}: {(np.abs(diff) >= t).sum()} ({(np.abs(diff) >= t).mean():.1%})")

    ok_mask = np.abs(diff) < 1
    check_cols = [
        "N_DOENCAS", "DIF_SEV_TOT", "N_LIMITACOES", "HS1_ord", "HATLEVEL_ord",
        "HHINCOME_ord", "IMC", "IN43_ord", "SS1_ord", "N_SEM_RESPOSTA",
    ]
    rows = []
    for kind, sel in (("mais_velho", diff >= th), ("mais_novo", diff <= -th)):
        for g in np.unique(y_train[sel]):
            a = (y_train == g) & sel
            b = (y_train == g) & ok_mask
            if a.sum() < 8 or b.sum() < 8:
                continue
            for col in check_cols:
                if col in X_train.columns:
                    rows.append((kind, labels[g], col, int(a.sum()), float(X_train.loc[a, col].mean()), float(X_train.loc[b, col].mean())))

    char_df = pd.DataFrame(rows, columns=["tipo", "grupo_real", "variavel", "n_anom", "media_anom", "media_pares_ok"])
    if not char_df.empty:
        char_df["dif"] = char_df["media_anom"] - char_df["media_pares_ok"]
    df.assign(
        tipo=np.where(diff >= th, "perfil_mais_velho", np.where(diff <= -th, "perfil_mais_novo", ""))
    ).query("tipo != ''").to_csv(report_dir / "anomalies.csv", index=False)
    char_df.round(3).to_csv(report_dir / "anomalias_caracterizacao.csv", index=False)
    print(f"Saved anomaly diagnostics to {report_dir}")


def _build_unsupervised_matrix(X_train: pd.DataFrame, meta: dict[str, Any]) -> np.ndarray:
    """Build scaled dense feature matrix strictly from X_train (no age label, no X_test)."""
    cat_cols = meta["cat_cols"]
    num_cols = meta["num_cols"]
    num_vals = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X_train[num_cols]))
    if cat_cols:
        ohe_vals = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False).fit_transform(
            X_train[cat_cols].astype(str)
        )
        return np.hstack([ohe_vals, num_vals]).astype(np.float32)
    return num_vals.astype(np.float32)


def run_unsupervised(cfg: dict[str, Any], stages: tuple[str, ...] = ("ae", "kmeans", "dbscan")) -> None:
    """Run Autoencoder reconstruction anomaly check, K-Means profiling, and DBSCAN outliers."""
    X_train, y_train, _, meta = _load_report_inputs(cfg)
    labels = cfg["age_labels"]
    th = int(cfg.get("anomaly_threshold", 3))
    oof_dir = Path(cfg["paths"]["oof_dir"])
    report_dir = Path(cfg["paths"]["report_dir"])

    probs = np.load(oof_dir / "stack_best.npy")
    diff = probs.argmax(axis=1) - y_train
    anom = np.abs(diff) >= th

    X_mat = _build_unsupervised_matrix(X_train, meta)
    print(f"Unsupervised matrix (train-only): {X_mat.shape} | supervised anomalies: {anom.sum()}")

    if "ae" in stages:
        ae = MLPRegressor(
            hidden_layer_sizes=(128, 24, 128),
            alpha=1e-3,
            max_iter=40,
            early_stopping=False,
            random_state=int(cfg["seed"]),
            learning_rate_init=1e-3,
        )
        ae.fit(X_mat, X_mat)
        err = ((ae.predict(X_mat) - X_mat) ** 2).mean(axis=1)
        z = pd.Series(err).groupby(y_train).transform(lambda s: (s - s.mean()) / (s.std() + 1e-8)).to_numpy()
        print(
            f"AE reconstruction error: mean={err.mean():.4f} | "
            f"AUC(anom)={roc_auc_score(anom, err):.3f} | AUC(z)={roc_auc_score(anom, z):.3f}"
        )
        pd.DataFrame(
            {"idx": np.arange(len(y_train)), "AGE_RANGE": y_train, "erro_ae": err.round(5), "z_grupo": z.round(3), "anomalia_sup": anom}
        ).to_csv(report_dir / "autoencoder_erros.csv", index=False)

    if "kmeans" in stages or "dbscan" in stages:
        z_pca = PCA(n_components=20, random_state=0).fit_transform(X_mat)

        if "kmeans" in stages:
            sil = {}
            for k in range(3, 9):
                lab = KMeans(k, n_init=5, random_state=0).fit_predict(z_pca)
                sil[k] = silhouette_score(z_pca, lab, sample_size=min(4000, len(z_pca)), random_state=0)
            best_k = max(sil, key=sil.get)
            lab = KMeans(best_k, n_init=10, random_state=0).fit_predict(z_pca)
            print("KMeans silhouette:", {k: round(v, 3) for k, v in sil.items()}, f"| chosen k={best_k}")

            rows = []
            for c in range(best_k):
                mask = lab == c
                rows.append({
                    "cluster": c,
                    "n": int(mask.sum()),
                    "idade_media_idx": round(float(y_train[mask].mean()), 2),
                    "idade_mediana": labels[int(np.median(y_train[mask]))],
                    "p10-p90": f"{labels[int(np.percentile(y_train[mask], 10))]}..{labels[int(np.percentile(y_train[mask], 90))]}",
                    "N_DOENCAS": round(float(X_train.loc[mask, "N_DOENCAS"].mean()), 2),
                    "DIF_SEV_TOT": round(float(X_train.loc[mask, "DIF_SEV_TOT"].mean()), 2),
                    "HS1_ord": round(float(X_train.loc[mask, "HS1_ord"].mean()), 2),
                    "HATLEVEL_ord": round(float(X_train.loc[mask, "HATLEVEL_ord"].mean()), 2),
                    "MAINSTAT": X_train.loc[mask, "MAINSTAT"].astype(str).value_counts().index[0][:40],
                    "%anomalia_sup": f"{anom[mask].mean():.1%}",
                })
            km_df = pd.DataFrame(rows).sort_values("idade_media_idx")
            km_df.to_csv(report_dir / "kmeans_perfis.csv", index=False)

        if "dbscan" in stages:
            z10 = z_pca[:, :10]
            dist = np.sort(NearestNeighbors(n_neighbors=10).fit(z10).kneighbors(z10)[0][:, -1])
            outliers = np.zeros(len(y_train), dtype=bool)
            for q in (0.90, 0.95):
                eps = float(np.quantile(dist, q))
                lab = DBSCAN(eps=eps, min_samples=10).fit_predict(z10)
                outliers = lab == -1
                print(
                    f"DBSCAN eps(q={q:.2f})={eps:.2f}: outliers={outliers.sum()} ({outliers.mean():.1%}) | "
                    f"AUC(anom)={roc_auc_score(anom, outliers):.3f}"
                )
            pd.DataFrame(
                {"idx": np.arange(len(y_train)), "AGE_RANGE": y_train, "outlier_dbscan": outliers, "anomalia_sup": anom}
            ).to_csv(report_dir / "dbscan_outliers.csv", index=False)


def run_characterization(cfg: dict[str, Any]) -> None:
    """Summarize health, lifestyle, and socioeconomic profiles per real age group."""
    X_train, y_train, tr, _ = _load_report_inputs(cfg)
    labels = cfg["age_labels"]
    num_classes = int(cfg["num_classes"])
    report_dir = Path(cfg["paths"]["report_dir"])

    pct = lambda m: 100.0 * pd.Series(m).groupby(y_train).mean()
    df = pd.DataFrame(index=labels)
    df["n"] = np.bincount(y_train, minlength=num_classes)
    df["%mulheres"] = pct(tr["SEX"] == "Mulher").values
    df["%saude_ma_muito_ma"] = pct(tr["HS1"].isin(["Mau", "Muito mau"])).values
    df["n_doencas_cronicas"] = pd.Series(X_train["N_DOENCAS"].values).groupby(y_train).mean().values
    df["%3+_doencas"] = pct(X_train["N_DOENCAS"] >= 3).values
    df["%dif_sensorial_motora"] = pct(X_train["DIF_N_PL"] > 0).values
    df["gravidade_PL"] = pd.Series(X_train["DIF_SEV_PL"].values).groupby(y_train).mean().values
    g = pd.Series(X_train["DIF_SEV_PCHA"].values).where(y_train >= 9).groupby(y_train).mean()
    df["gravidade_PC_HA(55+)"] = g.reindex(range(num_classes)).values

    imc = X_train["IMC"].where((X_train["IMC"] > 12) & (X_train["IMC"] < 60))
    df["IMC_medio"] = pd.Series(imc.values).groupby(y_train).mean().values
    df["%obesidade(IMC>=30)"] = 100.0 * pd.Series((imc >= 30).where(imc.notna())).groupby(y_train).mean().values
    df["%fuma_diariamente"] = pct(tr["IN43"] == "Fuma diariamente").values
    df["%alcool_diario"] = pct(tr["AL1"] == "Todos os dias ou quase todos os dias").values
    df["%rede_social_<=2"] = pct(tr["SS1"].isin(["Nenhuma", "1 ou 2"])).values
    df["%trabalho_fisico_pesado"] = pct(tr["PE1"] == "Em trabalhos pesados ou fisicamente exigentes").values

    basico = ["Nenhum nível completado", "Básico 1º e 2º ciclo", "Básico 3º ciclo"]
    sup = [
        "Bacharelato ou licenciatura 1º ciclo",
        "Licenciatura pré-bolonha ou mestrado 2º ciclo",
        "Doutoramento",
        "C.T.Superior Profissional",
    ]
    df["%ate_ensino_basico"] = pct(tr["HATLEVEL"].isin(basico)).values
    df["%ensino_superior"] = pct(tr["HATLEVEL"].isin(sup)).values
    df["%1o_quintil_rend"] = pct(tr["HHINCOME"] == "1º quintil").values
    df["%5o_quintil_rend"] = pct(tr["HHINCOME"] == "5º quintil").values
    df["%empregado"] = pct(tr["MAINSTAT"].str.startswith("Tem um emprego")).values
    df["%reformado"] = pct(tr["MAINSTAT"].str.startswith("Reformado")).values
    df["%desempregado"] = pct(tr["MAINSTAT"] == "Desempregado").values
    df["%estudante"] = pct(tr["MAINSTAT"].str.startswith("Aluno")).values
    df["%viuvo"] = pct(tr["MARSTALEGAL"] == "Viúvo").values
    df["%casado"] = pct(tr["MARSTALEGAL"] == "Casado").values
    df["%vive_so"] = pct(tr["HHTYPE"] == "Agregado unipessoal").values
    df["%zona_densa"] = pct(tr["DEG_URB"] == "Áreas densamente povoadas").values

    out_path = report_dir / "caracterizacao_faixas.csv"
    df.round(1).to_csv(out_path, index_label="faixa")
    print(f"Saved age group characterization to {out_path}")


def run_figures(cfg: dict[str, Any]) -> None:
    """Generate report PNG figures under out/report/figs/."""
    proc_dir = Path(cfg["paths"]["processed_dir"])
    oof_dir = Path(cfg["paths"]["oof_dir"])
    report_dir = Path(cfg["paths"]["report_dir"])
    figs_dir = Path(cfg["paths"]["figs_dir"])

    y = np.load(proc_dir / "y_train.npy")
    labels = cfg["age_labels"]
    num_classes = int(cfg["num_classes"])
    th = int(cfg.get("anomaly_threshold", 3))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    az, lj, cz = "#2a6f97", "#e07a1f", "#8a8f98"

    def _save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(figs_dir / f"{name}.png", dpi=150)
        plt.close()
        print("Saved figure:", figs_dir / f"{name}.png")

    counts = np.bincount(y, minlength=num_classes)
    plt.figure(figsize=(8, 3.4))
    plt.bar(labels, counts, color=az)
    plt.xticks(rotation=45)
    plt.ylabel("Pessoas no treino")
    plt.title("Distribuição do grupo etário no treino")
    for i, v in enumerate(counts):
        plt.text(i, v + 10, str(v), ha="center", fontsize=7)
    _save("01_distribuicao_classes")

    display_names = {
        "lgb_fe": "LightGBM (base)",
        "lgb_tuned": "LightGBM afinado",
        "xgb": "XGBoost",
        "cat": "CatBoost",
        "rf": "Random Forest",
        "lr": "Regressão logística",
        "et": "ExtraTrees",
        "mlp": "MLP",
        "ord": "Regressão ordinal",
        "stack_best": "Stacking",
    }
    res = [
        (label, float(f1_score(y, np.load(oof_dir / f"{key}.npy").argmax(1), average="macro")))
        for key, label in display_names.items()
        if (oof_dir / f"{key}.npy").exists()
    ]
    if res:
        res.sort(key=lambda item: item[1])
        plt.figure(figsize=(7.5, 3.8))
        colors = [lj if r[0] == "Stacking" else az for r in res]
        plt.barh([r[0] for r in res], [r[1] for r in res], color=colors)
        for i, r in enumerate(res):
            plt.text(r[1] + 0.003, i, f"{r[1]:.3f}", va="center", fontsize=8)
        plt.xlim(0.3, max(r[1] for r in res) + 0.04)
        plt.xlabel("Macro F1 (validação cruzada)")
        plt.title("Comparação de modelos")
        _save("02_comparacao_modelos")

    stack_path = oof_dir / "stack_best.npy"
    if stack_path.exists():
        preds = np.load(stack_path).argmax(axis=1)
        diff = preds - y

        cm = confusion_matrix(y, preds, labels=range(num_classes))
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        plt.figure(figsize=(6.6, 5.8))
        plt.imshow(cmn, cmap="Blues", vmin=0, vmax=0.9)
        plt.xticks(range(num_classes), labels, rotation=60, fontsize=7)
        plt.yticks(range(num_classes), labels, fontsize=7)
        plt.xlabel("Faixa prevista")
        plt.ylabel("Faixa real")
        plt.title("Matriz de confusão normalizada (stacking)")
        plt.colorbar(label="% da faixa real")
        _save("03_matriz_confusao")

        per_f1 = f1_score(y, preds, average=None, labels=range(num_classes))
        plt.figure(figsize=(8, 3.4))
        plt.bar(labels, per_f1, color=[lj if v < 0.35 else az for v in per_f1])
        plt.xticks(rotation=45)
        plt.axhline(per_f1.mean(), color=cz, ls="--", lw=1)
        plt.text(num_classes - 0.6, per_f1.mean() + 0.01, f"média {per_f1.mean():.2f}", ha="right", fontsize=8)
        plt.ylabel("F1")
        plt.title("F1 por faixa etária (stacking)")
        _save("04_f1_por_faixa")

        vc = pd.Series(np.abs(diff)).value_counts().sort_index()
        plt.figure(figsize=(6, 3.4))
        plt.bar(vc.index.astype(str), vc.values / len(y) * 100.0, color=az)
        for i, v in enumerate(vc.values):
            plt.text(i, v / len(y) * 100.0 + 0.5, str(v), ha="center", fontsize=8)
        plt.xlabel("|faixa prevista - faixa real|")
        plt.ylabel("% das pessoas")
        plt.title("Erro em número de faixas")
        _save("05_erro_em_faixas")

        mv = [int(((y == k) & (diff >= th)).sum()) for k in range(num_classes)]
        mn = [int(((y == k) & (diff <= -th)).sum()) for k in range(num_classes)]
        x = np.arange(num_classes)
        plt.figure(figsize=(8.5, 3.6))
        plt.bar(x, mv, color=lj, label=f"perfil mais velho (n={sum(mv)})")
        plt.bar(x, [-v for v in mn], color=az, label=f"perfil mais novo (n={sum(mn)})")
        plt.axhline(0, color="k", lw=0.6)
        plt.xticks(x, labels, rotation=45)
        plt.ylabel("Casos")
        plt.legend(frameon=False, fontsize=8)
        plt.title(f"Anomalias (discrepância >= {th} faixas)")
        _save("06_anomalias_por_faixa")

    fi_path = report_dir / "feature_importance.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path, index_col=0).sort_values("mean_abs_contrib", ascending=False).head(20)[::-1]
        route_cols = {"N_LIMITACOES", "PCHA_APLICAVEL", "NA_PC", "NA_HA"}
        plt.figure(figsize=(7.5, 5))
        plt.barh(fi.index, fi["mean_abs_contrib"], color=[lj if i in route_cols else az for i in fi.index])
        plt.xlabel("Contribuição absoluta média")
        plt.title("Variáveis mais importantes (laranja: encaminhamento 55+)")
        _save("07_importancia_variaveis")

    char_path = report_dir / "caracterizacao_faixas.csv"
    if char_path.exists():
        char_df = pd.read_csv(char_path, index_col=0)
        panels = [
            ("Saúde", ["%saude_ma_muito_ma", "%3+_doencas", "%dif_sensorial_motora"], ["Saúde má/muito má", "3+ doenças", "Dif. sensorial/motora"]),
            ("Trabalho e reforma", ["%empregado", "%reformado", "%ate_ensino_basico"], ["Empregado", "Reformado", "Até ensino básico"]),
            ("Vida familiar", ["%casado", "%viuvo", "%vive_so"], ["Casado", "Viúvo", "Vive só"]),
            ("Hábitos e corpo", ["%fuma_diariamente", "%alcool_diario", "%obesidade(IMC>=30)"], ["Fuma diariamente", "Álcool diário", "Obesidade"]),
        ]
        _, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
        for ax, (title, cols, names) in zip(axes.ravel(), panels):
            for c, nm, col in zip(cols, names, (az, lj, "#5a9e6f")):
                ax.plot(range(num_classes), char_df[c].values, marker="o", ms=3, label=nm, color=col)
            ax.set_title(title)
            ax.set_ylabel("%")
            ax.legend(frameon=False, fontsize=7)
            ax.set_xticks(range(num_classes))
            ax.set_xticklabels(labels, rotation=60, fontsize=7)
        plt.suptitle("Caracterização das faixas etárias")
        _save("08_caracterizacao_faixas")


def run_report(
    cfg: dict[str, Any],
    importance: bool = False,
    anomalies: bool = False,
    unsupervised: bool = False,
    characterization: bool = False,
    figures: bool = False,
    run_all: bool = False,
) -> None:
    """Execute requested diagnostic and reporting stages."""
    if run_all or not any((importance, anomalies, unsupervised, characterization, figures)):
        importance = anomalies = unsupervised = characterization = figures = True

    if importance:
        run_importance(cfg)
    if anomalies:
        run_anomalies(cfg)
    if unsupervised:
        run_unsupervised(cfg)
    if characterization:
        run_characterization(cfg)
    if figures:
        run_figures(cfg)


if __name__ == "__main__":
    run_report(load_config(), run_all=True)
