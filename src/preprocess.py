"""Leakage-free data cleaning, domain feature engineering, and frozen CV split generation."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import load_config


ORDINAL_MAPS: dict[str, list[str]] = {
    "HATLEVEL": [
        "Nenhum nível completado",
        "Básico 1º e 2º ciclo",
        "Básico 3º ciclo",
        "Secundário",
        "Pós secundário",
        "C.T.Superior Profissional",
        "Bacharelato ou licenciatura 1º ciclo",
        "Licenciatura pré-bolonha ou mestrado 2º ciclo",
        "Doutoramento",
    ],
    "HS1": ["Muito bom", "Bom", "Razoável", "Mau", "Muito mau"],
    "HS3": ["Nada limitado", "Limitado mas não severamente", "Severamente limitado"],
    "HHINCOME": ["1º quintil", "2º quintil", "3º quintil", "4º quintil", "5º quintil"],
    "SS1": ["Nenhuma", "1 ou 2", "3 a 5", "6 ou mais"],
    "IN43": [
        "Não fuma",
        "Já fumou, mas atualmente não fuma",
        "Fuma ocasionalmente",
        "Fuma diariamente",
    ],
}

SEVERITY_MAP: dict[str, int] = {
    "Alguma dificuldade": 1,
    "Muita dificuldade": 2,
    "Não consegue de todo": 3,
}


def _to_numeric_series(s: pd.Series) -> pd.Series:
    """Parse numeric values stored as strings (e.g. '>=7')."""
    cleaned = s.astype(str).str.replace(r"^>=\s*", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _is_hhmm_series(s: pd.Series) -> bool:
    """Detect whether a string column encodes 4-digit HHMM timestamps."""
    return bool(s.astype(str).str.fullmatch(r"\d{4}").sum() > 0)


def _parse_hhmm(s: pd.Series) -> pd.Series:
    """Convert HHMM strings into total minutes while keeping non-matching as NaN."""
    as_str = s.astype(str)
    valid = as_str.where(as_str.str.fullmatch(r"\d{4}"))
    return valid.str[:2].astype(float) * 60.0 + valid.str[2:].astype(float)


def _infer_column_schema(
    raw_train: pd.DataFrame,
    drop_cols: list[str],
) -> dict[str, Any]:
    """Infer text-numeric, HHMM, and categorical columns strictly from training data."""
    cols = [c for c in raw_train.columns if c not in drop_cols]
    text_num_cols: list[str] = []
    hhmm_cols: list[str] = []
    cat_cols: list[str] = []
    pure_num_cols: list[str] = []

    for c in cols:
        s = raw_train[c]
        if not pd.api.types.is_numeric_dtype(s):
            parsed = _to_numeric_series(s)
            if parsed.notna().mean() > 0.5:
                text_num_cols.append(c)
                if _is_hhmm_series(s):
                    hhmm_cols.append(c)
            else:
                cat_cols.append(c)
        else:
            pure_num_cols.append(c)

    return {
        "cols": cols,
        "text_num_cols": text_num_cols,
        "hhmm_cols": set(hhmm_cols),
        "cat_cols": cat_cols,
        "pure_num_cols": pure_num_cols,
    }


def _difficulty_flag(s: pd.Series) -> pd.Series:
    """Return 1 for any limitation level beyond 'Não'/'Nenhuma'/missing codes."""
    return (~s.isin(["Não", "Sem resposta", "Não aplicável", "Nenhuma"])).astype(np.int8)


def _engineer_rowwise(
    raw: pd.DataFrame,
    schema: dict[str, Any],
    min_prefix_cols: int = 3,
) -> pd.DataFrame:
    """Apply deterministic row-wise cleaning and domain feature engineering."""
    df = raw[schema["cols"]].copy()
    cols_dict: dict[str, Any] = {}

    n_sr = pd.Series(0, index=df.index, dtype=np.int16)
    n_na = pd.Series(0, index=df.index, dtype=np.int16)

    for c in schema["text_num_cols"]:
        s = df[c]
        n_sr = n_sr + (s == "Sem resposta").astype(np.int16)
        n_na = n_na + (s == "Não aplicável").astype(np.int16)
        num = _parse_hhmm(s) if c in schema["hhmm_cols"] else _to_numeric_series(s)
        cols_dict[c] = num.astype(np.float32)
        cols_dict[f"{c}_NA"] = (s == "Não aplicável").astype(np.int8)
        cols_dict[f"{c}_SR"] = (s == "Sem resposta").astype(np.int8)

    for c in schema["cat_cols"]:
        s = df[c]
        n_sr = n_sr + (s == "Sem resposta").astype(np.int16)
        n_na = n_na + (s == "Não aplicável").astype(np.int16)
        cols_dict[c] = s.fillna("MISSING").astype(str)

    for c in schema["pure_num_cols"]:
        cols_dict[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    # Known ordinal mappings
    for c, order in ORDINAL_MAPS.items():
        if c in df.columns:
            mapping = {v: float(i) for i, v in enumerate(order)}
            cols_dict[f"{c}_ord"] = df[c].map(mapping).astype(np.float32)

    # Chronic conditions & baseline limitations
    cd_cols = [c for c in df.columns if re.fullmatch(r"CD1[A-P]", c)]
    if cd_cols:
        cols_dict["N_DOENCAS"] = sum((df[c] == "Sim").astype(np.int16) for c in cd_cols)

    lim_cols = [c for c in df.columns if re.match(r"(PL[1-8]|PC1[A-E]|HA1[A-G])$", c)]
    if lim_cols:
        cols_dict["N_LIMITACOES"] = sum(_difficulty_flag(df[c]) for c in lim_cols)

    cols_dict["N_SEM_RESPOSTA"] = n_sr
    cols_dict["N_NAO_APLICAVEL"] = n_na

    # BMI (preserve valid extreme values per Macro F1 edge-case rules)
    if {"BM1", "BM2"} <= set(cols_dict):
        height_m = cols_dict["BM1"] / 100.0
        cols_dict["IMC"] = (cols_dict["BM2"] / (height_m ** 2)).replace([np.inf, -np.inf], np.nan).astype(np.float32)

    # Functional difficulty severity by domain (from fe2.py)
    domains = {
        "PL": [c for c in df.columns if re.fullmatch(r"PL[1-8]", c)],
        "PC": [c for c in df.columns if re.fullmatch(r"PC1[A-E]", c)],
        "HA": [c for c in df.columns if re.fullmatch(r"HA1[A-G]", c)],
    }
    for dom_name, cols in domains.items():
        if cols:
            sev_df = pd.concat([df[c].map(SEVERITY_MAP).fillna(0) for c in cols], axis=1)
            cols_dict[f"DIF_N_{dom_name}"] = (sev_df > 0).sum(axis=1).astype(np.int16)
            cols_dict[f"DIF_SEV_{dom_name}"] = sev_df.sum(axis=1).astype(np.int16)

    if all(f"DIF_N_{d}" in cols_dict for d in ("PL", "PC", "HA")):
        cols_dict["DIF_N_TOT"] = (cols_dict["DIF_N_PL"] + cols_dict["DIF_N_PC"] + cols_dict["DIF_N_HA"]).astype(np.int16)
        cols_dict["DIF_SEV_TOT"] = (cols_dict["DIF_SEV_PL"] + cols_dict["DIF_SEV_PC"] + cols_dict["DIF_SEV_HA"]).astype(np.int16)
        cols_dict["DIF_SEV_PCHA"] = (cols_dict["DIF_SEV_PC"] + cols_dict["DIF_SEV_HA"]).astype(np.int16)
        cols_dict["DIF_SEV_MEDIA"] = (cols_dict["DIF_SEV_TOT"] / (cols_dict["DIF_N_TOT"] + 1.0)).astype(np.float32)
        if "N_DOENCAS" in cols_dict:
            cols_dict["CD_POR_LIM"] = (cols_dict["N_DOENCAS"] / (cols_dict["DIF_N_TOT"] + 1.0)).astype(np.float32)

    if "PC1A" in df.columns:
        cols_dict["PCHA_APLICAVEL"] = (df["PC1A"] != "Não aplicável").astype(np.int8)

    # Missingness counts per questionnaire block prefix (from fe2.py)
    prefixes: dict[str, list[str]] = {}
    for c in df.columns:
        m = re.match(r"[A-Za-z]+", c)
        if m:
            prefixes.setdefault(m.group(0), []).append(c)
    for prefix, cols in prefixes.items():
        if len(cols) >= min_prefix_cols:
            cols_dict[f"NA_{prefix}"] = sum((df[c] == "Não aplicável").astype(np.int16) for c in cols)
            cols_dict[f"SR_{prefix}"] = sum((df[c] == "Sem resposta").astype(np.int16) for c in cols)

    # Life-stage flags and composite health/socioeconomic indicators
    if "MAINSTAT" in df.columns:
        ms = df["MAINSTAT"].astype(str)
        cols_dict["IS_RETIRED"] = ms.str.startswith("Reformado").astype(np.int8)
        cols_dict["IS_STUDENT"] = ms.str.startswith("Aluno").astype(np.int8)
        cols_dict["IS_EMPLOYED"] = ms.str.startswith("Tem um emprego").astype(np.int8)
        cols_dict["IS_UNEMPLOYED"] = (ms == "Desempregado").astype(np.int8)

    if "MARSTALEGAL" in df.columns:
        cols_dict["IS_WIDOWED"] = (df["MARSTALEGAL"] == "Viúvo").astype(np.int8)
        cols_dict["IS_MARRIED"] = (df["MARSTALEGAL"] == "Casado").astype(np.int8)
        cols_dict["IS_SINGLE"] = (df["MARSTALEGAL"] == "Solteiro").astype(np.int8)

    if "HHTYPE" in df.columns:
        hht = df["HHTYPE"].astype(str)
        cols_dict["LIVES_ALONE"] = (hht == "Agregado unipessoal").astype(np.int8)
        cols_dict["HAS_CHILD_LT25"] = hht.str.contains("< 25 anos", regex=False).astype(np.int8)

    if {"MD1", "MD2"} <= set(df.columns):
        cols_dict["USES_MEDICATION"] = ((df["MD1"] == "Sim") | (df["MD2"] == "Sim")).astype(np.int8)

    if {"HH_ACT", "HH_INACT"} <= set(cols_dict):
        denom = cols_dict["HH_ACT"].fillna(0) + cols_dict["HH_INACT"].fillna(0) + 1.0
        cols_dict["HH_ACT_RATIO"] = (cols_dict["HH_ACT"].fillna(0) / denom).astype(np.float32)

    # Cumulative health risk & socioeconomic stability scores
    if {"N_DOENCAS", "HS1_ord"} <= set(cols_dict):
        smoke = (cols_dict["IN43_ord"].fillna(0) >= 2).astype(np.float32) if "IN43_ord" in cols_dict else 0.0
        alcohol = (df["AL1"] == "Todos os dias ou quase todos os dias").astype(np.float32) if "AL1" in df.columns else 0.0
        cols_dict["HEALTH_RISK_SCORE"] = (
            cols_dict["N_DOENCAS"].fillna(0).astype(np.float32)
            + cols_dict["HS1_ord"].fillna(2).astype(np.float32)
            + smoke
            + alcohol
        ).astype(np.float32)

    if {"HATLEVEL_ord", "HHINCOME_ord"} <= set(cols_dict):
        emp = cols_dict.get("IS_EMPLOYED", pd.Series(0, index=df.index)).astype(np.float32)
        cols_dict["SES_STABILITY_SCORE"] = (
            cols_dict["HATLEVEL_ord"].fillna(0).astype(np.float32)
            + cols_dict["HHINCOME_ord"].fillna(0).astype(np.float32)
            + 2.0 * emp
        ).astype(np.float32)

    return pd.DataFrame(cols_dict, index=df.index)


def _audit_data(
    y_train: np.ndarray,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    age_labels: list[str],
) -> None:
    """Log training class distribution and feature-only train/test missingness drift."""
    counts = np.bincount(y_train)
    pct = 100.0 * counts / len(y_train)
    summary = {age_labels[i]: f"{counts[i]} ({pct[i]:.1f}%)" for i in range(len(counts))}
    print("Training class distribution:", summary)

    num_cols = [c for c in X_train.columns if str(X_train[c].dtype) != "category"]
    if num_cols:
        tr_na = X_train[num_cols].isna().mean()
        te_na = X_test[num_cols].isna().mean()
        drift = (tr_na - te_na).abs().sort_values(ascending=False)
        top_drift = drift[drift > 0.03].head(5)
        if len(top_drift) > 0:
            print("Feature NaN-rate drift |train - test| > 3%:", top_drift.round(4).to_dict())
        else:
            print("Covariate check: no numeric column has >3% NaN-rate drift between train and test.")


def run_preprocess(cfg: dict[str, Any]) -> None:
    """Run full preprocessing pipeline and save train/test parquet + frozen folds."""
    paths = cfg["paths"]
    target_col = cfg["target_col"]
    drop_cols = list(cfg["preprocess"]["drop_cols"])
    min_prefix = int(cfg["preprocess"].get("min_prefix_cols", 3))

    raw_train = pd.read_csv(paths["train_csv"])
    raw_test = pd.read_csv(paths["test_csv"])

    y_train = raw_train[target_col].to_numpy(dtype=np.int64)
    raw_train_X = raw_train.drop(columns=[target_col])

    schema = _infer_column_schema(raw_train_X, drop_cols)
    X_train = _engineer_rowwise(raw_train_X, schema, min_prefix_cols=min_prefix)
    X_test = _engineer_rowwise(raw_test, schema, min_prefix_cols=min_prefix)

    # Add binary missingness indicator flags strictly for columns with NaNs in X_train
    cat_set = set(schema["cat_cols"])
    na_flag_cols: list[str] = []
    for c in list(X_train.columns):
        if c not in cat_set and not c.endswith(("_NA", "_SR")) and X_train[c].isna().any():
            na_flag_cols.append(c)

    if na_flag_cols:
        tr_flags = pd.DataFrame(
            {f"{c}_isna": X_train[c].isna().astype(np.int8) for c in na_flag_cols},
            index=X_train.index,
        )
        te_flags = pd.DataFrame(
            {f"{c}_isna": X_test[c].isna().astype(np.int8) for c in na_flag_cols},
            index=X_test.index,
        )
        X_train = pd.concat([X_train, tr_flags], axis=1)
        X_test = pd.concat([X_test, te_flags], axis=1)

    # Drop columns that are constant in X_train (never inspect X_test variance)
    keep_cols = [c for c in X_train.columns if X_train[c].nunique(dropna=False) > 1]
    X_train = X_train[keep_cols].copy()
    X_test = X_test[keep_cols].copy()

    # Freeze categorical vocabulary strictly from X_train
    final_cat_cols = [c for c in schema["cat_cols"] if c in X_train.columns]
    for c in final_cat_cols:
        categories = sorted(X_train[c].astype(str).unique().tolist())
        cat_dtype = pd.CategoricalDtype(categories=categories, ordered=False)
        X_train[c] = X_train[c].astype(str).astype(cat_dtype)
        X_test[c] = X_test[c].astype(str).astype(cat_dtype)

    final_num_cols = [c for c in X_train.columns if c not in set(final_cat_cols)]

    # Freeze StratifiedKFold indices
    skf = StratifiedKFold(
        n_splits=int(cfg["n_splits"]),
        shuffle=True,
        random_state=int(cfg["seed"]),
    )
    folds = np.empty(len(X_train), dtype=np.int32)
    for fold_idx, (_, val_idx) in enumerate(skf.split(X_train, y_train)):
        folds[val_idx] = fold_idx

    proc_dir = Path(paths["processed_dir"])
    X_train.to_parquet(proc_dir / "X_train.parquet", index=False)
    X_test.to_parquet(proc_dir / "X_test.parquet", index=False)
    np.save(proc_dir / "y_train.npy", y_train)
    np.save(proc_dir / "folds.npy", folds)

    meta = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "num_classes": int(cfg["num_classes"]),
        "cat_cols": final_cat_cols,
        "num_cols": final_num_cols,
    }
    with (proc_dir / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(
        f"Saved processed data to {proc_dir}: "
        f"X_train={X_train.shape}, X_test={X_test.shape}, "
        f"categorical={len(final_cat_cols)}, numeric={len(final_num_cols)}"
    )
    _audit_data(y_train, X_train, X_test, cfg["age_labels"])


if __name__ == "__main__":
    run_preprocess(load_config())
