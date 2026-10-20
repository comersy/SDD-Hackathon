"""Expérience r1 : CatBoost avec variables catégorielles déclarées, à la place de LightGBM.

Usage : python experiments/r1_catboost.py [--name NOM] [--threads N] [--folds 5]
Sorties : submissions/submission_<nom>_<horodatage>.csv et results/<nom>_<horodatage>.json

Protocole d'évaluation, à conserver à l'identique dans toute expérience pour que les scores
restent comparables :
- entraînement sur train.csv + extra.csv concaténés (370 132 lignes), test.csv sert
  uniquement à prédire ;
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42), prédictions out-of-fold ;
- tout prétraitement qui utilise TARGET est ajusté dans chaque fold sur la partie
  entraînement seulement ;
- le seuil de décision est choisi sur les prédictions OOF pour maximiser le F1 ;
  le F1 reporté est ce F1 OOF, au seuil retenu.
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

IDEA = ("CatBoost à la place de LightGBM, mêmes colonnes : les colonnes à ≤ 15 valeurs distinctes "
        "sont déclarées catégorielles (NaN -> -1, int), les autres restent numériques avec NaN")
SEED = 42
MAX_CAT_CARD = 15  # colonnes avec au plus ce nombre de valeurs distinctes (hors NaN) -> catégorielles


def find_root():
    """Dossier du dépôt (celui qui contient train.csv), quel que soit l'emplacement du script."""
    here = Path(__file__).resolve().parent
    for d in [here, *here.parents]:
        if (d / "train.csv").exists():
            return d
    raise FileNotFoundError("train.csv introuvable : lance le script depuis le dépôt")


ROOT = find_root()

# Colonnes sans rapport avec la santé de la personne (cf. codebook BRFSS)
DATE_COLS = ["FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR"]
ADMIN_COLS = [
    "ID", "DISPCODE", "SEQNO", "_PSU", "QSTVER", "QSTLANG",
    # Écran d'appel (fixe / portable, logement, état de résidence)
    "CTELENM1", "PVTRESD1", "COLGHOUS", "STATERE1", "CELPHON1", "LADULT1", "COLGSEX1",
    "LANDSEX1", "RESPSLCT", "SAFETIME", "CTELNUM1", "CELLFON5", "CADULT1", "CELLSEX1",
    "PVTRESD3", "CCLGHOUS", "CSTATE1", "LANDLINE",
]
WEIGHT_COLS = [
    "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT",
    "_DUALUSE", "_DUALCOR", "_LLCPWT2", "_LLCPWT",
]
DROP_COLS = DATE_COLS + ADMIN_COLS + WEIGHT_COLS

CB_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "depth": 6,
    "learning_rate": 0.08,
    "iterations": 3000,
    "od_type": "Iter",
    "od_wait": 100,
    "l2_leaf_reg": 3,
    "border_count": 128,
    "one_hot_max_size": 4,
    "max_ctr_complexity": 1,
    "random_seed": SEED,
    "verbose": 200,
    "use_best_model": True,
    "allow_writing_files": False,
}


def _read_csv(path):
    df = pd.read_csv(path, low_memory=False)
    for c in df.columns:
        if c == "ID":
            df[c] = df[c].astype("int64")
        elif c == "TARGET":
            df[c] = df[c].astype("int8")
        else:
            df[c] = df[c].astype("float32")
    return df


def load():
    """train.csv + extra.csv concaténés, et test.csv. Cache parquet dans data/ (ignoré par git)."""
    cache = ROOT / "data"
    cache.mkdir(exist_ok=True)
    train_pq, test_pq = cache / "train_extra.parquet", cache / "test.parquet"
    if train_pq.exists() and test_pq.exists():
        return pd.read_parquet(train_pq), pd.read_parquet(test_pq)
    train = pd.concat([_read_csv(ROOT / "train.csv"), _read_csv(ROOT / "extra.csv")],
                      ignore_index=True)
    test = _read_csv(ROOT / "test.csv")
    for df, path in [(train, train_pq), (test, test_pq)]:
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp)
        os.replace(tmp, path)  # écriture atomique : plusieurs scripts peuvent tourner en même temps
    return train, test


def select_columns(train):
    feats = [c for c in train.columns if c not in DROP_COLS and c != "TARGET"]
    constant = [c for c in feats if train[c].nunique(dropna=False) <= 1]
    feats = [c for c in feats if c not in constant]
    print(f"{len(feats)} colonnes gardées, {len(DROP_COLS)} retirées par règle, "
          f"{len(constant)} constantes : {constant}")
    return feats


def split_cat_num(train, feats):
    """Colonnes catégorielles = ≤ MAX_CAT_CARD valeurs distinctes (hors NaN) ; ne dépend pas de TARGET."""
    nun = train[feats].nunique(dropna=True)
    cat_cols = [c for c in feats if nun[c] <= MAX_CAT_CARD]
    num_cols = [c for c in feats if c not in cat_cols]
    print(f"{len(cat_cols)} colonnes catégorielles (<= {MAX_CAT_CARD} valeurs), {len(num_cols)} numériques")
    return cat_cols, num_cols


def prepare(df, feats, cat_cols):
    """Catégorielles : NaN -> -1 puis int32 ; numériques : float32 avec NaN (CatBoost les gère)."""
    out = df[feats].copy()
    out[cat_cols] = out[cat_cols].fillna(-1).astype("int32")
    return out


def best_threshold(y, proba):
    grid = np.arange(0.05, 0.6, 0.005)
    scores = [f1_score(y, proba >= t) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def evaluate(y, oof):
    threshold, f1 = best_threshold(y, oof)
    pred = oof >= threshold
    return {
        "f1": f1,
        "threshold": threshold,
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "auc": float(roc_auc_score(y, oof)),
    }


def save_outputs(name, test, test_proba, metrics, info, t0):
    """Écrit la soumission et le JSON de résultats ; retourne le chemin du JSON."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (ROOT / "submissions").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    sub_path = ROOT / "submissions" / f"submission_{name}_{stamp}.csv"
    pred = (test_proba >= metrics["threshold"]).astype(int)
    pd.DataFrame({"ID": test["ID"], "TARGET": pred}).to_csv(sub_path, index=False)
    result = {
        "name": name,
        "idea": IDEA,
        "timestamp": stamp,
        "cv": metrics,
        "test_positive_rate": float(pred.mean()),
        "submission": f"submissions/{sub_path.name}",
        "runtime_s": round(time.time() - t0, 1),
        **info,
    }
    res_path = ROOT / "results" / f"{name}_{stamp}.json"
    res_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("CV :", {k: round(v, 4) for k, v in metrics.items()})
    print(f"Soumission : {sub_path}\nRésultats : {res_path}  ({result['runtime_s']} s)")
    return res_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="r1_catboost")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    t0 = time.time()
    train, test = load()
    feats = select_columns(train)
    cat_cols, num_cols = split_cat_num(train, feats)
    y = train["TARGET"].values.astype(int)
    X = prepare(train, feats, cat_cols)
    X_test = prepare(test, feats, cat_cols)
    test_ids = test["ID"]
    del train, test
    gc.collect()
    print(f"train {X.shape}, test {X_test.shape}, taux de positifs {y.mean():.3f}")

    params = {**CB_PARAMS, "thread_count": args.threads}
    oof = np.zeros(len(X))
    test_proba = np.zeros(len(X_test))
    importance = pd.Series(0.0, index=feats)
    test_pool = Pool(X_test, cat_features=cat_cols)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        tf = time.time()
        train_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_cols)
        valid_pool = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_cols)
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool)
        oof[va_idx] = model.predict_proba(valid_pool)[:, 1]
        test_proba += model.predict_proba(test_pool)[:, 1] / args.folds
        importance += pd.Series(model.get_feature_importance(), index=feats) / args.folds
        print(f"fold {fold}: {model.get_best_iteration() + 1} arbres, "
              f"AUC {roc_auc_score(y[va_idx], oof[va_idx]):.4f}, {time.time() - tf:.0f} s")
        del model, train_pool, valid_pool
        gc.collect()

    metrics = evaluate(y, oof)
    top = importance.sort_values(ascending=False).head(20)
    print("Top 20 variables (PredictionValuesChange) :")
    print(top.round(2).to_string())
    save_outputs(args.name, pd.DataFrame({"ID": test_ids}), test_proba, metrics,
                 {"model": "catboost", "params": params, "n_features": len(feats),
                  "n_cat_features": len(cat_cols), "cat_features": cat_cols,
                  "top_features": top.round(2).to_dict()}, t0)


if __name__ == "__main__":
    main()
