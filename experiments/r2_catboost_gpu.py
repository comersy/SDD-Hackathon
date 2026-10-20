"""Expérience r2 : CatBoost GPU avec variables catégorielles déclarées, sur les codes nettoyés.

Usage : python experiments/r2_catboost_gpu.py [--name NOM] [--threads N] [--folds 5]
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

IDEA = ("CatBoost GPU à la place de LightGBM sur les codes nettoyés (r1_clean_codes) : les colonnes "
        "à <= 15 valeurs distinctes sont déclarées catégorielles (NaN -> -1, int32), les autres restent "
        "numériques avec NaN ; depth=8 / lr=0.06 puis, si le temps le permet, seconde CV complète "
        "depth=6 / lr=0.08, la meilleure des deux est retenue")
SEED = 42
MAX_CAT_CARD = 15  # colonnes avec au plus ce nombre de valeurs distinctes (hors NaN) -> catégorielles
SECOND_CV_BUDGET_S = 8 * 60  # seconde configuration tentée seulement si la première CV a pris moins


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
    "task_type": "GPU",
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "depth": 8,
    "learning_rate": 0.06,
    "iterations": 3000,
    "od_type": "Iter",
    "od_wait": 150,
    "l2_leaf_reg": 3,
    "border_count": 128,
    "one_hot_max_size": 4,
    "random_seed": SEED,
    "verbose": 200,
    "use_best_model": True,
    "allow_writing_files": False,
}
# Seconde configuration (CV complète, mêmes folds), tentée si la première a tenu dans le budget
CB_PARAMS_ALT = {**CB_PARAMS, "depth": 6, "learning_rate": 0.08}


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


# --- Nettoyage des codes de non-réponse (transformation sans TARGET, faite une fois) ---------
# Variables continues calculées par le BRFSS : laissées intactes.
CONTINUOUS_COMPUTED = [
    "_BMI5", "_AGE80", "HTM4", "HTIN4", "WTKG3", "_PACKYRS", "_YRSSMOK", "_DRNKWK2",
    "DROCDY4_", "_PACKDAY", "_YRSQUIT", "_AGEG5YR", "_STATE",
]
# Colonnes à max <= 9 où 7 est une vraie catégorie (EMPLOY1 7 = retraité, _RACE1/_RACEPR1
# 7 = multiracial/hispanique, _INCOMG1 7 = 200 k$+) ; pour RRCOGNT2 7 = « constamment »
# et 8 = « ne sait pas ».
SEVEN_IS_REAL = ["EMPLOY1", "_RACE1", "_RACEPR1", "_INCOMG1", "RRCOGNT2"]
# Âges en années : 77/88 réels, 98 = ne sait pas, 99 = refus.
AGE_COLS = ["DIABAGE4", "CNCRAGE"]
# Âges / comptages codés sur 3 chiffres : 777/999 codes, 888 = « jamais » (LCSFIRST).
AGE3_COLS = ["LCSFIRST", "LCSLAST", "LCSNUMCG"]
# Codes mois-année (MMYYYY) : seuls 777777/999999 sont des codes.
MONTH_YEAR_COLS = ["FLSHTMY3", "HIVTSTD3", "COVIDFS1", "COVIDSE1"]
SPECIAL_COLS = (CONTINUOUS_COMPUTED + SEVEN_IS_REAL + AGE_COLS + AGE3_COLS + MONTH_YEAR_COLS
                + ["WEIGHT2", "HEIGHT3", "ALCDAY4", "CHKHEMO3"])
SHOW_COLS = ["GENHLTH", "PHYSHLTH", "CHILDREN", "WEIGHT2", "ALCDAY4"]


def _set(df, col, mask, value):
    df[col] = df[col].mask(mask, value).astype("float32")


def clean(df, feats, maxes):
    """Nettoie df en place. `maxes` = max par colonne calculé sur train (mêmes règles train/test)."""
    f = np.float32
    for c in feats:
        if c in SPECIAL_COLS:
            continue
        m = maxes[c]
        s = df[c]
        if m <= 9:
            _set(df, c, s.isin([7, 9]), np.nan)
        elif m <= 99:
            _set(df, c, s.isin([77, 99]), np.nan)
            _set(df, c, df[c] == 88, 0)
        elif m <= 999:
            _set(df, c, s.isin([777, 999]), np.nan)
            _set(df, c, df[c] == 888, 0)
        elif m <= 9999:
            _set(df, c, s.isin([7777, 9999]), np.nan)
            _set(df, c, df[c] == 8888, 0)
        # au-delà de 9999 : rien (variables calculées / dates non listées)

    for c in SEVEN_IS_REAL:
        _set(df, c, df[c] == 9, np.nan)
    _set(df, "RRCOGNT2", df["RRCOGNT2"] == 8, np.nan)
    for c in AGE_COLS:
        _set(df, c, df[c].isin([98, 99]), np.nan)
    for c in AGE3_COLS:
        _set(df, c, df[c].isin([777, 999]), np.nan)
        _set(df, c, df[c] == 888, 0)
    for c in MONTH_YEAR_COLS:
        _set(df, c, df[c].isin([777777, 999999]), np.nan)
    # CHKHEMO3 : 1-76 fois, 77 DK, 88 aucune, 98 jamais entendu parler, 99 refus
    _set(df, "CHKHEMO3", df["CHKHEMO3"].isin([77, 98, 99]), np.nan)
    _set(df, "CHKHEMO3", df["CHKHEMO3"] == 88, 0)

    # WEIGHT2 : 50-776 livres ; 9000-9998 = kilos ; 7777/9999 (et valeurs hors plage) -> NaN
    w = df["WEIGHT2"]
    kg = (w >= 9000) & (w <= 9998)
    out = w.where(~kg, (w - 9000) * f(2.20462))
    out = out.mask((w > 776) & (w < 9000) | (w == 9999), np.nan)
    df["WEIGHT2"] = out.astype("float32")

    # HEIGHT3 : 200-711 = pieds*100 + pouces ; 9000-9998 = centimètres ; 7777/9999 -> NaN
    h = df["HEIGHT3"]
    ftin = (h >= 200) & (h <= 711)
    cm = (h >= 9000) & (h <= 9998)
    out = h.where(~ftin, np.floor(h / 100) * 12 + (h % 100))
    out = out.where(~cm, (h - 9000) / f(2.54))
    out = out.mask(~(ftin | cm), np.nan)
    df["HEIGHT3"] = out.astype("float32")

    # ALCDAY4 : 101-107 jours/semaine -> x30/7 ; 201-230 jours/mois ; 888 = 0 ; 777/999 -> NaN
    a = df["ALCDAY4"]
    week = (a >= 101) & (a <= 107)
    month = (a >= 201) & (a <= 230)
    out = a.where(~week, (a - 100) * f(30 / 7))
    out = out.where(~month, a - 200)
    out = out.mask(a == 888, 0)
    out = out.mask(~(week | month | (a == 888)), np.nan)
    df["ALCDAY4"] = out.astype("float32")
    return df


def _show(df, title):
    print(f"--- value_counts {title} ---")
    for c in SHOW_COLS:
        vc = df[c].value_counts(dropna=False).sort_index()
        if len(vc) > 25:  # résumé pour les colonnes à beaucoup de modalités
            print(f"{c}: n_distinct={df[c].nunique()}, NaN={df[c].isna().sum()}, "
                  f"min={df[c].min():.1f}, max={df[c].max():.1f}, mean={df[c].mean():.2f}, "
                  f"tail={vc.tail(8).to_dict()}")
        else:
            print(f"{c}: {vc.to_dict()}")


# --- Catégorielles déclarées pour CatBoost (sans TARGET, après nettoyage) ---------------------
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


def run_cv(params, X, y, test_pool, cat_cols, feats, folds, label):
    """CV complète 5 folds avec une configuration CatBoost ; retourne oof, test_proba, importance, infos."""
    print(f"===== CV {label} : depth={params['depth']}, lr={params['learning_rate']}, "
          f"task_type={params['task_type']} =====")
    tcv = time.time()
    oof = np.zeros(len(X))
    test_proba = np.zeros(test_pool.num_row())
    importance = pd.Series(0.0, index=feats)
    best_iters = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        tf = time.time()
        train_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_cols)
        valid_pool = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_cols)
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool)
        if fold == 0:
            print(f"task_type effectif : {model.get_param('task_type')}")
        oof[va_idx] = model.predict_proba(valid_pool)[:, 1]
        test_proba += model.predict_proba(test_pool)[:, 1] / folds
        importance += pd.Series(model.get_feature_importance(), index=feats) / folds
        best_iters.append(int(model.get_best_iteration()) + 1)
        print(f"fold {fold}: {best_iters[-1]} arbres, "
              f"AUC {roc_auc_score(y[va_idx], oof[va_idx]):.4f}, {time.time() - tf:.0f} s", flush=True)
        del model, train_pool, valid_pool
        gc.collect()
    metrics = evaluate(y, oof)
    elapsed = round(time.time() - tcv, 1)
    print(f"CV {label} : {metrics}  ({elapsed} s)", flush=True)
    info = {"params": params, "cv": metrics, "best_iterations": best_iters, "cv_time_s": elapsed}
    return oof, test_proba, importance, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="r2_catboost_gpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    t0 = time.time()
    try:
        from catboost.utils import get_gpu_device_count
        print(f"GPU CatBoost détectés : {get_gpu_device_count()}")
    except Exception as e:  # pragma: no cover - diagnostic seulement
        print(f"Détection GPU impossible : {e}")

    train, test = load()
    feats = select_columns(train)
    maxes = train[feats].max()
    _show(train, "AVANT nettoyage (train)")
    nan_before = train[feats].isna().sum().sum()
    clean(train, feats, maxes)
    clean(test, feats, maxes)
    _show(train, "APRÈS nettoyage (train)")
    print(f"NaN dans train : {nan_before} -> {train[feats].isna().sum().sum()}")

    cat_cols, num_cols = split_cat_num(train, feats)
    y = train["TARGET"].values.astype(int)
    X = prepare(train, feats, cat_cols)
    X_test = prepare(test, feats, cat_cols)
    test_ids = test["ID"]
    del train, test
    gc.collect()
    print(f"train {X.shape}, test {X_test.shape}, taux de positifs {y.mean():.3f}")
    test_pool = Pool(X_test, cat_features=cat_cols)

    configs = [("A", {**CB_PARAMS, "thread_count": args.threads}),
               ("B", {**CB_PARAMS_ALT, "thread_count": args.threads})]
    runs = []
    for label, params in configs:
        if label != "A" and runs[0][4]["cv_time_s"] >= SECOND_CV_BUDGET_S:
            print(f"CV {label} sautée : la première CV a pris {runs[0][4]['cv_time_s']} s "
                  f"(>= {SECOND_CV_BUDGET_S} s)")
            continue
        runs.append((label, *run_cv(params, X, y, test_pool, cat_cols, feats, args.folds, label)))

    best = max(runs, key=lambda r: r[4]["cv"]["f1"])
    label, oof, test_proba, importance, info = best[0], best[1], best[2], best[3], best[4]
    print(f"Configuration retenue : {label} (F1 OOF {info['cv']['f1']:.4f})")
    metrics = info["cv"]
    top = importance.sort_values(ascending=False).head(20)
    print("Top 20 variables (PredictionValuesChange) :")
    print(top.round(2).to_string())
    save_outputs(args.name, pd.DataFrame({"ID": test_ids}), test_proba, metrics,
                 {"model": "catboost_gpu", "selected_config": label, "params": info["params"],
                  "best_iterations": info["best_iterations"],
                  "all_configs": {r[0]: r[4] for r in runs},
                  "n_features": len(feats), "n_cat_features": len(cat_cols), "cat_features": cat_cols,
                  "top_features": top.round(2).to_dict()}, t0)


if __name__ == "__main__":
    main()
