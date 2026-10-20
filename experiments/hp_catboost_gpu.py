"""Réglage CatBoost GPU : grille de 6-8 configurations en CV complète 5 folds, meilleure config conservée.

Usage : python experiments/hp_catboost_gpu.py [--name NOM] [--threads N] [--folds 5] [--debug]
Sorties : submissions/submission_<nom>_<horodatage>.csv, results/<nom>_<horodatage>.json,
          results/<nom>_<horodatage>_oof.csv et results/<nom>_<horodatage>_test.csv

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

IDEA = ("[copie de x_lgbbag_proba pour le nettoyage, préparation catégorielle de r2_catboost_gpu] "
        "Réglage CatBoost GPU (max_ctr_complexity=1, od_wait=100, 3000 itérations, une graine 42) : "
        "grille depth {6, 8} x learning_rate {0.05, 0.1} x l2_leaf_reg {3, 10} évaluée en CV COMPLÈTE "
        "5 folds (jamais sur un seul fold), des moins chères aux plus chères avec garde-fou de temps ; "
        "puis 2 raffinements de la meilleure config (border_count 254 + one_hot_max_size 10, puis "
        "bootstrap Bernoulli subsample 0.8) ; la meilleure config (F1 OOF) est retenue telle quelle : "
        "son OOF et sa moyenne des 5 folds sur test sont sauvegardés (seuil choisi sur l'OOF)")
SEED = 42
MAX_CAT_CARD = 15  # colonnes avec au plus ce nombre de valeurs distinctes (hors NaN) -> catégorielles

# Budget de temps (secondes depuis le début du script) : on ne lance pas une CV dont l'estimation
# dépasserait ce total ; le « final » ne coûte rien (la config retenue est déjà en OOF).
SEARCH_BUDGET_S = 40 * 60
SLOW_CV_S = 6 * 60  # une CV au-delà : on saute les configurations restantes les plus coûteuses
# Facteurs de coût relatifs pour estimer une CV non encore mesurée
DEPTH_FACTOR = {6: 1.0, 8: 1.9}
LR_FACTOR = {0.1: 1.0, 0.05: 1.6}


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

CB_BASE = {
    "task_type": "GPU",
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "iterations": 3000,
    "od_type": "Iter",
    "od_wait": 100,
    "max_ctr_complexity": 1,
    "random_seed": SEED,
    "verbose": 500,
    "use_best_model": True,
    "allow_writing_files": False,
}
# Grille principale, ordonnée des configurations les moins chères aux plus chères
GRID = [{"depth": d, "learning_rate": lr, "l2_leaf_reg": l2}
        for d in (6, 8) for lr in (0.1, 0.05) for l2 in (3, 10)]
# Raffinements appliqués à la meilleure configuration de la grille
REFINEMENTS = [{"border_count": 254, "one_hot_max_size": 10},
               {"bootstrap_type": "Bernoulli", "subsample": 0.8}]


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


def save_outputs(name, test, test_proba, metrics, info, t0, oof=None):
    """Écrit la soumission et le JSON de résultats ; retourne le chemin du JSON."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (ROOT / "submissions").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    sub_path = ROOT / "submissions" / f"submission_{name}_{stamp}.csv"
    pred = (test_proba >= metrics["threshold"]).astype(int)
    pd.DataFrame({"ID": test["ID"], "TARGET": pred}).to_csv(sub_path, index=False)
    if oof is not None:  # probabilités pour les blends (ordre OOF = train.csv puis extra.csv)
        pd.DataFrame({"oof": oof}).to_csv(ROOT / "results" / f"{name}_{stamp}_oof.csv", index=False, float_format="%.6f")
        pd.DataFrame({"ID": test["ID"], "proba": test_proba}).to_csv(
            ROOT / "results" / f"{name}_{stamp}_test.csv", index=False, float_format="%.6f")
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


# --- Recherche ---------------------------------------------------------------------------------
def describe(params):
    """Résumé lisible des hyperparamètres qui varient."""
    keys = ["depth", "learning_rate", "l2_leaf_reg", "border_count", "one_hot_max_size",
            "bootstrap_type", "subsample"]
    return {k: params[k] for k in keys if k in params}


def run_cv(params, X, y, X_test, cat_cols, folds, label):
    """CV complète (5 folds, mêmes folds que toutes les expériences) ; retourne oof, test_proba, infos."""
    print(f"===== CV {label} : {describe(params)} =====", flush=True)
    tcv = time.time()
    oof = np.zeros(len(X))
    test_proba = np.zeros(len(X_test))
    test_pool = Pool(X_test, cat_features=cat_cols)
    best_iters = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        tf = time.time()
        train_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_cols)
        valid_pool = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_cols)
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool)
        oof[va_idx] = model.predict_proba(valid_pool)[:, 1]
        test_proba += model.predict_proba(test_pool)[:, 1] / folds
        best_iters.append(int(model.get_best_iteration()) + 1)
        print(f"[{label}] fold {fold}: {best_iters[-1]} arbres, "
              f"AUC {roc_auc_score(y[va_idx], oof[va_idx]):.4f}, {time.time() - tf:.0f} s", flush=True)
        del model, train_pool, valid_pool
        gc.collect()
    metrics = evaluate(y, oof)
    elapsed = round(time.time() - tcv, 1)
    print(f"CV {label} : F1 {metrics['f1']:.4f}, AUC {metrics['auc']:.4f}, seuil {metrics['threshold']:.3f} "
          f"({elapsed} s)", flush=True)
    info = {"label": label, "params": describe(params), "f1": metrics["f1"], "auc": metrics["auc"],
            "threshold": metrics["threshold"], "precision": metrics["precision"], "recall": metrics["recall"],
            "best_iterations": best_iters, "cv_time_s": elapsed}
    return oof, test_proba, metrics, info


def estimate_cv_time(params, measured):
    """Estimation du coût d'une CV à partir des CV déjà mesurées (facteurs profondeur / learning_rate)."""
    if not measured:
        return 0.0
    key = (params["depth"], params["learning_rate"])
    if key in measured:
        return measured[key]
    ests = []
    for (d, lr), t in measured.items():
        ests.append(t * DEPTH_FACTOR[params["depth"]] / DEPTH_FACTOR[d]
                    * LR_FACTOR[params["learning_rate"]] / LR_FACTOR[lr])
    return float(np.mean(ests))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="hp_catboost_gpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--debug", action="store_true",
                        help="test local minuscule : 2 000 lignes, CPU, 30 itérations, 2 configs + 1 raffinement")
    args = parser.parse_args()

    t0 = time.time()
    base = {**CB_BASE, "thread_count": args.threads}
    grid, refinements, budget_s = GRID, REFINEMENTS, SEARCH_BUDGET_S
    if args.debug:
        base.update({"task_type": "CPU", "iterations": 30, "od_wait": 10, "verbose": 0})
        grid, refinements, budget_s = GRID[:2], REFINEMENTS[:1], 10 ** 9
    try:
        from catboost.utils import get_gpu_device_count
        print(f"GPU CatBoost détectés : {get_gpu_device_count()}")
    except Exception as e:  # pragma: no cover - diagnostic seulement
        print(f"Détection GPU impossible : {e}")

    train, test = load()
    if args.debug:
        train = train.sample(2000, random_state=SEED).reset_index(drop=True)
        test = test.head(500).reset_index(drop=True)
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

    trials = []  # table des essais (info) dans l'ordre d'exécution
    runs = {}  # label -> (oof, test_proba, metrics)
    skipped = []
    measured = {}  # (depth, lr) -> temps de CV mesuré
    slow_seen = False

    def try_run(label, params):
        nonlocal slow_seen
        est = estimate_cv_time(params, measured)
        elapsed = time.time() - t0
        if elapsed + est > budget_s:
            msg = f"{label} sautée : {elapsed:.0f} s écoulées + {est:.0f} s estimées > budget {budget_s} s"
            print(msg, flush=True)
            skipped.append({"label": label, "params": describe(params), "reason": msg})
            return
        oof, tp, metrics, info = run_cv(params, X, y, X_test, cat_cols, args.folds, label)
        trials.append(info)
        runs[label] = (oof, tp, metrics)
        key = (params["depth"], params["learning_rate"])
        measured[key] = max(measured.get(key, 0.0), info["cv_time_s"])
        if info["cv_time_s"] > SLOW_CV_S:
            slow_seen = True

    # 1) Grille principale, des configurations les moins chères aux plus chères
    for i, g in enumerate(grid):
        label = f"G{i + 1}_d{g['depth']}_lr{g['learning_rate']}_l2{g['l2_leaf_reg']}"
        params = {**base, **g}
        if slow_seen and g["depth"] == 8 and g["learning_rate"] == 0.05:
            msg = f"{label} sautée : une CV a déjà dépassé {SLOW_CV_S} s, configuration la plus coûteuse"
            print(msg, flush=True)
            skipped.append({"label": label, "params": describe(params), "reason": msg})
            continue
        try_run(label, params)

    if not runs:
        raise RuntimeError("aucune configuration évaluée")

    # 2) Raffinements de la meilleure config (chaque raffinement part de la meilleure config courante)
    for j, extra in enumerate(refinements):
        best_label = max(runs, key=lambda k: runs[k][2]["f1"])
        best_params = {**base, **next(t["params"] for t in trials if t["label"] == best_label)}
        label = f"R{j + 1}_" + "_".join(f"{k}{v}" for k, v in extra.items())
        try_run(label, {**best_params, **extra})

    # 3) Final = meilleure configuration (une graine, déjà en OOF)
    best_label = max(runs, key=lambda k: runs[k][2]["f1"])
    oof, test_proba, metrics = runs[best_label]
    best_info = next(t for t in trials if t["label"] == best_label)
    best_params = {**base, **best_info["params"]}
    print("\n--- Table des essais (F1 / AUC OOF, CV complète 5 folds) ---")
    for t in sorted(trials, key=lambda t: -t["f1"]):
        print(f"{t['label']:<40} F1 {t['f1']:.4f}  AUC {t['auc']:.4f}  seuil {t['threshold']:.3f}  "
              f"{t['cv_time_s']:.0f} s  arbres {t['best_iterations']}")
    print(f"Configuration retenue : {best_label} -> F1 OOF {metrics['f1']:.4f}, AUC {metrics['auc']:.4f}")
    save_outputs(args.name, pd.DataFrame({"ID": test_ids}), test_proba, metrics,
                 {"model": "catboost_gpu_grid", "selected_config": best_label, "params": best_params,
                  "best_iterations": best_info["best_iterations"], "n_trials": len(trials),
                  "trials": trials, "skipped": skipped, "search_budget_s": budget_s,
                  "n_features": len(feats), "n_cat_features": len(cat_cols), "cat_features": cat_cols},
                 t0, oof=oof)


if __name__ == "__main__":
    main()
