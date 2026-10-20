"""Optuna LightGBM en CV complète 5 folds (objectif AUC OOF), puis bagging 5 graines de la meilleure config (base x_lgbbag_proba).

Usage : python train.py [--name NOM] [--threads N] [--folds 5] [--search-s 1800] [--max-trials N] [--rows N]
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
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

IDEA = ("[copie de x_lgbbag_proba, partie modèle remplacée par une recherche Optuna] Réglage sérieux du "
        "LightGBM en CV COMPLÈTE 5 folds (mêmes SKF seed 42, early stopping 100 sur l'AUC de validation "
        "— pas la logloss, que scale_pos_weight > 1 fausse —, une graine par essai) : "
        "Optuna TPESampler(seed=42), objectif = AUC OOF (F1 OOF et logloss reportés), MedianPruner sur l'AUC "
        "de chaque fold (report après chaque fold), timeout ~30 min, premier essai = paramètres de la base "
        "(enqueue_trial) comme témoin. Espace : num_leaves 15-255 (log), min_child_samples 20-2000 (log), "
        "learning_rate 0.02-0.1 (log), feature_fraction 0.2-0.9, bagging_fraction 0.5-1.0 (bagging_freq 1), "
        "lambda_l1 0-10, lambda_l2 0.1-50 (log), min_split_gain 0-0.5, max_bin {63,127,255}, path_smooth 0-10, "
        "scale_pos_weight 1-3. Datasets LightGBM binnés mis en cache par (fold, max_bin) avec "
        "feature_pre_filter=False pour ne pas les reconstruire à chaque essai. Final : 5 graines par fold "
        "(42, 1, 2, 3, 4) avec la meilleure config, OOF et test = moyenne des probabilités, seuil choisi sur l'OOF")
SEED = 42
SEEDS = [42, 1, 2, 3, 4]  # la première graine (42) reproduit exactement la base
SEARCH_TIMEOUT_S = 30 * 60  # budget de la recherche Optuna (vérifié aussi avant chaque fold)
TOTAL_BUDGET_S = 50 * 60  # budget total du script : sert à ajuster le nombre de graines du final si besoin
MAX_BINS = [63, 127, 255]


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

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": SEED,
}
# Paramètres de la base exprimés dans l'espace Optuna (essai témoin n° 0)
BASE_TRIAL = {
    "num_leaves": 63, "min_child_samples": 100, "learning_rate": 0.05, "feature_fraction": 0.5,
    "bagging_fraction": 0.8, "lambda_l1": 0.0, "lambda_l2": 1.0, "min_split_gain": 0.0,
    "max_bin": 255, "path_smooth": 0.0, "scale_pos_weight": 1.0,
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


def best_threshold(y, proba):
    grid = np.arange(0.05, 0.6, 0.005)
    scores = [f1_score(y, proba >= t) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def fast_threshold(y, proba):
    """Même grille et même F1 que best_threshold (2TP / (npos + npred)), en vectorisé (pour les essais)."""
    grid = np.arange(0.05, 0.6, 0.005)
    order = np.argsort(-proba, kind="stable")
    p_sorted, y_sorted = proba[order], y[order]
    cum_tp = np.cumsum(y_sorted)
    npred = np.searchsorted(-p_sorted, -grid, side="right")  # nombre de proba >= t
    tp = np.where(npred > 0, cum_tp[np.maximum(npred - 1, 0)], 0)
    f1 = 2 * tp / np.maximum(y.sum() + npred, 1)
    i = int(np.argmax(f1))
    return float(grid[i]), float(f1[i])


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


# --- Recherche Optuna --------------------------------------------------------------------------
class FoldData:
    """Découpes train/valid par fold et Datasets LightGBM binnés, mis en cache par (fold, max_bin)."""

    def __init__(self, X, y, folds):
        self.X, self.y, self.folds = X, y, folds
        self.X_tr = [X.iloc[tr] for tr, _ in folds]
        self.X_va = [X.iloc[va] for _, va in folds]
        self.cache = {}

    def get(self, k, max_bin):
        key = (k, max_bin)
        if key not in self.cache:
            tr, va = self.folds[k]
            ds_params = {"max_bin": max_bin, "feature_pre_filter": False, "verbose": -1}
            dtr = lgb.Dataset(self.X_tr[k], self.y[tr], params=ds_params, free_raw_data=False)
            dva = lgb.Dataset(self.X_va[k], self.y[va], reference=dtr, params=ds_params, free_raw_data=False)
            self.cache[key] = (dtr, dva)
        return self.cache[key]


def suggest_params(trial):
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 2000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 50.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "max_bin": trial.suggest_categorical("max_bin", MAX_BINS),
        "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 3.0),
    }


def full_params(p, threads, seed=SEED):
    # early stopping sur l'AUC de validation : aligné sur l'objectif Optuna et insensible à scale_pos_weight
    # (avec la logloss non pondérée, un scale_pos_weight > 1 arrêtait l'entraînement après 10-40 arbres)
    return {"objective": "binary", "metric": "auc", "verbose": -1, "num_threads": threads, "bagging_freq": 1,
            "feature_pre_filter": False, "seed": seed, **p}


def train_fold(params, data, k, X_va):
    dtr, dva = data.get(k, params["max_bin"])
    model = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    return model, model.predict(X_va, num_iteration=model.best_iteration)


def make_objective(data, threads, deadline, log):
    y = data.y

    def objective(trial):
        p = suggest_params(trial)
        params = full_params(p, threads)
        t_trial = time.time()
        oof = np.zeros(len(y))
        fold_aucs, iters = [], []
        rec = {"number": trial.number, "params": p, "fold_auc": fold_aucs, "best_iterations": iters}
        log.append(rec)
        for k, (tr, va) in enumerate(data.folds):
            if time.time() > deadline:  # budget de recherche épuisé : on arrête proprement
                rec.update(status="timeout", duration_s=round(time.time() - t_trial, 1))
                print(f"essai {trial.number}: budget de recherche épuisé avant le fold {k}, arrêt de l'étude")
                trial.study.stop()
                raise optuna.TrialPruned()
            model, pred = train_fold(params, data, k, data.X_va[k])
            oof[va] = pred
            auc_k = float(roc_auc_score(y[va], pred))
            fold_aucs.append(auc_k)
            iters.append(int(model.best_iteration))
            trial.report(auc_k, k)
            if trial.should_prune():
                rec.update(status="pruned", duration_s=round(time.time() - t_trial, 1))
                print(f"essai {trial.number}: élagué après le fold {k} (AUC folds {[round(a, 4) for a in fold_aucs]}, "
                      f"{rec['duration_s']} s) {p}")
                raise optuna.TrialPruned()
        auc = float(roc_auc_score(y, oof))
        thr, f1 = fast_threshold(y, oof)
        ll = float(log_loss(y, np.clip(oof, 1e-7, 1 - 1e-7)))
        rec.update(status="complete", auc=auc, f1=f1, threshold=thr, logloss=ll,
                   duration_s=round(time.time() - t_trial, 1))
        trial.set_user_attr("f1", f1)
        trial.set_user_attr("logloss", ll)
        trial.set_user_attr("threshold", thr)
        trial.set_user_attr("duration_s", rec["duration_s"])
        print(f"essai {trial.number}: AUC {auc:.5f} F1 {f1:.4f} (seuil {thr:.3f}) logloss {ll:.5f} "
              f"arbres {iters} ({rec['duration_s']} s) {p}")
        return auc

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="hp_lgbm_optuna")
    parser.add_argument("--threads", type=int, default=0, help="0 = tous les cœurs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--search-s", type=int, default=SEARCH_TIMEOUT_S, help="timeout Optuna en secondes")
    parser.add_argument("--max-trials", type=int, default=None, help="plafond d'essais (test local)")
    parser.add_argument("--rows", type=int, default=0, help="sous-échantillon pour test local (0 = tout)")
    args = parser.parse_args()

    t0 = time.time()
    train, test = load()
    if args.rows:  # test local minuscule de la logique uniquement
        train = train.sample(args.rows, random_state=SEED).reset_index(drop=True)
        test = test.head(args.rows).reset_index(drop=True)
    feats = select_columns(train)
    maxes = train[feats].max()
    _show(train, "AVANT nettoyage (train)")
    nan_before = train[feats].isna().sum().sum()
    clean(train, feats, maxes)
    clean(test, feats, maxes)
    _show(train, "APRÈS nettoyage (train)")
    print(f"NaN dans train : {nan_before} -> {train[feats].isna().sum().sum()}")
    X, y = train[feats], train["TARGET"].values.astype(int)
    X_test = test[feats]
    print(f"train {X.shape}, test {X_test.shape}, taux de positifs {y.mean():.3f}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))
    data = FoldData(X, y, folds)

    # --- Recherche Optuna (une graine par essai, CV complète, pruning sur l'AUC des folds) ---
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=0),
    )
    study.enqueue_trial(BASE_TRIAL)  # essai 0 = paramètres de la base (témoin)
    log = []
    t_search = time.time()
    deadline = t_search + args.search_s
    print(f"Recherche Optuna : timeout {args.search_s} s, max_trials {args.max_trials}, "
          f"prétraitement {t_search - t0:.0f} s")
    study.optimize(make_objective(data, args.threads, deadline, log), timeout=args.search_s,
                   n_trials=args.max_trials, gc_after_trial=True, catch=(lgb.basic.LightGBMError,))
    search_s = time.time() - t_search
    complete = [r for r in log if r.get("status") == "complete"]
    if not complete:
        raise RuntimeError("aucun essai complet : impossible de choisir une configuration")
    best = max(complete, key=lambda r: r["auc"])
    base_rec = log[0]
    n_pruned = sum(r.get("status") == "pruned" for r in log)
    print(f"Recherche terminée : {len(log)} essais ({len(complete)} complets, {n_pruned} élagués) en {search_s:.0f} s")
    print(f"Témoin (essai 0) : {base_rec.get('status')} AUC {base_rec.get('auc')} F1 {base_rec.get('f1')}")
    print(f"Meilleur essai {best['number']} : AUC {best['auc']:.5f} F1 {best['f1']:.4f} ({best['duration_s']} s) "
          f"{best['params']}")
    ranking = sorted(complete, key=lambda r: -r["auc"])[:10]
    print("Top 10 (AUC OOF) :")
    for r in ranking:
        print(f"  essai {r['number']:3d}: AUC {r['auc']:.5f} F1 {r['f1']:.4f} ll {r['logloss']:.5f} {r['params']}")

    # --- Entraînement final : 5 graines par fold avec la meilleure config ---
    remaining = TOTAL_BUDGET_S - (time.time() - t0)
    n_seeds = min(len(SEEDS), max(2, int(remaining / (best["duration_s"] * 1.1)))) if args.rows == 0 else len(SEEDS)
    seeds = SEEDS[:n_seeds]
    if n_seeds < len(SEEDS):
        print(f"Budget restant {remaining:.0f} s : final réduit à {n_seeds} graines {seeds}")
    params = full_params(best["params"], args.threads)
    oof_seeds = np.zeros((n_seeds, len(X)))
    test_proba = np.zeros(len(X_test))
    importance = pd.Series(0.0, index=feats)
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        fold_iters = []
        for k, seed in enumerate(seeds):
            tf = time.time()
            model, pred = train_fold({**params, "seed": seed}, data, fold, data.X_va[fold])
            oof_seeds[k, va_idx] = pred
            test_proba += model.predict(X_test, num_iteration=model.best_iteration) / (n_seeds * args.folds)
            importance += model.feature_importance("gain") / (n_seeds * args.folds)
            fold_iters.append(int(model.best_iteration))
            print(f"fold {fold} seed {seed}: {model.best_iteration} arbres, "
                  f"AUC {roc_auc_score(y[va_idx], pred):.4f} ({time.time() - tf:.0f} s)")
        best_iters.append(fold_iters)
        print(f"fold {fold} moyenne {n_seeds} graines: AUC "
              f"{roc_auc_score(y[va_idx], oof_seeds[:, va_idx].mean(axis=0)):.4f}")

    oof = oof_seeds.mean(axis=0)
    metrics = evaluate(y, oof)
    f1_single_seed42 = evaluate(y, oof_seeds[0])
    f1_cumulative = {}
    for k in range(1, n_seeds + 1):
        m = evaluate(y, oof_seeds[:k].mean(axis=0))
        f1_cumulative[k] = {"f1": m["f1"], "auc": m["auc"], "threshold": m["threshold"]}
        print(f"{k} graine(s) cumulée(s) {seeds[:k]}: F1 {m['f1']:.4f}, AUC {m['auc']:.4f}, "
              f"seuil {m['threshold']:.3f}")
    print(f"Seed 42 seul : F1 {f1_single_seed42['f1']:.4f} -> {n_seeds} graines : F1 {metrics['f1']:.4f} "
          f"(gain {metrics['f1'] - f1_single_seed42['f1']:+.4f})")
    top = importance.sort_values(ascending=False).head(20)
    print("Top 20 variables (gain) :")
    print(top.round(0).to_string())
    save_outputs(args.name, test, test_proba, metrics,
                 {"model": "lightgbm_optuna_seed_bagging", "params": params, "seeds": seeds,
                  "n_features": len(feats), "best_iterations": best_iters,
                  "cv_seed42_only": f1_single_seed42, "cv_cumulative_seeds": f1_cumulative,
                  "optuna": {"objective": "auc_oof", "timeout_s": args.search_s, "search_time_s": round(search_s, 1),
                             "n_trials": len(log), "n_complete": len(complete), "n_pruned": n_pruned,
                             "sampler": "TPESampler(seed=42, n_startup_trials=8)",
                             "pruner": "MedianPruner(n_startup_trials=6, n_warmup_steps=0)",
                             "baseline_trial": base_rec, "best_trial": best, "trials": log},
                  "top_features": top.round(0).to_dict()}, t0, oof=oof)


if __name__ == "__main__":
    main()
