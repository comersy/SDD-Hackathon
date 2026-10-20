"""XGBoost GPU réglé par Optuna en CV complète (5 folds), puis bagging 5 graines par fold (base x_lgbbag_proba).

Usage : python train.py [--name NOM] [--threads N] [--folds 5] [--quick]
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

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

IDEA = ("[copie de x_lgbbag_proba : même load/select_columns/clean/folds/seuil/save_outputs] "
        "XGBoost GPU (device=cuda, hist, binary:logistic, logloss, early stopping 100, 5000 arbres max) "
        "réglé par Optuna en CV COMPLÈTE 5 folds (objectif = AUC OOF, TPESampler seed 42, MedianPruner "
        "sur l'AUC cumulée fold par fold, timeout ~26 min, DMatrix par fold construites une seule fois) ; "
        "espace : max_depth 4-10, learning_rate 0.02-0.1 log, min_child_weight 1-100 log, subsample 0.5-1, "
        "colsample_bytree 0.2-0.9, colsample_bynode 0.5-1, reg_alpha 0-10, reg_lambda 0.1-50 log, gamma 0-2, "
        "max_bin {128, 256} ; premier essai = config r4 connue (depth 8, mcw 100, colsample 0.3). "
        "Final : meilleure config entraînée avec 5 graines (42, 1, 2, 3, 4) par fold, chacune avec son early "
        "stopping ; OOF et test = moyenne des 5 graines x 5 folds ; seuil choisi sur l'OOF moyenné")
SEED = 42
SEEDS = [42, 1, 2, 3, 4]  # la première graine (42) reproduit exactement la base


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

XGB_BASE = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cuda",
    "seed": SEED,
}
NUM_BOOST_ROUND = 5000
EARLY_STOPPING = 100
# Budget temps : 45 min de calcul distant au total ; la recherche Optuna s'arrête à OPTUNA_TIMEOUT ou
# plus tôt s'il faut garder FINAL_RESERVE pour l'entraînement final (5 graines x 5 folds).
TOTAL_BUDGET_S = 44 * 60
OPTUNA_TIMEOUT_S = 26 * 60
FINAL_RESERVE_S = 12 * 60
MAX_TRIALS = 60
# Premier essai = meilleure config du réglage manuel r4_xgb_fullcv_tuning (AUC OOF 0.8546), pour partir
# d'un bon point ; le TPE explore ensuite.
WARM_START = {"max_depth": 8, "learning_rate": 0.05, "min_child_weight": 100, "subsample": 0.8,
              "colsample_bytree": 0.3, "colsample_bynode": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0,
              "gamma": 0.0, "max_bin": 256}


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


def suggest_params(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.9),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256]),
    }


def train_fold(params, dtrain, dvalid, dtest, n_rounds):
    """Un XGBoost avec early stopping sur le fold de validation ; renvoie (proba valid, proba test, best_it)."""
    bst = xgb.train(params, dtrain, n_rounds, evals=[(dvalid, "valid")],
                    early_stopping_rounds=EARLY_STOPPING, verbose_eval=False)
    rng = (0, bst.best_iteration + 1)
    pv = bst.predict(dvalid, iteration_range=rng)
    pt = bst.predict(dtest, iteration_range=rng) if dtest is not None else None
    it = int(bst.best_iteration)
    del bst
    return pv, pt, it


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="hp_xgb_optuna")
    parser.add_argument("--threads", type=int, default=0, help="0 = tous les cœurs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="test local minuscule (2 000 lignes, 2 essais, CPU)")
    args = parser.parse_args()

    t0 = time.time()
    train, test = load()
    feats = select_columns(train)
    maxes = train[feats].max()
    _show(train, "AVANT nettoyage (train)")
    nan_before = train[feats].isna().sum().sum()
    clean(train, feats, maxes)
    clean(test, feats, maxes)
    _show(train, "APRÈS nettoyage (train)")
    print(f"NaN dans train : {nan_before} -> {train[feats].isna().sum().sum()}")
    if args.quick:
        train = train.sample(2000, random_state=SEED).reset_index(drop=True)
        test = test.head(500).reset_index(drop=True)
    X, y = train[feats], train["TARGET"].values.astype(int)
    X_test = test[feats]
    print(f"train {X.shape}, test {X_test.shape}, taux de positifs {y.mean():.3f}")

    base = {**XGB_BASE, "nthread": args.threads if args.threads > 0 else os.cpu_count()}
    n_rounds, optuna_timeout, max_trials = NUM_BOOST_ROUND, OPTUNA_TIMEOUT_S, MAX_TRIALS
    if args.quick:
        base["device"] = "cpu"
        n_rounds, optuna_timeout, max_trials = 20, 60, 2
    print(f"XGBoost {xgb.__version__}, optuna {optuna.__version__}, device {base['device']}, "
          f"nthread {base['nthread']}")

    # DMatrix par fold construites une seule fois (NaN natifs ; aucun prétraitement n'utilise TARGET)
    ts = time.time()
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    folds = []
    for tr_idx, va_idx in skf.split(X, y):
        folds.append((va_idx, xgb.DMatrix(X.iloc[tr_idx], y[tr_idx]), xgb.DMatrix(X.iloc[va_idx], y[va_idx])))
    dtest = xgb.DMatrix(X_test)
    print(f"DMatrix des {args.folds} folds + test construites ({time.time() - ts:.0f} s)")

    # ---------------- Recherche Optuna en CV complète (objectif = AUC OOF) ----------------
    trials_log = []

    def objective(trial):
        hp = suggest_params(trial)
        params = {**base, **hp}
        tt = time.time()
        oof_t = np.zeros(len(y))
        iters, aucs = [], []
        for k, (va_idx, dtr, dva) in enumerate(folds):
            pv, _, it = train_fold(params, dtr, dva, None, n_rounds)
            oof_t[va_idx] = pv
            iters.append(it)
            aucs.append(float(roc_auc_score(y[va_idx], pv)))
            # valeur intermédiaire = AUC moyenne des folds déjà faits (MedianPruner)
            trial.report(float(np.mean(aucs)), step=k)
            if trial.should_prune():
                trials_log.append({"trial": trial.number, "params": hp, "state": "pruned", "folds_done": k + 1,
                                   "auc_partial": float(np.mean(aucs)), "best_iterations": iters,
                                   "time_s": round(time.time() - tt, 1)})
                print(f"essai {trial.number} élagué après {k + 1} fold(s) : AUC partielle {np.mean(aucs):.4f} "
                      f"({time.time() - tt:.0f} s) {hp}", flush=True)
                raise optuna.TrialPruned()
        auc = float(roc_auc_score(y, oof_t))
        thr, f1 = best_threshold(y, oof_t)
        trial.set_user_attr("f1", f1)
        trial.set_user_attr("threshold", thr)
        trials_log.append({"trial": trial.number, "params": hp, "state": "complete", "auc": auc, "f1": f1,
                           "threshold": thr, "best_iterations": iters, "time_s": round(time.time() - tt, 1)})
        print(f"essai {trial.number}: AUC {auc:.5f} F1 {f1:.4f} seuil {thr:.3f} arbres {iters} "
              f"({time.time() - tt:.0f} s) {hp}", flush=True)
        return auc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    study.enqueue_trial(WARM_START)
    remaining = TOTAL_BUDGET_S - (time.time() - t0) - FINAL_RESERVE_S
    timeout = max(60.0, min(optuna_timeout, remaining))
    print(f"Optuna : timeout {timeout:.0f} s, au plus {max_trials} essais", flush=True)
    ts = time.time()
    study.optimize(objective, n_trials=max_trials, timeout=timeout, gc_after_trial=True)
    search_time = time.time() - ts
    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    best = study.best_trial
    best_hp = dict(best.params)
    print(f"Recherche : {len(study.trials)} essais ({n_complete} complets, {n_pruned} élagués) en "
          f"{search_time:.0f} s ; meilleur essai {best.number} AUC {best.value:.5f} "
          f"F1 {best.user_attrs.get('f1', float('nan')):.4f} {best_hp}", flush=True)
    table = sorted((t for t in trials_log if t["state"] == "complete"), key=lambda d: -d["auc"])
    print("Top 10 essais (AUC OOF / F1 OOF) :")
    for t in table[:10]:
        print(f"  #{t['trial']:2d} AUC {t['auc']:.5f} F1 {t['f1']:.4f} {t['params']}")

    # ---------------- Entraînement final : 5 graines par fold avec la meilleure config ----------------
    params = {**base, **best_hp}
    n_seeds = len(SEEDS)
    oof_seeds = np.zeros((n_seeds, len(X)))
    test_proba = np.zeros(len(X_test))
    best_iters = []
    tf0 = time.time()
    for fold, (va_idx, dtr, dva) in enumerate(folds):
        fold_iters = []
        for k, seed in enumerate(SEEDS):
            tf = time.time()
            pv, pt, it = train_fold({**params, "seed": seed}, dtr, dva, dtest, n_rounds)
            oof_seeds[k, va_idx] = pv
            test_proba += pt / (n_seeds * args.folds)
            fold_iters.append(it)
            print(f"fold {fold} seed {seed}: {it} arbres, AUC {roc_auc_score(y[va_idx], pv):.4f} "
                  f"({time.time() - tf:.0f} s)", flush=True)
        best_iters.append(fold_iters)
        print(f"fold {fold} moyenne {n_seeds} graines: AUC "
              f"{roc_auc_score(y[va_idx], oof_seeds[:, va_idx].mean(axis=0)):.4f}", flush=True)
    final_time = time.time() - tf0

    oof = oof_seeds.mean(axis=0)
    metrics = evaluate(y, oof)
    f1_single_seed42 = evaluate(y, oof_seeds[0])
    f1_cumulative = {}
    for k in range(1, n_seeds + 1):
        m = evaluate(y, oof_seeds[:k].mean(axis=0))
        f1_cumulative[k] = {"f1": m["f1"], "auc": m["auc"], "threshold": m["threshold"]}
        print(f"{k} graine(s) cumulée(s) {SEEDS[:k]}: F1 {m['f1']:.4f}, AUC {m['auc']:.4f}, "
              f"seuil {m['threshold']:.3f}")
    print(f"Seed 42 seul : F1 {f1_single_seed42['f1']:.4f} -> {n_seeds} graines : F1 {metrics['f1']:.4f} "
          f"(gain {metrics['f1'] - f1_single_seed42['f1']:+.4f})")
    save_outputs(args.name, test, test_proba, metrics,
                 {"model": "xgboost_gpu_optuna_seed_bagging", "params": params, "seeds": SEEDS,
                  "n_features": len(feats), "best_iterations": best_iters,
                  "cv_seed42_only": f1_single_seed42, "cv_cumulative_seeds": f1_cumulative,
                  "optuna": {"n_trials": len(study.trials), "n_complete": n_complete, "n_pruned": n_pruned,
                             "search_time_s": round(search_time, 1), "final_time_s": round(final_time, 1),
                             "best_trial": best.number, "best_auc": float(best.value),
                             "best_f1": best.user_attrs.get("f1"), "best_params": best_hp,
                             "warm_start": WARM_START, "trials": trials_log}}, t0, oof=oof)


if __name__ == "__main__":
    main()
