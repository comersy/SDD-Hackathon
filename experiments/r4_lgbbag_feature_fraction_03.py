"""Blend LightGBM seed-bagged (5 graines, feature_fraction 0.3) + XGBoost GPU, base r3_lgbbag_xgb_blend.

Usage : python train.py [--name NOM] [--threads N] [--folds 5]
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
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

IDEA = ("feature_fraction 0.3 (au lieu de 0.5) pour le LightGBM seed-bagged (5 graines, early stopping "
        "100) du blend LGB+XGB de r3_lgbbag_xgb_blend : un sous-échantillonnage de colonnes plus fort "
        "rend les graines plus différentes et amplifie le gain du seed averaging ; XGBoost cuda, "
        "BLEND_WEIGHTS, graines, folds (StratifiedKFold random_state=42) et choix du seuil inchangés ; "
        "si le fold 0 est rapide (< 90 s), seconde CV complète à feature_fraction 0.4 (XGBoost réutilisé), "
        "les deux stockées dans all_configs et la meilleure retenue sur le F1 OOF blendé")
SEED = 42
SEEDS = [42, 1, 2, 3, 4]  # mêmes graines que la base
BLEND_WEIGHTS = [0.5, 0.6, 0.7]  # poids du LightGBM seed-bagged dans le blend
FEATURE_FRACTIONS = [0.3, 0.4]  # 0.3 = l'idée ; 0.4 seulement si le fold 0 est assez rapide
SECOND_CONFIG_MAX_FOLD0_S = 90.0  # seuil de temps du fold 0 (5 graines LGB + XGB) pour lancer ff 0.4

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cuda",
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "colsample_bynode": 0.8,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "max_bin": 256,
    "seed": SEED,
}


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
    "feature_fraction": 0.3,  # seul changement par rapport à r3_lgbbag_xgb_blend (0.5)
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": SEED,
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


def run_xgb(xgb_params, X, y, X_test, feats, folds):
    """XGBoost GPU, un modèle par fold (identique à la base) ; entraîné une seule fois et partagé."""
    oof_xgb = np.zeros(len(X))
    test_proba_xgb = np.zeros(len(X_test))
    importance_xgb = pd.Series(0.0, index=feats)
    best_iters_xgb = []
    fold_times = []
    dtest = xgb.DMatrix(X_test)  # construit une fois, NaN natifs
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        tf = time.time()
        dtrain = xgb.DMatrix(X.iloc[tr_idx], y[tr_idx])
        dvalid = xgb.DMatrix(X.iloc[va_idx], y[va_idx])
        bst = xgb.train(xgb_params, dtrain, 5000, evals=[(dvalid, "valid")],
                        early_stopping_rounds=100, verbose_eval=False)
        it_range = (0, bst.best_iteration + 1)
        oof_xgb[va_idx] = bst.predict(dvalid, iteration_range=it_range)
        test_proba_xgb += bst.predict(dtest, iteration_range=it_range) / folds
        gain = bst.get_score(importance_type="gain")
        importance_xgb += pd.Series(gain).reindex(feats).fillna(0.0) / folds
        best_iters_xgb.append(int(bst.best_iteration))
        if fold == 0:
            # vérification que le GPU est bien utilisé (sinon plusieurs minutes par fold sur 8 cœurs)
            cfg = json.loads(bst.save_config())
            print("XGBoost config device :", cfg.get("learner", {}).get("generic_param", {}).get("device"),
                  "| tree_method :", cfg.get("learner", {}).get("gradient_booster", {})
                  .get("gbtree_train_param", {}).get("tree_method"))
        fold_times.append(time.time() - tf)
        print(f"fold {fold} xgb: {bst.best_iteration} arbres, "
              f"AUC {roc_auc_score(y[va_idx], oof_xgb[va_idx]):.4f} ({fold_times[-1]:.0f} s)")
        del dtrain, dvalid, bst
    return {"oof": oof_xgb, "test_proba": test_proba_xgb, "importance": importance_xgb,
            "best_iters": best_iters_xgb, "fold_times": fold_times}


def run_lgb_bag(params, X, y, X_test, feats, folds):
    """LightGBM seed-bagged (5 graines, un Dataset par graine, early stopping 100), comme la base."""
    n_seeds = len(SEEDS)
    oof_seeds = np.zeros((n_seeds, len(X)))  # une ligne par graine, pour mesurer le gain cumulé
    test_proba_lgb = np.zeros(len(X_test))
    importance = pd.Series(0.0, index=feats)
    best_iters = []
    fold_times = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        tf0 = time.time()
        fold_iters = []
        for k, seed in enumerate(SEEDS):
            tf = time.time()
            model = lgb.train(
                {**params, "seed": seed},
                lgb.Dataset(X.iloc[tr_idx], y[tr_idx]),
                num_boost_round=5000,
                valid_sets=[lgb.Dataset(X.iloc[va_idx], y[va_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            # chaque modèle prédit avec SON best_iteration (pas de moyenne des itérations)
            oof_seeds[k, va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
            test_proba_lgb += model.predict(X_test, num_iteration=model.best_iteration) / (n_seeds * folds)
            importance += model.feature_importance("gain") / (n_seeds * folds)
            fold_iters.append(int(model.best_iteration))
            print(f"ff {params['feature_fraction']} fold {fold} seed {seed}: {model.best_iteration} arbres, "
                  f"AUC {roc_auc_score(y[va_idx], oof_seeds[k, va_idx]):.4f} ({time.time() - tf:.0f} s)")
        best_iters.append(fold_iters)
        fold_times.append(time.time() - tf0)
        print(f"ff {params['feature_fraction']} fold {fold} moyenne 5 graines: AUC "
              f"{roc_auc_score(y[va_idx], oof_seeds[:, va_idx].mean(axis=0)):.4f} ({fold_times[-1]:.0f} s)")
    return {"oof_seeds": oof_seeds, "test_proba": test_proba_lgb, "importance": importance,
            "best_iters": best_iters, "fold_times": fold_times}


def blend_and_report(y, lgb_res, xgb_res, tag):
    """Blend LGB bag / XGB (w choisi sur le F1 OOF), gain par graine cumulée ; retourne tout."""
    oof_seeds = lgb_res["oof_seeds"]
    oof_lgb = oof_seeds.mean(axis=0)
    oof_xgb = xgb_res["oof"]
    metrics_lgb = evaluate(y, oof_lgb)
    metrics_xgb = evaluate(y, oof_xgb)
    corr = float(np.corrcoef(oof_lgb, oof_xgb)[0, 1])
    print(f"[{tag}] LGB seed-bag seul : F1 {metrics_lgb['f1']:.4f}, AUC {metrics_lgb['auc']:.4f}, "
          f"seuil {metrics_lgb['threshold']:.3f}")
    print(f"[{tag}] XGB seul          : F1 {metrics_xgb['f1']:.4f}, AUC {metrics_xgb['auc']:.4f}, "
          f"seuil {metrics_xgb['threshold']:.3f}")
    print(f"[{tag}] Corrélation OOF LGB / XGB : {corr:.4f}")

    # --- Blend : p = w*lgb + (1-w)*xgb, w choisi sur le F1 OOF, même w sur le test --------------
    blend_table = {}
    best_w, best_f1 = None, -1.0
    for w in BLEND_WEIGHTS:
        m = evaluate(y, w * oof_lgb + (1 - w) * oof_xgb)
        blend_table[str(w)] = m
        print(f"[{tag}] blend w_lgb={w}: F1 {m['f1']:.4f}, AUC {m['auc']:.4f}, seuil {m['threshold']:.3f}")
        if m["f1"] > best_f1:
            best_w, best_f1 = w, m["f1"]
    oof = best_w * oof_lgb + (1 - best_w) * oof_xgb
    test_proba = best_w * lgb_res["test_proba"] + (1 - best_w) * xgb_res["test_proba"]
    metrics = evaluate(y, oof)
    print(f"[{tag}] Blend retenu w_lgb={best_w} : F1 {metrics['f1']:.4f} "
          f"(gain vs LGB seed-bag {metrics['f1'] - metrics_lgb['f1']:+.4f}, "
          f"vs XGB seul {metrics['f1'] - metrics_xgb['f1']:+.4f})")

    # Gain exact : F1 OOF du seul seed 42 et F1 pour 1..5 graines cumulées
    f1_single_seed42 = evaluate(y, oof_seeds[0])
    f1_cumulative = {}
    for k in range(1, len(SEEDS) + 1):
        m = evaluate(y, oof_seeds[:k].mean(axis=0))
        f1_cumulative[k] = {"f1": m["f1"], "auc": m["auc"], "threshold": m["threshold"]}
        print(f"[{tag}] {k} graine(s) cumulée(s) {SEEDS[:k]}: F1 {m['f1']:.4f}, AUC {m['auc']:.4f}, "
              f"seuil {m['threshold']:.3f}")
    print(f"[{tag}] Seed 42 seul : F1 {f1_single_seed42['f1']:.4f} -> 5 graines : F1 {metrics_lgb['f1']:.4f} "
          f"(gain {metrics_lgb['f1'] - f1_single_seed42['f1']:+.4f})")
    top = lgb_res["importance"].sort_values(ascending=False).head(20)
    print(f"[{tag}] Top 20 variables LGB (gain) :")
    print(top.round(0).to_string())
    return {
        "metrics": metrics, "test_proba": test_proba, "blend_weight_lgb": best_w,
        "blend_table": blend_table, "cv_lgb_bag_only": metrics_lgb, "cv_xgb_only": metrics_xgb,
        "oof_corr_lgb_xgb": corr, "best_iterations": lgb_res["best_iters"],
        "cv_seed42_only": f1_single_seed42, "cv_cumulative_seeds": f1_cumulative,
        "top_features": top.round(0).to_dict(),
        "lgb_fold_times_s": [round(t, 1) for t in lgb_res["fold_times"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="baseline_lgbm")
    parser.add_argument("--threads", type=int, default=0, help="0 = tous les cœurs")
    parser.add_argument("--folds", type=int, default=5)
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
    X, y = train[feats], train["TARGET"].values.astype(int)
    X_test = test[feats]
    print(f"train {X.shape}, test {X_test.shape}, taux de positifs {y.mean():.3f}")

    params = {**LGB_PARAMS, "num_threads": args.threads}
    xgb_params = {**XGB_PARAMS, "nthread": args.threads}
    print(f"XGBoost {xgb.__version__}, device demandé : {xgb_params['device']}")

    # --- XGBoost GPU (inchangé) : entraîné une fois, partagé par toutes les configurations LGB ----
    xgb_res = run_xgb(xgb_params, X, y, X_test, feats, args.folds)

    # --- LightGBM seed-bagged : ff 0.3 (l'idée), puis ff 0.4 si le fold 0 est assez rapide ---------
    all_configs = {}
    results = {}
    for i, ff in enumerate(FEATURE_FRACTIONS):
        p = {**params, "feature_fraction": ff}
        if i > 0:
            fold0 = results[FEATURE_FRACTIONS[0]]["lgb_fold0_s"] + xgb_res["fold_times"][0]
            elapsed = time.time() - t0
            if fold0 >= SECOND_CONFIG_MAX_FOLD0_S:
                print(f"ff {ff} non lancé : fold 0 (LGB 5 graines + XGB) = {fold0:.0f} s "
                      f">= {SECOND_CONFIG_MAX_FOLD0_S:.0f} s (temps écoulé {elapsed:.0f} s)")
                continue
            print(f"ff {ff} lancé : fold 0 (LGB 5 graines + XGB) = {fold0:.0f} s "
                  f"< {SECOND_CONFIG_MAX_FOLD0_S:.0f} s (temps écoulé {elapsed:.0f} s)")
        tc = time.time()
        lgb_res = run_lgb_bag(p, X, y, X_test, feats, args.folds)
        rep = blend_and_report(y, lgb_res, xgb_res, f"ff {ff}")
        rep["lgb_fold0_s"] = lgb_res["fold_times"][0]
        rep["params"] = p
        rep["config_time_s"] = round(time.time() - tc, 1)
        results[ff] = rep
        all_configs[str(ff)] = {k: v for k, v in rep.items() if k not in ("test_proba",)}
        print(f"ff {ff} : F1 blend {rep['metrics']['f1']:.4f}, LGB seul {rep['cv_lgb_bag_only']['f1']:.4f}, "
              f"arbres moyens {np.mean(lgb_res['best_iters']):.0f} ({rep['config_time_s']} s)")

    # --- Sélection de la meilleure configuration sur le F1 OOF blendé ----------------------------
    best_ff = max(results, key=lambda k: results[k]["metrics"]["f1"])
    rep = results[best_ff]
    for ff in FEATURE_FRACTIONS:
        if ff in results:
            print(f"ff {ff}: F1 blend {results[ff]['metrics']['f1']:.4f} "
                  f"(AUC {results[ff]['metrics']['auc']:.4f}, w_lgb {results[ff]['blend_weight_lgb']})")
    print(f"Configuration retenue : feature_fraction {best_ff} (F1 {rep['metrics']['f1']:.4f})")
    top_xgb = xgb_res["importance"].sort_values(ascending=False).head(20)
    print("Top 20 variables XGB (gain) :")
    print(top_xgb.round(2).to_string())
    save_outputs(args.name, test, rep["test_proba"], rep["metrics"],
                 {"model": "lgbm_seed_bagging_xgb_blend", "params": rep["params"], "seeds": SEEDS,
                  "feature_fraction": best_ff, "feature_fractions_tried": sorted(results),
                  "xgb_params": xgb_params, "blend_weight_lgb": rep["blend_weight_lgb"],
                  "blend_weights": BLEND_WEIGHTS, "blend_table": rep["blend_table"],
                  "cv_lgb_bag_only": rep["cv_lgb_bag_only"], "cv_xgb_only": rep["cv_xgb_only"],
                  "oof_corr_lgb_xgb": rep["oof_corr_lgb_xgb"], "n_features": len(feats),
                  "best_iterations": rep["best_iterations"], "best_iterations_xgb": xgb_res["best_iters"],
                  "cv_seed42_only": rep["cv_seed42_only"], "cv_cumulative_seeds": rep["cv_cumulative_seeds"],
                  "top_features": rep["top_features"], "top_features_xgb": top_xgb.round(2).to_dict(),
                  "all_configs": all_configs}, t0)


if __name__ == "__main__":
    main()
