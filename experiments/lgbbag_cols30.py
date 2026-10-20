"""Liste MINIMALE de 30 colonnes (élimination récursive) évaluée avec LightGBM baggé 5 graines par fold.

Usage : python experiments/lgbbag_cols30.py [--name NOM] [--threads N] [--folds 5] [--rows N]
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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

IDEA = ("[copie de fs_final (LightGBM baggé 5 graines par fold, protocole x_lgbbag_proba inchangé) ; "
        "seule modification : select_columns() ne garde que COLUMNS] 30 colonnes RFE : liste minimale "
        "issue d'une élimination récursive (rfe_curve) à partir de la liste finale de 50 colonnes, "
        "évaluée seule (pas de variante étendue).")
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


# Liste minimale de 30 colonnes issue de l'élimination récursive (rfe_curve), embarquée en constante.
COLUMNS = [
    "_AGE80", "GENHLTH", "EMPLOY1", "LCSCTSC1", "CVDSTRK3", "SEXVAR", "_BMI5", "_PACKYRS",
    "PERSDOC3", "DIABAGE4", "WEIGHT2", "_YRSSMOK", "CHCKDNY2", "_STATE", "CHCCOPD3", "DIABETE4",
    "PHYSHLTH", "_MAM5023", "INCOME3", "SLEPTIM1", "HEIGHT3", "POORHLTH", "ALCDAY4", "HADHYST2",
    "HAVARTH4", "_RACE1", "FLSHTMY3", "_EXTETH3", "DIFFWALK", "PRIMINSR",
]
FINAL_COLUMNS = COLUMNS
SELECTION_NOTE = ("la liste a été construite par élimination récursive à partir d'importances calculées "
                  "dans chaque fold puis agrégées, donc l'évaluation finale en CV comporte un léger "
                  "optimisme de sélection (à dire dans le notebook)")


def select_columns(train, columns=FINAL_COLUMNS):
    missing = [c for c in columns if c not in train.columns]
    assert not missing, f"colonnes absentes du train : {missing}"
    assert not set(columns) & set(DROP_COLS) and "TARGET" not in columns
    assert len(set(columns)) == len(columns), "doublon dans la liste"
    print(f"{len(columns)} colonnes gardées (30 colonnes RFE) sur {train.shape[1] - 2} candidates")
    return list(columns)


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


def run_cv(X, y, X_test, params, folds, label):
    """LightGBM baggé 5 graines dans chaque fold (protocole x_lgbbag_proba inchangé)."""
    feats = list(X.columns)
    n_seeds = len(SEEDS)
    oof_seeds = np.zeros((n_seeds, len(X)))
    test_proba = np.zeros(len(X_test))
    importance = pd.Series(0.0, index=feats)
    best_iters = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
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
            oof_seeds[k, va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
            test_proba += model.predict(X_test, num_iteration=model.best_iteration) / (n_seeds * folds)
            importance += model.feature_importance("gain") / (n_seeds * folds)
            fold_iters.append(int(model.best_iteration))
            print(f"[{label}] fold {fold} seed {seed}: {model.best_iteration} arbres, "
                  f"AUC {roc_auc_score(y[va_idx], oof_seeds[k, va_idx]):.4f} ({time.time() - tf:.0f} s)")
        best_iters.append(fold_iters)
        print(f"[{label}] fold {fold} moyenne 5 graines: AUC "
              f"{roc_auc_score(y[va_idx], oof_seeds[:, va_idx].mean(axis=0)):.4f}")
    oof = oof_seeds.mean(axis=0)
    metrics = evaluate(y, oof)
    seed42 = evaluate(y, oof_seeds[0])
    print(f"[{label}] seed 42 seul : F1 {seed42['f1']:.4f} -> 5 graines : F1 {metrics['f1']:.4f} "
          f"AUC {metrics['auc']:.4f} (gain {metrics['f1'] - seed42['f1']:+.4f})")
    top = importance.sort_values(ascending=False)
    print(f"[{label}] Top 20 variables (gain) :")
    print(top.head(20).round(0).to_string())
    return oof, test_proba, metrics, seed42, best_iters, top


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="lgbbag_cols30")
    parser.add_argument("--threads", type=int, default=0, help="0 = tous les cœurs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rows", type=int, default=0, help="test local : sous-échantillon de train (0 = tout)")
    args = parser.parse_args()

    t0 = time.time()
    train, test = load()
    if args.rows:
        train = train.sample(n=args.rows, random_state=SEED).reset_index(drop=True)
    # Nettoyage des codes sur toutes les colonnes candidates (identique à la base), puis sélection
    all_feats = [c for c in train.columns if c not in DROP_COLS and c != "TARGET"]
    maxes = train[all_feats].max()
    _show(train, "AVANT nettoyage (train)")
    nan_before = train[all_feats].isna().sum().sum()
    clean(train, all_feats, maxes)
    clean(test, all_feats, maxes)
    _show(train, "APRÈS nettoyage (train)")
    print(f"NaN dans train : {nan_before} -> {train[all_feats].isna().sum().sum()}")
    feats = select_columns(train)
    y = train["TARGET"].values.astype(int)
    params = {**LGB_PARAMS, "num_threads": args.threads}

    X, X_test = train[feats], test[feats]
    print(f"train {X.shape}, test {X_test.shape}, taux de positifs {y.mean():.3f}")
    oof, test_proba, metrics, seed42, best_iters, top = run_cv(X, y, X_test, params, args.folds, "final")

    print(f"30 colonnes RFE : F1 {metrics['f1']:.4f} AUC {metrics['auc']:.4f}")

    save_outputs(args.name, test, test_proba, metrics,
                 {"model": "lightgbm_seed_bagging", "params": params, "seeds": SEEDS,
                  "n_features": len(feats), "n_columns": len(feats), "columns": feats,
                  "selection_rule": IDEA, "selection_note": SELECTION_NOTE,
                  "sources": ["results/rfe_curve_20261020_093321_study.json"],
                  "best_iterations": best_iters, "cv_seed42_only": seed42,
                  "top_features": top.round(0).to_dict(),
                  "rows": args.rows}, t0, oof=oof)


if __name__ == "__main__":
    main()
