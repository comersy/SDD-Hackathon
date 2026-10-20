"""Étude de sélection de colonnes : élimination récursive (RFE) dans chaque fold, courbe F1 / AUC.

Usage : python experiments/rfe_curve.py [--name NOM] [--threads N] [--folds 5] [--rows N]
Sorties : submissions/submission_<nom>_<horodatage>.csv et results/<nom>_<horodatage>.json
          (palier n = REPORT_N) + results/<nom>_<horodatage>_study.json (courbe, listes par fold, consensus)

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

IDEA = ("Élimination récursive de colonnes (RFE) dans chaque fold, à partir des 62 colonnes de "
        "features/selected_columns.json (50 + variante étendue). Base = x_drop_nan95 (nettoyage des codes "
        "de non-réponse). Dans chaque fold, sur la partie entraînement uniquement : LightGBM rapide "
        "(lr 0.1, feature_fraction 0.8, 150 arbres, sans early stopping) -> on retire les colonnes de plus "
        "faible gain par paquets 62->55->50->45->40->35->30->25->20->15->10 ; à chaque palier n de "
        "50 à 10, LightGBM de la base (early stopping 100 sur le fold de validation) sur les n colonnes du "
        "fold -> OOF et test par palier. Courbe F1/AUC par palier, liste consensus par palier (colonnes "
        "présentes dans >= 3 folds, classées par nombre de folds puis gain moyen, ajustée à n). "
        "Sorties standard pour le palier n = 30 ; détails dans results/<name>_<stamp>_study.json")
SEED = 42
# Les 62 colonnes de features/selected_columns.json (50 "columns" + 12 "extended_variant")
START_COLUMNS = [
    "SEXVAR", "GENHLTH", "PERSDOC3", "MEDCOST1", "CVDSTRK3", "CHCSCNC1", "CHCOCNC1", "CHCCOPD3",
    "ADDEPEV3", "CHCKDNY2", "HAVARTH4", "DIABETE4", "DIABAGE4", "NUMHHOL4", "DEAF", "BLIND",
    "DIFFWALK", "DIFFDRES", "HADHYST2", "_CRCREC2", "LCSCTSC1", "LCSSCNCR", "CNCRTYP2", "_RACE1",
    "_ALTETH3", "_AGE80", "_PACKYRS", "EMPLOY1", "_BMI5", "_MAM5023", "WEIGHT2", "PHYSHLTH",
    "_YRSSMOK", "_STATE", "SLEPTIM1", "INCOME3", "POORHLTH", "CHECKUP1", "LASTDEN4", "PRIMINSR",
    "HEIGHT3", "FLSHTMY3", "MENTHLTH", "MARITAL", "HIVTSTD3", "PNEUVAC4", "ALCDAY4", "TETANUS1",
    "COVIDFS1", "HHADULT",
    # variante étendue (+12)
    "AVEDRNK3", "CRVCLCNC", "EMTSUPRT", "MAXDRNKS", "EDUCA", "COVIDSE1", "SDHSTRE1", "_MRACE2",
    "DRNK3GE5", "CNCRAGE", "RRCOGNT2", "_EXTETH3",
]
LEVELS = [55, 50, 45, 40, 35, 30, 25, 20, 15, 10]  # paliers d'élimination
EVAL_LEVELS = [50, 45, 40, 35, 30, 25, 20, 15, 10]  # paliers évalués avec le LightGBM de la base
REPORT_N = 30  # palier des sorties standard (soumission + results JSON)
MIN_FOLDS = 3  # consensus : colonne gardée si présente dans >= MIN_FOLDS folds
FAST_PARAMS = {"learning_rate": 0.1, "feature_fraction": 0.8}  # LightGBM de classement, sans early stopping
FAST_ROUNDS = 150

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



def save_outputs(name, test, test_proba, metrics, info, t0, stamp=None):
    """Écrit la soumission et le JSON de résultats ; retourne le chemin du JSON."""
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
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


def consensus(fold_sets, fold_gains, n):
    """Colonnes présentes dans >= MIN_FOLDS folds au palier n, classées par (nb de folds, gain moyen),
    tronquées ou complétées (par les suivantes du classement) à exactement n."""
    union = sorted({c for s in fold_sets for c in s})
    rows = []
    for c in union:
        folds = [i for i, s in enumerate(fold_sets) if c in s]
        gains = [fold_gains[i][c] for i in folds]
        rows.append((c, len(folds), float(np.mean(gains))))
    rows.sort(key=lambda r: (-r[1], -r[2], r[0]))
    kept = [r for r in rows if r[1] >= MIN_FOLDS]
    rest = [r for r in rows if r[1] < MIN_FOLDS]
    ranked = (kept + rest)[:n]
    return [r[0] for r in ranked], {r[0]: {"folds": r[1], "gain_mean": round(r[2], 1)} for r in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="rfe_curve")
    parser.add_argument("--threads", type=int, default=0, help="0 = tous les cœurs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rows", type=int, default=0, help="test local : n lignes (0 = toutes)")
    args = parser.parse_args()

    t0 = time.time()
    train, test = load()
    if args.rows:
        train = train.sample(args.rows, random_state=SEED).reset_index(drop=True)
    feats = list(START_COLUMNS)
    assert len(feats) == len(set(feats)) == 62, len(feats)
    missing = [c for c in feats if c not in train.columns or c not in test.columns]
    assert not missing, f"colonnes absentes : {missing}"
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
    fast_params = {**params, **FAST_PARAMS}
    oof = {n: np.zeros(len(X)) for n in EVAL_LEVELS}
    test_proba = {n: np.zeros(len(X_test)) for n in EVAL_LEVELS}
    trees = {n: [] for n in EVAL_LEVELS}
    fold_auc = {n: [] for n in EVAL_LEVELS}
    per_fold = {n: [] for n in EVAL_LEVELS}       # n -> [cols du fold 0, cols du fold 1, ...]
    per_fold_gain = {n: [] for n in EVAL_LEVELS}  # n -> [{col: gain du LightGBM rapide sur ces n cols}, ...]
    elimination_order = []                        # par fold : colonnes dans l'ordre d'élimination
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t_fold = time.time()
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]
        cols = list(feats)
        order = []
        for n_next in [*LEVELS, None]:
            # (1) LightGBM rapide sur les colonnes courantes, partie entraînement uniquement
            t_s = time.time()
            fast = lgb.train(fast_params, lgb.Dataset(X_tr[cols], y_tr), num_boost_round=FAST_ROUNDS)
            gain = pd.Series(fast.feature_importance("gain"), index=cols).sort_values(ascending=False)
            n = len(cols)
            if n in EVAL_LEVELS:
                per_fold[n].append(list(cols))
                per_fold_gain[n].append(gain.round(1).to_dict())
                # (2) LightGBM de la base sur les n colonnes du fold, early stopping sur la validation
                model = lgb.train(
                    params,
                    lgb.Dataset(X_tr[cols], y_tr),
                    num_boost_round=5000,
                    valid_sets=[lgb.Dataset(X_va[cols], y_va)],
                    callbacks=[lgb.early_stopping(100, verbose=False)],
                )
                oof[n][va_idx] = model.predict(X_va[cols], num_iteration=model.best_iteration)
                test_proba[n] += model.predict(X_test[cols], num_iteration=model.best_iteration) / args.folds
                trees[n].append(int(model.best_iteration))
                auc = roc_auc_score(y_va, oof[n][va_idx])
                fold_auc[n].append(float(auc))
                print(f"  fold {fold} n={n:3d}: {model.best_iteration} arbres, AUC {auc:.4f} "
                      f"({time.time() - t_s:.0f} s)", flush=True)
            if n_next is None:
                # dernier palier : on complète l'ordre d'élimination par les colonnes restantes (gain croissant)
                order.extend(list(gain.index[::-1]))
                break
            # (3) retrait des colonnes de plus faible gain pour atteindre le palier suivant
            dropped = list(gain.index[::-1][: n - n_next])
            order.extend(dropped)
            cols = [c for c in cols if c not in set(dropped)]
            print(f"  fold {fold} {n}->{n_next}: retirées {dropped}", flush=True)
        elimination_order.append(order)
        print(f"fold {fold} terminé en {time.time() - t_fold:.0f} s (total {time.time() - t0:.0f} s), "
              f"10 dernières colonnes : {cols}", flush=True)

    # Courbe F1 / AUC par palier (seuil choisi sur l'OOF, par palier)
    curve = {}
    for n in EVAL_LEVELS:
        m = evaluate(y, oof[n])
        curve[n] = {**m, "trees": float(np.mean(trees[n])), "fold_auc": fold_auc[n]}
    print("\nCourbe F1 / AUC en fonction du nombre de colonnes n :")
    print(f"{'n':>4} {'F1':>7} {'AUC':>7} {'prec':>6} {'rec':>6} {'seuil':>6} {'arbres':>7}")
    for n in EVAL_LEVELS:
        c = curve[n]
        print(f"{n:>4} {c['f1']:.4f} {c['auc']:.4f} {c['precision']:.4f} {c['recall']:.4f} "
              f"{c['threshold']:.3f} {c['trees']:>7.0f}")

    # Consensus par palier
    cons, cons_detail = {}, {}
    for n in EVAL_LEVELS:
        cons[n], cons_detail[n] = consensus(per_fold[n], per_fold_gain[n], n)
        n_all = sum(1 for c in cons[n] if cons_detail[n][c]["folds"] == args.folds)
        print(f"consensus n={n}: {n_all}/{n} colonnes communes aux {args.folds} folds : {cons[n]}")
    always = [c for c in feats if all(c in s for s in per_fold[10])]
    print(f"Colonnes gardées jusqu'à n=10 dans tous les folds : {always}")
    print(f"Ordre d'élimination (fold 0) : {elimination_order[0]}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    (ROOT / "results").mkdir(exist_ok=True)
    study = {
        "name": args.name, "timestamp": stamp, "start_columns": feats, "levels": EVAL_LEVELS,
        "report_n": REPORT_N, "min_folds": MIN_FOLDS,
        "curve": [{"n": n, "f1": curve[n]["f1"], "auc": curve[n]["auc"], "precision": curve[n]["precision"],
                   "recall": curve[n]["recall"], "threshold": curve[n]["threshold"],
                   "trees": curve[n]["trees"], "fold_auc": curve[n]["fold_auc"]} for n in EVAL_LEVELS],
        "per_fold": {str(n): per_fold[n] for n in EVAL_LEVELS},
        "per_fold_gain": {str(n): per_fold_gain[n] for n in EVAL_LEVELS},
        "consensus": {str(n): cons[n] for n in EVAL_LEVELS},
        "consensus_detail": {str(n): cons_detail[n] for n in EVAL_LEVELS},
        "elimination_order_fold0": elimination_order[0],
        "elimination_order_by_fold": elimination_order,
        "always_kept_at_10": always,
    }
    study_path = ROOT / "results" / f"{args.name}_{stamp}_study.json"
    study_path.write_text(json.dumps(study, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Étude : {study_path}")

    metrics = {k: curve[REPORT_N][k] for k in ["f1", "threshold", "precision", "recall", "auc"]}
    save_outputs(args.name, test, test_proba[REPORT_N], metrics,
                 {"model": "lightgbm", "params": params, "n_candidates": len(feats),
                  "n_features": REPORT_N, "report_n": REPORT_N, "levels": EVAL_LEVELS,
                  "curve": {str(n): {kk: round(v, 5) for kk, v in curve[n].items() if kk != "fold_auc"}
                            for n in EVAL_LEVELS},
                  "consensus_report_n": cons[REPORT_N],
                  "consensus": {str(n): cons[n] for n in EVAL_LEVELS},
                  "always_kept_at_10": always,
                  "study": f"results/{study_path.name}"}, t0, stamp=stamp)


if __name__ == "__main__":
    main()
