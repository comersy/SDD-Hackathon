"""Étude de sélection de colonnes : courbe F1 / AUC en fonction du nombre K de colonnes (top-K par fold).

Usage : python experiments/fs_topk_curve.py [--name NOM] [--threads N] [--folds 5] [--rows N]
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

IDEA = ("Étude de sélection de colonnes : courbe F1/AUC en fonction du nombre K de colonnes gardées. "
        "Base = x_drop_nan95 (nettoyage des codes de non-réponse, colonnes vides > 95 % retirées -> 214 "
        "candidates). Dans chaque fold : LightGBM rapide (lr 0.1, feature_fraction 0.8, 150 arbres) sur la "
        "partie entraînement -> gain par colonne ; puis, pour chaque K de K_GRID, LightGBM de la base "
        "(early stopping 100) sur les top-K colonnes du fold -> OOF et test par K. Sorties standard pour le "
        "meilleur K par F1 OOF (à moins de 0.0003 près, le plus petit K) ; détails dans "
        "results/<name>_<stamp>_study.json")
SEED = 42
NAN_MAX = 0.95  # colonnes vides à plus de 95 % : retirées
K_GRID = [15, 25, 40, 60, 80, 100, 120, 150, 214]
F1_TIE = 0.0003  # en dessous de cet écart de F1, on préfère le K le plus petit
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


def select_columns(train):
    feats = [c for c in train.columns if c not in DROP_COLS and c != "TARGET"]
    constant = [c for c in feats if train[c].nunique(dropna=False) <= 1]
    feats = [c for c in feats if c not in constant]
    # Colonnes presque vides (modules optionnels du BRFSS) : critère sans TARGET, calculé sur train
    nan_rate = train[feats].isna().mean()
    sparse = sorted(nan_rate[nan_rate > NAN_MAX].index)
    feats = [c for c in feats if c not in sparse]
    print(f"{len(feats)} colonnes gardées, {len(DROP_COLS)} retirées par règle, "
          f"{len(constant)} constantes : {constant}, "
          f"{len(sparse)} vides à plus de {NAN_MAX:.0%} : {sparse}")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="fs_topk_curve")
    parser.add_argument("--threads", type=int, default=0, help="0 = tous les cœurs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rows", type=int, default=0, help="test local : n lignes (0 = toutes)")
    args = parser.parse_args()

    t0 = time.time()
    train, test = load()
    if args.rows:
        train = train.sample(args.rows, random_state=SEED).reset_index(drop=True)
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
    fast_params = {**params, **FAST_PARAMS}
    k_grid = sorted({min(k, len(feats)) for k in K_GRID})
    oof = {k: np.zeros(len(X)) for k in k_grid}
    test_proba = {k: np.zeros(len(X_test)) for k in k_grid}
    trees = {k: [] for k in k_grid}
    fold_auc = {k: [] for k in k_grid}
    ranks = []  # par fold : colonnes triées par gain décroissant (LightGBM rapide)
    gains_fold0 = None
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t_fold = time.time()
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]
        # (1) classement des colonnes par gain, sur la partie entraînement du fold uniquement
        fast = lgb.train(fast_params, lgb.Dataset(X_tr, y_tr), num_boost_round=FAST_ROUNDS)
        gain = pd.Series(fast.feature_importance("gain"), index=feats).sort_values(ascending=False)
        rank = list(gain.index)
        ranks.append(rank)
        if fold == 0:
            gains_fold0 = gain
        n_zero = int((gain == 0).sum())
        print(f"fold {fold}: classement rapide en {time.time() - t_fold:.0f} s, "
              f"{n_zero} colonnes à gain nul, top 10 = {rank[:10]}", flush=True)
        # (2) un LightGBM de la base par K, sur les top-K colonnes du fold
        for k in k_grid:
            cols = rank[:k]
            t_k = time.time()
            model = lgb.train(
                params,
                lgb.Dataset(X_tr[cols], y_tr),
                num_boost_round=5000,
                valid_sets=[lgb.Dataset(X_va[cols], y_va)],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            oof[k][va_idx] = model.predict(X_va[cols], num_iteration=model.best_iteration)
            test_proba[k] += model.predict(X_test[cols], num_iteration=model.best_iteration) / args.folds
            trees[k].append(int(model.best_iteration))
            auc = roc_auc_score(y_va, oof[k][va_idx])
            fold_auc[k].append(float(auc))
            print(f"  fold {fold} K={k:3d}: {model.best_iteration} arbres, AUC {auc:.4f} "
                  f"({time.time() - t_k:.0f} s)", flush=True)
        print(f"fold {fold} terminé en {time.time() - t_fold:.0f} s (total {time.time() - t0:.0f} s)",
              flush=True)

    # Courbe F1 / AUC en fonction de K (seuil choisi sur l'OOF, par K)
    curve = {}
    for k in k_grid:
        m = evaluate(y, oof[k])
        curve[k] = {**m, "trees": float(np.mean(trees[k])), "fold_auc": fold_auc[k]}
    print("\nCourbe F1 / AUC en fonction du nombre de colonnes K :")
    print(f"{'K':>4} {'F1':>7} {'AUC':>7} {'prec':>6} {'rec':>6} {'seuil':>6} {'arbres':>7}")
    for k in k_grid:
        c = curve[k]
        print(f"{k:>4} {c['f1']:.4f} {c['auc']:.4f} {c['precision']:.4f} {c['recall']:.4f} "
              f"{c['threshold']:.3f} {c['trees']:>7.0f}")
    best_f1 = max(c["f1"] for c in curve.values())
    best_k = min(k for k in k_grid if curve[k]["f1"] >= best_f1 - F1_TIE)
    print(f"Meilleur K par F1 : {best_k} (F1 {curve[best_k]['f1']:.4f}, max {best_f1:.4f}, "
          f"tolérance {F1_TIE})")

    # Stabilité de la sélection entre folds
    stable = {k: [c for c in ranks[0][:k] if all(c in set(r[:k]) for r in ranks[1:])]
              for k in k_grid}
    selected_fold0 = {k: ranks[0][:k] for k in k_grid}
    print("Colonnes présentes dans le top-K de tous les folds :",
          {k: f"{len(stable[k])}/{k}" for k in k_grid})
    print(f"Top 30 colonnes du fold 0 (gain rapide) :\n{gains_fold0.head(30).round(0).to_string()}")
    print(f"Colonnes stables pour K={best_k} : {stable[best_k]}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    (ROOT / "results").mkdir(exist_ok=True)
    study = {
        "name": args.name, "timestamp": stamp, "k_grid": k_grid, "best_k": best_k,
        "curve": {str(k): curve[k] for k in k_grid},
        "selected_fold0": {str(k): selected_fold0[k] for k in k_grid},
        "stable": {str(k): stable[k] for k in k_grid},
        "rank_fold0": ranks[0],
        "gain_fold0": gains_fold0.round(1).to_dict(),
        "ranks_by_fold": ranks,
        "n_candidates": len(feats),
    }
    study_path = ROOT / "results" / f"{args.name}_{stamp}_study.json"
    study_path.write_text(json.dumps(study, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Étude : {study_path}")

    metrics = {k: curve[best_k][k] for k in ["f1", "threshold", "precision", "recall", "auc"]}
    save_outputs(args.name, test, test_proba[best_k], metrics,
                 {"model": "lightgbm", "params": params, "n_candidates": len(feats),
                  "n_features": best_k, "best_k": best_k, "k_grid": k_grid,
                  "curve": {str(k): {kk: round(v, 5) for kk, v in curve[k].items() if kk != "fold_auc"}
                            for k in k_grid},
                  "n_stable": {str(k): len(stable[k]) for k in k_grid},
                  "selected_fold0_best_k": selected_fold0[best_k],
                  "stable_best_k": stable[best_k],
                  "top_features": gains_fold0.head(30).round(0).to_dict(),
                  "study": f"results/{study_path.name}"}, t0, stamp=stamp)


if __name__ == "__main__":
    main()
