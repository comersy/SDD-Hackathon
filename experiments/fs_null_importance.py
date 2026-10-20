"""Sélection de colonnes par « null importance » : on garde les colonnes dont l'importance
(gain et nombre de splits) dépasse ce qu'obtient un LightGBM entraîné sur une cible permutée.

Usage : python experiments/fs_null_importance.py [--name NOM] [--threads N] [--folds 5] [--perms 20]
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

IDEA = ("Sélection de colonnes par null importance, faite dans chaque fold sur la partie entraînement : "
        "LightGBM rapide (lr 0.1, feature_fraction 0.8, 200 arbres) -> gain et splits réels par colonne ; "
        "20 répétitions avec TARGET permuté -> distributions nulles ; on garde les colonnes dont "
        "gain réel > p95 du gain nul ET splits réels > p95 des splits nuls ; puis LightGBM de la base "
        "(early stopping) sur les colonnes gardées du fold -> OOF et test. Consensus = colonnes gardées "
        "dans >= 4 folds sur 5. Base : x_drop_nan95 (nettoyage des codes, 214 colonnes candidates)")
SEED = 42
NAN_MAX = 0.95  # colonnes vides à plus de 95 % : retirées
N_PERM = 20  # répétitions avec cible permutée (réduit à 12 si le temps prévu dépasse 25 min)
TIME_BUDGET_S = 25 * 60
FAST_PARAMS = {"learning_rate": 0.1, "feature_fraction": 0.8}
FAST_ROUNDS = 200
PCT = 95


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


def fast_importance(params, ds):
    """Gain et nombre de splits par colonne d'un LightGBM rapide (200 arbres, pas d'early stopping)."""
    model = lgb.train(params, ds, num_boost_round=FAST_ROUNDS)
    return model.feature_importance("gain"), model.feature_importance("split").astype(float)


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
    parser.add_argument("--name", default="fs_null_importance")
    parser.add_argument("--threads", type=int, default=0, help="0 = tous les cœurs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--perms", type=int, default=N_PERM)
    parser.add_argument("--rows", type=int, default=0, help="test local : sous-échantillon de train")
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
    n_perm = args.perms
    oof = np.zeros(len(X))
    test_proba = np.zeros(len(X_test))
    importance = pd.Series(0.0, index=feats)
    kept_count = pd.Series(0, index=feats)
    per_fold, scores_fold0 = [], None
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t_fold = time.time()
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        # --- 1) importance réelle avec un LightGBM rapide (sur la partie entraînement seulement) ---
        ds = lgb.Dataset(X_tr, y_tr, free_raw_data=False).construct()  # binning fait une seule fois
        gain_real, splits_real = fast_importance(fast_params, ds)
        t_one = time.time() - t_fold
        if fold == 0:
            projected = args.folds * ((n_perm + 1) * t_one + 60)
            print(f"fit rapide : {t_one:.1f} s -> temps total prévu {projected / 60:.1f} min "
                  f"avec {n_perm} permutations")
            if projected > TIME_BUDGET_S and n_perm > 12:
                n_perm = 12
                print(f"-> réduit à {n_perm} permutations")
        # --- 2) distributions nulles : même modèle sur TARGET permuté -------------------------
        gain_null = np.zeros((n_perm, len(feats)))
        splits_null = np.zeros((n_perm, len(feats)))
        for i in range(n_perm):
            rng = np.random.default_rng(1000 * fold + i)
            ds.set_label(rng.permutation(y_tr))
            gain_null[i], splits_null[i] = fast_importance(fast_params, ds)
        # --- 3) critère : gain ET splits réels au-dessus du 95e percentile nul ------------------
        gain_p95 = np.percentile(gain_null, PCT, axis=0)
        splits_p95 = np.percentile(splits_null, PCT, axis=0)
        gain_p75 = np.percentile(gain_null, 75, axis=0)
        keep_mask = (gain_real > gain_p95) & (splits_real > splits_p95)
        score = np.log((gain_real + 1) / (gain_p75 + 1))
        kept = [c for c, k in zip(feats, keep_mask) if k]
        if len(kept) < 2:  # garde-fou, jamais atteint en pratique
            kept = [c for c, _ in sorted(zip(feats, score), key=lambda t: -t[1])[:20]]
        kept_count[kept] += 1
        per_fold.append({"n_kept": len(kept), "kept": kept,
                         "n_gain_only": int(((gain_real > gain_p95) & ~keep_mask).sum()),
                         "n_splits_only": int(((splits_real > splits_p95) & ~keep_mask).sum())})
        if fold == 0:
            scores_fold0 = {c: {"gain_real": float(gain_real[j]), "gain_null_p95": float(gain_p95[j]),
                                "splits_real": float(splits_real[j]),
                                "splits_null_p95": float(splits_p95[j]),
                                "score": float(score[j]), "kept": bool(keep_mask[j])}
                            for j, c in enumerate(feats)}
        print(f"fold {fold}: {len(kept)} colonnes gardées / {len(feats)} "
              f"(sélection {time.time() - t_fold:.0f} s, {n_perm} permutations)")
        del ds
        # --- 4) LightGBM de la base (early stopping) sur les colonnes gardées du fold ----------
        model = lgb.train(
            params,
            lgb.Dataset(X_tr[kept], y_tr),
            num_boost_round=5000,
            valid_sets=[lgb.Dataset(X.iloc[va_idx][kept], y[va_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[va_idx] = model.predict(X.iloc[va_idx][kept], num_iteration=model.best_iteration)
        test_proba += model.predict(X_test[kept], num_iteration=model.best_iteration) / args.folds
        importance[kept] += model.feature_importance("gain") / args.folds
        print(f"fold {fold}: {model.best_iteration} arbres, "
              f"AUC {roc_auc_score(y[va_idx], oof[va_idx]):.4f} ({time.time() - t_fold:.0f} s)")

    metrics = evaluate(y, oof)
    min_folds = max(1, args.folds - 1)
    consensus = [c for c in feats if kept_count[c] >= min_folds]
    rejected = [c for c in feats if (args.folds - kept_count[c]) >= min_folds]
    top = importance.sort_values(ascending=False).head(20)
    n_kept = [f["n_kept"] for f in per_fold]
    print(f"Colonnes gardées par fold : {n_kept} ; consensus (>= {min_folds} folds) : "
          f"{len(consensus)} ; rejetées dans >= {min_folds} folds : {len(rejected)}")
    print("Top 20 variables (gain) :")
    print(top.round(0).to_string())
    study = {"per_fold": per_fold, "consensus": consensus, "n_consensus": len(consensus),
             "scores_fold0": scores_fold0, "rejected_consensus": rejected,
             "n_permutations": n_perm, "kept_count": kept_count.astype(int).to_dict()}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (ROOT / "results").mkdir(exist_ok=True)
    study_path = ROOT / "results" / f"{args.name}_{stamp}_study.json"
    study_path.write_text(json.dumps(study, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Étude : {study_path}")
    save_outputs(args.name, test, test_proba, metrics,
                 {"model": "lightgbm", "params": params, "n_candidates": len(feats),
                  "n_features": int(round(np.mean(n_kept))), "n_kept_per_fold": n_kept,
                  "n_consensus": len(consensus), "consensus": consensus,
                  "n_rejected_consensus": len(rejected), "n_permutations": n_perm,
                  "fast_params": {**FAST_PARAMS, "rounds": FAST_ROUNDS}, "percentile": PCT,
                  "study": f"results/{study_path.name}",
                  "top_features": top.round(0).to_dict()}, t0)


if __name__ == "__main__":
    main()
