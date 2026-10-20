"""Modèle de référence : tri des colonnes par règles + LightGBM + seuil optimisé pour le F1.

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

Expérience r1_lgbm_tuned : réglage des hyperparamètres LightGBM par recherche guidée sur le
fold 0 (critère = F1 au meilleur seuil sur le fold de validation), puis CV 5 folds complète
avec la configuration retenue (seul score reporté).
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

IDEA = ("Réglage des hyperparamètres LightGBM : recherche guidée (~12-15 essais, early stopping) "
        "sur le fold 0 avec le F1 au meilleur seuil comme critère, puis CV 5 folds avec la "
        "meilleure configuration")
SEED = 42


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

# --- Recherche d'hyperparamètres sur le fold 0 -------------------------------------------
# Point de départ = LGB_PARAMS (la référence), complété par les dimensions non réglées jusqu'ici.
SEARCH_START = {"scale_pos_weight": 1.0, "max_bin": 255, "min_split_gain": 0.0}
# Dimensions explorées dans cet ordre (descente par coordonnées) ; pour chaque dimension on
# part de la valeur courante et on avance dans une direction tant que le F1 s'améliore,
# puis dans l'autre direction (donc jamais la grille complète).
SEARCH_SPACE = [
    ("num_leaves", [31, 63, 127, 255]),
    ("min_child_samples", [50, 100, 300, 1000]),
    ("feature_fraction", [0.3, 0.5, 0.7]),
    ("lambda_l2", [1.0, 10.0, 30.0]),
    ("scale_pos_weight", [1.0, 2.0, 4.0]),
    ("max_bin", [63, 255]),
    ("bagging_fraction", [0.7, 0.8, 0.9]),
    ("min_split_gain", [0.0, 0.1]),
    ("learning_rate", [0.02, 0.03, 0.05]),
]
# Première direction essayée par dimension (+1 = valeur supérieure, -1 = inférieure)
FIRST_DIRECTION = {"num_leaves": +1, "min_child_samples": +1, "feature_fraction": -1,
                   "lambda_l2": +1, "scale_pos_weight": +1, "max_bin": -1,
                   "bagging_fraction": +1, "min_split_gain": +1, "learning_rate": -1}
MAX_TRIALS = 15
SEARCH_BUDGET_S = 13 * 60   # garde-fou : la CV 5 folds finale doit tenir dans les 25 min


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


def best_threshold(y, proba):
    # Grille élargie jusqu'à 0.9 : avec scale_pos_weight > 1 les probabilités sont décalées
    # vers le haut et le seuil optimal peut dépasser 0.6.
    grid = np.arange(0.05, 0.9, 0.005)
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


def search_fold0(X, y, skf, threads):
    """Recherche guidée sur le fold 0 ; retourne (meilleure config, liste des essais)."""
    tr_idx, va_idx = next(iter(skf.split(X, y)))
    X_tr, X_va, y_tr, y_va = X.iloc[tr_idx], X.iloc[va_idx], y[tr_idx], y[va_idx]
    datasets = {}  # un couple (train, valid) de Datasets construits par valeur de max_bin

    def get_datasets(max_bin):
        if max_bin not in datasets:
            dtr = lgb.Dataset(X_tr, y_tr, params={"max_bin": max_bin, "verbose": -1},
                              free_raw_data=False)
            dva = lgb.Dataset(X_va, y_va, reference=dtr, free_raw_data=False)
            datasets[max_bin] = (dtr, dva)
        return datasets[max_bin]

    trials, t_search = [], time.time()
    cache = {}

    def run_trial(cfg):
        key = tuple(sorted(cfg.items()))
        if key in cache:
            return cache[key]
        t1 = time.time()
        dtr, dva = get_datasets(cfg["max_bin"])
        params = {**cfg, "num_threads": threads}
        model = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        proba = model.predict(X_va, num_iteration=model.best_iteration)
        thr, f1 = best_threshold(y_va, proba)
        rec = {"trial": len(trials) + 1, "f1_fold0": f1, "threshold": thr,
               "auc_fold0": float(roc_auc_score(y_va, proba)),
               "best_iteration": int(model.best_iteration), "time_s": round(time.time() - t1, 1),
               "params": {k: v for k, v in cfg.items()
                          if k in dict(SEARCH_SPACE) or k == "learning_rate"}}
        trials.append(rec)
        cache[key] = f1
        print(f"essai {rec['trial']:2d}: F1 {f1:.4f} seuil {thr:.3f} AUC {rec['auc_fold0']:.4f} "
              f"{rec['best_iteration']} arbres {rec['time_s']}s | {rec['params']}")
        return f1

    def budget_left():
        return len(trials) < MAX_TRIALS and time.time() - t_search < SEARCH_BUDGET_S

    best_cfg = {**LGB_PARAMS, **SEARCH_START}
    best_f1 = run_trial(best_cfg)
    for dim, values in SEARCH_SPACE:
        if not budget_left():
            break
        start = values.index(best_cfg[dim])
        for direction in (FIRST_DIRECTION[dim], -FIRST_DIRECTION[dim]):
            i = start + direction
            improved_in_dir = False
            while 0 <= i < len(values) and budget_left():
                cfg = {**best_cfg, dim: values[i]}
                f1 = run_trial(cfg)
                if f1 > best_f1:
                    best_f1, best_cfg, improved_in_dir = f1, cfg, True
                    i += direction
                else:
                    break
            if improved_in_dir:
                break  # on ne teste l'autre direction que si la première n'a rien donné
    print(f"Recherche : {len(trials)} essais en {time.time() - t_search:.0f}s, "
          f"meilleur F1 fold 0 = {best_f1:.4f}")
    del datasets, X_tr, X_va
    return best_cfg, best_f1, trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="r1_lgbm_tuned")
    parser.add_argument("--threads", type=int, default=0, help="0 = tous les cœurs")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    t0 = time.time()
    train, test = load()
    feats = select_columns(train)
    X, y = train[feats], train["TARGET"].values.astype(int)
    X_test = test[feats]
    del train
    print(f"train {X.shape}, test {X_test.shape}, taux de positifs {y.mean():.3f}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    best_cfg, best_f1_fold0, trials = search_fold0(X, y, skf, args.threads)
    print("Configuration retenue :", best_cfg)

    params = {**best_cfg, "num_threads": args.threads}
    oof = np.zeros(len(X))
    test_proba = np.zeros(len(X_test))
    importance = pd.Series(0.0, index=feats)
    n_trees = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        model = lgb.train(
            params,
            lgb.Dataset(X.iloc[tr_idx], y[tr_idx]),
            num_boost_round=5000,
            valid_sets=[lgb.Dataset(X.iloc[va_idx], y[va_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        test_proba += model.predict(X_test, num_iteration=model.best_iteration) / args.folds
        importance += model.feature_importance("gain") / args.folds
        n_trees.append(int(model.best_iteration))
        print(f"fold {fold}: {model.best_iteration} arbres, "
              f"AUC {roc_auc_score(y[va_idx], oof[va_idx]):.4f}")

    metrics = evaluate(y, oof)
    top = importance.sort_values(ascending=False).head(20)
    print("Top 20 variables (gain) :")
    print(top.round(0).to_string())
    save_outputs(args.name, test, test_proba, metrics,
                 {"model": "lightgbm", "params": params, "n_features": len(feats),
                  "n_trees_per_fold": n_trees,
                  "search": {"fold": 0, "criterion": "F1 au meilleur seuil sur le fold de validation",
                             "n_trials": len(trials), "best_f1_fold0": best_f1_fold0,
                             "best_config": best_cfg, "trials": trials},
                  "top_features": top.round(0).to_dict()}, t0)


if __name__ == "__main__":
    main()
