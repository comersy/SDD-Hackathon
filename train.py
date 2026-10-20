"""Baseline : tri des colonnes par règles + LightGBM + seuil optimisé pour le F1.

Usage : python train.py [--data-dir .] [--folds 5]
Sorties : submissions/submission_<horodatage>.csv et results/<horodatage>.json
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
SEED = 42

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
    "n_jobs": -1,
    "verbose": -1,
    "seed": SEED,
}


def load(data_dir):
    train = pd.concat(
        [pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "extra.csv")],
        ignore_index=True,
    )
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def select_columns(train):
    feats = [c for c in train.columns if c not in DROP_COLS and c != "TARGET"]
    constant = [c for c in feats if train[c].nunique(dropna=False) <= 1]
    feats = [c for c in feats if c not in constant]
    print(f"{len(feats)} colonnes gardées, {len(DROP_COLS)} retirées par règle, "
          f"{len(constant)} constantes : {constant}")
    return feats


def best_threshold(y, proba):
    grid = np.arange(0.05, 0.6, 0.005)
    scores = [f1_score(y, proba >= t) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    t0 = time.time()
    train, test = load(args.data_dir)
    feats = select_columns(train)
    X, y = train[feats], train["TARGET"].astype(int).values
    X_test = test[feats]
    print(f"train {X.shape}, test {X_test.shape}, taux de positifs {y.mean():.3f}")

    oof = np.zeros(len(X))
    test_proba = np.zeros(len(X_test))
    importance = pd.Series(0.0, index=feats)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        model = lgb.train(
            LGB_PARAMS,
            lgb.Dataset(X.iloc[tr_idx], y[tr_idx]),
            num_boost_round=5000,
            valid_sets=[lgb.Dataset(X.iloc[va_idx], y[va_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        test_proba += model.predict(X_test, num_iteration=model.best_iteration) / args.folds
        importance += model.feature_importance("gain") / args.folds
        print(f"fold {fold}: {model.best_iteration} arbres, "
              f"AUC {roc_auc_score(y[va_idx], oof[va_idx]):.4f}")

    threshold, f1 = best_threshold(y, oof)
    pred = oof >= threshold
    metrics = {
        "f1": f1,
        "threshold": threshold,
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "auc": float(roc_auc_score(y, oof)),
    }
    print("CV :", {k: round(v, 4) for k, v in metrics.items()})
    print("Top 20 variables (gain) :")
    print(importance.sort_values(ascending=False).head(20).round(0).to_string())

    stamp = time.strftime("%Y%m%d_%H%M%S")
    (ROOT / "submissions").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    sub_path = ROOT / "submissions" / f"submission_{stamp}.csv"
    pd.DataFrame({"ID": test["ID"], "TARGET": (test_proba >= threshold).astype(int)}).to_csv(
        sub_path, index=False)
    result = {
        "timestamp": stamp,
        "model": "lightgbm",
        "params": LGB_PARAMS,
        "n_features": len(feats),
        "cv": metrics,
        "test_positive_rate": float((test_proba >= threshold).mean()),
        "submission": sub_path.name,
        "runtime_s": round(time.time() - t0, 1),
    }
    (ROOT / "results" / f"{stamp}.json").write_text(json.dumps(result, indent=2))
    print(f"Soumission : {sub_path}  ({result['runtime_s']} s)")


if __name__ == "__main__":
    main()
