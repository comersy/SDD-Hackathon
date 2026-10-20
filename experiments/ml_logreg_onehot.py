"""Régression logistique one-hot (odds ratios interprétables) : sélection top-120 par fold + sklearn.

Usage (Modal, CPU) :
    python -W ignore -m modal run modal_run.py --script experiments/ml_logreg_onehot.py --name ml_logreg_onehot --threads 16

Protocole identique à train.py / dl_base.py (même CV 5 folds seed 42, même seuil choisi sur l'OOF).
Dans chaque fold :
1. sélection : LightGBM rapide ajusté sur la partie entraînement du fold -> top-K colonnes par gain
   (K = CONFIG["top_k"]) ; aucune fuite ;
2. prétraitement ajusté sur la partie entraînement du fold (sans TARGET) : colonnes à <= 15 modalités
   -> codes entiers (0 = manquant / inconnu) puis OneHotEncoder(handle_unknown="ignore") creux, la
   modalité 0 (manquant) devient une colonne one-hot comme les autres ; autres colonnes ->
   QuantileTransformer (loi normale), NaN -> 0 + indicateur de valeur manquante ; scipy.sparse.hstack ;
3. LogisticRegression(C, solver="lbfgs", max_iter=500). Sur le fold 0 seulement, C est comparé dans
   CONFIG["c_grid"] (AUC de validation) et la meilleure C est gardée pour les 5 folds.
Le JSON contient en plus les 25 coefficients les plus forts du fold 0 (colonne=modalité, odds ratio).
En plus du JSON et de la soumission, les probabilités OOF et test sont écrites dans results/
(<name>_<stamp>_oof.csv et <name>_<stamp>_test.csv) pour les blends ultérieurs.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_clean_codes import LGB_PARAMS, ROOT, SEED, clean, evaluate, load, select_columns  # noqa: E402

IDEA = ("Régression logistique one-hot : colonnes vides à > 95 % retirées, top-120 colonnes par fold "
        "(gain LightGBM), one-hot creux des catégorielles (<= 15 modalités, 0 = manquant comme modalité) "
        "+ quantiles normaux et indicateurs de manquant des numériques ; LogisticRegression lbfgs "
        "max_iter=500, C choisie sur le fold 0 parmi {0.1, 0.3, 1} ; 25 odds ratios les plus forts sauvegardés")
NAN_MAX = 0.95          # colonnes vides à plus de 95 % : retirées d'office (critère sans TARGET)
CAT_MAX_LEVELS = 15     # <= 15 modalités distinctes -> catégorielle (one-hot)
CONFIG = {
    "top_k": 120,           # colonnes gardées par fold
    "C": 0.3,               # valeur par défaut, remplacée par la meilleure du fold 0
    "c_grid": [0.1, 0.3, 1.0],
    "solver": "lbfgs",
    "max_iter": 500,
    "n_top_coefs": 25,
}


# ------------------------------------------------------------------ sélection et prétraitement
def drop_sparse(train, feats):
    nan_rate = train[feats].isna().mean()
    sparse = sorted(nan_rate[nan_rate > NAN_MAX].index)
    print(f"{len(sparse)} colonnes vides à plus de {NAN_MAX:.0%} retirées, {len(feats) - len(sparse)} candidates")
    return [c for c in feats if c not in sparse]


def select_top_k(df_tr, y_tr, feats, k, threads):
    """Top-K colonnes par gain d'un LightGBM rapide ajusté sur la partie entraînement du fold."""
    params = {**LGB_PARAMS, "learning_rate": 0.1, "feature_fraction": 0.8, "num_threads": threads}
    model = lgb.train(params, lgb.Dataset(df_tr[feats], y_tr), num_boost_round=150)
    gain = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    return list(gain.index[:k]), gain


class FoldPreprocessor:
    """Ajusté sur la partie entraînement du fold uniquement ; n'utilise jamais TARGET.
    Sortie : matrice creuse [one-hot des codes catégoriels | quantiles | indicateurs de manquant]."""

    def __init__(self, cats, nums):
        self.cats, self.nums = cats, nums

    def _codes(self, df):
        if not self.cats:
            return np.zeros((len(df), 0), dtype=np.int64)
        return np.stack([df[c].astype("float64").map(self.maps[c]).fillna(0).to_numpy(dtype=np.int64)
                         for c in self.cats], axis=1)

    def fit(self, df):
        self.maps = {c: {v: i + 1 for i, v in enumerate(sorted(df[c].dropna().unique().tolist()))}
                     for c in self.cats}
        self.cardinalities = [len(self.maps[c]) + 1 for c in self.cats]  # 0 = manquant / inconnu
        self.ohe = (OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32).fit(self._codes(df))
                    if self.cats else None)
        self.qt = (QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                       subsample=200_000, random_state=SEED).fit(df[self.nums])
                   if self.nums else None)
        # Noms des colonnes de la matrice finale (pour les coefficients)
        names = []
        if self.cats:
            inv = {c: {i: v for v, i in self.maps[c].items()} for c in self.cats}
            for c, levels in zip(self.cats, self.ohe.categories_):
                for code in levels:
                    label = "manquant" if int(code) == 0 else f"{inv[c][int(code)]:g}"
                    names.append(f"{c}={label}")
        names += [f"{c}(quantile)" for c in self.nums] + [f"{c}=manquant" for c in self.nums]
        self.feature_names = names
        return self

    def transform(self, df):
        n = len(df)
        parts = []
        if self.cats:
            parts.append(self.ohe.transform(self._codes(df)))
        if self.nums:
            num = self.qt.transform(df[self.nums]).astype(np.float32)
            missing = np.isnan(num)
            num[missing] = 0.0
            parts.append(sp.csr_matrix(np.concatenate([num, missing.astype(np.float32)], axis=1)))
        if not parts:
            return sp.csr_matrix((n, 0), dtype=np.float32)
        return sp.hstack(parts, format="csr", dtype=np.float32)


# ------------------------------------------------------------------------------------ modèle
def build_model(cfg, C):
    return LogisticRegression(C=C, solver=cfg["solver"], max_iter=cfg["max_iter"],
                              n_jobs=16, random_state=SEED)


def fit_predict(cfg, C, X_tr, y_tr, X_va):
    model = build_model(cfg, C).fit(X_tr, y_tr)
    return model, model.predict_proba(X_va)[:, 1]


def top_coefficients(model, names, n):
    coef = model.coef_.ravel()
    order = np.argsort(-np.abs(coef))[:n]
    return [{"feature": names[i], "coef": round(float(coef[i]), 4),
             "odds_ratio": round(float(np.exp(coef[i])), 4)} for i in order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ml_logreg_onehot")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=None, help="remplace CONFIG['top_k']")
    parser.add_argument("--nrows", type=int, default=None, help="test local : sous-échantillon")
    args = parser.parse_args()
    cfg = dict(CONFIG)
    if args.top_k:
        cfg["top_k"] = args.top_k
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(max(1, args.threads)))
    t0 = time.time()

    train, test = load()
    if args.nrows:
        train = train.sample(args.nrows, random_state=SEED).reset_index(drop=True)
        test = test.head(args.nrows).reset_index(drop=True)
    feats = drop_sparse(train, select_columns(train))
    maxes = train[feats].max()
    clean(train, feats, maxes)
    clean(test, feats, maxes)
    y = train["TARGET"].values.astype(int)
    nunique = train[feats].nunique()
    cats_all = {c for c in feats if nunique[c] <= CAT_MAX_LEVELS}
    print(f"{len(feats)} colonnes candidates dont {len(cats_all)} catégorielles ; top-{cfg['top_k']} par fold")

    oof = np.zeros(len(train))
    test_proba = np.zeros(len(test))
    folds_info = []
    c_search = None
    top_coefs = None
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        tf = time.time()
        df_tr, df_va = train.iloc[tr_idx], train.iloc[va_idx]
        selected, gain = select_top_k(df_tr, y[tr_idx], feats, cfg["top_k"], args.threads)
        cats = [c for c in selected if c in cats_all]
        nums = [c for c in selected if c not in cats_all]
        pre = FoldPreprocessor(cats, nums).fit(df_tr)
        X_tr, X_va, X_te = pre.transform(df_tr), pre.transform(df_va), pre.transform(test)
        if fold == 0:
            print(f"matrice creuse fold 0 : {X_tr.shape}, {len(pre.feature_names)} colonnes nommées")
            # Comparaison rapide de C sur le fold 0 seulement (AUC de validation), puis C fixée
            c_search = []
            best_C, best_auc, best_model, best_p = None, -1.0, None, None
            for C in cfg["c_grid"]:
                tc = time.time()
                model, p = fit_predict(cfg, C, X_tr, y[tr_idx], X_va)
                auc = roc_auc_score(y[va_idx], p)
                ll = log_loss(y[va_idx], np.clip(p, 1e-7, 1 - 1e-7))
                c_search.append({"C": C, "auc": round(float(auc), 5), "logloss": round(float(ll), 5),
                                 "n_iter": int(model.n_iter_[0]), "s": round(time.time() - tc, 1)})
                print(f"  fold 0 C={C}: AUC {auc:.5f} logloss {ll:.5f} ({model.n_iter_[0]} itérations, "
                      f"{time.time() - tc:.0f} s)", flush=True)
                if auc > best_auc:
                    best_C, best_auc, best_model, best_p = C, auc, model, p
            cfg["C"] = best_C
            print(f"C retenue pour les 5 folds : {best_C}", flush=True)
            model, p = best_model, best_p
            top_coefs = top_coefficients(model, pre.feature_names, cfg["n_top_coefs"])
            print("Top coefficients fold 0 (odds ratio) :")
            for row in top_coefs:
                print(f"  {row['feature']:<28} coef {row['coef']:+.3f}  OR {row['odds_ratio']:.3f}")
        else:
            model, p = fit_predict(cfg, cfg["C"], X_tr, y[tr_idx], X_va)
        oof[va_idx] = p
        test_proba += model.predict_proba(X_te)[:, 1] / args.folds
        auc = roc_auc_score(y[va_idx], oof[va_idx])
        print(f"fold {fold}: AUC {auc:.4f}, {len(cats)} cat + {len(nums)} num -> {X_tr.shape[1]} colonnes, "
              f"{int(model.n_iter_[0])} itérations, {time.time() - tf:.0f} s", flush=True)
        folds_info.append({"selected": selected, "n_cat": len(cats), "n_num": len(nums),
                           "n_features_sparse": int(X_tr.shape[1]), "n_iter": int(model.n_iter_[0]),
                           "auc": round(float(auc), 5), "s": round(time.time() - tf, 1),
                           "top10_gain": {k: round(float(v)) for k, v in gain.head(10).items()}})
        del X_tr, X_va, X_te, model

    metrics = evaluate(y, oof)
    save_outputs(args.name, test, oof, test_proba, metrics,
                 {"model": "logreg_onehot", "config": cfg, "c_search_fold0": c_search,
                  "top_coefficients_fold0": top_coefs, "nan_max": NAN_MAX,
                  "cat_max_levels": CAT_MAX_LEVELS, "folds": folds_info}, t0)


def save_outputs(name, test, oof, test_proba, metrics, info, t0):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (ROOT / "submissions").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    sub_path = ROOT / "submissions" / f"submission_{name}_{stamp}.csv"
    pred = (test_proba >= metrics["threshold"]).astype(int)
    pd.DataFrame({"ID": test["ID"], "TARGET": pred}).to_csv(sub_path, index=False)
    # Probabilités pour les blends (ordre OOF = train.csv puis extra.csv)
    pd.DataFrame({"oof": oof}).to_csv(ROOT / "results" / f"{name}_{stamp}_oof.csv", index=False, float_format="%.6f")
    pd.DataFrame({"ID": test["ID"], "proba": test_proba}).to_csv(
        ROOT / "results" / f"{name}_{stamp}_test.csv", index=False, float_format="%.6f")
    result = {
        "name": name, "idea": IDEA, "timestamp": stamp, "cv": metrics,
        "test_positive_rate": float(pred.mean()), "submission": f"submissions/{sub_path.name}",
        "runtime_s": round(time.time() - t0, 1), **info,
    }
    res_path = ROOT / "results" / f"{name}_{stamp}.json"
    res_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("CV :", {k: round(v, 4) for k, v in metrics.items()})
    print(f"Soumission : {sub_path}\nRésultats : {res_path}  ({result['runtime_s']} s)")
    return res_path


if __name__ == "__main__":
    main()
