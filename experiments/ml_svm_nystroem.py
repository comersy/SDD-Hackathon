"""SVM à noyau RBF approché (Nystroem) + SVM linéaire calibré, top-80 colonnes par fold.

Usage (Modal, CPU 16 cœurs) :
    python -W ignore -m modal run modal_run.py --script experiments/ml_svm_nystroem.py --name ml_svm_nystroem --threads 16

Protocole identique à train.py / dl_base.py (même CV 5 folds seed 42, même seuil choisi sur l'OOF).
Dans chaque fold :
1. sélection : LightGBM rapide ajusté sur la partie entraînement du fold -> top-K colonnes par gain ;
2. prétraitement ajusté sur la partie entraînement du fold (sans TARGET) : colonnes à <= 15 modalités
   -> codes entiers puis one-hot (OneHotEncoder, modalité inconnue -> zéros) ; autres colonnes ->
   QuantileTransformer (loi normale), NaN -> 0 + indicateur de valeur manquante ; matrice dense float32 ;
   si plus de 600 colonnes, le one-hot est limité aux modalités de fréquence >= 1 % ;
3. Nystroem(rbf, gamma = 1 / n_features, n_components) ajusté sur 100 000 lignes de la partie
   entraînement du fold, puis LinearSVC(C=0.5, dual=False) dans CalibratedClassifierCV(cv=3, sigmoid)
   ajusté sur la partie entraînement du fold seulement (la calibration interne découpe cette partie
   en 3 ; le fold de validation n'est jamais vu). OOF = predict_proba[:, 1] ; test = moyenne des folds.
Pourquoi Nystroem : un SVM à noyau exact est quadratique en nombre de lignes (matrice de Gram
370 000 x 370 000 ~ 1 To, temps O(n^2) à O(n^3)) et donc impossible ici ; l'approximation de Nystroem
projette les données dans un espace de dimension n_components où le produit scalaire approche le noyau
RBF, ce qui ramène le problème à un SVM linéaire (liblinear, linéaire en n) : c'est l'alternative
standard aux SVM à noyau sur de gros jeux de données.
En plus du JSON et de la soumission, les probabilités OOF et test sont écrites dans results/
(<name>_<stamp>_oof.csv et <name>_<stamp>_test.csv) pour les blends ultérieurs.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_clean_codes import LGB_PARAMS, ROOT, SEED, clean, evaluate, load, select_columns  # noqa: E402

IDEA = ("SVM à noyau RBF approché : colonnes vides à > 95 % retirées, top-80 colonnes par fold (gain LightGBM), "
        "one-hot des catégorielles + quantiles des numériques + indicateurs de valeur manquante (dense float32), "
        "Nystroem(rbf, gamma=1/n_features, n_components=1500, random_state=42) ajusté sur 100 000 lignes "
        "de la partie entraînement du fold, puis LinearSVC(C=0.5, dual=False, max_iter=2000) dans "
        "CalibratedClassifierCV(cv=3, sigmoid) ajusté sur la partie entraînement du fold ; "
        "un SVM à noyau exact est quadratique en nombre de lignes, donc impossible sur 370 000 lignes")
NAN_MAX = 0.95          # colonnes vides à plus de 95 % : retirées d'office (critère sans TARGET)
CAT_MAX_LEVELS = 15     # <= 15 modalités distinctes -> catégorielle (one-hot)
CONFIG = {
    "top_k": 80,                 # colonnes gardées par fold
    "onehot_max_cols": 600,      # au-delà : one-hot limité aux modalités fréquentes
    "onehot_min_frequency": 0.01,
    "n_components": 1500,        # composantes Nystroem
    "nystroem_fit_rows": 100_000,
    "nystroem_seed": 42,
    "C": 0.5,
    "max_iter": 2000,
    "calib_cv": 3,
    "calib_method": "sigmoid",
    "calib_n_jobs": 2,           # ajustements de la calibration en parallèle (threads) ; ~7 Go par ajustement
    "transform_chunk": 50_000,   # lignes par bloc pour la transformation Nystroem (mémoire)
    "perm_importance_rows": 20_000,  # fold 0 : importance par permutation sur un échantillon de validation
    "perm_importance_top": 15,
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
    """Ajusté sur la partie entraînement du fold uniquement ; n'utilise jamais TARGET."""

    def __init__(self, cats, nums):
        self.cats, self.nums = cats, nums

    def fit(self, df):
        self.maps = {c: {v: i + 1 for i, v in enumerate(sorted(df[c].dropna().unique().tolist()))}
                     for c in self.cats}
        self.cardinalities = [len(self.maps[c]) + 1 for c in self.cats]  # 0 = manquant / inconnu
        self.qt = (QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                       subsample=200_000, random_state=SEED).fit(df[self.nums])
                   if self.nums else None)
        return self

    def transform(self, df):
        n = len(df)
        if self.cats:
            cat = np.stack([df[c].astype("float64").map(self.maps[c]).fillna(0).to_numpy(dtype=np.int64)
                            for c in self.cats], axis=1)
        else:
            cat = np.zeros((n, 0), dtype=np.int64)
        if self.nums:
            num = self.qt.transform(df[self.nums]).astype(np.float32)
            missing = np.isnan(num)
            num[missing] = 0.0
            num = np.concatenate([num, missing.astype(np.float32)], axis=1)
        else:
            num = np.zeros((n, 0), dtype=np.float32)
        return cat, num


class DenseEncoder:
    """One-hot des codes catégoriels (ajusté sur la partie entraînement du fold) + numériques, dense float32."""

    def __init__(self, pre, cfg):
        self.pre, self.cfg = pre, cfg

    def fit(self, df):
        cat, num = self.pre.transform(df)
        self.ohe = None
        self.min_frequency = None
        if cat.shape[1]:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32).fit(cat)
            n_cols = sum(len(c) for c in self.ohe.categories_) + num.shape[1]
            if n_cols > self.cfg["onehot_max_cols"]:
                self.min_frequency = self.cfg["onehot_min_frequency"]
                self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32,
                                         min_frequency=self.min_frequency).fit(cat)
                print(f"  {n_cols} colonnes denses > {self.cfg['onehot_max_cols']} : one-hot limité aux "
                      f"modalités de fréquence >= {self.min_frequency:.0%}")
        self.n_features = self.transform(df.head(2)).shape[1]
        return self

    def transform(self, df):
        cat, num = self.pre.transform(df)
        parts = [num]
        if self.ohe is not None:
            parts.insert(0, self.ohe.transform(cat).astype(np.float32))
        return np.concatenate(parts, axis=1)


# ------------------------------------------------------------------------------------ modèle
class NystroemSVM:
    """Nystroem (rbf) ajusté sur un sous-échantillon de la partie entraînement du fold, puis LinearSVC
    calibré (sigmoid, cv=3) ajusté sur toute la partie entraînement du fold."""

    def __init__(self, cfg, fold):
        self.cfg, self.fold = cfg, fold

    def _nystroem_transform(self, X, dtype):
        cfg = self.cfg
        out = np.empty((len(X), self.nys.n_components), dtype=dtype)
        for i in range(0, len(X), cfg["transform_chunk"]):
            out[i:i + cfg["transform_chunk"]] = self.nys.transform(X[i:i + cfg["transform_chunk"]])
        return out

    def fit(self, X, y):
        cfg = self.cfg
        t = time.time()
        rng = np.random.RandomState(SEED + self.fold)
        sub = rng.choice(len(X), size=min(cfg["nystroem_fit_rows"], len(X)), replace=False)
        self.nys = Nystroem(kernel="rbf", gamma=1.0 / X.shape[1], n_components=cfg["n_components"],
                            random_state=cfg["nystroem_seed"]).fit(X[sub])
        Z = self._nystroem_transform(X, np.float64)  # float64 : évite une copie dans LinearSVC
        self.t_nystroem = time.time() - t
        print(f"  fold {self.fold}: Nystroem {Z.shape} en {self.t_nystroem:.0f} s", flush=True)
        t = time.time()
        base = LinearSVC(C=cfg["C"], dual=False, max_iter=cfg["max_iter"], random_state=SEED)
        self.clf = CalibratedClassifierCV(
            base, method=cfg["calib_method"], n_jobs=cfg["calib_n_jobs"],
            cv=StratifiedKFold(n_splits=cfg["calib_cv"], shuffle=True, random_state=SEED + self.fold))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            with parallel_backend("threading"):  # liblinear libère le GIL ; pas de copie des données
                self.clf.fit(Z, y)
        self.t_svm = time.time() - t
        self.n_iter = [int(np.ravel(c.estimator.n_iter_)[0]) for c in self.clf.calibrated_classifiers_]
        self.sigmoid = [(float(c.calibrators[0].a_), float(c.calibrators[0].b_))
                        for c in self.clf.calibrated_classifiers_]
        print(f"  fold {self.fold}: LinearSVC calibré en {self.t_svm:.0f} s, itérations {self.n_iter}, "
              f"sigmoïdes (a, b) {[(round(a, 3), round(b, 3)) for a, b in self.sigmoid]}", flush=True)
        del Z
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self._nystroem_transform(X, np.float32))[:, 1].astype(np.float64)


def permutation_importance(df_va, y_va, enc, model, columns, rows, fold):
    """Baisse d'AUC quand une colonne est permutée, sur un échantillon de la partie validation du fold."""
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(df_va), size=min(rows, len(df_va)), replace=False)
    sample, ys = df_va.iloc[idx].reset_index(drop=True), y_va[idx]
    base = roc_auc_score(ys, model.predict_proba(enc.transform(sample)))
    drops = {}
    for c in columns:
        shuffled = sample.copy()
        shuffled[c] = shuffled[c].to_numpy()[rng.permutation(len(shuffled))]
        drops[c] = round(float(base - roc_auc_score(ys, model.predict_proba(enc.transform(shuffled)))), 5)
    drops = dict(sorted(drops.items(), key=lambda kv: -kv[1]))
    print(f"  fold {fold}: importance par permutation (baisse d'AUC, {len(sample)} lignes) : {drops}", flush=True)
    return drops


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ml_svm_nystroem")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=None, help="remplace CONFIG['top_k']")
    parser.add_argument("--n-components", type=int, default=None, help="remplace CONFIG['n_components']")
    parser.add_argument("--rows", type=int, default=None, help="test local : n'utilise que les N premières lignes")
    args = parser.parse_args()
    cfg = dict(CONFIG)
    if args.top_k:
        cfg["top_k"] = args.top_k
    if args.n_components:
        cfg["n_components"] = args.n_components
    t0 = time.time()

    train, test = load()
    if args.rows:  # test local de l'API uniquement, jamais pour un résultat
        train, test = train.head(args.rows).copy(), test.head(args.rows).copy()
        cfg.update(nystroem_fit_rows=min(cfg["nystroem_fit_rows"], args.rows // 2),
                   n_components=min(cfg["n_components"], args.rows // 4),
                   perm_importance_rows=min(cfg["perm_importance_rows"], args.rows // 5))
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
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        tf = time.time()
        df_tr, df_va = train.iloc[tr_idx], train.iloc[va_idx]
        selected, gain = select_top_k(df_tr, y[tr_idx], feats, cfg["top_k"], args.threads)
        cats = [c for c in selected if c in cats_all]
        nums = [c for c in selected if c not in cats_all]
        pre = FoldPreprocessor(cats, nums).fit(df_tr)
        enc = DenseEncoder(pre, cfg).fit(df_tr)
        X_tr = enc.transform(df_tr)
        print(f"fold {fold}: {len(cats)} cat + {len(nums)} num -> {X_tr.shape[1]} colonnes denses "
              f"({X_tr.nbytes / 1e9:.2f} Go), sélection + encodage {time.time() - tf:.0f} s", flush=True)
        model = NystroemSVM(cfg, fold).fit(X_tr, y[tr_idx])
        del X_tr
        oof[va_idx] = model.predict_proba(enc.transform(df_va))
        test_proba += model.predict_proba(enc.transform(test)) / args.folds
        auc = roc_auc_score(y[va_idx], oof[va_idx])
        perm = None
        if fold == 0 and cfg["perm_importance_top"]:
            perm = permutation_importance(df_va, y[va_idx], enc, model, selected[:cfg["perm_importance_top"]],
                                          cfg["perm_importance_rows"], fold)
        print(f"fold {fold}: AUC {auc:.4f}, {len(cats)} cat + {len(nums)} num, {time.time() - tf:.0f} s", flush=True)
        folds_info.append({"selected": selected, "n_cat": len(cats), "n_num": len(nums),
                           "n_dense": int(enc.n_features), "onehot_min_frequency": enc.min_frequency,
                           "auc": round(float(auc), 5), "s": round(time.time() - tf, 1),
                           "s_nystroem": round(model.t_nystroem, 1), "s_svm": round(model.t_svm, 1),
                           "svm_n_iter": model.n_iter, "sigmoid_ab": model.sigmoid,
                           "top10_gain": {k: round(float(v)) for k, v in gain.head(10).items()},
                           **({"perm_importance_auc_drop": perm} if perm else {})})
        del model, enc, pre

    metrics = evaluate(y, oof)
    save_outputs(args.name, test, oof, test_proba, metrics,
                 {"model": "nystroem_rbf_linearsvc_calibrated", "config": cfg, "nan_max": NAN_MAX,
                  "cat_max_levels": CAT_MAX_LEVELS, "folds": folds_info}, t0)


if __name__ == "__main__":
    main()
