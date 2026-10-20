"""Extra-Trees (scikit-learn) sur toutes les colonnes nettoyées, même entrée que le Random Forest.

Usage (Modal, CPU 16 cœurs) :
    python -W ignore -m modal run modal_run.py --script experiments/ml_extra_trees.py --name ml_extra_trees --threads 16

Protocole identique à train.py / dl_base.py (même CV 5 folds seed 42, même seuil choisi sur l'OOF).
Dans chaque fold :
1. sélection : LightGBM rapide ajusté sur la partie entraînement du fold -> top-K colonnes par gain ;
   ici K = toutes les colonnes candidates (214), la sélection ne sert qu'à documenter le gain ;
2. prétraitement ajusté sur la partie entraînement du fold (sans TARGET) : colonnes à <= 15 modalités
   -> codes entiers (0 = manquant / modalité inconnue) ; autres colonnes -> QuantileTransformer
   (loi normale), NaN -> 0 + indicateur de valeur manquante ; X = [codes | quantiles | indicateurs] ;
3. ExtraTreesClassifier(n_estimators=600, max_features=0.3, min_samples_leaf=3, n_jobs=threads,
   random_state=42+fold). Si les 600 arbres du fold 0 dépassent 5 min (projection faite après
   300 arbres via warm_start), tous les folds utilisent 300 arbres. OOF = predict_proba du fold ;
   test = moyenne des folds ; feature_importances_ moyennées (top 20 sauvegardé).
En plus du JSON et de la soumission, les probabilités OOF et test sont écrites dans results/
(<name>_<stamp>_oof.csv et <name>_<stamp>_test.csv) pour les blends / stacking ultérieurs.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_clean_codes import LGB_PARAMS, ROOT, SEED, clean, evaluate, load, select_columns  # noqa: E402

IDEA = ("Extra-Trees sklearn sur toutes les colonnes nettoyées (214 candidates, vides à > 95 % retirées) : "
        "codes entiers des catégorielles + quantiles normaux et indicateurs de NaN des numériques, "
        "ExtraTreesClassifier(n_estimators=600, max_features=0.3, min_samples_leaf=3, "
        "random_state=42+fold) ; même entrée que le Random Forest, pour le stacking")
NAN_MAX = 0.95          # colonnes vides à plus de 95 % : retirées d'office (critère sans TARGET)
CAT_MAX_LEVELS = 15     # <= 15 modalités distinctes -> codes entiers
CONFIG = {
    "top_k": None,          # None = toutes les colonnes candidates (fixé à len(feats) dans main)
    "n_estimators": 600,
    "n_estimators_fallback": 300,   # si les 600 arbres d'un fold dépassent FOLD_MAX_S
    "max_features": 0.3,
    "min_samples_leaf": 3,
}
FOLD_MAX_S = 300        # budget d'entraînement par fold (5 min)


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

    def feature_names(self):
        return list(self.cats) + list(self.nums) + [f"{c}__nan" for c in self.nums]


def to_matrix(cat, num):
    """X = [codes entiers | quantiles | indicateurs de NaN], float32 (format natif des arbres sklearn)."""
    return np.concatenate([cat.astype(np.float32), num], axis=1)


# ------------------------------------------------------------------------------ entraînement
def build_model(n_estimators, cfg, threads, fold, warm_start=False):
    return ExtraTreesClassifier(n_estimators=n_estimators, max_features=cfg["max_features"],
                                min_samples_leaf=cfg["min_samples_leaf"], n_jobs=threads,
                                random_state=SEED + fold, warm_start=warm_start)


def fit_fold(X_tr, y_tr, cfg, threads, fold, n_estimators):
    """Ajuste le modèle ; au fold 0 (n_estimators None), décide 600 ou 300 arbres d'après le temps
    des 300 premiers (warm_start) : projection > FOLD_MAX_S -> 300 arbres pour tous les folds."""
    if n_estimators is not None:
        model = build_model(n_estimators, cfg, threads, fold)
        model.fit(X_tr, y_tr)
        return model, n_estimators
    half = cfg["n_estimators_fallback"]
    model = build_model(half, cfg, threads, fold, warm_start=True)
    t = time.time()
    model.fit(X_tr, y_tr)
    half_s = time.time() - t
    projected = half_s * cfg["n_estimators"] / half
    print(f"  fold {fold}: {half} arbres en {half_s:.0f} s, projection {cfg['n_estimators']} arbres "
          f"= {projected:.0f} s (budget {FOLD_MAX_S} s)", flush=True)
    if projected <= FOLD_MAX_S:
        model.set_params(n_estimators=cfg["n_estimators"])
        model.fit(X_tr, y_tr)  # warm_start : ajoute les arbres manquants
        return model, cfg["n_estimators"]
    print(f"  -> trop lent : {half} arbres pour tous les folds", flush=True)
    return model, half


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
    parser.add_argument("--name", default="ml_extra_trees")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=None, help="remplace CONFIG['top_k'] (défaut : toutes)")
    parser.add_argument("--sample", type=int, default=None, help="test local : n lignes tirées au hasard")
    args = parser.parse_args()
    cfg = dict(CONFIG)
    t0 = time.time()

    train, test = load()
    if args.sample:
        train = train.sample(args.sample, random_state=SEED).reset_index(drop=True)
        test = test.head(args.sample).reset_index(drop=True)
        cfg["n_estimators"], cfg["n_estimators_fallback"] = 20, 10
    feats = drop_sparse(train, select_columns(train))
    maxes = train[feats].max()
    clean(train, feats, maxes)
    clean(test, feats, maxes)
    cfg["top_k"] = args.top_k or len(feats)
    y = train["TARGET"].values.astype(int)
    nunique = train[feats].nunique()
    cats_all = {c for c in feats if nunique[c] <= CAT_MAX_LEVELS}
    print(f"{len(feats)} colonnes candidates dont {len(cats_all)} catégorielles ; top-{cfg['top_k']} par fold")

    oof = np.zeros(len(train))
    test_proba = np.zeros(len(test))
    importance = pd.Series(dtype="float64")
    folds_info = []
    n_estimators = None  # décidé au fold 0
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        tf = time.time()
        df_tr, df_va = train.iloc[tr_idx], train.iloc[va_idx]
        selected, gain = select_top_k(df_tr, y[tr_idx], feats, cfg["top_k"], args.threads)
        cats = [c for c in selected if c in cats_all]
        nums = [c for c in selected if c not in cats_all]
        pre = FoldPreprocessor(cats, nums).fit(df_tr)
        X_tr = to_matrix(*pre.transform(df_tr))
        X_va = to_matrix(*pre.transform(df_va))
        X_te = to_matrix(*pre.transform(test))
        names = pre.feature_names()
        if fold == 0:
            print(f"X : {X_tr.shape[1]} colonnes ({len(cats)} codes + {len(nums)} quantiles + {len(nums)} indicateurs NaN)")
        t_fit = time.time()
        model, n_estimators = fit_fold(X_tr, y[tr_idx], cfg, args.threads, fold, n_estimators)
        fit_s = time.time() - t_fit
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_proba += model.predict_proba(X_te)[:, 1] / args.folds
        imp = pd.Series(model.feature_importances_, index=names)
        importance = importance.add(imp / args.folds, fill_value=0.0)
        auc = roc_auc_score(y[va_idx], oof[va_idx])
        print(f"fold {fold}: {n_estimators} arbres, AUC {auc:.4f}, {len(cats)} cat + {len(nums)} num, "
              f"fit {fit_s:.0f} s, total {time.time() - tf:.0f} s", flush=True)
        folds_info.append({"n_estimators": n_estimators, "n_cat": len(cats), "n_num": len(nums),
                           "n_features": int(X_tr.shape[1]), "auc": round(float(auc), 5),
                           "fit_s": round(fit_s, 1), "fold_s": round(time.time() - tf, 1),
                           "top10_imp": {k: round(float(v), 5) for k, v in imp.sort_values(ascending=False).head(10).items()},
                           "top10_gain": {k: round(float(v)) for k, v in gain.head(10).items()}})
        del model, X_tr, X_va, X_te

    metrics = evaluate(y, oof)
    top = importance.sort_values(ascending=False).head(20)
    print("Top 20 variables (feature_importances_ moyennées) :")
    print(top.round(5).to_string())
    cfg["n_estimators_used"] = n_estimators
    save_outputs(args.name, test, oof, test_proba, metrics,
                 {"model": "extra_trees", "config": cfg, "nan_max": NAN_MAX,
                  "cat_max_levels": CAT_MAX_LEVELS, "n_candidates": len(feats),
                  "top_features": {k: round(float(v), 5) for k, v in top.items()},
                  "folds": folds_info}, t0)


if __name__ == "__main__":
    main()
