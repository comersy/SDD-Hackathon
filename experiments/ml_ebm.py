"""Explainable Boosting Machine (interpret) : sélection de colonnes par fold + EBM sur CPU.

Usage (Modal, CPU 16 cœurs) :
    python -W ignore -m modal run modal_run.py --script experiments/ml_ebm.py --name ml_ebm --threads 16

Protocole identique à train.py / dl_base.py (même CV 5 folds seed 42, même seuil choisi sur l'OOF).
Dans chaque fold :
1. sélection : LightGBM rapide ajusté sur la partie entraînement du fold -> top-K colonnes par gain
   (K = CONFIG["top_k"]) ; la liste peut différer d'un fold à l'autre, aucune fuite ;
2. prétraitement ajusté sur la partie entraînement du fold (sans TARGET) : colonnes à <= 15 modalités
   -> codes entiers (0 = manquant / modalité inconnue), déclarées "nominal" à l'EBM ; autres colonnes ->
   QuantileTransformer (loi normale), déclarées "continuous" ; les indicateurs de valeur manquante
   sont retirés (l'EBM gère nativement les NaN avec un bin « manquant », on les lui restitue) ;
3. ExplainableBoostingClassifier (interpret) ajusté sur le fold ; OOF = predict_proba du fold ;
   test = moyenne des folds. Les 15 termes les plus importants du fold 0 sont écrits dans le JSON.
Si le fold 0 dépasse FALLBACK_AFTER_S (6 min), la CV repart de zéro avec CONFIG_FAST
(interactions=0, outer_bags=1, top_k=40).
En plus du JSON et de la soumission, les probabilités OOF et test sont écrites dans results/
(<name>_<stamp>_oof.csv et <name>_<stamp>_test.csv) pour les blends ultérieurs.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

try:
    from interpret.glassbox import ExplainableBoostingClassifier
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "interpret-core"])
    from interpret.glassbox import ExplainableBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_clean_codes import LGB_PARAMS, ROOT, SEED, clean, evaluate, load, select_columns  # noqa: E402

IDEA = ("ML EBM : colonnes vides à > 95 % retirées, top-60 colonnes par fold (gain LightGBM), "
        "codes des catégorielles (nominal) + quantiles des numériques sans indicateurs (NaN natifs), "
        "ExplainableBoostingClassifier(max_bins=128, interactions=5, outer_bags=2, inner_bags=0, "
        "learning_rate=0.03, max_rounds=2000, early_stopping_rounds=50) sur CPU")
NAN_MAX = 0.95          # colonnes vides à plus de 95 % : retirées d'office (critère sans TARGET)
CAT_MAX_LEVELS = 15     # <= 15 modalités distinctes -> catégorielle (nominal)
FALLBACK_AFTER_S = 360  # fold 0 plus long que 6 min -> CONFIG_FAST
CONFIG = {
    "top_k": 60,            # colonnes gardées par fold
    "max_bins": 128,
    "interactions": 5,
    "outer_bags": 2,
    "inner_bags": 0,
    "learning_rate": 0.03,
    "max_rounds": 2000,
    "early_stopping_rounds": 50,
}
CONFIG_FAST = {**CONFIG, "top_k": 40, "interactions": 0, "outer_bags": 1}
TOP_TERMS = 15


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
    Variante EBM : pas d'indicateurs de valeur manquante, les NaN des numériques sont conservés."""

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
            num = self.qt.transform(df[self.nums]).astype(np.float32)  # NaN conservés (bin « manquant » EBM)
        else:
            num = np.zeros((n, 0), dtype=np.float32)
        return cat, num


def to_matrix(cat, num):
    """Matrice float64 : codes des catégorielles (valeurs entières) puis numériques quantiles."""
    return np.concatenate([cat.astype(np.float64), num.astype(np.float64)], axis=1)


# ------------------------------------------------------------------------------------ modèle
def build_model(cfg, threads, fold):
    return ExplainableBoostingClassifier(
        max_bins=cfg["max_bins"], interactions=cfg["interactions"], outer_bags=cfg["outer_bags"],
        inner_bags=cfg["inner_bags"], learning_rate=cfg["learning_rate"], max_rounds=cfg["max_rounds"],
        early_stopping_rounds=cfg["early_stopping_rounds"], n_jobs=threads, random_state=SEED + fold,
    )


def top_terms(model, k=TOP_TERMS):
    imp = model.term_importances()
    order = np.argsort(imp)[::-1][:k]
    return {str(model.term_names_[i]): round(float(imp[i]), 5) for i in order}


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


# ------------------------------------------------------------------------------ validation croisée
def run_cv(train, test, y, feats, cats_all, cfg, args, allow_fallback):
    """Retourne (oof, test_proba, folds_info) ou None si le fold 0 est trop lent (repli demandé)."""
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
        x_tr = to_matrix(*pre.transform(df_tr))
        x_va = to_matrix(*pre.transform(df_va))
        x_te = to_matrix(*pre.transform(test))
        names = cats + nums
        types = ["nominal"] * len(cats) + ["continuous"] * len(nums)
        model = build_model(cfg, args.threads, fold)
        model.set_params(feature_names=names, feature_types=types)
        model.fit(x_tr, y[tr_idx])
        oof[va_idx] = model.predict_proba(x_va)[:, 1]
        test_proba += model.predict_proba(x_te)[:, 1] / args.folds
        auc = roc_auc_score(y[va_idx], oof[va_idx])
        n_terms = len(model.term_names_)
        n_inter = sum(len(t) > 1 for t in model.term_features_)
        elapsed = time.time() - tf
        print(f"fold {fold}: AUC {auc:.4f}, {len(cats)} cat + {len(nums)} num, {n_terms} termes "
              f"dont {n_inter} interactions, {elapsed:.0f} s", flush=True)
        info = {"selected": selected, "n_cat": len(cats), "n_num": len(nums), "n_terms": n_terms,
                "n_interactions": n_inter, "auc": round(float(auc), 5), "s": round(elapsed, 1),
                "top10_gain": {k: round(float(v)) for k, v in gain.head(10).items()}}
        if fold == 0:
            info["top_terms"] = top_terms(model)
            print(f"  top {TOP_TERMS} termes du fold 0 :", json.dumps(info["top_terms"], ensure_ascii=False))
            if allow_fallback and elapsed > FALLBACK_AFTER_S:
                print(f"fold 0 trop lent ({elapsed:.0f} s > {FALLBACK_AFTER_S} s) : repli sur CONFIG_FAST", flush=True)
                return None
        folds_info.append(info)
        del model, x_tr, x_va, x_te
    return oof, test_proba, folds_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ml_ebm")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=None, help="remplace CONFIG['top_k']")
    parser.add_argument("--sample", type=int, default=0, help="test local : n lignes tirées au hasard (0 = tout)")
    args = parser.parse_args()
    cfg = dict(CONFIG)
    if args.top_k:
        cfg["top_k"] = args.top_k
    t0 = time.time()

    train, test = load()
    if args.sample:
        train = train.sample(args.sample, random_state=SEED).reset_index(drop=True)
        test = test.head(args.sample).reset_index(drop=True)
    feats = drop_sparse(train, select_columns(train))
    maxes = train[feats].max()
    clean(train, feats, maxes)
    clean(test, feats, maxes)
    y = train["TARGET"].values.astype(int)
    nunique = train[feats].nunique()
    cats_all = {c for c in feats if nunique[c] <= CAT_MAX_LEVELS}
    print(f"{len(feats)} colonnes candidates dont {len(cats_all)} catégorielles ; top-{cfg['top_k']} par fold")
    print("config :", cfg, flush=True)

    out = run_cv(train, test, y, feats, cats_all, cfg, args, allow_fallback=not args.sample)
    fallback = out is None
    if fallback:
        cfg = dict(CONFIG_FAST)
        print("config de repli :", cfg, flush=True)
        out = run_cv(train, test, y, feats, cats_all, cfg, args, allow_fallback=False)
    oof, test_proba, folds_info = out

    metrics = evaluate(y, oof)
    save_outputs(args.name, test, oof, test_proba, metrics,
                 {"model": "ebm", "config": cfg, "fallback_used": fallback, "nan_max": NAN_MAX,
                  "cat_max_levels": CAT_MAX_LEVELS, "top_terms_fold0": folds_info[0]["top_terms"],
                  "folds": folds_info}, t0)


if __name__ == "__main__":
    main()
