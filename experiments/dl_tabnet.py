"""TabNet (pytorch-tabnet) avec masques d'attention sur les top-80 colonnes par fold, GPU.

Usage (Modal, GPU) :
    python -W ignore -m modal run modal_run.py --script experiments/dl_tabnet.py --name dl_tabnet --threads 8 --gpu

Copie de dl_base.py où seule la partie modèle/entraînement change : TabNetClassifier remplace le MLP
PyTorch et sa boucle AMP. Tout le reste est identique (protocole CV 5 folds seed 42, colonnes vides à
> 95 % retirées, top-K colonnes par fold via LightGBM sur la partie entraînement du fold, codes entiers
des catégorielles + quantiles des numériques ajustés sur la partie entraînement, seuil choisi sur
l'OOF, save_outputs). Dans chaque fold :
- X = [codes entiers des catégorielles | numériques quantilisées | indicateurs de valeur manquante],
  cat_idxs = indices des catégorielles, cat_dims = cardinalités (0 = manquant), cat_emb_dim = 4 ;
- TabNetClassifier (n_d = n_a = 32, 4 étapes, gamma 1.5, entmax, lambda_sparse 1e-4, Adam lr 2e-2,
  StepLR 10 époques x 0.9), fit avec early stopping sur la logloss de validation (patience 4,
  meilleurs poids restaurés par pytorch-tabnet) ; OOF = predict_proba du fold, test = moyenne des folds ;
- les importances globales de TabNet (feature_importances_, issues des masques d'attention) sont
  moyennées sur les folds, alignées sur les noms de colonnes, et écrites dans le JSON.
Garde-fou de temps : si un fold dépasse 7 min, les folds suivants passent à 20 époques / batch 8192.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_clean_codes import LGB_PARAMS, ROOT, SEED, clean, evaluate, load, select_columns  # noqa: E402

IDEA = ("TabNet (pytorch-tabnet, masques d'attention entmax, n_d=n_a=32, 4 étapes, embeddings dim 4) "
        "sur les top-80 colonnes par fold (gain LightGBM) ; même sélection, prétraitement et seuil que "
        "dl_base ; importances des masques moyennées sur les folds")
NAN_MAX = 0.95          # colonnes vides à plus de 95 % : retirées d'office (critère sans TARGET)
CAT_MAX_LEVELS = 15     # <= 15 modalités distinctes -> catégorielle (embedding)
CONFIG = {
    "top_k": 80,            # colonnes gardées par fold
    "max_epochs": 30,
    "patience": 4,          # époques sans amélioration de la logloss de validation
    "batch_size": 4096,
    "virtual_batch_size": 512,
    "lr": 2e-2,
    "scheduler_step_size": 10,
    "scheduler_gamma": 0.9,
    "cat_emb_dim": 4,
    "n_d": 32,
    "n_a": 32,
    "n_steps": 4,
    "gamma": 1.5,
    "lambda_sparse": 1e-4,
    "mask_type": "entmax",
    "slow_fold_s": 420,     # au-delà de 7 min par fold : max_epochs 20 et batch_size 8192 ensuite
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

    def matrix(self, df):
        """Matrice TabNet : catégorielles (codes entiers) d'abord, puis numériques + indicateurs."""
        cat, num = self.transform(df)
        return np.concatenate([cat.astype(np.float32), num], axis=1)

    def column_names(self):
        return self.cats + self.nums + [f"{c}__missing" for c in self.nums]


# ------------------------------------------------------------------------------------ modèle
def build_model(cardinalities, cfg, device, fold):
    """TabNetClassifier : les catégorielles occupent les premières colonnes de X (cat_idxs)."""
    return TabNetClassifier(
        cat_idxs=list(range(len(cardinalities))), cat_dims=list(cardinalities), cat_emb_dim=cfg["cat_emb_dim"],
        n_d=cfg["n_d"], n_a=cfg["n_a"], n_steps=cfg["n_steps"], gamma=cfg["gamma"],
        lambda_sparse=cfg["lambda_sparse"], mask_type=cfg["mask_type"],
        optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=cfg["lr"]),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=cfg["scheduler_step_size"], gamma=cfg["scheduler_gamma"]),
        seed=SEED + fold, device_name=device.type, verbose=1,
    )


# ------------------------------------------------------------------------------ entraînement
def train_fold(model, X_tr, y_tr, X_va, y_va, cfg):
    """Boucle d'entraînement propre à pytorch-tabnet : early stopping sur la logloss de validation,
    meilleurs poids restaurés en fin de fit par le callback EarlyStopping de la librairie."""
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_name=["val"], eval_metric=["logloss"],
              max_epochs=cfg["max_epochs"], patience=cfg["patience"], batch_size=cfg["batch_size"],
              virtual_batch_size=cfg["virtual_batch_size"], num_workers=0, drop_last=False)
    hist = model.history
    history = [{"epoch": e, "train_loss": round(float(hist["loss"][e]), 5),
                "val_logloss": round(float(hist["val_logloss"][e]), 5), "lr": float(hist["lr"][e])}
               for e in range(len(hist["loss"]))]
    return history, int(model.best_epoch if model.best_epoch is not None else len(history) - 1)


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
    parser.add_argument("--name", default="dl_tabnet")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=None, help="remplace CONFIG['top_k']")
    args = parser.parse_args()
    cfg = dict(CONFIG)
    if args.top_k:
        cfg["top_k"] = args.top_k
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    t0 = time.time()

    train, test = load()
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
    imp_sum, imp_count = {}, {}   # importances TabNet (masques) cumulées par nom de colonne
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        tf = time.time()
        df_tr, df_va = train.iloc[tr_idx], train.iloc[va_idx]
        selected, gain = select_top_k(df_tr, y[tr_idx], feats, cfg["top_k"], args.threads)
        cats = [c for c in selected if c in cats_all]
        nums = [c for c in selected if c not in cats_all]
        pre = FoldPreprocessor(cats, nums).fit(df_tr)
        X_tr, X_va, X_te = pre.matrix(df_tr), pre.matrix(df_va), pre.matrix(test)
        torch.manual_seed(SEED + fold)
        model = build_model(pre.cardinalities, cfg, device, fold)
        if fold == 0:
            print(f"TabNet : X {X_tr.shape}, {len(cats)} cat (emb {cfg['cat_emb_dim']}) + {len(nums)} num "
                  f"+ {len(nums)} indicateurs ; n_d={cfg['n_d']} n_a={cfg['n_a']} n_steps={cfg['n_steps']} "
                  f"gamma={cfg['gamma']} {cfg['mask_type']} lr={cfg['lr']} batch={cfg['batch_size']}")
        history, best_epoch = train_fold(model, X_tr, y[tr_idx], X_va, y[va_idx], cfg)
        if fold == 0:
            print("paramètres :", sum(p.numel() for p in model.network.parameters()))
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_proba += model.predict_proba(X_te)[:, 1] / args.folds
        auc = roc_auc_score(y[va_idx], oof[va_idx])
        names = pre.column_names()
        imp = np.asarray(model.feature_importances_, dtype=float)
        assert len(imp) == len(names), (len(imp), len(names))
        for n_, v in zip(names, imp):
            imp_sum[n_] = imp_sum.get(n_, 0.0) + float(v)
            imp_count[n_] = imp_count.get(n_, 0) + 1
        fold_imp = pd.Series(imp, index=names).sort_values(ascending=False)
        elapsed = time.time() - tf
        print(f"fold {fold}: {len(history)} époques (meilleure {best_epoch}), AUC {auc:.4f}, "
              f"{len(cats)} cat + {len(nums)} num, {elapsed:.0f} s", flush=True)
        print("  top 10 importances TabNet :", {k: round(float(v), 4) for k, v in fold_imp.head(10).items()})
        folds_info.append({"selected": selected, "n_cat": len(cats), "n_num": len(nums),
                           "epochs": len(history), "best_epoch": best_epoch, "auc": round(float(auc), 5),
                           "seconds": round(elapsed, 1), "history": history,
                           "config": {k: cfg[k] for k in ("max_epochs", "batch_size")},
                           "top10_gain": {k: round(float(v)) for k, v in gain.head(10).items()},
                           "top10_tabnet_importance": {k: round(float(v), 5) for k, v in fold_imp.head(10).items()}})
        if elapsed > cfg["slow_fold_s"] and cfg["max_epochs"] > 20:
            cfg["max_epochs"], cfg["batch_size"] = 20, 8192
            print(f"  fold lent (> {cfg['slow_fold_s']} s) : folds suivants à 20 époques, batch 8192", flush=True)
        del model, X_tr, X_va, X_te
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Importances TabNet moyennées sur les folds (somme / nombre de folds ; colonne absente d'un fold = 0)
    importance = pd.Series({k: v / args.folds for k, v in imp_sum.items()}).sort_values(ascending=False)
    print("Top 20 importances TabNet (moyenne des folds) :")
    print(importance.head(20).round(4).to_string())
    metrics = evaluate(y, oof)
    save_outputs(args.name, test, oof, test_proba, metrics,
                 {"model": "dl", "config": cfg, "device": str(device), "nan_max": NAN_MAX,
                  "cat_max_levels": CAT_MAX_LEVELS,
                  "tabnet_importance": {k: round(float(v), 5) for k, v in importance.items()},
                  "tabnet_importance_folds": imp_count, "folds": folds_info}, t0)


if __name__ == "__main__":
    main()
