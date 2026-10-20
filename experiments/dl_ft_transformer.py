"""Expérience DL : FT-Transformer (rtdl_revisiting_models) sur les top-80 colonnes par fold.

Copie de dl_base.py ; seuls IDEA, CONFIG, build_model() (adaptateur FT-Transformer) et les groupes
de paramètres AdamW (model.make_parameter_groups()) changent, plus un garde-fou de temps par fold.

Usage (Modal, GPU) :
    python -W ignore -m modal run modal_run.py --script experiments/dl_ft_transformer.py --name dl_ft_transformer --threads 8 --gpu

Protocole identique à train.py (même CV 5 folds seed 42, même seuil choisi sur l'OOF). Dans chaque fold :
1. sélection : LightGBM rapide ajusté sur la partie entraînement du fold -> top-K colonnes par gain
   (K = CONFIG["top_k"]) ; la liste peut différer d'un fold à l'autre, aucune fuite ;
2. prétraitement ajusté sur la partie entraînement du fold (sans TARGET) : colonnes à <= 15 modalités
   -> codes entiers (0 = manquant / modalité inconnue) pour des embeddings ; autres colonnes ->
   QuantileTransformer (loi normale), NaN -> 0 + indicateur de valeur manquante ;
3. réseau PyTorch construit par build_model(), AdamW + early stopping sur la logloss de validation
   (AMP bfloat16 sur GPU). OOF = meilleur état du fold ; test = moyenne des folds.
Une expérience DL copie ce script et remplace build_model() et/ou CONFIG (une seule idée par script).
En plus du JSON et de la soumission, les probabilités OOF et test sont écrites dans results/
(<name>_<stamp>_oof.csv et <name>_<stamp>_test.csv) pour les blends ultérieurs.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from rtdl_revisiting_models import FTTransformer
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_clean_codes import LGB_PARAMS, ROOT, SEED, clean, evaluate, load, select_columns  # noqa: E402

IDEA = ("FT-Transformer (rtdl_revisiting_models, 3 blocs, d_block 192, 8 têtes) sur les top-80 colonnes "
        "par fold : tokens = embeddings des catégorielles + embeddings linéaires des quantiles et des "
        "indicateurs de valeur manquante ; AdamW lr 1e-4 avec make_parameter_groups(), batch 1024")
NAN_MAX = 0.95          # colonnes vides à plus de 95 % : retirées d'office (critère sans TARGET)
CAT_MAX_LEVELS = 15     # <= 15 modalités distinctes -> catégorielle (embedding)
CONFIG = {
    "top_k": 80,            # colonnes gardées par fold
    "epochs": 25,
    "patience": 4,          # époques sans amélioration de la logloss de validation
    "batch_size": 1024,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "n_blocks": 3,          # FTTransformer.get_default_kwargs(n_blocks=...)
    "d_block": None,        # None = défaut rtdl (192) ; 128 pour accélérer
    "attention_dropout": None,  # None = défaut rtdl (0.2)
    "fold_time_budget_s": 450,  # garde-fou : plus d'époque démarrée au-delà (meilleur état conservé)
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


# ------------------------------------------------------------------------------------ modèle
class FTTransformerAdapter(nn.Module):
    """Adaptateur : (x_cat, x_num) comme dans dl_base -> FTTransformer(x_cont, x_cat) -> logits [batch]."""

    def __init__(self, cardinalities, n_num, cfg):
        super().__init__()
        kwargs = FTTransformer.get_default_kwargs(n_blocks=cfg["n_blocks"])
        for key in ("d_block", "attention_dropout"):
            if cfg.get(key) is not None:
                kwargs[key] = cfg[key]
                kwargs["_is_default"] = False
        self.has_cat = len(cardinalities) > 0
        self.ft = FTTransformer(n_cont_features=n_num, cat_cardinalities=list(cardinalities), d_out=1, **kwargs)

    def forward(self, x_cat, x_num):
        return self.ft(x_num, x_cat if self.has_cat else None).squeeze(1)

    def make_parameter_groups(self):
        return self.ft.make_parameter_groups()


def build_model(cardinalities, n_num, cfg):
    """FT-Transformer : chaque colonne (catégorielle, quantile ou indicateur de manquant) devient un token.
    cardinalities : taille du vocabulaire de chaque colonne catégorielle (index 0 = manquant) ;
    n_num : nombre de colonnes numériques (indicateurs de valeur manquante inclus)."""
    return FTTransformerAdapter(cardinalities, n_num, cfg)


# ------------------------------------------------------------------------------ entraînement
def to_tensors(cat, num, y, device):
    tensors = [torch.from_numpy(cat).to(device), torch.from_numpy(num).to(device)]
    if y is not None:
        tensors.append(torch.from_numpy(y.astype(np.float32)).to(device))
    return tensors


@torch.no_grad()
def predict(model, xc, xn, batch_size=8192):
    model.eval()
    out = []
    for i in range(0, len(xc), batch_size):
        with torch.autocast(device_type=xc.device.type, dtype=torch.bfloat16, enabled=xc.is_cuda):
            logits = model(xc[i:i + batch_size], xn[i:i + batch_size])
        out.append(torch.sigmoid(logits.float()))
    return torch.cat(out).cpu().numpy().astype(np.float64)


def train_fold(model, tr, va, cfg, device, fold):
    xc_tr, xn_tr, y_tr = tr
    xc_va, xn_va, y_va = va
    model.to(device)
    params = model.make_parameter_groups() if hasattr(model, "make_parameter_groups") else model.parameters()
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(SEED + fold)
    y_np = y_va.cpu().numpy()
    best, best_state, bad, history = float("inf"), None, 0, []
    n, bs = len(xc_tr), cfg["batch_size"]
    t_fold = time.time()
    for epoch in range(cfg["epochs"]):
        if epoch and time.time() - t_fold > cfg["fold_time_budget_s"]:
            print(f"  fold {fold}: budget de temps atteint après {epoch} époques", flush=True)
            break
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        t0 = time.time()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(xc_tr[idx], xn_tr[idx])
            loss = loss_fn(logits.float(), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        p = predict(model, xc_va, xn_va)
        ll = log_loss(y_np, np.clip(p, 1e-7, 1 - 1e-7))
        auc = roc_auc_score(y_np, p)
        history.append({"epoch": epoch, "val_logloss": round(ll, 5), "val_auc": round(auc, 5),
                        "s": round(time.time() - t0, 1)})
        print(f"  fold {fold} époque {epoch}: logloss {ll:.5f} AUC {auc:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if ll < best - 1e-5:
            best, best_state, bad = ll, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return history


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
    parser.add_argument("--name", default="dl_ft_transformer")
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
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        tf = time.time()
        df_tr, df_va = train.iloc[tr_idx], train.iloc[va_idx]
        selected, gain = select_top_k(df_tr, y[tr_idx], feats, cfg["top_k"], args.threads)
        cats = [c for c in selected if c in cats_all]
        nums = [c for c in selected if c not in cats_all]
        pre = FoldPreprocessor(cats, nums).fit(df_tr)
        tr = to_tensors(*pre.transform(df_tr), y[tr_idx], device)
        va = to_tensors(*pre.transform(df_va), y[va_idx], device)
        te = to_tensors(*pre.transform(test), None, device)
        torch.manual_seed(SEED + fold)
        model = build_model(pre.cardinalities, tr[1].shape[1], cfg)
        if fold == 0:
            print(model)
            print("paramètres :", sum(p.numel() for p in model.parameters()))
        history = train_fold(model, tr, va, cfg, device, fold)
        oof[va_idx] = predict(model, va[0], va[1])
        test_proba += predict(model, te[0], te[1]) / args.folds
        auc = roc_auc_score(y[va_idx], oof[va_idx])
        print(f"fold {fold}: {len(history)} époques, AUC {auc:.4f}, {len(cats)} cat + {len(nums)} num, "
              f"{time.time() - tf:.0f} s", flush=True)
        folds_info.append({"selected": selected, "n_cat": len(cats), "n_num": len(nums),
                           "epochs": len(history), "auc": round(float(auc), 5), "history": history,
                           "top10_gain": {k: round(float(v)) for k, v in gain.head(10).items()}})
        del tr, va, te, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = evaluate(y, oof)
    save_outputs(args.name, test, oof, test_proba, metrics,
                 {"model": "dl", "config": cfg, "device": str(device), "nan_max": NAN_MAX,
                  "cat_max_levels": CAT_MAX_LEVELS, "folds": folds_info}, t0)


if __name__ == "__main__":
    main()
