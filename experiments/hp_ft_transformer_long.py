"""Réglage : FT-Transformer entraîné « complet » (budget par fold élargi) avec lr réglé et planifié.

Copie de experiments/dl_ft_transformer.py (base dl_base : top-80 colonnes par fold, FT-Transformer rtdl
n_blocks 3, d_block 192). Ce qui change :
- fold_time_budget_s 450 -> 900 : les folds 0 et 2 de dl_ft_transformer avaient été coupés à 15 époques
  alors que la logloss de validation baissait encore ;
- epochs 25 -> 30, patience 4 -> 5, lr 1e-4 -> 1.5e-4 avec planification par pas : warmup linéaire sur
  2 époques puis cosinus jusqu'à lr * schedule_min_ratio ; batch 1024, weight_decay 1e-5
  (make_parameter_groups), 1 graine ;
- garde-fou GLOBAL (total_budget_s, 44 min) imposé par la limite Modal (timeout dur 60 min) : 5 x 900 s
  dépasserait l'heure ; le budget effectif d'un fold = min(fold_time_budget_s, temps restant / folds
  restants) et la longueur du cosinus est adaptée au nombre d'époques planifiables dans ce budget
  (estimation du temps par époque, mesurée dès le fold 0), pour que le lr soit bien recuit à la fin.

Usage (Modal, GPU) :
    python -W ignore -m modal run modal_run.py --script experiments/hp_ft_transformer_long.py --name hp_ft_transformer_long --threads 8 --gpu

Protocole identique à train.py (même CV 5 folds seed 42, même seuil choisi sur l'OOF). Dans chaque fold :
1. sélection : LightGBM rapide ajusté sur la partie entraînement du fold -> top-K colonnes par gain ;
2. prétraitement ajusté sur la partie entraînement du fold (sans TARGET) : colonnes à <= 15 modalités
   -> codes entiers (0 = manquant / inconnu) pour des embeddings ; autres -> QuantileTransformer (normale),
   NaN -> 0 + indicateur de valeur manquante ;
3. FT-Transformer, AdamW + warmup/cosinus + early stopping sur la logloss de validation (AMP bfloat16).
   OOF = meilleur état du fold ; test = moyenne des folds.
En plus du JSON et de la soumission, les probabilités OOF et test sont écrites dans results/
(<name>_<stamp>_oof.csv et <name>_<stamp>_test.csv) pour les blends ultérieurs.
"""
import argparse
import copy
import json
import math
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

IDEA = ("FT-Transformer (rtdl, 3 blocs, d_block 192) top-80 colonnes par fold, entraînement long : budget "
        "900 s/fold (plafonné par un garde-fou global de 44 min), 30 époques max, patience 5, AdamW lr 1.5e-4 "
        "avec warmup 2 époques puis cosinus adapté aux époques planifiables, batch 1024, wd 1e-5, 1 graine")
NAN_MAX = 0.95          # colonnes vides à plus de 95 % : retirées d'office (critère sans TARGET)
CAT_MAX_LEVELS = 15     # <= 15 modalités distinctes -> catégorielle (embedding)
CONFIG = {
    "top_k": 80,            # colonnes gardées par fold
    "epochs": 30,
    "patience": 5,          # époques sans amélioration de la logloss de validation
    "batch_size": 1024,
    "lr": 1.5e-4,
    "weight_decay": 1e-5,
    "warmup_epochs": 2,     # warmup linéaire (par pas) puis cosinus jusqu'à lr * schedule_min_ratio
    "schedule_min_ratio": 0.01,
    "n_blocks": 3,          # FTTransformer.get_default_kwargs(n_blocks=...)
    "d_block": None,        # None = défaut rtdl (192)
    "attention_dropout": None,  # None = défaut rtdl (0.2)
    "fold_time_budget_s": 900,  # garde-fou par fold : plus d'époque démarrée au-delà (meilleur état conservé)
    "total_budget_s": 2640,     # garde-fou global (44 min) : timeout Modal dur = 60 min
    "reserve_s": 45,            # temps gardé pour l'évaluation et l'écriture des sorties
    "epoch_s_est": 34.0,        # estimation initiale du temps par époque (A10, 32 s mesurés) ; mise à jour
    "min_sched_epochs": 8,      # longueur minimale du cosinus
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
    """FT-Transformer : chaque colonne (catégorielle, quantile ou indicateur de manquant) devient un token."""
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


def make_scheduler(opt, cfg, steps_per_epoch, sched_epochs):
    """Warmup linéaire sur warmup_epochs (par pas) puis cosinus jusqu'à lr * schedule_min_ratio
    à la fin de sched_epochs ; constant au minimum au-delà."""
    warm = max(1, int(cfg["warmup_epochs"] * steps_per_epoch))
    total = max(warm + 1, int(sched_epochs * steps_per_epoch))
    floor = cfg["schedule_min_ratio"]

    def f(step):
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total - warm))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def train_fold(model, tr, va, cfg, device, fold, fold_budget_s, sched_epochs):
    xc_tr, xn_tr, y_tr = tr
    xc_va, xn_va, y_va = va
    model.to(device)
    params = model.make_parameter_groups() if hasattr(model, "make_parameter_groups") else model.parameters()
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n, bs = len(xc_tr), cfg["batch_size"]
    steps_per_epoch = math.ceil(n / bs)
    sched = make_scheduler(opt, cfg, steps_per_epoch, sched_epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(SEED + fold)
    y_np = y_va.cpu().numpy()
    best, best_state, bad, history = float("inf"), None, 0, []
    t_fold = time.time()
    max_epochs = min(cfg["epochs"], sched_epochs)
    for epoch in range(max_epochs):
        if epoch and time.time() - t_fold > fold_budget_s:
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
            sched.step()
        p = predict(model, xc_va, xn_va)
        ll = log_loss(y_np, np.clip(p, 1e-7, 1 - 1e-7))
        auc = roc_auc_score(y_np, p)
        lr_now = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "val_logloss": round(ll, 5), "val_auc": round(auc, 5),
                        "lr": float(f"{lr_now:.3e}"), "s": round(time.time() - t0, 1)})
        print(f"  fold {fold} époque {epoch}: logloss {ll:.5f} AUC {auc:.4f} lr {lr_now:.2e} "
              f"({time.time() - t0:.0f} s)", flush=True)
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
    parser.add_argument("--name", default="hp_ft_transformer_long")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=None, help="remplace CONFIG['top_k']")
    parser.add_argument("--max-rows", type=int, default=None, help="test local : sous-échantillon")
    parser.add_argument("--epochs", type=int, default=None, help="test local : remplace CONFIG['epochs']")
    args = parser.parse_args()
    cfg = dict(CONFIG)
    if args.top_k:
        cfg["top_k"] = args.top_k
    if args.epochs:
        cfg["epochs"] = args.epochs
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    t0 = time.time()

    train, test = load()
    if args.max_rows:
        train = train.sample(args.max_rows, random_state=SEED).reset_index(drop=True)
        test = test.head(args.max_rows).reset_index(drop=True)
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
    epoch_s_est = cfg["epoch_s_est"]
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
        # Budget effectif du fold : garde-fou par fold ET garde-fou global (temps restant / folds restants)
        remaining = cfg["total_budget_s"] - cfg["reserve_s"] - (time.time() - t0)
        fold_budget = min(cfg["fold_time_budget_s"], remaining / (args.folds - fold))
        sched_epochs = int(max(cfg["min_sched_epochs"], min(cfg["epochs"], (fold_budget - 5) / epoch_s_est)))
        print(f"fold {fold}: budget {fold_budget:.0f} s, {sched_epochs} époques planifiées "
              f"(estimation {epoch_s_est:.1f} s/époque)", flush=True)
        history = train_fold(model, tr, va, cfg, device, fold, fold_budget, sched_epochs)
        measured = float(np.mean([h["s"] for h in history])) * 1.03
        epoch_s_est = measured if fold == 0 else max(epoch_s_est, measured)
        oof[va_idx] = predict(model, va[0], va[1])
        test_proba += predict(model, te[0], te[1]) / args.folds
        auc = roc_auc_score(y[va_idx], oof[va_idx])
        print(f"fold {fold}: {len(history)} époques, AUC {auc:.4f}, {len(cats)} cat + {len(nums)} num, "
              f"{time.time() - tf:.0f} s (écoulé {time.time() - t0:.0f} s)", flush=True)
        folds_info.append({"selected": selected, "n_cat": len(cats), "n_num": len(nums),
                           "epochs": len(history), "sched_epochs": sched_epochs,
                           "fold_budget_s": round(fold_budget), "auc": round(float(auc), 5),
                           "best_logloss": min(h["val_logloss"] for h in history), "history": history,
                           "top10_gain": {k: round(float(v)) for k, v in gain.head(10).items()}})
        del tr, va, te, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = evaluate(y, oof)
    trial = {"params": {k: cfg[k] for k in ("epochs", "patience", "batch_size", "lr", "weight_decay",
                                            "warmup_epochs", "schedule_min_ratio", "fold_time_budget_s",
                                            "total_budget_s", "n_blocks", "d_block", "top_k")},
             "auc_oof": round(metrics["auc"], 5), "f1_oof": round(metrics["f1"], 5),
             "epochs_per_fold": [f["epochs"] for f in folds_info]}
    save_outputs(args.name, test, oof, test_proba, metrics,
                 {"model": "dl", "config": cfg, "device": str(device), "nan_max": NAN_MAX,
                  "cat_max_levels": CAT_MAX_LEVELS, "n_trials": 1, "trials": [trial], "best_params": trial["params"],
                  "reference": {"name": "dl_ft_transformer_20261020_085906", "f1": 0.42037, "auc": 0.85320},
                  "folds": folds_info}, t0)


if __name__ == "__main__":
    main()
