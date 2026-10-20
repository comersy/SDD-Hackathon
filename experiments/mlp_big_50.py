"""Gros MLP (1024-512-256) sur les 50 colonnes sélectionnées, 3 graines moyennées par fold, OneCycleLR.

Usage (Modal, GPU) :
    python -W ignore -m modal run modal_run.py --script experiments/mlp_big_50.py --name mlp_big_50 --threads 8 --gpu
Test minuscule en local (CPU) : python experiments/mlp_big_50.py --quick 2000

Copie adaptée de dl_base.py (même protocole : CV 5 folds seed 42, seuil choisi sur l'OOF) :
1. PLUS de sélection top-K par fold : liste FIXE de 50 colonnes (COLUMNS, copiée depuis
   features/selected_columns.json, clé "columns"), filtrée par drop_sparse (critère sans TARGET) ;
2. prétraitement FoldPreprocessor inchangé, ajusté sur la partie entraînement du fold ;
3. MLP 1024-512-256 (BatchNorm + ReLU + dropout 0.35, embeddings 16), AdamW + OneCycleLR
   (max_lr = lr, pct_start 0.3, total_steps = epochs x batches), early stopping sur la logloss de
   validation (patience 5) en conservant le meilleur état ;
4. MOYENNAGE DE GRAINES : N_SEEDS = 3 réseaux par fold (torch.manual_seed(SEED + 100*s + fold)),
   OOF du fold = moyenne des 3, test += moyenne des 3 / n_folds. Le F1 OOF de la graine 0 seule est
   consigné dans le JSON (single_seed_cv) pour mesurer l'apport du moyennage.
Sorties : JSON, soumission, et probabilités OOF / test dans results/ (save_outputs de dl_base).
"""
import argparse
import copy
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dl_base  # noqa: E402
from dl_base import CAT_MAX_LEVELS, NAN_MAX, FoldPreprocessor, drop_sparse, predict, to_tensors  # noqa: E402
from r1_clean_codes import SEED, clean, evaluate, load  # noqa: E402

IDEA = ("Gros MLP 1024-512-256 (BatchNorm + ReLU, dropout 0.35, embeddings 16) sur les 50 colonnes "
        "sélectionnées (liste fixe, sans top-K par fold), AdamW + OneCycleLR, early stopping logloss, "
        "3 graines moyennées par fold")
dl_base.IDEA = IDEA  # save_outputs de dl_base lit dl_base.IDEA

# features/selected_columns.json, clé "columns" (n = 50)
COLUMNS = [
    "SEXVAR", "GENHLTH", "PERSDOC3", "MEDCOST1", "CVDSTRK3", "CHCSCNC1", "CHCOCNC1", "CHCCOPD3",
    "ADDEPEV3", "CHCKDNY2", "HAVARTH4", "DIABETE4", "DIABAGE4", "NUMHHOL4", "DEAF", "BLIND",
    "DIFFWALK", "DIFFDRES", "HADHYST2", "_CRCREC2", "LCSCTSC1", "LCSSCNCR", "CNCRTYP2", "_RACE1",
    "_ALTETH3", "_AGE80", "_PACKYRS", "EMPLOY1", "_BMI5", "_MAM5023", "WEIGHT2", "PHYSHLTH",
    "_YRSSMOK", "_STATE", "SLEPTIM1", "INCOME3", "POORHLTH", "CHECKUP1", "LASTDEN4", "PRIMINSR",
    "HEIGHT3", "FLSHTMY3", "MENTHLTH", "MARITAL", "HIVTSTD3", "PNEUVAC4", "ALCDAY4", "TETANUS1",
    "COVIDFS1", "HHADULT",
]
N_SEEDS = 3
CONFIG = {
    "epochs": 25,
    "patience": 5,
    "batch_size": 1024,
    "lr": 1e-3,             # max_lr du OneCycleLR
    "pct_start": 0.3,
    "weight_decay": 1e-4,
    "emb_dim": 16,
    "hidden": [1024, 512, 256],
    "dropout": 0.35,
    "n_seeds": N_SEEDS,
}
TRAIN_EVAL_ROWS = 60_000  # sous-échantillon de la partie entraînement pour mesurer l'écart train/val


# ------------------------------------------------------------------------------------ modèle
class BigMLP(nn.Module):
    def __init__(self, cardinalities, n_num, cfg):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(card, cfg["emb_dim"]) for card in cardinalities])
        d = len(cardinalities) * cfg["emb_dim"] + n_num
        layers = []
        for h in cfg["hidden"]:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(cfg["dropout"])]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        parts = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)] + [x_num]
        return self.mlp(torch.cat(parts, dim=1)).squeeze(1)


def build_model(cardinalities, n_num, cfg):
    return BigMLP(cardinalities, n_num, cfg)


# ------------------------------------------------------------------------------ entraînement
def train_one(model, tr, va, cfg, device, fold, seed):
    """AdamW + OneCycleLR (pas par batch), early stopping sur la logloss de validation (meilleur état)."""
    xc_tr, xn_tr, y_tr = tr
    xc_va, xn_va, y_va = va
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n, bs = len(xc_tr), cfg["batch_size"]
    batches = (n + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * batches,
                                                pct_start=cfg["pct_start"])
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    y_np = y_va.cpu().numpy()
    best, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(cfg["epochs"]):
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
        history.append({"epoch": epoch, "val_logloss": round(ll, 5), "val_auc": round(auc, 5),
                        "lr": round(sched.get_last_lr()[0], 7), "s": round(time.time() - t0, 1)})
        print(f"  fold {fold} graine {seed} époque {epoch}: logloss {ll:.5f} AUC {auc:.4f} "
              f"lr {sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f} s)", flush=True)
        if ll < best - 1e-5:
            best, best_state, bad = ll, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return history


def train_gap(model, tr, va, gen):
    """Écart train/val du meilleur état : logloss et AUC sur un sous-échantillon de la partie entraînement."""
    xc_tr, xn_tr, y_tr = tr
    n = len(xc_tr)
    idx = torch.randperm(n, generator=gen)[:TRAIN_EVAL_ROWS].to(xc_tr.device)
    p_tr = predict(model, xc_tr[idx], xn_tr[idx])
    y_s = y_tr[idx].cpu().numpy()
    p_va = predict(model, va[0], va[1])
    y_v = va[2].cpu().numpy()
    return {"train_logloss": round(float(log_loss(y_s, np.clip(p_tr, 1e-7, 1 - 1e-7))), 5),
            "train_auc": round(float(roc_auc_score(y_s, p_tr)), 5),
            "val_logloss": round(float(log_loss(y_v, np.clip(p_va, 1e-7, 1 - 1e-7))), 5),
            "val_auc": round(float(roc_auc_score(y_v, p_va)), 5)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="mlp_big_50")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--quick", type=int, default=0,
                        help="test local minuscule : N lignes, 1 époque, 1 graine (jamais pour un vrai run)")
    args = parser.parse_args()
    cfg = dict(CONFIG)
    n_seeds = N_SEEDS
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    t0 = time.time()

    train, test = load()
    if args.quick:
        train = train.sample(args.quick, random_state=SEED).reset_index(drop=True)
        test = test.head(args.quick).reset_index(drop=True)
        cfg["epochs"], n_seeds = 1, 1
        dl_base.ROOT = Path(tempfile.mkdtemp(prefix="mlp_big_50_quick_"))  # ne pollue pas results/
        print(f"MODE QUICK : {len(train)} lignes, 1 époque, 1 graine (résultats non comparables), "
              f"sorties dans {dl_base.ROOT}")
    missing = [c for c in COLUMNS if c not in train.columns or c not in test.columns]
    assert not missing, f"colonnes absentes : {missing}"
    feats = drop_sparse(train, list(COLUMNS))
    maxes = train[feats].max()
    clean(train, feats, maxes)
    clean(test, feats, maxes)
    y = train["TARGET"].values.astype(int)
    nunique = train[feats].nunique()
    cats = [c for c in feats if nunique[c] <= CAT_MAX_LEVELS]
    nums = [c for c in feats if nunique[c] > CAT_MAX_LEVELS]
    print(f"{len(feats)} colonnes fixes dont {len(cats)} catégorielles et {len(nums)} numériques ; "
          f"{n_seeds} graines par fold")

    oof = np.zeros(len(train))
    oof_seed0 = np.zeros(len(train))
    test_proba = np.zeros(len(test))
    folds_info = []
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        tf = time.time()
        df_tr, df_va = train.iloc[tr_idx], train.iloc[va_idx]
        pre = FoldPreprocessor(cats, nums).fit(df_tr)
        tr = to_tensors(*pre.transform(df_tr), y[tr_idx], device)
        va = to_tensors(*pre.transform(df_va), y[va_idx], device)
        te = to_tensors(*pre.transform(test), None, device)
        oof_fold = np.zeros(len(va_idx))
        test_fold = np.zeros(len(test))
        seeds_info = []
        for s in range(n_seeds):
            seed = SEED + 100 * s + fold
            torch.manual_seed(seed)
            model = build_model(pre.cardinalities, tr[1].shape[1], cfg)
            if fold == 0 and s == 0:
                print(model)
                print("paramètres :", sum(p.numel() for p in model.parameters()))
            history = train_one(model, tr, va, cfg, device, fold, seed)
            p_va = predict(model, va[0], va[1])
            p_te = predict(model, te[0], te[1])
            if s == 0:
                oof_seed0[va_idx] = p_va
            oof_fold += p_va / n_seeds
            test_fold += p_te / n_seeds
            gap = train_gap(model, tr, va, torch.Generator().manual_seed(seed))
            auc_s = roc_auc_score(y[va_idx], p_va)
            best_epoch = int(np.argmin([h["val_logloss"] for h in history]))
            print(f"  fold {fold} graine {seed}: {len(history)} époques (meilleure {best_epoch}), AUC {auc_s:.4f}, "
                  f"train/val logloss {gap['train_logloss']:.4f}/{gap['val_logloss']:.4f} "
                  f"AUC {gap['train_auc']:.4f}/{gap['val_auc']:.4f}", flush=True)
            seeds_info.append({"seed": seed, "epochs": len(history), "best_epoch": best_epoch,
                               "auc": round(float(auc_s), 5), "gap": gap, "history": history})
            del model
        oof[va_idx] = oof_fold
        test_proba += test_fold / args.folds
        auc = roc_auc_score(y[va_idx], oof_fold)
        auc0 = roc_auc_score(y[va_idx], oof_seed0[va_idx])
        print(f"fold {fold}: AUC moyenne des {n_seeds} graines {auc:.4f} (graine 0 seule {auc0:.4f}), "
              f"{len(cats)} cat + {len(nums)} num, {time.time() - tf:.0f} s", flush=True)
        folds_info.append({"n_cat": len(cats), "n_num": len(nums), "auc": round(float(auc), 5),
                           "auc_seed0": round(float(auc0), 5), "epochs": [si["epochs"] for si in seeds_info],
                           "seconds": round(time.time() - tf, 1), "seeds": seeds_info})
        del tr, va, te
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = evaluate(y, oof)
    metrics_seed0 = evaluate(y, oof_seed0)
    print("CV graine 0 seule :", {k: round(v, 4) for k, v in metrics_seed0.items()})
    gaps = [si["gap"] for f in folds_info for si in f["seeds"]]
    mean_gap = {k: round(float(np.mean([g[k] for g in gaps])), 5) for k in gaps[0]}
    dl_base.save_outputs(args.name, test, oof, test_proba, metrics,
                         {"model": "dl", "architecture": "MLP 1024-512-256 BatchNorm+ReLU", "config": cfg,
                          "n_seeds": n_seeds, "single_seed_cv": metrics_seed0, "single_seed_f1": metrics_seed0["f1"],
                          "columns": feats, "n_features": len(feats), "n_cat": len(cats), "n_num": len(nums),
                          "mean_train_val_gap": mean_gap, "device": str(device), "nan_max": NAN_MAX,
                          "cat_max_levels": CAT_MAX_LEVELS, "quick": args.quick, "folds": folds_info}, t0)


if __name__ == "__main__":
    main()
