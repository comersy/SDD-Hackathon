"""ACP (64 composantes) sur les 50 colonnes encodées, puis MLP 1024-512-256 sans embeddings, 3 graines.

Usage (Modal, GPU) :
    python -W ignore -m modal run modal_run.py --script experiments/mlp_pca_50.py --name mlp_pca_50 --threads 8 --gpu

Protocole identique à dl_base.py (même CV 5 folds seed 42, même seuil choisi sur l'OOF), mais :
1. PLUS de sélection top-K par fold : liste FIXE de 50 colonnes (features/selected_columns.json, embarquée
   dans COLUMNS) ; drop_sparse / clean / FoldPreprocessor conservés (fit sur la partie entraînement du fold) ;
2. dans chaque fold : codes des catégorielles -> one-hot (OneHotEncoder(handle_unknown="ignore")),
   numériques -> quantiles + indicateurs NaN ; matrice dense float32 -> PCA(n_components=64, random_state=42)
   ajustée sur 150 000 lignes de la partie entraînement du fold, transform de train / val / test ;
3. MLP sans embeddings (entrée = 64 composantes) hidden [1024, 512, 256], dropout 0.35, batch 1024,
   AdamW lr 1e-3 + OneCycleLR (pct_start 0.3), weight_decay 1e-4, 25 époques, early stopping patience 5
   sur la logloss de validation (meilleur état conservé) ;
4. moyennage de 3 graines par fold (torch.manual_seed(SEED + 100*s + fold)) : OOF du fold = moyenne,
   test += moyenne / n_folds ; le F1 OOF de la graine 0 seule est consigné dans le JSON.
Objectif : mesurer l'effet d'une ACP sur des codes de questionnaire (attendu : perte d'AUC).
La variance expliquée par les 64 composantes est consignée par fold.
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dl_base  # noqa: E402
from dl_base import CAT_MAX_LEVELS, NAN_MAX, FoldPreprocessor, drop_sparse  # noqa: E402
from r1_clean_codes import SEED, clean, evaluate, load, select_columns  # noqa: E402

IDEA = ("ACP 64 composantes sur les 50 colonnes encodées (one-hot des catégorielles + quantiles + indicateurs "
        "NaN, ajustée par fold) puis MLP 1024-512-256 sans embeddings, OneCycleLR, 3 graines moyennées : "
        "mesure la perte due à l'ACP sur des codes de questionnaire")
dl_base.IDEA = IDEA  # save_outputs() de dl_base lit IDEA dans son module

# Les 50 colonnes de features/selected_columns.json (clé "columns"), embarquées car le conteneur
# Modal ne contient pas le dossier features/.
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
PCA_COMPONENTS = 64
PCA_FIT_ROWS = 150_000
CONFIG = {
    "epochs": 25,
    "patience": 5,
    "batch_size": 1024,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "hidden": [1024, 512, 256],
    "dropout": 0.35,
    "pct_start": 0.3,
}


# ------------------------------------------------------------------------- encodage + ACP par fold
class FoldPCA:
    """One-hot des codes + numériques quantiles + indicateurs NaN -> matrice dense -> ACP.
    Ajustée sur la partie entraînement du fold uniquement ; n'utilise jamais TARGET."""

    def __init__(self, cats, nums, n_components, fit_rows, seed):
        self.pre = FoldPreprocessor(cats, nums)
        self.n_components, self.fit_rows, self.seed = n_components, fit_rows, seed

    def _dense(self, df):
        cat, num = self.pre.transform(df)
        parts = [num]
        if cat.shape[1]:
            parts.insert(0, self.ohe.transform(cat).astype(np.float32))
        return np.concatenate(parts, axis=1)

    def fit(self, df):
        self.pre.fit(df)
        cat, _ = self.pre.transform(df)
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        if cat.shape[1]:
            self.ohe.fit(cat)
        x = self._dense(df)
        self.n_input = x.shape[1]
        rng = np.random.RandomState(self.seed)
        sub = x[rng.choice(len(x), min(self.fit_rows, len(x)), replace=False)] if len(x) > self.fit_rows else x
        n_comp = min(self.n_components, sub.shape[1], len(sub))
        self.pca = PCA(n_components=n_comp, random_state=self.seed).fit(sub)
        self.explained = float(self.pca.explained_variance_ratio_.sum())
        return self

    def transform(self, df):
        return self.pca.transform(self._dense(df)).astype(np.float32)


# ------------------------------------------------------------------------------------ modèle
class PlainMLP(nn.Module):
    """MLP sans embeddings : entrée = composantes principales."""

    def __init__(self, n_in, cfg):
        super().__init__()
        d, layers = n_in, []
        for h in cfg["hidden"]:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(cfg["dropout"])]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x).squeeze(1)


def build_model(n_in, cfg):
    return PlainMLP(n_in, cfg)


# ------------------------------------------------------------------------------ entraînement
@torch.no_grad()
def predict(model, x, batch_size=16384):
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            logits = model(x[i:i + batch_size])
        out.append(torch.sigmoid(logits.float()))
    return torch.cat(out).cpu().numpy().astype(np.float64)


def train_one(model, x_tr, y_tr, x_va, y_va, cfg, device, fold, seed_idx):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n, bs = len(x_tr), cfg["batch_size"]
    steps_per_epoch = (n + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps_per_epoch,
                                                pct_start=cfg["pct_start"])
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(SEED + 100 * seed_idx + fold)
    y_np = y_va.cpu().numpy()
    best, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        t0, tot = time.time(), 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x_tr[idx])
            loss = loss_fn(logits.float(), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        p = predict(model, x_va)
        ll = log_loss(y_np, np.clip(p, 1e-7, 1 - 1e-7))
        auc = roc_auc_score(y_np, p)
        history.append({"epoch": epoch, "train_logloss": round(tot / n, 5), "val_logloss": round(ll, 5),
                        "val_auc": round(auc, 5), "lr": round(sched.get_last_lr()[0], 7),
                        "s": round(time.time() - t0, 1)})
        print(f"  fold {fold} graine {seed_idx} époque {epoch}: train {tot / n:.5f} logloss {ll:.5f} "
              f"AUC {auc:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if ll < best - 1e-5:
            best, best_state, bad = ll, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="mlp_pca_50")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=None, help="remplace CONFIG['epochs'] (test local)")
    parser.add_argument("--max-rows", type=int, default=None, help="sous-échantillon de train (test local)")
    args = parser.parse_args()
    cfg = dict(CONFIG)
    if args.epochs:
        cfg["epochs"] = args.epochs
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    t0 = time.time()

    train, test = load()
    if args.max_rows:
        train = train.sample(args.max_rows, random_state=SEED).reset_index(drop=True)
    feats_all = drop_sparse(train, select_columns(train))
    maxes = train[feats_all].max()
    clean(train, feats_all, maxes)
    clean(test, feats_all, maxes)
    missing = [c for c in COLUMNS if c not in feats_all]
    feats = [c for c in COLUMNS if c in feats_all]
    print(f"{len(feats)} colonnes fixes utilisées" + (f", absentes : {missing}" if missing else ""))
    y = train["TARGET"].values.astype(int)
    nunique = train[feats].nunique()
    cats = [c for c in feats if nunique[c] <= CAT_MAX_LEVELS]
    nums = [c for c in feats if nunique[c] > CAT_MAX_LEVELS]
    print(f"{len(cats)} catégorielles (one-hot) + {len(nums)} numériques (quantiles + indicateur NaN) "
          f"-> ACP {PCA_COMPONENTS} composantes")

    oof = np.zeros(len(train))
    oof_seed0 = np.zeros(len(train))
    test_proba = np.zeros(len(test))
    folds_info = []
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        tf = time.time()
        df_tr, df_va = train.iloc[tr_idx], train.iloc[va_idx]
        enc = FoldPCA(cats, nums, PCA_COMPONENTS, PCA_FIT_ROWS, SEED).fit(df_tr)
        x_tr = torch.from_numpy(enc.transform(df_tr)).to(device)
        x_va = torch.from_numpy(enc.transform(df_va)).to(device)
        x_te = torch.from_numpy(enc.transform(test)).to(device)
        y_tr = torch.from_numpy(y[tr_idx].astype(np.float32)).to(device)
        y_va = torch.from_numpy(y[va_idx].astype(np.float32)).to(device)
        print(f"fold {fold}: {enc.n_input} colonnes encodées -> {x_tr.shape[1]} composantes, "
              f"variance expliquée {enc.explained:.4f} ({time.time() - tf:.0f} s)", flush=True)
        va_pred, te_pred, seeds_info = [], [], []
        for s in range(N_SEEDS):
            torch.manual_seed(SEED + 100 * s + fold)
            model = build_model(x_tr.shape[1], cfg)
            if fold == 0 and s == 0:
                print(model)
                print("paramètres :", sum(p.numel() for p in model.parameters()))
            history = train_one(model, x_tr, y_tr, x_va, y_va, cfg, device, fold, s)
            p_va = predict(model, x_va)
            va_pred.append(p_va)
            te_pred.append(predict(model, x_te))
            best = min(history, key=lambda h: h["val_logloss"])
            seeds_info.append({"seed": s, "epochs": len(history), "best_epoch": best["epoch"],
                               "auc": round(float(roc_auc_score(y[va_idx], p_va)), 5),
                               "best_val_logloss": best["val_logloss"], "train_logloss_at_best": best["train_logloss"],
                               "history": history})
            del model
        oof[va_idx] = np.mean(va_pred, axis=0)
        oof_seed0[va_idx] = va_pred[0]
        test_proba += np.mean(te_pred, axis=0) / args.folds
        auc = roc_auc_score(y[va_idx], oof[va_idx])
        print(f"fold {fold}: AUC moyenne des {N_SEEDS} graines {auc:.4f} "
              f"(graines : {[si['auc'] for si in seeds_info]}), {time.time() - tf:.0f} s", flush=True)
        folds_info.append({"n_input_encoded": enc.n_input, "n_components": int(x_tr.shape[1]),
                           "explained_variance_ratio": round(enc.explained, 5),
                           "n_cat": len(cats), "n_num": len(nums), "auc": round(float(auc), 5),
                           "seeds": seeds_info})
        del x_tr, x_va, x_te, y_tr, y_va
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = evaluate(y, oof)
    metrics_seed0 = evaluate(y, oof_seed0)
    print("CV graine 0 seule :", {k: round(v, 4) for k, v in metrics_seed0.items()})
    dl_base.save_outputs(args.name, test, oof, test_proba, metrics,
                         {"model": "dl", "config": cfg, "n_seeds": N_SEEDS, "columns": feats,
                          "pca_components": PCA_COMPONENTS, "pca_fit_rows": PCA_FIT_ROWS,
                          "explained_variance_mean": round(float(np.mean([f["explained_variance_ratio"]
                                                                          for f in folds_info])), 5),
                          "cv_seed0": metrics_seed0, "single_seed_f1": metrics_seed0["f1"],
                          "device": str(device), "nan_max": NAN_MAX, "cat_max_levels": CAT_MAX_LEVELS,
                          "folds": folds_info}, t0)


if __name__ == "__main__":
    main()
