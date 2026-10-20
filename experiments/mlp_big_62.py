"""Gros MLP (1024-512-256) sur une liste FIXE de 62 colonnes (50 sélectionnées + 12 de la variante
étendue de features/selected_columns.json), 3 graines moyennées par fold, OneCycleLR.

Usage (Modal, GPU) :
    python -W ignore -m modal run modal_run.py --script experiments/mlp_big_62.py --name mlp_big_62 --threads 8 --gpu

Copie adaptée de dl_base.py. Différences :
1. plus de sélection top-K par fold : COLUMNS (constante ci-dessous) = les 62 colonnes ; drop_sparse /
   clean / FoldPreprocessor (ajusté sur la partie entraînement du fold) conservés ;
2. moyennage de graines : dans chaque fold, N_SEEDS réseaux (torch.manual_seed(SEED + 100*s + fold)),
   OOF du fold = moyenne des graines, test += moyenne des graines / n_folds ; le F1 OOF de la graine 0
   seule est consigné dans le JSON (cv_seed0) pour mesurer l'apport du moyennage ;
3. AdamW + OneCycleLR (max_lr = lr, pct_start 0.3), early stopping sur la logloss de validation
   (patience 5) en conservant le meilleur état ;
4. le reste (AMP bf16, save_outputs avec OOF/test CSV) inchangé.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_clean_codes import ROOT, SEED, clean, evaluate, load  # noqa: E402

IDEA = ("Gros MLP 1024-512-256 (dropout 0.35, emb 16) sur une liste fixe de 62 colonnes "
        "(50 sélectionnées + variante étendue de features/selected_columns.json), 3 graines moyennées "
        "par fold, AdamW + OneCycleLR, early stopping logloss (patience 5)")
NAN_MAX = 0.95          # colonnes vides à plus de 95 % : retirées d'office (critère sans TARGET)
CAT_MAX_LEVELS = 15     # <= 15 modalités distinctes -> catégorielle (embedding)
N_SEEDS = 3
CONFIG = {
    "epochs": 25,
    "patience": 5,          # époques sans amélioration de la logloss de validation
    "batch_size": 1024,
    "lr": 1e-3,             # max_lr du OneCycleLR
    "weight_decay": 1e-4,
    "emb_dim": 16,
    "hidden": [1024, 512, 256],
    "dropout": 0.35,
    "pct_start": 0.3,
    "n_seeds": N_SEEDS,
}

# features/selected_columns.json : "columns" (50) + "extended_variant.extra_columns" (12) = 62
COLUMNS = [
    "SEXVAR", "GENHLTH", "PERSDOC3", "MEDCOST1", "CVDSTRK3", "CHCSCNC1", "CHCOCNC1", "CHCCOPD3",
    "ADDEPEV3", "CHCKDNY2", "HAVARTH4", "DIABETE4", "DIABAGE4", "NUMHHOL4", "DEAF", "BLIND",
    "DIFFWALK", "DIFFDRES", "HADHYST2", "_CRCREC2", "LCSCTSC1", "LCSSCNCR", "CNCRTYP2", "_RACE1",
    "_ALTETH3", "_AGE80", "_PACKYRS", "EMPLOY1", "_BMI5", "_MAM5023", "WEIGHT2", "PHYSHLTH",
    "_YRSSMOK", "_STATE", "SLEPTIM1", "INCOME3", "POORHLTH", "CHECKUP1", "LASTDEN4", "PRIMINSR",
    "HEIGHT3", "FLSHTMY3", "MENTHLTH", "MARITAL", "HIVTSTD3", "PNEUVAC4", "ALCDAY4", "TETANUS1",
    "COVIDFS1", "HHADULT",
    # variante étendue (+12)
    "AVEDRNK3", "CRVCLCNC", "EMTSUPRT", "MAXDRNKS", "EDUCA", "COVIDSE1", "SDHSTRE1", "_MRACE2",
    "DRNK3GE5", "CNCRAGE", "RRCOGNT2", "_EXTETH3",
]
assert len(COLUMNS) == 62 and len(set(COLUMNS)) == 62


# ------------------------------------------------------------------ sélection et prétraitement
def drop_sparse(train, feats):
    nan_rate = train[feats].isna().mean()
    sparse = sorted(nan_rate[nan_rate > NAN_MAX].index)
    print(f"{len(sparse)} colonnes vides à plus de {NAN_MAX:.0%} retirées, {len(feats) - len(sparse)} candidates")
    return [c for c in feats if c not in sparse]


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
class MLP(nn.Module):
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
    return MLP(cardinalities, n_num, cfg)


# ------------------------------------------------------------------------------ entraînement
def to_tensors(cat, num, y, device):
    tensors = [torch.from_numpy(cat).to(device), torch.from_numpy(num).to(device)]
    if y is not None:
        tensors.append(torch.from_numpy(y.astype(np.float32)).to(device))
    return tensors


@torch.no_grad()
def predict(model, xc, xn, batch_size=16384):
    model.eval()
    out = []
    for i in range(0, len(xc), batch_size):
        with torch.autocast(device_type=xc.device.type, dtype=torch.bfloat16, enabled=xc.is_cuda):
            logits = model(xc[i:i + batch_size], xn[i:i + batch_size])
        out.append(torch.sigmoid(logits.float()))
    return torch.cat(out).cpu().numpy().astype(np.float64)


def train_fold(model, tr, va, cfg, device, fold, seed_idx):
    xc_tr, xn_tr, y_tr = tr
    xc_va, xn_va, y_va = va
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n, bs = len(xc_tr), cfg["batch_size"]
    batches = (n + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * batches,
                                                pct_start=cfg["pct_start"])
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(SEED + 100 * seed_idx + fold)
    y_np = y_va.cpu().numpy()
    best, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        t0 = time.time()
        tr_loss, tr_n = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(xc_tr[idx], xn_tr[idx])
            loss = loss_fn(logits.float(), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tr_loss += float(loss.detach()) * len(idx)
            tr_n += len(idx)
        p = predict(model, xc_va, xn_va)
        ll = log_loss(y_np, np.clip(p, 1e-7, 1 - 1e-7))
        auc = roc_auc_score(y_np, p)
        history.append({"epoch": epoch, "train_logloss": round(tr_loss / tr_n, 5), "val_logloss": round(ll, 5),
                        "val_auc": round(auc, 5), "lr": round(sched.get_last_lr()[0], 6),
                        "s": round(time.time() - t0, 1)})
        print(f"  fold {fold} graine {seed_idx} époque {epoch}: train {tr_loss / tr_n:.5f} val {ll:.5f} "
              f"AUC {auc:.4f} ({time.time() - t0:.0f} s)", flush=True)
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
    parser.add_argument("--name", default="mlp_big_62")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--nrows", type=int, default=None, help="test local minuscule : sous-échantillon")
    parser.add_argument("--epochs", type=int, default=None, help="remplace CONFIG['epochs'] (test local)")
    args = parser.parse_args()
    cfg = dict(CONFIG)
    if args.epochs:
        cfg["epochs"] = args.epochs
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    t0 = time.time()

    train, test = load()
    if args.nrows:
        train = train.sample(args.nrows, random_state=SEED).reset_index(drop=True)
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
          f"{N_SEEDS} graines par fold")

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
        fold_oof = np.zeros(len(va_idx))
        fold_test = np.zeros(len(test))
        seeds_info = []
        for s in range(N_SEEDS):
            torch.manual_seed(SEED + 100 * s + fold)
            model = build_model(pre.cardinalities, tr[1].shape[1], cfg)
            if fold == 0 and s == 0:
                print(model)
                print("paramètres :", sum(p.numel() for p in model.parameters()))
            history = train_fold(model, tr, va, cfg, device, fold, s)
            p_va = predict(model, va[0], va[1])
            p_te = predict(model, te[0], te[1])
            if s == 0:
                oof_seed0[va_idx] = p_va
            fold_oof += p_va / N_SEEDS
            fold_test += p_te / N_SEEDS
            best_ep = min(history, key=lambda h: h["val_logloss"])
            seeds_info.append({"seed": s, "epochs": len(history), "best_epoch": best_ep["epoch"],
                               "best_val_logloss": best_ep["val_logloss"], "best_train_logloss": best_ep["train_logloss"],
                               "auc": round(float(roc_auc_score(y[va_idx], p_va)), 5), "history": history})
            print(f"  fold {fold} graine {s}: {len(history)} époques, AUC {seeds_info[-1]['auc']:.4f}", flush=True)
            del model
        oof[va_idx] = fold_oof
        test_proba += fold_test / args.folds
        auc = roc_auc_score(y[va_idx], fold_oof)
        print(f"fold {fold}: AUC moyenne des graines {auc:.4f}, {len(cats)} cat + {len(nums)} num, "
              f"{time.time() - tf:.0f} s", flush=True)
        folds_info.append({"n_cat": len(cats), "n_num": len(nums), "auc": round(float(auc), 5),
                           "seeds": seeds_info, "s": round(time.time() - tf, 1)})
        del tr, va, te
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = evaluate(y, oof)
    metrics_seed0 = evaluate(y, oof_seed0)
    print("CV graine 0 seule :", {k: round(v, 4) for k, v in metrics_seed0.items()})
    save_outputs(args.name, test, oof, test_proba, metrics,
                 {"model": "dl", "config": cfg, "device": str(device), "nan_max": NAN_MAX,
                  "cat_max_levels": CAT_MAX_LEVELS, "n_seeds": N_SEEDS, "columns": feats,
                  "cv_seed0": metrics_seed0, "folds": folds_info}, t0)


if __name__ == "__main__":
    main()
