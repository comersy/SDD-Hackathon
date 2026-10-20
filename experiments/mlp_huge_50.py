"""MLP énorme (2048-1024-512-256) sur les 50 colonnes fixes, dropout 0.45, 3 graines moyennées par fold.

Usage (Modal, GPU) :
    python -W ignore -m modal run modal_run.py --script experiments/mlp_huge_50.py --name mlp_huge_50 --threads 8 --gpu

Copie adaptée de dl_base.py. Différences :
1. PLUS de sélection top-K par fold : liste FIXE de 50 colonnes (constante COLUMNS, copiée de
   features/selected_columns.json, clé "columns") ; drop_sparse / clean / FoldPreprocessor conservés
   (prétraitement ajusté sur la partie entraînement du fold uniquement) ;
2. moyennage de graines : dans chaque fold, N_SEEDS = 3 réseaux (torch.manual_seed(SEED + 100*s + fold)),
   OOF du fold = moyenne des 3, test += moyenne des 3 / n_folds ; le F1 OOF de la graine 0 seule est
   consigné dans le JSON (cv_seed0) pour mesurer l'apport du moyennage ;
3. AdamW + OneCycleLR (max_lr = lr, total_steps = epochs x batches, pct_start 0.3), early stopping sur la
   logloss de validation (patience 5) en conservant le meilleur état ;
4. la logloss d'entraînement (mode eval, sous-échantillon fixe de 60 000 lignes de la partie entraînement)
   est suivie à chaque époque pour mesurer l'écart entraînement / validation (sur-apprentissage) ;
5. AMP bf16, save_outputs (JSON + OOF/test CSV + soumission) inchangés.
Question testée : la capacité (4 couches larges, ~5 M de paramètres) aide-t-elle sur 50 colonnes ?
Attendu : sur-apprentissage plus rapide, score <= mlp_big_50.
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

IDEA = ("MLP énorme 2048-1024-512-256 (dropout 0.45, emb 24) sur les 50 colonnes fixes de "
        "features/selected_columns.json, AdamW + OneCycleLR, early stopping patience 5, 3 graines moyennées "
        "par fold ; test : la capacité aide-t-elle ou sur-apprend-elle plus vite ?")
NAN_MAX = 0.95          # colonnes vides à plus de 95 % : retirées d'office (critère sans TARGET)
CAT_MAX_LEVELS = 15     # <= 15 modalités distinctes -> catégorielle (embedding)
N_SEEDS = 3
TRAIN_LL_ROWS = 60_000  # sous-échantillon fixe de la partie entraînement pour la logloss d'entraînement
CONFIG = {
    "epochs": 30,
    "patience": 5,          # époques sans amélioration de la logloss de validation
    "batch_size": 2048,
    "lr": 2e-3,             # max_lr du OneCycleLR
    "weight_decay": 3e-4,
    "emb_dim": 24,
    "hidden": [2048, 1024, 512, 256],
    "dropout": 0.45,
    "scheduler": "OneCycleLR",
    "pct_start": 0.3,
    "n_seeds": N_SEEDS,
}
# 50 colonnes de features/selected_columns.json (clé "columns") : consensus null importance + top-K stable,
# une colonne par groupe redondant, aucune colonne administrative / de date / de pondération.
COLUMNS = [
    "SEXVAR", "GENHLTH", "PERSDOC3", "MEDCOST1", "CVDSTRK3", "CHCSCNC1", "CHCOCNC1", "CHCCOPD3",
    "ADDEPEV3", "CHCKDNY2", "HAVARTH4", "DIABETE4", "DIABAGE4", "NUMHHOL4", "DEAF", "BLIND",
    "DIFFWALK", "DIFFDRES", "HADHYST2", "_CRCREC2", "LCSCTSC1", "LCSSCNCR", "CNCRTYP2", "_RACE1",
    "_ALTETH3", "_AGE80", "_PACKYRS", "EMPLOY1", "_BMI5", "_MAM5023", "WEIGHT2", "PHYSHLTH",
    "_YRSSMOK", "_STATE", "SLEPTIM1", "INCOME3", "POORHLTH", "CHECKUP1", "LASTDEN4", "PRIMINSR",
    "HEIGHT3", "FLSHTMY3", "MENTHLTH", "MARITAL", "HIVTSTD3", "PNEUVAC4", "ALCDAY4", "TETANUS1",
    "COVIDFS1", "HHADULT",
]
assert len(COLUMNS) == 50 and len(set(COLUMNS)) == 50


# ------------------------------------------------------------------ sélection et prétraitement
def drop_sparse(train, feats):
    nan_rate = train[feats].isna().mean()
    sparse = sorted(nan_rate[nan_rate > NAN_MAX].index)
    print(f"{len(sparse)} colonnes vides à plus de {NAN_MAX:.0%} retirées {sparse}, {len(feats) - len(sparse)} gardées")
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


def _ll(y, p):
    return log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])


def train_fold(model, tr, va, cfg, device, fold, seed_idx):
    xc_tr, xn_tr, y_tr = tr
    xc_va, xn_va, y_va = va
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    seed = SEED + 100 * seed_idx + fold
    gen = torch.Generator().manual_seed(seed)
    n, bs = len(xc_tr), cfg["batch_size"]
    batches = (n + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * batches,
                                                pct_start=cfg["pct_start"])
    # Sous-échantillon fixe de la partie entraînement pour la logloss d'entraînement (mode eval, sans dropout)
    sub = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:min(TRAIN_LL_ROWS, n)].to(device)
    xc_sub, xn_sub, y_sub = xc_tr[sub], xn_tr[sub], y_tr[sub].cpu().numpy()
    y_np = y_va.cpu().numpy()
    best, best_state, best_epoch, bad, history = float("inf"), None, -1, 0, []
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
        ll = _ll(y_np, p)
        auc = roc_auc_score(y_np, p)
        tr_ll = _ll(y_sub, predict(model, xc_sub, xn_sub))
        history.append({"epoch": epoch, "train_logloss": round(tr_ll, 5), "val_logloss": round(ll, 5),
                        "val_auc": round(auc, 5), "lr": round(sched.get_last_lr()[0], 6),
                        "s": round(time.time() - t0, 1)})
        print(f"  fold {fold} graine {seed_idx} époque {epoch}: train {tr_ll:.5f} val {ll:.5f} "
              f"(écart {ll - tr_ll:+.5f}) AUC {auc:.4f} lr {sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f} s)",
              flush=True)
        if ll < best - 1e-5:
            best, best_state, best_epoch, bad = ll, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    h = history[best_epoch]
    summary = {"epochs_run": len(history), "best_epoch": best_epoch, "best_val_logloss": h["val_logloss"],
               "train_logloss_at_best": h["train_logloss"], "gap_at_best": round(h["val_logloss"] - h["train_logloss"], 5),
               "gap_last": round(history[-1]["val_logloss"] - history[-1]["train_logloss"], 5),
               "stopped_early": len(history) < cfg["epochs"]}
    return history, summary


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
    parser.add_argument("--name", default="mlp_huge_50")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=None, help="remplace CONFIG['epochs'] (test local)")
    parser.add_argument("--seeds", type=int, default=None, help="remplace N_SEEDS (test local)")
    parser.add_argument("--debug-rows", type=int, default=0, help="sous-échantillon de lignes (test local)")
    args = parser.parse_args()
    cfg = dict(CONFIG)
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.seeds:
        cfg["n_seeds"] = args.seeds
    n_seeds = cfg["n_seeds"]
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    t0 = time.time()

    train, test = load()
    if args.debug_rows:
        train = train.sample(args.debug_rows, random_state=SEED).reset_index(drop=True)
        test = test.head(args.debug_rows).reset_index(drop=True)
    missing = [c for c in COLUMNS if c not in train.columns or c not in test.columns]
    assert not missing, f"colonnes absentes : {missing}"
    feats = drop_sparse(train, COLUMNS)
    maxes = train[feats].max()
    clean(train, feats, maxes)
    clean(test, feats, maxes)
    y = train["TARGET"].values.astype(int)
    nunique = train[feats].nunique()
    cats = [c for c in feats if nunique[c] <= CAT_MAX_LEVELS]
    nums = [c for c in feats if nunique[c] > CAT_MAX_LEVELS]
    print(f"{len(feats)} colonnes fixes : {len(cats)} catégorielles (embeddings) + {len(nums)} numériques ; "
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
        va_mean = np.zeros(len(va_idx))
        te_mean = np.zeros(len(test))
        seeds_info = []
        for s in range(n_seeds):
            ts = time.time()
            torch.manual_seed(SEED + 100 * s + fold)
            model = build_model(pre.cardinalities, tr[1].shape[1], cfg)
            if fold == 0 and s == 0:
                print(model)
                print("paramètres :", sum(p.numel() for p in model.parameters()))
            history, summary = train_fold(model, tr, va, cfg, device, fold, s)
            p_va = predict(model, va[0], va[1])
            if s == 0:
                oof_seed0[va_idx] = p_va
            va_mean += p_va / n_seeds
            te_mean += predict(model, te[0], te[1]) / n_seeds
            auc_s = roc_auc_score(y[va_idx], p_va)
            print(f" fold {fold} graine {s}: {summary['epochs_run']} époques (meilleure {summary['best_epoch']}), "
                  f"val logloss {summary['best_val_logloss']:.5f}, écart train/val {summary['gap_at_best']:+.5f}, "
                  f"AUC {auc_s:.4f}, {time.time() - ts:.0f} s", flush=True)
            seeds_info.append({"seed": s, "auc": round(float(auc_s), 5), **summary, "history": history})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        oof[va_idx] = va_mean
        test_proba += te_mean / args.folds
        auc = roc_auc_score(y[va_idx], va_mean)
        print(f"fold {fold}: AUC moyenne des {n_seeds} graines {auc:.4f} "
              f"(graine 0 seule {roc_auc_score(y[va_idx], oof_seed0[va_idx]):.4f}), {time.time() - tf:.0f} s", flush=True)
        folds_info.append({"n_cat": len(cats), "n_num": len(nums), "auc": round(float(auc), 5),
                           "epochs": [si["epochs_run"] for si in seeds_info],
                           "gap_at_best": [si["gap_at_best"] for si in seeds_info],
                           "seeds": seeds_info})
        del tr, va, te
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = evaluate(y, oof)
    metrics_seed0 = evaluate(y, oof_seed0)
    print("CV graine 0 seule :", {k: round(v, 4) for k, v in metrics_seed0.items()})
    all_epochs = [e for fi in folds_info for e in fi["epochs"]]
    all_gaps = [g for fi in folds_info for g in fi["gap_at_best"]]
    print(f"époques par réseau : {all_epochs} (moyenne {np.mean(all_epochs):.1f}) ; "
          f"écart train/val au meilleur état : moyenne {np.mean(all_gaps):+.5f}")
    save_outputs(args.name, test, oof, test_proba, metrics,
                 {"model": "dl", "cv_seed0": metrics_seed0, "columns": feats, "n_features": len(feats),
                  "config": cfg, "device": str(device), "nan_max": NAN_MAX, "cat_max_levels": CAT_MAX_LEVELS,
                  "mean_epochs": round(float(np.mean(all_epochs)), 2),
                  "mean_gap_at_best": round(float(np.mean(all_gaps)), 5), "folds": folds_info}, t0)


if __name__ == "__main__":
    main()
