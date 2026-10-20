"""Blend LightGBM seed-bagged (5 graines) + MLP PyTorch à embeddings (2 graines), base r2_lgbm_seed_bagging.

Usage : python experiments/r3_lgbbag_mlp_blend.py [--name NOM] [--threads N] [--folds 5]
Sorties : submissions/submission_<nom>_<horodatage>.csv et results/<nom>_<horodatage>.json

Protocole d'évaluation, à conserver à l'identique dans toute expérience pour que les scores
restent comparables :
- entraînement sur train.csv + extra.csv concaténés (370 132 lignes), test.csv sert
  uniquement à prédire ;
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42), prédictions out-of-fold ;
- tout prétraitement qui utilise TARGET est ajusté dans chaque fold sur la partie
  entraînement seulement ;
- le seuil de décision est choisi sur les prédictions OOF pour maximiser le F1 ;
  le F1 reporté est ce F1 OOF, au seuil retenu.
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from torch import nn

IDEA = ("Blend de deux familles de modèles : LightGBM seed-bagged (5 graines par fold, identique à "
        "r2_lgbm_seed_bagging) + MLP PyTorch GPU à embeddings (identique à r2_mlp_embeddings_gpu, "
        "2 graines 42 et 1 par fold, moyenne des sigmoid) ; p = w*LGB + (1-w)*MLP avec w dans "
        "{0.6, 0.7, 0.8} choisi sur l'OOF, même w pour le test ; seuil choisi sur l'OOF blendé ; "
        "StratifiedKFold inchangé (random_state=42)")
SEED = 42
SEEDS = [42, 1, 2, 3, 4]  # la première graine (42) reproduit exactement la base
MLP_SEEDS = [42, 1]  # la première graine (42) reproduit exactement r2_mlp_embeddings_gpu
BLEND_WEIGHTS = [0.6, 0.7, 0.8]  # poids du LightGBM (le MLP est plus faible : jamais sous 0.6)

# Hyperparamètres du MLP (recopiés de r2_mlp_embeddings_gpu)
CAT_MAX_DISTINCT = 15
EMB_DIM = 4
HIDDEN = (512, 256)
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 2048
MAX_EPOCHS = 20
PATIENCE = 3
EVAL_BATCH = 65536


def find_root():
    """Dossier du dépôt (celui qui contient train.csv), quel que soit l'emplacement du script."""
    here = Path(__file__).resolve().parent
    for d in [here, *here.parents]:
        if (d / "train.csv").exists():
            return d
    raise FileNotFoundError("train.csv introuvable : lance le script depuis le dépôt")


ROOT = find_root()

# Colonnes sans rapport avec la santé de la personne (cf. codebook BRFSS)
DATE_COLS = ["FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR"]
ADMIN_COLS = [
    "ID", "DISPCODE", "SEQNO", "_PSU", "QSTVER", "QSTLANG",
    # Écran d'appel (fixe / portable, logement, état de résidence)
    "CTELENM1", "PVTRESD1", "COLGHOUS", "STATERE1", "CELPHON1", "LADULT1", "COLGSEX1",
    "LANDSEX1", "RESPSLCT", "SAFETIME", "CTELNUM1", "CELLFON5", "CADULT1", "CELLSEX1",
    "PVTRESD3", "CCLGHOUS", "CSTATE1", "LANDLINE",
]
WEIGHT_COLS = [
    "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT",
    "_DUALUSE", "_DUALCOR", "_LLCPWT2", "_LLCPWT",
]
DROP_COLS = DATE_COLS + ADMIN_COLS + WEIGHT_COLS

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": SEED,
}


def _read_csv(path):
    df = pd.read_csv(path, low_memory=False)
    for c in df.columns:
        if c == "ID":
            df[c] = df[c].astype("int64")
        elif c == "TARGET":
            df[c] = df[c].astype("int8")
        else:
            df[c] = df[c].astype("float32")
    return df


def load():
    """train.csv + extra.csv concaténés, et test.csv. Cache parquet dans data/ (ignoré par git)."""
    cache = ROOT / "data"
    cache.mkdir(exist_ok=True)
    train_pq, test_pq = cache / "train_extra.parquet", cache / "test.parquet"
    if train_pq.exists() and test_pq.exists():
        return pd.read_parquet(train_pq), pd.read_parquet(test_pq)
    train = pd.concat([_read_csv(ROOT / "train.csv"), _read_csv(ROOT / "extra.csv")],
                      ignore_index=True)
    test = _read_csv(ROOT / "test.csv")
    for df, path in [(train, train_pq), (test, test_pq)]:
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp)
        os.replace(tmp, path)  # écriture atomique : plusieurs scripts peuvent tourner en même temps
    return train, test


def select_columns(train):
    feats = [c for c in train.columns if c not in DROP_COLS and c != "TARGET"]
    constant = [c for c in feats if train[c].nunique(dropna=False) <= 1]
    feats = [c for c in feats if c not in constant]
    print(f"{len(feats)} colonnes gardées, {len(DROP_COLS)} retirées par règle, "
          f"{len(constant)} constantes : {constant}")
    return feats


# --- Nettoyage des codes de non-réponse (transformation sans TARGET, faite une fois) ---------
# Variables continues calculées par le BRFSS : laissées intactes.
CONTINUOUS_COMPUTED = [
    "_BMI5", "_AGE80", "HTM4", "HTIN4", "WTKG3", "_PACKYRS", "_YRSSMOK", "_DRNKWK2",
    "DROCDY4_", "_PACKDAY", "_YRSQUIT", "_AGEG5YR", "_STATE",
]
# Colonnes à max <= 9 où 7 est une vraie catégorie (EMPLOY1 7 = retraité, _RACE1/_RACEPR1
# 7 = multiracial/hispanique, _INCOMG1 7 = 200 k$+) ; pour RRCOGNT2 7 = « constamment »
# et 8 = « ne sait pas ».
SEVEN_IS_REAL = ["EMPLOY1", "_RACE1", "_RACEPR1", "_INCOMG1", "RRCOGNT2"]
# Âges en années : 77/88 réels, 98 = ne sait pas, 99 = refus.
AGE_COLS = ["DIABAGE4", "CNCRAGE"]
# Âges / comptages codés sur 3 chiffres : 777/999 codes, 888 = « jamais » (LCSFIRST).
AGE3_COLS = ["LCSFIRST", "LCSLAST", "LCSNUMCG"]
# Codes mois-année (MMYYYY) : seuls 777777/999999 sont des codes.
MONTH_YEAR_COLS = ["FLSHTMY3", "HIVTSTD3", "COVIDFS1", "COVIDSE1"]
SPECIAL_COLS = (CONTINUOUS_COMPUTED + SEVEN_IS_REAL + AGE_COLS + AGE3_COLS + MONTH_YEAR_COLS
                + ["WEIGHT2", "HEIGHT3", "ALCDAY4", "CHKHEMO3"])
SHOW_COLS = ["GENHLTH", "PHYSHLTH", "CHILDREN", "WEIGHT2", "ALCDAY4"]


def _set(df, col, mask, value):
    df[col] = df[col].mask(mask, value).astype("float32")


def clean(df, feats, maxes):
    """Nettoie df en place. `maxes` = max par colonne calculé sur train (mêmes règles train/test)."""
    f = np.float32
    for c in feats:
        if c in SPECIAL_COLS:
            continue
        m = maxes[c]
        s = df[c]
        if m <= 9:
            _set(df, c, s.isin([7, 9]), np.nan)
        elif m <= 99:
            _set(df, c, s.isin([77, 99]), np.nan)
            _set(df, c, df[c] == 88, 0)
        elif m <= 999:
            _set(df, c, s.isin([777, 999]), np.nan)
            _set(df, c, df[c] == 888, 0)
        elif m <= 9999:
            _set(df, c, s.isin([7777, 9999]), np.nan)
            _set(df, c, df[c] == 8888, 0)
        # au-delà de 9999 : rien (variables calculées / dates non listées)

    for c in SEVEN_IS_REAL:
        _set(df, c, df[c] == 9, np.nan)
    _set(df, "RRCOGNT2", df["RRCOGNT2"] == 8, np.nan)
    for c in AGE_COLS:
        _set(df, c, df[c].isin([98, 99]), np.nan)
    for c in AGE3_COLS:
        _set(df, c, df[c].isin([777, 999]), np.nan)
        _set(df, c, df[c] == 888, 0)
    for c in MONTH_YEAR_COLS:
        _set(df, c, df[c].isin([777777, 999999]), np.nan)
    # CHKHEMO3 : 1-76 fois, 77 DK, 88 aucune, 98 jamais entendu parler, 99 refus
    _set(df, "CHKHEMO3", df["CHKHEMO3"].isin([77, 98, 99]), np.nan)
    _set(df, "CHKHEMO3", df["CHKHEMO3"] == 88, 0)

    # WEIGHT2 : 50-776 livres ; 9000-9998 = kilos ; 7777/9999 (et valeurs hors plage) -> NaN
    w = df["WEIGHT2"]
    kg = (w >= 9000) & (w <= 9998)
    out = w.where(~kg, (w - 9000) * f(2.20462))
    out = out.mask((w > 776) & (w < 9000) | (w == 9999), np.nan)
    df["WEIGHT2"] = out.astype("float32")

    # HEIGHT3 : 200-711 = pieds*100 + pouces ; 9000-9998 = centimètres ; 7777/9999 -> NaN
    h = df["HEIGHT3"]
    ftin = (h >= 200) & (h <= 711)
    cm = (h >= 9000) & (h <= 9998)
    out = h.where(~ftin, np.floor(h / 100) * 12 + (h % 100))
    out = out.where(~cm, (h - 9000) / f(2.54))
    out = out.mask(~(ftin | cm), np.nan)
    df["HEIGHT3"] = out.astype("float32")

    # ALCDAY4 : 101-107 jours/semaine -> x30/7 ; 201-230 jours/mois ; 888 = 0 ; 777/999 -> NaN
    a = df["ALCDAY4"]
    week = (a >= 101) & (a <= 107)
    month = (a >= 201) & (a <= 230)
    out = a.where(~week, (a - 100) * f(30 / 7))
    out = out.where(~month, a - 200)
    out = out.mask(a == 888, 0)
    out = out.mask(~(week | month | (a == 888)), np.nan)
    df["ALCDAY4"] = out.astype("float32")
    return df


def _show(df, title):
    print(f"--- value_counts {title} ---")
    for c in SHOW_COLS:
        vc = df[c].value_counts(dropna=False).sort_index()
        if len(vc) > 25:  # résumé pour les colonnes à beaucoup de modalités
            print(f"{c}: n_distinct={df[c].nunique()}, NaN={df[c].isna().sum()}, "
                  f"min={df[c].min():.1f}, max={df[c].max():.1f}, mean={df[c].mean():.2f}, "
                  f"tail={vc.tail(8).to_dict()}")
        else:
            print(f"{c}: {vc.to_dict()}")


def best_threshold(y, proba):
    grid = np.arange(0.05, 0.6, 0.005)
    scores = [f1_score(y, proba >= t) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def evaluate(y, oof):
    threshold, f1 = best_threshold(y, oof)
    pred = oof >= threshold
    return {
        "f1": f1,
        "threshold": threshold,
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "auc": float(roc_auc_score(y, oof)),
    }


def save_outputs(name, test, test_proba, metrics, info, t0):
    """Écrit la soumission et le JSON de résultats ; retourne le chemin du JSON."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (ROOT / "submissions").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    sub_path = ROOT / "submissions" / f"submission_{name}_{stamp}.csv"
    pred = (test_proba >= metrics["threshold"]).astype(int)
    pd.DataFrame({"ID": test["ID"], "TARGET": pred}).to_csv(sub_path, index=False)
    result = {
        "name": name,
        "idea": IDEA,
        "timestamp": stamp,
        "cv": metrics,
        "test_positive_rate": float(pred.mean()),
        "submission": f"submissions/{sub_path.name}",
        "runtime_s": round(time.time() - t0, 1),
        **info,
    }
    res_path = ROOT / "results" / f"{name}_{stamp}.json"
    res_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("CV :", {k: round(v, 4) for k, v in metrics.items()})
    print(f"Soumission : {sub_path}\nRésultats : {res_path}  ({result['runtime_s']} s)")
    return res_path


# --- Préparation des entrées du MLP (sans TARGET : faite une fois sur train et test) ----------
def prepare_inputs(X, X_test):
    """Retourne (cat_tr, cat_te, num_tr, num_te, cardinalités, cat_cols, num_cols).

    Catégorielles = colonnes à <= CAT_MAX_DISTINCT valeurs distinctes sur train+test :
    NaN -> code 0, puis 1..k via pd.factorize sur train+test concaténés.
    Numériques = les autres : QuantileTransformer normal ajusté sur train, NaN -> 0 après
    transformation, plus un indicateur isna par colonne.
    """
    both = pd.concat([X, X_test], ignore_index=True)
    n_tr = len(X)
    nunique = both.nunique()
    cat_cols = [c for c in X.columns if nunique[c] <= CAT_MAX_DISTINCT]
    num_cols = [c for c in X.columns if c not in cat_cols]

    cat = np.zeros((len(both), len(cat_cols)), dtype=np.int32)
    cards = []
    for j, c in enumerate(cat_cols):
        codes, uniques = pd.factorize(both[c], sort=True)  # NaN -> -1
        cat[:, j] = codes + 1
        cards.append(len(uniques) + 1)

    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             subsample=n_tr, random_state=SEED)
    qt.fit(X[num_cols].values)
    num = qt.transform(both[num_cols].values).astype(np.float32)
    isna = np.isnan(num).astype(np.float32)
    num = np.nan_to_num(num, nan=0.0)
    num = np.hstack([num, isna]).astype(np.float32)

    print(f"{len(cat_cols)} catégorielles (cardinalité totale {sum(cards)}), "
          f"{len(num_cols)} numériques (+{len(num_cols)} indicateurs isna) "
          f"-> dimension d'entrée {len(cat_cols) * EMB_DIM + num.shape[1]}")
    return (cat[:n_tr], cat[n_tr:], num[:n_tr], num[n_tr:], cards, cat_cols, num_cols)


class MLP(nn.Module):
    def __init__(self, cards, n_num):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(k, EMB_DIM) for k in cards])
        d_in = len(cards) * EMB_DIM + n_num
        layers = []
        for h in HIDDEN:
            layers += [nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(DROPOUT)]
            d_in = h
        layers.append(nn.Linear(d_in, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat, num):
        embs = [emb(cat[:, j]) for j, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [num], dim=1)
        return self.mlp(x).squeeze(1)


@torch.no_grad()
def predict_logits(model, cat, num):
    model.eval()
    out = torch.empty(len(cat), device=cat.device)
    for i in range(0, len(cat), EVAL_BATCH):
        out[i:i + EVAL_BATCH] = model(cat[i:i + EVAL_BATCH], num[i:i + EVAL_BATCH])
    return out


def train_fold(cards, cat_tr, num_tr, y_tr, cat_va, num_va, y_va_np, device, fold, seed):
    """Entraîne un MLP avec early stopping sur l'AUC de validation ; retourne (modèle, meilleure époque, AUC).

    `seed` pilote l'initialisation des poids (torch.manual_seed) et, via seed + fold, l'ordre des
    lots ; seed 42 reproduit r2_mlp_embeddings_gpu.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = MLP(cards, num_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + fold)
    n = len(cat_tr)
    best_auc, best_epoch, best_state, bad = -1.0, -1, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device, generator=gen)
        total = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2:  # BatchNorm exige au moins 2 lignes
                continue
            opt.zero_grad(set_to_none=True)
            logits = model(cat_tr[idx], num_tr[idx])
            loss = loss_fn(logits, y_tr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        va_logits = predict_logits(model, cat_va, num_va).cpu().numpy()
        auc = roc_auc_score(y_va_np, va_logits)
        print(f"  fold {fold} mlp seed {seed} époque {epoch + 1:2d}: loss {total / n:.4f}, AUC val {auc:.4f}",
              flush=True)
        if auc > best_auc:
            best_auc, best_epoch, bad = auc, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)  # restauration des meilleurs poids (meilleures époques 2-4)
    return model, best_epoch, best_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="r3_lgbbag_mlp_blend")
    parser.add_argument("--threads", type=int, default=0, help="0 = tous les cœurs")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    t0 = time.time()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train, test = load()
    feats = select_columns(train)
    maxes = train[feats].max()
    _show(train, "AVANT nettoyage (train)")
    nan_before = train[feats].isna().sum().sum()
    clean(train, feats, maxes)
    clean(test, feats, maxes)
    _show(train, "APRÈS nettoyage (train)")
    print(f"NaN dans train : {nan_before} -> {train[feats].isna().sum().sum()}")
    X, y = train[feats], train["TARGET"].values.astype(int)
    X_test = test[feats]
    print(f"train {X.shape}, test {X_test.shape}, taux de positifs {y.mean():.3f}")

    # Entrées du MLP (sans TARGET : préparées une fois sur train et test), tout sur le GPU
    cat_tr, cat_te, num_tr, num_te, cards, cat_cols, num_cols = prepare_inputs(X, X_test)
    t_prep = time.time() - t0
    print(f"préparation MLP : {t_prep:.1f} s")
    cat_all = torch.from_numpy(cat_tr).to(device)
    num_all = torch.from_numpy(num_tr).to(device)
    y_all = torch.from_numpy(y.astype(np.float32)).to(device)
    cat_test = torch.from_numpy(cat_te).to(device)
    num_test = torch.from_numpy(num_te).to(device)

    params = {**LGB_PARAMS, "num_threads": args.threads}
    n_seeds = len(SEEDS)
    n_mlp = len(MLP_SEEDS)
    oof_seeds = np.zeros((n_seeds, len(X)))  # une ligne par graine, pour mesurer le gain cumulé
    test_proba = np.zeros(len(X_test))  # LightGBM bag
    oof_mlp_seeds = np.zeros((n_mlp, len(X)))
    test_mlp = np.zeros(len(X_test))  # moyenne sur graines et folds
    importance = pd.Series(0.0, index=feats)
    best_iters, best_epochs, mlp_fold_aucs = [], [], []
    t_lgb_total, t_mlp_total = 0.0, 0.0
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        fold_iters = []
        t_lgb = time.time()
        for k, seed in enumerate(SEEDS):
            tf = time.time()
            # Dataset reconstruit par graine (comme la base) : seed 42 reproduit exactement la base
            model = lgb.train(
                {**params, "seed": seed},
                lgb.Dataset(X.iloc[tr_idx], y[tr_idx]),
                num_boost_round=5000,
                valid_sets=[lgb.Dataset(X.iloc[va_idx], y[va_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            # chaque modèle prédit avec SON best_iteration (pas de moyenne des itérations)
            oof_seeds[k, va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
            test_proba += model.predict(X_test, num_iteration=model.best_iteration) / (n_seeds * args.folds)
            importance += model.feature_importance("gain") / (n_seeds * args.folds)
            fold_iters.append(int(model.best_iteration))
            print(f"fold {fold} seed {seed}: {model.best_iteration} arbres, "
                  f"AUC {roc_auc_score(y[va_idx], oof_seeds[k, va_idx]):.4f} ({time.time() - tf:.0f} s)")
        best_iters.append(fold_iters)
        t_lgb_total += time.time() - t_lgb
        print(f"fold {fold} moyenne 5 graines LightGBM: AUC "
              f"{roc_auc_score(y[va_idx], oof_seeds[:, va_idx].mean(axis=0)):.4f} ({time.time() - t_lgb:.0f} s)")

        # MLP : 2 graines par fold, moyenne des sigmoid (OOF) ; test = moyenne sur graines et folds
        t_mlp = time.time()
        tr_t = torch.from_numpy(tr_idx).to(device)
        va_t = torch.from_numpy(va_idx).to(device)
        fold_epochs = []
        for k, seed in enumerate(MLP_SEEDS):
            ts = time.time()
            model, best_epoch, best_auc = train_fold(
                cards, cat_all[tr_t], num_all[tr_t], y_all[tr_t],
                cat_all[va_t], num_all[va_t], y[va_idx], device, fold, seed)
            oof_mlp_seeds[k, va_idx] = torch.sigmoid(
                predict_logits(model, cat_all[va_t], num_all[va_t])).cpu().numpy()
            test_mlp += torch.sigmoid(predict_logits(model, cat_test, num_test)).cpu().numpy() / (n_mlp * args.folds)
            fold_epochs.append(int(best_epoch))
            print(f"fold {fold} mlp seed {seed}: meilleure époque {best_epoch}, AUC {best_auc:.4f} "
                  f"({time.time() - ts:.0f} s)", flush=True)
            del model
        best_epochs.append(fold_epochs)
        t_mlp_total += time.time() - t_mlp
        mlp_fold_aucs.append(float(roc_auc_score(y[va_idx], oof_mlp_seeds[:, va_idx].mean(axis=0))))
        print(f"fold {fold} moyenne 2 graines MLP: AUC {mlp_fold_aucs[-1]:.4f} ({time.time() - t_mlp:.0f} s)",
              flush=True)

    oof_lgb = oof_seeds.mean(axis=0)
    oof_mlp = oof_mlp_seeds.mean(axis=0)
    metrics_lgb = evaluate(y, oof_lgb)
    metrics_mlp = evaluate(y, oof_mlp)
    metrics_mlp_seed42 = evaluate(y, oof_mlp_seeds[0])
    corr = float(np.corrcoef(oof_lgb, oof_mlp)[0, 1])
    corr_rank = float(pd.Series(oof_lgb).corr(pd.Series(oof_mlp), method="spearman"))
    print(f"LGB bag seul : F1 {metrics_lgb['f1']:.4f}, AUC {metrics_lgb['auc']:.4f}, seuil {metrics_lgb['threshold']:.3f}")
    print(f"MLP seed 42 seul : F1 {metrics_mlp_seed42['f1']:.4f}, AUC {metrics_mlp_seed42['auc']:.4f}")
    print(f"MLP 2 graines seul : F1 {metrics_mlp['f1']:.4f}, AUC {metrics_mlp['auc']:.4f}, "
          f"seuil {metrics_mlp['threshold']:.3f}")
    print(f"corrélation OOF LGB/MLP : Pearson {corr:.4f}, Spearman {corr_rank:.4f}")

    # Blend p = w*LGB + (1-w)*MLP, w choisi sur l'OOF (même w pour le test), seuil sur l'OOF blendé
    blend_table = {}
    best_w, best_f1 = None, -1.0
    for w in BLEND_WEIGHTS:
        m = evaluate(y, w * oof_lgb + (1 - w) * oof_mlp)
        blend_table[str(w)] = m
        print(f"w_lgb={w:.1f}: F1 {m['f1']:.4f}, AUC {m['auc']:.4f}, seuil {m['threshold']:.3f}, "
              f"P {m['precision']:.4f}, R {m['recall']:.4f}")
        if m["f1"] > best_f1:
            best_w, best_f1 = w, m["f1"]
    print(f"meilleur poids LGB : {best_w} (F1 {best_f1:.4f}, gain vs LGB bag seul {best_f1 - metrics_lgb['f1']:+.4f})")
    oof = best_w * oof_lgb + (1 - best_w) * oof_mlp
    test_blend = best_w * test_proba + (1 - best_w) * test_mlp
    metrics = evaluate(y, oof)
    # Gain exact : F1 OOF du seul seed 42 (= la base) et F1 pour 1..5 graines cumulées
    f1_single_seed42 = evaluate(y, oof_seeds[0])
    f1_cumulative = {}
    for k in range(1, n_seeds + 1):
        m = evaluate(y, oof_seeds[:k].mean(axis=0))
        f1_cumulative[k] = {"f1": m["f1"], "auc": m["auc"], "threshold": m["threshold"]}
        print(f"{k} graine(s) cumulée(s) {SEEDS[:k]}: F1 {m['f1']:.4f}, AUC {m['auc']:.4f}, "
              f"seuil {m['threshold']:.3f}")
    print(f"Seed 42 seul (base) : F1 {f1_single_seed42['f1']:.4f} -> 5 graines : F1 {metrics_lgb['f1']:.4f} "
          f"(gain {metrics_lgb['f1'] - f1_single_seed42['f1']:+.4f}) -> blend MLP : F1 {metrics['f1']:.4f} "
          f"(gain {metrics['f1'] - metrics_lgb['f1']:+.4f})")
    top = importance.sort_values(ascending=False).head(20)
    print("Top 20 variables (gain) :")
    print(top.round(0).to_string())
    save_outputs(args.name, test, test_blend, metrics,
                 {"model": "blend_lightgbm_seed_bagging_mlp_torch", "params": params, "seeds": SEEDS,
                  "mlp_seeds": MLP_SEEDS, "device": device.type, "n_features": len(feats),
                  "blend_weight_lgb": best_w, "blend_weights_tried": BLEND_WEIGHTS, "blend_table": blend_table,
                  "cv_lgb_bag_only": metrics_lgb, "cv_mlp_only": metrics_mlp,
                  "cv_mlp_seed42_only": metrics_mlp_seed42,
                  "oof_corr_lgb_mlp_pearson": corr, "oof_corr_lgb_mlp_spearman": corr_rank,
                  "best_iterations": best_iters, "mlp_best_epochs": best_epochs, "mlp_fold_aucs": mlp_fold_aucs,
                  "cv_seed42_only": f1_single_seed42, "cv_cumulative_seeds": f1_cumulative,
                  "time_lgb_s": round(t_lgb_total, 1), "time_mlp_s": round(t_mlp_total, 1),
                  "prep_time_s": round(t_prep, 1),
                  "mlp_config": {"n_cat": len(cat_cols), "n_num": len(num_cols), "emb_dim": EMB_DIM,
                                 "hidden": list(HIDDEN), "dropout": DROPOUT, "lr": LR,
                                 "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                                 "max_epochs": MAX_EPOCHS, "patience": PATIENCE},
                  "top_features": top.round(0).to_dict()}, t0)

if __name__ == "__main__":
    main()
