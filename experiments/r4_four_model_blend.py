"""Blend à 4 composants : LightGBM seed-bagged (5 graines) + XGBoost GPU + CatBoost GPU (ctr 1)
+ MLP PyTorch à embeddings (2 graines), poids ajustés par SLSQP sur la logloss OOF. Base r3_lgbbag_xgb_blend.

Usage : python experiments/r4_four_model_blend.py [--name NOM] [--threads N] [--folds 5]
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
import gc
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from torch import nn

IDEA = ("Blend à 4 composants entraînés dans les mêmes folds : LightGBM seed-bagged (5 graines, inchangé) "
        "+ XGBoost GPU (hist/cuda, depth 7, inchangé vs r3_lgbbag_xgb_blend) + CatBoost GPU (colonnes <= 15 "
        "valeurs catégorielles, NaN -> -1, max_ctr_complexity=1, recopié de r3_lgbbag_catboost_blend) "
        "+ MLP PyTorch à embeddings (2 graines 42 et 1, recopié de r3_lgbbag_mlp_blend) ; poids "
        "(w_lgb, w_xgb, w_cb, w_mlp) >= 0 de somme 1 obtenus par scipy.optimize.minimize (SLSQP) de la "
        "logloss OOF du mélange, mêmes poids sur le test ; seuil choisi sur l'OOF blendé ; références : poids "
        "fixes (0.4, 0.2, 0.2, 0.2) et blend LGB+XGB w=0.7 ; StratifiedKFold inchangé (random_state=42)")
SEED = 42
SEEDS = [42, 1, 2, 3, 4]  # la première graine (42) reproduit exactement la base
MLP_SEEDS = [42, 1]  # la première graine (42) reproduit exactement r2_mlp_embeddings_gpu
FIXED_WEIGHTS = (0.4, 0.2, 0.2, 0.2)  # référence : poids fixes (lgb, xgb, cb, mlp)
REF_W_LGB_XGB = 0.7  # référence : blend LGB+XGB de r3_lgbbag_xgb_blend (doit redonner F1 0.4250)
FOLD0_LIMIT_S = 210.0  # garde-fou : si le fold 0 dépasse 3 min 30, MLP_SEEDS -> [42] pour la suite
MAX_CAT_CARD = 15  # colonnes avec au plus ce nombre de valeurs distinctes (hors NaN) -> catégorielles CatBoost

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cuda",
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "colsample_bynode": 0.8,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "max_bin": 256,
    "seed": SEED,
}

CB_PARAMS = {
    "task_type": "GPU",
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "depth": 8,
    "learning_rate": 0.06,
    "iterations": 3000,
    "od_type": "Iter",
    "od_wait": 150,
    "l2_leaf_reg": 3,
    "border_count": 128,
    "one_hot_max_size": 4,
    "max_ctr_complexity": 1,  # CTR simples uniquement : divise le temps CatBoost (125-209 s/fold sinon)
    "random_seed": SEED,
    "verbose": 200,
    "use_best_model": True,
    "allow_writing_files": False,
}

# Hyperparamètres du MLP (recopiés de r3_lgbbag_mlp_blend / r2_mlp_embeddings_gpu)
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


# --- Préparation CatBoost (sans TARGET, appliquée APRÈS clean()) ----------------------------
def split_cat_num(train, feats):
    """Colonnes catégorielles = ≤ MAX_CAT_CARD valeurs distinctes (hors NaN) ; ne dépend pas de TARGET."""
    nun = train[feats].nunique(dropna=True)
    cat_cols = [c for c in feats if nun[c] <= MAX_CAT_CARD]
    num_cols = [c for c in feats if c not in cat_cols]
    print(f"{len(cat_cols)} colonnes catégorielles (<= {MAX_CAT_CARD} valeurs), {len(num_cols)} numériques")
    return cat_cols, num_cols


def prepare(df, feats, cat_cols):
    """Catégorielles : NaN -> -1 puis int32 ; numériques : float32 avec NaN (CatBoost les gère)."""
    out = df[feats].copy()
    out[cat_cols] = out[cat_cols].fillna(-1).astype("int32")
    return out


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


# --- Blend à 4 composants : poids par minimisation SLSQP de la logloss OOF ----------------------
COMPONENTS = ["lgb", "xgb", "cb", "mlp"]


def blend(weights, mats):
    """mats : tableau (4, n) ; weights : (4,) -> probabilité mélangée (n,)."""
    return np.asarray(weights) @ mats


def optimize_weights(y, oof_mats, x0):
    """Poids >= 0 de somme 1 minimisant la logloss OOF du mélange (SLSQP, départ x0)."""
    eps = 1e-7

    def objective(w):
        p = np.clip(blend(w, oof_mats), eps, 1 - eps)
        return float(log_loss(y, p))

    res = minimize(objective, np.asarray(x0, dtype=float), method="SLSQP",
                   bounds=[(0.0, 1.0)] * len(x0),
                   constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum()
    return w, float(res.fun), bool(res.success), str(res.message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="r4_four_model_blend")
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

    # Matrices CatBoost préparées APRÈS le nettoyage (les -1 remplacent les NaN créés par clean)
    cat_cols_cb, num_cols_cb = split_cat_num(train, feats)
    X_cb = prepare(train, feats, cat_cols_cb)
    X_test_cb = prepare(test, feats, cat_cols_cb)
    test_pool = Pool(X_test_cb, cat_features=cat_cols_cb)

    # Entrées du MLP (sans TARGET : préparées une fois sur train et test), tout sur le GPU
    cat_tr, cat_te, num_tr, num_te, cards, cat_cols_mlp, num_cols_mlp = prepare_inputs(X, X_test)
    t_prep = time.time() - t0
    print(f"préparation (chargement + nettoyage + CatBoost + MLP) : {t_prep:.1f} s")
    cat_all = torch.from_numpy(cat_tr).to(device)
    num_all = torch.from_numpy(num_tr).to(device)
    y_all = torch.from_numpy(y.astype(np.float32)).to(device)
    cat_test = torch.from_numpy(cat_te).to(device)
    num_test = torch.from_numpy(num_te).to(device)

    params = {**LGB_PARAMS, "num_threads": args.threads}
    xgb_params = {**XGB_PARAMS, "nthread": args.threads}
    cb_threads = args.threads if args.threads > 0 else -1
    cb_params = {**CB_PARAMS, "thread_count": cb_threads}
    print(f"XGBoost {xgb.__version__}, device demandé : {xgb_params['device']}")
    n_seeds = len(SEEDS)
    mlp_seeds = list(MLP_SEEDS)  # peut être réduit à [42] par le garde-fou après le fold 0
    mlp_seeds_per_fold = []
    oof_seeds = np.zeros((n_seeds, len(X)))  # une ligne par graine, pour mesurer le gain cumulé
    test_proba_lgb = np.zeros(len(X_test))
    oof_xgb, test_proba_xgb = np.zeros(len(X)), np.zeros(len(X_test))
    oof_cb, test_proba_cb = np.zeros(len(X)), np.zeros(len(X_test))
    oof_mlp, test_proba_mlp = np.zeros(len(X)), np.zeros(len(X_test))
    importance = pd.Series(0.0, index=feats)
    importance_xgb = pd.Series(0.0, index=feats)
    importance_cb = pd.Series(0.0, index=feats)
    best_iters, best_iters_xgb, trees_cb, best_epochs, mlp_fold_aucs = [], [], [], [], []
    time_lgb, time_xgb, time_cb, time_mlp, time_fold = [], [], [], [], []
    dtest = xgb.DMatrix(X_test)  # construit une fois, NaN natifs
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t_fold = time.time()
        # (a) LightGBM seed-bagged, inchangé vs r2_lgbm_seed_bagging
        fold_iters = []
        tl = time.time()
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
            test_proba_lgb += model.predict(X_test, num_iteration=model.best_iteration) / (n_seeds * args.folds)
            importance += model.feature_importance("gain") / (n_seeds * args.folds)
            fold_iters.append(int(model.best_iteration))
            print(f"fold {fold} seed {seed}: {model.best_iteration} arbres, "
                  f"AUC {roc_auc_score(y[va_idx], oof_seeds[k, va_idx]):.4f} ({time.time() - tf:.0f} s)",
                  flush=True)
            del model
            gc.collect()
        best_iters.append(fold_iters)
        time_lgb.append(round(time.time() - tl, 1))
        print(f"fold {fold} moyenne 5 graines: AUC "
              f"{roc_auc_score(y[va_idx], oof_seeds[:, va_idx].mean(axis=0)):.4f} "
              f"({time_lgb[-1]:.0f} s pour les 5 LightGBM)", flush=True)

        # (b) XGBoost GPU : un modèle par fold, NaN gérés nativement (inchangé vs r3_lgbbag_xgb_blend)
        tf = time.time()
        dtrain = xgb.DMatrix(X.iloc[tr_idx], y[tr_idx])
        dvalid = xgb.DMatrix(X.iloc[va_idx], y[va_idx])
        bst = xgb.train(xgb_params, dtrain, 5000, evals=[(dvalid, "valid")],
                        early_stopping_rounds=100, verbose_eval=False)
        it_range = (0, bst.best_iteration + 1)
        oof_xgb[va_idx] = bst.predict(dvalid, iteration_range=it_range)
        test_proba_xgb += bst.predict(dtest, iteration_range=it_range) / args.folds
        gain = bst.get_score(importance_type="gain")
        importance_xgb += pd.Series(gain).reindex(feats).fillna(0.0) / args.folds
        best_iters_xgb.append(int(bst.best_iteration))
        if fold == 0:
            # vérification que le GPU est bien utilisé (sinon plusieurs minutes par fold sur 8 cœurs)
            cfg = json.loads(bst.save_config())
            print("XGBoost config device :", cfg.get("learner", {}).get("generic_param", {}).get("device"),
                  "| tree_method :", cfg.get("learner", {}).get("gradient_booster", {})
                  .get("gbtree_train_param", {}).get("tree_method"))
        time_xgb.append(round(time.time() - tf, 1))
        print(f"fold {fold} xgb: {bst.best_iteration} arbres, "
              f"AUC {roc_auc_score(y[va_idx], oof_xgb[va_idx]):.4f} ({time_xgb[-1]:.0f} s)", flush=True)
        del dtrain, dvalid, bst
        gc.collect()

        # (c) CatBoost GPU avec variables catégorielles déclarées (max_ctr_complexity=1)
        tf = time.time()
        train_pool = Pool(X_cb.iloc[tr_idx], y[tr_idx], cat_features=cat_cols_cb)
        valid_pool = Pool(X_cb.iloc[va_idx], y[va_idx], cat_features=cat_cols_cb)
        cb = CatBoostClassifier(**cb_params)
        cb.fit(train_pool, eval_set=valid_pool)
        oof_cb[va_idx] = cb.predict_proba(valid_pool)[:, 1]
        test_proba_cb += cb.predict_proba(test_pool)[:, 1] / args.folds
        importance_cb += pd.Series(cb.get_feature_importance(), index=feats) / args.folds
        trees_cb.append(int(cb.get_best_iteration()) + 1)
        time_cb.append(round(time.time() - tf, 1))
        print(f"fold {fold} CatBoost: {cb.get_best_iteration() + 1} arbres, "
              f"AUC {roc_auc_score(y[va_idx], oof_cb[va_idx]):.4f}, {time_cb[-1]:.0f} s", flush=True)
        del cb, train_pool, valid_pool
        gc.collect()
        torch.cuda.empty_cache()  # libère la mémoire GPU entre CatBoost et le MLP

        # (d) MLP : graines par fold, moyenne des sigmoid (OOF) ; test = moyenne sur graines et folds
        t_mlp = time.time()
        tr_t = torch.from_numpy(tr_idx).to(device)
        va_t = torch.from_numpy(va_idx).to(device)
        fold_epochs = []
        n_mlp = len(mlp_seeds)
        mlp_seeds_per_fold.append(list(mlp_seeds))
        for k, seed in enumerate(mlp_seeds):
            ts = time.time()
            model, best_epoch, best_auc = train_fold(
                cards, cat_all[tr_t], num_all[tr_t], y_all[tr_t],
                cat_all[va_t], num_all[va_t], y[va_idx], device, fold, seed)
            oof_mlp[va_idx] += torch.sigmoid(
                predict_logits(model, cat_all[va_t], num_all[va_t])).cpu().numpy() / n_mlp
            test_proba_mlp += torch.sigmoid(
                predict_logits(model, cat_test, num_test)).cpu().numpy() / (n_mlp * args.folds)
            fold_epochs.append(int(best_epoch))
            print(f"fold {fold} mlp seed {seed}: meilleure époque {best_epoch}, AUC {best_auc:.4f} "
                  f"({time.time() - ts:.0f} s)", flush=True)
            del model
        del tr_t, va_t
        torch.cuda.empty_cache()
        best_epochs.append(fold_epochs)
        time_mlp.append(round(time.time() - t_mlp, 1))
        mlp_fold_aucs.append(float(roc_auc_score(y[va_idx], oof_mlp[va_idx])))
        print(f"fold {fold} moyenne {n_mlp} graine(s) MLP: AUC {mlp_fold_aucs[-1]:.4f} ({time_mlp[-1]:.0f} s)",
              flush=True)
        time_fold.append(round(time.time() - t_fold, 1))
        print(f"fold {fold} total : {time_fold[-1]:.0f} s (LGB {time_lgb[-1]:.0f}, XGB {time_xgb[-1]:.0f}, "
              f"CB {time_cb[-1]:.0f}, MLP {time_mlp[-1]:.0f})", flush=True)
        if fold == 0 and time_fold[-1] > FOLD0_LIMIT_S:
            mlp_seeds = [42]
            print(f"GARDE-FOU : fold 0 = {time_fold[-1]:.0f} s > {FOLD0_LIMIT_S:.0f} s, "
                  f"MLP_SEEDS -> {mlp_seeds} pour les folds suivants", flush=True)

    # --- Chaque composant seul ---------------------------------------------------------------
    oof_lgb = oof_seeds.mean(axis=0)
    oof_mats = np.vstack([oof_lgb, oof_xgb, oof_cb, oof_mlp])
    test_mats = np.vstack([test_proba_lgb, test_proba_xgb, test_proba_cb, test_proba_mlp])
    metrics_single = {}
    for name, row in zip(COMPONENTS, oof_mats):
        metrics_single[name] = evaluate(y, row)
        print(f"{name:>4} seul : F1 {metrics_single[name]['f1']:.4f}, AUC {metrics_single[name]['auc']:.4f}, "
              f"seuil {metrics_single[name]['threshold']:.3f}, "
              f"logloss {log_loss(y, np.clip(row, 1e-7, 1 - 1e-7)):.5f}")
    corr_mat = np.corrcoef(oof_mats)
    corr_df = pd.DataFrame(corr_mat, index=COMPONENTS, columns=COMPONENTS)
    print("Matrice de corrélation OOF (Pearson) :")
    print(corr_df.round(4).to_string())

    # --- Références : blend LGB+XGB w=0.7 (r3_lgbbag_xgb_blend) et poids fixes (0.4,0.2,0.2,0.2) -----
    metrics_ref_lgb_xgb = evaluate(y, REF_W_LGB_XGB * oof_lgb + (1 - REF_W_LGB_XGB) * oof_xgb)
    print(f"Référence LGB+XGB w_lgb={REF_W_LGB_XGB} : F1 {metrics_ref_lgb_xgb['f1']:.4f} "
          f"(attendu 0.4250), AUC {metrics_ref_lgb_xgb['auc']:.4f}, seuil {metrics_ref_lgb_xgb['threshold']:.3f}")
    oof_fixed = blend(FIXED_WEIGHTS, oof_mats)
    metrics_fixed = evaluate(y, oof_fixed)
    ll_fixed = float(log_loss(y, np.clip(oof_fixed, 1e-7, 1 - 1e-7)))
    print(f"Référence poids fixes {FIXED_WEIGHTS} : F1 {metrics_fixed['f1']:.4f}, AUC {metrics_fixed['auc']:.4f}, "
          f"seuil {metrics_fixed['threshold']:.3f}, logloss {ll_fixed:.5f}")

    # --- Blend retenu : poids SLSQP minimisant la logloss OOF, mêmes poids sur le test ---------
    t_opt = time.time()
    w_opt, ll_opt, ok, msg = optimize_weights(y, oof_mats, [0.25, 0.25, 0.25, 0.25])
    weights = {name: float(w) for name, w in zip(COMPONENTS, w_opt)}
    print(f"Poids SLSQP (logloss OOF {ll_opt:.5f}, succès={ok}, {msg}, {time.time() - t_opt:.1f} s) : "
          f"{ {k: round(v, 4) for k, v in weights.items()} }")
    oof = blend(w_opt, oof_mats)
    test_proba = blend(w_opt, test_mats)
    metrics = evaluate(y, oof)
    print(f"Blend SLSQP : F1 {metrics['f1']:.4f}, AUC {metrics['auc']:.4f}, seuil {metrics['threshold']:.3f} "
          f"(gain vs LGB seed-bag {metrics['f1'] - metrics_single['lgb']['f1']:+.4f}, "
          f"vs LGB+XGB w=0.7 {metrics['f1'] - metrics_ref_lgb_xgb['f1']:+.4f}, "
          f"vs poids fixes {metrics['f1'] - metrics_fixed['f1']:+.4f})")

    # Gain exact : F1 OOF du seul seed 42 (= la base) et F1 pour 1..5 graines cumulées
    f1_single_seed42 = evaluate(y, oof_seeds[0])
    f1_cumulative = {}
    for k in range(1, n_seeds + 1):
        m = evaluate(y, oof_seeds[:k].mean(axis=0))
        f1_cumulative[k] = {"f1": m["f1"], "auc": m["auc"], "threshold": m["threshold"]}
        print(f"{k} graine(s) cumulée(s) {SEEDS[:k]}: F1 {m['f1']:.4f}, AUC {m['auc']:.4f}, "
              f"seuil {m['threshold']:.3f}")
    print(f"Seed 42 seul (base) : F1 {f1_single_seed42['f1']:.4f} -> 5 graines : F1 "
          f"{metrics_single['lgb']['f1']:.4f} (gain {metrics_single['lgb']['f1'] - f1_single_seed42['f1']:+.4f})")
    print(f"temps par fold : total {time_fold}, LGB x5 {time_lgb}, XGB {time_xgb}, CatBoost {time_cb}, MLP {time_mlp}")
    top = importance.sort_values(ascending=False).head(20)
    print("Top 20 variables LGB (gain) :")
    print(top.round(0).to_string())
    top_xgb = importance_xgb.sort_values(ascending=False).head(20)
    print("Top 20 variables XGB (gain) :")
    print(top_xgb.round(2).to_string())
    top_cb = importance_cb.sort_values(ascending=False).head(20)
    print("Top 20 variables CatBoost (PredictionValuesChange) :")
    print(top_cb.round(2).to_string())
    save_outputs(args.name, test, test_proba, metrics,
                 {"model": "blend4_lgbm_seed_bagging_xgb_catboost_mlp_slsqp_logloss",
                  "components": COMPONENTS, "blend_weights": weights,
                  "blend_weights_init": [0.25, 0.25, 0.25, 0.25],
                  "blend_logloss_oof": ll_opt, "blend_opt_success": ok, "blend_opt_message": msg,
                  "cv_components_only": metrics_single,
                  "f1_components_only": {k: v["f1"] for k, v in metrics_single.items()},
                  "oof_corr_matrix": {r: {c: float(corr_df.loc[r, c]) for c in COMPONENTS} for r in COMPONENTS},
                  "ref_fixed_weights": list(FIXED_WEIGHTS), "cv_ref_fixed_weights": metrics_fixed,
                  "ref_fixed_weights_logloss_oof": ll_fixed,
                  "ref_w_lgb_xgb": REF_W_LGB_XGB, "cv_ref_lgb_xgb": metrics_ref_lgb_xgb,
                  "params": params, "seeds": SEEDS, "xgb_params": xgb_params, "cb_params": cb_params,
                  "mlp_seeds_initial": MLP_SEEDS, "mlp_seeds_per_fold": mlp_seeds_per_fold,
                  "fold0_limit_s": FOLD0_LIMIT_S,
                  "guard_triggered": bool(any(len(s) < len(MLP_SEEDS) for s in mlp_seeds_per_fold)),
                  "device": device.type, "n_features": len(feats),
                  "n_cat_features_cb": len(cat_cols_cb),
                  "best_iterations": best_iters, "best_iterations_xgb": best_iters_xgb, "trees_cb": trees_cb,
                  "mlp_best_epochs": best_epochs, "mlp_fold_aucs": mlp_fold_aucs,
                  "cv_seed42_only": f1_single_seed42, "cv_cumulative_seeds": f1_cumulative,
                  "time_fold_s": time_fold, "time_lgb_per_fold_s": time_lgb, "time_xgb_per_fold_s": time_xgb,
                  "time_cb_per_fold_s": time_cb, "time_mlp_per_fold_s": time_mlp, "prep_time_s": round(t_prep, 1),
                  "mlp_config": {"n_cat": len(cat_cols_mlp), "n_num": len(num_cols_mlp), "emb_dim": EMB_DIM,
                                 "hidden": list(HIDDEN), "dropout": DROPOUT, "lr": LR,
                                 "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                                 "max_epochs": MAX_EPOCHS, "patience": PATIENCE},
                  "top_features": top.round(0).to_dict(),
                  "top_features_xgb": top_xgb.round(2).to_dict(),
                  "top_features_cb": top_cb.round(2).to_dict()}, t0)


if __name__ == "__main__":
    main()
