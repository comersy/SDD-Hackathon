"""Modèle « intelligent » : MLP à quatre tours (A amont, B comorbidités, C parcours, D contexte) sommées en logit.

Usage (Modal, GPU) :
    python -W ignore -m modal run modal_run.py --script experiments/smart_mlp_towers.py --name smart_mlp_towers --threads 8 --gpu
Test local minuscule (construction des colonnes/dérivées, jamais un entraînement complet) :
    python experiments/smart_mlp_towers.py --sample 2000 --seeds 1 --epochs 2 --retrain-seeds 1 --no-save

Idée. Au lieu de donner 283 colonnes à un réseau, on ne garde que les colonnes DÉFENDABLES de
features/tiers.json (copiées ici en constantes), classées en quatre niveaux :
  A = facteurs de risque en amont (âge, sexe, tabac, IMC, diabète, activité, alcool, sommeil, santé mentale),
  B = comorbidités (AVC, rein, BPCO, arthrite, asthme, cancers, surdité, cécité),
  C = parcours et conséquences (santé déclarée, limitations, suivi médical, vaccins, arrêt du tabac...) :
      causalité inversée, utilisable pour la compétition mais pas pour la prévention,
  D = contexte socio-démographique (revenu, éducation, État, logement...).
Tout le reste (intrusif, doublons calculés, artefacts du questionnaire) est exclu.
Le réseau a UNE TOUR PAR NIVEAU (embeddings 8 des catégorielles + numériques en quantiles -> 256 -> 128 -> 1,
BatchNorm, ReLU, dropout 0.3) et le logit final est ADDITIF entre niveaux :
    logit = biais + tour_A + tour_B + tour_C + tour_D.
Les interactions sont libres à l'intérieur d'un niveau (ex. âge x tabac dans A) mais impossibles entre niveaux,
ce qui rend chaque prédiction décomposable : on lit, pour chaque personne, ce que valent ses facteurs d'amont,
ses comorbidités, son parcours et son contexte, en logit. Le même réseau donne aussi un SCORE PRÉVENTION
sigmoid(biais + tour_A + tour_B [+ décalage constant = moyenne de tour_C + tour_D sur le fold d'entraînement])
qui ne regarde que ce qu'un outil de prévention a le droit de regarder. Un petit réseau A+B réentraîné seul
(2 graines) donne une seconde estimation de la valeur des facteurs d'amont, sans l'effet d'ablation.

Variables dérivées (construites une fois sur train et test, sans TARGET) : SMOKING_STATUS, BMI_CALC,
DRINKS_WEEK, DIABETES_BIN, DIAB_YEARS (niveau A) ; N_COMORB (B) ; N_LIMITS, CARE_INTENSITY (C).
Valeurs manquantes : code 0 (embedding « manquant ») pour les catégorielles, 0 + indicateur pour les numériques :
l'absence d'une question de module ou conditionnelle garde son sens.

Protocole identique à train.py : train + extra concaténés, StratifiedKFold(5, shuffle, seed 42), OOF,
prétraitement ajusté sur la partie entraînement du fold, 5 graines par fold (OneCycleLR 12 époques, lr 1e-3,
wd 1e-4, batch 1024, arrêt anticipé patience 4 sur la logloss de validation), seuil choisi sur l'OOF,
test = moyenne folds x graines. Sorties : JSON (colonnes par niveau, statistiques des tours, décomposition de
5 profils types, métriques du modèle prévention), OOF/test CSV du modèle complet et des modèles prévention.
"""
import argparse
import copy
import json
import math
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
from r1_clean_codes import ROOT, SEED, clean, evaluate, load, select_columns  # noqa: E402

IDEA = ("Modèle intelligent : colonnes défendables seulement (niveaux A amont / B comorbidités / C parcours / "
        "D contexte de features/tiers.json + 8 dérivées), MLP à quatre tours sommées en logit (additif entre "
        "niveaux, interactions libres dans un niveau), 5 graines par fold ; le même réseau donne un score "
        "prévention A+B et une décomposition lisible de chaque prédiction")
NAN_MAX = 0.95          # colonnes vides à plus de 95 % : retirées (critère sans TARGET)
CAT_MAX_LEVELS = 15     # <= 15 modalités distinctes -> catégorielle (embedding)
# Codes nominaux à mettre en embedding quel que soit le nombre de modalités (sinon quantiles absurdes)
FORCE_CAT = {"_STATE", "_RACE1", "MARITAL", "EMPLOY1", "PRIMINSR", "LASTSMK2", "HOWLONG", "_CRCREC2",
             "_MAM5023", "SMOKING_STATUS", "DIABETES_BIN"}
# Comptages / mesures à garder numériques même avec peu de modalités
FORCE_NUM = {"N_COMORB", "N_LIMITS", "CARE_INTENSITY", "DIAB_YEARS", "BMI_CALC", "DRINKS_WEEK", "_AGE80",
             "SLEPTIM1", "MENTHLTH", "PHYSHLTH", "POORHLTH", "CHILDREN", "HHADULT"}

# Copie de features/tiers.json (le conteneur Modal n'a pas features/)
TIERS = {
    "A": ["_AGE80", "SEXVAR", "SMOKE100", "SMOKDAY2", "_SMOKER3", "_PACKYRS", "_YRSSMOK", "_PACKDAY", "USENOW3",
          "ECIGNOW2", "_CURECI2", "WEIGHT2", "HEIGHT3", "_BMI5", "DIABETE4", "DIABAGE4", "PREDIAB2", "EXERANY2",
          "ALCDAY4", "AVEDRNK3", "DRNK3GE5", "MAXDRNKS", "_DRNKWK2", "SLEPTIM1", "MENTHLTH", "ADDEPEV3", "SDHSTRE1",
          "PREGNANT"],
    "B": ["CVDSTRK3", "CHCKDNY2", "CHCCOPD3", "HAVARTH4", "ASTHMA3", "ASTHNOW", "CHCSCNC1", "CHCOCNC1", "DEAF", "BLIND"],
    "C": ["GENHLTH", "PHYSHLTH", "POORHLTH", "DIFFWALK", "DIFFDRES", "DIFFALON", "DECIDE", "EMPLOY1", "PERSDOC3",
          "CHECKUP1", "MEDCOST1", "PRIMINSR", "FLUSHOT7", "FLSHTMY3", "PNEUVAC4", "TETANUS1", "COVIDVA1", "_YRSQUIT",
          "LASTSMK2", "STOPSMK2", "LCSCTSC1", "LCSSCNCR", "LCSFIRST", "LCSLAST", "LCSNUMCG", "LASTDEN4", "RMVTETH4",
          "_CRCREC2", "_MAM5023", "HOWLONG"],
    "D": ["INCOME3", "EDUCA", "MARITAL", "_RACE1", "_HISPANC", "_STATE", "_METSTAT", "_URBSTAT", "HHADULT", "CHILDREN",
          "VETERAN3", "RENTHOM1", "SDHFOOD1", "SDHBILLS", "SDHISOLT", "EMTSUPRT", "LSATISFY"],
}
TIER_NAMES = {"A": "A_amont", "B": "B_comorbidites", "C": "C_parcours_consequences", "D": "D_contexte"}
DERIVED_TIER = {"SMOKING_STATUS": "A", "BMI_CALC": "A", "DRINKS_WEEK": "A", "DIABETES_BIN": "A", "DIAB_YEARS": "A",
                "N_COMORB": "B", "N_LIMITS": "C", "CARE_INTENSITY": "C"}
LIMIT_COLS = ["DIFFWALK", "DIFFDRES", "DIFFALON", "DECIDE"]

CONFIG = {
    "epochs": 12,
    "patience": 4,
    "batch_size": 1024,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "emb_dim": 8,
    "hidden": [256, 128],
    "dropout": 0.3,
    "seeds": 5,                   # graines par fold (modèle complet)
    "prevention_retrain_seeds": 2,  # graines par fold du petit réseau A+B réentraîné seul
}


# ------------------------------------------------------------------ colonnes et dérivées
def drop_sparse(train, feats):
    nan_rate = train[feats].isna().mean()
    sparse = sorted(nan_rate[nan_rate > NAN_MAX].index)
    print(f"{len(sparse)} colonnes vides à plus de {NAN_MAX:.0%} retirées, {len(feats) - len(sparse)} candidates")
    return [c for c in feats if c not in sparse]


def add_derived(df):
    """Variables dérivées, sans TARGET, mêmes règles train/test (appelée après clean())."""
    s100, sday = df["SMOKE100"], df["SMOKDAY2"]
    status = pd.Series(np.nan, index=df.index, dtype="float32")
    status = status.mask(s100 == 2, 0)                 # jamais fumé 100 cigarettes
    status = status.mask((s100 == 1) & (sday == 3), 1)  # ancien fumeur
    status = status.mask((s100 == 1) & (sday == 2), 2)  # occasionnel
    status = status.mask((s100 == 1) & (sday == 1), 3)  # quotidien
    df["SMOKING_STATUS"] = status.astype("float32")

    kg = df["WEIGHT2"] * 0.453592            # WEIGHT2 nettoyé = livres
    m = df["HEIGHT3"] * 0.0254               # HEIGHT3 nettoyé = pouces
    bmi = kg / (m ** 2)
    df["BMI_CALC"] = bmi.where((bmi >= 10) & (bmi <= 100)).astype("float32")

    df["_DRNKWK2"] = df["_DRNKWK2"].mask(df["_DRNKWK2"] >= 99900, np.nan).astype("float32")  # 99900 = ne sait pas
    days, avg = df["ALCDAY4"], df["AVEDRNK3"]  # ALCDAY4 nettoyé = jours par mois, 0 = ne boit pas
    dw = (days / 30 * 7) * avg
    df["DRINKS_WEEK"] = dw.mask(days == 0, 0).astype("float32")

    diab = df["DIABETE4"]
    df["DIABETES_BIN"] = (diab == 1).astype("float32").mask(diab.isna(), np.nan).astype("float32")
    dy = df["_AGE80"] - df["DIABAGE4"]
    df["DIAB_YEARS"] = dy.where((diab == 1) & (dy >= 0)).astype("float32")

    comorb = [c for c in TIERS["B"] if c != "ASTHNOW"]  # ASTHNOW est une sous-question d'ASTHMA3 (pas de double compte)
    df["N_COMORB"] = sum((df[c] == 1).astype("float32") for c in comorb).astype("float32")
    df["N_LIMITS"] = sum((df[c] == 1).astype("float32") for c in LIMIT_COLS).astype("float32")
    df["CARE_INTENSITY"] = (df["PERSDOC3"].isin([1, 2]).astype("float32") + (df["CHECKUP1"] == 1)
                            + (df["FLUSHOT7"] == 1) + (df["PNEUVAC4"] == 1)).astype("float32")

    # FLSHTMY3 = MMYYYY (ex. 92022 = septembre 2022) -> index mensuel (année x 12 + mois)
    f = df["FLSHTMY3"]
    month, year = np.floor(f / 10000), f % 10000
    idx = (year * 12 + month).where((year >= 2015) & (year <= 2023) & (month >= 1) & (month <= 12))
    df["FLSHTMY3"] = idx.astype("float32")
    return df


def tier_columns(feats):
    """Colonnes finales par niveau : tiers ∩ colonnes disponibles, plus les dérivées."""
    avail = set(feats)
    cols = {t: [c for c in TIERS[t] if c in avail] for t in TIERS}
    for c, t in DERIVED_TIER.items():
        cols[t].append(c)
    return cols


# ------------------------------------------------------------------ prétraitement par fold
class TierPreprocessor:
    """Une tour = ses catégorielles (codes, 0 = manquant) + ses numériques (quantiles, NaN -> 0 + indicateur).
    Ajusté sur la partie entraînement du fold uniquement ; n'utilise jamais TARGET."""

    def __init__(self, cats, nums):
        self.cats, self.nums = cats, nums

    def fit(self, df):
        self.maps = {c: {v: i + 1 for i, v in enumerate(sorted(df[c].dropna().unique().tolist()))}
                     for c in self.cats}
        self.cardinalities = [len(self.maps[c]) + 1 for c in self.cats]
        self.qt = (QuantileTransformer(n_quantiles=min(1000, len(df)), output_distribution="normal",
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


def split_types(train, cols):
    nunique = train[cols].nunique()
    cats = [c for c in cols if c in FORCE_CAT or (c not in FORCE_NUM and nunique[c] <= CAT_MAX_LEVELS)]
    nums = [c for c in cols if c not in cats]
    return cats, nums


# ------------------------------------------------------------------------------------ modèle
class Tower(nn.Module):
    def __init__(self, cardinalities, n_num, cfg):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(card, cfg["emb_dim"]) for card in cardinalities])
        d = len(cardinalities) * cfg["emb_dim"] + n_num
        layers = []
        for h in cfg["hidden"]:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(cfg["dropout"])]
            d = h
        layers.append(nn.Linear(d, 1, bias=False))  # le biais est global, pour une décomposition propre
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        parts = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)] + [x_num]
        return self.mlp(torch.cat(parts, dim=1)).squeeze(1)


class TowerMLP(nn.Module):
    """logit = biais + somme des tours ; towers_out() rend la contribution de chaque tour."""

    def __init__(self, specs, cfg):
        super().__init__()
        self.towers = nn.ModuleList([Tower(card, n_num, cfg) for card, n_num in specs])
        self.bias = nn.Parameter(torch.zeros(1))

    def towers_out(self, xs):
        return torch.stack([tower(xc, xn) for tower, (xc, xn) in zip(self.towers, xs)], dim=1)

    def forward(self, xs):
        return self.bias + self.towers_out(xs).sum(dim=1)


# ------------------------------------------------------------------------------ entraînement
def to_device(parts, device):
    return [(torch.from_numpy(c).to(device), torch.from_numpy(n).to(device)) for c, n in parts]


def take(xs, idx):
    return [(xc[idx], xn[idx]) for xc, xn in xs]


@torch.no_grad()
def predict_towers(model, xs, batch_size=16384):
    """Contributions (N, n_tours) en float64 ; la probabilité complète = sigmoid(biais + somme)."""
    model.eval()
    out = []
    n = len(xs[0][0])
    for i in range(0, n, batch_size):
        batch = take(xs, slice(i, i + batch_size))
        with torch.autocast(device_type=xs[0][0].device.type, dtype=torch.bfloat16, enabled=xs[0][0].is_cuda):
            out.append(model.towers_out(batch).float())
    return torch.cat(out).cpu().numpy().astype(np.float64)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train_one(model, xs_tr, y_tr, xs_va, y_va, cfg, device, seed, tag):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    n, bs = len(y_tr), cfg["batch_size"]
    starts = [i for i in range(0, n, bs) if min(bs, n - i) >= 2]  # pas de lot de taille 1 (BatchNorm)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * len(starts))
    gen = torch.Generator().manual_seed(seed)
    y_np = y_va.cpu().numpy()
    best, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        t0 = time.time()
        for i in starts:
            idx = perm[i:i + bs]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(take(xs_tr, idx))
            loss = loss_fn(logits.float(), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        t = predict_towers(model, xs_va)
        p = sigmoid(model.bias.item() + t.sum(axis=1))
        ll = log_loss(y_np, np.clip(p, 1e-7, 1 - 1e-7))
        auc = roc_auc_score(y_np, p)
        history.append({"epoch": epoch, "val_logloss": round(ll, 5), "val_auc": round(auc, 5),
                        "s": round(time.time() - t0, 1)})
        print(f"  {tag} époque {epoch}: logloss {ll:.5f} AUC {auc:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if ll < best - 1e-5:
            best, best_state, bad = ll, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return history


# ------------------------------------------------------------------------------ lecture du modèle
PROFILE_RULES = {
    "jeune_sain": lambda d: (d["_AGE80"] <= 30) & (d["SMOKING_STATUS"] == 0) & (d["DIABETES_BIN"] == 0)
    & (d["GENHLTH"] <= 2) & (d["N_COMORB"] == 0),
    "fumeur_age": lambda d: (d["_AGE80"] >= 65) & (d["SMOKING_STATUS"] == 3),
    "diabetique_ancien": lambda d: (d["DIABETES_BIN"] == 1) & (d["DIAB_YEARS"] >= 10) & (d["_AGE80"].between(50, 65))
    & (d["SMOKING_STATUS"] == 0),
    "mauvaise_sante_suivi_lourd": lambda d: (d["GENHLTH"] == 5) & (d["DIFFWALK"] == 1) & (d["CARE_INTENSITY"] == 4)
    & (d["_AGE80"] >= 60),
    "precaire_age_moyen": lambda d: (d["INCOME3"] <= 3) & (d["EDUCA"] <= 3) & (d["MEDCOST1"] == 1)
    & (d["_AGE80"].between(40, 60)),
}
PROFILE_SHOW = ["_AGE80", "SEXVAR", "SMOKING_STATUS", "_PACKYRS", "BMI_CALC", "DIABETES_BIN", "DIAB_YEARS", "EXERANY2",
                "DRINKS_WEEK", "SLEPTIM1", "MENTHLTH", "N_COMORB", "CVDSTRK3", "GENHLTH", "N_LIMITS", "CARE_INTENSITY",
                "INCOME3", "EDUCA", "_STATE"]


def _f(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)


def pick_profiles(train, y, towers_oof, bias, oof_full, oof_prev):
    """5 individus types choisis dans l'OOF (membre médian de chaque règle) et leur décomposition en logit."""
    profiles = []
    for name, rule in PROFILE_RULES.items():
        mask = rule(train).fillna(False).to_numpy()
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            profiles.append({"profile": name, "n_matching": 0})
            continue
        i = int(idx[np.argsort(oof_full[idx])[len(idx) // 2]])
        row = train.iloc[i]
        profiles.append({
            "profile": name, "n_matching": int(len(idx)), "row": i, "target": int(y[i]),
            "positive_rate_in_group": round(float(y[idx].mean()), 4),
            "values": {c: _f(row[c]) for c in PROFILE_SHOW},
            "logit": {"bias": round(float(bias), 4), **{TIER_NAMES[t]: round(float(towers_oof[i, k]), 4)
                                                        for k, t in enumerate("ABCD")},
                      "total": round(float(bias + towers_oof[i].sum()), 4)},
            "proba_full": round(float(oof_full[i]), 4), "proba_prevention": round(float(oof_prev[i]), 4),
        })
    return profiles


def tower_stats(towers_oof, bias):
    total = bias + towers_oof.sum(axis=1)
    stats = {}
    for k, t in enumerate("ABCD"):
        v = towers_oof[:, k]
        stats[TIER_NAMES[t]] = {"mean": _f(v.mean()), "std": _f(v.std()), "var": _f(v.var()),
                                "min": _f(v.min()), "q05": _f(np.quantile(v, 0.05)), "q95": _f(np.quantile(v, 0.95)),
                                "max": _f(v.max()), "var_share_of_total_logit": _f(v.var() / total.var()),
                                "corr_with_total_logit": _f(np.corrcoef(v, total)[0, 1])}
    corr = np.corrcoef(towers_oof.T)
    return {"bias": _f(bias), "total_logit": {"mean": _f(total.mean()), "var": _f(total.var())}, "towers": stats,
            "tower_correlation": {TIER_NAMES[t]: {TIER_NAMES[u]: _f(corr[k, j]) for j, u in enumerate("ABCD")}
                                  for k, t in enumerate("ABCD")}}


# ------------------------------------------------------------------------------ sorties
def save_outputs(name, test, oof, test_proba, metrics, info, t0, extra):
    """Comme dl_base.save_outputs, plus les CSV OOF/test des modèles prévention (même horodatage)."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (ROOT / "submissions").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    sub_path = ROOT / "submissions" / f"submission_{name}_{stamp}.csv"
    pred = (test_proba >= metrics["threshold"]).astype(int)
    pd.DataFrame({"ID": test["ID"], "TARGET": pred}).to_csv(sub_path, index=False)
    csvs = {}
    for suffix, (o, tp) in [("", (oof, test_proba)), *extra.items()]:
        base = f"{name}{suffix}_{stamp}"
        pd.DataFrame({"oof": o}).to_csv(ROOT / "results" / f"{base}_oof.csv", index=False, float_format="%.6f")
        pd.DataFrame({"ID": test["ID"], "proba": tp}).to_csv(ROOT / "results" / f"{base}_test.csv", index=False,
                                                            float_format="%.6f")
        csvs[suffix or "full"] = {"oof_csv": f"results/{base}_oof.csv", "test_csv": f"results/{base}_test.csv"}
    for suffix, key in [("_prevention", "prevention_model"), ("_prevention_retrained", "prevention_retrained")]:
        if suffix in csvs and key in info:
            info[key].update(csvs[suffix])
    result = {
        "name": name, "idea": IDEA, "timestamp": stamp, "cv": metrics,
        "test_positive_rate": float(pred.mean()), "submission": f"submissions/{sub_path.name}",
        "oof_csv": csvs["full"]["oof_csv"], "test_csv": csvs["full"]["test_csv"],
        "runtime_s": round(time.time() - t0, 1), **info,
    }
    res_path = ROOT / "results" / f"{name}_{stamp}.json"
    res_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("CV :", {k: round(v, 4) for k, v in metrics.items()})
    print(f"Soumission : {sub_path}\nRésultats : {res_path}  ({result['runtime_s']} s)")
    return res_path


# ------------------------------------------------------------------------------ programme
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="smart_mlp_towers")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=None, help="remplace CONFIG['seeds']")
    parser.add_argument("--epochs", type=int, default=None, help="remplace CONFIG['epochs']")
    parser.add_argument("--retrain-seeds", type=int, default=None, help="remplace CONFIG['prevention_retrain_seeds']")
    parser.add_argument("--sample", type=int, default=0, help="test local : n lignes de train seulement")
    parser.add_argument("--no-save", action="store_true", help="test local : n'écrit rien")
    args = parser.parse_args()
    cfg = dict(CONFIG)
    for k, v in [("seeds", args.seeds), ("epochs", args.epochs), ("prevention_retrain_seeds", args.retrain_seeds)]:
        if v is not None:
            cfg[k] = v
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    t0 = time.time()

    # 1. colonnes disponibles, nettoyage des codes, dérivées (sans TARGET, une fois)
    train, test = load()
    if args.sample:
        train = train.sample(args.sample, random_state=SEED).reset_index(drop=True)
        test = test.head(args.sample).reset_index(drop=True)
    feats = drop_sparse(train, select_columns(train))
    maxes = train[feats].max()
    clean(train, feats, maxes)
    clean(test, feats, maxes)
    add_derived(train)
    add_derived(test)
    cols = tier_columns(feats)
    types = {t: split_types(train, cols[t]) for t in cols}
    all_cols = [c for t in "ABCD" for c in cols[t]]
    excluded = sorted(set(TIERS["A"] + TIERS["B"] + TIERS["C"] + TIERS["D"]) - set(feats))
    for t in "ABCD":
        print(f"{TIER_NAMES[t]} : {len(cols[t])} colonnes ({len(types[t][0])} cat + {len(types[t][1])} num) : {cols[t]}")
    print(f"{len(all_cols)} colonnes au total ; absentes ou vides > {NAN_MAX:.0%} : {excluded}")
    y = train["TARGET"].values.astype(int)
    n_tr, n_te = len(train), len(test)

    # 2. validation croisée
    oof_full, oof_prev, oof_retr = np.zeros(n_tr), np.zeros(n_tr), np.zeros(n_tr)
    te_full, te_prev, te_retr = np.zeros(n_te), np.zeros(n_te), np.zeros(n_te)
    towers_oof, towers_te = np.zeros((n_tr, 4)), np.zeros((n_te, 4))
    bias_oof, folds_info = 0.0, []
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        tf = time.time()
        df_tr, df_va = train.iloc[tr_idx], train.iloc[va_idx]
        pres = [TierPreprocessor(*types[t]).fit(df_tr) for t in "ABCD"]
        xs_tr = to_device([p.transform(df_tr) for p in pres], device)
        xs_va = to_device([p.transform(df_va) for p in pres], device)
        xs_te = to_device([p.transform(test) for p in pres], device)
        y_tr = torch.from_numpy(y[tr_idx].astype(np.float32)).to(device)
        y_va = torch.from_numpy(y[va_idx].astype(np.float32)).to(device)
        specs = [(p.cardinalities, xs_tr[k][1].shape[1]) for k, p in enumerate(pres)]
        seeds_info = []

        # (a) modèle complet, 5 graines ; (b) score prévention = même réseau, tours C et D remplacées par leur
        #     moyenne sur le fold d'entraînement (décalage constant, sans TARGET)
        S = cfg["seeds"]
        for s in range(S):
            seed = SEED + 100 * fold + s
            torch.manual_seed(seed)
            model = TowerMLP(specs, cfg)
            if fold == 0 and s == 0:
                print(model)
                print("paramètres :", sum(p.numel() for p in model.parameters()))
            history = train_one(model, xs_tr, y_tr, xs_va, y_va, cfg, device, seed, f"fold {fold} graine {s}")
            b = model.bias.item()
            t_va, t_te, t_tr = predict_towers(model, xs_va), predict_towers(model, xs_te), predict_towers(model, xs_tr)
            shift = float((t_tr[:, 2] + t_tr[:, 3]).mean())
            oof_full[va_idx] += sigmoid(b + t_va.sum(axis=1)) / S
            te_full += sigmoid(b + t_te.sum(axis=1)) / (S * args.folds)
            oof_prev[va_idx] += sigmoid(b + t_va[:, 0] + t_va[:, 1] + shift) / S
            te_prev += sigmoid(b + t_te[:, 0] + t_te[:, 1] + shift) / (S * args.folds)
            towers_oof[va_idx] += t_va / S
            towers_te += t_te / (S * args.folds)
            bias_oof += b / (S * args.folds)
            seeds_info.append({"seed": seed, "epochs": len(history), "best_val_logloss": min(h["val_logloss"] for h in history),
                               "val_auc": round(float(roc_auc_score(y[va_idx], sigmoid(b + t_va.sum(axis=1)))), 5),
                               "bias": round(b, 4), "shift_CD": round(shift, 4),
                               "tower_means_train": [round(float(v), 4) for v in t_tr.mean(axis=0)],
                               "tower_stds_train": [round(float(v), 4) for v in t_tr.std(axis=0)]})
            del model
        # (c) petit réseau A+B réentraîné seul : valeur des facteurs d'amont sans effet d'ablation
        R = cfg["prevention_retrain_seeds"]
        retr_info = []
        for s in range(R):
            seed = SEED + 100 * fold + 50 + s
            torch.manual_seed(seed)
            model = TowerMLP(specs[:2], cfg)
            history = train_one(model, xs_tr[:2], y_tr, xs_va[:2], y_va, cfg, device, seed, f"fold {fold} A+B graine {s}")
            b = model.bias.item()
            t_va, t_te = predict_towers(model, xs_va[:2]), predict_towers(model, xs_te[:2])
            oof_retr[va_idx] += sigmoid(b + t_va.sum(axis=1)) / R
            te_retr += sigmoid(b + t_te.sum(axis=1)) / (R * args.folds)
            retr_info.append({"seed": seed, "epochs": len(history), "best_val_logloss": min(h["val_logloss"] for h in history)})
            del model

        auc = roc_auc_score(y[va_idx], oof_full[va_idx])
        auc_p = roc_auc_score(y[va_idx], oof_prev[va_idx])
        auc_r = roc_auc_score(y[va_idx], oof_retr[va_idx]) if R else None
        print(f"fold {fold}: AUC complet {auc:.4f}, prévention (ablation) {auc_p:.4f}, prévention (A+B réentraîné) "
              f"{auc_r if auc_r is None else round(auc_r, 4)}, {time.time() - tf:.0f} s", flush=True)
        folds_info.append({"auc_full": round(float(auc), 5), "auc_prevention": round(float(auc_p), 5),
                           "auc_prevention_retrained": None if auc_r is None else round(float(auc_r), 5),
                           "seeds": seeds_info, "retrained_seeds": retr_info, "s": round(time.time() - tf)})
        del xs_tr, xs_va, xs_te
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 3. métriques, lecture du modèle, sorties
    metrics = evaluate(y, oof_full)
    prev_metrics = evaluate(y, oof_prev)
    retr_metrics = evaluate(y, oof_retr) if cfg["prevention_retrain_seeds"] else None
    print("prévention (ablation C+D) :", {k: round(v, 4) for k, v in prev_metrics.items()})
    if retr_metrics:
        print("prévention (A+B réentraîné) :", {k: round(v, 4) for k, v in retr_metrics.items()})
    stats = tower_stats(towers_oof, bias_oof)
    profiles = pick_profiles(train, y, towers_oof, bias_oof, oof_full, oof_prev)
    print("tours :", json.dumps(stats["towers"], ensure_ascii=False))
    for p in profiles:
        print("profil :", json.dumps(p, ensure_ascii=False))
    if args.no_save:
        print("--no-save : rien d'écrit")
        return
    info = {
        "model": "mlp_towers", "config": cfg, "device": str(device), "nan_max": NAN_MAX,
        "cat_max_levels": CAT_MAX_LEVELS, "n_columns": len(all_cols),
        "columns_by_tier": {TIER_NAMES[t]: cols[t] for t in "ABCD"},
        "categorical_by_tier": {TIER_NAMES[t]: types[t][0] for t in "ABCD"},
        "derived": {c: TIER_NAMES[t] for c, t in DERIVED_TIER.items()},
        "excluded_absent_or_sparse": excluded,
        "prevention_model": {"method": "même réseau, sigmoid(biais + tour_A + tour_B + moyenne(tour_C + tour_D) sur "
                                       "le fold d'entraînement)", "n_columns": len(cols["A"]) + len(cols["B"]),
                             "columns": cols["A"] + cols["B"], **prev_metrics},
        "tower_decomposition": stats, "profiles": profiles, "folds": folds_info,
    }
    extra = {"_prevention": (oof_prev, te_prev)}
    if retr_metrics:
        info["prevention_retrained"] = {"method": "réseau A+B seul réentraîné, mêmes hyperparamètres, "
                                                  f"{cfg['prevention_retrain_seeds']} graines par fold", **retr_metrics}
        extra["_prevention_retrained"] = (oof_retr, te_retr)
    save_outputs(args.name, test, oof_full, te_full, metrics, info, t0, extra)


if __name__ == "__main__":
    main()
