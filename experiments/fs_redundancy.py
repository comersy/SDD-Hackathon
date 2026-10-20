"""Modèle de référence : tri des colonnes par règles + LightGBM + seuil optimisé pour le F1.

Usage : python train.py [--name NOM] [--threads N] [--folds 5]
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
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

IDEA = ("Étude de SÉLECTION DE COLONNES par redondance (une colonne par concept BRFSS). Base = x_drop_nan95 "
        "(nettoyage des codes de non-réponse, colonnes administratives et vides à plus de 95 % retirées : "
        "214 candidates). Analyse sans TARGET faite une fois sur train : Spearman entre toutes les paires sur "
        "100 000 lignes (lignes communes, min 5 000) + V de Cramér pour les paires à <= 15 modalités ; paire "
        "redondante si |rho| >= 0.9 ou V >= 0.9. Regroupement par concept (variables calculées « _ » dérivées "
        "des brutes : âge, corpulence, tabac, alcool, santé générale, assurance, race, dépistages...) ; dans "
        "chaque groupe on garde la colonne au plus fort gain (LightGBM rapide 200 arbres ajusté sur tout train, "
        "décision structurelle, 2 graines) et une seconde seulement si elle n'est pas redondante avec la première "
        "et apporte >= 20 % du gain de la première ; puis passe globale : toute colonne redondante avec une "
        "colonne déjà gardée de plus fort gain est retirée. Même LightGBM que la base (une graine, early "
        "stopping) sur la liste curée ; variante courte = liste curée privée des colonnes à <= 1 % de gain cumulé.")
SEED = 42
NAN_MAX = 0.95  # colonnes vides à plus de 95 % : retirées
CORR_ROWS = 100_000   # échantillon pour les corrélations
MIN_COMMON = 5_000    # lignes communes minimales pour qu'une corrélation compte
MAX_CARD = 15         # V de Cramér pour les colonnes à <= 15 modalités
REDUNDANT = 0.9       # seuil de redondance (|rho| ou V)
NEAR = 0.8            # paires « presque redondantes », pour information
SECOND_SHARE = 0.2    # 2e colonne d'un groupe gardée si gain >= 20 % de la 1re (et non redondante)
SHORT_CUM_GAIN = 0.01  # variante courte : colonnes à <= 1 % de gain cumulé retirées

# Concepts BRFSS : une variable calculée (« _ ») dérive des brutes du même groupe ; chaque groupe
# garde une colonne (deux si information distincte). Les colonnes absentes des 214 candidates sont ignorées.
GROUPS = {
    "age": ["_AGE80", "_AGEG5YR", "_AGE_G", "_AGE65YR"],
    "sexe": ["SEXVAR", "_SEX", "BIRTHSEX"],
    "imc": ["_BMI5", "_BMI5CAT", "_RFBMI5"],
    "poids": ["WTKG3", "WEIGHT2"],
    "taille": ["HTM4", "HTIN4", "HEIGHT3"],
    "tabac_statut": ["_SMOKER3", "_RFSMOK3", "_SMOKGRP", "SMOKE100", "SMOKDAY2", "LASTSMK2", "STOPSMK2"],
    "tabac_historique": ["_PACKYRS", "_YRSSMOK", "_PACKDAY", "_YRSQUIT", "LCSNUMCG", "LCSFIRST", "LCSLAST"],
    "ecig": ["ECIGNOW2", "_CURECI2"],
    "depistage_poumon": ["LCSCTSC1", "LCSSCNCR", "LCSCTWHN", "_LCSREC"],
    "alcool_frequence": ["ALCDAY4", "DRNKANY6", "DROCDY4_"],
    "alcool_quantite": ["AVEDRNK3", "_DRNKWK2", "_RFDRHV8"],
    "alcool_excès": ["DRNK3GE5", "_RFBING6", "MAXDRNKS"],
    "sante_generale": ["GENHLTH", "_RFHLTH"],
    "sante_physique": ["PHYSHLTH", "_PHYS14D"],
    "sante_mentale": ["MENTHLTH", "_MENT14D"],
    "assurance": ["PRIMINSR", "_HLTHPLN", "_HCVU652"],
    "activite_physique": ["EXERANY2", "_TOTINDA"],
    "arthrite": ["HAVARTH4", "_DRDXAR2"],
    "asthme": ["ASTHMA3", "ASTHNOW", "_LTASTH1", "_CASTHM1", "_ASTHMS1"],
    "dents": ["RMVTETH4", "_EXTETH3", "_ALTETH3"],
    "dentiste": ["LASTDEN4", "_DENVST3"],
    "race_ethnie": ["_PRACE2", "_MRACE2", "_HISPANC", "_RACE1", "_RACEG22", "_RACEGR4", "_RACEPR1", "_IMPRACE"],
    "education": ["EDUCA", "_EDUCAG"],
    "revenu": ["INCOME3", "_INCOMG1"],
    "enfants": ["CHILDREN", "_CHLDCNT"],
    "vaccin_grippe": ["FLUSHOT7", "FLSHTMY3", "_FLSHOT7"],
    "vaccin_pneumo": ["PNEUVAC4", "_PNEUMO3"],
    "vih": ["HIVTST7", "HIVTSTD3", "_AIDTST4"],
    "depistage_colorectal": ["HADSIGM4", "COLNSIGM", "COLNTES1", "SIGMTES1", "LASTSIG4", "COLNCNCR", "VIRCOLO1",
                             "VCLNTES2", "SMALSTOL", "STOLTEST", "STOOLDN2", "BLDSTFIT", "SDNATES1", "_HADCOLN",
                             "_CLNSCP1", "_HADSIGM", "_SGMSCP1", "_SGMS101", "_RFBLDS5", "_STOLDN1", "_VIRCOL1",
                             "_SBONTI1", "_CRCREC2"],
    "mammographie": ["HADMAM", "HOWLONG", "_RFMAM22", "_MAM5023"],
    "col_uterus": ["CERVSCRN", "CRVCLCNC", "CRVCLPAP", "CRVCLHPV", "HADHYST2"],
    "metropole": ["_METSTAT", "_URBSTAT", "MSCODE"],
    "menage": ["NUMADULT", "NUMMEN", "NUMWOMEN", "HHADULT"],
    "telephones": ["NUMHHOL4", "NUMPHON4", "CPDEMO1C"],
    "diabete": ["DIABETE4", "DIABAGE4", "PDIABTS1", "PREDIAB2", "DIABTYPE", "INSULIN1", "CHKHEMO3",
                "EYEEXAM1", "DIABEYE1", "DIABEDU1", "FEETSORE"],
    "covid_infection": ["COVIDPOS", "COVIDSMP", "COVIDPRM"],
    "covid_vaccin": ["COVIDVA1", "COVIDNU1", "COVIDFS1", "COVIDSE1"],
    "cancer_survivant": ["CNCRDIFF", "CNCRAGE", "CNCRTYP2"],
    "aidant": ["CAREGIV1", "CRGVEXPT"],
    "ace_enfance": ["ACEDEPRS", "ACEDRINK", "ACEDRUGS", "ACEPRISN", "ACEDIVRC", "ACEPUNCH", "ACEHURT1",
                    "ACESWEAR", "ACETOUCH", "ACETTHEM", "ACEHVSEX", "ACEADSAF", "ACEADNED"],
    "social_sdoh": ["LSATISFY", "EMTSUPRT", "SDHISOLT", "SDHEMPLY", "FOODSTMP", "SDHFOOD1", "SDHBILLS",
                    "SDHUTILS", "SDHTRNSP", "SDHSTRE1"],
    "alcool_conseil_asbi": ["ASBIALCH", "ASBIDRNK", "ASBIBING", "ASBIADVC", "ASBIRDUC"],
    "reactions_race": ["RRCLASS3", "RRCOGNT2", "RRTREAT", "RRATWRK2", "RRHCARE4", "RRPHYSM2"],
    "orientation_sexuelle": ["SOMALE", "SOFEMALE", "TRNSGNDR"],
    "enfant_selectionne": ["_CHISPNC", "_CRACE2", "_CPRACE2", "CAGEG", "RCSGEND1", "RCSRLTN2", "CASTHDX2"],
}


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
    # Colonnes presque vides (modules optionnels du BRFSS) : critère sans TARGET, calculé sur train
    nan_rate = train[feats].isna().mean()
    sparse = sorted(nan_rate[nan_rate > NAN_MAX].index)
    feats = [c for c in feats if c not in sparse]
    print(f"{len(feats)} colonnes gardées, {len(DROP_COLS)} retirées par règle, "
          f"{len(constant)} constantes : {constant}, "
          f"{len(sparse)} vides à plus de {NAN_MAX:.0%} : {sparse}")
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


def save_outputs(name, test, test_proba, metrics, info, t0, stamp=None):
    """Écrit la soumission et le JSON de résultats ; retourne le chemin du JSON."""
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
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


# --- Étude de redondance (sans TARGET, faite une fois sur un échantillon de train) ---------------
def spearman_matrix(df):
    """Spearman pairwise sur les lignes communes à chaque paire (rangs calculés par colonne,
    NaN conservés). Produits matriciels : exact pour le Pearson des rangs sur lignes communes."""
    R = df.rank().to_numpy(dtype=np.float64)
    M = (~np.isnan(R)).astype(np.float64)
    X = np.nan_to_num(R)
    n = M.T @ M                      # lignes communes
    sx = X.T @ M                     # sx[i, j] = somme de x_i sur les lignes communes à (i, j)
    sxx = (X ** 2).T @ M
    sxy = X.T @ X
    with np.errstate(divide="ignore", invalid="ignore"):
        mx, my = sx / n, sx.T / n
        cov = sxy / n - mx * my
        vx, vy = sxx / n - mx ** 2, sxx.T / n - my ** 2
        rho = cov / np.sqrt(vx * vy)
    rho[n < MIN_COMMON] = np.nan
    np.fill_diagonal(rho, np.nan)
    # Les rangs globaux sous-estiment le Spearman des paires à NaN asymétriques (ex. SMOKDAY2 /
    # _SMOKER3 : 0.95 au lieu de 1.0) : recalcul exact (re-rangs sur lignes communes) des paires
    # candidates, |rho| >= 0.6.
    cols = list(df.columns)
    ii, jj = np.nonzero(np.triu(np.abs(np.nan_to_num(rho)) >= 0.6, k=1))
    for i, j in zip(ii, jj):
        sub = df.iloc[:, [i, j]].dropna()
        rho[i, j] = rho[j, i] = float(sub.iloc[:, 0].corr(sub.iloc[:, 1], method="spearman"))
    print(f"  {len(ii)} paires |rho| >= 0.6 recalculées exactement (rangs sur lignes communes)")
    return rho, n


def cramer_matrix(df, cols):
    """V de Cramér entre toutes les paires de colonnes à peu de modalités, via une matrice
    one-hot creuse : O.T @ O donne toutes les tables de contingence d'un coup."""
    n_rows = len(df)
    rows, idx, offsets, sizes = [], [], [], []
    off = 0
    for c in cols:
        codes, uniq = pd.factorize(df[c], sort=True)   # NaN -> -1
        ok = codes >= 0
        rows.append(np.nonzero(ok)[0])
        idx.append(off + codes[ok])
        offsets.append(off)
        sizes.append(len(uniq))
        off += len(uniq)
    rows, idx = np.concatenate(rows), np.concatenate(idx)
    O = sp.csr_matrix((np.ones(len(rows)), (rows, idx)), shape=(n_rows, off))
    G = (O.T @ O).toarray()
    k = len(cols)
    V = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(i + 1, k):
            C = G[offsets[i]:offsets[i] + sizes[i], offsets[j]:offsets[j] + sizes[j]]
            C = C[C.sum(1) > 0][:, C.sum(0) > 0]
            n = C.sum()
            if n < MIN_COMMON or min(C.shape) < 2:
                continue
            E = np.outer(C.sum(1), C.sum(0)) / n
            chi2 = ((C - E) ** 2 / E).sum()
            V[i, j] = V[j, i] = float(np.sqrt(chi2 / (n * (min(C.shape) - 1))))
    return V


def quick_gain(X, y, threads, seeds=(SEED, SEED + 1)):
    """Gain LightGBM rapide (200 arbres, lr 0.1) ajusté sur TOUT train : sert uniquement à
    ordonner les colonnes d'un même concept (décision structurelle, signalée dans les sorties)."""
    gain = pd.Series(0.0, index=X.columns)
    for s in seeds:
        params = {**LGB_PARAMS, "learning_rate": 0.1, "num_threads": threads, "seed": s}
        m = lgb.train(params, lgb.Dataset(X, y), num_boost_round=200)
        gain += m.feature_importance("gain")
    return gain / len(seeds)


def redundancy_study(train, feats, threads):
    """Retourne (final_list, gain, study) : liste curée, gains rapides et détails (paires, groupes)."""
    y = train["TARGET"].values.astype(int)
    t = time.time()
    gain = quick_gain(train[feats], y, threads)
    print(f"gain LightGBM rapide : {time.time() - t:.0f} s")
    nuniq = train[feats].nunique()
    nan_rate = train[feats].isna().mean()

    sample = train[feats].sample(n=min(CORR_ROWS, len(train)), random_state=SEED)
    t = time.time()
    rho, n_common = spearman_matrix(sample)
    print(f"Spearman {len(feats)}x{len(feats)} : {time.time() - t:.0f} s")
    low = [c for c in feats if nuniq[c] <= MAX_CARD]
    t = time.time()
    V_low = cramer_matrix(sample, low)
    print(f"Cramér sur {len(low)} colonnes à <= {MAX_CARD} modalités : {time.time() - t:.0f} s")
    pos = {c: i for i, c in enumerate(feats)}
    lpos = {c: i for i, c in enumerate(low)}

    def assoc(a, b):
        """(valeur max, type) entre deux colonnes : spearman |rho| ou V de Cramér."""
        r = rho[pos[a], pos[b]]
        best = (abs(float(r)) if not np.isnan(r) else -1.0, "spearman")
        if a in lpos and b in lpos and not np.isnan(V_low[lpos[a], lpos[b]]):
            v = float(V_low[lpos[a], lpos[b]])
            if v > best[0]:
                best = (v, "cramer")
        return best

    # Toutes les paires redondantes / presque redondantes
    redundant, near = {}, []
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            v, kind = assoc(a, b)
            if v >= REDUNDANT:
                redundant[frozenset((a, b))] = (kind, v)
            elif v >= NEAR:
                near.append([a, b, kind, round(v, 3)])
    print(f"{len(redundant)} paires redondantes (>= {REDUNDANT}), {len(near)} presque (>= {NEAR})")

    def is_red(a, b):
        return frozenset((a, b)) in redundant

    def rank_key(c):  # plus de gain, puis continue (plus de modalités), puis moins de NaN
        return (-gain[c], -nuniq[c], nan_rate[c])

    # (2)-(3) : une ou deux colonnes par concept
    groups, kept_by_group, grouped = {}, set(), set()
    for concept, cols in GROUPS.items():
        cols = [c for c in cols if c in pos]
        if not cols:
            continue
        grouped.update(cols)
        order = sorted(cols, key=rank_key)
        kept = [order[0]]
        why = [f"{order[0]} : gain max du groupe ({gain[order[0]]:.0f})"]
        for c in order[1:]:
            share = gain[c] / gain[order[0]] if gain[order[0]] > 0 else 0
            if any(is_red(c, k) for k in kept):
                continue
            if share >= SECOND_SHARE:
                kept.append(c)
                why.append(f"{c} : information distincte (non redondante, gain {gain[c]:.0f} = "
                           f"{share:.0%} de la 1re)")
                break
        dropped = [c for c in cols if c not in kept]
        parts = []
        for c in dropped:
            v, kind = assoc(c, kept[0])
            parts.append(f"{c} (gain {gain[c]:.0f}" + (f", {kind} {v:.2f} avec {kept[0]}" if is_red(c, kept[0]) else "") + ")")
        why.append("retirées : " + ", ".join(parts))
        groups[concept] = {"kept": kept, "dropped": dropped, "why": " ; ".join(why)}
        kept_by_group.update(kept)
        print(f"  {concept:22s} garde {kept}  retire {len(dropped)}")

    singles = [c for c in feats if c not in grouped]
    candidates = sorted(list(kept_by_group) + singles, key=rank_key)
    # Passe globale : paires redondantes entre concepts différents / singletons
    final, cross_dropped = [], {}
    for c in candidates:
        hit = [k for k in final if is_red(c, k)]
        if hit:
            cross_dropped[c] = hit[0]
        else:
            final.append(c)
    print(f"{len(singles)} colonnes hors groupe, {len(kept_by_group)} gardées par les groupes, "
          f"{len(cross_dropped)} retirées par la passe globale : {cross_dropped}")

    final_set = set(final)
    pairs = []
    for key, (kind, v) in sorted(redundant.items(), key=lambda kv: -kv[1][1]):
        a, b = sorted(key, key=rank_key)
        kept_col = a if a in final_set else (b if b in final_set else None)
        pairs.append([a, b, kind, round(v, 4), kept_col])

    study = {
        "n_candidates": len(feats),
        "n_final": len(final),
        "final_list": final,
        "gain_source": "LightGBM rapide 200 arbres lr 0.1, 2 graines, ajusté sur TOUT train (décision "
                       "structurelle de sélection, signalée : léger risque d'optimisme)",
        "gain_quick": {c: round(float(gain[c]), 1) for c in sorted(feats, key=rank_key)},
        "n_redundant_pairs": len(pairs),
        "pairs": pairs,
        "near_pairs": sorted(near, key=lambda p: -p[3]),
        "groups": groups,
        "singletons": singles,
        "cross_group_dropped": cross_dropped,
        "thresholds": {"redundant": REDUNDANT, "near": NEAR, "corr_rows": int(len(sample)),
                       "min_common": MIN_COMMON, "max_card_cramer": MAX_CARD, "second_share": SECOND_SHARE},
    }
    return final, gain, study


def run_cv(X, y, X_test, params, folds):
    oof = np.zeros(len(X))
    test_proba = np.zeros(len(X_test))
    importance = pd.Series(0.0, index=X.columns)
    iters = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        model = lgb.train(
            params,
            lgb.Dataset(X.iloc[tr_idx], y[tr_idx]),
            num_boost_round=5000,
            valid_sets=[lgb.Dataset(X.iloc[va_idx], y[va_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        test_proba += model.predict(X_test, num_iteration=model.best_iteration) / folds
        importance += model.feature_importance("gain") / folds
        iters.append(int(model.best_iteration))
        print(f"fold {fold}: {model.best_iteration} arbres, "
              f"AUC {roc_auc_score(y[va_idx], oof[va_idx]):.4f}")
    return oof, test_proba, importance, iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="fs_redundancy")
    parser.add_argument("--threads", type=int, default=0, help="0 = tous les cœurs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--debug-rows", type=int, default=0, help="test local : sous-échantillon de train")
    args = parser.parse_args()

    t0 = time.time()
    train, test = load()
    if args.debug_rows:
        train = train.sample(n=args.debug_rows, random_state=SEED).reset_index(drop=True)
    feats = select_columns(train)
    maxes = train[feats].max()
    clean(train, feats, maxes)
    clean(test, feats, maxes)
    y = train["TARGET"].values.astype(int)

    final, gain, study = redundancy_study(train, feats, args.threads)
    print(f"Liste curée : {len(final)} colonnes sur {len(feats)}")

    params = {**LGB_PARAMS, "num_threads": args.threads}
    print(f"=== LightGBM sur la liste curée ({len(final)} colonnes) ===")
    oof, test_proba, importance, iters = run_cv(train[final], y, test[final], params, args.folds)
    metrics = evaluate(y, oof)
    top = importance.sort_values(ascending=False)
    print("CV liste curée :", {k: round(v, 4) for k, v in metrics.items()})

    # Variante courte : colonnes à <= 1 % de gain cumulé (gain du modèle final évalué) retirées
    share = (top / top.sum()).sort_values()
    tail = share[share.cumsum() <= SHORT_CUM_GAIN].index.tolist()
    short = [c for c in final if c not in tail]
    print(f"=== Variante courte : {len(short)} colonnes (retirées à <= {SHORT_CUM_GAIN:.0%} "
          f"de gain cumulé : {tail}) ===")
    oof_s, test_proba_s, importance_s, iters_s = run_cv(train[short], y, test[short], params, args.folds)
    metrics_s = evaluate(y, oof_s)
    print("CV variante courte :", {k: round(v, 4) for k, v in metrics_s.items()})

    stamp = time.strftime("%Y%m%d_%H%M%S")
    (ROOT / "submissions").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    sub_short = ROOT / "submissions" / f"submission_{args.name}_short_{stamp}.csv"
    pd.DataFrame({"ID": test["ID"], "TARGET": (test_proba_s >= metrics_s["threshold"]).astype(int)}
                 ).to_csv(sub_short, index=False)

    study["cv_curated"] = metrics
    study["best_iterations_curated"] = iters
    study["top_features_curated"] = top.round(0).to_dict()
    study["short_variant"] = {"n_columns": len(short), "dropped": tail, "cv": metrics_s,
                              "best_iterations": iters_s, "columns": short,
                              "top_features": importance_s.sort_values(ascending=False).round(0).to_dict(),
                              "submission": f"submissions/{sub_short.name}"}
    study_path = ROOT / "results" / f"{args.name}_{stamp}_study.json"
    study_path.write_text(json.dumps(study, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Étude : {study_path}")

    info = {
        "model": "lightgbm", "params": params,
        "n_features": len(final), "n_candidates": len(feats),
        "columns_per_fold": [len(final)] * args.folds,  # sélection structurelle : identique dans chaque fold
        "final_list": final,
        "n_redundant_pairs": study["n_redundant_pairs"],
        "groups_summary": {k: {"kept": v["kept"], "n_dropped": len(v["dropped"])}
                           for k, v in study["groups"].items()},
        "cross_group_dropped": study["cross_group_dropped"],
        "gain_source": study["gain_source"],
        "short_variant": {"n_columns": len(short), "cv": metrics_s, "dropped": tail,
                          "submission": f"submissions/{sub_short.name}"},
        "study_json": f"results/{study_path.name}",
        "top_features": top.head(20).round(0).to_dict(),
    }
    save_outputs(args.name, test, test_proba, metrics, info, t0, stamp=stamp)


if __name__ == "__main__":
    main()
