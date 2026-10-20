"""Modèle « intelligent » XGBoost GPU : colonnes défendables (niveaux A-D de features/tiers.json),
variables dérivées lisibles, contraintes monotones sur les facteurs d'amont, 3 graines par fold,
et un second modèle PRÉVENTION (A+B seulement) évalué dans le même script.

Usage : python experiments/smart_xgb.py [--name NOM] [--threads N] [--folds 5]
                                        [--device cuda|cpu] [--sample N] [--rounds N]
Sorties : submissions/submission_<nom>_<horodatage>.csv, results/<nom>_<horodatage>.json,
          results/<nom>_<horodatage>_oof.csv, results/<nom>_<horodatage>_test.csv,
          results/<nom>_prevention_<horodatage>_oof.csv, results/<nom>_prevention_<horodatage>_test.csv

Protocole d'évaluation, à conserver à l'identique dans toute expérience pour que les scores
restent comparables :
- entraînement sur train.csv + extra.csv concaténés (370 132 lignes), test.csv sert
  uniquement à prédire ;
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42), prédictions out-of-fold ;
- tout prétraitement qui utilise TARGET est ajusté dans chaque fold sur la partie
  entraînement seulement (ici : aucun ; toutes les transformations sont sans TARGET, faites une fois) ;
- le seuil de décision est choisi sur les prédictions OOF pour maximiser le F1 ;
  le F1 reporté est ce F1 OOF, au seuil retenu.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# load / select_columns / clean : repris tels quels de la base r1_clean_codes (même dossier)
from r1_clean_codes import clean, load, select_columns

IDEA = ("[structure de x_lgbbag_proba : folds, seuil OOF, save_outputs avec oof=] "
        "Modèle « intelligent » : seules les colonnes des niveaux A_amont / B_comorbidites / "
        "C_parcours_consequences / D_contexte de features/tiers.json (listes embarquées), intersectées avec "
        "select_columns() et sans les colonnes vides à > 95 % (STOPSMK2 exclue) ; clean() de r1_clean_codes, "
        "plus _DRNKWK2 == 99900 (code inconnu) -> NaN. Dérivées sans TARGET, faites une fois sur train et test : "
        "SMOKING_STATUS (0 jamais / 1 ancien / 2 occasionnel / 3 quotidien), BMI_CALC depuis WEIGHT2/HEIGHT3 "
        "nettoyés, DRINKS_WEEK = ALCDAY4/30*7 x AVEDRNK3, DIABETES_BIN, DIAB_YEARS = _AGE80 - DIABAGE4, N_COMORB "
        "(somme des B), N_LIMITS (DIFFWALK/DIFFDRES/DIFFALON/DECIDE), CARE_INTENSITY (PERSDOC3 1-2, CHECKUP1 1, "
        "FLUSHOT7 1, PNEUVAC4 1) ; colonnes brutes conservées ; NaN gardés pour les arbres (l'absence d'un module "
        "ou d'une question conditionnelle a un sens). XGBoost GPU (device=cuda, hist, binary:logistic, logloss, "
        "max_depth 7, lr 0.05, min_child_weight 20, subsample 0.8, colsample_bytree 0.5, reg_lambda 5, early "
        "stopping 100, 5000 arbres max) avec monotone_constraints croissantes sur _AGE80, _PACKYRS, _YRSSMOK, "
        "_PACKDAY, _BMI5, WEIGHT2, MENTHLTH, _DRNKWK2 et DIABETES_BIN (= « DIABAGE4_present » de tiers.json, "
        "mêmes colonnes que smart_lgbm) ; 3 graines (42, 1, 2) par fold, chacune avec son early stopping ; OOF et "
        "test = moyenne 3 graines x 5 folds. Hyperparamètres prescrits (aucun results/hp_xgb_optuna_*.json "
        "disponible au lancement). Deux évaluations : modèle COMPLET A+B+C+D (soumis) et modèle PRÉVENTION A+B "
        "(mêmes hyperparamètres, mêmes folds), dont les métriques OOF vont sous \"prevention_model\" et les "
        "OOF/test CSV sous <nom>_prevention")
SEED = 42
SEEDS = [42, 1, 2]

# --- Niveaux de colonnes (copie de features/tiers.json ; le conteneur Modal n'a pas features/) ---
TIERS = {
    "A_amont": ["_AGE80", "SEXVAR", "SMOKE100", "SMOKDAY2", "_SMOKER3", "_PACKYRS", "_YRSSMOK", "_PACKDAY",
                "USENOW3", "ECIGNOW2", "_CURECI2", "WEIGHT2", "HEIGHT3", "_BMI5", "DIABETE4", "DIABAGE4",
                "PREDIAB2", "EXERANY2", "ALCDAY4", "AVEDRNK3", "DRNK3GE5", "MAXDRNKS", "_DRNKWK2", "SLEPTIM1",
                "MENTHLTH", "ADDEPEV3", "SDHSTRE1", "PREGNANT"],
    "B_comorbidites": ["CVDSTRK3", "CHCKDNY2", "CHCCOPD3", "HAVARTH4", "ASTHMA3", "ASTHNOW", "CHCSCNC1",
                       "CHCOCNC1", "DEAF", "BLIND"],
    "C_parcours_consequences": ["GENHLTH", "PHYSHLTH", "POORHLTH", "DIFFWALK", "DIFFDRES", "DIFFALON", "DECIDE",
                                "EMPLOY1", "PERSDOC3", "CHECKUP1", "MEDCOST1", "PRIMINSR", "FLUSHOT7", "FLSHTMY3",
                                "PNEUVAC4", "TETANUS1", "COVIDVA1", "_YRSQUIT", "LASTSMK2", "STOPSMK2",
                                "LCSCTSC1", "LCSSCNCR", "LCSFIRST", "LCSLAST", "LCSNUMCG", "LASTDEN4", "RMVTETH4",
                                "_CRCREC2", "_MAM5023", "HOWLONG"],
    "D_contexte": ["INCOME3", "EDUCA", "MARITAL", "_RACE1", "_HISPANC", "_STATE", "_METSTAT", "_URBSTAT",
                   "HHADULT", "CHILDREN", "VETERAN3", "RENTHOM1", "SDHFOOD1", "SDHBILLS", "SDHISOLT", "EMTSUPRT",
                   "LSATISFY"],
}
# Niveau des variables dérivées
DERIVED_TIER = {
    "SMOKING_STATUS": "A_amont", "BMI_CALC": "A_amont", "DRINKS_WEEK": "A_amont",
    "DIABETES_BIN": "A_amont", "DIAB_YEARS": "A_amont",
    "N_COMORB": "B_comorbidites",
    "N_LIMITS": "C_parcours_consequences", "CARE_INTENSITY": "C_parcours_consequences",
}
PREVENTION_TIERS = ["A_amont", "B_comorbidites"]
# Contraintes monotones croissantes (tiers.json « monotone_increasing » ; DIABAGE4_present -> DIABETES_BIN)
MONOTONE_INC = ["_AGE80", "_PACKYRS", "_YRSSMOK", "_PACKDAY", "_BMI5", "WEIGHT2", "MENTHLTH", "_DRNKWK2",
                "DIABETES_BIN"]
EMPTY_MAX = 0.95  # colonnes vides à plus de 95 % (sur train, sans TARGET) : ignorées

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cuda",
    "max_depth": 7,
    "learning_rate": 0.05,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "reg_lambda": 5,
    "seed": SEED,
}
NUM_BOOST_ROUND = 5000
EARLY_STOPPING = 100


def find_root():
    """Dossier du dépôt (celui qui contient train.csv), quel que soit l'emplacement du script."""
    here = Path(__file__).resolve().parent
    for d in [here, *here.parents]:
        if (d / "train.csv").exists():
            return d
    raise FileNotFoundError("train.csv introuvable : lance le script depuis le dépôt")


ROOT = find_root()


# --- Colonnes et dérivées (transformations sans TARGET, faites une fois sur train et test) ----------
def tier_columns(train, feats):
    """Colonnes retenues par niveau : tiers.json ∩ select_columns() ∩ non vides à > 95 %."""
    kept, dropped = {}, {}
    for tier, cols in TIERS.items():
        kept[tier] = [c for c in cols if c in feats and train[c].isna().mean() <= EMPTY_MAX]
        dropped[tier] = [c for c in cols if c not in kept[tier]]
    return kept, dropped


def add_derived(df, tiers_kept):
    """Variables dérivées, calculées sur les colonnes nettoyées par clean(). Codes inconnus -> NaN."""
    f32 = "float32"
    # _DRNKWK2 : 99900 = ne sait pas / refus (laissé intact par clean) -> NaN
    df["_DRNKWK2"] = df["_DRNKWK2"].mask(df["_DRNKWK2"] >= 99900, np.nan).astype(f32)

    # SMOKING_STATUS : 0 jamais (SMOKE100 == 2), 1 ancien (SMOKDAY2 == 3), 2 occasionnel (2), 3 quotidien (1)
    s100, sday = df["SMOKE100"], df["SMOKDAY2"]
    status = pd.Series(np.nan, index=df.index, dtype=f32)
    status = status.mask(s100 == 2, 0)
    status = status.mask((s100 == 1) & (sday == 3), 1)
    status = status.mask((s100 == 1) & (sday == 2), 2)
    status = status.mask((s100 == 1) & (sday == 1), 3)
    df["SMOKING_STATUS"] = status.astype(f32)

    # BMI_CALC : WEIGHT2 en livres, HEIGHT3 en pouces (après clean) ; plages plausibles sinon NaN
    w_kg = df["WEIGHT2"] * 0.45359237
    h_m = df["HEIGHT3"] * 0.0254
    bmi = w_kg / (h_m ** 2)
    bmi = bmi.mask(~(df["HEIGHT3"].between(36, 96) & df["WEIGHT2"].between(50, 776)), np.nan)
    df["BMI_CALC"] = bmi.astype(f32)

    # DRINKS_WEEK : ALCDAY4 (jours/mois après clean) / 30 * 7 x AVEDRNK3 ; 0 si ne boit pas
    days = df["ALCDAY4"]
    dw = days / 30 * 7 * df["AVEDRNK3"]
    dw = dw.mask(days == 0, 0)
    df["DRINKS_WEEK"] = dw.astype(f32)

    # DIABETES_BIN et DIAB_YEARS (NaN si pas diabétique ou âge de diagnostic inconnu)
    diab = df["DIABETE4"]
    df["DIABETES_BIN"] = (diab == 1).astype(f32).mask(diab.isna(), np.nan).astype(f32)
    years = (df["_AGE80"] - df["DIABAGE4"]).clip(lower=0)
    df["DIAB_YEARS"] = years.mask(df["DIABETES_BIN"] != 1, np.nan).astype(f32)

    # N_COMORB : nombre de comorbidités B déclarées (== 1)
    b_cols = tiers_kept["B_comorbidites"]
    df["N_COMORB"] = (df[b_cols] == 1).sum(axis=1).astype(f32)

    # N_LIMITS : limitations fonctionnelles (niveau C)
    lim_cols = [c for c in ["DIFFWALK", "DIFFDRES", "DIFFALON", "DECIDE"] if c in df.columns]
    df["N_LIMITS"] = (df[lim_cols] == 1).sum(axis=1).astype(f32)

    # CARE_INTENSITY : intensité du suivi médical (niveau C)
    care = (df["PERSDOC3"].isin([1, 2]).astype(int) + (df["CHECKUP1"] == 1).astype(int)
            + (df["FLUSHOT7"] == 1).astype(int) + (df["PNEUVAC4"] == 1).astype(int))
    df["CARE_INTENSITY"] = care.astype(f32)
    return df


def build_features(train, test):
    """Sélection, nettoyage, dérivées. Retourne (colonnes par niveau, colonnes exclues par niveau)."""
    feats = select_columns(train)
    maxes = train[feats].max()
    clean(train, feats, maxes)
    clean(test, feats, maxes)
    tiers_kept, tiers_dropped = tier_columns(train, feats)
    for df in (train, test):
        add_derived(df, tiers_kept)
    for name, tier in DERIVED_TIER.items():
        tiers_kept[tier].append(name)
    return tiers_kept, tiers_dropped


def columns_for(tiers_kept, tiers):
    return [c for t in tiers for c in tiers_kept[t]]


# --- Évaluation et sorties (copie de x_lgbbag_proba, avec horodatage partagé) ----------------------
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


def save_proba(name, stamp, test, test_proba, oof):
    """Probabilités OOF et test pour les blends (ordre OOF = train.csv puis extra.csv)."""
    pd.DataFrame({"oof": oof}).to_csv(ROOT / "results" / f"{name}_{stamp}_oof.csv", index=False, float_format="%.6f")
    pd.DataFrame({"ID": test["ID"], "proba": test_proba}).to_csv(
        ROOT / "results" / f"{name}_{stamp}_test.csv", index=False, float_format="%.6f")


def save_outputs(name, test, test_proba, metrics, info, t0, oof=None, stamp=None):
    """Écrit la soumission et le JSON de résultats ; retourne le chemin du JSON."""
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    (ROOT / "submissions").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    sub_path = ROOT / "submissions" / f"submission_{name}_{stamp}.csv"
    pred = (test_proba >= metrics["threshold"]).astype(int)
    pd.DataFrame({"ID": test["ID"], "TARGET": pred}).to_csv(sub_path, index=False)
    if oof is not None:
        save_proba(name, stamp, test, test_proba, oof)
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


# --- Entraînement ----------------------------------------------------------------------------------
def run_model(label, X, y, X_test, params, folds, rounds):
    """CV 5 folds x 3 graines ; retourne oof, test_proba, importance (gain), best_iterations."""
    cols = list(X.columns)
    constraints = tuple(1 if c in MONOTONE_INC else 0 for c in cols)
    params = {**params, "monotone_constraints": constraints}
    print(f"[{label}] {len(cols)} colonnes, contraintes monotones sur "
          f"{[c for c in cols if c in MONOTONE_INC]}")
    dtest = xgb.DMatrix(X_test, feature_names=cols, missing=np.nan)
    n_seeds = len(SEEDS)
    oof_seeds = np.zeros((n_seeds, len(X)))
    test_proba = np.zeros(len(X_test))
    importance = pd.Series(0.0, index=cols)
    best_iters = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        dtr = xgb.DMatrix(X.iloc[tr_idx], y[tr_idx], feature_names=cols, missing=np.nan)
        dva = xgb.DMatrix(X.iloc[va_idx], y[va_idx], feature_names=cols, missing=np.nan)
        fold_iters = []
        for k, seed in enumerate(SEEDS):
            tf = time.time()
            model = xgb.train({**params, "seed": seed}, dtr, num_boost_round=rounds,
                              evals=[(dva, "valid")], early_stopping_rounds=EARLY_STOPPING, verbose_eval=False)
            rng = (0, model.best_iteration + 1)
            oof_seeds[k, va_idx] = model.predict(dva, iteration_range=rng)
            test_proba += model.predict(dtest, iteration_range=rng) / (n_seeds * folds)
            gain = model.get_score(importance_type="gain")
            importance += pd.Series(gain).reindex(cols).fillna(0.0) / (n_seeds * folds)
            fold_iters.append(int(model.best_iteration))
            print(f"[{label}] fold {fold} seed {seed}: {model.best_iteration} arbres, "
                  f"AUC {roc_auc_score(y[va_idx], oof_seeds[k, va_idx]):.4f} ({time.time() - tf:.0f} s)", flush=True)
            del model
        best_iters.append(fold_iters)
        print(f"[{label}] fold {fold} moyenne {n_seeds} graines: AUC "
              f"{roc_auc_score(y[va_idx], oof_seeds[:, va_idx].mean(axis=0)):.4f}", flush=True)
        del dtr, dva
    oof = oof_seeds.mean(axis=0)
    m1 = evaluate(y, oof_seeds[0])
    print(f"[{label}] graine 42 seule : F1 {m1['f1']:.4f} AUC {m1['auc']:.4f}")
    return oof, test_proba, importance, best_iters, m1


def tier_of(col, tiers_kept):
    for t, cols in tiers_kept.items():
        if col in cols:
            return t
    return "?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="smart_xgb")
    parser.add_argument("--threads", type=int, default=0, help="0 = tous les cœurs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cuda", help="cuda (Modal) ou cpu (test local)")
    parser.add_argument("--sample", type=int, default=0, help="test local : n lignes de train (0 = tout)")
    parser.add_argument("--rounds", type=int, default=NUM_BOOST_ROUND)
    args = parser.parse_args()

    t0 = time.time()
    train, test = load()
    if args.sample:
        train = train.sample(args.sample, random_state=SEED).reset_index(drop=True)
        test = test.head(500).reset_index(drop=True)
    tiers_kept, tiers_dropped = build_features(train, test)
    for t in TIERS:
        print(f"{t}: {len(tiers_kept[t])} colonnes {tiers_kept[t]}\n   exclues : {tiers_dropped[t]}")
    for c in DERIVED_TIER:
        s = train[c]
        print(f"{c}: NaN={s.isna().mean():.3f} min={s.min():.2f} max={s.max():.2f} mean={s.mean():.3f}")

    full_cols = columns_for(tiers_kept, list(TIERS))
    prev_cols = columns_for(tiers_kept, PREVENTION_TIERS)
    y = train["TARGET"].values.astype(int)
    print(f"train {train[full_cols].shape}, test {test[full_cols].shape}, taux de positifs {y.mean():.3f} ; "
          f"modèle prévention : {len(prev_cols)} colonnes")
    params = {**XGB_PARAMS, "device": args.device, "nthread": args.threads or None}
    params = {k: v for k, v in params.items() if v is not None}

    # (a) modèle COMPLET A+B+C+D (soumis)
    oof, test_proba, importance, best_iters, m_seed42 = run_model(
        "complet", train[full_cols], y, test[full_cols], params, args.folds, args.rounds)
    metrics = evaluate(y, oof)
    print(f"[complet] 3 graines : F1 {metrics['f1']:.4f} AUC {metrics['auc']:.4f} seuil {metrics['threshold']:.3f}")
    top = importance.sort_values(ascending=False).head(25)
    print("Top 25 variables (gain moyen) :")
    print(pd.DataFrame({"gain": top.round(1), "niveau": [tier_of(c, tiers_kept) for c in top.index]}).to_string())
    gain_by_tier = importance.groupby(importance.index.map(lambda c: tier_of(c, tiers_kept))).sum()
    gain_share = (gain_by_tier / gain_by_tier.sum()).round(4).to_dict()
    print("Part du gain par niveau :", gain_share)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    # (b) modèle PRÉVENTION A+B (mêmes hyperparamètres, mêmes folds)
    oof_p, test_p, imp_p, iters_p, mp_seed42 = run_model(
        "prevention", train[prev_cols], y, test[prev_cols], params, args.folds, args.rounds)
    metrics_p = evaluate(y, oof_p)
    print(f"[prevention] 3 graines : F1 {metrics_p['f1']:.4f} AUC {metrics_p['auc']:.4f} "
          f"seuil {metrics_p['threshold']:.3f} (écart complet - prévention : F1 {metrics['f1'] - metrics_p['f1']:+.4f}, "
          f"AUC {metrics['auc'] - metrics_p['auc']:+.4f})")
    top_p = imp_p.sort_values(ascending=False).head(15)
    print("Top 15 variables prévention (gain moyen) :")
    print(top_p.round(1).to_string())
    (ROOT / "results").mkdir(exist_ok=True)
    save_proba(f"{args.name}_prevention", stamp, test, test_p, oof_p)

    save_outputs(args.name, test, test_proba, metrics,
                 {"model": "xgboost_gpu_monotone_seed_bagging", "params": {**params, "monotone_constraints": MONOTONE_INC},
                  "seeds": SEEDS, "num_boost_round": args.rounds, "early_stopping_rounds": EARLY_STOPPING,
                  "n_features": len(full_cols), "columns_by_tier": tiers_kept, "columns_dropped_by_tier": tiers_dropped,
                  "derived_columns": DERIVED_TIER, "monotone_increasing": [c for c in full_cols if c in MONOTONE_INC],
                  "best_iterations": best_iters, "cv_seed42_only": m_seed42,
                  "gain_share_by_tier": gain_share, "top_features": top.round(1).to_dict(),
                  "prevention_model": {"tiers": PREVENTION_TIERS, "n_features": len(prev_cols), "columns": prev_cols,
                                       "cv": metrics_p, "cv_seed42_only": mp_seed42, "best_iterations": iters_p,
                                       "top_features": top_p.round(1).to_dict(),
                                       "oof_csv": f"results/{args.name}_prevention_{stamp}_oof.csv",
                                       "test_csv": f"results/{args.name}_prevention_{stamp}_test.csv",
                                       "delta_full_minus_prevention": {"f1": metrics["f1"] - metrics_p["f1"],
                                                                       "auc": metrics["auc"] - metrics_p["auc"]}}},
                 t0, oof=oof, stamp=stamp)


if __name__ == "__main__":
    main()
