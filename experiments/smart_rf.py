"""Modèle « intelligent » Random Forest : colonnes défendables (niveaux A-D de features/tiers.json) + dérivées.

Usage (Modal, CPU 16 cœurs) :
    python -W ignore -m modal run modal_run.py --script experiments/smart_rf.py --name smart_rf --threads 16

Protocole identique à train.py (train+extra, StratifiedKFold 5 seed 42, OOF, seuil F1 sur l'OOF).
Colonnes : union des niveaux A_amont, B_comorbidites, C_parcours_consequences, D_contexte (copiés de
features/tiers.json, le conteneur Modal n'embarque pas features/), intersectée avec select_columns() et
débarrassée des colonnes vides à > 95 % (critère sans TARGET) ; codes nettoyés par clean() de r1_clean_codes.
Dérivées construites une fois sur train et test (sans TARGET) : SMOKING_STATUS, BMI_CALC, DRINKS_WEEK,
DIABETES_BIN, DIAB_YEARS (A), N_COMORB (B), N_LIMITS, CARE_INTENSITY (C). Colonnes brutes conservées.
NaN -> -1 pour la forêt (l'absence d'une question de module ou conditionnelle a un sens).
Deux évaluations : (a) modèle COMPLET A+B+C+D (soumis) ; (b) modèle PRÉVENTION A+B seulement, mêmes
hyperparamètres, métriques OOF sous "prevention_model" et CSV <name>_prevention_<stamp>_{oof,test}.csv.
Importances : permutation (AUC) sur 20 000 lignes de validation du fold 0, 5 répétitions, part par niveau.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_clean_codes import ROOT, SEED, clean, evaluate, load, select_columns  # noqa: E402

IDEA = ("Random Forest « intelligent » : uniquement les colonnes défendables des niveaux A (amont), "
        "B (comorbidités), C (parcours/conséquences), D (contexte) de features/tiers.json, codes nettoyés, "
        "colonnes vides à > 95 % retirées, + 8 dérivées (SMOKING_STATUS, BMI_CALC, DRINKS_WEEK, DIABETES_BIN, "
        "DIAB_YEARS, N_COMORB, N_LIMITS, CARE_INTENSITY), NaN -> -1 ; RandomForestClassifier(500 arbres, "
        "max_features='sqrt', min_samples_leaf=5, random_state=42+fold) ; second modèle PRÉVENTION A+B seulement "
        "avec les mêmes hyperparamètres ; importances par permutation (AUC) agrégées par niveau")
NAN_MAX = 0.95
CONFIG = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "fill_value": -1.0,
    "perm_rows": 20_000,
    "perm_repeats": 5,
}

# ---------------------------------------------------------------- niveaux (copie de features/tiers.json)
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
DERIVED_TIER = {
    "SMOKING_STATUS": "A_amont", "BMI_CALC": "A_amont", "DRINKS_WEEK": "A_amont", "DIABETES_BIN": "A_amont",
    "DIAB_YEARS": "A_amont", "N_COMORB": "B_comorbidites", "N_LIMITS": "C_parcours_consequences",
    "CARE_INTENSITY": "C_parcours_consequences",
}
PREVENTION_TIERS = ["A_amont", "B_comorbidites"]
LIMIT_COLS = ["DIFFWALK", "DIFFDRES", "DIFFALON", "DECIDE"]


# ---------------------------------------------------------------- colonnes et dérivées (sans TARGET)
def tier_columns(train, feats_all):
    """Colonnes A-D disponibles après select_columns() et retrait des vides à > 95 % (critère sans TARGET)."""
    available = set(feats_all)
    nan_rate = train[feats_all].isna().mean()
    kept, dropped = {}, {}
    for tier, cols in TIERS.items():
        kept[tier] = [c for c in cols if c in available and nan_rate[c] <= NAN_MAX]
        dropped[tier] = [c for c in cols if c not in kept[tier]]
    return kept, dropped


def add_derived(df, kept):
    """Dérivées construites à partir des colonnes nettoyées. Modifie df en place, retourne df."""
    f32 = "float32"
    s100, sday = df["SMOKE100"], df["SMOKDAY2"]
    status = pd.Series(np.nan, index=df.index, dtype=f32)
    status = status.mask(s100 == 2, 0).mask((s100 == 1) & (sday == 3), 1).mask(sday == 2, 2).mask(sday == 1, 3)
    df["SMOKING_STATUS"] = status.astype(f32)

    kg = df["WEIGHT2"] * np.float32(0.45359237)
    m = df["HEIGHT3"] * np.float32(0.0254)
    bmi = kg / (m * m)
    df["BMI_CALC"] = bmi.where((bmi >= 10) & (bmi <= 100), np.nan).astype(f32)

    days = df["ALCDAY4"]  # jours/mois après clean()
    drinks = (days / 30.0 * 7.0) * df["AVEDRNK3"]
    drinks = drinks.mask(days == 0, 0.0)
    df["DRINKS_WEEK"] = drinks.astype(f32)

    df["DIABETES_BIN"] = (df["DIABETE4"] == 1).astype(f32)
    diab_years = (df["_AGE80"] - df["DIABAGE4"]).clip(lower=0)
    df["DIAB_YEARS"] = diab_years.where(df["DIABETE4"] == 1, np.nan).astype(f32)

    b_cols = [c for c in kept["B_comorbidites"]]
    df["N_COMORB"] = (df[b_cols] == 1).sum(axis=1).astype(f32)
    lim = [c for c in LIMIT_COLS if c in kept["C_parcours_consequences"]]
    df["N_LIMITS"] = (df[lim] == 1).sum(axis=1).astype(f32)
    care = (df["PERSDOC3"].isin([1, 2]).astype(int) + (df["CHECKUP1"] == 1).astype(int)
            + (df["FLUSHOT7"] == 1).astype(int) + (df["PNEUVAC4"] == 1).astype(int))
    df["CARE_INTENSITY"] = care.astype(f32)
    return df


def columns_by_tier(kept):
    """Colonnes finales (brutes + dérivées) par niveau."""
    out = {t: list(cols) for t, cols in kept.items()}
    for d, t in DERIVED_TIER.items():
        out[t].append(d)
    return out


def to_matrix(df, cols, fill):
    return df[cols].to_numpy(dtype=np.float32, na_value=fill)


# ---------------------------------------------------------------------------------------- modèle
def build_model(cfg, threads, fold):
    return RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_features=cfg["max_features"],
        min_samples_leaf=cfg["min_samples_leaf"],
        n_jobs=threads,
        random_state=SEED + fold,
    )


def run_cv(train, test, y, cols, cfg, threads, folds, label, perm=None):
    """CV 5 folds ; retourne oof, test_proba, infos par fold et (fold 0) importances par permutation."""
    X_all = to_matrix(train, cols, cfg["fill_value"])
    X_test = to_matrix(test, cols, cfg["fill_value"])
    print(f"[{label}] X {X_all.shape} float32 ({X_all.nbytes / 1e9:.2f} Go), {len(cols)} colonnes", flush=True)
    oof = np.zeros(len(train))
    test_proba = np.zeros(len(test))
    folds_info, perm_result, impurity = [], None, pd.Series(0.0, index=cols)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y)):
        tf = time.time()
        model = build_model(cfg, threads, fold)
        model.fit(X_all[tr_idx], y[tr_idx])
        fit_s = time.time() - tf
        oof[va_idx] = model.predict_proba(X_all[va_idx])[:, 1]
        test_proba += model.predict_proba(X_test)[:, 1] / folds
        auc = roc_auc_score(y[va_idx], oof[va_idx])
        impurity += pd.Series(model.feature_importances_, index=cols) / folds
        n_nodes = int(sum(t.tree_.node_count for t in model.estimators_))
        print(f"[{label}] fold {fold}: {cfg['n_estimators']} arbres ({n_nodes / cfg['n_estimators']:.0f} nœuds/arbre), "
              f"AUC {auc:.4f}, fit {fit_s:.0f} s, total {time.time() - tf:.0f} s", flush=True)
        folds_info.append({"auc": round(float(auc), 5), "fit_s": round(fit_s, 1),
                           "nodes_per_tree": round(n_nodes / cfg["n_estimators"]),
                           "fold_s": round(time.time() - tf, 1)})
        if fold == 0 and perm is not None:
            tp = time.time()
            rng = np.random.RandomState(SEED)
            sub = rng.choice(va_idx, size=min(perm["rows"], len(va_idx)), replace=False)
            pi = permutation_importance(model, X_all[sub], y[sub], scoring="roc_auc",
                                        n_repeats=perm["repeats"], random_state=SEED, n_jobs=1)
            perm_result = pd.DataFrame({"mean": pi.importances_mean, "std": pi.importances_std}, index=cols)
            print(f"[{label}] permutation : {len(sub)} lignes x {perm['repeats']} répétitions, {time.time() - tp:.0f} s",
                  flush=True)
        del model
    return oof, test_proba, folds_info, impurity, perm_result


def tier_shares(perm_result, by_tier):
    """Part de l'importance totale (permutation, valeurs négatives ramenées à 0) par niveau."""
    pos = perm_result["mean"].clip(lower=0)
    total = float(pos.sum()) or 1.0
    shares = {}
    for tier, cols in by_tier.items():
        shares[tier] = {"share": round(float(pos[cols].sum()) / total, 4), "n_columns": len(cols),
                        "top5": {c: round(float(perm_result.loc[c, "mean"]), 5)
                                 for c in perm_result.loc[cols, "mean"].sort_values(ascending=False).head(5).index}}
    return shares


def write_proba(name, stamp, test, oof, test_proba):
    pd.DataFrame({"oof": oof}).to_csv(ROOT / "results" / f"{name}_{stamp}_oof.csv", index=False, float_format="%.6f")
    pd.DataFrame({"ID": test["ID"], "proba": test_proba}).to_csv(
        ROOT / "results" / f"{name}_{stamp}_test.csv", index=False, float_format="%.6f")


def save_outputs(name, stamp, test, oof, test_proba, metrics, info, t0):
    """Même format que save_outputs de dl_base (soumission + JSON + OOF/test CSV)."""
    (ROOT / "submissions").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    sub_path = ROOT / "submissions" / f"submission_{name}_{stamp}.csv"
    pred = (test_proba >= metrics["threshold"]).astype(int)
    pd.DataFrame({"ID": test["ID"], "TARGET": pred}).to_csv(sub_path, index=False)
    write_proba(name, stamp, test, oof, test_proba)
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
    parser.add_argument("--name", default="smart_rf")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=None, help="remplace CONFIG['n_estimators']")
    parser.add_argument("--nrows", type=int, default=None, help="test local : sous-échantillon de lignes")
    args = parser.parse_args()
    cfg = dict(CONFIG)
    if args.n_estimators:
        cfg["n_estimators"] = args.n_estimators
    t0 = time.time()

    train, test = load()
    if args.nrows:  # test d'API uniquement, jamais pour un résultat
        train = train.sample(args.nrows, random_state=SEED).reset_index(drop=True)
        test = test.head(args.nrows).reset_index(drop=True)
    feats_all = select_columns(train)
    maxes = train[feats_all].max()
    clean(train, feats_all, maxes)
    clean(test, feats_all, maxes)
    kept, dropped = tier_columns(train, feats_all)
    for df in (train, test):
        add_derived(df, kept)
    by_tier = columns_by_tier(kept)
    full_cols = [c for t in by_tier for c in by_tier[t]]
    prev_cols = [c for t in PREVENTION_TIERS for c in by_tier[t]]
    y = train["TARGET"].values.astype(int)
    for t in by_tier:
        print(f"{t}: {len(by_tier[t])} colonnes (retirées : {dropped[t]})")
    print(f"modèle complet {len(full_cols)} colonnes, prévention {len(prev_cols)} colonnes")
    derived = list(DERIVED_TIER)
    print("dérivées (train) :")
    print(train[derived].describe().T[["count", "mean", "min", "max"]].round(2).to_string())

    # (a) modèle complet A+B+C+D
    oof, test_proba, folds_info, impurity, perm_result = run_cv(
        train, test, y, full_cols, cfg, args.threads, args.folds, "complet",
        perm={"rows": cfg["perm_rows"], "repeats": cfg["perm_repeats"]})
    metrics = evaluate(y, oof)
    shares = tier_shares(perm_result, by_tier)
    print("Part de l'importance (permutation AUC) par niveau :",
          {t: s["share"] for t, s in shares.items()})
    top_perm = perm_result["mean"].sort_values(ascending=False).head(20)
    print("Top 20 variables (permutation AUC, fold 0) :")
    print(top_perm.round(5).to_string())

    # (b) modèle prévention A+B, mêmes hyperparamètres
    oof_p, test_p, folds_p, impurity_p, _ = run_cv(
        train, test, y, prev_cols, cfg, args.threads, args.folds, "prévention")
    metrics_p = evaluate(y, oof_p)
    print("CV prévention :", {k: round(v, 4) for k, v in metrics_p.items()})

    stamp = time.strftime("%Y%m%d_%H%M%S")
    (ROOT / "results").mkdir(exist_ok=True)
    write_proba(f"{args.name}_prevention", stamp, test, oof_p, test_p)
    info = {
        "model": "random_forest", "config": cfg, "nan_max": NAN_MAX,
        "n_columns": len(full_cols), "columns_by_tier": by_tier, "derived_columns": DERIVED_TIER,
        "dropped_by_tier": dropped,
        "importance_shares_by_tier": shares,
        "top_features_permutation": {k: round(float(v), 6) for k, v in top_perm.items()},
        "permutation_importance": {c: {"mean": round(float(r["mean"]), 6), "std": round(float(r["std"]), 6)}
                                   for c, r in perm_result.iterrows()},
        "top_features_impurity": {k: round(float(v), 6) for k, v in
                                  impurity.sort_values(ascending=False).head(20).items()},
        "folds": folds_info,
        "prevention_model": {
            "tiers": PREVENTION_TIERS, "n_columns": len(prev_cols), "columns": prev_cols,
            "cv": metrics_p, "folds": folds_p,
            "oof_csv": f"results/{args.name}_prevention_{stamp}_oof.csv",
            "test_csv": f"results/{args.name}_prevention_{stamp}_test.csv",
            "top_features_impurity": {k: round(float(v), 6) for k, v in
                                      impurity_p.sort_values(ascending=False).head(15).items()},
            "delta_vs_full": {"f1": round(metrics["f1"] - metrics_p["f1"], 5),
                              "auc": round(metrics["auc"] - metrics_p["auc"], 5)},
        },
    }
    save_outputs(args.name, stamp, test, oof, test_proba, metrics, info, t0)


if __name__ == "__main__":
    main()
