"""Modèle « intelligent » : LightGBM baggé sur les seules colonnes défendables (niveaux A-D de
features/tiers.json), variables dérivées lisibles, contraintes de monotonie sur les facteurs d'amont,
et un second modèle PRÉVENTION (A+B seulement) pour chiffrer ce que valent les facteurs d'amont seuls.

Usage : python experiments/smart_lgbm.py [--name NOM] [--threads N] [--folds 5] [--rows N]
Sorties : submissions/submission_<nom>_<horodatage>.csv, results/<nom>_<horodatage>.json,
          results/<nom>_<horodatage>_oof.csv / _test.csv (modèle complet)
          results/<nom>_prevention_<horodatage>_oof.csv / _test.csv (modèle A+B)

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
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Base : x_lgbbag_proba (LightGBM 5 graines par fold, protocole inchangé) ; nettoyage des codes
# de r1_clean_codes (importé, pas recopié).
from x_lgbbag_proba import LGB_PARAMS, ROOT, SEED, SEEDS, clean, evaluate, load, select_columns

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)  # ajout de colonnes dérivées

IDEA = ("[copie de x_lgbbag_proba : LightGBM 5 graines par fold, early stopping par modèle, OOF/test "
        "moyennés] Modèle « intelligent » : colonnes restreintes aux niveaux A_amont, B_comorbidites, "
        "C_parcours_consequences, D_contexte de features/tiers.json (colonnes intrusives, doublons "
        "calculés et artefacts exclus ; vides à plus de 95 % retirées) ; 8 variables dérivées lisibles "
        "(SMOKING_STATUS, BMI_CALC, DRINKS_WEEK, DIABETES_BIN, DIAB_YEARS, N_COMORB, N_LIMITS, "
        "CARE_INTENSITY) ; contraintes de monotonie croissante sur les facteurs d'amont continus "
        "(monotone_constraints_method='advanced') ; même modèle réentraîné sans contraintes et en "
        "version PRÉVENTION (A+B seulement) pour chiffrer contraintes et facteurs d'amont")

# --- Listes de colonnes (copiées de features/tiers.json : le conteneur Modal n'a pas features/) -----
TIERS = {
    "A_amont": ["_AGE80", "SEXVAR", "SMOKE100", "SMOKDAY2", "_SMOKER3", "_PACKYRS", "_YRSSMOK", "_PACKDAY",
                "USENOW3", "ECIGNOW2", "_CURECI2", "WEIGHT2", "HEIGHT3", "_BMI5", "DIABETE4", "DIABAGE4",
                "PREDIAB2", "EXERANY2", "ALCDAY4", "AVEDRNK3", "DRNK3GE5", "MAXDRNKS", "_DRNKWK2", "SLEPTIM1",
                "MENTHLTH", "ADDEPEV3", "SDHSTRE1", "PREGNANT"],
    "B_comorbidites": ["CVDSTRK3", "CHCKDNY2", "CHCCOPD3", "HAVARTH4", "ASTHMA3", "ASTHNOW", "CHCSCNC1",
                       "CHCOCNC1", "DEAF", "BLIND"],
    "C_parcours_consequences": ["GENHLTH", "PHYSHLTH", "POORHLTH", "DIFFWALK", "DIFFDRES", "DIFFALON", "DECIDE",
                                "EMPLOY1", "PERSDOC3", "CHECKUP1", "MEDCOST1", "PRIMINSR", "FLUSHOT7", "FLSHTMY3",
                                "PNEUVAC4", "TETANUS1", "COVIDVA1", "_YRSQUIT", "LASTSMK2", "STOPSMK2", "LCSCTSC1",
                                "LCSSCNCR", "LCSFIRST", "LCSLAST", "LCSNUMCG", "LASTDEN4", "RMVTETH4", "_CRCREC2",
                                "_MAM5023", "HOWLONG"],
    "D_contexte": ["INCOME3", "EDUCA", "MARITAL", "_RACE1", "_HISPANC", "_STATE", "_METSTAT", "_URBSTAT",
                   "HHADULT", "CHILDREN", "VETERAN3", "RENTHOM1", "SDHFOOD1", "SDHBILLS", "SDHISOLT", "EMTSUPRT",
                   "LSATISFY"],
}
# Niveau des variables dérivées (construites une fois sur train et test, sans TARGET)
DERIVED_TIER = {
    "SMOKING_STATUS": "A_amont", "BMI_CALC": "A_amont", "DRINKS_WEEK": "A_amont", "DIABETES_BIN": "A_amont",
    "DIAB_YEARS": "A_amont", "N_COMORB": "B_comorbidites", "N_LIMITS": "C_parcours_consequences",
    "CARE_INTENSITY": "C_parcours_consequences",
}
PREVENTION_TIERS = ["A_amont", "B_comorbidites"]
# Facteurs d'amont dont le risque doit croître avec la valeur (contrainte +1) ; 0 ailleurs
MONOTONE_UP = ["_AGE80", "_PACKYRS", "_YRSSMOK", "_PACKDAY", "_BMI5", "BMI_CALC", "WEIGHT2", "MENTHLTH",
               "DRINKS_WEEK", "N_COMORB", "N_LIMITS", "CARE_INTENSITY", "DIAB_YEARS"]
EMPTY_MAX = 0.95  # colonnes vides à plus de 95 % (sur train, sans TARGET) retirées
LIMIT_COLS = ["DIFFWALK", "DIFFDRES", "DIFFALON", "DECIDE"]


def build_columns(train, feats):
    """Union A ∪ B ∪ C ∪ D ∩ colonnes disponibles après select_columns(), moins les vides à > 95 %."""
    avail = set(feats)
    empty = {c for c in avail if train[c].isna().mean() > EMPTY_MAX}
    kept, dropped = {}, {}
    for tier, cols in TIERS.items():
        kept[tier] = [c for c in cols if c in avail and c not in empty]
        dropped[tier] = [c for c in cols if c not in avail or c in empty]
    return kept, dropped


def add_derived(df):
    """Variables dérivées lisibles, calculées après clean() (codes 7/9 -> NaN, 88 -> 0, unités réelles)."""
    f32 = "float32"
    # Statut tabagique : 0 jamais (SMOKE100 = 2), 1 ancien (fume plus du tout), 2 occasionnel, 3 quotidien
    s100, sday = df["SMOKE100"], df["SMOKDAY2"]
    st = pd.Series(np.nan, index=df.index, dtype=f32)
    st = st.mask(s100 == 2, 0).mask((s100 == 1) & (sday == 3), 1).mask(sday == 2, 2).mask(sday == 1, 3)
    df["SMOKING_STATUS"] = st.astype(f32)
    # IMC recalculé : WEIGHT2 en livres, HEIGHT3 en pouces (après clean) ; hors [10, 100] -> NaN
    kg, m = df["WEIGHT2"] * 0.453592, df["HEIGHT3"] * 0.0254
    bmi = kg / (m * m)
    df["BMI_CALC"] = bmi.mask((bmi < 10) | (bmi > 100), np.nan).astype(f32)
    # Verres par semaine : ALCDAY4 en jours/mois après clean (0 = ne boit pas) × verres par jour de consommation
    days = df["ALCDAY4"]
    dw = (days / 30 * 7) * df["AVEDRNK3"]
    df["DRINKS_WEEK"] = dw.mask(days == 0, 0).astype(f32)
    # Diabète (hors gestationnel / prédiabète) et ancienneté du diabète (NaN si pas diabétique)
    diab = (df["DIABETE4"] == 1)
    df["DIABETES_BIN"] = diab.astype(f32)
    dy = (df["_AGE80"] - df["DIABAGE4"]).clip(lower=0)
    df["DIAB_YEARS"] = dy.where(diab & df["DIABAGE4"].notna(), np.nan).astype(f32)
    # Comptages lisibles
    df["N_COMORB"] = sum((df[c] == 1).astype(f32) for c in TIERS["B_comorbidites"]).astype(f32)
    df["N_LIMITS"] = sum((df[c] == 1).astype(f32) for c in LIMIT_COLS).astype(f32)
    care = (df["PERSDOC3"].isin([1, 2]).astype(f32) + (df["CHECKUP1"] == 1).astype(f32)
            + (df["FLUSHOT7"] == 1).astype(f32) + (df["PNEUVAC4"] == 1).astype(f32))
    df["CARE_INTENSITY"] = care.astype(f32)
    return df


def monotone_check(train, cols, y):
    """Diagnostic (ne touche pas au modèle) : corrélation de Spearman avec TARGET et taux de positifs
    par quartile, pour vérifier que les colonnes contraintes croissent bien avec le risque."""
    out = {}
    for c in cols:
        s = train[c]
        ok = s.notna()
        rho = float(pd.Series(y[ok.values]).corr(s[ok].reset_index(drop=True), method="spearman"))
        try:
            q = pd.qcut(s[ok], 4, labels=False, duplicates="drop")
            rates = pd.Series(y[ok.values]).groupby(q.values).mean().round(4).tolist()
        except ValueError:
            rates = []
        out[c] = {"spearman": round(rho, 4), "pos_rate_by_quartile": rates, "nan_rate": round(float(1 - ok.mean()), 3)}
        flag = "" if rho > 0 else "  <-- NON croissant"
        print(f"  {c:15s} rho={rho:+.4f} quartiles={rates} NaN={1 - ok.mean():.2f}{flag}")
    return out


def run_cv(X, y, X_test, params, seeds, folds, label):
    """LightGBM baggé : `seeds` graines par fold, early stopping par modèle ; renvoie OOF, test, infos."""
    n_seeds = len(seeds)
    oof_seeds = np.zeros((n_seeds, len(X)))
    test_proba = np.zeros(len(X_test))
    importance = pd.Series(0.0, index=X.columns)
    best_iters = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        fold_iters = []
        for k, seed in enumerate(seeds):
            tf = time.time()
            model = lgb.train(
                {**params, "seed": seed},
                lgb.Dataset(X.iloc[tr_idx], y[tr_idx]),
                num_boost_round=5000,
                valid_sets=[lgb.Dataset(X.iloc[va_idx], y[va_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            oof_seeds[k, va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
            test_proba += model.predict(X_test, num_iteration=model.best_iteration) / (n_seeds * folds)
            importance += model.feature_importance("gain") / (n_seeds * folds)
            fold_iters.append(int(model.best_iteration))
            print(f"[{label}] fold {fold} seed {seed}: {model.best_iteration} arbres, "
                  f"AUC {roc_auc_score(y[va_idx], oof_seeds[k, va_idx]):.4f} ({time.time() - tf:.0f} s)", flush=True)
        best_iters.append(fold_iters)
    oof = oof_seeds.mean(axis=0)
    metrics = evaluate(y, oof)
    print(f"[{label}] CV :", {k: round(v, 4) for k, v in metrics.items()}, flush=True)
    top = importance.sort_values(ascending=False)
    info = {"best_iterations": best_iters, "cv_seed42_only": evaluate(y, oof_seeds[0]),
            "top_features": top.head(25).round(0).to_dict(),
            "importance_share_by_tier": None}
    return oof, test_proba, metrics, info, top


def save_outputs(name, test, test_proba, metrics, info, t0, oof, prevention=None):
    """Copie de x_lgbbag_proba.save_outputs, étendue : écrit aussi les OOF/test du modèle prévention
    sous <name>_prevention avec le même horodatage."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (ROOT / "submissions").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    sub_path = ROOT / "submissions" / f"submission_{name}_{stamp}.csv"
    pred = (test_proba >= metrics["threshold"]).astype(int)
    pd.DataFrame({"ID": test["ID"], "TARGET": pred}).to_csv(sub_path, index=False)
    pd.DataFrame({"oof": oof}).to_csv(ROOT / "results" / f"{name}_{stamp}_oof.csv", index=False, float_format="%.6f")
    pd.DataFrame({"ID": test["ID"], "proba": test_proba}).to_csv(
        ROOT / "results" / f"{name}_{stamp}_test.csv", index=False, float_format="%.6f")
    if prevention is not None:
        p_oof, p_test = prevention
        pname = f"{name}_prevention"
        pd.DataFrame({"oof": p_oof}).to_csv(ROOT / "results" / f"{pname}_{stamp}_oof.csv", index=False, float_format="%.6f")
        pd.DataFrame({"ID": test["ID"], "proba": p_test}).to_csv(
            ROOT / "results" / f"{pname}_{stamp}_test.csv", index=False, float_format="%.6f")
        info["prevention_model"]["oof_csv"] = f"results/{pname}_{stamp}_oof.csv"
        info["prevention_model"]["test_csv"] = f"results/{pname}_{stamp}_test.csv"
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


def tier_shares(top, tier_of):
    share = top / top.sum() if top.sum() > 0 else top
    out = {}
    for c, v in share.items():
        out[tier_of[c]] = out.get(tier_of[c], 0.0) + float(v)
    return {k: round(v, 4) for k, v in sorted(out.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="smart_lgbm")
    parser.add_argument("--threads", type=int, default=0, help="0 = tous les cœurs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rows", type=int, default=0, help="test local : sous-échantillon de train (0 = tout)")
    parser.add_argument("--dry", action="store_true", help="test local : construit les colonnes sans entraîner")
    args = parser.parse_args()

    t0 = time.time()
    train, test = load()
    if args.rows:
        train = train.sample(n=args.rows, random_state=SEED).reset_index(drop=True)
        test = test.head(args.rows).reset_index(drop=True)
    feats = select_columns(train)
    kept, dropped = build_columns(train, feats)
    raw_cols = [c for tier in kept for c in kept[tier]]
    print("Colonnes retenues par niveau :", {k: len(v) for k, v in kept.items()})
    print("Colonnes de niveaux A-D écartées (absentes ou vides > 95 %) :", {k: v for k, v in dropped.items() if v})

    # Nettoyage des codes (r1_clean_codes), transformation sans TARGET faite une fois sur train et test
    maxes = train[feats].max()
    clean(train, feats, maxes)
    clean(test, feats, maxes)
    for df in (train, test):  # _DRNKWK2 : 99900 = ne sait pas / refus (laissé intact par clean)
        df["_DRNKWK2"] = df["_DRNKWK2"].mask(df["_DRNKWK2"] >= 99900, np.nan).astype("float32")
        add_derived(df)

    tier_of = {c: tier for tier in kept for c in kept[tier]}
    tier_of.update(DERIVED_TIER)
    full_cols = raw_cols + list(DERIVED_TIER)
    prev_cols = [c for c in full_cols if tier_of[c] in PREVENTION_TIERS]
    columns_by_tier = {tier: [c for c in full_cols if tier_of[c] == tier] for tier in TIERS}
    print(f"Modèle complet : {len(full_cols)} colonnes ; modèle prévention (A+B) : {len(prev_cols)} colonnes")
    for tier, cols in columns_by_tier.items():
        print(f"  {tier} ({len(cols)}) : {cols}")

    y = train["TARGET"].values.astype(int)
    print("Dérivées (train) :")
    for c in DERIVED_TIER:
        s = train[c]
        print(f"  {c}: NaN={s.isna().mean():.3f} min={s.min():.2f} max={s.max():.2f} mean={s.mean():.3f} "
              f"n_distinct={s.nunique()}")
    print("Vérification de la monotonie (Spearman avec TARGET, taux de positifs par quartile) :")
    mono = monotone_check(train, MONOTONE_UP, y)
    if args.dry:
        print("Mode --dry : arrêt avant entraînement.")
        return

    X, X_test = train[full_cols], test[full_cols]
    print(f"train {X.shape}, test {X_test.shape}, taux de positifs {y.mean():.3f}", flush=True)
    base_params = {**LGB_PARAMS, "num_threads": args.threads}

    def constrained(cols):
        return {**base_params, "monotone_constraints": [1 if c in MONOTONE_UP else 0 for c in cols],
                "monotone_constraints_method": "advanced"}

    # (a) modèle COMPLET A+B+C+D, contraint, 5 graines : celui soumis
    oof, test_proba, metrics, info, top = run_cv(X, y, X_test, constrained(full_cols), SEEDS, args.folds, "complet")
    info["importance_share_by_tier"] = tier_shares(top, tier_of)
    print("Top 20 variables (gain) :")
    print(top.head(20).round(0).to_string())
    print("Part du gain par niveau :", info["importance_share_by_tier"], flush=True)

    # Même modèle SANS contraintes (5 graines, même protocole) pour chiffrer l'effet des contraintes
    oof_u, _, metrics_u, info_u, top_u = run_cv(X, y, X_test, base_params, SEEDS, args.folds, "sans contraintes")

    # (b) modèle PRÉVENTION A+B seulement, mêmes hyperparamètres (contraint, 5 graines)
    Xp, Xp_test = train[prev_cols], test[prev_cols]
    oof_p, test_p, metrics_p, info_p, top_p = run_cv(Xp, y, Xp_test, constrained(prev_cols), SEEDS, args.folds, "prévention A+B")
    info_p["importance_share_by_tier"] = tier_shares(top_p, tier_of)

    print(f"\nComplet contraint : F1 {metrics['f1']:.4f} AUC {metrics['auc']:.4f} | sans contraintes : "
          f"F1 {metrics_u['f1']:.4f} AUC {metrics_u['auc']:.4f} (effet contraintes F1 {metrics['f1'] - metrics_u['f1']:+.4f}, "
          f"AUC {metrics['auc'] - metrics_u['auc']:+.4f})")
    print(f"Prévention A+B : F1 {metrics_p['f1']:.4f} AUC {metrics_p['auc']:.4f} "
          f"(écart vs complet F1 {metrics_p['f1'] - metrics['f1']:+.4f}, AUC {metrics_p['auc'] - metrics['auc']:+.4f})")

    result_info = {
        "model": "lightgbm_seed_bagging_monotone_tiers",
        "params": constrained(full_cols), "seeds": SEEDS, "rows": args.rows,
        "n_features": len(full_cols), "columns_by_tier": columns_by_tier,
        "derived_columns": DERIVED_TIER, "dropped_tier_columns": dropped,
        "monotone_increasing": MONOTONE_UP, "monotone_check": mono,
        "full_model": {"constrained": True, "seeds": SEEDS, **info},
        "unconstrained_model": {"cv": metrics_u, "seeds": SEEDS, "cv_seed42_only": info_u["cv_seed42_only"],
                                "best_iterations": info_u["best_iterations"],
                                "top_features": top_u.head(25).round(0).to_dict(),
                                "effect_of_constraints": {"f1": round(metrics["f1"] - metrics_u["f1"], 5),
                                                          "auc": round(metrics["auc"] - metrics_u["auc"], 5)}},
        "prevention_model": {"tiers": PREVENTION_TIERS, "n_features": len(prev_cols), "columns": prev_cols,
                             "cv": metrics_p, "seeds": SEEDS, **info_p,
                             "gap_vs_full": {"f1": round(metrics_p["f1"] - metrics["f1"], 5),
                                             "auc": round(metrics_p["auc"] - metrics["auc"], 5)}},
    }
    save_outputs(args.name, test, test_proba, metrics, result_info, t0, oof, prevention=(oof_p, test_p))


if __name__ == "__main__":
    main()
