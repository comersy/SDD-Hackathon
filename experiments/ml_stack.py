"""Stacking et blends pondérés de tous les modèles de base disponibles, à partir de leurs
probabilités OOF / test sauvegardées (results/*_oof.csv, *_test.csv). Tout se calcule en local,
sans réentraîner les modèles de base.

Usage : python experiments/ml_stack.py [--name ml_stack]
Sorties : results/<nom>_<horodatage>.json, results/<nom>_<horodatage>_oof.csv, _test.csv,
          submissions/submission_<nom>_<horodatage>.csv

Protocole d'évaluation, à conserver à l'identique dans toute expérience pour que les scores
restent comparables :
- entraînement sur train.csv + extra.csv concaténés (370 132 lignes), test.csv sert
  uniquement à prédire ;
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42), prédictions out-of-fold ;
- tout prétraitement qui utilise TARGET est ajusté dans chaque fold sur la partie
  entraînement seulement ;
- le seuil de décision est choisi sur les prédictions OOF pour maximiser le F1 ;
  le F1 reporté est ce F1 OOF, au seuil retenu.

Niveau 2 : le méta-modèle de stacking est lui-même évalué par cross_val_predict (mêmes folds) sur
la matrice OOF, de sorte que chaque ligne est prédite par un méta-modèle qui ne l'a pas vue. Les
blends à poids fixes choisissent leurs poids sur l'OOF complet (légèrement optimiste, grille grossière).
"""
import argparse
import glob
import itertools
import json
import os
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
BASE = "x_lgbbag_proba"  # référence : LightGBM baggé (F1 0.4247)
EXCLUDE_PREFIXES = ("dl_blend", "ml_stack")  # combinaisons, pas des modèles de base
N_TRAIN = 370_132
N_TEST = 75_000
SEED = 42
TOL = 1e-6
SIMPLICITY_MARGIN = 0.0003  # en dessous de cet écart de F1, on préfère la solution la plus simple

IDEA = ("Combinaison de tous les modèles de base disponibles (results/*_oof.csv) sans réentraînement : "
        "(a) stacking honnête, méta-modèle LogisticRegression(C=1) sur les logits des composants, évalué par "
        "cross_val_predict StratifiedKFold(5, shuffle, 42) sur la matrice OOF, seuil sur les prédictions OOF de "
        "niveau 2, méta-modèle refit sur tout l'OOF pour le test ; variante LightGBM régularisé (num_leaves 4, "
        "200 arbres, lr 0.03) ; (b) blends pondérés LightGBM baggé + un autre composant (w 0.1/0.2/0.3), "
        "LightGBM + les 2-3 meilleurs composants (grille grossière, poids LightGBM >= 0.5), moyenne de rangs ; "
        "choix final par F1 OOF, solution la plus simple si l'écart est < 0.0003")


# ---------------------------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------------------------
GRID = np.arange(0.05, 0.6, 0.005)  # même grille que les scripts de base


def f1_on_grid(y, proba, grid=GRID):
    """F1 pour chaque seuil de la grille (pred = proba >= t), calcul vectorisé exact."""
    order = np.argsort(-proba, kind="stable")
    p_sorted = proba[order]
    tp_cum = np.cumsum(y[order])
    n_pos = int(y.sum())
    # nombre de prédictions positives pour chaque seuil : #(proba >= t)
    n_pred = np.searchsorted(-p_sorted, -grid, side="right")
    tp = np.where(n_pred > 0, tp_cum[np.maximum(n_pred - 1, 0)], 0)
    denom = n_pred + n_pos
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


RANK_GRID = np.arange(0.5, 0.995, 0.005)  # scores de rangs dans [0, 1] : le seuil utile est un quantile élevé


def best_threshold(y, proba, grid=GRID):
    scores = f1_on_grid(y, proba, grid)
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def evaluate(y, oof, grid=GRID):
    threshold, f1 = best_threshold(y, oof, grid)
    pred = oof >= threshold
    return {
        "f1": f1,
        "threshold": threshold,
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "auc": float(roc_auc_score(y, oof)),
    }


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------------------------
# Chargement des composants
# ---------------------------------------------------------------------------------------------
def short_name(prefix):
    return re.sub(r"_\d{8}_\d{6}$", "", prefix)


def load_components(y, test_ids):
    comps, excluded = {}, []
    for oof_path in sorted(glob.glob(str(ROOT / "results" / "*_oof.csv"))):
        base = os.path.basename(oof_path)[: -len("_oof.csv")]
        if base.startswith(EXCLUDE_PREFIXES):
            continue
        name = short_name(base)
        test_path = ROOT / "results" / f"{base}_test.csv"
        json_path = ROOT / "results" / f"{base}.json"
        if not test_path.exists() or not json_path.exists():
            excluded.append({"name": name, "reason": "jumeau _test.csv ou JSON manquant"})
            continue
        oof = pd.read_csv(oof_path)["oof"].to_numpy(dtype=np.float64)
        test = pd.read_csv(test_path)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        if len(oof) != N_TRAIN or len(y) != N_TRAIN:
            excluded.append({"name": name, "reason": f"OOF de {len(oof)} lignes au lieu de {N_TRAIN}"})
            continue
        if len(test) != N_TEST or not np.array_equal(test["ID"].to_numpy(), test_ids):
            excluded.append({"name": name, "reason": "IDs test différents de test.parquet"})
            continue
        if not (np.isfinite(oof).all() and np.isfinite(test["proba"]).all()):
            excluded.append({"name": name, "reason": "valeurs non finies"})
            continue
        thr, f1_json = meta["cv"]["threshold"], meta["cv"]["f1"]
        f1_check = f1_score(y, oof >= thr)
        gap = abs(f1_check - f1_json)
        if gap > TOL:
            n_edge = int((np.abs(oof - thr) <= 5e-7).sum())  # lignes dont l'arrondi à 6 décimales touche le seuil
            excluded.append({"name": name, "reason": (f"F1 recalculé {f1_check:.6f} vs JSON {f1_json:.6f} (écart {gap:.2e}) ; "
                                                      f"{n_edge} ligne(s) OOF à moins de 5e-7 du seuil : l'arrondi du CSV à 6 "
                                                      "décimales fait basculer ce(s) cas, d'où un F1 non reproduit à 1e-6 près")})
            continue
        comps[name] = {
            "file": base,
            "oof": oof,
            "test": test["proba"].to_numpy(dtype=np.float64),
            "f1_json": float(f1_json),
            "threshold_json": float(thr),
            "f1_check": float(f1_check),
            "auc": float(roc_auc_score(y, oof)),
        }
        print(f"  {name:<28} F1 JSON {f1_json:.6f}  recalculé {f1_check:.6f}  AUC {comps[name]['auc']:.5f}")
    return comps, excluded


# ---------------------------------------------------------------------------------------------
# Combinaisons
# ---------------------------------------------------------------------------------------------
def blend(ws, names, comps, key):
    out = np.zeros_like(comps[names[0]][key])
    for w, n in zip(ws, names):
        out += w * comps[n][key]
    return out


def rank01(x):
    return (rankdata(x) - 1) / (len(x) - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ml_stack")
    args = parser.parse_args()
    t0 = time.time()
    stamp = time.strftime("%Y%m%d_%H%M%S")

    y = pd.read_parquet(ROOT / "data" / "train_extra.parquet", columns=["TARGET"])["TARGET"].to_numpy(dtype=np.int64)
    test_ids = pd.read_parquet(ROOT / "data" / "test.parquet", columns=["ID"])["ID"].to_numpy()
    assert len(y) == N_TRAIN and len(test_ids) == N_TEST
    print(f"Cible : {len(y)} lignes, taux positif {y.mean():.4f} ; test : {len(test_ids)} IDs")

    print("\n== Composants ==")
    comps, excluded = load_components(y, test_ids)
    for e in excluded:
        print(f"  EXCLU {e['name']}: {e['reason']}")
    if BASE not in comps:
        raise SystemExit(f"Composant de référence {BASE} introuvable ou exclu")
    names = [BASE] + sorted(n for n in comps if n != BASE)
    print(f"\n{len(names)} composants retenus : {names}")

    oof_mat = np.column_stack([comps[n]["oof"] for n in names])
    test_mat = np.column_stack([comps[n]["test"] for n in names])
    corr = pd.DataFrame(oof_mat, columns=names).corr()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n== Corrélation des OOF (Pearson) ==")
    print(corr.round(3).to_string())

    rows = []  # tableau complet
    cands = {}  # nom -> dict(oof, test, metrics, complexity, détails)

    def add(label, oof, test, complexity, grid=GRID, **details):
        m = evaluate(y, oof, grid)
        rows.append({"combination": label, **{k: m[k] for k in ("f1", "auc", "threshold", "precision", "recall")}})
        cands[label] = {"oof": oof, "test": test, "metrics": m, "complexity": complexity, **details}
        print(f"  {label:<60} F1 {m['f1']:.5f}  AUC {m['auc']:.5f}  seuil {m['threshold']:.3f}")
        return m

    # -- composants seuls
    print("\n== Composants seuls ==")
    for n in names:
        add(n, comps[n]["oof"], comps[n]["test"], complexity=0 if n == BASE else 0.5,
            method="single", weights={n: 1.0})

    # -- stacking honnête
    print("\n== Stacking (niveau 2 évalué par cross_val_predict, mêmes folds) ==")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    X_logit, Xt_logit = logit(oof_mat), logit(test_mat)

    lr = LogisticRegression(C=1.0, max_iter=2000)
    oof_lr = cross_val_predict(lr, X_logit, y, cv=skf, method="predict_proba")[:, 1]
    lr.fit(X_logit, y)
    test_lr = lr.predict_proba(Xt_logit)[:, 1]
    coefs_lr = {n: float(c) for n, c in zip(names, lr.coef_[0])}
    add("stack_logreg_logits", oof_lr, test_lr, complexity=4, method="stacking_logreg",
        coefficients={**coefs_lr, "intercept": float(lr.intercept_[0])})
    print("    coefficients :", {k: round(v, 3) for k, v in coefs_lr.items()})

    lgb_meta = lgb.LGBMClassifier(num_leaves=4, n_estimators=200, learning_rate=0.03, min_child_samples=200,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                  random_state=SEED, verbose=-1)
    oof_lgbm = cross_val_predict(lgb_meta, oof_mat, y, cv=skf, method="predict_proba")[:, 1]
    lgb_meta.fit(oof_mat, y)
    test_lgbm = lgb_meta.predict_proba(test_mat)[:, 1]
    add("stack_lgbm_meta", oof_lgbm, test_lgbm, complexity=5, method="stacking_lgbm",
        feature_importance={n: int(v) for n, v in zip(names, lgb_meta.feature_importances_)})

    # -- blends à deux composants
    print("\n== Blends LightGBM baggé + 1 composant ==")
    others = [n for n in names if n != BASE]
    pair_best = {}
    for n in others:
        for w in (0.1, 0.2, 0.3):
            ws, ns = (1 - w, w), (BASE, n)
            m = add(f"blend {BASE} {1 - w:.1f} + {n} {w:.1f}", blend(ws, ns, comps, "oof"), blend(ws, ns, comps, "test"),
                    complexity=1, method="blend", weights=dict(zip(ns, ws)))
            if n not in pair_best or m["f1"] > pair_best[n]:
                pair_best[n] = m["f1"]
    ranked = sorted(others, key=lambda n: -pair_best[n])
    top3 = ranked[:3]
    print(f"\nMeilleurs composants en blend à deux : {[(n, round(pair_best[n], 5)) for n in ranked]}")

    # -- blends à 3 / 4 composants (grille grossière, poids LightGBM >= 0.5)
    print("\n== Blends LightGBM baggé + 2-3 meilleurs composants ==")
    steps = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
    for k in (2, 3):
        for subset in itertools.combinations(top3, k):
            for ws_o in itertools.product(steps, repeat=k):
                if sum(ws_o) > 0.5 + 1e-9:
                    continue
                ws, ns = (1 - sum(ws_o), *ws_o), (BASE, *subset)
                label = "blend " + " + ".join(f"{n} {w:.2f}" for n, w in zip(ns, ws))
                add(label, blend(ws, ns, comps, "oof"), blend(ws, ns, comps, "test"),
                    complexity=k, method="blend", weights=dict(zip(ns, ws)))

    # -- moyenne de rangs pour la meilleure combinaison de blend
    blend_labels = [l for l in cands if cands[l]["method"] == "blend"]
    best_blend = max(blend_labels, key=lambda l: cands[l]["metrics"]["f1"])
    bw = cands[best_blend]["weights"]
    print(f"\n== Moyenne de rangs pondérée pour {best_blend} ==")
    oof_rank = sum(w * rank01(comps[n]["oof"]) for n, w in bw.items())
    test_rank = sum(w * rank01(comps[n]["test"]) for n, w in bw.items())
    add("rank " + " + ".join(f"{n} {w:.2f}" for n, w in bw.items()), oof_rank, test_rank,
        complexity=len(bw) + 0.5, grid=RANK_GRID, method="rank_average", weights=bw,
        note="scores = moyenne pondérée des rangs normalisés dans [0, 1] ; seuil cherché sur [0.5, 0.99]")

    # -- tableau et choix
    table = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    print("\n== Tableau (trié par F1 OOF) ==")
    print(table.round(5).to_string())

    best_f1 = table["f1"].iloc[0]
    close = [l for l in cands if cands[l]["metrics"]["f1"] >= best_f1 - SIMPLICITY_MARGIN]
    chosen = min(close, key=lambda l: (cands[l]["complexity"], -cands[l]["metrics"]["f1"]))
    c = cands[chosen]
    print(f"\nMeilleur F1 : {best_f1:.5f} ({table['combination'].iloc[0]})")
    print(f"Retenu (le plus simple à < {SIMPLICITY_MARGIN} du meilleur) : {chosen}  F1 {c['metrics']['f1']:.5f}")

    # -- sorties au format standard
    name = args.name
    metrics = c["metrics"]
    pred = (c["test"] >= metrics["threshold"]).astype(int)
    sub_dir = ROOT / "submissions"
    sub_dir.mkdir(exist_ok=True)
    sub_path = sub_dir / f"submission_{name}_{stamp}.csv"
    pd.DataFrame({"ID": test_ids, "TARGET": pred}).to_csv(sub_path, index=False)
    pd.DataFrame({"oof": c["oof"]}).to_csv(ROOT / "results" / f"{name}_{stamp}_oof.csv", index=False, float_format="%.6f")
    pd.DataFrame({"ID": test_ids, "proba": c["test"]}).to_csv(
        ROOT / "results" / f"{name}_{stamp}_test.csv", index=False, float_format="%.6f")

    result = {
        "name": name,
        "idea": IDEA,
        "timestamp": stamp,
        "cv": metrics,
        "test_positive_rate": float(pred.mean()),
        "submission": f"submissions/{sub_path.name}",
        "runtime_s": round(time.time() - t0, 1),
        "components": {n: {"file": comps[n]["file"], "f1": comps[n]["f1_json"], "threshold": comps[n]["threshold_json"],
                           "auc": comps[n]["auc"]} for n in names},
        "excluded_components": excluded,
        "selected": chosen,
        "method": c["method"],
        "weights": c.get("weights"),
        "coefficients": c.get("coefficients"),
        "best_by_f1": {"combination": table["combination"].iloc[0], "f1": float(best_f1)},
        "selection_rule": f"meilleur F1 OOF ; solution la plus simple si écart < {SIMPLICITY_MARGIN}",
        "optimism_note": ("Les poids des blends et le choix de la combinaison sont faits sur l'OOF complet : le F1 "
                          "retenu est légèrement optimiste (grille grossière, ~100 combinaisons comparées). Le stacking "
                          "est évalué honnêtement par cross_val_predict (méta-modèle jamais ajusté sur la ligne prédite), "
                          "seul le seuil est choisi sur l'OOF de niveau 2 comme dans toutes les expériences."),
        "stacking": {
            "logreg": {"f1": cands["stack_logreg_logits"]["metrics"]["f1"], "auc": cands["stack_logreg_logits"]["metrics"]["auc"],
                       "coefficients": cands["stack_logreg_logits"]["coefficients"]},
            "lgbm": {"f1": cands["stack_lgbm_meta"]["metrics"]["f1"], "auc": cands["stack_lgbm_meta"]["metrics"]["auc"],
                     "params": {"num_leaves": 4, "n_estimators": 200, "learning_rate": 0.03},
                     "feature_importance": cands["stack_lgbm_meta"]["feature_importance"]},
        },
        "pairwise_best_f1": {n: pair_best[n] for n in ranked},
        "correlation": corr.round(4).to_dict(),
        "table": table.to_dict(orient="records"),
    }
    res_path = ROOT / "results" / f"{name}_{stamp}.json"
    res_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nCV :", {k: round(v, 4) for k, v in metrics.items()})
    print(f"Soumission : {sub_path}\nRésultats : {res_path}  ({result['runtime_s']} s)")


if __name__ == "__main__":
    main()
