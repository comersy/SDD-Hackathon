"""Stacking et blends pondérés de tous les modèles de base disponibles (v3), à partir de leurs
probabilités OOF / test sauvegardées (results/*_oof.csv, *_test.csv). Tout se calcule en local,
sans réentraîner les modèles de base.

Usage : python experiments/ml_stack_v3.py [--name ml_stack_v3]
Sorties : results/<nom>_<horodatage>.json, results/<nom>_<horodatage>_oof.csv, _test.csv,
          submissions/submission_<nom>_<horodatage>.csv

Différences avec ml_stack.py :
- l'ancre des blends est le meilleur composant par F1 OOF (recalculé), pas forcément x_lgbbag_proba ;
- blend des GBDT réglés (hp_lgbm_optuna, hp_xgb_optuna, hp_catboost_gpu, x_lgbbag_proba : moyenne
  pondérée, grille grossière pas 0.1), puis ce blend GBDT combiné aux réseaux (dl_ft_transformer,
  hp_ft_transformer_long, dl_resnet, mlp_huge_50 ; poids total réseaux 0.1-0.4, répartition interne pas 0.2) ;
- stacking honnête (cross_val_predict) inchangé.

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
EXCLUDE_PREFIXES = ("dl_blend", "ml_stack")  # combinaisons, pas des modèles de base
GBDT_NAMES = ["hp_lgbm_optuna", "hp_xgb_optuna", "hp_catboost_gpu", "x_lgbbag_proba"]
NET_NAMES = ["dl_ft_transformer", "hp_ft_transformer_long", "dl_resnet", "mlp_huge_50"]
GBDT_STEP = 0.1  # grille des poids GBDT (somme 1)
NET_TOTALS = (0.1, 0.2, 0.3, 0.4)  # poids total des réseaux dans le blend GBDT + réseaux
NET_FRAC_STEP = 0.2  # répartition interne des réseaux (fractions de somme 1)
N_TRAIN = 370_132
N_TEST = 75_000
SEED = 42
TOL = 1e-6
SIMPLICITY_MARGIN = 0.0003  # en dessous de cet écart de F1, on préfère la solution la plus simple

IDEA = ("Combinaison de tous les modèles de base disponibles (results/*_oof.csv) sans réentraînement, v3 : "
        "(a) stacking honnête, méta-modèle LogisticRegression(C=1) sur les logits des composants, évalué par "
        "cross_val_predict StratifiedKFold(5, shuffle, 42) sur la matrice OOF, seuil sur les prédictions OOF de "
        "niveau 2, méta-modèle refit sur tout l'OOF pour le test ; variante LightGBM régularisé (num_leaves 4, "
        "200 arbres, lr 0.03) ; (b) blends pondérés ancre (= meilleur composant par F1 OOF) + un autre composant "
        "(w 0.1/0.2/0.3), ancre + les 2-3 meilleurs composants (grille grossière, poids ancre >= 0.5) ; "
        "(c) blend des GBDT réglés (hp_lgbm_optuna, hp_xgb_optuna, hp_catboost_gpu, x_lgbbag_proba ; grille pas 0.1) "
        "puis blend GBDT + réseaux (dl_ft_transformer, hp_ft_transformer_long, dl_resnet, mlp_huge_50 ; poids total "
        "réseaux 0.1-0.4, répartition pas 0.2) ; moyenne de rangs du meilleur blend ; "
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


def simplex_grid(k, step):
    """Toutes les répartitions de k poids >= 0, multiples de `step`, de somme 1."""
    units = int(round(1 / step))
    for cuts in itertools.combinations(range(units + k - 1), k - 1):
        parts, prev = [], -1
        for c in cuts:
            parts.append(c - prev - 1)
            prev = c
        parts.append(units + k - 1 - prev - 1)
        yield tuple(round(p * step, 6) for p in parts)


def fmt_weights(ws, ns):
    return " + ".join(f"{n} {w:.2f}" for n, w in zip(ns, ws) if w > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ml_stack_v3")
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
    if not comps:
        raise SystemExit("Aucun composant retenu")
    # (1) ancre = meilleur composant par F1 OOF recalculé
    base = max(comps, key=lambda n: comps[n]["f1_check"])
    names = [base] + sorted(n for n in comps if n != base)
    print(f"\nAncre des blends (meilleur F1 OOF) : {base}  F1 {comps[base]['f1_check']:.5f}")
    print(f"{len(names)} composants retenus : {names}")

    gbdt = [n for n in GBDT_NAMES if n in comps]
    nets = [n for n in NET_NAMES if n in comps]
    missing = [n for n in GBDT_NAMES + NET_NAMES if n not in comps]
    if missing:
        print(f"ATTENTION composants GBDT/réseaux absents ou exclus : {missing}")

    oof_mat = np.column_stack([comps[n]["oof"] for n in names])
    test_mat = np.column_stack([comps[n]["test"] for n in names])
    corr = pd.DataFrame(oof_mat, columns=names).corr()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\n== Corrélation des OOF (Pearson) ==")
    print(corr.round(3).to_string())

    rows = []  # tableau complet
    cands = {}  # nom -> dict(oof, test, metrics, complexity, détails)

    def add(label, oof, test, complexity, grid=GRID, verbose=True, **details):
        m = evaluate(y, oof, grid)
        rows.append({"combination": label, "method": details.get("method"),
                     **{k: m[k] for k in ("f1", "auc", "threshold", "precision", "recall")}})
        cands[label] = {"oof": oof, "test": test, "metrics": m, "complexity": complexity, **details}
        if verbose:
            print(f"  {label:<75} F1 {m['f1']:.5f}  AUC {m['auc']:.5f}  seuil {m['threshold']:.3f}")
        return m

    # -- composants seuls
    print("\n== Composants seuls ==")
    for n in names:
        add(n, comps[n]["oof"], comps[n]["test"], complexity=0 if n == base else 0.5,
            method="single", weights={n: 1.0})

    # -- stacking honnête (inchangé)
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

    # -- blends ancre + 1 composant
    print(f"\n== Blends ancre ({base}) + 1 composant ==")
    others = [n for n in names if n != base]
    pair_best = {}
    for n in others:
        for w in (0.1, 0.2, 0.3):
            ws, ns = (1 - w, w), (base, n)
            m = add(f"blend {base} {1 - w:.1f} + {n} {w:.1f}", blend(ws, ns, comps, "oof"), blend(ws, ns, comps, "test"),
                    complexity=1, method="blend", weights=dict(zip(ns, ws)))
            if n not in pair_best or m["f1"] > pair_best[n]:
                pair_best[n] = m["f1"]
    ranked = sorted(others, key=lambda n: -pair_best[n])
    top3 = ranked[:3]
    print(f"\nMeilleurs composants en blend à deux : {[(n, round(pair_best[n], 5)) for n in ranked]}")

    # -- blends ancre + 2-3 composants (grille grossière, poids ancre >= 0.5)
    print(f"\n== Blends ancre ({base}) + 2-3 meilleurs composants ==")
    steps = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
    for k in (2, 3):
        for subset in itertools.combinations(top3, k):
            for ws_o in itertools.product(steps, repeat=k):
                if sum(ws_o) > 0.5 + 1e-9:
                    continue
                ws, ns = (1 - sum(ws_o), *ws_o), (base, *subset)
                add("blend " + fmt_weights(ws, ns), blend(ws, ns, comps, "oof"), blend(ws, ns, comps, "test"),
                    complexity=k, method="blend", weights=dict(zip(ns, ws)))

    # -- (2) blend des GBDT réglés : moyenne pondérée, grille grossière (pas GBDT_STEP, somme 1)
    gbdt_best, gbdt_oof, gbdt_test, gbdt_w = None, None, None, None
    if len(gbdt) >= 2:
        print(f"\n== Blend GBDT {gbdt} (grille pas {GBDT_STEP}) ==")
        n_eval = 0
        for ws in simplex_grid(len(gbdt), GBDT_STEP):
            nz = [(w, n) for w, n in zip(ws, gbdt) if w > 0]
            if len(nz) < 2:
                continue  # les composants seuls sont déjà évalués
            label = "gbdt " + fmt_weights(ws, gbdt)
            m = add(label, blend(ws, gbdt, comps, "oof"), blend(ws, gbdt, comps, "test"),
                    complexity=len(nz) - 1, verbose=False, method="blend_gbdt", weights={n: w for w, n in nz})
            n_eval += 1
            if gbdt_best is None or m["f1"] > cands[gbdt_best]["metrics"]["f1"]:
                gbdt_best = label
        gbdt_w = cands[gbdt_best]["weights"]
        gbdt_oof, gbdt_test = cands[gbdt_best]["oof"], cands[gbdt_best]["test"]
        mg = cands[gbdt_best]["metrics"]
        print(f"  {n_eval} combinaisons évaluées ; meilleur : {gbdt_best}  F1 {mg['f1']:.5f}  AUC {mg['auc']:.5f}")
        # égalité de poids pour référence
        ws_eq = tuple(round(1 / len(gbdt), 6) for _ in gbdt)
        add("gbdt_equal " + fmt_weights(ws_eq, gbdt), blend(ws_eq, gbdt, comps, "oof"), blend(ws_eq, gbdt, comps, "test"),
            complexity=len(gbdt) - 1, method="blend_gbdt", weights=dict(zip(gbdt, ws_eq)))
    else:
        print("\nBlend GBDT ignoré : moins de 2 composants GBDT disponibles")

    # -- (2bis) blend GBDT (poids figés) + réseaux : poids total réseaux dans NET_TOTALS, répartition interne pas NET_FRAC_STEP
    gn_best = None
    if gbdt_best is not None and nets:
        print(f"\n== Blend GBDT ({gbdt_best}) + réseaux {nets} (total réseaux {NET_TOTALS}, répartition pas {NET_FRAC_STEP}) ==")
        comps["_gbdt_blend"] = {"oof": gbdt_oof, "test": gbdt_test}
        n_eval = 0
        for total in NET_TOTALS:
            for fracs in simplex_grid(len(nets), NET_FRAC_STEP):
                ws = (1 - total, *[round(total * f, 6) for f in fracs])
                ns = ("_gbdt_blend", *nets)
                nz_nets = [(w, n) for w, n in zip(ws[1:], nets) if w > 0]
                label = f"gbdt+nets [gbdt {1 - total:.2f}] " + " + ".join(f"{n} {w:.2f}" for w, n in nz_nets)
                full_w = {**{n: round((1 - total) * w, 6) for n, w in gbdt_w.items()}, **{n: w for w, n in nz_nets}}
                m = add(label, blend(ws, ns, comps, "oof"), blend(ws, ns, comps, "test"),
                        complexity=len(gbdt_w) - 1 + len(nz_nets), verbose=False, method="blend_gbdt_nets",
                        weights=full_w, gbdt_weights=gbdt_w, net_total=total,
                        net_weights={n: w for w, n in nz_nets})
                n_eval += 1
                if gn_best is None or m["f1"] > cands[gn_best]["metrics"]["f1"]:
                    gn_best = label
        del comps["_gbdt_blend"]
        mg = cands[gn_best]["metrics"]
        print(f"  {n_eval} combinaisons évaluées ; meilleur : {gn_best}  F1 {mg['f1']:.5f}  AUC {mg['auc']:.5f}")
        print(f"    poids développés : {cands[gn_best]['weights']}")

    # -- moyenne de rangs pour la meilleure combinaison de blend (toutes familles)
    blend_labels = [l for l in cands if cands[l]["method"] in ("blend", "blend_gbdt", "blend_gbdt_nets")]
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
    print("\n== Tableau (40 premières lignes, trié par F1 OOF) ==")
    print(table.head(40).round(5).to_string())
    print("\n== Meilleur par famille ==")
    for meth, sub in table.groupby("method"):
        r = sub.iloc[0]
        print(f"  {meth:<18} F1 {r['f1']:.5f}  AUC {r['auc']:.5f}  {r['combination']}")

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

    def summary(label):
        if label is None:
            return None
        cc = cands[label]
        return {"combination": label, "f1": cc["metrics"]["f1"], "auc": cc["metrics"]["auc"],
                "threshold": cc["metrics"]["threshold"], "weights": cc.get("weights"),
                "gbdt_weights": cc.get("gbdt_weights"), "net_total": cc.get("net_total"), "net_weights": cc.get("net_weights")}

    result = {
        "name": name,
        "idea": IDEA,
        "timestamp": stamp,
        "cv": metrics,
        "test_positive_rate": float(pred.mean()),
        "submission": f"submissions/{sub_path.name}",
        "runtime_s": round(time.time() - t0, 1),
        "anchor": {"name": base, "f1": comps[base]["f1_check"], "rule": "meilleur composant par F1 OOF recalculé"},
        "components": {n: {"file": comps[n]["file"], "f1": comps[n]["f1_json"], "threshold": comps[n]["threshold_json"],
                           "auc": comps[n]["auc"]} for n in names},
        "excluded_components": excluded,
        "missing_gbdt_or_nets": missing,
        "selected": chosen,
        "method": c["method"],
        "weights": c.get("weights"),
        "coefficients": c.get("coefficients"),
        "best_by_f1": {"combination": table["combination"].iloc[0], "f1": float(best_f1)},
        "best_by_method": {meth: summary(sub.iloc[0]["combination"]) for meth, sub in table.groupby("method")},
        "gbdt_blend": {"components": gbdt, "grid_step": GBDT_STEP, "best": summary(gbdt_best)},
        "gbdt_nets_blend": {"nets": nets, "net_totals": list(NET_TOTALS), "net_frac_step": NET_FRAC_STEP, "best": summary(gn_best)},
        "selection_rule": f"meilleur F1 OOF ; solution la plus simple si écart < {SIMPLICITY_MARGIN}",
        "optimism_note": ("Les poids des blends et le choix de la combinaison sont faits sur l'OOF complet : le F1 "
                          f"retenu est légèrement optimiste (grille grossière, {len(rows)} combinaisons comparées). Le stacking "
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
