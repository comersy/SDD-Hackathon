"""Blend LightGBM seed-bagged + modèles deep learning, à partir des probabilités OOF/test sauvegardées.

Usage : python experiments/dl_blend.py [--name dl_blend] [--tol 1e-6] [--exclude-failed]
Sorties : results/<nom>_<horodatage>.json, results/<nom>_<horodatage>_oof.csv,
          results/<nom>_<horodatage>_test.csv, submissions/submission_<nom>_<horodatage>.csv

Tout se calcule en local (numpy/pandas), sans réentraînement : les composants sont les fichiers
*_oof.csv / *_test.csv écrits par x_lgbbag_proba.py et les scripts dl_*.py.

Protocole d'évaluation (inchangé) : les OOF proviennent du même StratifiedKFold(5, shuffle=True,
random_state=42) sur train.csv + extra.csv (370 132 lignes) ; le seuil de décision est choisi sur
l'OOF du blend pour maximiser le F1 ; le F1 reporté est ce F1 OOF, au seuil retenu.

Contrôles avant tout blend :
- 370 132 lignes OOF par composant, 75 000 lignes test avec les mêmes ID que test.csv, pas de NaN ;
- le F1 OOF recalculé de chaque composant seul doit retrouver celui de son JSON à --tol près
  (1e-6 par défaut) ; sinon le script s'arrête et explique, sauf --exclude-failed qui continue
  sans le composant fautif (le diagnostic est conservé dans le JSON de sortie).
"""
import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

IDEA = ("Blend post-hoc, sans réentraînement, du LightGBM seed-bagged (x_lgbbag_proba, meilleur modèle, "
        "F1 CV 0.4247) avec les modèles deep learning validés (MLP, FT-Transformer, TabTransformer, "
        "ResNet, FT-Transformer k40, TabNet), à partir de leurs probabilités OOF/test sauvegardées : "
        "(a) LightGBM + un DL avec poids DL w dans {0.1..0.5}, (b) LightGBM + les deux meilleurs DL "
        "(grille à pas 0.1, poids DL total <= 0.5), (c) moyenne de rangs (rankdata/n) LightGBM + un DL "
        "avec w = 0.3 ; seuil choisi sur l'OOF du blend ; sélection du meilleur F1 OOF avec préférence "
        "pour le blend le plus simple si l'écart est < 0.0003")

ROOT = Path(__file__).resolve().parent.parent
N_TRAIN, N_TEST = 370_132, 75_000
LGB_NAME = "lgbm_bagged"
COMPONENTS = [
    {"name": "lgbm_bagged", "stem": "x_lgbbag_proba_20261020_082151"},
    {"name": "dl_base_mlp", "stem": "dl_base_20261020_081814"},
    {"name": "dl_ft_transformer", "stem": "dl_ft_transformer_20261020_085906"},
    {"name": "dl_tabtransformer", "stem": "dl_tabtransformer_20261020_085010"},
    {"name": "dl_resnet", "stem": "dl_resnet_20261020_082804"},
    {"name": "dl_ft_transformer_k40", "stem": "dl_ft_transformer_k40_20261020_084542"},
    {"name": "dl_tabnet", "stem": "dl_tabnet_20261020_083445"},
]
SINGLE_WEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5]  # (a) poids du DL
PAIR_GRID = [0.1, 0.2, 0.3, 0.4]            # (b) poids de chaque DL, total <= 0.5
RANK_WEIGHT = 0.3                            # (c)
TIE_MARGIN = 0.0003                          # préférence pour le plus simple sous cet écart

PROBA_GRID = np.arange(0.05, 0.6, 0.005)    # identique à r1_clean_codes.best_threshold
RANK_GRID = np.arange(0.70, 0.98, 0.0025)   # scores de rang : seuil = quantile (positifs ~ 9 %)


# --- métriques (même logique que experiments/r1_clean_codes.py, grille paramétrable pour les rangs)
def best_threshold(y, proba, grid=PROBA_GRID):
    scores = [f1_score(y, proba >= t) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def evaluate(y, oof, grid=PROBA_GRID):
    threshold, f1 = best_threshold(y, oof, grid)
    pred = oof >= threshold
    return {
        "f1": f1,
        "threshold": threshold,
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "auc": float(roc_auc_score(y, oof)),
    }


# --- chargement et contrôles ---------------------------------------------------------------------
def load_components(y, test_ids, tol, exclude_failed):
    comps, failures = [], []
    for c in COMPONENTS:
        stem = c["stem"]
        js = json.loads((ROOT / "results" / f"{stem}.json").read_text(encoding="utf-8"))
        oof = pd.read_csv(ROOT / "results" / f"{stem}_oof.csv")["oof"].to_numpy(dtype=np.float64)
        te = pd.read_csv(ROOT / "results" / f"{stem}_test.csv")
        problems = []
        if len(oof) != N_TRAIN:
            problems.append(f"OOF : {len(oof)} lignes au lieu de {N_TRAIN}")
        if len(te) != N_TEST:
            problems.append(f"test : {len(te)} lignes au lieu de {N_TEST}")
        elif not np.array_equal(te["ID"].to_numpy(), test_ids):
            problems.append("test : ID différents de test.csv (ordre ou valeurs)")
        if np.isnan(oof).any() or te["proba"].isna().any():
            problems.append("NaN dans les probabilités")
        if problems:  # alignement cassé : on ne va pas plus loin
            raise SystemExit(f"ARRÊT — composant {c['name']} ({stem}) : " + " ; ".join(problems))
        m = evaluate(y, oof)
        diff = m["f1"] - js["cv"]["f1"]
        entry = {
            **c,
            "oof_csv": f"results/{stem}_oof.csv", "test_csv": f"results/{stem}_test.csv",
            "results_json": f"results/{stem}.json",
            "cv_f1_json": js["cv"]["f1"], "cv_f1_recomputed": m["f1"], "f1_diff": diff,
            "threshold_json": js["cv"]["threshold"], "threshold_recomputed": m["threshold"],
            "auc_json": js["cv"]["auc"], "auc_recomputed": m["auc"],
            "oof": oof, "test": te["proba"].to_numpy(dtype=np.float64), "metrics": m,
        }
        ok = abs(diff) <= tol
        print(f"  {c['name']:22s} F1 recalculé {m['f1']:.8f}  JSON {js['cv']['f1']:.8f}  "
              f"écart {diff:+.2e}  seuil {m['threshold']:.3f}/{js['cv']['threshold']:.3f}  "
              f"AUC {m['auc']:.6f}  -> {'OK' if ok else 'ÉCART > tol'}")
        if ok:
            comps.append(entry)
            continue
        # Diagnostic : le CSV est écrit avec 6 décimales ; des valeurs juste sous le seuil peuvent
        # être arrondies exactement au seuil et basculer en positifs (>=), ce qui change le F1.
        t = js["cv"]["threshold"]
        at = np.isclose(oof, t, atol=1e-9)
        f1_strict = f1_score(y, oof > t)
        explain = (f"{int(at.sum())} valeur(s) OOF exactement égales au seuil {t:.3f} après arrondi à 6 "
                   f"décimales (dont {int(y[at].sum())} positif(s)) ; F1 avec '>' au lieu de '>=' = "
                   f"{f1_strict:.8f} (écart au JSON {f1_strict - js['cv']['f1']:+.2e})")
        print(f"    diagnostic : {explain}")
        failures.append({k: entry[k] for k in ("name", "stem", "results_json", "cv_f1_json",
                                                 "cv_f1_recomputed", "f1_diff")} | {"diagnostic": explain})
        if not exclude_failed:
            raise SystemExit(
                f"ARRÊT — le F1 OOF recalculé de {c['name']} ne retrouve pas son JSON à {tol:g} près "
                f"(écart {diff:+.2e}). Cause probable : {explain}. Relance avec --exclude-failed pour "
                f"continuer sans ce composant, ou --tol plus large si l'écart est jugé acceptable.")
        print(f"    -> composant {c['name']} exclu du blend (--exclude-failed)")
    return comps, failures


# --- blends --------------------------------------------------------------------------------------
def blend_proba(comps, weights):
    oof = sum(w * c["oof"] for c, w in zip(comps, weights))
    test = sum(w * c["test"] for c, w in zip(comps, weights))
    return oof, test


def blend_rank(comps, weights):
    oof = sum(w * rankdata(c["oof"]) / len(c["oof"]) for c, w in zip(comps, weights))
    test = sum(w * rankdata(c["test"]) / len(c["test"]) for c, w in zip(comps, weights))
    return oof, test


def run_trial(y, kind, comps, weights, label):
    oof, test = (blend_rank if kind == "rank" else blend_proba)(comps, weights)
    m = evaluate(y, oof, RANK_GRID if kind == "rank" else PROBA_GRID)
    return {
        "label": label, "type": kind, "n_components": len(comps),
        "components": [c["name"] for c in comps],
        "weights": {c["name"]: float(w) for c, w in zip(comps, weights)},
        **m, "_oof": oof, "_test": test,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dl_blend")
    ap.add_argument("--tol", type=float, default=1e-6, help="tolérance sur le F1 recalculé de chaque composant")
    ap.add_argument("--exclude-failed", action="store_true",
                    help="continuer sans les composants dont le F1 ne se reproduit pas (au lieu de s'arrêter)")
    args = ap.parse_args()
    t0 = time.time()

    y = pd.read_parquet(ROOT / "data" / "train_extra.parquet", columns=["TARGET"])["TARGET"].to_numpy().astype(int)
    test_ids = pd.read_parquet(ROOT / "data" / "test.parquet", columns=["ID"])["ID"].to_numpy()
    assert len(y) == N_TRAIN and len(test_ids) == N_TEST, (len(y), len(test_ids))
    print(f"cible : {len(y)} lignes, taux de positifs {y.mean():.4f} ; test : {len(test_ids)} ID")

    print("1. contrôle des composants")
    comps, failures = load_components(y, test_ids, args.tol, args.exclude_failed)
    lgb = next(c for c in comps if c["name"] == LGB_NAME)
    dls = sorted([c for c in comps if c["name"] != LGB_NAME], key=lambda c: -c["cv_f1_json"])
    print(f"   LightGBM : {lgb['name']} ; DL ({len(dls)}) : {[c['name'] for c in dls]}")

    print("2. blends")
    trials = []
    for c in [lgb] + dls:  # références : composants seuls
        trials.append({"label": f"single:{c['name']}", "type": "single", "n_components": 1,
                       "components": [c["name"]], "weights": {c["name"]: 1.0}, **c["metrics"],
                       "_oof": c["oof"], "_test": c["test"]})
    for d in dls:  # (a)
        for w in SINGLE_WEIGHTS:
            trials.append(run_trial(y, "proba", [lgb, d], [1 - w, w], f"a:lgb+{d['name']} w={w}"))
    top2 = dls[:2]  # (b) deux meilleurs DL d'après leur F1 CV individuel (JSON)
    for w1, w2 in product(PAIR_GRID, PAIR_GRID):
        if w1 + w2 <= 0.5 + 1e-9:
            trials.append(run_trial(y, "proba", [lgb, *top2], [1 - w1 - w2, w1, w2],
                                    f"b:lgb+{top2[0]['name']}({w1})+{top2[1]['name']}({w2})"))
    for d in dls:  # (c)
        trials.append(run_trial(y, "rank", [lgb, d], [1 - RANK_WEIGHT, RANK_WEIGHT],
                                f"c:rank lgb+{d['name']} w={RANK_WEIGHT}"))
    all_c = [lgb] + dls  # poids égaux sur tous les composants valides (référence)
    equal = run_trial(y, "proba", all_c, [1 / len(all_c)] * len(all_c), f"equal:{len(all_c)} composants")
    trials.append(equal)

    table = pd.DataFrame([{k: v for k, v in t.items() if not k.startswith("_")} for t in trials])
    table["weights"] = table["weights"].map(lambda d: " ".join(f"{k}={v:.3g}" for k, v in d.items()))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(table[["label", "type", "f1", "auc", "threshold", "precision", "recall"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("3. sélection")
    candidates = [t for t in trials if t["type"] in ("proba", "rank")]
    best = max(candidates, key=lambda t: t["f1"])
    near = [t for t in candidates if t["f1"] >= best["f1"] - TIE_MARGIN]
    # plus simple = moins de composants, puis moyenne de probabilités plutôt que de rangs, puis meilleur F1
    chosen = min(near, key=lambda t: (t["n_components"], t["type"] == "rank", -t["f1"]))
    print(f"   meilleur F1 : {best['label']} ({best['f1']:.6f}) ; {len(near)} candidat(s) à moins de "
          f"{TIE_MARGIN} ; retenu : {chosen['label']} ({chosen['f1']:.6f})")
    print(f"   F1 du LightGBM seul : {lgb['metrics']['f1']:.6f} -> blend : {chosen['f1']:.6f} "
          f"(gain {chosen['f1'] - lgb['metrics']['f1']:+.6f}) ; poids égaux : {equal['f1']:.6f}")

    print("4. sorties")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (ROOT / "submissions").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    metrics = {k: chosen[k] for k in ("f1", "threshold", "precision", "recall", "auc")}
    pred = (chosen["_test"] >= metrics["threshold"]).astype(int)
    sub_path = ROOT / "submissions" / f"submission_{args.name}_{stamp}.csv"
    pd.DataFrame({"ID": test_ids, "TARGET": pred}).to_csv(sub_path, index=False)
    pd.DataFrame({"oof": chosen["_oof"]}).to_csv(ROOT / "results" / f"{args.name}_{stamp}_oof.csv",
                                                 index=False, float_format="%.6f")
    pd.DataFrame({"ID": test_ids, "proba": chosen["_test"]}).to_csv(
        ROOT / "results" / f"{args.name}_{stamp}_test.csv", index=False, float_format="%.6f")
    comp_info = [{k: c[k] for k in ("name", "oof_csv", "test_csv", "results_json", "cv_f1_json",
                                     "cv_f1_recomputed", "f1_diff", "auc_json")}
                 | {"weight": chosen["weights"].get(c["name"], 0.0)} for c in all_c]
    result = {
        "name": args.name,
        "idea": IDEA,
        "timestamp": stamp,
        "cv": metrics,
        "test_positive_rate": float(pred.mean()),
        "submission": f"submissions/{sub_path.name}",
        "runtime_s": round(time.time() - t0, 1),
        "model": "blend_lgbm_bagged_dl",
        "blend_type": chosen["type"],
        "blend_label": chosen["label"],
        "components": comp_info,
        "weights": chosen["weights"],
        "equal_weight_f1": equal["f1"],
        "equal_weight_components": equal["components"],
        "lgbm_alone_f1": lgb["metrics"]["f1"],
        "best_trial": {k: v for k, v in best.items() if not k.startswith("_")},
        "selection_rule": (f"meilleur F1 OOF parmi les blends, mais le plus simple (moins de composants, "
                           f"probabilités plutôt que rangs) est préféré si l'écart est < {TIE_MARGIN}"),
        "caveat": ("les poids du blend et le seuil sont choisis sur les mêmes OOF que le F1 reporté : "
                   "légère optimisme (quelques paramètres discrets : 1-2 poids sur une grille à pas 0.1 "
                   "et un seuil), non corrigé par CV imbriquée ; les composants DL eux-mêmes n'ont pas "
                   "été réentraînés ; les probabilités lues sont celles des CSV (6 décimales)"),
        "rank_blend_note": ("pour les blends de rangs, le score = somme pondérée de rankdata/n calculée "
                            "séparément sur l'OOF et sur le test ; le seuil est donc un quantile, appliqué "
                            "au test comme fraction de positifs"),
        "check_tolerance": args.tol,
        "excluded_components": failures,
        "trials": [{k: v for k, v in t.items() if not k.startswith("_")} for t in trials],
    }
    res_path = ROOT / "results" / f"{args.name}_{stamp}.json"
    res_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("CV :", {k: round(v, 4) for k, v in metrics.items()})
    print(f"Soumission : {sub_path}\nRésultats : {res_path}  ({result['runtime_s']} s)")


if __name__ == "__main__":
    main()
