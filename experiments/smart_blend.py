"""Blend local (numpy/pandas, secondes) des modèles « intelligents » validés : smart_xgb, smart_lgbm,
smart_mlp_towers, smart_rf, à partir de leurs probabilités OOF/test sauvegardées.

Usage : python experiments/smart_blend.py [--name smart_blend] [--tol 1e-6] [--exclude-failed]
Sorties : results/<nom>_<horodatage>.json, results/<nom>_<horodatage>_oof.csv,
          results/<nom>_<horodatage>_test.csv, submissions/submission_<nom>_<horodatage>.csv,
          results/<nom>_prevention_<horodatage>_oof.csv, results/<nom>_prevention_<horodatage>_test.csv

Même protocole que experiments/dl_blend.py : les OOF viennent du même StratifiedKFold(5, shuffle=True,
random_state=42) sur train_extra.parquet (370 132 lignes) ; le seuil est choisi sur l'OOF du blend pour
maximiser le F1 ; le F1 reporté est ce F1 OOF, au seuil retenu.

Contrôles avant tout blend : 370 132 lignes OOF par composant, 75 000 lignes test avec les mêmes ID que
test.parquet, pas de NaN, et le F1 OOF recalculé de chaque composant seul doit retrouver celui de son JSON
à --tol près (sinon arrêt, sauf --exclude-failed). Pour les composants prévention, --tol-prevention
(1e-5) : le CSV OOF de smart_xgb prévention, arrondi à 6 décimales, redonne le F1 du JSON à -4.4e-6 près.

Deux blends sont construits avec la même procédure (poids égaux, grille grossière à pas 0.1, moyenne de
rangs ; sélection du meilleur F1 OOF avec préférence pour le plus simple si l'écart < 0.0003) :
- le blend complet (fichiers <stem>_oof/_test), qui donne la soumission ;
- le blend « prévention » (fichiers <stem>_prevention_*, tiers A+B seulement), dont les métriques vont
  sous "prevention_blend" dans le JSON. Pour smart_mlp_towers, le composant prévention retenu est le réseau
  A+B réentraîné (prevention_retrained, F1 0.3826), vrai modèle amont+comorbidités, plutôt que la
  décomposition des tours du réseau complet (prevention_model, F1 0.3745).
"""
import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

IDEA = ("Blend post-hoc, sans réentraînement, des quatre modèles « intelligents » (tiers de colonnes, "
        "monotonie, dérivées) : XGBoost GPU bagged, LightGBM bagged, MLP à tours, Random Forest, à partir "
        "de leurs OOF/test : (a) moyenne à poids égaux, (b) grille grossière à pas 0.1 (paires xgb+lgbm, "
        "xgb+mlp, lgbm+mlp ; puis GBDT = (xgb+lgbm)/2 avec poids mlp/rf <= 0.5 au total), (c) moyennes de "
        "rangs ; seuil choisi sur l'OOF du blend ; sélection du meilleur F1 OOF avec préférence pour le "
        "plus simple si l'écart est < 0.0003. Même procédure sur les modèles prévention (tiers A+B) -> "
        "\"prevention_blend\".")

ROOT = Path(__file__).resolve().parent.parent
N_TRAIN, N_TEST = 370_132, 75_000
COMPONENTS = [  # ordre = F1 CV décroissant (JSON)
    {"name": "smart_xgb", "stem": "smart_xgb_20261020_112119", "short": "xgb"},
    {"name": "smart_lgbm", "stem": "smart_lgbm_20261020_112854", "short": "lgbm"},
    {"name": "smart_mlp_towers", "stem": "smart_mlp_towers_20261020_114011", "short": "mlp"},
    {"name": "smart_rf", "stem": "smart_rf_20261020_112427", "short": "rf"},
]
PAIR_WEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5]  # poids du second composant d'une paire (le plus faible)
XGB_LGBM_WEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # poids du LightGBM face au XGBoost
EXTRA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4]    # poids mlp / rf autour du bloc GBDT (xgb+lgbm à parts égales)
EXTRA_TOTAL_MAX = 0.5
TIE_MARGIN = 0.0003

PROBA_GRID = np.arange(0.05, 0.6, 0.005)    # identique à r1_clean_codes.best_threshold
RANK_GRID = np.arange(0.70, 0.98, 0.0025)   # scores de rang : seuil = quantile


# --- métriques ---------------------------------------------------------------------------------
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


# --- chargement et contrôles ----------------------------------------------------------------------
def json_metrics(js, prevention):
    """Métriques de référence (F1, seuil, AUC) du composant dans son JSON."""
    if not prevention:
        return js["cv"]
    if "prevention_retrained" in js:          # smart_mlp_towers : réseau A+B réentraîné
        return js["prevention_retrained"]
    pm = js["prevention_model"]
    return pm["cv"] if "cv" in pm else pm


def prevention_files(js):
    if "prevention_retrained" in js:          # smart_mlp_towers : réseau A+B réentraîné
        return js["prevention_retrained"]["oof_csv"], js["prevention_retrained"]["test_csv"]
    return js["prevention_model"]["oof_csv"], js["prevention_model"]["test_csv"]


def load_components(y, test_ids, tol, exclude_failed, prevention=False):
    comps, failures = [], []
    for c in COMPONENTS:
        stem = c["stem"]
        js = json.loads((ROOT / "results" / f"{stem}.json").read_text(encoding="utf-8"))
        if prevention:
            oof_csv, test_csv = prevention_files(js)
        else:
            oof_csv, test_csv = f"results/{stem}_oof.csv", f"results/{stem}_test.csv"
        ref = json_metrics(js, prevention)
        oof = pd.read_csv(ROOT / oof_csv)["oof"].to_numpy(dtype=np.float64)
        te = pd.read_csv(ROOT / test_csv)
        problems = []
        if len(oof) != N_TRAIN:
            problems.append(f"OOF : {len(oof)} lignes au lieu de {N_TRAIN}")
        if len(te) != N_TEST:
            problems.append(f"test : {len(te)} lignes au lieu de {N_TEST}")
        elif not np.array_equal(te["ID"].to_numpy(), test_ids):
            problems.append("test : ID différents de test.parquet (ordre ou valeurs)")
        if np.isnan(oof).any() or te["proba"].isna().any():
            problems.append("NaN dans les probabilités")
        if problems:
            raise SystemExit(f"ARRÊT — composant {c['name']} ({oof_csv}) : " + " ; ".join(problems))
        m = evaluate(y, oof)
        diff = m["f1"] - ref["f1"]
        entry = {
            **c,
            "oof_csv": oof_csv, "test_csv": test_csv, "results_json": f"results/{stem}.json",
            "cv_f1_json": ref["f1"], "cv_f1_recomputed": m["f1"], "f1_diff": diff,
            "threshold_json": ref["threshold"], "threshold_recomputed": m["threshold"],
            "auc_json": ref["auc"], "auc_recomputed": m["auc"],
            "oof": oof, "test": te["proba"].to_numpy(dtype=np.float64), "metrics": m,
        }
        ok = abs(diff) <= tol
        print(f"  {c['name']:18s} F1 recalculé {m['f1']:.8f}  JSON {ref['f1']:.8f}  écart {diff:+.2e}  "
              f"seuil {m['threshold']:.3f}/{ref['threshold']:.3f}  AUC {m['auc']:.6f}  "
              f"-> {'OK' if ok else 'ÉCART > tol'}")
        if ok:
            comps.append(entry)
            continue
        t = ref["threshold"]
        at = np.isclose(oof, t, atol=1e-9)
        f1_strict = f1_score(y, oof > t)
        explain = (f"{int(at.sum())} valeur(s) OOF exactement égales au seuil {t:.3f} après arrondi à 6 "
                   f"décimales (dont {int(y[at].sum())} positif(s)) ; F1 avec '>' au lieu de '>=' = "
                   f"{f1_strict:.8f} (écart au JSON {f1_strict - ref['f1']:+.2e})")
        print(f"    diagnostic : {explain}")
        failures.append({k: entry[k] for k in ("name", "stem", "oof_csv", "results_json", "cv_f1_json",
                                                 "cv_f1_recomputed", "f1_diff")} | {"diagnostic": explain})
        if not exclude_failed:
            raise SystemExit(
                f"ARRÊT — le F1 OOF recalculé de {c['name']} ne retrouve pas son JSON à {tol:g} près "
                f"(écart {diff:+.2e}). Cause probable : {explain}. Relance avec --exclude-failed pour "
                f"continuer sans ce composant, ou --tol plus large si l'écart est jugé acceptable.")
        print(f"    -> composant {c['name']} exclu du blend (--exclude-failed)")
    return comps, failures


# --- blends ----------------------------------------------------------------------------------------
def blend_proba(comps, weights):
    oof = sum(w * c["oof"] for c, w in zip(comps, weights))
    test = sum(w * c["test"] for c, w in zip(comps, weights))
    return oof, test


def blend_rank(comps, weights):
    oof = sum(w * rankdata(c["oof"]) / len(c["oof"]) for c, w in zip(comps, weights))
    test = sum(w * rankdata(c["test"]) / len(c["test"]) for c, w in zip(comps, weights))
    return oof, test


def run_trial(y, kind, comps, weights, label, equal=False):
    keep = [(c, w) for c, w in zip(comps, weights) if w > 0]
    comps, weights = [c for c, _ in keep], [w for _, w in keep]
    oof, test = (blend_rank if kind == "rank" else blend_proba)(comps, weights)
    m = evaluate(y, oof, RANK_GRID if kind == "rank" else PROBA_GRID)
    return {
        "label": label, "type": kind, "n_components": len(comps), "equal_weights": equal,
        "components": [c["name"] for c in comps],
        "weights": {c["name"]: float(round(w, 6)) for c, w in zip(comps, weights)},
        **m, "_oof": oof, "_test": test,
    }


def build_trials(y, comps):
    by = {c["short"]: c for c in comps}
    trials = []
    for c in comps:  # références : composants seuls
        trials.append({"label": f"single:{c['short']}", "type": "single", "n_components": 1,
                       "equal_weights": True, "components": [c["name"]], "weights": {c["name"]: 1.0},
                       **c["metrics"], "_oof": c["oof"], "_test": c["test"]})
    # (a) moyennes à poids égaux
    n = len(comps)
    trials.append(run_trial(y, "proba", comps, [1 / n] * n, f"equal:{'+'.join(c['short'] for c in comps)}",
                            equal=True))
    if all(k in by for k in ("xgb", "lgbm", "mlp")):
        trio = [by["xgb"], by["lgbm"], by["mlp"]]
        trials.append(run_trial(y, "proba", trio, [1 / 3] * 3, "equal:xgb+lgbm+mlp", equal=True))
    # (b) grille grossière, pas 0.1
    n_grid = 0
    if "xgb" in by and "lgbm" in by:
        for w in XGB_LGBM_WEIGHTS:
            trials.append(run_trial(y, "proba", [by["xgb"], by["lgbm"]], [1 - w, w],
                                    f"b:xgb+lgbm w_lgbm={w}", equal=(w == 0.5)))
            n_grid += 1
    for first, second in (("xgb", "mlp"), ("lgbm", "mlp")):
        if first in by and second in by:
            for w in PAIR_WEIGHTS:
                trials.append(run_trial(y, "proba", [by[first], by[second]], [1 - w, w],
                                        f"b:{first}+{second} w_{second}={w}", equal=(w == 0.5)))
                n_grid += 1
    if all(k in by for k in ("xgb", "lgbm")):
        extras = [k for k in ("mlp", "rf") if k in by]
        for ws in product(EXTRA_GRID, repeat=len(extras)):
            tot = sum(ws)
            if tot == 0 or tot > EXTRA_TOTAL_MAX + 1e-9:
                continue
            g = (1 - tot) / 2
            label = "b:gbdt(xgb=lgbm=%.2f)+" % g + "+".join(f"{k}({w})" for k, w in zip(extras, ws) if w > 0)
            trials.append(run_trial(y, "proba", [by["xgb"], by["lgbm"]] + [by[k] for k in extras],
                                    [g, g] + list(ws), label))
            n_grid += 1
    # (c) moyennes de rangs
    trials.append(run_trial(y, "rank", comps, [1 / n] * n, f"rank:{'+'.join(c['short'] for c in comps)}",
                            equal=True))
    if all(k in by for k in ("xgb", "lgbm", "mlp")):
        trio = [by["xgb"], by["lgbm"], by["mlp"]]
        trials.append(run_trial(y, "rank", trio, [1 / 3] * 3, "rank:xgb+lgbm+mlp", equal=True))
        trials.append(run_trial(y, "rank", trio, [0.35, 0.35, 0.3], "rank:xgb(0.35)+lgbm(0.35)+mlp(0.3)"))
    if "xgb" in by and "lgbm" in by:
        trials.append(run_trial(y, "rank", [by["xgb"], by["lgbm"]], [0.5, 0.5], "rank:xgb+lgbm", equal=True))
    print(f"   {n_grid} combinaisons de grille (pas 0.1), {len(trials)} essais au total")
    return trials


def select(trials, comps):
    candidates = [t for t in trials if t["type"] in ("proba", "rank")]
    best = max(candidates, key=lambda t: t["f1"])
    near = [t for t in candidates if t["f1"] >= best["f1"] - TIE_MARGIN]
    # plus simple = moins de composants, puis probabilités plutôt que rangs, puis poids égaux, puis F1
    chosen = min(near, key=lambda t: (t["n_components"], t["type"] == "rank", not t["equal_weights"], -t["f1"]))
    top = comps[0]
    print(f"   meilleur F1 : {best['label']} ({best['f1']:.6f}) ; {len(near)} candidat(s) à moins de "
          f"{TIE_MARGIN} ; retenu : {chosen['label']} ({chosen['f1']:.6f})")
    print(f"   meilleur composant seul ({top['name']}) : {top['metrics']['f1']:.6f} -> blend : "
          f"{chosen['f1']:.6f} (gain {chosen['f1'] - top['metrics']['f1']:+.6f})")
    return best, chosen, near


def print_table(trials):
    table = pd.DataFrame([{k: v for k, v in t.items() if not k.startswith("_")} for t in trials])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(table[["label", "type", "f1", "auc", "threshold", "precision", "recall"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))


def public(t):
    return {k: v for k, v in t.items() if not k.startswith("_")}


def run_blend(y, comps, title):
    print(f"2{title}. blends")
    trials = build_trials(y, comps)
    print_table(trials)
    print(f"3{title}. sélection")
    best, chosen, near = select(trials, comps)
    equal = next(t for t in trials if t["label"].startswith("equal:") and t["n_components"] == len(comps))
    return trials, best, chosen, near, equal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="smart_blend")
    ap.add_argument("--tol", type=float, default=1e-6, help="tolérance sur le F1 recalculé de chaque composant")
    ap.add_argument("--tol-prevention", type=float, default=1e-5,
                    help="tolérance pour les composants prévention (CSV arrondis à 6 décimales : smart_xgb "
                         "prévention a 2 OOF exactement au seuil 0.185, écart F1 -4.4e-6)")
    ap.add_argument("--exclude-failed", action="store_true",
                    help="continuer sans les composants dont le F1 ne se reproduit pas (au lieu de s'arrêter)")
    args = ap.parse_args()
    t0 = time.time()

    y = pd.read_parquet(ROOT / "data" / "train_extra.parquet", columns=["TARGET"])["TARGET"].to_numpy().astype(int)
    test_ids = pd.read_parquet(ROOT / "data" / "test.parquet", columns=["ID"])["ID"].to_numpy()
    assert len(y) == N_TRAIN and len(test_ids) == N_TEST, (len(y), len(test_ids))
    print(f"cible : {len(y)} lignes, taux de positifs {y.mean():.4f} ; test : {len(test_ids)} ID")

    print("1. contrôle des composants (modèles complets)")
    comps, failures = load_components(y, test_ids, args.tol, args.exclude_failed)
    comps.sort(key=lambda c: -c["cv_f1_json"])
    trials, best, chosen, near, equal = run_blend(y, comps, "")

    print("1p. contrôle des composants (modèles prévention, tiers A+B)")
    pcomps, pfailures = load_components(y, test_ids, args.tol_prevention, args.exclude_failed, prevention=True)
    pcomps.sort(key=lambda c: -c["cv_f1_json"])
    ptrials, pbest, pchosen, pnear, pequal = run_blend(y, pcomps, "p")

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
    pmetrics = {k: pchosen[k] for k in ("f1", "threshold", "precision", "recall", "auc")}
    ppred = (pchosen["_test"] >= pmetrics["threshold"]).astype(int)
    pd.DataFrame({"oof": pchosen["_oof"]}).to_csv(ROOT / "results" / f"{args.name}_prevention_{stamp}_oof.csv",
                                                  index=False, float_format="%.6f")
    pd.DataFrame({"ID": test_ids, "proba": pchosen["_test"]}).to_csv(
        ROOT / "results" / f"{args.name}_prevention_{stamp}_test.csv", index=False, float_format="%.6f")

    def comp_info(cs, ch):
        return [{k: c[k] for k in ("name", "oof_csv", "test_csv", "results_json", "cv_f1_json",
                                   "cv_f1_recomputed", "f1_diff", "auc_json")}
                | {"weight": ch["weights"].get(c["name"], 0.0)} for c in cs]

    result = {
        "name": args.name,
        "idea": IDEA,
        "timestamp": stamp,
        "cv": metrics,
        "test_positive_rate": float(pred.mean()),
        "submission": f"submissions/{sub_path.name}",
        "oof_csv": f"results/{args.name}_{stamp}_oof.csv",
        "test_csv": f"results/{args.name}_{stamp}_test.csv",
        "runtime_s": round(time.time() - t0, 1),
        "model": "blend_smart_xgb_lgbm_mlp_rf",
        "blend_type": chosen["type"],
        "blend_label": chosen["label"],
        "components": comp_info(comps, chosen),
        "weights": chosen["weights"],
        "equal_weight_f1": equal["f1"],
        "equal_weight_auc": equal["auc"],
        "equal_weight_components": equal["components"],
        "best_single_name": comps[0]["name"],
        "best_single_f1": comps[0]["metrics"]["f1"],
        "best_trial": public(best),
        "n_candidates_within_margin": len(near),
        "selection_rule": (f"meilleur F1 OOF parmi les blends, mais le plus simple (moins de composants, "
                           f"probabilités plutôt que rangs, poids égaux plutôt qu'ajustés) est préféré si "
                           f"l'écart est < {TIE_MARGIN}"),
        "caveat": ("les poids du blend et le seuil sont choisis sur les mêmes OOF que le F1 reporté : léger "
                   "optimisme (1-2 poids sur une grille à pas 0.1 et un seuil), non corrigé par CV imbriquée ; "
                   "les composants ne sont pas réentraînés ; probabilités lues dans les CSV (6 décimales)"),
        "rank_blend_note": ("pour les blends de rangs, score = somme pondérée de rankdata/n calculée séparément "
                            "sur l'OOF et sur le test ; le seuil est un quantile, appliqué au test comme "
                            "fraction de positifs"),
        "check_tolerance": args.tol,
        "excluded_components": failures,
        "trials": [public(t) for t in trials],
        "prevention_blend": {
            "tiers": ["A_amont", "B_comorbidites"],
            "cv": pmetrics,
            "f1": pmetrics["f1"], "auc": pmetrics["auc"], "threshold": pmetrics["threshold"],
            "precision": pmetrics["precision"], "recall": pmetrics["recall"],
            "test_positive_rate": float(ppred.mean()),
            "blend_type": pchosen["type"],
            "blend_label": pchosen["label"],
            "weights": pchosen["weights"],
            "components": comp_info(pcomps, pchosen),
            "mlp_component_note": ("smart_mlp_towers : réseau A+B réentraîné (prevention_retrained) plutôt que "
                                   "la décomposition des tours du réseau complet (prevention_model)"),
            "equal_weight_f1": pequal["f1"],
            "equal_weight_auc": pequal["auc"],
            "best_single_name": pcomps[0]["name"],
            "best_single_f1": pcomps[0]["metrics"]["f1"],
            "best_trial": public(pbest),
            "n_candidates_within_margin": len(pnear),
            "oof_csv": f"results/{args.name}_prevention_{stamp}_oof.csv",
            "test_csv": f"results/{args.name}_prevention_{stamp}_test.csv",
            "check_tolerance": args.tol_prevention,
            "check_note": ("smart_xgb prévention : F1 recalculé sur le CSV (6 décimales) à -4.4e-6 du JSON, "
                           "2 valeurs OOF arrondies exactement au seuil 0.185 ; tolérance 1e-5 acceptée"),
            "excluded_components": pfailures,
            "trials": [public(t) for t in ptrials],
        },
        "delta_full_minus_prevention": {"f1": metrics["f1"] - pmetrics["f1"], "auc": metrics["auc"] - pmetrics["auc"]},
    }
    res_path = ROOT / "results" / f"{args.name}_{stamp}.json"
    res_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("CV :", {k: round(v, 4) for k, v in metrics.items()})
    print("Prévention :", {k: round(v, 4) for k, v in pmetrics.items()}, "|", pchosen["label"])
    print(f"Soumission : {sub_path}\nRésultats : {res_path}  ({result['runtime_s']} s)")


if __name__ == "__main__":
    main()
