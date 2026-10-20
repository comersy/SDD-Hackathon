"""Exécute un script d'expérience sur Modal (CPU 16 cœurs / 32 Go, ou GPU) et rapatrie ses sorties.

Usage :
    python -W ignore -m modal run modal_run.py --script experiments/x.py --name x [--threads 16] [--gpu]

Le script tourne dans un conteneur avec les parquet de data/ (Volume Modal `sdd-heart-risk-data`,
à remplir une fois avec `modal volume put`). Ses fichiers results/*.json et submissions/*.csv sont
recopiés en local aux mêmes chemins relatifs, comme s'il avait tourné ici.
Avec --gpu : catboost task_type="GPU", xgboost device="cuda", torch cuda disponibles.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = "/root/sdd"
OUTPUT_DIRS = ("results", "submissions")

# --- Garde-fou budget (estimation locale, tarifs Modal au 2026-10-20 : CPU 0.0000131 $/cœur/s,
# mémoire 0.00000222 $/GiB/s, A10 0.000306 $/s). Le plafond est dans modal_budget.json
# ({"cap_usd": ...}) ; chaque exécution ajoute son coût estimé dans logs/modal_usage.jsonl.
BUDGET_FILE = LOCAL_ROOT / "modal_budget.json"
USAGE_FILE = LOCAL_ROOT / "logs" / "modal_usage.jsonl"
RATE_PER_S = {"cpu": 16 * 0.0000131 + 32 * 0.00000222, "gpu": 0.000306 + 8 * 0.0000131 + 32 * 0.00000222}


def usage_total():
    if not USAGE_FILE.exists():
        return 0.0
    total = 0.0
    for line in USAGE_FILE.read_text(encoding="utf-8").splitlines():
        try:
            total += json.loads(line).get("usd", 0.0)
        except ValueError:
            continue
    return total


def budget_cap():
    if not BUDGET_FILE.exists():
        return None
    return float(json.loads(BUDGET_FILE.read_text(encoding="utf-8")).get("cap_usd", 0))


def record_usage(name, kind, seconds):
    USAGE_FILE.parent.mkdir(exist_ok=True)
    usd = round(seconds * RATE_PER_S[kind], 4)
    with USAGE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"date": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name, "kind": kind,
                            "seconds": round(seconds), "usd": usd}) + "\n")
    return usd

app = modal.App("sdd-heart-risk")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "pandas", "pyarrow", "scikit-learn", "lightgbm", "catboost", "xgboost", "optuna", "torch", "psutil",
    # Modèles tabulaires deep learning
    "rtdl_revisiting_models", "pytorch-tabnet", "tab-transformer-pytorch", "einops",
)
data_volume = modal.Volume.from_name("sdd-heart-risk-data", create_if_missing=True)


def _run(script, name, files, threads, extra_args):
    root = Path(REMOTE_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.csv").touch()  # marqueur pour find_root() ; les données viennent du cache parquet
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    before = {p for d in OUTPUT_DIRS for p in (root / d).glob("*")}

    cmd = [sys.executable, "-u", script, "--name", name, "--threads", str(threads), *extra_args]
    print("$", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=root).returncode

    outputs = {}
    for d in OUTPUT_DIRS:
        for p in sorted((root / d).glob("*")):
            if p not in before and p.is_file():
                outputs[f"{d}/{p.name}"] = p.read_text(encoding="utf-8")
    return {"returncode": code, "files": outputs}


@app.function(image=image, volumes={f"{REMOTE_ROOT}/data": data_volume}, cpu=16, memory=32768, timeout=3600)
def run_cpu(script, name, files, threads, extra_args):
    return _run(script, name, files, threads, extra_args)


@app.function(image=image, volumes={f"{REMOTE_ROOT}/data": data_volume}, cpu=8, memory=32768, gpu="A10G",
              timeout=3600)
def run_gpu(script, name, files, threads, extra_args):
    return _run(script, name, files, threads, extra_args)


@app.local_entrypoint()
def main(script: str, name: str = "", threads: int = 16, gpu: bool = False, extra: str = ""):
    """script : chemin relatif au dépôt (ex. experiments/x.py) ; extra : arguments supplémentaires."""
    script = Path(script).as_posix()
    name = name or Path(script).stem
    # Le script, train.py et toutes les expériences (un script peut importer une base précédente)
    files = {p.relative_to(LOCAL_ROOT).as_posix(): p.read_text(encoding="utf-8")
             for p in [LOCAL_ROOT / "train.py", *(LOCAL_ROOT / "experiments").glob("*.py"), LOCAL_ROOT / script]}
    cap, spent = budget_cap(), usage_total()
    if cap is not None and spent >= cap:
        print(f"Budget Modal atteint ({spent:.2f} $ estimés sur un plafond de {cap:.2f} $) : exécution refusée. "
              f"Augmente cap_usd dans {BUDGET_FILE.name} si tu veux continuer.")
        sys.exit(2)
    fn = run_gpu if gpu else run_cpu
    t0 = time.time()
    try:
        out = fn.remote(script, name, files, threads, extra.split())
    finally:
        usd = record_usage(name, "gpu" if gpu else "cpu", time.time() - t0)
        print(f"Coût estimé : {usd:.2f} $ (cumul {usage_total():.2f} $" + (f" / plafond {cap:.2f} $)" if cap is not None else ")"))
    for rel, content in out["files"].items():
        path = LOCAL_ROOT / rel
        path.parent.mkdir(exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print("Rapatrié :", rel)
    if out["returncode"] != 0:
        print(f"Échec distant (code {out['returncode']})")
        sys.exit(out["returncode"])
    print("Terminé.")
