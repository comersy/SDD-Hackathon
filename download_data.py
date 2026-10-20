"""Télécharge les données de la compétition Kaggle à la racine du repo."""
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parent

path = kagglehub.competition_download("sdd-hackaton-2026-heart-risk", output_dir=str(ROOT))
print("Données téléchargées dans :", path)
