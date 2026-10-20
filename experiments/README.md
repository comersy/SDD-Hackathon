# Expériences

Un script par idée, autonome (copie du meilleur script du moment + une seule modification).
Chaque script suit le protocole d'évaluation décrit en tête de `train.py`, pour que les F1 soient comparables.

```bash
python experiments/<nom>.py --name <nom> --threads 4
```

Résultats dans `results/<nom>_<horodatage>.json`, soumission dans `submissions/`.
Le journal de toutes les expériences est dans `EXPERIMENTS.md`, l'état courant dans `results/state.json`.
