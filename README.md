# SDD Hackathon 2026 – Heart Risk

Compétition Kaggle : https://www.kaggle.com/competitions/sdd-hackaton-2026-heart-risk

## Récupérer les données

Les CSV ne sont pas versionnés : ils sont trop lourds pour GitHub (`train.csv` fait environ 210 Mo).
Chacun les télécharge en local.

| Fichier                 | Taille  |
|-------------------------|---------|
| `train.csv`             | ~211 Mo |
| `extra.csv`             | ~137 Mo |
| `test.csv`              | ~70 Mo  |
| `sample_submission.csv` | ~1 Mo   |

### Option 1 : avec le script (recommandé)

1. **Accepter le règlement** sur la page de la compétition (bouton *Join Competition*).
   Sinon, le téléchargement est refusé.
2. **Créer un token API** : https://www.kaggle.com/settings, section *API*, puis *Create New Token*.
   Ça télécharge un fichier `kaggle.json`.
3. **Placer le token** :
   - Windows : `C:\Users\<toi>\.kaggle\kaggle.json`
   - Linux/Mac : `~/.kaggle/kaggle.json`

   Ne le commit jamais : il est déjà dans le `.gitignore`.
4. **Installer et lancer** :
   ```bash
   pip install kagglehub
   python download_data.py
   ```

Les CSV arrivent à la racine du repo.

### Option 2 : à la main

Sur la page de la compétition, onglet *Data*, clique sur *Download All*.
Dézippe ensuite les CSV à la racine du repo.

## Structure attendue

```
SDD-Hackathon/
├── train.csv              (ignoré par git)
├── extra.csv              (ignoré par git)
├── test.csv               (ignoré par git)
├── sample_submission.csv  (ignoré par git)
├── download_data.py
└── README.md
```

## Entraîner un modèle

```bash
pip install lightgbm scikit-learn pandas
python train.py
```

Le script écrit la soumission dans `submissions/` et le score de validation croisée dans `results/`.
Pour soumettre sur Kaggle :

```bash
kaggle competitions submit -c sdd-hackaton-2026-heart-risk -f submissions/<fichier>.csv -m "message"
```

## Expériences et boucle d'autoresearch

- `experiments/` : un script par idée (copie du meilleur script du moment + une seule modification), `EXPERIMENTS.md` : journal de toutes les expériences (F1 CV, AUC, score Kaggle), `results/` : un JSON par exécution + `state.json` (meilleurs scores).
- `features/selected_columns.json` : liste de 50 colonnes (sur 324) qui conserve l'AUC du modèle complet (0,8542 contre 0,8550), avec la justification colonne par colonne (null importance, courbe top-K, redondance par concept).
- Les entraînements tournent sur Modal (`modal_run.py`, conteneur 16 cœurs ou GPU A10G) :
  ```bash
  pip install modal && modal setup
  modal volume create sdd-heart-risk-data
  modal volume put sdd-heart-risk-data data/train_extra.parquet /train_extra.parquet   # cache créé par train.py
  modal volume put sdd-heart-risk-data data/test.parquet /test.parquet
  PYTHONUTF8=1 MSYS_NO_PATHCONV=1 python -m modal run modal_run.py --script experiments/x.py --name x --threads 16 [--gpu]
  ```
  Un garde-fou (`modal_budget.json`) refuse tout lancement au-delà du budget estimé.

## Reproduire le record (Kaggle 0,43062)

Blend de trois modèles entraînés avec la même validation croisée (5 folds, seed 42), poids choisis sur les prédictions out-of-fold :
`0,55 × LightGBM baggé (experiments/x_lgbbag_proba.py) + 0,20 × FT-Transformer (experiments/dl_ft_transformer.py, GPU) + 0,25 × ResNet tabulaire (experiments/dl_resnet.py, GPU)`, seuil 0,22.
Lancer ces trois scripts sur Modal (ils écrivent leurs probabilités OOF/test dans `results/`), puis `python experiments/ml_stack.py` en local.
