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
