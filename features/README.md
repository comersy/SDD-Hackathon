# Colonnes sélectionnées (liste finale)

**Méthode** : consensus null importance (33 colonnes battant la cible permutée dans >= 4 folds sur 5) ramené à une colonne par concept via la liste curée `fs_redundancy` (-> 27 colonnes), complété par les colonnes stables du top-K au coude de la courbe `fs_topk_curve` (K = 60, F1 >= 99 % du maximum ; +23 colonnes, PRIMINSR préférée à sa dérivée binaire _HCVU652). Aucune paire redondante (rho/V >= 0.9), aucune colonne administrative.

**Résultat** (LightGBM baggé 5 graines par fold, `experiments/fs_final.py`, CV 5 folds OOF) : **50 colonnes -> F1 0.4219, AUC 0.8542** (seuil 0.215), contre F1 0.4247 / AUC 0.8550 avec 283 colonnes (`x_lgbbag_proba`). Variante étendue (62 colonnes, stables du top-100) : F1 0.4238 / AUC 0.8544.
Attention : la liste vient d'importances calculées dans chaque fold puis agrégées, donc ce score CV comporte un léger optimisme de sélection (à dire dans le notebook).

**Utilisation** : `cols = json.load(open("features/selected_columns.json"))["columns"]` puis `X = train[cols]` (après le nettoyage des codes de non-réponse de `train.py`) ; `rationale` donne, par colonne, le concept, le nombre de folds null importance, le rang/gain du top-K et l'origine dans la règle ; `sources` liste les study JSON utilisés ; `extended_variant` la liste de repli si l'on veut quelques colonnes de plus.

## Listes minimales (RFE)

**Courbe** (`experiments/rfe_curve.py`, LightGBM une graine, élimination récursive 62 -> 10 colonnes dans chaque fold puis consensus) : F1/AUC CV = 50:0.4216/0.8532, 45:0.4204/0.8530, 40:0.4200/0.8528, 35:0.4195/0.8522, 30:0.4171/0.8515, 25:0.4171/0.8505, 20:0.4158/0.8500, 15:0.4095/0.8474, 10:0.4037/0.8439. La perte reste faible jusqu'à 30 colonnes, puis décroche sous 20.

**Listes disponibles** : `features/selected_columns_min.json` -> `lists["15"|"20"|"25"|"30"|"35"|"40"]` (ordre = importance décroissante), `curve` pour la courbe, `evaluations` pour les scores.

**Scores** (LightGBM baggé 5 graines, CV 5 folds OOF / Kaggle) : 40 colonnes F1 0.4215 / AUC 0.8534 / Kaggle 0.42639 ; 30 colonnes F1 0.4200 / AUC 0.8522 / Kaggle 0.4282 ; 20 colonnes F1 0.4169 / AUC 0.8506 / Kaggle 0.42047 ; gros MLP 3 graines sur 30 colonnes F1 0.4180 / AUC 0.8511 / Kaggle 0.4262. À comparer à 50 colonnes (F1 0.4219, Kaggle ~0.43) : 30 colonnes coûtent ~0.002 de F1 Kaggle, 20 colonnes ~0.01.

**Utilisation** : `cols = json.load(open("features/selected_columns_min.json"))["lists"]["30"]` puis `X = train[cols]` (même nettoyage des codes de non-réponse que `train.py`). Même réserve que ci-dessus : sélection faite dans chaque fold puis agrégée, donc léger optimisme de sélection sur le score CV.
