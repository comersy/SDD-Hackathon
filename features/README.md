# Colonnes sélectionnées (liste finale)

**Méthode** : consensus null importance (33 colonnes battant la cible permutée dans >= 4 folds sur 5) ramené à une colonne par concept via la liste curée `fs_redundancy` (-> 27 colonnes), complété par les colonnes stables du top-K au coude de la courbe `fs_topk_curve` (K = 60, F1 >= 99 % du maximum ; +23 colonnes, PRIMINSR préférée à sa dérivée binaire _HCVU652). Aucune paire redondante (rho/V >= 0.9), aucune colonne administrative.

**Résultat** (LightGBM baggé 5 graines par fold, `experiments/fs_final.py`, CV 5 folds OOF) : **50 colonnes -> F1 0.4219, AUC 0.8542** (seuil 0.215), contre F1 0.4247 / AUC 0.8550 avec 283 colonnes (`x_lgbbag_proba`). Variante étendue (62 colonnes, stables du top-100) : F1 0.4238 / AUC 0.8544.
Attention : la liste vient d'importances calculées dans chaque fold puis agrégées, donc ce score CV comporte un léger optimisme de sélection (à dire dans le notebook).

**Utilisation** : `cols = json.load(open("features/selected_columns.json"))["columns"]` puis `X = train[cols]` (après le nettoyage des codes de non-réponse de `train.py`) ; `rationale` donne, par colonne, le concept, le nombre de folds null importance, le rang/gain du top-K et l'origine dans la règle ; `sources` liste les study JSON utilisés ; `extended_variant` la liste de repli si l'on veut quelques colonnes de plus.
