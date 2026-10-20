# Journal des expériences

Métrique : F1 sur la classe positive. CV = validation croisée 5 folds sur train + extra (370 132 lignes),
protocole décrit en tête de `train.py`. Kaggle = score public, soumis seulement quand le F1 CV bat le record.

| Date | Nom | Idée | F1 CV | Précision | Rappel | AUC | Seuil | Kaggle | Statut |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-20 | baseline_lgbm | Colonnes triées par règles, LightGBM brut, seuil optimisé pour le F1 | 0.4211 | 0.350 | 0.529 | 0.854 | 0.215 | 0.43024 | record (PR #1) |
| 2026-10-20 | r1_clean_codes | Nettoyage des codes de non-réponse BRFSS (7/9 → NaN, 88 → 0, unités) | 0.4221 | 0.364 | 0.502 | 0.854 | 0.230 | — | pas mieux |
| 2026-10-20 | r1_lgbm_tuned | Réglage des hyperparamètres LightGBM (recherche sur le fold 0, puis CV complète) | 0.4208 | 0.340 | 0.551 | 0.853 | 0.205 | — | pas mieux |
| 2026-10-20 | r1_catboost | CatBoost avec variables catégorielles déclarées, à la place de LightGBM | — | — | — | — | — | — | échec : Error: agent({schema}): subagent completed without calling StructuredOutput (after in-conversation nudge) |
| 2026-10-20 | x_drop_nan95 | r1_clean_codes + retrait des 69 colonnes vides à plus de 95 % (214 colonnes gardées), lancé à la main sur Modal | 0.4225 | 0.342 | 0.553 | 0.854 | 0.205 | — | léger plus (+0.0005), candidat base |
| 2026-10-20 | x_drop_nan50 | r1_clean_codes + retrait des 168 colonnes vides à plus de 50 % (115 colonnes gardées), lancé à la main sur Modal | 0.4208 | 0.344 | 0.542 | 0.853 | 0.210 | — | pas mieux (trop agressif) |
| 2026-10-20 | r2_lgbm_catboost_blend | Blend LightGBM + CatBoost GPU (moyenne des probabilités OOF et test, même script) | 0.4244 | 0.349 | 0.542 | 0.855 | 0.210 | — | pas mieux |
| 2026-10-20 | r2_catboost_gpu | CatBoost GPU avec catégorielles déclarées, sur les codes nettoyés (reprise de r1_catboost jamais évaluée) | 0.4222 | 0.343 | 0.549 | 0.854 | 0.205 | — | pas mieux |
| 2026-10-20 | r2_derived_features | Features dérivées : nombre de maladies chroniques, compte de NaN par ligne, âge×tabac, BMI recalculé | 0.4217 | 0.341 | 0.551 | 0.854 | 0.205 | — | pas mieux |
| 2026-10-20 | r2_lgbm_seed_bagging | Seed averaging LightGBM : 5 graines par fold, moyenne des probabilités OOF et test | 0.4247 | 0.348 | 0.545 | 0.855 | 0.210 | 0.43024 | meilleur CV, mais Kaggle 0.43024 ≤ 0.43024 |
| 2026-10-20 | r2_mlp_embeddings_gpu | MLP PyTorch GPU avec embeddings des catégorielles (composant diversifié pour blend futur) | 0.4177 | 0.332 | 0.562 | 0.850 | 0.220 | — | pas mieux |
| 2026-10-20 | r3_lgbbag_catboost_blend | Blend du LightGBM seed-bagged (5 graines) avec CatBoost GPU, réentraînés dans les mêmes folds | 0.4247 | 0.349 | 0.543 | 0.855 | 0.210 | — | pas mieux |
| 2026-10-20 | r3_lgbbag_mlp_blend | Blend du LightGBM seed-bagged avec le MLP PyTorch à embeddings (2 graines), famille de modèles différente | 0.4249 | 0.339 | 0.568 | 0.855 | 0.210 | — | pas mieux |
| 2026-10-20 | r3_lgb_nominal_cats | Colonnes nominales déclarées catégorielles pour LightGBM (EMPLOY1, _STATE, MARITAL, PRIMINSR, race, assurance…) sur le seed bagging | 0.4242 | 0.348 | 0.543 | 0.855 | 0.210 | — | pas mieux |
| 2026-10-20 | r3_lgbbag_low_lr | Réglage en CV complète du learning rate : 0.02 au lieu de 0.05 avec le seed bagging (early stopping 200) | 0.4245 | 0.361 | 0.515 | 0.855 | 0.225 | — | pas mieux |
| 2026-10-20 | r3_lgbbag_xgb_blend | Blend du LightGBM seed-bagged avec XGBoost GPU (hist, cuda), composant rapide jamais essayé | 0.4250 | 0.353 | 0.535 | 0.855 | 0.215 | — | nouvelle base (meilleur CV) |
| 2026-10-20 | fs_topk_curve | Courbe F1 et AUC en fonction du nombre de colonnes K (sélection top-K par fold) (150 colonnes) [sélection de colonnes] | 0.4229 | 0.347 | 0.543 | 0.854 | 0.210 | — | étude |
| 2026-10-20 | fs_null_importance | Null importance : colonnes dont l importance bat celle obtenue avec une cible mélangée (50 colonnes) [sélection de colonnes] | 0.4200 | 0.344 | 0.540 | 0.852 | 0.210 | — | étude |
| 2026-10-20 | fs_redundancy | Redondance : une colonne par concept (variables calculées BRFSS vs brutes, corrélations) (99 colonnes) [sélection de colonnes] | 0.4210 | 0.345 | 0.540 | 0.854 | 0.210 | — | étude |
| 2026-10-20 | fs_final | Liste finale de 50 colonnes + LightGBM baggé [sélection de colonnes] | 0.4219 | 0.350 | 0.531 | 0.854 | 0.215 | 0.42524 | 50 colonnes : Kaggle 0.42524 |
| 2026-10-20 | fs_final_extended | Variante étendue de la liste finale (62 colonnes) + LightGBM baggé [sélection de colonnes] | 0.4238 | — | — | 0.854 | 0.220 | 0.42970 | 62 colonnes : Kaggle 0.42970, à -0.0005 du record |
| 2026-10-20 | r4_lgbbag_feature_fraction_03 | feature_fraction 0.3 (au lieu de 0.5) pour le LightGBM seed-bagged du blend LGB+XGB, réglage en CV complète | 0.4253 | 0.353 | 0.535 | 0.855 | 0.215 | 0.43025 | RECORD Kaggle 0.43025 (+0.00001 : bruit) — PR après le stacking |
| 2026-10-20 | r4_weight_cols_as_features | Réintégrer les 9 colonnes de pondération BRFSS (_STSTR, _LLCPWT, _RAWRAKE…) comme variables numériques | 0.4252 | 0.348 | 0.546 | 0.855 | 0.210 | — | pas mieux |
| 2026-10-20 | r4_four_model_blend | Blend à 4 composants dans le même script : LGB seed-bagged + XGB cuda + CatBoost GPU (ctr 1) + MLP embeddings (2 graines), poids ajustés sur la logloss OOF | 0.4247 | 0.339 | 0.568 | 0.856 | 0.205 | — | pas mieux |
| 2026-10-20 | r4_xgb_fullcv_tuning | Réglage en CV complète (5 folds, ~30 s par config sur A10G) du XGBoost cuda du blend, sélection sur l'AUC OOF | 0.4244 | 0.348 | 0.544 | 0.855 | 0.210 | — | pas mieux |
| 2026-10-20 | r4_lgbbag_xgb_drop_nan95 | Retrait des 75 colonnes vides à plus de 95 % (validé +0.0005 sur LightGBM seul) combiné au meilleur blend LGB-bag + XGB | 0.4247 | 0.344 | 0.555 | 0.855 | 0.205 | — | pas mieux |
| 2026-10-20 | dl_ft_transformer | FT-Transformer (rtdl) sur top-80 colonnes par fold [piste DL] | 0.4204 | 0.338 | 0.555 | 0.853 | 0.210 | — | pas mieux (seul) |
| 2026-10-20 | dl_ft_transformer_k40 | FT-Transformer (rtdl) sur top-40 colonnes par fold (sélection agressive) [piste DL] | 0.4181 | 0.342 | 0.536 | 0.852 | 0.215 | — | pas mieux (seul) |
| 2026-10-20 | dl_resnet | ResNet tabulaire (rtdl) avec embeddings, top-80 [piste DL] | 0.4188 | 0.335 | 0.557 | 0.852 | 0.210 | — | pas mieux (seul) |
| 2026-10-20 | dl_tabnet | TabNet (pytorch-tabnet) avec masques d attention, top-80 [piste DL] | 0.4139 | 0.337 | 0.537 | 0.847 | 0.210 | — | pas mieux (seul) |
| 2026-10-20 | dl_tabtransformer | TabTransformer (tab-transformer-pytorch), top-80 [piste DL] | 0.4191 | 0.345 | 0.534 | 0.851 | 0.215 | — | pas mieux (seul) |
| 2026-10-20 | dl_base | Base DL : top-80 colonnes par fold + MLP GPU (embeddings + quantiles) | 0.4181 | 0.329 | 0.573 | 0.851 | 0.215 | — | base de la piste DL |
| 2026-10-20 | dl_blend | Blend LightGBM baggé + modèles DL (Retenu : 0.8*lgbm_bagged + 0.2*dl_ft_transformer (proba), F1 0.425141 (LGBM seul 0.424674, +0.00047 ; test_positive_rate) | 0.4251 | 0.347 | 0.549 | 0.855 | 0.210 | 0.43050 | RECORD Kaggle 0.43050 — PR après le stacking |
| 2026-10-20 | ml_random_forest | Random Forest (sklearn) sur toutes les colonnes nettoyées [ML classique] | 0.4100 | 0.327 | 0.549 | 0.845 | 0.205 | — | pas mieux (seul) |
| 2026-10-20 | ml_extra_trees | Extra-Trees (sklearn) sur toutes les colonnes nettoyées [ML classique] | 0.4075 | 0.332 | 0.528 | 0.841 | 0.220 | — | pas mieux (seul) |
| 2026-10-20 | ml_logreg_onehot | Régression logistique one-hot (odds ratios interprétables), top-120 [ML classique] | 0.4179 | 0.336 | 0.553 | 0.851 | 0.190 | — | pas mieux (seul) |
| 2026-10-20 | ml_ebm | Explainable Boosting Machine (interpret), top-60 [ML classique] | 0.4173 | 0.342 | 0.535 | 0.852 | 0.200 | — | pas mieux (seul) |
| 2026-10-20 | ml_svm_nystroem | SVM à noyau RBF approché (Nystroem) + SVM linéaire calibré, top-80 [ML classique] | 0.4202 | 0.351 | 0.524 | 0.849 | 0.170 | — | pas mieux (seul) |
| 2026-10-20 | ml_stack | Stacking / blends de tous les modèles disponibles (Composants (12 retenus, F1 OOF recalculé = JSON à 1e-6) : x_lgbbag_proba 0.4247 (réf.), fs_final 0.4219, ml_svm_nystroem) | 0.4257 | 0.354 | 0.533 | 0.856 | 0.220 | 0.43062 | RECORD Kaggle 0.43062 — PR à valider |
| 2026-10-20 | mlp_big_50 | Gros MLP (1024-512-256) sur les 50 colonnes sélectionnées, 3 graines [gros MLP] | 0.4206 | 0.346 | 0.536 | 0.852 | 0.220 | — | pas soumis (1 graine : 0.4184) |
| 2026-10-20 | mlp_big_62 | Gros MLP (1024-512-256) sur les 62 colonnes (variante étendue), 3 graines [gros MLP] | 0.4195 | 0.350 | 0.524 | 0.852 | 0.225 | — | pas soumis (1 graine : 0.4172) |
| 2026-10-20 | mlp_huge_50 | MLP énorme (2048-1024-512-256) sur les 50 colonnes, dropout 0.45, 3 graines [gros MLP] | 0.4212 | 0.337 | 0.562 | 0.852 | 0.205 | 0.429 | Kaggle 0.429 (1 graine : 0.4184) |
| 2026-10-20 | mlp_pca_50 | ACP (64 composantes) sur les 50 colonnes encodées, puis MLP 1024-512-256, 3 graines [gros MLP] | 0.4108 | 0.328 | 0.550 | 0.848 | 0.210 | — | pas soumis (1 graine : 0.4084) |
| 2026-10-20 | ml_stack_v2 | Re-stacking de tous les modèles avec les gros MLP | 0.4259 | 0.344 | 0.559 | 0.855 | 0.205 | 0.43046 | Kaggle 0.43046 |
