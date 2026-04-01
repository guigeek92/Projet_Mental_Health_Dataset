

# Inégalités de prévalence des troubles mentaux selon les conditions de naissance

## Auteurs
Projet Data Science - Majeure Energie Environnement

## Source des données
Dataset Kaggle : Global Mental Health Disorder Statistics

## Contexte

Les études épidémiologiques montrent que la prévalence des troubles mentaux varie selon les régions du monde et le sexe.

Comprendre ces écarts permet d’analyser dans quelle mesure les conditions de naissance peuvent être associées à la santé mentale.

## Problématique

Dans quelle mesure les conditions de naissance (continent, sexe, année) sont-elles associées aux différences de prévalence des troubles mentaux ?

## Données utilisées

- Source : Kaggle
- Nombre d'observations : 390
- Variables : année, continent, sexe

## Méthodologie

Plusieurs modèles de machine learning ont été testés :

- Régression linéaire
- Ridge Regression
- Random Forest

Les performances sont évaluées à l'aide de :

- RMSE
- MAE
- R²

## Résultats

             modele       mae      rmse        r2
1             Ridge  1.406872  1.956963 -0.059373
0  LinearRegression  1.406107  1.960660 -0.063378
2      RandomForest  1.399379  1.974430 -0.078367

Meilleur modèle : Ridge

R² : -0.059

## Importance des variables

None

## Interprétation

Les résultats montrent que les variables associées aux conditions de naissance expliquent une partie des variations de prévalence observées entre régions et sexes.

Cependant ces facteurs ne suffisent pas à expliquer entièrement les différences observées.

## Limites

Les données sont agrégées au niveau des pays et ne représentent pas des trajectoires individuelles.

Les résultats décrivent donc des corrélations et ne permettent pas d'établir une causalité.

## Perspectives

Intégrer d'autres variables :

- revenu moyen
- accès aux soins
- niveau d’éducation
- facteurs socio-économiques

---

Fiche générée automatiquement le 10/03/2026

