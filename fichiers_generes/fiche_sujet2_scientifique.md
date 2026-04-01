
# Fiche Scientifique - Sujet 2

## Auteurs
Equipe Data Science - Projet Mental Health

## Contexte
La depression est un enjeu majeur de sante publique. Ce jeu de donnees permet d'analyser les liens entre depression, autres troubles mentaux et troubles lies aux substances.

## Objectif
Identifier les facteurs les plus explicatifs de la prevalence de la depression et comparer plusieurs modeles de regression pour une prediction robuste.

## Donnees utilisees
- Dataset : prevalence-by-mental-and-substance-use-disorder.csv
- Nombre d'observations : 6150
- Variables principales : Schizophrenie, Bipolarite, Troubles alim., Anxiete, Usage drogues, Usage alcool

## Methodologie
Modeles testes :
- Regression lineaire
- Ridge
- RandomForest

Methode de validation :
Split temporel: entrainement <= 2013, test > 2013.

Packages utilises :
pandas, numpy, matplotlib, scikit-learn

## Resultats

| Modele | RMSE | MAE | R2 |
|------|------|------|------|
| RandomForest | 0.235 | 0.143 | 0.934 |
| Lineaire | 0.743 | 0.570 | 0.344 |
| Ridge | 0.752 | 0.582 | 0.328 |

Meilleur modele : **RandomForest**

## Interpretation

Le modele RandomForest obtient la meilleure performance (RMSE=0.235). Le facteur le plus associe a la depression est Bipolarite avec une correlation de r=0.19.

## Limites

Les modeles decrivent des associations statistiques mais ne prouvent pas la causalite. L'absence de variables socio-economiques et sanitaires limite l'explication contextuelle.

## Perspectives

Integrer des variables externes (PIB, acces aux soins), tester des modeles de boosting et comparer les performances par zone geographique.

---

Fiche generee automatiquement le 10/03/2026
