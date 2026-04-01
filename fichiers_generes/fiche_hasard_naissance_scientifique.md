
# Fiche Scientifique - Hasard de la naissance

## Auteurs
Equipe Data Science - Projet Mental Health

## Contexte
Selon le pays, le continent et le sexe, l'exposition moyenne aux troubles mentaux n'est pas identique. La question est donc de mesurer dans quelle proportion les conditions de naissance structurent ces ecarts.

## Objectif
Comparer les differences geographiques et de sexe, puis tester dans quelle mesure des variables de naissance (continent, sexe, annee) permettent de predire la prevalence des troubles mentaux.

## Donnees utilisees
- Dataset principal : share-with-mental-and-substance-disorders.csv
- Dataset complementaire : share-with-mental-or-substance-disorders-by-sex.csv
- Nombre d'observations : 390
- Variables principales : Annee, Continent, Sexe

## Methodologie
Modeles testes :
- Regression lineaire
- Ridge
- RandomForest

Methode de validation :
Split aleatoire: 75% entrainement, 25% test (dataset disponible sur un millesime exploitable).

Packages utilises :
pandas, numpy, matplotlib, scikit-learn

## Resultats

| Modele | RMSE | MAE | R2 |
|------|------|------|------|
| Ridge | 1.957 | 1.407 | -0.059 |
| Lineaire | 1.961 | 1.406 | -0.063 |
| RandomForest | 1.975 | 1.399 | -0.079 |

Meilleur modele : **Ridge**

## Interpretation

Le continent le plus eleve en 2015 est Oceania (17.08%). L'ecart entre continents atteint 4.57 points et le meilleur modele base uniquement sur annee, continent et sexe atteint R2=-0.059.

## Limites

Les donnees sont aggregees au niveau pays et ne decrivent pas des trajectoires individuelles. Le lieu de naissance n'explique donc pas tout et ne doit pas etre interprete comme une causalite stricte.

## Perspectives

Ajouter revenu, acces aux soins, conflits ou education pour distinguer ce qui releve de la geographie et ce qui releve du contexte socio-economique.

---

Fiche generee automatiquement le 10/03/2026
