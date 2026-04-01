# Fiche de révision : Analyse2 – Corrélation et Prédiction de la Dépression

## 1. Objectif de l’analyse
- Étudier la corrélation entre différents troubles mentaux.
- Prédire la prévalence de la dépression à partir d’autres troubles et de l’année.

## 2. Données utilisées
- Source : Fichier CSV sur la prévalence de troubles mentaux et d’addictions.
- Variables :
  - Année
  - Prévalence de : dépression, schizophrénie, trouble bipolaire, troubles alimentaires, anxiété, usage de drogues, usage d’alcool (tous en %)

## 3. Étapes principales
### a) Corrélation entre troubles
- Calcul de la matrice de corrélation entre les troubles.
- Visualisation par heatmap (matrice_correlation_troubles.png).

### b) Problème prédictif
- Cible : prévalence de la dépression.
- Variables explicatives : autres troubles + année.
- Séparation temporelle : données jusqu’à 2013 pour l’entraînement, après 2013 pour le test.

### c) Modélisation
- Modèles testés :
  - Régression linéaire
  - Ridge
  - Random Forest
- Évaluation : MAE, RMSE, R2 sur le jeu de test.
- Sélection du meilleur modèle selon le RMSE.

### d) Amélioration itérative
- Tuning des hyperparamètres du Random Forest.
- Comparaison des performances pour différentes configurations.

### e) Analyse des résultats
- Affichage des erreurs, des résidus, et de l’importance des variables (importance_variables_random_forest.png).

## 4. Concepts clés
- **Corrélation** : mesure la force du lien entre deux variables.
- **Régression** : prédire une valeur numérique à partir de plusieurs variables.
- **Séparation temporelle** : évite la fuite d’information du futur vers le passé.
- **Tuning** : ajuster les paramètres d’un modèle pour améliorer ses performances.
- **Importance des variables** : indique quelles variables sont les plus utiles pour la prédiction.

## 5. Métriques utilisées
- **MAE** (Mean Absolute Error) : erreur moyenne absolue.
- **RMSE** (Root Mean Squared Error) : racine de l’erreur quadratique moyenne.
- **R2** : proportion de la variance expliquée par le modèle.

## 6. Outils Python mobilisés
- pandas : manipulation de données
- matplotlib : visualisation
- numpy : calculs numériques
- scikit-learn : modèles prédictifs et métriques

## 7. Schéma du workflow

Entrée (données CSV) → Nettoyage → Corrélation → Sélection des variables → Split train/test → Modélisation → Évaluation → Tuning → Analyse des résultats

---

# Concepts généraux (Machine Learning supervisé)

- On apprend à partir de données d’entrée et de cibles connues (étiquettes).
- Le but est de généraliser à de nouveaux cas.
- Importance de séparer entraînement et test pour évaluer la vraie performance.
- On compare plusieurs modèles et on choisit le meilleur selon une métrique adaptée.

---

# Pour aller plus loin
- Tester d’autres modèles (ex : XGBoost, SVM)
- Ajouter d’autres variables explicatives (ex : facteurs socio-économiques)
- Utiliser la validation croisée pour une évaluation plus robuste

---

# Fiche réalisée le 01/04/2026
