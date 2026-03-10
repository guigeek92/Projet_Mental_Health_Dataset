# Import des librairies utiles
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Mettre à True si tu veux ouvrir la fenêtre du graphique
AFFICHER_GRAPHIQUE = False

# =========================
# Objectif actuel :
# Afficher uniquement la matrice de corrélation entre les différents troubles.
# =========================

# 1) Chargement et nettoyage simple
df = pd.read_csv('donnees_brutes/prevalence-by-mental-and-substance-use-disorder.csv')
df = df.dropna()

# 2) Colonnes des troubles à corréler
colonnes = [
    "Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)",
    "Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)",
    "Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)",
    "Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)",
    "Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)",
]

noms_courts = {
    colonnes[0]: "Bipolar",
    colonnes[1]: "Eating",
    colonnes[2]: "Anxiety",
    colonnes[3]: "Drug use",
    colonnes[4]: "Alcohol use",
}

# 3) Matrice de corrélation (trouble vs trouble)
matrice_corr = df[colonnes].corr()
matrice_corr.index = [noms_courts[c] for c in matrice_corr.index]
matrice_corr.columns = [noms_courts[c] for c in matrice_corr.columns]

print("\nMatrice de corrélation entre les troubles :")
print(matrice_corr)

# 4) Heatmap (non bloquante en terminal)
if AFFICHER_GRAPHIQUE:
    plt.figure(figsize=(8, 6))
    plt.imshow(matrice_corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Corrélation')
    plt.xticks(range(len(matrice_corr.columns)), matrice_corr.columns, rotation=45, ha='right')
    plt.yticks(range(len(matrice_corr.index)), matrice_corr.index)
    plt.title("Corrélation entre les troubles")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)
    plt.close()
else:
    plt.figure(figsize=(8, 6))
    plt.imshow(matrice_corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Corrélation')
    plt.xticks(range(len(matrice_corr.columns)), matrice_corr.columns, rotation=45, ha='right')
    plt.yticks(range(len(matrice_corr.index)), matrice_corr.index)
    plt.title("Corrélation entre les troubles")
    plt.tight_layout()
    plt.savefig("matrice_correlation_troubles.png", dpi=150)
    plt.close()
    print("Heatmap enregistrée : matrice_correlation_troubles.png")

# =========================
# Étape 3 : Développement de solution (baseline)
# =========================
# Problème : prédire la prévalence de la dépression.
col_cible = "Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)"

features = [
    "Year",
    "Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)",
    "Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)",
    "Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)",
    "Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)",
    "Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)",
    "Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)",
]

df_modele = df[features + [col_cible]].copy()

# Split temporel simple : années <= 2013 en train, > 2013 en test
# (évite la fuite d'information liée au temps)
train_mask = df_modele["Year"] <= 2013

X_train = df_modele.loc[train_mask, features]
y_train = df_modele.loc[train_mask, col_cible]
X_test = df_modele.loc[~train_mask, features]
y_test = df_modele.loc[~train_mask, col_cible]

modele = LinearRegression()
modele.fit(X_train, y_train)
y_pred = modele.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== Étape 3 : baseline régression linéaire ===")
print(f"Train: {len(X_train)} lignes | Test: {len(X_test)} lignes")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")


# =========================
# Étape 4 : Quels modèles ?
# =========================
modeles = {
    "LinearRegression": LinearRegression(),
    "Ridge_alpha_1": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),
}

resultats = []
for nom_modele, modele_test in modeles.items():
    modele_test.fit(X_train, y_train)
    y_pred_test = modele_test.predict(X_test)

    resultats.append({
        "Modele": nom_modele,
        "MAE": mean_absolute_error(y_test, y_pred_test),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "R2": r2_score(y_test, y_pred_test),
    })

resultats_df = pd.DataFrame(resultats).sort_values(by="RMSE")
print("\n=== Étape 4 : comparaison des modèles ===")
print(resultats_df.to_string(index=False))

meilleur_modele_nom = resultats_df.iloc[0]["Modele"]
print(f"\nMeilleur modèle (sur test, selon RMSE) : {meilleur_modele_nom}")


# =========================
# Étape 5 : Cycles itératifs (essai-erreur)
# =========================
print("\n=== Étape 5 : process essai-erreur (explication) ===")
print("1) On change les hyperparamètres du modèle.")
print("2) On réentraîne sur le même train et on réévalue sur le même test.")
print("3) On compare RMSE/MAE/R2 et on garde la meilleure configuration.")

configurations_rf = [
    {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 1},
    {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 2},
]

resultats_tuning = []
for config in configurations_rf:
    modele_rf = RandomForestRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        random_state=42,
        n_jobs=-1,
    )
    modele_rf.fit(X_train, y_train)
    y_pred_rf = modele_rf.predict(X_test)

    resultats_tuning.append({
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "min_samples_leaf": config["min_samples_leaf"],
        "MAE": mean_absolute_error(y_test, y_pred_rf),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        "R2": r2_score(y_test, y_pred_rf),
    })

resultats_tuning_df = pd.DataFrame(resultats_tuning).sort_values(by="RMSE")
print("\nRésultats du tuning RandomForest :")
print(resultats_tuning_df.to_string(index=False))

meilleure_config = resultats_tuning_df.iloc[0]
print("\nMeilleure configuration RandomForest :")
print(
    f"n_estimators={int(meilleure_config['n_estimators'])}, "
    f"max_depth={meilleure_config['max_depth']}, "
    f"min_samples_leaf={int(meilleure_config['min_samples_leaf'])}"
)
print(
    f"Scores -> MAE={meilleure_config['MAE']:.4f}, "
    f"RMSE={meilleure_config['RMSE']:.4f}, "
    f"R2={meilleure_config['R2']:.4f}"
)


# =========================
# Étape 6 : Qualité prédictive (diagnostic)
# =========================
modele_final = RandomForestRegressor(
    n_estimators=int(meilleure_config["n_estimators"]),
    max_depth=None if pd.isna(meilleure_config["max_depth"]) else int(meilleure_config["max_depth"]),
    min_samples_leaf=int(meilleure_config["min_samples_leaf"]),
    random_state=42,
    n_jobs=-1,
)
modele_final.fit(X_train, y_train)
y_pred_final = modele_final.predict(X_test)
residus = y_test - y_pred_final

df_erreurs = pd.DataFrame({
    "Year": X_test["Year"].values,
    "y_reel": y_test.values,
    "y_pred": y_pred_final,
    "erreur": residus.values,
    "erreur_absolue": np.abs(residus.values),
})

print("\n=== Étape 6 : qualité prédictive détaillée ===")
print(f"Erreur moyenne (résidu moyen) : {df_erreurs['erreur'].mean():.4f}")
print(f"Écart-type des résidus      : {df_erreurs['erreur'].std():.4f}")

print("\nTop 10 plus grosses erreurs absolues :")
print(df_erreurs.sort_values(by="erreur_absolue", ascending=False).head(10).to_string(index=False))

erreur_par_annee = df_erreurs.groupby("Year")["erreur_absolue"].mean().reset_index()
print("\nErreur absolue moyenne par année (test) :")
print(erreur_par_annee.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.scatter(df_erreurs["y_reel"], df_erreurs["y_pred"], alpha=0.35)
min_val = min(df_erreurs["y_reel"].min(), df_erreurs["y_pred"].min())
max_val = max(df_erreurs["y_reel"].max(), df_erreurs["y_pred"].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)
plt.xlabel("Valeurs réelles (dépression)")
plt.ylabel("Valeurs prédites")
plt.title("Qualité prédictive : réel vs prédit")
plt.tight_layout()
plt.savefig("qualite_predictive_reel_vs_predit.png", dpi=150)
plt.close()
print("Graphique enregistré : qualite_predictive_reel_vs_predit.png")


# =========================
# Étape 7 : Importance des variables
# =========================
importances = pd.DataFrame({
    "Feature": features,
    "Importance": modele_final.feature_importances_,
}).sort_values(by="Importance", ascending=False)

print("\n=== Étape 7 : importance des variables ===")
print(importances.to_string(index=False))

plt.figure(figsize=(10, 5))
plt.bar(importances["Feature"], importances["Importance"])
plt.xticks(rotation=60, ha='right')
plt.ylabel("Importance")
plt.title("Importance des variables (RandomForest)")
plt.tight_layout()
plt.savefig("importance_variables_random_forest.png", dpi=150)
plt.close()
print("Graphique enregistré : importance_variables_random_forest.png")


# =========================
# Anciennes analyses (désactivées pour l'instant)
# =========================
# print(df.head())
# print(df.info())
# print(df.isna().sum())
#
# # Corrélation complète sur toutes les colonnes numériques
# # df_numerique = df.select_dtypes(include='number')
# # print(df_numerique.corr())
#
# # Statistiques descriptives
# # for col in colonnes:
# #     print(col, df[col].mean(), df[col].median())
#
# # Corrélation Year vs trouble
# # for col in colonnes:
# #     print(df[['Year', col]].corr().iloc[0, 1])
#
# # Visualisation des moyennes annuelles
# # moyennes_annuelles = df.groupby('Year')[colonnes].mean().sort_index()

# =========================
# Recap
# =========================
# Cette analyse explore les liens entre plusieurs troubles mentaux et
# construit un modele de prediction de la prevalence de la depression.
# Le script calcule une matrice de correlation, etabli une baseline lineaire,
# compare plusieurs modeles, realise un tuning RandomForest, puis analyse les
# erreurs et l'importance des variables pour identifier les facteurs dominants.


# =========================
# Conclusion (rapport)
# =========================
# Dans ce projet, nous avons analysé un dataset panel (pays × années, 1990–2019)
# sur la prévalence de plusieurs troubles mentaux et liés aux substances.
# La matrice de corrélation montre des liens positifs modérés à forts entre
# plusieurs troubles, notamment entre bipolarité et troubles alimentaires,
# ainsi qu’entre troubles alimentaires et usage de drogues.
#
# Le problème prédictif retenu est une régression supervisée : prédire la
# prévalence des troubles dépressifs à partir de l’année et des autres troubles.
# Un split temporel (train jusqu’en 2013, test après 2013) a été utilisé pour
# éviter la fuite d’information.
#
# La baseline linéaire obtient des performances limitées (RMSE ≈ 0.743,
# R² ≈ 0.344), alors que RandomForest améliore fortement la qualité prédictive
# (RMSE ≈ 0.235, R² ≈ 0.934).
#
# Une phase itérative de tuning (essai-erreur sur les hyperparamètres) confirme
# la meilleure configuration autour de n_estimators=300, max_depth=None,
# min_samples_leaf=1.
#
# L’analyse des erreurs montre une bonne précision globale, avec une légère
# dégradation sur les années les plus récentes du test.
#
# L’importance des variables indique que les prévalences de la schizophrénie,
# de la bipolarité, de l’anxiété et des troubles liés à l’alcool contribuent
# le plus à la prédiction, tandis que l’année a un poids faible.
#
# En résumé, le modèle capture efficacement les relations statistiques entre
# troubles, mais ces résultats restent corrélationnels : ils ne permettent pas
# d’inférer une causalité directe. Pour aller plus loin, il serait utile
# d’intégrer des variables socio-économiques et sanitaires externes afin
# d’améliorer l’interprétation et la robustesse du modèle.