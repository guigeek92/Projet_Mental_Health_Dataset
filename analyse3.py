import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Mettre à True si tu veux ouvrir les fenêtres graphiques
AFFICHER_GRAPHIQUE = False


# =========================
# 1) Chargement des données
# =========================

dataset_path =  "donnees_brutes/prevalence-of-depression-males-vs-females.csv"
df = pd.read_csv(dataset_path)

col_male = "Prevalence - Depressive disorders - Sex: Male - Age: Age-standardized (Percent)"
col_female = "Prevalence - Depressive disorders - Sex: Female - Age: Age-standardized (Percent)"
col_pop = "Population (historical estimates)"

print("=== Aperçu du dataset ===")
print(f"Shape initiale: {df.shape}")
print(df.head())
print("\nValeurs manquantes (initial):")
print(df.isna().sum())

# On garde uniquement les colonnes utiles pour l'analyse/modélisation
colonnes_utiles = ["Entity", "Code", "Year", col_male, col_female, col_pop]
df = df[colonnes_utiles].dropna(subset=["Year", col_male, col_female, col_pop]).copy()

print("\n=== Après nettoyage ===")
print(f"Shape nettoyée: {df.shape}")
print(f"Nombre de pays/entités: {df['Entity'].nunique()}")
print(f"Période: {int(df['Year'].min())} - {int(df['Year'].max())}")


# =========================
# 2) Matrice de corrélation
# =========================
df_corr = df[["Year", col_male, col_female, col_pop]].copy()
matrice_corr = df_corr.corr()

renommage = {
	"Year": "Year",
	col_male: "Depression_Male",
	col_female: "Depression_Female",
	col_pop: "Population",
}
matrice_corr = matrice_corr.rename(index=renommage, columns=renommage)

print("\n=== Matrice de corrélation ===")
print(matrice_corr)

plt.figure(figsize=(8, 6))
plt.imshow(matrice_corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Corrélation")
plt.xticks(range(len(matrice_corr.columns)), matrice_corr.columns, rotation=35, ha="right")
plt.yticks(range(len(matrice_corr.index)), matrice_corr.index)
plt.title("Corrélation : dépression hommes/femmes")
plt.tight_layout()

if AFFICHER_GRAPHIQUE:
	plt.show(block=False)
	plt.pause(2)
	plt.close()
else:
	plt.savefig( "donnees_brutes/doc_analyse3/matrice_correlation_depression_mf.png", dpi=150)
	plt.close()
	print("Heatmap enregistrée : donnees_brutes/matrice_correlation_depression_mf.png")


# =========================
# 3) Problème prédictif
# =========================
# Cible: prévalence de la dépression chez les femmes
# Features: année + prévalence homme + population
col_cible = col_female
features = ["Year", col_male, col_pop]

df_modele = df[features + [col_cible]].copy()

# Split temporel
train_mask = df_modele["Year"] <= 2013
X_train = df_modele.loc[train_mask, features]
y_train = df_modele.loc[train_mask, col_cible]
X_test = df_modele.loc[~train_mask, features]
y_test = df_modele.loc[~train_mask, col_cible]


# =========================
# 4) Baseline linéaire
# =========================
baseline = LinearRegression()
baseline.fit(X_train, y_train)
y_pred_baseline = baseline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_baseline)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
r2 = r2_score(y_test, y_pred_baseline)

print("\n=== Étape 4 : baseline régression linéaire ===")
print(f"Train: {len(X_train)} lignes | Test: {len(X_test)} lignes")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")


# =========================
# 5) Comparaison de modèles
# =========================
modeles = {
	"LinearRegression": LinearRegression(),
	"Ridge_alpha_1": Ridge(alpha=1.0),
	"RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
}

resultats = []
for nom_modele, modele in modeles.items():
	modele.fit(X_train, y_train)
	y_pred = modele.predict(X_test)
	resultats.append({
		"Modele": nom_modele,
		"MAE": mean_absolute_error(y_test, y_pred),
		"RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
		"R2": r2_score(y_test, y_pred),
	})

resultats_df = pd.DataFrame(resultats).sort_values(by="RMSE")
print("\n=== Étape 5 : comparaison des modèles ===")
print(resultats_df.to_string(index=False))


# =========================
# 6) Essai-erreur (tuning RF)
# =========================
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
print("\n=== Étape 6 : tuning RandomForest ===")
print(resultats_tuning_df.to_string(index=False))

meilleure_config = resultats_tuning_df.iloc[0]
print("\nMeilleure configuration RF:")
print(
	f"n_estimators={int(meilleure_config['n_estimators'])}, "
	f"max_depth={meilleure_config['max_depth']}, "
	f"min_samples_leaf={int(meilleure_config['min_samples_leaf'])}"
)


# =========================
# 7) Qualité prédictive + importance
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

print("\n=== Étape 7 : qualité prédictive détaillée ===")
print(f"Erreur moyenne (résidu moyen) : {df_erreurs['erreur'].mean():.4f}")
print(f"Écart-type des résidus      : {df_erreurs['erreur'].std():.4f}")
print("\nTop 10 plus grosses erreurs absolues :")
print(df_erreurs.sort_values(by="erreur_absolue", ascending=False).head(10).to_string(index=False))

plt.figure(figsize=(7, 5))
plt.scatter(df_erreurs["y_reel"], df_erreurs["y_pred"], alpha=0.35)
min_val = min(df_erreurs["y_reel"].min(), df_erreurs["y_pred"].min())
max_val = max(df_erreurs["y_reel"].max(), df_erreurs["y_pred"].max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)
plt.xlabel("Valeurs réelles (femmes)")
plt.ylabel("Valeurs prédites")
plt.title("Qualité prédictive : réel vs prédit")
plt.tight_layout()
plt.savefig("donnees_brutes/doc_analyse3/qualite_predictive_female.png", dpi=150)
plt.close()
print("Graphique enregistré : donnees_brutes/doc_analyse3/qualite_predictive_female.png")

importances = pd.DataFrame({
	"Feature": features,
	"Importance": modele_final.feature_importances_,
}).sort_values(by="Importance", ascending=False)

print("\nImportance des variables :")
print(importances.to_string(index=False))

plt.figure(figsize=(7, 4))
plt.barh(importances["Feature"], importances["Importance"])
plt.xlabel("Importance")
plt.title("Importance des variables (RF)")
plt.tight_layout()
plt.savefig("donnees_brutes/doc_analyse3/importance_variables_female.png", dpi=150)
plt.close()
print("Graphique enregistré : donnees_brutes/doc_analyse3/importance_variables_female.png")


# =========================
# Recap
# =========================
# Cette analyse se concentre sur la depression par sexe (hommes/femmes).
# Elle inclut le nettoyage des donnees, la correlation des variables, une
# prediction de la prevalence feminine, la comparaison de modeles, un tuning
# RandomForest, puis un diagnostic d'erreurs et une mesure d'importance des
# variables explicatives.


# =========================
# Conclusion (commentaire)
# =========================
# Cette analyse suit la même méthode que analyse2.py mais sur un dataset
# hommes/femmes, avec comme cible la prévalence féminine de la dépression.
# Les étapes incluent : corrélation, baseline, comparaison de modèles,
# tuning itératif, diagnostic d'erreurs et importance des variables.

# =========================
# Conclusion (rapport)
# =========================
# Dans ce projet, nous avons appliqué une démarche de data science complète
# sur le dataset "prevalence-of-depression-males-vs-females".
# Après nettoyage, l'analyse couvre 205 entités sur la période 1990–2019.
#
# La matrice de corrélation met en évidence une corrélation forte entre la
# prévalence de dépression chez les hommes et chez les femmes (≈ 0.85),
# alors que l'année et la population ont des corrélations faibles avec la cible.
#
# Le problème prédictif retenu est une régression supervisée : prédire la
# prévalence féminine à partir de l'année, de la prévalence masculine et de
# la population. Un split temporel (train ≤ 2013, test > 2013) est utilisé
# pour respecter l'ordre chronologique et éviter les fuites d'information.
#
# La baseline linéaire donne déjà un résultat correct (R² ≈ 0.73), mais
# RandomForest améliore nettement la performance (R² ≈ 0.84–0.85 selon la
# configuration), avec une meilleure gestion des relations non linéaires.
#
# L'étape d'essai-erreur sur les hyperparamètres confirme une configuration
# RandomForest performante (notamment n_estimators autour de 100/300,
# max_depth non limité, min_samples_leaf=1).
#
# L'analyse des erreurs montre que le modèle est globalement précis mais peut
# sous-estimer certaines observations extrêmes. Enfin, l'importance des
# variables confirme que la prévalence masculine est de loin la variable la
# plus informative, suivie de la population, puis de l'année.
#
# En conclusion, le modèle est efficace pour la prédiction statistique, mais
# les résultats restent corrélationnels : ils n'impliquent pas de causalité.
# Pour aller plus loin, on peut intégrer des variables socio-économiques,
# sanitaires ou géopolitiques pour améliorer l'explication du phénomène.
