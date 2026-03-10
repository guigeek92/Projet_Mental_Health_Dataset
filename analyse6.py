import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Mettre a True si tu veux ouvrir les fenetres graphiques
AFFICHER_GRAPHIQUE = False

# Dossier de sorties
doc_dir = Path("donnees_brutes/doc_analyse6")
doc_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 1) Chargement des donnees
# =========================
df = pd.read_csv("donnees_brutes/share-with-mental-or-substance-disorders-by-sex.csv")

col_male = "Prevalence - Mental and substance use disorders - Sex: Male - Age: Age-standardized (Percent)"
col_female = "Prevalence - Mental and substance use disorders - Sex: Female - Age: Age-standardized (Percent)"
col_pop = "Population (historical estimates)"

print("=== Apercu du dataset (analyse6) ===")
print(f"Shape initiale: {df.shape}")
print(df.head())
print("\nValeurs manquantes (initial):")
print(df.isna().sum())

# Filtre temporel pour rester comparable aux autres analyses
# et eviter les lignes historiques tres anciennes.
df = df[(df["Year"] >= 1990) & (df["Year"] <= 2019)].copy()

colonnes_utiles = ["Entity", "Year", col_male, col_female, col_pop, "Continent"]
df = df[colonnes_utiles].dropna(subset=["Year", col_male, col_female]).copy()

print("\n=== Apres nettoyage ===")
print(f"Shape nettoyee: {df.shape}")
print(f"Nombre d'entites: {df['Entity'].nunique()}")
print(f"Periode: {int(df['Year'].min())} - {int(df['Year'].max())}")


# =========================
# 2) Matrice de correlation
# =========================
df_corr = df[["Year", col_male, col_female, col_pop]].copy()
matrice_corr = df_corr.corr()

renommage = {
    "Year": "Year",
    col_male: "Mental_Substance_Male",
    col_female: "Mental_Substance_Female",
    col_pop: "Population",
}
matrice_corr = matrice_corr.rename(index=renommage, columns=renommage)

print("\n=== Matrice de correlation ===")
print(matrice_corr)

plt.figure(figsize=(8, 6))
plt.imshow(matrice_corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(matrice_corr.columns)), matrice_corr.columns, rotation=35, ha="right")
plt.yticks(range(len(matrice_corr.index)), matrice_corr.index)
plt.title("Correlation hommes/femmes (mental+substances)")
plt.tight_layout()

if AFFICHER_GRAPHIQUE:
    plt.show(block=False)
    plt.pause(2)
    plt.close()
else:
    out = doc_dir / "matrice_correlation_mental_substance_mf.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Heatmap enregistree: {out}")


# =========================
# 3) Ecart hommes/femmes
# =========================
df["Gap_female_minus_male"] = df[col_female] - df[col_male]

gap_annuel = df.groupby("Year")["Gap_female_minus_male"].mean().reset_index()
print("\n=== Ecart moyen annuel (femmes - hommes) ===")
print(gap_annuel.head())

plt.figure(figsize=(10, 5))
plt.plot(gap_annuel["Year"], gap_annuel["Gap_female_minus_male"], marker="o")
plt.axhline(0, color="black", linewidth=1)
plt.title("Evolution de l'ecart moyen Femmes - Hommes")
plt.xlabel("Annee")
plt.ylabel("Ecart de prevalence (points de %) ")
plt.grid(alpha=0.3)
plt.tight_layout()
out = doc_dir / "evolution_gap_female_minus_male.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"Graphique enregistre: {out}")

if df["Continent"].notna().any():
    gap_continent = (
        df.dropna(subset=["Continent"])
        .groupby("Continent")["Gap_female_minus_male"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nEcart moyen (femmes-hommes) par continent:")
    print(gap_continent.to_string())

    plt.figure(figsize=(8, 5))
    plt.bar(gap_continent.index, gap_continent.values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Ecart moyen (points de %)")
    plt.title("Ecart moyen Femmes - Hommes par continent")
    plt.tight_layout()
    out = doc_dir / "gap_moyen_par_continent.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Graphique enregistre: {out}")


# =========================
# 4) Probleme predictif
# =========================
# Cible: prevalence femmes
# Features: annee + prevalence hommes + population
features = ["Year", col_male, col_pop]
col_cible = col_female

df_modele = df[features + [col_cible]].dropna().copy()

train_mask = df_modele["Year"] <= 2013
X_train = df_modele.loc[train_mask, features]
y_train = df_modele.loc[train_mask, col_cible]
X_test = df_modele.loc[~train_mask, features]
y_test = df_modele.loc[~train_mask, col_cible]


# =========================
# 5) Comparaison de modeles
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

    resultats.append(
        {
            "Modele": nom_modele,
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
        }
    )

resultats_df = pd.DataFrame(resultats).sort_values(by="RMSE")
print("\n=== Comparaison des modeles ===")
print(resultats_df.to_string(index=False))

meilleur_modele_nom = resultats_df.iloc[0]["Modele"]
print(f"\nMeilleur modele (selon RMSE): {meilleur_modele_nom}")


# =========================
# 6) Tuning RandomForest
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

    resultats_tuning.append(
        {
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "min_samples_leaf": config["min_samples_leaf"],
            "MAE": mean_absolute_error(y_test, y_pred_rf),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            "R2": r2_score(y_test, y_pred_rf),
        }
    )

resultats_tuning_df = pd.DataFrame(resultats_tuning).sort_values(by="RMSE")
print("\n=== Tuning RandomForest ===")
print(resultats_tuning_df.to_string(index=False))

meilleure_config = resultats_tuning_df.iloc[0]


# =========================
# 7) Qualite predictive + importance
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

df_erreurs = pd.DataFrame(
    {
        "Year": X_test["Year"].values,
        "y_reel": y_test.values,
        "y_pred": y_pred_final,
        "erreur": residus.values,
        "erreur_absolue": np.abs(residus.values),
    }
)

print("\n=== Qualite predictive detaillee ===")
print(f"Erreur moyenne (residu moyen): {df_erreurs['erreur'].mean():.4f}")
print(f"Ecart-type des residus      : {df_erreurs['erreur'].std():.4f}")
print("\nTop 10 plus grosses erreurs absolues:")
print(df_erreurs.sort_values(by="erreur_absolue", ascending=False).head(10).to_string(index=False))

plt.figure(figsize=(7, 5))
plt.scatter(df_erreurs["y_reel"], df_erreurs["y_pred"], alpha=0.35)
min_val = min(df_erreurs["y_reel"].min(), df_erreurs["y_pred"].min())
max_val = max(df_erreurs["y_reel"].max(), df_erreurs["y_pred"].max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)
plt.xlabel("Valeurs reelles (femmes)")
plt.ylabel("Valeurs predites")
plt.title("Qualite predictive : reel vs predit")
plt.tight_layout()
out = doc_dir / "qualite_predictive_female.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"Graphique enregistre: {out}")

importances = pd.DataFrame(
    {
        "Feature": features,
        "Importance": modele_final.feature_importances_,
    }
).sort_values(by="Importance", ascending=False)

print("\nImportance des variables:")
print(importances.to_string(index=False))

plt.figure(figsize=(8, 4))
plt.bar(importances["Feature"], importances["Importance"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Importance")
plt.title("Importance des variables (RandomForest)")
plt.tight_layout()
out = doc_dir / "importance_variables_female.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"Graphique enregistre: {out}")


# =========================
# Recap
# =========================
# Cette analyse compare la prevalence des troubles mentaux et de substances
# entre hommes et femmes. Le script filtre la periode recente, etudie les
# correlations, mesure l'ecart moyen femmes-hommes (global et continent), puis
# construit un modele predictif de la prevalence feminine avec comparaison de
# modeles, tuning RandomForest, diagnostic d'erreurs et importance des variables.

# =========================
# Conclusion
# =========================
# L'ecart femmes-hommes est quantifiable et varie selon les zones geographiques.
# La prevalence masculine est un signal fort pour predire la prevalence feminine,
# et RandomForest ameliore generalement la precision par rapport aux modeles
# lineaires. Les resultats restent statistiques; pour une lecture explicative,
# il faut completer avec des determinants socio-sanitaires externes.
