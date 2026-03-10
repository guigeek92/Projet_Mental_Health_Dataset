import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Mettre a True si tu veux ouvrir les fenetres graphiques
AFFICHER_GRAPHIQUE = False

# Dossier de sorties
doc_dir = Path("donnees_brutes/doc_analyse4")
doc_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 1) Chargement des donnees
# =========================
df = pd.read_csv("donnees_brutes/share-with-depression.csv")
col = "Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)"

print("=== Apercu du dataset (analyse4) ===")
print(f"Shape initiale: {df.shape}")
print(df.head())
print("\nValeurs manquantes (initial):")
print(df.isna().sum())

df = df.dropna(subset=["Year", col]).copy()
if "Code" in df.columns:
    df = df.drop(columns=["Code"])

print("\n=== Apres nettoyage ===")
print(f"Shape nettoyee: {df.shape}")
print(f"Nombre d'entites: {df['Entity'].nunique()}")
print(f"Periode: {int(df['Year'].min())} - {int(df['Year'].max())}")


# ========================================
# 2) Statistiques et correlations de base
# ========================================
print("\n=== Stats descriptives ===")
print(f"Moyenne prevalence depression: {df[col].mean():.4f}")
print(f"Mediane prevalence depression: {df[col].median():.4f}")

correl_brute = df[["Year", col]].corr()
print("\nMatrice de correlation (donnees brutes pays-annee):")
print(correl_brute)


# =============================================
# 3) Serie agregee : moyenne par annee
# =============================================
serie_annuelle = df.groupby("Year")[col].mean().sort_index()
df_annuel = serie_annuelle.reset_index(name="Depression_moyenne")

correl_annuelle = df_annuel[["Year", "Depression_moyenne"]].corr()
print("\nMatrice de correlation (Year vs moyenne annuelle):")
print(correl_annuelle)

plt.figure(figsize=(10, 5))
plt.plot(df_annuel["Year"], df_annuel["Depression_moyenne"], marker="o", linewidth=1.5)
plt.title("Evolution de la prevalence moyenne de depression")
plt.xlabel("Annee")
plt.ylabel("Prevalence depression (%)")
plt.grid(alpha=0.3)
plt.tight_layout()

if AFFICHER_GRAPHIQUE:
    plt.show(block=False)
    plt.pause(2)
    plt.close()
else:
    out = doc_dir / "evolution_prevalence_depression.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Graphique enregistre: {out}")


# =======================================================
# 4) Modelisation (split temporel + comparaison)
# =======================================================
X = df_annuel[["Year"]].values
y = df_annuel["Depression_moyenne"].values

n = len(df_annuel)
split_index = int(n * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

modeles = {
    "Lineaire": LinearRegression(),
    "Polynome_deg2": make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression(),
    ),
    "Ridge_alpha_1": Ridge(alpha=1.0),
}

resultats = []
for nom_modele, modele in modeles.items():
    modele.fit(X_train, y_train)
    y_pred_test = modele.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)

    resultats.append(
        {
            "Modele": nom_modele,
            "MAE": mae,
            "RMSE": rmse,
            "R2_test": r2,
        }
    )

resultats_df = pd.DataFrame(resultats).sort_values(by="R2_test", ascending=False)
print("\n=== Comparaison des modeles ===")
print(resultats_df.to_string(index=False))

meilleur_modele_nom = resultats_df.iloc[0]["Modele"]
meilleur_modele = modeles[meilleur_modele_nom]

annee_prediction = 2030
prediction = meilleur_modele.predict([[annee_prediction]])
print(f"\nMeilleur modele: {meilleur_modele_nom}")
print(f"Prediction prevalence moyenne pour {annee_prediction}: {prediction[0]:.4f}")


# ======================================
# 5) Qualite d'ajustement globale
# ======================================
x = df_annuel["Year"].values
y_annuel = df_annuel["Depression_moyenne"].values
y_pred_global = meilleur_modele.predict(x.reshape(-1, 1))

pearson = np.corrcoef(x, y_annuel)[0, 1]
r2_global = 1 - np.sum((y_annuel - y_pred_global) ** 2) / np.sum((y_annuel - y_annuel.mean()) ** 2)

print(f"\nPearson (Year vs prevalence moyenne): {pearson:.4f}")
print(f"R2 global (modele retenu): {r2_global:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(x, y_annuel, marker="o", label="Prevalence moyenne")
plt.plot(x, y_pred_global, color="red", linewidth=2, label=f"Tendance ({meilleur_modele_nom})")
plt.title("Tendance temporelle de la prevalence de depression")
plt.xlabel("Annee")
plt.ylabel("Prevalence depression (%)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(
    0.02,
    0.95,
    f"Pearson = {pearson:.3f}\nR2 global = {r2_global:.3f}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
)
plt.tight_layout()
out = doc_dir / "tendance_prevalence_depression_modele.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"Graphique enregistre: {out}")


# ================================================
# 6) Visualisation lisible : top 10 entites
# ================================================
top_10_entites = (
    df.groupby("Entity")[col]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .index
)

plt.figure(figsize=(14, 8))
for entity, groupe in df[df["Entity"].isin(top_10_entites)].groupby("Entity"):
    g = groupe.sort_values("Year")
    plt.plot(g["Year"], g[col], label=entity, linewidth=1.1, alpha=0.85)

plt.title("Prevalence de depression (%) pour 10 entites les plus elevees")
plt.xlabel("Annee")
plt.ylabel("Prevalence depression (%)")
plt.grid(True, alpha=0.25)
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
out = doc_dir / "top10_entites_depression.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"Graphique enregistre: {out}")


# =====================================
# 7) Pentes par entite
# =====================================
pentes = []
for entity, groupe in df.groupby("Entity"):
    g = groupe.sort_values("Year")
    if len(g) >= 2:
        pente, _ = np.polyfit(g["Year"], g[col], 1)
        pentes.append({"Entity": entity, "Pente_par_an": pente})

pentes_df = pd.DataFrame(pentes).sort_values(by="Pente_par_an", ascending=False)

print("\nTop 10 entites avec la plus forte hausse:")
print(pentes_df.head(10).to_string(index=False))

print("\nTop 10 entites avec la plus forte baisse:")
print(pentes_df.tail(10).to_string(index=False))


# =========================
# Recap
# =========================
# Cette analyse etudie la prevalence de la depression (sexes confondus).
# Le script nettoie les donnees, calcule les statistiques de base, mesure la
# correlation avec l'annee, agrège la serie annuelle, compare plusieurs
# modeles de tendance, puis fournit des visualisations globales et par entite.

# =========================
# Conclusion
# =========================
# La prevalence moyenne de la depression suit une dynamique temporelle
# observable, mais heterogene selon les entites. Le meilleur modele choisi
# apporte une tendance utile pour la projection a court terme. Les resultats
# restent correlatifs; l'ajout de variables socio-economiques permettrait de
# mieux expliquer les variations observees.
