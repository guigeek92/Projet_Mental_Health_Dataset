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
doc_dir = Path("donnees_brutes/doc_analyse5")
doc_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 1) Chargement des donnees
# =========================
df = pd.read_csv("donnees_brutes/share-with-mental-and-substance-disorders.csv")
col = "Prevalence - Mental disorders - Sex: Both - Age: Age-standardized (Percent)"

print("=== Apercu du dataset (analyse5) ===")
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
print(f"Moyenne prevalence troubles mentaux: {df[col].mean():.4f}")
print(f"Mediane prevalence troubles mentaux: {df[col].median():.4f}")

correl_brute = df[["Year", col]].corr()
print("\nMatrice de correlation (donnees brutes pays-annee):")
print(correl_brute)


# =============================================
# 3) Serie agregee : moyenne par annee
# =============================================
serie_annuelle = df.groupby("Year")[col].mean().sort_index()
df_annuel = serie_annuelle.reset_index(name="Mental_moyenne")

correl_annuelle = df_annuel[["Year", "Mental_moyenne"]].corr()
print("\nMatrice de correlation (Year vs moyenne annuelle):")
print(correl_annuelle)

plt.figure(figsize=(10, 5))
plt.plot(df_annuel["Year"], df_annuel["Mental_moyenne"], marker="o", linewidth=1.5)
plt.title("Evolution de la prevalence moyenne des troubles mentaux")
plt.xlabel("Annee")
plt.ylabel("Prevalence troubles mentaux (%)")
plt.grid(alpha=0.3)
plt.tight_layout()

if AFFICHER_GRAPHIQUE:
    plt.show(block=False)
    plt.pause(2)
    plt.close()
else:
    out = doc_dir / "evolution_prevalence_mental.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Graphique enregistre: {out}")


# =======================================================
# 4) Modelisation (split temporel + comparaison)
# =======================================================
X = df_annuel[["Year"]].values
y = df_annuel["Mental_moyenne"].values

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
y_annuel = df_annuel["Mental_moyenne"].values
y_pred_global = meilleur_modele.predict(x.reshape(-1, 1))

pearson = np.corrcoef(x, y_annuel)[0, 1]
r2_global = 1 - np.sum((y_annuel - y_pred_global) ** 2) / np.sum((y_annuel - y_annuel.mean()) ** 2)

print(f"\nPearson (Year vs prevalence moyenne): {pearson:.4f}")
print(f"R2 global (modele retenu): {r2_global:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(x, y_annuel, marker="o", label="Prevalence moyenne")
plt.plot(x, y_pred_global, color="red", linewidth=2, label=f"Tendance ({meilleur_modele_nom})")
plt.title("Tendance temporelle des troubles mentaux")
plt.xlabel("Annee")
plt.ylabel("Prevalence troubles mentaux (%)")
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
out = doc_dir / "tendance_prevalence_mental_modele.png"
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

plt.title("Prevalence troubles mentaux (%) pour 10 entites les plus elevees")
plt.xlabel("Annee")
plt.ylabel("Prevalence troubles mentaux (%)")
plt.grid(True, alpha=0.25)
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
out = doc_dir / "top10_entites_mental.png"
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
# Cette analyse porte sur la prevalence globale des troubles mentaux.
# Le script realise le nettoyage, les statistiques descriptives, l'etude de
# correlation temporelle, la modelisation de tendance annuelle, puis des
# visualisations interpretablees (serie globale, top entites, pentes).

# =========================
# Conclusion
# =========================
# La prevalence moyenne des troubles mentaux presente une tendance temporelle
# mesurable sur la periode observee, avec des ecarts marqués entre entites.
# Le modele retenu decrit correctement la tendance generale, mais ne capture
# pas toute la complexite locale. Une extension multi-facteurs est necessaire
# pour une interpretation causale plus solide.
