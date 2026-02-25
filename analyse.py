# Import des librairies utiles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#import seaborn as sns 
import pandas as pd
df = pd.read_csv('donnees_brutes/mental-and-substance-use-as-share-of-disease.csv')
print(df.head())
print(df.info())
df.isna().sum()
df.dropna().sum()
# Supprimer la colonne "Code"
df = df.drop(columns=["Code"])

# Vérifier les colonnes restantes
col = "DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)"

moyenne = df[col].mean()
mediane = df[col].median()

print("Moyenne :", moyenne)
print("Médiane :", mediane)
print(df.loc[df['Entity'] == 'Afghanistan', col])
"""La matrice de corrélation est un tableau qui montre la relation linéaire entre toutes les paires de variables numériques.

Valeurs de corrélation :

+1 : Corrélation positive parfaite (quand A augmente, B augmente)
-1 : Corrélation négative parfaite (quand A augmente, B diminue)
0 : Aucune corrélation linéaire

"""
# Calculer la matrice de corrélation
correl = df[['Year', col]].corr()
heatmap = plt.matshow(correl)
print(correl)

# Tracer les DALYs en fonction des années (moyenne sur tous les pays)
dalys_par_annee = df.groupby('Year')[col].mean().sort_index()

# Matrice de corrélation sur les moyennes annuelles des DALYs
moyennes_annuelles_df = dalys_par_annee.reset_index(name='DALYs_moyens')
correl_moyennes_annuelles = moyennes_annuelles_df[['Year', 'DALYs_moyens']].corr()
print("\nMatrice de corrélation (Year vs moyenne annuelle des DALYs) :")
print(correl_moyennes_annuelles)

# Régression linéaire avec train/test sur Year -> DALYs_moyens
X = moyennes_annuelles_df[['Year']].values
y = moyennes_annuelles_df['DALYs_moyens'].values

print("Dimensions X, y :", X.ndim, X.shape, y.ndim, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


# =========================
# 1) Chargement des données
# =========================
df = pd.read_csv('donnees_brutes/mental-and-substance-use-as-share-of-disease.csv')
df = df.drop(columns=['Code'])

col = "DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)"

print("Aperçu des données :")
print(df.head())
print("\nInfos :")
print(df.info())
print("\nValeurs manquantes :")
print(df.isna().sum())


# ========================================
# 2) Statistiques et corrélations de base
# ========================================
print("\nMoyenne DALYs :", df[col].mean())
print("Médiane DALYs :", df[col].median())

correl_brute = df[['Year', col]].corr()
print("\nMatrice de corrélation (données brutes pays-année) :")
print(correl_brute)


# =============================================
# 3) Série agrégée : moyenne DALYs par année
# =============================================
dalys_par_annee = df.groupby('Year')[col].mean().sort_index()
moyennes_annuelles_df = dalys_par_annee.reset_index(name='DALYs_moyens')

correl_moyennes_annuelles = moyennes_annuelles_df[['Year', 'DALYs_moyens']].corr()
print("\nMatrice de corrélation (Year vs moyenne annuelle des DALYs) :")
print(correl_moyennes_annuelles)


# =======================================================
# 4) Modélisation propre (split temporel + comparaison)
# =======================================================
X = moyennes_annuelles_df[['Year']].values
y = moyennes_annuelles_df['DALYs_moyens'].values

print("\nDimensions X, y :", X.ndim, X.shape, y.ndim, y.shape)

n = len(moyennes_annuelles_df)
split_index = int(n * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

modeles = {
    'Lineaire': LinearRegression(),
    # Modèle polynômial degré 2 : ajoute le terme Year^2 pour capter une courbure légère.
    # Si la relation n'est pas strictement une droite, ce modèle peut mieux s'ajuster.
    'Polynome_deg2': make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression()),
    # Ridge (alpha=1.0) : régression linéaire régularisée.
    # Le paramètre alpha contrôle la pénalisation des coefficients:
    # - alpha plus grand => modèle plus simple/stable (moins de surapprentissage)
    # - alpha plus petit => comportement proche d'une régression linéaire classique
    'Ridge_alpha_1': Ridge(alpha=1.0)
}

resultats = []
for nom_modele, modele in modeles.items():
    modele.fit(X_train, y_train)
    y_pred_test = modele.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    erreur_relative = (mae / y_test.mean()) * 100

    resultats.append({
        'Modele': nom_modele,
        'MAE': mae,
        'RMSE': rmse,
        'R2_test': r2,
        'Erreur_relative_%': erreur_relative
    })

resultats_df = pd.DataFrame(resultats).sort_values(by='R2_test', ascending=False)
print("\nComparaison des modèles (split temporel) :")
print(resultats_df)

meilleur_modele_nom = resultats_df.iloc[0]['Modele']
meilleur_modele = modeles[meilleur_modele_nom]

annee_prediction = 2030
prediction = meilleur_modele.predict([[annee_prediction]])
print(f"\nMeilleur modèle : {meilleur_modele_nom}")
print(f"Prédiction DALYs moyens pour {annee_prediction} : {prediction[0]:.4f}")


# ======================================
# 5) Visualisation moyenne + tendance
# ======================================
x = dalys_par_annee.index.values
y_annuel = dalys_par_annee.values

y_pred_global = meilleur_modele.predict(x.reshape(-1, 1))
pearson = np.corrcoef(x, y_annuel)[0, 1]
r2_global = 1 - np.sum((y_annuel - y_pred_global) ** 2) / np.sum((y_annuel - y_annuel.mean()) ** 2)

print(f"\nCorrélation Pearson (Year vs DALYs moyens) : {pearson:.4f}")
print(f"R² global (modèle retenu sur série complète) : {r2_global:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(x, y_annuel, marker='o', label='DALYs moyens par année')
plt.plot(x, y_pred_global, color='red', linewidth=2, label=f'Tendance ({meilleur_modele_nom})')
plt.title("Évolution des DALYs moyens (troubles mentaux) en fonction des années")
plt.xlabel("Année")
plt.ylabel("DALYs (%)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(
    0.02,
    0.95,
    f"Pearson = {pearson:.3f}\nR² global = {r2_global:.3f}",
    transform=plt.gca().transAxes,
    verticalalignment='top'
)
plt.tight_layout()
plt.show()


# ================================================
# 6) Visualisation lisible : 10 pays uniquement
# ================================================
top_10_pays = (
    df.groupby('Entity')[col]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .index
)

plt.figure(figsize=(14, 8))
for entity, groupe in df[df['Entity'].isin(top_10_pays)].groupby('Entity'):
    groupe_trie = groupe.sort_values('Year')
    plt.plot(
        groupe_trie['Year'],
        groupe_trie[col],
        label=entity,
        linewidth=1.2,
        alpha=0.8
    )

plt.title("DALYs (%) par année pour 10 pays (DALYs moyens les plus élevés)")
plt.xlabel("Année")
plt.ylabel("DALYs (%)")
plt.grid(True, alpha=0.25)
plt.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.show()


# =====================================
# 7) Infos en plus : pentes par pays
# =====================================
pentes_pays = []
for entity, groupe in df.groupby('Entity'):
    groupe_trie = groupe.sort_values('Year')
    if len(groupe_trie) >= 2:
        pente, intercept = np.polyfit(groupe_trie['Year'], groupe_trie[col], 1)
        pentes_pays.append({'Entity': entity, 'Pente_par_an': pente})

pentes_df = pd.DataFrame(pentes_pays).sort_values(by='Pente_par_an', ascending=False)

print("\nTop 10 pays avec la plus forte hausse :")
print(pentes_df.head(10))

print("\nTop 10 pays avec la plus forte baisse :")
print(pentes_df.tail(10))