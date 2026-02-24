# Import des librairies utiles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
df.loc[df['Entity'] == 'Afghanistan', col] = moyenne
print(df.loc[df['Entity'] == 'Afghanistan', col])       
df.loc[df['Year'] == 2010, col] = moyenne
"""La matrice de corrélation est un tableau qui montre la relation linéaire entre toutes les paires de variables numériques.

Valeurs de corrélation :

+1 : Corrélation positive parfaite (quand A augmente, B augmente)
-1 : Corrélation négative parfaite (quand A augmente, B diminue)
0 : Aucune corrélation linéaire
En pandas :

Exemple de résultat :
                wind_speed  power_output  temperature
wind_speed           1.00          0.95         0.30
power_output         0.95          1.00         0.25
temperature          0.30          0.25         1.00
Interprétation :

wind_speed ↔ power_output : 0.95 → Forte corrélation positive (plus de vent = plus de production)
wind_speed ↔ temperature : 0.30 → Faible corrélation positive
"""
# Calculer la matrice de corrélation
correl = df[['Year', col]].corr()
heatmap = plt.matshow(correl)
print(correl)