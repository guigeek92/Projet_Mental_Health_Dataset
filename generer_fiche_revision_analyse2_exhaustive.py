import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from datetime import datetime
import textwrap

import matplotlib.patches as mpatches


# Dossier de sortie
OUT_DIR = Path(__file__).resolve().parent / "fichiers_generes"
OUT_DIR.mkdir(exist_ok=True)
OUT_PDF = OUT_DIR / "fiche_revision_analyse2_exhaustive.pdf"



TITRE = "Fiche de révision : Analyse2 – Corrélation et Prédiction"
DATE = datetime.now().strftime("%d/%m/%Y")
AUTEURS = "Auteur : [Ton Nom]"


SECTIONS = [
    # ... (toutes les sections pédagogiques ici, comme avant, à restaurer depuis la version fonctionnelle précédente) ...
]

# Ajout à la toute fin du fichier, après la définition de SECTIONS
SECTIONS.append(("Pour aller plus loin : extraits de code commentés", [
    ("Chargement et nettoyage des données", '#37474f', 'bold',
     """df = pd.read_csv('donnees_brutes/prevalence-by-mental-and-substance-use-disorder.csv')\ndf = df.dropna()"""),
    ("Explication", '#263238', 'normal',
     "On charge le fichier CSV contenant les données, puis on supprime les lignes incomplètes pour éviter les biais ou erreurs dans les analyses statistiques."),
    ("Calcul de la matrice de corrélation", '#37474f', 'bold',
     """matrice_corr = df[colonnes].corr()\nmatrice_corr.index = [noms_courts[c] for c in matrice_corr.index]\nmatrice_corr.columns = [noms_courts[c] for c in matrice_corr.columns]"""),
    ("Explication", '#263238', 'normal',
     "On calcule la corrélation linéaire entre chaque trouble pour repérer ceux qui évoluent ensemble. Cela aide à comprendre les liens statistiques entre maladies."),
    ("Séparation train/test et modélisation", '#37474f', 'bold',
     """from sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nmodele = LinearRegression()\nmodele.fit(X_train, y_train)\ny_pred = modele.predict(X_test)"""),
    ("Explication", '#263238', 'normal',
         "On sépare les données en deux groupes : un pour entraîner le modèle, un pour tester sa capacité à généraliser. On entraîne une régression linéaire pour prédire la dépression à partir des autres troubles."),
        ("Tuning (essai-erreur) du RandomForest", '#37474f', 'bold',
         """configurations_rf = [\n    {'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1},\n    {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 1},\n    {'n_estimators': 500, 'max_depth': None, 'min_samples_leaf': 1},\n    {'n_estimators': 300, 'max_depth': 12, 'min_samples_leaf': 1},\n    {'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 2},\n]\nfor config in configurations_rf:\n    modele_rf = RandomForestRegressor(**config, random_state=42, n_jobs=-1)\n    modele_rf.fit(X_train, y_train)\n    y_pred_rf = modele_rf.predict(X_test)"""),
        ("Explication", '#263238', 'normal',
         "On teste différentes configurations du modèle RandomForest (nombre d'arbres, profondeur, etc.) pour trouver celle qui donne les meilleurs résultats.\nC'est la démarche d'essai-erreur : on ajuste, on teste, on compare, on retient la meilleure configuration selon les scores obtenus sur le jeu de test."),
    ], None))
    ("Concepts fondamentaux du machine learning", [
        ("Apprentissage supervisé", '#1b5e20', 'bold', "C’est une méthode où l’on fournit au modèle des exemples composés de données d’entrée (features) et de la bonne réponse (variable cible ou label). Le but est que le modèle apprenne à prédire la cible à partir des entrées. Exemple : prédire si un email est un spam (entrée = texte de l’email, cible = spam ou non)."),
        ("Données d’entrée (features)", '#1976d2', 'bold', "Ce sont les caractéristiques mesurées pour chaque observation. Elles peuvent être numériques (âge, taille), catégorielles (sexe, couleur), ou textuelles. Exemple : pour prédire le prix d’une maison, les features peuvent être la surface, le nombre de pièces, la localisation, etc."),
        ("Variable cible (target/label)", '#c62828', 'bold', "C’est la variable que l’on souhaite prédire. Elle peut être continue (prix, température) ou discrète (catégorie, classe). Exemple : dans un diagnostic médical, la cible est la présence ou non d’une maladie."),
        ("Modèle", '#6a1b9a', 'bold', "C’est un algorithme mathématique ou statistique qui apprend à faire le lien entre les données d’entrée et la variable cible. Il existe de nombreux types de modèles (régression, arbres, réseaux de neurones, etc.)."),
        ("Entraînement (training/fit)", '#ff8f00', 'bold', "C’est la phase où le modèle ajuste ses paramètres internes pour minimiser l’erreur entre ses prédictions et la vraie valeur de la cible sur les exemples connus. On parle aussi de 'fit' du modèle."),
        ("Prédiction (prediction)", '#00838f', 'bold', "Une fois entraîné, le modèle peut être utilisé pour estimer la cible sur de nouvelles données jamais vues. Exemple : prédire le prix d’une maison à partir de ses caractéristiques."),
        ("Généralisation", '#455a64', 'bold', "C’est la capacité du modèle à bien fonctionner sur des données nouvelles, différentes de celles utilisées pour l’entraînement. Un bon modèle généralise bien, il ne se contente pas de mémoriser les exemples d’entraînement.")
    ], None),
    ("Comprendre le code analyse2.py", [
        ("1. Chargement et nettoyage des données", '#1b5e20', 'bold', "Le script commence par charger le fichier CSV contenant la prévalence de plusieurs troubles mentaux dans le monde, puis supprime les lignes incomplètes pour garantir la qualité des analyses."),
        ("Code", '#263238', 'normal', """df = pd.read_csv('donnees_brutes/prevalence-by-mental-and-substance-use-disorder.csv')\ndf = df.dropna()"""),
        ("Explication détaillée", '#263238', 'normal', "On utilise pandas pour lire le fichier CSV. La fonction dropna() supprime toutes les lignes où il manque au moins une valeur, ce qui évite d'introduire des biais ou des erreurs dans les calculs statistiques ultérieurs."),

        ("2. Sélection des variables", '#1976d2', 'bold', "On sélectionne les colonnes du CSV correspondant à la prévalence (en %) de chaque trouble mental étudié (dépression, schizophrénie, etc.)."),
        ("Code", '#263238', 'normal', """colonnes = [\n    'Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)',\n    ...\n]"""),
        ("Explication détaillée", '#263238', 'normal', "On crée une liste de noms de colonnes pour ne garder que les variables pertinentes pour l'analyse. Cela permet de simplifier la suite du traitement et de se concentrer sur les troubles d'intérêt."),

        ("3. Matrice de corrélation", '#c62828', 'bold', "On calcule la corrélation linéaire entre chaque paire de troubles pour repérer ceux qui évoluent ensemble."),
        ("Code", '#263238', 'normal', """matrice_corr = df[colonnes].corr()\nmatrice_corr.index = [noms_courts[c] for c in matrice_corr.index]\nmatrice_corr.columns = [noms_courts[c] for c in matrice_corr.columns]"""),
        ("Explication détaillée", '#263238', 'normal', "La corrélation de Pearson mesure à quel point deux variables évoluent ensemble. On renomme les lignes et colonnes pour une lecture plus claire. Une heatmap est ensuite générée pour visualiser ces liens."),

        ("4. Problème prédictif", '#6a1b9a', 'bold', "On cherche à prédire la prévalence de la dépression à partir des autres troubles et de l’année. Les données sont séparées temporellement pour éviter la fuite d’information."),
        ("Code", '#263238', 'normal', """from sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"""),
        ("Explication détaillée", '#263238', 'normal', "On utilise train_test_split de scikit-learn pour séparer les données en un jeu d'entraînement et un jeu de test. Cela permet d'évaluer la capacité du modèle à généraliser sur des données nouvelles."),

        ("5. Modélisation", '#ff8f00', 'bold', "On teste plusieurs modèles (régression linéaire, Ridge, Random Forest) pour prédire la dépression à partir des autres troubles."),
        ("Code", '#263238', 'normal', """modeles = {\n    'LinearRegression': LinearRegression(),\n    'Ridge_alpha_1': Ridge(alpha=1.0),\n    'RandomForest': RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),\n}\nfor nom_modele, modele_test in modeles.items():\n    modele_test.fit(X_train, y_train)\n    y_pred_test = modele_test.predict(X_test)"""),
        ("Explication détaillée", '#263238', 'normal', "On compare les performances de chaque modèle à l'aide de métriques comme MAE, RMSE et R2. Cela permet de choisir le modèle le plus adapté au problème."),

        ("6. Tuning et analyse (essai-erreur)", '#00838f', 'bold', "On ajuste les hyperparamètres du Random Forest pour améliorer la performance, puis on analyse les erreurs et l’importance des variables pour l’interprétation."),
        ("Code", '#263238', 'normal', """configurations_rf = [\n    {'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1},\n    ...\n]\nfor config in configurations_rf:\n    modele_rf = RandomForestRegressor(**config, random_state=42, n_jobs=-1)\n    modele_rf.fit(X_train, y_train)\n    y_pred_rf = modele_rf.predict(X_test)"""),
        ("Explication détaillée", '#263238', 'normal', "On teste différentes configurations du modèle RandomForest (nombre d'arbres, profondeur, etc.) pour trouver celle qui donne les meilleurs résultats. C'est la démarche d'essai-erreur : on ajuste, on teste, on compare, on retient la meilleure configuration selon les scores obtenus sur le jeu de test. On analyse aussi l'importance des variables pour comprendre quels troubles influencent le plus la dépression."),

        ("7. Visualisation", '#455a64', 'bold', "Des graphiques sont générés pour visualiser la qualité des prédictions et l’importance des variables. Cela permet d’interpréter et de communiquer les résultats de façon claire."),
    ], None),
    ("Types de problèmes", [
        "Régression : Problème où la variable cible est numérique et continue. Exemple : prédire le prix d’une maison, la température, le taux de chômage. Le modèle doit estimer une valeur réelle.",
        "Classification : Problème où la variable cible est une catégorie. Exemple : reconnaître le type d’animal sur une photo (chat, chien, oiseau).",
        "Classification binaire : Cas particulier de classification où il n’y a que deux classes possibles (ex : maladie présente/absente, spam/non spam).",
        "Classification multiclasse : La cible peut prendre plus de deux valeurs (ex : type de fruit : pomme, poire, banane). Chaque observation appartient à une seule classe parmi plusieurs possibles."
    ], None),
    ("Données & variables", [
        "Variable numérique : Variable qui prend des valeurs mesurables, continues ou discrètes. Exemple : taille, poids, nombre d’enfants.",
        "Variable catégorielle : Variable qui prend des valeurs qualitatives, appelées modalités. Exemple : couleur (rouge, vert, bleu), sexe (homme, femme).",
        "Feature engineering : Processus de création de nouvelles variables à partir des données existantes pour améliorer la performance du modèle. Exemple : transformer une date de naissance en âge, ou combiner plusieurs variables en une seule.",
        "Sélection de variables (feature selection) : Choisir les variables les plus utiles pour la prédiction, afin de simplifier le modèle et d’éviter le bruit. Techniques : sélection automatique, analyse de corrélation, importance des variables.",
        "Encodage : Transformation des variables catégorielles en format numérique pour que les modèles puissent les utiliser. Les modèles mathématiques ne comprennent que les nombres.",
        "One-hot encoding : Méthode d’encodage qui crée une colonne binaire (0/1) pour chaque modalité d’une variable catégorielle. Exemple : pour la variable 'couleur' avec les modalités rouge, vert, bleu, on crée trois colonnes (couleur_rouge, couleur_vert, couleur_bleu).",
        "Label encoding : Méthode d’encodage qui attribue un nombre entier à chaque modalité. Exemple : rouge=0, vert=1, bleu=2. Attention : cela introduit un ordre arbitraire qui n’a pas toujours de sens.",
        "Normalisation / standardisation : Techniques pour ramener les variables numériques sur une même échelle (par exemple entre 0 et 1, ou moyenne 0 et écart-type 1). Cela évite qu’une variable avec de grandes valeurs domine l’apprentissage."
    ], None),
    ("Préparation des données", [
        "Données manquantes (missing values) : Il arrive que certaines valeurs soient absentes dans le jeu de données. Il faut les traiter pour éviter des erreurs lors de l’entraînement. Solutions : supprimer les lignes/colonnes concernées, ou remplacer les valeurs manquantes par la moyenne, la médiane, ou une valeur spécifique (imputation).",
        "Outliers (valeurs aberrantes) : Ce sont des valeurs très éloignées de la majorité des données. Elles peuvent fausser l’analyse et l’entraînement du modèle. On peut les détecter par des graphiques (boxplot) ou des règles statistiques, puis décider de les corriger ou de les exclure.",
        "Nettoyage des données : Ensemble des opérations pour corriger les erreurs, supprimer les doublons, harmoniser les formats, et rendre les données cohérentes et exploitables.",
        "Split train / test : On divise le jeu de données en deux parties : un jeu d’entraînement (train set) pour apprendre, et un jeu de test (test set) pour évaluer la performance sur des données nouvelles. Cela permet de mesurer la capacité de généralisation du modèle."
    ], None),
    ("Modèles (à connaître au moins de nom)", [
        "Régression linéaire : Modèle mathématique qui cherche la meilleure droite pour prédire une variable numérique à partir d’autres variables. Simple, rapide, mais limité aux relations linéaires.",
        "Régression logistique : Modèle utilisé pour la classification binaire. Il prédit la probabilité d’appartenir à une classe (ex : malade/pas malade).",
        "Arbre de décision (Decision Tree) : Modèle qui segmente les données en branches selon des règles simples (ex : si âge > 50 alors…). Facile à interpréter, mais sensible au surapprentissage.",
        "Random Forest : Ensemble de nombreux arbres de décision construits sur des sous-échantillons des données. Les prédictions sont moyennées (régression) ou votées (classification). Plus robuste et précis qu’un arbre seul.",
        "KNN (k plus proches voisins) : Pour prédire la cible d’un point, on regarde les k points les plus proches dans l’espace des features et on fait la moyenne (régression) ou le vote majoritaire (classification). Simple mais lent sur de gros jeux de données.",
        "SVM (Support Vector Machine) : Modèle qui cherche la frontière optimale entre les classes, en maximisant la marge entre les points de classes différentes. Efficace pour les problèmes complexes, mais nécessite un bon réglage des paramètres."
    ], None),
    ("Entraînement & amélioration", [
        "Fit du modèle : C’est l’action d’entraîner le modèle, c’est-à-dire d’ajuster ses paramètres internes pour qu’il prédise au mieux la cible à partir des données d’entraînement.",
        "Hyperparamètres : Ce sont des paramètres du modèle qui ne sont pas appris automatiquement, mais fixés par l’utilisateur avant l’entraînement. Exemple : profondeur maximale d’un arbre, nombre de voisins pour KNN, taux d’apprentissage pour un réseau de neurones.",
        "Optimisation : Processus de recherche des meilleurs hyperparamètres pour maximiser la performance du modèle. Peut se faire par essais successifs (grid search, random search) ou algorithmes plus avancés.",
        "Démarche itérative : On ne trouve pas le meilleur modèle du premier coup. On teste plusieurs modèles, on ajuste les paramètres, on compare les résultats, et on retient la meilleure solution.",
        "Comparaison de modèles : On évalue plusieurs modèles selon des métriques adaptées (MAE, RMSE, accuracy, etc.) pour choisir celui qui généralise le mieux sur de nouvelles données."
    ], None),
    ("1. Objectif de l’analyse", [
        "Comprendre les liens entre différents troubles mentaux (corrélation).",
        "Prédire la prévalence de la dépression à partir d’autres troubles et de l’année (modélisation prédictive)."
    ], None),
    ("2. Données utilisées", [
        "Source : Données mondiales sur la prévalence de troubles mentaux et d’addictions.",
        "Variables : Année, prévalence (%) de : dépression, schizophrénie, trouble bipolaire, troubles alimentaires, anxiété, usage de drogues, usage d’alcool."
    ], None),
    ("3. Identification du problème prédictif", [
        "Variables : année, prévalence (%) de dépression, schizophrénie, trouble bipolaire, troubles alimentaires, anxiété, usage de drogues, usage d’alcool.",
        "Problème : prédire la prévalence de la dépression (régression).",
        "Modèles testés : régression linéaire, Ridge, Random Forest."
    ], "On cherche à prédire une variable continue (régression) à partir de plusieurs facteurs.") ,
    ("4. Sélection et préparation des variables", [
        "Variables explicatives choisies selon leur pertinence clinique/statistique.",
        "Pas de variables catégorielles à encoder ici (toutes quantitatives).",
        "Possibilité de créer de nouvelles variables à partir des données brutes."
    ], "La qualité de la sélection des variables influe directement sur la performance du modèle."),
    ("5. Développement du modèle prédictif (vulgarisé)", [
        "On 'fit' le modèle : il ajuste ses paramètres sur les données d'entraînement.",
        "On compare plusieurs modèles et on ajuste leurs paramètres (tuning) pour améliorer la performance.",
        "La démarche est itérative : on teste, on compare, on améliore."
    ], "L'amélioration incrémentale consiste à tester différentes variantes et à retenir la meilleure."),
    ("6. Méthodes d’évaluation et comparaison", [
        "On évalue la généralisation sur un jeu de test (jamais vu à l'entraînement).",
        "Pour petits jeux de données, on utilise la validation croisée (cross-validation)."
    ], "Évaluer sur des données non vues permet de mesurer la capacité à généraliser."),
    ("7. Choix et compréhension des métriques", [
        "MAE : erreur moyenne absolue (plus c'est bas, mieux c'est).",
        "RMSE : racine de l'erreur quadratique moyenne (pénalise les grosses erreurs).",
        "R2 : proportion de la variance expliquée (proche de 1 = très bon)."
    ], "Chaque métrique donne une vision différente de la performance du modèle."),
    ("8. Interprétation des scores et choix du meilleur modèle", [
        "On choisit le modèle avec le meilleur compromis entre performance et robustesse.",
        "On surveille le surapprentissage (overfitting) : bon score sur train, mauvais sur test = modèle trop complexe.",
        "On surveille le sous-apprentissage (underfitting) : mauvais scores partout = modèle trop simple."
    ], "Comparer les scores train/test permet de détecter overfitting et underfitting."),
    ("9. Visualisation et outils", [
        "Graphiques adaptés (matrice de corrélation, importance des variables, courbes de prédiction).",
        "Outils : pandas (données), matplotlib (graphiques), numpy (calculs), scikit-learn (modèles, métriques)."
    ], "La visualisation aide à interpréter et à communiquer les résultats."),
    ("10. Pour aller plus loin", [
        "Tester d’autres modèles avancés (XGBoost, SVM, réseaux de neurones).",
        "Ajouter des variables contextuelles (socio-économiques, accès aux soins).",
        "Utiliser la validation croisée systématique pour une évaluation plus fiable."
    ], "L'exploration de nouveaux modèles et de nouvelles variables peut améliorer la prédiction.")
]

SECTIONS.append(("Pour aller plus loin : extraits de code commentés", [
    ("Chargement et nettoyage des données", '#37474f', 'bold',
     """df = pd.read_csv('donnees_brutes/prevalence-by-mental-and-substance-use-disorder.csv')\ndf = df.dropna()"""),
    ("Explication", '#263238', 'normal',
     "On charge le fichier CSV contenant les données, puis on supprime les lignes incomplètes pour éviter les biais ou erreurs dans les analyses statistiques."),
    ("Calcul de la matrice de corrélation", '#37474f', 'bold',
     """matrice_corr = df[colonnes].corr()\nmatrice_corr.index = [noms_courts[c] for c in matrice_corr.index]\nmatrice_corr.columns = [noms_courts[c] for c in matrice_corr.columns]"""),
    ("Explication", '#263238', 'normal',
     "On calcule la corrélation linéaire entre chaque trouble pour repérer ceux qui évoluent ensemble. Cela aide à comprendre les liens statistiques entre maladies."),
    ("Séparation train/test et modélisation", '#37474f', 'bold',
     """from sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nmodele = LinearRegression()\nmodele.fit(X_train, y_train)\ny_pred = modele.predict(X_test)"""),
    ("Explication", '#263238', 'normal',
     "On sépare les données en deux groupes : un pour entraîner le modèle, un pour tester sa capacité à généraliser. On entraîne une régression linéaire pour prédire la dépression à partir des autres troubles."),
    ("Tuning (essai-erreur) du RandomForest", '#37474f', 'bold',
     """configurations_rf = [\n    {'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1},\n    {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 1},\n    {'n_estimators': 500, 'max_depth': None, 'min_samples_leaf': 1},\n    {'n_estimators': 300, 'max_depth': 12, 'min_samples_leaf': 1},\n    {'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 2},\n]\nfor config in configurations_rf:\n    modele_rf = RandomForestRegressor(**config, random_state=42, n_jobs=-1)\n    modele_rf.fit(X_train, y_train)\n    y_pred_rf = modele_rf.predict(X_test)"""),
    ("Explication", '#263238', 'normal',
     "On teste différentes configurations du modèle RandomForest (nombre d'arbres, profondeur, etc.) pour trouver celle qui donne les meilleurs résultats.\nC'est la démarche d'essai-erreur : on ajuste, on teste, on compare, on retient la meilleure configuration selon les scores obtenus sur le jeu de test."),
], None))

# Paramètres de mise en page

FONT_SIZE_TITLE = 16
FONT_SIZE_SECTION = 15
FONT_SIZE_POINT = 10.5
FONT_SIZE_BOX = 10.5
LINE_SPACING = 0.038
POINT_SPACING = 0.027
MARGIN_TOP = 0.96
MARGIN_BOTTOM = 0.045
MARGIN_LEFT = 0.01
MARGIN_RIGHT = 0.99
MAX_LINES_PER_PAGE = 30


with PdfPages(OUT_PDF) as pdf:
    y = MARGIN_TOP
    page = 1
    fig, ax = plt.subplots(figsize=(8.3, 11.7))  # Format A4
    ax.axis('off')
    # Titre page 1
    ax.text(MARGIN_LEFT, y, TITRE, fontsize=FONT_SIZE_TITLE, fontweight='bold', ha='left', va='top', color='#1a237e')
    y -= 2 * LINE_SPACING
    ax.text(MARGIN_LEFT, y, f"{AUTEURS} | Date : {DATE}", fontsize=9, ha='left', va='top', color='#37474f')
    y -= 2 * LINE_SPACING
    # Plus de schéma, on commence directement après le titre
    line_count = 3

    for section, points, box in SECTIONS:
        if line_count > MAX_LINES_PER_PAGE:
            ax.text(0.5, MARGIN_BOTTOM, f"Fiche générée automatiquement | Page {page}", fontsize=8, ha='center', va='bottom', color='gray')
            fig.tight_layout(rect=[0, 0, 1, 1])
            pdf.savefig(fig)
            plt.close(fig)
            page += 1
            fig, ax = plt.subplots(figsize=(8.3, 11.7))
            ax.axis('off')
            y = MARGIN_TOP
            line_count = 0
        # Ajoute une barre latérale colorée à gauche du titre pour une séparation visuelle forte
        bar_height = LINE_SPACING * 1.5
        ax.add_patch(mpatches.FancyBboxPatch((MARGIN_LEFT-0.005, y-bar_height/2), 0.012, bar_height, boxstyle="round,pad=0.01", ec='#1976d2', fc='#1976d2', mutation_scale=0.01))
        ax.text(MARGIN_LEFT+0.012, y, section, fontsize=FONT_SIZE_SECTION, fontweight='bold', ha='left', va='center', color='#0d47a1', bbox=dict(facecolor='#e3f2fd', edgecolor='none', boxstyle='round,pad=0.2'))
        y -= LINE_SPACING * 1.7
        line_count += 2
        for point in points:
            # Si point est un tuple stylisé (mot, couleur, poids, texte)
            if isinstance(point, tuple):
                mot, couleur, poids, texte = point
                # Bloc de code : encadré gris, police monospace, plus d'espace
                if mot == "Code":
                    code_lines = texte.split('\n')
                    code_box_height = POINT_SPACING * (len(code_lines)+1) + 0.01
                    ax.add_patch(mpatches.FancyBboxPatch((MARGIN_LEFT+0.03, y-code_box_height+0.01), 0.90-MARGIN_LEFT, code_box_height, boxstyle="round,pad=0.04", ec="#b0bec5", fc="#eceff1", alpha=0.97, mutation_scale=0.01))
                    for i, code_line in enumerate(code_lines):
                        ax.text(MARGIN_LEFT+0.045, y-POINT_SPACING*i, code_line, fontsize=FONT_SIZE_POINT+0.3, ha='left', va='top', family='monospace', color='#263238')
                    y -= code_box_height + 0.01
                    line_count += len(code_lines)+1
                # Explication détaillée : encadré bleu clair
                elif mot == "Explication détaillée":
                    wrapped = textwrap.wrap(texte, width=85)
                    box_height = 0.045 * (len(wrapped) + 1)
                    ax.add_patch(mpatches.FancyBboxPatch((MARGIN_LEFT+0.01, y-box_height+0.01), 0.93-MARGIN_LEFT, box_height, boxstyle="round,pad=0.02", ec="#1976d2", fc="#e3f2fd", alpha=0.95, mutation_scale=0.01))
                    for i, line in enumerate(wrapped):
                        ax.text(MARGIN_LEFT+0.025, y-POINT_SPACING*i, line, fontsize=FONT_SIZE_POINT, ha='left', va='top', color='#0d47a1')
                    y -= box_height + 0.01
                    line_count += len(wrapped)+1
                else:
                    wrapped = textwrap.wrap(texte, width=85)
                    for i, line in enumerate(wrapped):
                        if line_count > MAX_LINES_PER_PAGE:
                            ax.text(0.5, MARGIN_BOTTOM, f"Fiche générée automatiquement | Page {page}", fontsize=8, ha='center', va='bottom', color='gray')
                            fig.tight_layout(rect=[0, 0, 1, 1])
                            pdf.savefig(fig)
                            plt.close(fig)
                            page += 1
                            fig, ax = plt.subplots(figsize=(8.3, 11.7))
                            ax.axis('off')
                            y = MARGIN_TOP
                            line_count = 0
                        if i == 0:
                            ax.text(MARGIN_LEFT + 0.01, y, f"- ", fontsize=FONT_SIZE_POINT, ha='left', va='top', wrap=True, color='#263238')
                            ax.text(MARGIN_LEFT + 0.04, y, mot, fontsize=FONT_SIZE_POINT, ha='left', va='top', fontweight=poids, color=couleur)
                            ax.text(MARGIN_LEFT + 0.04 + 0.18, y, ": " + line, fontsize=FONT_SIZE_POINT, ha='left', va='top', wrap=True, color='#263238')
                        else:
                            ax.text(MARGIN_LEFT + 0.04 + 0.18, y, line, fontsize=FONT_SIZE_POINT, ha='left', va='top', wrap=True, color='#263238')
                        y -= POINT_SPACING
                        line_count += 1
            else:
                wrapped = textwrap.wrap(point, width=90)
                for line in wrapped:
                    if line_count > MAX_LINES_PER_PAGE:
                        ax.text(0.5, MARGIN_BOTTOM, f"Fiche générée automatiquement | Page {page}", fontsize=8, ha='center', va='bottom', color='gray')
                        fig.tight_layout(rect=[0, 0, 1, 1])
                        pdf.savefig(fig)
                        plt.close(fig)
                        page += 1
                        fig, ax = plt.subplots(figsize=(8.3, 11.7))
                        ax.axis('off')
                        y = MARGIN_TOP
                        line_count = 0
                    ax.text(MARGIN_LEFT + 0.01, y, f"- {line}", fontsize=FONT_SIZE_POINT, ha='left', va='top', wrap=True, color='#263238')
                    y -= POINT_SPACING
                    line_count += 1
        # Encadré "À retenir" ou "Exemple"
        if box:
            if line_count > MAX_LINES_PER_PAGE - 2:
                ax.text(0.5, MARGIN_BOTTOM, f"Fiche générée automatiquement | Page {page}", fontsize=8, ha='center', va='bottom', color='gray')
                fig.tight_layout(rect=[0, 0, 1, 1])
                pdf.savefig(fig)
                plt.close(fig)
                page += 1
                fig, ax = plt.subplots(figsize=(8.3, 11.7))
                ax.axis('off')
                y = MARGIN_TOP
                line_count = 0
            box_height = 0.045 * (len(textwrap.wrap(box, width=90)) + 1)
            rect = mpatches.FancyBboxPatch((MARGIN_LEFT, y-box_height+0.01), 0.95-MARGIN_LEFT, box_height, boxstyle="round,pad=0.02", ec="#1976d2", fc="#e3f2fd", alpha=0.95, mutation_scale=0.01)
            ax.add_patch(rect)
            ax.text(MARGIN_LEFT+0.015, y-0.01, box, fontsize=FONT_SIZE_BOX, ha='left', va='top', color='#0d47a1')
            y -= box_height + 0.01
            line_count += int(box_height / POINT_SPACING)
        y -= LINE_SPACING / 2
        line_count += 1
    ax.text(0.5, MARGIN_BOTTOM, f"Fiche générée automatiquement avec matplotlib | Page {page}", fontsize=8, ha='center', va='bottom', color='gray')
    fig.tight_layout(rect=[0, 0, 1, 1])
    pdf.savefig(fig)
    plt.close(fig)

print(f"Fiche PDF exhaustive générée : {OUT_PDF}")
