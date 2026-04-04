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
        ("1. Chargement et nettoyage des données", '#1b5e20', 'bold', "Le script charge un fichier CSV contenant la prévalence de différents troubles mentaux, puis supprime les lignes incomplètes (données manquantes)."),
        ("2. Sélection des variables", '#1976d2', 'bold', "Il sélectionne les colonnes correspondant à différents troubles (dépression, schizophrénie, etc.) pour l’analyse de corrélation et la modélisation."),
        ("3. Matrice de corrélation", '#c62828', 'bold', "Il calcule la corrélation entre les troubles pour voir lesquels sont liés. Une heatmap est générée pour visualiser ces liens."),
        ("4. Problème prédictif", '#6a1b9a', 'bold', "Le but est de prédire la prévalence de la dépression à partir des autres troubles et de l’année. Les données sont séparées temporellement (avant/après 2013) pour éviter la fuite d’information."),
        ("5. Modélisation", '#ff8f00', 'bold', "Plusieurs modèles sont testés (régression linéaire, Ridge, Random Forest). Leurs performances sont comparées à l’aide de métriques (MAE, RMSE, R2)."),
        ("6. Tuning et analyse", '#00838f', 'bold', "Le script ajuste les hyperparamètres du Random Forest pour améliorer la performance, puis analyse les erreurs et l’importance des variables pour l’interprétation."),
        ("7. Visualisation", '#455a64', 'bold', "Des graphiques sont générés pour visualiser la qualité des prédictions et l’importance des variables.")
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
    ], "L'exploration de nouveaux modèles et de nouvelles variables peut améliorer la prédiction."),

    # Nouvelle section : Optimisation du modèle
    ("Optimisation du modèle : essai-erreur, cross-validation, GridSearchCV", [
        ("Essai-erreur", '#00838f', 'bold', "On modifie manuellement les hyperparamètres du modèle (ex : nombre d'arbres, profondeur, etc.), on réentraîne et on compare les scores (MAE, RMSE, R2) pour trouver la meilleure configuration. C'est la méthode la plus intuitive, mais elle peut être longue et ne garantit pas d'explorer toutes les possibilités."),
        ("Validation croisée (cross-validation)", '#1b5e20', 'bold', "On divise les données en plusieurs sous-ensembles (folds). Le modèle est entraîné sur certains folds et testé sur les autres, puis on recommence en changeant les folds. Cela donne une estimation plus fiable de la performance et limite le surapprentissage. Exemple : cross-validation à 5 folds."),
        ("Recherche automatique (GridSearchCV)", '#c62828', 'bold', "On définit une grille de valeurs possibles pour chaque hyperparamètre. L'algorithme teste automatiquement toutes les combinaisons avec validation croisée, puis retient la meilleure. C'est la méthode la plus systématique pour optimiser un modèle. Elle est utilisée dans le script analyse2.py pour le Random Forest.")
    ], "L'optimisation passe par plusieurs étapes : essais manuels, validation croisée, puis recherche automatique des meilleurs paramètres. Cela permet d'obtenir un modèle performant et robuste.")
]

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


        # Section spéciale pour Concepts fondamentaux et Comprendre le code : titres au-dessus, termes en gras
        if section in ["Concepts fondamentaux du machine learning", "Comprendre le code analyse2.py"]:
            ax.text(MARGIN_LEFT, y, section, fontsize=FONT_SIZE_SECTION, fontweight='bold', ha='left', va='top', color='#0d47a1', bbox=dict(facecolor='#e3f2fd', edgecolor='none', boxstyle='round,pad=0.2'))
            y -= LINE_SPACING * 1.3
            line_count += 1
            for point in points:
                mot, couleur, poids, texte = point
                # Titre du concept ou étape, bien séparé, en couleur et en gras
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
                ax.text(MARGIN_LEFT + 0.01, y, mot, fontsize=FONT_SIZE_POINT+1, fontweight='bold', color=couleur, ha='left', va='top')
                y -= POINT_SPACING * 1.1
                line_count += 1
                wrapped = textwrap.wrap(texte, width=85)
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
                    ax.text(MARGIN_LEFT + 0.04, y, line, fontsize=FONT_SIZE_POINT, ha='left', va='top', wrap=True, color='#263238')
                    y -= POINT_SPACING
                    line_count += 1
                y -= POINT_SPACING * 0.5
                line_count += 1
            continue

        # Section spéciale pour Données & variables et Optimisation du modèle : termes définis en gras
        if section in ["Données & variables", "Optimisation du modèle : essai-erreur, cross-validation, GridSearchCV"]:
            ax.text(MARGIN_LEFT, y, section, fontsize=FONT_SIZE_SECTION, fontweight='bold', ha='left', va='top', color='#0d47a1', bbox=dict(facecolor='#e3f2fd', edgecolor='none', boxstyle='round,pad=0.2'))
            y -= LINE_SPACING * 1.3
            line_count += 1
            import re
            for point in points:
                # Si c'est un tuple stylisé (mot, couleur, poids, texte)
                if isinstance(point, tuple):
                    mot, couleur, poids, texte = point
                    # Mot-clé en gras
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
                    ax.text(MARGIN_LEFT + 0.01, y, mot, fontsize=FONT_SIZE_POINT, fontweight='bold', color=couleur, ha='left', va='top')
                    y -= POINT_SPACING * 0.9
                    line_count += 1
                    wrapped = textwrap.wrap(texte, width=80)
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
                        ax.text(MARGIN_LEFT + 0.04, y, line, fontsize=FONT_SIZE_POINT, ha='left', va='top', wrap=True, color='#263238')
                        y -= POINT_SPACING
                        line_count += 1
                    y -= POINT_SPACING * 0.5
                    line_count += 1
                # Sinon, gestion classique (définition avec ':')
                elif ":" in point:
                    terme, definition = point.split(":", 1)
                    terme = terme.strip()
                    definition = definition.strip()
                    wrapped = textwrap.wrap(definition, width=80)
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
                    ax.text(MARGIN_LEFT + 0.01, y, terme, fontsize=FONT_SIZE_POINT, fontweight='bold', color='#263238', ha='left', va='top')
                    y -= POINT_SPACING * 0.9
                    line_count += 1
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
                        ax.text(MARGIN_LEFT + 0.04, y, line, fontsize=FONT_SIZE_POINT, ha='left', va='top', wrap=True, color='#263238')
                        y -= POINT_SPACING
                        line_count += 1
                    y -= POINT_SPACING * 0.5
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
            continue

        # Pour toutes les autres sections, mettre en gras le terme défini (avant le ':')
        ax.text(MARGIN_LEFT, y, section, fontsize=FONT_SIZE_SECTION, fontweight='bold', ha='left', va='top', color='#0d47a1', bbox=dict(facecolor='#e3f2fd', edgecolor='none', boxstyle='round,pad=0.2'))
        y -= LINE_SPACING * 1.3
        line_count += 1
        import re
        for point in points:
            # Si c'est un tuple stylisé, on garde l'ancien affichage
            if isinstance(point, tuple):
                mot, couleur, poids, texte = point
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
            # Si c'est une définition (mot : explication), on met le mot en gras
            elif ":" in point:
                terme, definition = point.split(":", 1)
                terme = terme.strip()
                definition = definition.strip()
                wrapped = textwrap.wrap(definition, width=80)
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
                ax.text(MARGIN_LEFT + 0.01, y, terme, fontsize=FONT_SIZE_POINT, fontweight='bold', color='#263238', ha='left', va='top')
                y -= POINT_SPACING * 0.9
                line_count += 1
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
                    ax.text(MARGIN_LEFT + 0.04, y, line, fontsize=FONT_SIZE_POINT, ha='left', va='top', wrap=True, color='#263238')
                    y -= POINT_SPACING
                    line_count += 1
                y -= POINT_SPACING * 0.5
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
