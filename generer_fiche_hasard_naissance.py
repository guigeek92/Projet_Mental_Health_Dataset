from datetime import datetime
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "donnees_brutes"
OUT_DIR = ROOT / "fichiers_generes"
OUT_PNG = OUT_DIR / "fiche_hasard_naissance_poster_oral_visuel.png"
OUT_PDF = OUT_DIR / "fiche_hasard_naissance_poster_oral_visuel.pdf"
OUT_MD = OUT_DIR / "fiche_hasard_naissance_scientifique.md"


def base_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def style_plot(ax):
    ax.set_facecolor("#111827")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, color="#334155", linewidth=0.8, alpha=0.35)
    ax.tick_params(colors="#cbd5e1", labelsize=7)
    ax.xaxis.label.set_color("#e5e7eb")
    ax.yaxis.label.set_color("#e5e7eb")


def panel_title(ax, title):
    ax.set_title(title, loc="left", color="#e5e7eb", fontweight="bold", pad=10)


def add_card(ax, title, value, subtitle, color="#22d3ee"):
    ax.axis("off")
    card = FancyBboxPatch(
        (0.01, 0.08),
        0.98,
        0.84,
        boxstyle="round,pad=0.016,rounding_size=12",
        linewidth=1.2,
        edgecolor="#334155",
        facecolor="#111827",
        transform=ax.transAxes,
    )
    ax.add_patch(card)
    ax.add_patch(
        FancyBboxPatch(
            (0.03, 0.78),
            0.94,
            0.08,
            boxstyle="round,pad=0.004,rounding_size=8",
            linewidth=0,
            facecolor=color,
            alpha=0.2,
            transform=ax.transAxes,
        )
    )
    ax.text(0.06, 0.72, title, fontsize=9, color="#cbd5e1", fontweight="bold", transform=ax.transAxes)
    ax.text(0.06, 0.42, value, fontsize=16, color="#f8fafc", fontweight="bold", transform=ax.transAxes)
    ax.text(0.06, 0.18, subtitle, fontsize=8.4, color="#94a3b8", transform=ax.transAxes)


def add_text_panel(ax, title, lines, wrap_width=42):
    ax.axis("off")
    panel = FancyBboxPatch(
        (0.01, 0.04),
        0.98,
        0.92,
        boxstyle="round,pad=0.018,rounding_size=12",
        linewidth=1.2,
        edgecolor="#334155",
        facecolor="#111827",
        transform=ax.transAxes,
    )
    ax.add_patch(panel)
    ax.text(0.04, 0.90, title, fontsize=11, color="#f8fafc", fontweight="bold", va="top", transform=ax.transAxes)
    wrapped = [textwrap.fill(line, width=wrap_width) for line in lines]
    ax.text(0.04, 0.78, "\n".join(wrapped), fontsize=9.1, color="#d1d5db", va="top", linespacing=1.5, transform=ax.transAxes)


def generate_scientific_sheet(data):
    return f"""
# {data['title']}

## Auteurs
{data['authors']}

## Contexte
{data['context']}

## Objectif
{data['objective']}

## Donnees utilisees
- Dataset principal : {data['dataset_main']}
- Dataset complementaire : {data['dataset_aux']}
- Nombre d'observations : {data['n_samples']}
- Variables principales : {", ".join(data['features'])}

## Methodologie
Modeles testes :
{chr(10).join(["- " + m for m in data['models']])}

Methode de validation :
{data['validation']}

Packages utilises :
{", ".join(data['packages'])}

## Resultats

| Modele | RMSE | MAE | R2 |
|------|------|------|------|
{chr(10).join([f"| {row['modele']} | {row['rmse']} | {row['mae']} | {row['r2']} |" for row in data['results']])}

Meilleur modele : **{data['best_model']}**

## Interpretation

{data['interpretation']}

## Limites

{data['limits']}

## Perspectives

{data['future_work']}

---

Fiche generee automatiquement le {datetime.now().strftime("%d/%m/%Y")}
"""


def build_dataset():
    overall_path = DATA / "share-with-mental-and-substance-disorders.csv"
    by_sex_path = DATA / "share-with-mental-or-substance-disorders-by-sex.csv"

    overall_col = "Prevalence - Mental disorders - Sex: Both - Age: Age-standardized (Percent)"
    male_col = "Prevalence - Mental and substance use disorders - Sex: Male - Age: Age-standardized (Percent)"
    female_col = "Prevalence - Mental and substance use disorders - Sex: Female - Age: Age-standardized (Percent)"
    pop_col = "Population (historical estimates)"

    overall = pd.read_csv(overall_path)[["Entity", "Code", "Year", overall_col]].dropna().copy()
    by_sex = pd.read_csv(by_sex_path)[["Entity", "Code", "Year", male_col, female_col, pop_col, "Continent"]].dropna().copy()

    meta = by_sex[["Entity", "Code", "Year", pop_col, "Continent"]].copy()
    merged = overall.merge(meta, on=["Entity", "Code", "Year"], how="inner")

    latest_year = int(merged["Year"].max())
    latest = merged.loc[merged["Year"] == latest_year].copy()

    continent_rows = []
    for continent, group in latest.groupby("Continent"):
        continent_rows.append(
            {
                "Continent": continent,
                "prevalence": float(np.average(group[overall_col], weights=group[pop_col])),
            }
        )
    continent_latest = pd.DataFrame(continent_rows).sort_values("prevalence", ascending=False)

    country_latest = latest[["Entity", overall_col]].rename(columns={overall_col: "prevalence"}).sort_values("prevalence")
    bottom5 = country_latest.head(5)
    top5 = country_latest.tail(5).sort_values("prevalence", ascending=False)
    extremes = pd.concat([top5, bottom5], ignore_index=True)

    long_sex = by_sex.melt(
        id_vars=["Entity", "Code", "Year", pop_col, "Continent"],
        value_vars=[male_col, female_col],
        var_name="sex_col",
        value_name="prevalence",
    )
    long_sex["Sexe"] = long_sex["sex_col"].map({male_col: "Hommes", female_col: "Femmes"})

    sex_latest = long_sex.loc[long_sex["Year"] == latest_year].copy()
    sex_rows = []
    for (continent, sexe), group in sex_latest.groupby(["Continent", "Sexe"]):
        sex_rows.append(
            {
                "Continent": continent,
                "Sexe": sexe,
                "prevalence": float(np.average(group["prevalence"], weights=group[pop_col])),
            }
        )
    sex_continent = pd.DataFrame(sex_rows)
    sex_pivot = sex_continent.pivot(index="Continent", columns="Sexe", values="prevalence").dropna().reset_index()
    sex_pivot["gap_f_h"] = sex_pivot["Femmes"] - sex_pivot["Hommes"]
    sex_pivot = sex_pivot.sort_values("gap_f_h", ascending=False)

    model_df = long_sex[["Year", "Continent", "Sexe", "prevalence"]].copy()
    model_x = pd.get_dummies(model_df[["Year", "Continent", "Sexe"]], columns=["Continent", "Sexe"], drop_first=False)
    x_train, x_test, y_train, y_test = train_test_split(
        model_x,
        model_df["prevalence"],
        test_size=0.25,
        random_state=42,
    )

    models = {
        "Lineaire": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=350, random_state=42, n_jobs=-1),
    }

    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        rows.append(
            {
                "modele": name,
                "mae": float(mean_absolute_error(y_test, pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "r2": float(r2_score(y_test, pred)),
            }
        )

    scores = pd.DataFrame(rows).sort_values("rmse")
    best_row = scores.iloc[0]

    weighted_global_female = np.average(sex_latest.loc[sex_latest["Sexe"] == "Femmes", "prevalence"], weights=sex_latest.loc[sex_latest["Sexe"] == "Femmes", pop_col])
    weighted_global_male = np.average(sex_latest.loc[sex_latest["Sexe"] == "Hommes", "prevalence"], weights=sex_latest.loc[sex_latest["Sexe"] == "Hommes", pop_col])

    return {
        "latest_year": latest_year,
        "continent_latest": continent_latest,
        "extremes": extremes,
        "sex_pivot": sex_pivot,
        "scores": scores,
        "best_model": best_row["modele"],
        "best_rmse": float(best_row["rmse"]),
        "best_r2": float(best_row["r2"]),
        "best_mae": float(best_row["mae"]),
        "continent_gap": float(continent_latest["prevalence"].max() - continent_latest["prevalence"].min()),
        "top_continent": str(continent_latest.iloc[0]["Continent"]),
        "top_continent_value": float(continent_latest.iloc[0]["prevalence"]),
        "bottom_country": str(bottom5.iloc[0]["Entity"]),
        "bottom_country_value": float(bottom5.iloc[0]["prevalence"]),
        "top_country": str(top5.iloc[0]["Entity"]),
        "top_country_value": float(top5.iloc[0]["prevalence"]),
        "global_sex_gap": float(weighted_global_female - weighted_global_male),
        "n_total": int(len(model_df)),
        "feature_names": ["Annee", "Continent", "Sexe"],
    }


def build_sheet_data(res):
    results_rows = []
    for _, row in res["scores"].iterrows():
        results_rows.append(
            {
                "modele": row["modele"],
                "rmse": f"{row['rmse']:.3f}",
                "mae": f"{row['mae']:.3f}",
                "r2": f"{row['r2']:.3f}",
            }
        )

    return {
        "title": "Fiche Scientifique - Hasard de la naissance",
        "authors": "Equipe Data Science - Projet Mental Health",
        "context": (
            "Selon le pays, le continent et le sexe, l'exposition moyenne aux troubles mentaux n'est pas identique. "
            "La question est donc de mesurer dans quelle proportion les conditions de naissance structurent ces ecarts."
        ),
        "objective": (
            "Comparer les differences geographiques et de sexe, puis tester dans quelle mesure des variables de naissance "
            "(continent, sexe, annee) permettent de predire la prevalence des troubles mentaux."
        ),
        "dataset_main": "share-with-mental-and-substance-disorders.csv",
        "dataset_aux": "share-with-mental-or-substance-disorders-by-sex.csv",
        "n_samples": res["n_total"],
        "features": res["feature_names"],
        "models": ["Regression lineaire", "Ridge", "RandomForest"],
        "validation": "Split aleatoire: 75% entrainement, 25% test (dataset disponible sur un millesime exploitable).",
        "packages": ["pandas", "numpy", "matplotlib", "scikit-learn"],
        "results": results_rows,
        "best_model": res["best_model"],
        "interpretation": (
            f"Le continent le plus eleve en {res['latest_year']} est {res['top_continent']} ({res['top_continent_value']:.2f}%). "
            f"L'ecart entre continents atteint {res['continent_gap']:.2f} points et le meilleur modele base uniquement sur annee, continent et sexe atteint R2={res['best_r2']:.3f}."
        ),
        "limits": (
            "Les donnees sont aggregees au niveau pays et ne decrivent pas des trajectoires individuelles. "
            "Le lieu de naissance n'explique donc pas tout et ne doit pas etre interprete comme une causalite stricte."
        ),
        "future_work": (
            "Ajouter revenu, acces aux soins, conflits ou education pour distinguer ce qui releve de la geographie "
            "et ce qui releve du contexte socio-economique."
        ),
    }


def generate_fiche_hasard_naissance():
    base_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = build_dataset()
    sheet_data = build_sheet_data(res)
    OUT_MD.write_text(generate_scientific_sheet(sheet_data), encoding="utf-8")

    fig = plt.figure(figsize=(11.69, 16.54), dpi=300)
    fig.patch.set_facecolor("#0b0f14")
    gs = fig.add_gridspec(
        20,
        12,
        left=0.04,
        right=0.98,
        top=0.985,
        bottom=0.03,
        wspace=0.78,
        hspace=1.35,
    )

    ax_header = fig.add_subplot(gs[0:2, :])
    ax_header.axis("off")
    ax_header.text(0.0, 0.88, "FICHE SCIENTIFIQUE - NAISSANCE ET TROUBLES MENTAUX", fontsize=21, fontweight="bold", color="#f8fafc")
    ax_header.text(0.0, 0.56, "Le hasard de la naissance determine-t-il nos troubles mentaux ?", fontsize=13, color="#cbd5e1")
    ax_header.text(
        0.0,
        0.24,
        "Datasets: prevalence globale + lecture par sexe/continent | Comparaison geographique et predictive",
        fontsize=9.5,
        color="#94a3b8",
    )
    ax_header.text(0.0, 0.06, f"Auteurs: {sheet_data['authors']} | Date: {datetime.now().strftime('%d/%m/%Y')}", fontsize=8.8, color="#94a3b8")
    ax_header.plot([0, 1], [0.02, 0.02], color="#22d3ee", linewidth=3, transform=ax_header.transAxes)

    fig.text(
        0.045,
        0.94,
        "PROBLEMATIQUE - Le lieu et les conditions de naissance expliquent-ils une part mesurable des troubles mentaux ?",
        fontsize=10.8,
        color="#f8fafc",
        fontweight="bold",
        ha="left",
        va="center",
        bbox=dict(facecolor="#0b1220", edgecolor="#22d3ee", boxstyle="round,pad=0.30"),
    )

    ax_card1 = fig.add_subplot(gs[2:4, 0:3])
    add_card(ax_card1, "Continent le plus eleve", res["top_continent"], f"{res['top_continent_value']:.2f}% en {res['latest_year']}", color="#22d3ee")
    ax_card2 = fig.add_subplot(gs[2:4, 3:6])
    add_card(ax_card2, "Ecart entre continents", f"{res['continent_gap']:.2f} pts", "prevalence ponderee par population", color="#34d399")
    ax_card3 = fig.add_subplot(gs[2:4, 6:9])
    add_card(ax_card3, "Gap femmes-hommes", f"{res['global_sex_gap']:.2f} pts", "moyenne mondiale ponderee", color="#f59e0b")
    ax_card4 = fig.add_subplot(gs[2:4, 9:12])
    add_card(ax_card4, "Pouvoir predictif", f"R2={res['best_r2']:.3f}", f"modele {res['best_model']} avec annee/continent/sexe", color="#fb7185")

    ax_cont = fig.add_subplot(gs[4:9, 1:6])
    style_plot(ax_cont)
    panel_title(ax_cont, "1) Prevalence par continent")
    cont = res["continent_latest"]
    ax_cont.bar(cont["Continent"], cont["prevalence"], color="#22d3ee")
    ax_cont.set_ylabel("Prevalence ponderee (%)")
    ax_cont.set_xlabel("")
    ax_cont.tick_params(axis="x", rotation=18, labelsize=7)

    ax_country = fig.add_subplot(gs[4:9, 7:12])
    style_plot(ax_country)
    panel_title(ax_country, "2) Pays extremes au dernier millesime")
    ext = res["extremes"].copy()
    colors = ["#38bdf8"] * 5 + ["#fb7185"] * 5
    ax_country.barh(ext["Entity"], ext["prevalence"], color=colors)
    ax_country.invert_yaxis()
    ax_country.set_xlabel("Prevalence (%)")
    ax_country.tick_params(axis="y", labelsize=6.5)

    ax_gap = fig.add_subplot(gs[10:15, 1:6])
    style_plot(ax_gap)
    panel_title(ax_gap, "3) Effet du sexe selon le continent")
    gap = res["sex_pivot"]
    ax_gap.barh(gap["Continent"], gap["gap_f_h"], color="#f59e0b")
    ax_gap.axvline(0, color="#cbd5e1", linewidth=0.9)
    ax_gap.set_xlabel("Femmes - Hommes (points)")
    ax_gap.tick_params(axis="y", labelsize=7)

    ax_model = fig.add_subplot(gs[10:15, 7:12])
    style_plot(ax_model)
    panel_title(ax_model, "4) Prediction avec variables de naissance")
    scores = res["scores"]
    ax_model.bar(scores["modele"], scores["r2"], color=["#7dd3fc", "#38bdf8", "#0ea5e9"])
    ax_model.set_ylabel("R2 sur test")
    ax_model.set_xlabel("Modeles")
    ax_model.tick_params(axis="x", labelsize=7)
    ax_model.text(
        0.03,
        0.95,
        f"RMSE min={res['best_rmse']:.3f} | MAE={res['best_mae']:.3f}",
        transform=ax_model.transAxes,
        va="top",
        fontsize=8.3,
        color="#d1d5db",
        bbox=dict(facecolor="#0b1220", edgecolor="#334155", boxstyle="round,pad=0.28"),
    )

    ax_method = fig.add_subplot(gs[16:20, 0:6])
    add_text_panel(
        ax_method,
        "Methodologie",
        [
            "1. Calcul des prevalences ponderees par population pour comparer les continents au dernier millesime.",
            "2. Identification des pays extremes pour mesurer l'amplitude brute de l'inegalite geographique.",
            "3. Mesure du gap femmes-hommes par continent pour integrer le sexe comme condition de naissance.",
            f"4. Test predictif avec {', '.join(sheet_data['models'])} sur les seules variables annee, continent et sexe.",
        ],
        wrap_width=44,
    )

    ax_conc = fig.add_subplot(gs[16:20, 6:12])
    add_text_panel(
        ax_conc,
        "Interpretation, limites et perspectives",
        [
            f"Resultat cle: {sheet_data['interpretation']}",
            f"Limites: {sheet_data['limits']}",
            f"Perspectives: {sheet_data['future_work']}",
        ],
        wrap_width=44,
    )

    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    print(f"Fiche naissance Markdown generee: {OUT_MD}")
    print(f"Fiche naissance PNG generee: {OUT_PNG}")
    print(f"Fiche naissance PDF generee: {OUT_PDF}")


if __name__ == "__main__":
    generate_fiche_hasard_naissance()