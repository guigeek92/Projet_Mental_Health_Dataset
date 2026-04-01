from datetime import datetime
import os
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "donnees_brutes"
OUT_DIR = ROOT / "fichiers_generes"
OUT_PNG = OUT_DIR / "fiche_sujet2_poster_scientifique.png"
OUT_PDF = OUT_DIR / "fiche_sujet2_poster_scientifique.pdf"
OUT_MD = OUT_DIR / "fiche_sujet2_scientifique.md"
OUT_PDF_ULTRA = OUT_DIR / "fiche_sujet2_poster_ultra_sobre.pdf"
OUT_PDF_ORAL = OUT_DIR / "fiche_sujet2_poster_oral_visuel.pdf"
OUT_PDF_BRANDED = OUT_DIR / "fiche_sujet2_poster_noms_logo_epf.pdf"
LOGO_EPF = ROOT / "epf - Recherche Images_files" / "epf.png"


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


def short_label(name):
    mapping = {
        "Year": "Annee",
        "Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)": "Schizophrenie",
        "Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)": "Bipolarite",
        "Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)": "Troubles alim.",
        "Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)": "Anxiete",
        "Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)": "Usage drogues",
        "Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)": "Usage alcool",
        "Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)": "Depression",
    }
    return mapping.get(name, name)


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


def add_text_panel(ax, title, lines, wrap_width=78, body_fontsize=9.0, body_linespacing=1.35):
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
    ax.add_patch(
        FancyBboxPatch(
            (0.03, 0.83),
            0.94,
            0.10,
            boxstyle="round,pad=0.005,rounding_size=8",
            linewidth=0,
            facecolor="#1f2937",
            alpha=0.9,
            transform=ax.transAxes,
        )
    )
    ax.text(0.05, 0.885, title, fontsize=12, color="#f8fafc", fontweight="bold", va="center", transform=ax.transAxes)
    wrapped = [textwrap.fill(line, width=wrap_width) for line in lines]
    ax.text(
        0.04,
        0.76,
        "\n".join(wrapped),
        fontsize=body_fontsize,
        color="#d1d5db",
        va="top",
        linespacing=body_linespacing,
        transform=ax.transAxes,
    )


def build_dataset():
    path = DATA / "prevalence-by-mental-and-substance-use-disorder.csv"
    df = pd.read_csv(path).dropna().copy()

    target = "Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)"
    features = [
        "Year",
        "Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)",
        "Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)",
        "Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)",
        "Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)",
        "Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)",
        "Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)",
    ]

    corr_cols = features[1:] + [target]
    corr = df[corr_cols].corr()
    corr.index = [short_label(c) for c in corr.index]
    corr.columns = [short_label(c) for c in corr.columns]

    data = df[features + [target]].copy()
    train_mask = data["Year"] <= 2013
    x_train = data.loc[train_mask, features]
    x_test = data.loc[~train_mask, features]
    y_train = data.loc[train_mask, target]
    y_test = data.loc[~train_mask, target]

    models = {
        "Lineaire": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    }

    rows = []
    predictions = {}
    fitted = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        fitted[name] = model
        predictions[name] = pred
        rows.append(
            {
                "modele": name,
                "mae": float(mean_absolute_error(y_test, pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "r2": float(r2_score(y_test, pred)),
            }
        )

    scores_df = pd.DataFrame(rows).sort_values("rmse")
    best_model = scores_df.iloc[0]["modele"]
    best_row = scores_df.iloc[0]

    rf = fitted["RandomForest"]
    importances = pd.DataFrame(
        {
            "feature": [short_label(c) for c in features],
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    dep_corr = corr["Depression"].drop("Depression").sort_values(ascending=False)

    return {
        "corr": corr,
        "scores": scores_df,
        "best_model": best_model,
        "best_rmse": float(best_row["rmse"]),
        "best_r2": float(best_row["r2"]),
        "best_mae": float(best_row["mae"]),
        "y_true": y_test.values,
        "y_pred": predictions[best_model],
        "importances": importances,
        "top_corr_name": dep_corr.index[0],
        "top_corr_value": float(dep_corr.iloc[0]),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_total": int(len(data)),
        "feature_names": [short_label(c) for c in features if c != "Year"],
    }


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
- Dataset : {data['dataset']}
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
        "title": "Fiche Scientifique - Sujet 2",
        "authors": "Equipe Data Science - Projet Mental Health",
        "context": (
            "La depression est un enjeu majeur de sante publique. Ce jeu de donnees permet "
            "d'analyser les liens entre depression, autres troubles mentaux et troubles lies aux substances."
        ),
        "objective": (
            "Identifier les facteurs les plus explicatifs de la prevalence de la depression "
            "et comparer plusieurs modeles de regression pour une prediction robuste."
        ),
        "dataset": "prevalence-by-mental-and-substance-use-disorder.csv",
        "n_samples": res["n_total"],
        "features": res["feature_names"],
        "models": ["Regression lineaire", "Ridge", "RandomForest"],
        "validation": "Split temporel: entrainement <= 2013, test > 2013.",
        "packages": ["pandas", "numpy", "matplotlib", "scikit-learn"],
        "results": results_rows,
        "best_model": res["best_model"],
        "interpretation": (
            f"Le modele {res['best_model']} obtient la meilleure performance (RMSE={res['best_rmse']:.3f}). "
            f"Le facteur le plus associe a la depression est {res['top_corr_name']} "
            f"avec une correlation de r={res['top_corr_value']:.2f}."
        ),
        "limits": (
            "Les modeles decrivent des associations statistiques mais ne prouvent pas la causalite. "
            "L'absence de variables socio-economiques et sanitaires limite l'explication contextuelle."
        ),
        "future_work": (
            "Integrer des variables externes (PIB, acces aux soins), tester des modeles de boosting "
            "et comparer les performances par zone geographique."
        ),
    }


def generate_ultra_sobre_pdf(res, sheet_data, out_pdf):
    fig = plt.figure(figsize=(11.69, 16.54), dpi=300)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(18, 12, left=0.06, right=0.96, top=0.97, bottom=0.04, wspace=0.5, hspace=0.8)

    ax_title = fig.add_subplot(gs[0:2, :])
    ax_title.axis("off")
    ax_title.text(0.0, 0.72, "Fiche Scientifique - Version Ultra Sobre", fontsize=20, fontweight="bold", color="black")
    ax_title.text(0.0, 0.38, "Sujet 2: prediction de la prevalence de la depression", fontsize=11, color="black")
    ax_title.text(
        0.0,
        0.10,
        f"Auteurs: {sheet_data['authors']} | Date: {datetime.now().strftime('%d/%m/%Y')}",
        fontsize=9,
        color="black",
    )

    ax_resume = fig.add_subplot(gs[2:5, :])
    ax_resume.axis("off")
    resume = [
        f"Donnees: {sheet_data['dataset']} ({res['n_total']} observations)",
        f"Modeles testes: {', '.join(sheet_data['models'])}",
        f"Meilleur modele: {res['best_model']} | RMSE={res['best_rmse']:.3f} | MAE={res['best_mae']:.3f} | R2={res['best_r2']:.3f}",
        f"Variable la plus associee a la depression: {res['top_corr_name']} (r={res['top_corr_value']:.2f})",
    ]
    ax_resume.text(0.0, 0.95, "Resume", fontsize=12, fontweight="bold", va="top")
    ax_resume.text(0.0, 0.76, "\n".join([f"- {x}" for x in resume]), fontsize=10, va="top", linespacing=1.5)

    ax_scores = fig.add_subplot(gs[5:10, 0:6])
    ax_scores.set_facecolor("white")
    ax_scores.grid(True, color="#dddddd")
    sc = res["scores"]
    ax_scores.bar(sc["modele"], sc["rmse"], color="#333333")
    ax_scores.set_title("Comparaison des modeles (RMSE)", loc="left", fontsize=11, fontweight="bold")
    ax_scores.set_ylabel("RMSE")

    ax_imp = fig.add_subplot(gs[5:10, 6:12])
    ax_imp.set_facecolor("white")
    ax_imp.grid(True, color="#dddddd")
    imp = res["importances"].head(7).copy()
    ax_imp.barh(imp["feature"], imp["importance"], color="#666666")
    ax_imp.invert_yaxis()
    ax_imp.set_title("Importance des variables", loc="left", fontsize=11, fontweight="bold")

    ax_txt = fig.add_subplot(gs[10:18, :])
    ax_txt.axis("off")
    bloc = [
        f"Interpretation: {sheet_data['interpretation']}",
        f"Limites: {sheet_data['limits']}",
        f"Perspectives: {sheet_data['future_work']}",
    ]
    ax_txt.text(0.0, 0.98, "Discussion", fontsize=12, fontweight="bold", va="top")
    ax_txt.text(0.0, 0.90, "\n\n".join(bloc), fontsize=10, va="top", linespacing=1.6)

    fig.savefig(out_pdf)
    plt.close(fig)


def generate_branded_pdf_from_png(sheet_data, poster_png, out_pdf):
    fig = plt.figure(figsize=(11.69, 16.54), dpi=300)
    fig.patch.set_facecolor("white")

    ax_top = fig.add_axes([0.04, 0.92, 0.92, 0.07])
    ax_top.axis("off")
    ax_top.text(0.0, 0.65, "Fiche Scientifique - Version Auteurs + Logo Ecole", fontsize=14, fontweight="bold", color="#0f172a")

    authors_real = os.environ.get("AUTEURS_REELS") or os.environ.get("AUTHORS_REAL") or sheet_data["authors"]
    ax_top.text(0.0, 0.20, f"Auteurs: {authors_real}", fontsize=10, color="#334155")

    if LOGO_EPF.exists():
        logo_ax = fig.add_axes([0.84, 0.925, 0.12, 0.06])
        logo_ax.axis("off")
        logo_ax.imshow(mpimg.imread(LOGO_EPF))

    ax_img = fig.add_axes([0.04, 0.04, 0.92, 0.87])
    ax_img.axis("off")
    ax_img.imshow(mpimg.imread(poster_png))

    fig.savefig(out_pdf)
    plt.close(fig)


def generate_fiche_sujet2():
    base_style()
    res = build_dataset()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sheet_data = build_sheet_data(res)
    OUT_MD.write_text(generate_scientific_sheet(sheet_data), encoding="utf-8")

    fig = plt.figure(figsize=(11.69, 16.54), dpi=300)
    fig.patch.set_facecolor("#0b0f14")
    gs = fig.add_gridspec(
        22,
        12,
        left=0.025,
        right=0.975,
        top=0.985,
        bottom=0.03,
        wspace=0.44,
        hspace=1.2,
    )

    ax_header = fig.add_subplot(gs[0:2, :])
    ax_header.axis("off")
    ax_header.text(0.0, 0.88, "FICHE SCIENTIFIQUE - SUJET 2", fontsize=22, fontweight="bold", color="#f8fafc")
    ax_header.text(
        0.0,
        0.56,
        "Quels troubles expliquent le mieux la prevalence de la depression ?",
        fontsize=13,
        color="#cbd5e1",
    )
    ax_header.text(
        0.0,
        0.24,
        "Dataset: prevalence-by-mental-and-substance-use-disorder.csv | Panel pays-annee | Evaluation sur periode recente",
        fontsize=9.5,
        color="#94a3b8",
    )
    ax_header.text(
        0.0,
        0.06,
        f"Auteurs: {sheet_data['authors']} | Date: {datetime.now().strftime('%d/%m/%Y')}",
        fontsize=8.8,
        color="#94a3b8",
    )
    ax_header.plot([0, 1], [0.02, 0.02], color="#22d3ee", linewidth=3, transform=ax_header.transAxes)

    # Force the problem title above the blue line at figure-level.
    fig.text(
        0.042,
        0.94,
        "PROBLEMATIQUE - Quels troubles expliquent le mieux la prevalence de la depression ?",
        fontsize=13.2,
        color="#f8fafc",
        fontweight="bold",
        ha="left",
        va="center",
        bbox=dict(facecolor="#0b1220", edgecolor="#22d3ee", boxstyle="round,pad=0.30"),
    )

    ax_card1 = fig.add_subplot(gs[2:4, 0:3])
    add_card(ax_card1, "Meilleur modele", res["best_model"], "selection selon RMSE test", color="#22d3ee")
    ax_card2 = fig.add_subplot(gs[2:4, 3:6])
    add_card(ax_card2, "RMSE (test)", f"{res['best_rmse']:.3f}", "plus bas = meilleur", color="#34d399")
    ax_card3 = fig.add_subplot(gs[2:4, 6:9])
    add_card(ax_card3, "R2 (test)", f"{res['best_r2']:.3f}", "qualite explicative", color="#f59e0b")
    ax_card4 = fig.add_subplot(gs[2:4, 9:12])
    add_card(
        ax_card4,
        "Plus forte correlation",
        f"{res['top_corr_name']}",
        f"avec depression: r={res['top_corr_value']:.2f}",
        color="#fb7185",
    )

    ax_hm = fig.add_subplot(gs[4:9, 0:5])
    style_plot(ax_hm)
    panel_title(ax_hm, "1) Matrice de correlations")
    hm = ax_hm.imshow(res["corr"].values, cmap="RdYlGn", vmin=-1, vmax=1)
    labels = res["corr"].columns.tolist()
    ax_hm.set_xticks(range(len(labels)))
    ax_hm.set_yticks(range(len(labels)))
    ax_hm.set_xticklabels(labels, rotation=24, ha="right", fontsize=7)
    ax_hm.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax_hm.text(j, i, f"{res['corr'].values[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="#0f172a")
    cbar = fig.colorbar(hm, ax=ax_hm, fraction=0.032, pad=0.03)
    cbar.ax.tick_params(labelsize=7, colors="#cbd5e1")
    cbar.outline.set_edgecolor("#334155")

    ax_rmse = fig.add_subplot(gs[4:9, 6:12])
    style_plot(ax_rmse)
    panel_title(ax_rmse, "2) Comparaison des modeles")
    sc = res["scores"]
    colors = ["#7dd3fc", "#38bdf8", "#0ea5e9"]
    ax_rmse.bar(sc["modele"], sc["rmse"], color=colors[: len(sc)])
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_xlabel("Modeles")
    ax_rmse.tick_params(axis="x", labelsize=7)
    ax_rmse.text(
        0.02,
        0.96,
        f"Train: {res['n_train']} | Test: {res['n_test']} | Total: {res['n_total']}",
        transform=ax_rmse.transAxes,
        va="top",
        fontsize=8.5,
        color="#d1d5db",
        bbox=dict(facecolor="#0b1220", edgecolor="#334155", boxstyle="round,pad=0.28"),
    )

    ax_scatter = fig.add_subplot(gs[10:15, 0:5])
    style_plot(ax_scatter)
    panel_title(ax_scatter, "3) Validation du meilleur modele")
    y_true = res["y_true"]
    y_pred = res["y_pred"]
    ax_scatter.scatter(y_true, y_pred, alpha=0.30, s=11, color="#38bdf8", edgecolors="none")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax_scatter.plot([lo, hi], [lo, hi], "--", color="#fb7185", linewidth=1.2)
    ax_scatter.set_xlabel("Depression reelle (%)")
    ax_scatter.set_ylabel("Depression predite (%)")
    ax_scatter.set_xlim(lo - 0.03, hi + 0.03)
    ax_scatter.set_ylim(lo - 0.03, hi + 0.03)

    ax_imp = fig.add_subplot(gs[10:15, 6:12])
    style_plot(ax_imp)
    panel_title(ax_imp, "4) Variables les plus explicatives")
    imp = res["importances"].head(7).copy()
    ax_imp.barh(imp["feature"], imp["importance"], color="#22d3ee")
    ax_imp.invert_yaxis()
    ax_imp.set_xlabel("Importance relative")
    ax_imp.tick_params(axis="y", labelsize=7)

    ax_method = fig.add_subplot(gs[16:22, 0:6])
    add_text_panel(
        ax_method,
        "Methodologie",
        [
            "1) Preparation des donnees: selection des prevalences par trouble et par annee.",
            f"2) Modeles compares: {', '.join(sheet_data['models'])}.",
            f"3) Strategie de validation: {sheet_data['validation']}",
            f"4) Environnement technique: {', '.join(sheet_data['packages'])}.",
        ],
        wrap_width=44,
        body_fontsize=8.5,
        body_linespacing=1.3,
    )

    ax_conc = fig.add_subplot(gs[16:22, 6:12])
    add_text_panel(
        ax_conc,
        "Interpretation, limites et perspectives",
        [
            f"Resultat principal: {sheet_data['interpretation']}",
            f"Limites de l'analyse: {sheet_data['limits']}",
            f"Pistes d'amelioration: {sheet_data['future_work']}",
        ],
        wrap_width=44,
        body_fontsize=8.15,
        body_linespacing=1.28,
    )

    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PDF_ORAL)
    plt.close(fig)

    generate_ultra_sobre_pdf(res, sheet_data, OUT_PDF_ULTRA)
    generate_branded_pdf_from_png(sheet_data, OUT_PNG, OUT_PDF_BRANDED)

    print(f"Fiche sujet 2 Markdown generee: {OUT_MD}")
    print(f"Fiche sujet 2 PNG generee: {OUT_PNG}")
    print(f"Fiche sujet 2 PDF generee: {OUT_PDF}")
    print(f"Version oral visuelle generee: {OUT_PDF_ORAL}")
    print(f"Version ultra sobre generee: {OUT_PDF_ULTRA}")
    print(f"Version auteurs + logo generee: {OUT_PDF_BRANDED}")


if __name__ == "__main__":
    generate_fiche_sujet2()
