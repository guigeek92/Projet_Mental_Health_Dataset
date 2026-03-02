from pathlib import Path
import textwrap
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "donnees_brutes"
OUT_DIR = ROOT / "fichiers_generes"
OUT_PNG = OUT_DIR / "poster_scientifique_depression_portrait.png"
OUT_PDF = OUT_DIR / "poster_scientifique_depression_portrait.pdf"


def compute_analyse1():
    path = DATA / "mental-and-substance-use-as-share-of-disease.csv"
    col = "DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)"
    df = pd.read_csv(path)[["Entity", "Year", col]].dropna().copy()

    yearly = df.groupby("Year", as_index=False)[col].mean().sort_values("Year")
    x = yearly["Year"].values
    y = yearly[col].values
    slope, intercept = np.polyfit(x, y, 1)
    trend = slope * x + intercept
    corr = np.corrcoef(x, y)[0, 1]

    top10 = (
        df.groupby("Entity", as_index=False)[col]
        .mean()
        .sort_values(col, ascending=False)
        .head(10)
    )

    return {
        "yearly": yearly,
        "trend": trend,
        "top10": top10,
        "mean": float(df[col].mean()),
        "median": float(df[col].median()),
        "corr": float(corr),
        "label": col,
    }


def compute_analyse2():
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
    corr_cols = [
        "Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)",
        "Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)",
        "Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)",
        "Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)",
        "Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)",
        target,
    ]
    short = {
        corr_cols[0]: "Bipolar",
        corr_cols[1]: "Troubles alim.",
        corr_cols[2]: "Anxiete",
        corr_cols[3]: "Usage drogues",
        corr_cols[4]: "Usage alcool",
        corr_cols[5]: "Depression",
    }

    corr = df[corr_cols].corr()
    corr.index = [short[c] for c in corr.index]
    corr.columns = [short[c] for c in corr.columns]

    d = df[features + [target]].copy()
    train_mask = d["Year"] <= 2013
    x_train, x_test = d.loc[train_mask, features], d.loc[~train_mask, features]
    y_train, y_test = d.loc[train_mask, target], d.loc[~train_mask, target]

    models = {
        "Lineaire": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    }

    scores = []
    preds = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        p = model.predict(x_test)
        preds[name] = p
        scores.append({
            "modele": name,
            "mae": float(mean_absolute_error(y_test, p)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, p))),
            "r2": float(r2_score(y_test, p)),
        })

    scores_df = pd.DataFrame(scores).sort_values("rmse")
    best_name = scores_df.iloc[0]["modele"]
    y_best = preds[best_name]

    return {
        "corr": corr,
        "scores": scores_df,
        "best_name": best_name,
        "y_true": y_test.values,
        "y_pred": y_best,
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
    }


def compute_analyse3():
    path = DATA / "prevalence-of-depression-males-vs-females.csv"
    df = pd.read_csv(path)

    col_m = "Prevalence - Depressive disorders - Sex: Male - Age: Age-standardized (Percent)"
    col_f = "Prevalence - Depressive disorders - Sex: Female - Age: Age-standardized (Percent)"
    col_pop = "Population (historical estimates)"

    d = df[["Entity", "Year", col_m, col_f, col_pop]].dropna().copy()
    by_year = d.groupby("Year", as_index=False)[[col_m, col_f]].mean().sort_values("Year")
    by_year["gap"] = by_year[col_f] - by_year[col_m]

    corr_mf = float(np.corrcoef(d[col_m], d[col_f])[0, 1])

    features = ["Year", col_m, col_pop]
    target = col_f
    dm = d[features + [target]].copy()
    train_mask = dm["Year"] <= 2013
    x_train, x_test = dm.loc[train_mask, features], dm.loc[~train_mask, features]
    y_train, y_test = dm.loc[train_mask, target], dm.loc[~train_mask, target]

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    imp = pd.DataFrame({
        "feature": ["Annee", "Depression hommes", "Population"],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "by_year": by_year,
        "corr_mf": corr_mf,
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "importance": imp,
    }


def panel_title(ax, title):
    wrapped_title = textwrap.fill(title, width=30)
    ax.text(
        0.015,
        0.985,
        wrapped_title,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color="#e5e7eb",
        va="top",
        ha="left",
        bbox=dict(facecolor="#111827", edgecolor="#334155", linewidth=0.8, boxstyle="round,pad=0.22"),
        zorder=40,
        clip_on=True,
    )


def style_axes(ax):
    ax.set_facecolor("#111827")
    ax.patch.set_edgecolor("#1f2937")
    ax.patch.set_linewidth(1.2)
    for spine in ax.spines.values():
        spine.set_color("#374151")
    ax.tick_params(colors="#cbd5e1", labelsize=8)
    ax.xaxis.label.set_color("#e5e7eb")
    ax.yaxis.label.set_color("#e5e7eb")
    ax.grid(True, color="#334155", alpha=0.25, linewidth=0.6)


def add_reasoning(ax, text):
    wrapped = "\n".join(textwrap.fill(line, width=54) for line in text.split("\n"))
    ax.text(
        0.98,
        0.02,
        wrapped,
        transform=ax.transAxes,
        fontsize=7.4,
        color="#d1d5db",
        va="bottom",
        ha="right",
        bbox=dict(facecolor="#0b1220", alpha=0.9, edgecolor="#1f2937", boxstyle="round,pad=0.35"),
    )


def add_section_box(ax, title, lines):
    ax.axis("off")
    ax.set_facecolor("#111827")
    ax.patch.set_edgecolor("#1f2937")
    ax.patch.set_linewidth(1.2)
    ax.text(0.02, 0.92, title, fontsize=12, color="#f9fafb", fontweight="bold", va="top")
    ax.text(0.02, 0.74, "\n".join(lines), fontsize=9.4, color="#d1d5db", va="top", linespacing=1.45)


def add_group_frame(fig, axes, title, accent="#22d3ee", label_x_shift=0.0, label_x_abs=None):
    boxes = [ax.get_position() for ax in axes]
    left = max(0.02, min(b.x0 for b in boxes) - 0.008)
    right = min(0.985, max(b.x1 for b in boxes) + 0.008)
    bottom = max(0.02, min(b.y0 for b in boxes) - 0.012)
    top = min(0.985, max(b.y1 for b in boxes) + 0.010)

    border = Rectangle(
        (left, bottom),
        right - left,
        top - bottom,
        transform=fig.transFigure,
        facecolor="none",
        edgecolor="#334155",
        linewidth=1.1,
        zorder=-8,
    )
    fig.add_artist(border)

    accent_line = Line2D(
        [left, right],
        [top, top],
        transform=fig.transFigure,
        color=accent,
        linewidth=1.4,
        zorder=-7,
    )
    fig.add_artist(accent_line)

    label_x = max(0.02, min(0.965, left + 0.010 + label_x_shift))
    if label_x_abs is not None:
        label_x = max(0.02, min(0.965, label_x_abs))

    title_wrapped = textwrap.fill(title, width=34)

    fig.text(
        label_x,
        min(0.994, top + 0.003),
        title_wrapped,
        color="#e2e8f0",
        fontsize=8.9,
        fontweight="bold",
        va="bottom",
        ha="left",
        zorder=30,
        bbox=dict(facecolor="#0b0f14", edgecolor=accent, linewidth=0.8, boxstyle="round,pad=0.20"),
    )


def add_poster_guides(fig):
    vline = Line2D([0.505, 0.505], [0.04, 0.97], transform=fig.transFigure, color="#1f2937", linewidth=1.0, zorder=-20)
    fig.add_artist(vline)
    for y in [0.79, 0.615, 0.44, 0.265, 0.12]:
        hline = Line2D([0.04, 0.98], [y, y], transform=fig.transFigure, color="#172030", linewidth=0.8, zorder=-20)
        fig.add_artist(hline)


def shorten_labels(series, max_len=22):
    return [s if len(s) <= max_len else (s[: max_len - 1] + "…") for s in series]


def generate_poster():
    a1 = compute_analyse1()
    a2 = compute_analyse2()
    a3 = compute_analyse3()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.style.use("default")
    fig = plt.figure(figsize=(11.69, 16.54), dpi=300)  # A3 portrait
    fig.patch.set_facecolor("#0b0f14")
    gs = fig.add_gridspec(
        6,
        2,
        height_ratios=[0.9, 1.35, 1.35, 1.35, 1.25, 1.1],
        hspace=0.62,
        wspace=0.34,
        left=0.045,
        right=0.98,
        top=0.985,
        bottom=0.03,
    )

    ax_t = fig.add_subplot(gs[0, :])
    add_section_box(
        ax_t,
        "POSTER SCIENTIFIQUE — DEPRESSION ET SANTE MENTALE (FORMAT PORTRAIT)",
        [
            "Problematique: comprendre l'evolution de la depression et identifier les variables les plus utiles pour la prediction.",
            "Definition utile: DALYs = annees de vie en bonne sante perdues (plus la valeur est haute, plus l'impact sanitaire est fort).",
            "Demarche: 1) Observation globale, 2) Liens entre troubles, 3) Modelisation, 4) Analyse femmes-hommes, 5) Interpretation.",
        ],
    )

    ax11 = fig.add_subplot(gs[1, 0])
    style_axes(ax11)
    panel_title(ax11, "1) Evolution globale (DALYs)")
    ax11.plot(a1["yearly"]["Year"], a1["yearly"][a1["label"]], color="#2563eb", label="Moyenne annuelle")
    ax11.plot(a1["yearly"]["Year"], a1["trend"], "--", color="#dc2626", label="Tendance")
    ax11.set_xlabel("Annee")
    ax11.set_ylabel("DALYs (%)")
    ax11.legend(fontsize=8, facecolor="#111827", edgecolor="#334155", labelcolor="#e5e7eb")
    ax11.text(
        0.02,
        0.16,
        f"Moyenne={a1['mean']:.3f} | Mediane={a1['median']:.3f} | Lien temps-niveau={a1['corr']:.3f}",
        transform=ax11.transAxes,
        fontsize=8,
        color="#e5e7eb",
        bbox=dict(facecolor="#0b1220", alpha=0.88, edgecolor="#1f2937"),
    )
    add_reasoning(
        ax11,
        "Question: la charge globale evolue-t-elle ?\nResultat: la courbe montre la dynamique annuelle.\nInterpretation: la ligne pointillee resume la tendance generale.",
    )

    ax12 = fig.add_subplot(gs[1, 1])
    style_axes(ax12)
    panel_title(ax12, "2) Entites les plus touchees")
    entities_short = shorten_labels(a1["top10"]["Entity"].tolist(), max_len=22)
    ax12.barh(entities_short, a1["top10"][a1["label"]], color="#60a5fa")
    ax12.invert_yaxis()
    ax12.set_xlabel("DALYs moyens (%)")
    add_reasoning(
        ax12,
        "Question: ou l'impact est-il le plus fort ?\nResultat: comparaison des niveaux moyens.\nInterpretation: cible les contextes prioritaires pour la sante mentale.",
    )

    ax21 = fig.add_subplot(gs[2, 0])
    style_axes(ax21)
    panel_title(ax21, "3) Liens troubles / depression")
    corr = a2["corr"].values
    labels = a2["corr"].columns.tolist()
    im = ax21.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    ax21.set_aspect("auto")
    ax21.set_anchor("W")
    cax = inset_axes(
        ax21,
        width="2.8%",
        height="74%",
        loc="lower left",
        bbox_to_anchor=(1.015, 0.12, 1, 1),
        bbox_transform=ax21.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(colors="#cbd5e1", labelsize=8)
    cbar.outline.set_edgecolor("#334155")
    ax21.set_xticks(range(len(labels)))
    ax21.set_yticks(range(len(labels)))
    ax21.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
    ax21.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax21.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7, color="#111827")
    add_reasoning(
        ax21,
        "Question: quels troubles vont souvent avec la depression ?\nResultat: plus la case est vive, plus le lien est fort.\nInterpretation: cela guide le choix des variables de prediction.",
    )

    ax22 = fig.add_subplot(gs[2, 1])
    style_axes(ax22)
    panel_title(ax22, "4) Comparaison des modeles de prediction")
    sc = a2["scores"]
    ax22.bar(sc["modele"], sc["rmse"], color=["#93c5fd", "#60a5fa", "#2563eb"])
    ax22.set_ylabel("RMSE (plus petit = mieux)")
    ax22.margins(x=0.12)
    ax22.text(
        0.02,
        0.94,
        f"Train={a2['train_size']} | Test={a2['test_size']}\nMeilleur={a2['best_name']}",
        transform=ax22.transAxes,
        fontsize=9,
        va="top",
        color="#e5e7eb",
        bbox=dict(facecolor="#0b1220", alpha=0.88, edgecolor="#1f2937"),
    )
    add_reasoning(
        ax22,
        "Question: quel modele predit le mieux la depression ?\nResultat: on compare l'erreur RMSE.\nInterpretation: la barre la plus basse indique la meilleure generalisation.",
    )

    ax31 = fig.add_subplot(gs[3, 0])
    style_axes(ax31)
    panel_title(ax31, "5) Qualite predictive (reel vs predit)")
    ax31.scatter(a2["y_true"], a2["y_pred"], alpha=0.25, s=10, color="#2563eb")
    lim_min = float(min(np.min(a2["y_true"]), np.min(a2["y_pred"])))
    lim_max = float(max(np.max(a2["y_true"]), np.max(a2["y_pred"])))
    ax31.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="#dc2626", linewidth=1.2)
    ax31.set_xlabel("Valeurs reelles")
    ax31.set_ylabel("Valeurs predites")
    ax31.set_xlim(lim_min - 0.03, lim_max + 0.03)
    ax31.set_ylim(lim_min - 0.03, lim_max + 0.03)
    add_reasoning(
        ax31,
        "Question: le modele est-il fiable point par point ?\nResultat: les points proches de la diagonale sont bien predits.\nInterpretation: un nuage serre autour de la ligne indique une bonne precision.",
    )

    ax32 = fig.add_subplot(gs[3, 1])
    style_axes(ax32)
    panel_title(ax32, "6) Depression hommes vs femmes")
    by = a3["by_year"]
    ax32.plot(by["Year"], by.iloc[:, 1], label="Hommes", color="#2563eb")
    ax32.plot(by["Year"], by.iloc[:, 2], label="Femmes", color="#f59e0b")
    ax32.set_xlabel("Annee")
    ax32.set_ylabel("Prevalence (%)")
    ax32.legend(fontsize=8, facecolor="#111827", edgecolor="#334155", labelcolor="#e5e7eb")
    ax32.text(
        0.02,
        0.16,
        f"Lien hommes-femmes={a3['corr_mf']:.3f}",
        transform=ax32.transAxes,
        fontsize=8,
        color="#e5e7eb",
        bbox=dict(facecolor="#0b1220", alpha=0.88, edgecolor="#1f2937"),
    )
    add_reasoning(
        ax32,
        "Question: les tendances sont-elles similaires selon le sexe ?\nResultat: comparaison directe des deux courbes.\nInterpretation: un ecart durable suggere des facteurs differencies a investiguer.",
    )

    ax41 = fig.add_subplot(gs[4, 0])
    style_axes(ax41)
    panel_title(ax41, "7) Ecart femmes - hommes")
    ax41.plot(by["Year"], by["gap"], color="#ef4444")
    ax41.axhline(0, color="#cbd5e1", linewidth=0.8)
    ax41.set_xlabel("Annee")
    ax41.set_ylabel("Gap F-M (points)")
    ax41.text(
        0.02,
        0.92,
        "Au-dessus de 0: prevalence moyenne plus elevee chez les femmes.",
        transform=ax41.transAxes,
        fontsize=8,
        color="#e5e7eb",
        va="top",
        bbox=dict(facecolor="#0b1220", alpha=0.88, edgecolor="#1f2937"),
    )
    add_reasoning(
        ax41,
        "Question: comment evolue l'ecart femmes-hommes ?\nResultat: la courbe quantifie l'ecart annuel.\nInterpretation: au-dessus de 0, les femmes sont plus touchees en moyenne.",
    )

    ax42 = fig.add_subplot(gs[4, 1])
    style_axes(ax42)
    panel_title(ax42, "8) Variables utiles pour predire la depression feminine")
    imp = a3["importance"]
    imp_labels = shorten_labels(imp["feature"].tolist(), max_len=24)
    ax42.barh(imp_labels, imp["importance"], color="#60a5fa")
    ax42.invert_yaxis()
    ax42.set_xlabel("Importance relative")
    ax42.text(
        0.02,
        0.92,
        f"RMSE={a3['rmse']:.3f} | R²={a3['r2']:.3f}",
        transform=ax42.transAxes,
        fontsize=8,
        color="#e5e7eb",
        va="top",
        bbox=dict(facecolor="#0b1220", alpha=0.88, edgecolor="#1f2937"),
    )
    add_reasoning(
        ax42,
        "Question: quelles variables sont les plus informatives ?\nResultat: barres plus longues = poids plus important.\nInterpretation: aide a expliquer le comportement du modele.",
    )

    ax_c = fig.add_subplot(gs[5, :])
    add_section_box(
        ax_c,
        "CONCLUSION ET RAISONNEMENT FINAL",
        [
            "1) Le niveau global de sante mentale presente une dynamique temporelle mesurable.",
            "2) La depression est liee a d'autres troubles, ce qui justifie une approche multi-variables.",
            "3) Les modeles non lineaires ameliorent la prediction, surtout sur des relations complexes.",
            "4) L'analyse par sexe complete l'interpretation et aide au ciblage des actions de prevention.",
            "5) Ces resultats montrent des associations statistiques, pas des preuves de causalite.",
        ],
    )

    add_poster_guides(fig)
    add_group_frame(fig, [ax_t], "Contexte et objectif", accent="#38bdf8")
    add_group_frame(fig, [ax11, ax12], "Partie A — Charge globale et territoires", accent="#60a5fa")
    add_group_frame(fig, [ax21], "Partie B — Liens troubles/depression", accent="#34d399", label_x_abs=0.055)
    add_group_frame(fig, [ax31, ax32], "Partie C — Validation du modele et comparaison par sexe", accent="#f59e0b")
    add_group_frame(fig, [ax41, ax42], "Partie D — Ecart F-H et variables explicatives", accent="#f97316")
    add_group_frame(fig, [ax_c], "Synthese", accent="#a78bfa")

    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    print(f"Poster scientifique PNG genere: {OUT_PNG}")
    print(f"Poster scientifique PDF genere: {OUT_PDF}")


if __name__ == "__main__":
    generate_poster()
