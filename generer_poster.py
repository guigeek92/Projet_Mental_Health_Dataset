import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "donnees_brutes"
OUT_HTML = ROOT / "poster_mental_health.html"


def safe_float(value):
    if value is None:
        return None
    try:
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return None
        return round(val, 6)
    except Exception:
        return None


def model_scores(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": safe_float(mae),
        "rmse": safe_float(rmse),
        "r2": safe_float(r2),
    }


def analyse_1():
    path = DATA / "mental-and-substance-use-as-share-of-disease.csv"
    df = pd.read_csv(path)

    dalys_col = "DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)"
    keep = ["Entity", "Year", dalys_col]
    df = df[keep].dropna().copy()

    yearly = df.groupby("Year", as_index=False)[dalys_col].mean().sort_values("Year")
    x = yearly["Year"].values
    y = yearly[dalys_col].values

    corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
    slope, intercept = np.polyfit(x, y, 1)
    trend = slope * x + intercept
    r2_global = 1 - np.sum((y - trend) ** 2) / np.sum((y - y.mean()) ** 2)

    top10 = (
        df.groupby("Entity", as_index=False)[dalys_col]
        .mean()
        .sort_values(dalys_col, ascending=False)
        .head(10)
    )

    return {
        "title": "Analyse 1 — Évolution DALYs (troubles mentaux)",
        "summary": {
            "moyenne_dalys": safe_float(df[dalys_col].mean()),
            "mediane_dalys": safe_float(df[dalys_col].median()),
            "pearson_year_dalys": safe_float(corr),
            "r2_tendance": safe_float(r2_global),
        },
        "line": {
            "labels": yearly["Year"].astype(int).tolist(),
            "dalys": [safe_float(v) for v in yearly[dalys_col].tolist()],
            "trend": [safe_float(v) for v in trend.tolist()],
        },
        "bar": {
            "labels": top10["Entity"].tolist(),
            "values": [safe_float(v) for v in top10[dalys_col].tolist()],
        },
        "text": [
            "La moyenne annuelle mondiale des DALYs liés aux troubles mentaux est suivie sur toute la période.",
            "La tendance linéaire permet de visualiser l'évolution de fond, sans conclure à une causalité.",
            "Le top 10 met en évidence les entités avec la charge moyenne la plus élevée.",
        ],
    }


def analyse_2():
    path = DATA / "prevalence-by-mental-and-substance-use-disorder.csv"
    df = pd.read_csv(path).dropna().copy()

    col_cible = "Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)"
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
        col_cible,
    ]

    short = {
        corr_cols[0]: "Bipolar",
        corr_cols[1]: "Eating",
        corr_cols[2]: "Anxiety",
        corr_cols[3]: "Drug use",
        corr_cols[4]: "Alcohol use",
        corr_cols[5]: "Depression",
    }

    corr = df[corr_cols].corr()
    corr.index = [short[c] for c in corr.index]
    corr.columns = [short[c] for c in corr.columns]

    data_model = df[features + [col_cible]].copy()
    mask_train = data_model["Year"] <= 2013
    x_train = data_model.loc[mask_train, features]
    y_train = data_model.loc[mask_train, col_cible]
    x_test = data_model.loc[~mask_train, features]
    y_test = data_model.loc[~mask_train, col_cible]

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge_alpha_1": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    }

    rows = []
    preds = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metrics = model_scores(y_test, y_pred)
        rows.append({"modele": name, **metrics})
        preds[name] = y_pred

    results_df = pd.DataFrame(rows).sort_values("rmse")
    best_name = results_df.iloc[0]["modele"]

    best_model = models[best_name]
    best_model.fit(x_train, y_train)
    importances = None
    if hasattr(best_model, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": features,
            "importance": best_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importances = {
            "labels": imp["feature"].tolist(),
            "values": [safe_float(v) for v in imp["importance"].tolist()],
        }

    test_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": preds[best_name],
        "year": x_test["Year"].values,
    })

    return {
        "title": "Analyse 2 — Modélisation de la dépression",
        "summary": {
            "train_size": int(len(x_train)),
            "test_size": int(len(x_test)),
            "best_model": best_name,
            "best_rmse": safe_float(results_df.iloc[0]["rmse"]),
            "best_r2": safe_float(results_df.iloc[0]["r2"]),
        },
        "corr": {
            "labels": corr.columns.tolist(),
            "matrix": [[safe_float(v) for v in row] for row in corr.values.tolist()],
        },
        "models": results_df.to_dict(orient="records"),
        "scatter": {
            "x": [safe_float(v) for v in test_df["y_true"].tolist()],
            "y": [safe_float(v) for v in test_df["y_pred"].tolist()],
        },
        "importance": importances,
        "text": [
            "Le split temporel est conservé pour éviter la fuite d'information.",
            "La comparaison de modèles met en évidence le meilleur compromis erreur/précision.",
            "Le nuage réel vs prédit permet de diagnostiquer la qualité globale du modèle.",
        ],
    }


def analyse_3():
    path = DATA / "prevalence-of-depression-males-vs-females.csv"
    df = pd.read_csv(path)

    col_male = "Prevalence - Depressive disorders - Sex: Male - Age: Age-standardized (Percent)"
    col_female = "Prevalence - Depressive disorders - Sex: Female - Age: Age-standardized (Percent)"
    col_pop = "Population (historical estimates)"

    keep = ["Entity", "Year", col_male, col_female, col_pop]
    df = df[keep].dropna().copy()

    by_year = (
        df.groupby("Year", as_index=False)[[col_male, col_female]]
        .mean()
        .sort_values("Year")
    )
    by_year["gap_f_minus_m"] = by_year[col_female] - by_year[col_male]

    corr_mf = np.corrcoef(df[col_male], df[col_female])[0, 1]

    features = ["Year", col_male, col_pop]
    col_cible = col_female
    data_model = df[features + [col_cible]].copy()
    mask_train = data_model["Year"] <= 2013

    x_train = data_model.loc[mask_train, features]
    y_train = data_model.loc[mask_train, col_cible]
    x_test = data_model.loc[~mask_train, features]
    y_test = data_model.loc[~mask_train, col_cible]

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    metrics = model_scores(y_test, y_pred)

    importances = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "title": "Analyse 3 — Dépression hommes vs femmes",
        "summary": {
            "corr_h_f": safe_float(corr_mf),
            "rf_rmse": metrics["rmse"],
            "rf_r2": metrics["r2"],
        },
        "line": {
            "labels": by_year["Year"].astype(int).tolist(),
            "male": [safe_float(v) for v in by_year[col_male].tolist()],
            "female": [safe_float(v) for v in by_year[col_female].tolist()],
            "gap": [safe_float(v) for v in by_year["gap_f_minus_m"].tolist()],
        },
        "importance": {
            "labels": importances["feature"].tolist(),
            "values": [safe_float(v) for v in importances["importance"].tolist()],
        },
        "text": [
            "La comparaison temporelle permet de suivre l'écart moyen femmes-hommes.",
            "La corrélation hommes/femmes mesure l'alignement global des trajectoires.",
            "L'importance des variables indique les facteurs les plus utiles à la prédiction féminine.",
        ],
    }


def build_html(payload):
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang=\"fr\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Poster dynamique - Projet Mental Health</title>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
  <style>
    :root {{
      --bg: #f7f9fc;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --primary: #2563eb;
      --border: #e5e7eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Roboto, Arial, sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
      color: white;
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 18px;
    }}
    .hero h1 {{ margin: 0 0 8px 0; font-size: 30px; }}
    .hero p {{ margin: 0; opacity: .95; }}
    .tabs {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0 18px 0; }}
    .tab-btn {{
      border: 1px solid var(--border);
      background: var(--card);
      color: var(--text);
      padding: 10px 14px;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 600;
    }}
    .tab-btn.active {{
      background: var(--primary);
      color: white;
      border-color: var(--primary);
    }}
    .panel {{ display: none; }}
    .panel.active {{ display: block; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
    }}
    .kpi-label {{ color: var(--muted); font-size: 13px; }}
    .kpi-value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    .card h3 {{ margin: 0 0 10px 0; font-size: 17px; }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .row.one {{ grid-template-columns: 1fr; }}
    @media (max-width: 900px) {{ .row {{ grid-template-columns: 1fr; }} }}
    .table-wrap {{ overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--border); padding: 8px; text-align: left; }}
    .note {{ margin: 0; color: var(--muted); line-height: 1.45; }}
    .heat-cell {{ text-align: center; font-weight: 600; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class=\"container\">
    <section class=\"hero\">
      <h1>Poster dynamique — Santé mentale & dépression</h1>
      <p>Compte rendu automatisé des scripts analyse1, analyse2 et analyse3 avec vocabulaire simplifié et focus sur la dépression.</p>
    </section>

    <div class=\"tabs\">
      <button class=\"tab-btn active\" data-panel=\"a1\">Analyse 1</button>
      <button class=\"tab-btn\" data-panel=\"a2\">Analyse 2</button>
      <button class=\"tab-btn\" data-panel=\"a3\">Analyse 3</button>
    </div>

    <section id=\"a1\" class=\"panel active\"></section>
    <section id=\"a2\" class=\"panel\"></section>
    <section id=\"a3\" class=\"panel\"></section>
  </div>

  <script>
    const DATA = {data_json};

    function fmt(v) {{
      if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
      return Number(v).toFixed(3);
    }}

    function kpiCard(label, value) {{
      return `<div class=\"card\"><div class=\"kpi-label\">${{label}}</div><div class=\"kpi-value\">${{value}}</div></div>`;
    }}

    function notesCard(lines) {{
      return `<div class=\"card\"><h3>Explication</h3>${{lines.map(l => `<p class=\"note\">• ${{l}}</p>`).join("")}}</div>`;
    }}

    function explainCard(title, lines) {{
      return `<div class=\"card\"><h3>${{title}}</h3>${{lines.map(l => `<p class=\"note\">• ${{l}}</p>`).join("")}}</div>`;
    }}

    function renderA1() {{
      const a = DATA.analyse1;
      const root = document.getElementById("a1");
      root.innerHTML = `
        <div class=\"grid\">
          ${{kpiCard("Niveau moyen (DALYs*)", fmt(a.summary.moyenne_dalys))}}
          ${{kpiCard("Valeur centrale (DALYs*)", fmt(a.summary.mediane_dalys))}}
          ${{kpiCard("Lien temps-niveau", fmt(a.summary.pearson_year_dalys))}}
          ${{kpiCard("Qualité de la tendance", fmt(a.summary.r2_tendance))}}
        </div>
        <div class=\"row\">
          <div class=\"card\"><h3>Évolution moyenne annuelle</h3><canvas id=\"a1-line\"></canvas></div>
          <div class=\"card\"><h3>Top 10 entités (DALYs moyens)</h3><canvas id=\"a1-bar\"></canvas></div>
        </div>
        <div class=\"row\">
          ${{explainCard("Comment lire les graphiques", [
            "DALYs* représente la perte de bonne santé due aux troubles mentaux (plus haut = situation globale plus lourde).",
            "Courbe bleue: évolution moyenne dans le temps; courbe rouge: direction générale (hausse ou baisse).",
            "Barres horizontales: les 10 pays/entités les plus touchés en moyenne."
          ])}}
          ${{explainCard("Indicateurs utilisés", [
            "*DALYs: nombre d'années de vie en bonne santé perdues (indicateur global de charge de maladie).",
            "Niveau moyen / valeur centrale: donnent une idée du niveau typique sur l'ensemble des données.",
            "Lien temps-niveau: proche de +1 = tendance à la hausse avec les années; proche de 0 = pas de tendance claire.",
            "Qualité de la tendance: proche de 1 = la ligne rouge résume bien l'évolution générale."
          ])}}
        </div>
        <div class=\"row\">
          ${{explainCard("Comment lire les graphiques", [
            "Matrice de corrélation: montre quels troubles évoluent ensemble avec la dépression (plus foncé = lien plus fort).",
            "Table des modèles: compare quelle méthode prédit le mieux la dépression.",
            "Nuage réel vs prédit: plus les points sont proches de la diagonale, plus la prédiction est juste.",
            "Importance des variables: indique quelles informations aident le plus à prédire la dépression."
          ])}}
          ${{explainCard("Indicateurs utilisés", [
            "MAE: écart moyen entre la valeur réelle et la valeur prédite (plus petit = mieux).",
            "RMSE: ressemble au MAE mais pénalise davantage les grosses erreurs (plus petit = mieux).",
            "R²: note de qualité globale du modèle (proche de 1 = très bon, proche de 0 = faible).",
            "Train/Test: on entraîne sur les années anciennes et on vérifie sur les années récentes pour simuler une vraie prévision."
          ])}}
        </div>
        <div class=\"row one\">${{notesCard(a.text)}}</div>
      `;

      new Chart(document.getElementById("a1-line"), {{
        type: "line",
        data: {{
          labels: a.line.labels,
          datasets: [
            {{ label: "DALYs moyens", data: a.line.dalys, borderColor: "#2563eb", tension: 0.2, pointRadius: 2 }},
            {{ label: "Tendance", data: a.line.trend, borderColor: "#ef4444", borderDash: [6, 4], pointRadius: 0 }}
          ]
        }},
        options: {{ responsive: true, plugins: {{ legend: {{ position: "bottom" }} }} }}
      }});

      new Chart(document.getElementById("a1-bar"), {{
        type: "bar",
        data: {{ labels: a.bar.labels, datasets: [{{ label: "DALYs", data: a.bar.values, backgroundColor: "#60a5fa" }}] }},
        options: {{ responsive: true, indexAxis: "y", plugins: {{ legend: {{ display: false }} }} }}
      }});
    }}

    function heatColor(v) {{
      const n = Math.max(-1, Math.min(1, Number(v)));
      if (n >= 0) {{
        const alpha = Math.abs(n) * 0.75 + 0.15;
        return `rgba(16,185,129,${{alpha}})`;
      }}
      const alpha = Math.abs(n) * 0.75 + 0.15;
      return `rgba(239,68,68,${{alpha}})`;
    }}

    function renderCorrTable(labels, matrix) {{
      let h = `<div class=\"table-wrap\"><table><thead><tr><th></th>${{labels.map(l => `<th>${{l}}</th>`).join("")}}</tr></thead><tbody>`;
      labels.forEach((rowName, i) => {{
        h += `<tr><th>${{rowName}}</th>`;
        matrix[i].forEach(v => {{
          h += `<td class=\"heat-cell\" style=\"background:${{heatColor(v)}}\">${{fmt(v)}}</td>`;
        }});
        h += "</tr>";
      }});
      h += "</tbody></table></div>";
      return h;
    }}

    function renderA2() {{
      const a = DATA.analyse2;
      const root = document.getElementById("a2");
      root.innerHTML = `
        <div class=\"grid\">
          ${{kpiCard("Train", a.summary.train_size)}}
          ${{kpiCard("Test", a.summary.test_size)}}
          ${{kpiCard("Meilleur modèle", a.summary.best_model)}}
          ${{kpiCard("Précision globale (R²)", fmt(a.summary.best_r2))}}
        </div>
        <div class=\"row\">
          <div class=\"card\"><h3>Matrice de corrélation des troubles</h3>${{renderCorrTable(a.corr.labels, a.corr.matrix)}}</div>
          <div class=\"card\"><h3>Comparaison des modèles</h3><div class=\"table-wrap\"><table id=\"a2-model-table\"></table></div></div>
        </div>
        <div class=\"row\">
          <div class=\"card\"><h3>Réel vs Prédit (meilleur modèle)</h3><canvas id=\"a2-scatter\"></canvas></div>
          <div class=\"card\"><h3>Importance des variables</h3><canvas id=\"a2-imp\"></canvas></div>
        </div>
        <div class=\"row one\">${{notesCard(a.text)}}</div>
      `;

      const headers = ["modele", "mae", "rmse", "r2"];
      const table = document.getElementById("a2-model-table");
      table.innerHTML = `<thead><tr>${{headers.map(h => `<th>${{h.toUpperCase()}}</th>`).join("")}}</tr></thead>` +
        `<tbody>${{a.models.map(r => `<tr>${{headers.map(h => `<td>${{h === "modele" ? r[h] : fmt(r[h])}}</td>`).join("")}}</tr>`).join("")}}</tbody>`;

      new Chart(document.getElementById("a2-scatter"), {{
        type: "scatter",
        data: {{
          datasets: [{{
            label: "Points test",
            data: a.scatter.x.map((x, i) => ({{x, y: a.scatter.y[i]}})),
            pointBackgroundColor: "#2563eb",
            pointRadius: 3,
          }}]
        }},
        options: {{ responsive: true, plugins: {{ legend: {{ position: "bottom" }} }} }}
      }});

      if (a.importance) {{
        new Chart(document.getElementById("a2-imp"), {{
          type: "bar",
          data: {{ labels: a.importance.labels, datasets: [{{ label: "Importance", data: a.importance.values, backgroundColor: "#34d399" }}] }},
          options: {{ responsive: true, indexAxis: "y", plugins: {{ legend: {{ display: false }} }} }}
        }});
      }}
    }}

    function renderA3() {{
      const a = DATA.analyse3;
      const root = document.getElementById("a3");
      root.innerHTML = `
        <div class=\"grid\">
          ${{kpiCard("Lien hommes-femmes", fmt(a.summary.corr_h_f))}}
          ${{kpiCard("Erreur moyenne forte (RMSE)", fmt(a.summary.rf_rmse))}}
          ${{kpiCard("Précision globale (R²)", fmt(a.summary.rf_r2))}}
          ${{kpiCard("Période", `${{a.line.labels[0]}} - ${{a.line.labels[a.line.labels.length - 1]}}`)}}
        </div>
        <div class=\"row\">
          <div class=\"card\"><h3>Évolution moyenne hommes/femmes</h3><canvas id=\"a3-line\"></canvas></div>
          <div class=\"card\"><h3>Écart femmes - hommes</h3><canvas id=\"a3-gap\"></canvas></div>
        </div>
        <div class=\"row\">
          <div class=\"card\"><h3>Importance des variables</h3><canvas id=\"a3-imp\"></canvas></div>
          ${{notesCard(a.text)}}
        </div>
        <div class=\"row\">
          ${{explainCard("Comment lire les graphiques", [
            "Courbes hommes/femmes: comparent l'évolution moyenne de la dépression dans les deux groupes.",
            "Gap F-M: si la courbe est au-dessus de 0, la dépression est en moyenne plus élevée chez les femmes.",
            "Importance des variables: montre ce qui pèse le plus dans la prédiction de la dépression féminine."
          ])}}
          ${{explainCard("Indicateurs utilisés", [
            "Lien hommes-femmes: proche de 1 = les deux courbes montent et baissent ensemble.",
            "RMSE: taille moyenne des erreurs importantes du modèle (plus petit = mieux).",
            "R²: capacité du modèle à bien reproduire les valeurs observées (plus grand = mieux).",
            "Période: années couvertes par l'analyse affichée."
          ])}}
        </div>
      `;

      new Chart(document.getElementById("a3-line"), {{
        type: "line",
        data: {{
          labels: a.line.labels,
          datasets: [
            {{ label: "Hommes", data: a.line.male, borderColor: "#2563eb", tension: 0.2, pointRadius: 1 }},
            {{ label: "Femmes", data: a.line.female, borderColor: "#f59e0b", tension: 0.2, pointRadius: 1 }}
          ]
        }},
        options: {{ responsive: true, plugins: {{ legend: {{ position: "bottom" }} }} }}
      }});

      new Chart(document.getElementById("a3-gap"), {{
        type: "line",
        data: {{ labels: a.line.labels, datasets: [{{ label: "Gap F-M", data: a.line.gap, borderColor: "#ef4444", fill: false, tension: 0.2, pointRadius: 1 }}] }},
        options: {{ responsive: true, plugins: {{ legend: {{ position: "bottom" }} }} }}
      }});

      new Chart(document.getElementById("a3-imp"), {{
        type: "bar",
        data: {{ labels: a.importance.labels, datasets: [{{ label: "Importance", data: a.importance.values, backgroundColor: "#60a5fa" }}] }},
        options: {{ responsive: true, indexAxis: "y", plugins: {{ legend: {{ display: false }} }} }}
      }});
    }}

    function initTabs() {{
      const buttons = document.querySelectorAll(".tab-btn");
      const panels = document.querySelectorAll(".panel");
      buttons.forEach(btn => {{
        btn.addEventListener("click", () => {{
          buttons.forEach(b => b.classList.remove("active"));
          panels.forEach(p => p.classList.remove("active"));
          btn.classList.add("active");
          document.getElementById(btn.dataset.panel).classList.add("active");
        }});
      }});
    }}

    renderA1();
    renderA2();
    renderA3();
    initTabs();
  </script>
</body>
</html>
"""


def main():
    payload = {
        "analyse1": analyse_1(),
        "analyse2": analyse_2(),
        "analyse3": analyse_3(),
    }

    html = build_html(payload)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Poster généré: {OUT_HTML}")


if __name__ == "__main__":
    main()
