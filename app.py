import io
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="Markowitz BVMT", layout="wide")

st.title("📊 Tableau de bord Markowitz - BVMT")
st.write("Analyse financière et optimisation de portefeuille selon la théorie de Markowitz.")

# ==============================
# Chargement automatique Excel
# ==============================

BASE_DIR = Path(__file__).parent

excel_files = sorted(
    list(BASE_DIR.glob("*.xlsx")) +
    list(BASE_DIR.glob("*.xls")) +
    list(BASE_DIR.glob("*.xlsx.xlsx"))
)

if not excel_files:
    st.error("Aucun fichier Excel trouvé à côté de app.py.")
    st.write("Chemin recherché :", BASE_DIR)
    st.write("Fichiers trouvés :", list(BASE_DIR.glob("*")))
    st.stop()

st.sidebar.success(f"{len(excel_files)} fichier(s) Excel détecté(s)")

all_data = []

for file in excel_files:
    try:
        sheets = pd.read_excel(file, sheet_name=None)

        for sheet_name, df in sheets.items():
            df["Source"] = file.name
            df["Feuille"] = sheet_name
            all_data.append(df)

    except Exception as e:
        st.warning(f"Erreur lecture {file.name} : {e}")

if not all_data:
    st.error("Aucun fichier Excel lisible.")
    st.stop()

data = pd.concat(all_data, ignore_index=True)

# ==============================
# Nettoyage des données
# ==============================

data.columns = data.columns.astype(str).str.strip()
data = data.loc[:, ~data.columns.duplicated()]

required_columns = ["SEANCE", "VALEUR", "CLOTURE"]
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    st.error(f"Colonnes manquantes : {missing_columns}")
    st.write("Colonnes trouvées :", list(data.columns))
    st.stop()

data = data[["SEANCE", "VALEUR", "CLOTURE"]]
data.columns = ["Date", "Societe", "Close"]

data["Date"] = pd.to_datetime(data["Date"], errors="coerce", dayfirst=True)

data["Close"] = (
    data["Close"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .str.replace(" ", "", regex=False)
)

data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

data = data.dropna()
data = data[data["Close"] > 0]

if data.empty:
    st.error("Les données sont vides après nettoyage.")
    st.stop()

data["Année"] = data["Date"].dt.year

st.sidebar.write("📅 Période détectée :")
st.sidebar.write(data["Date"].min(), "→", data["Date"].max())

st.sidebar.write("📊 Années disponibles :")
st.sidebar.write(sorted(data["Année"].dropna().unique()))

prices = data.pivot_table(
    index="Date",
    columns="Societe",
    values="Close",
    aggfunc="last"
)

prices = prices.sort_index().ffill()

# ==============================
# Banques
# ==============================

banques = [
    "BIAT",
    "ATB",
    "STB",
    "BT",
    "AMEN BANK",
    "UIB",
    "UBCI",
    "BH",
    "BNA",
    "ATTIJARI BANK",
    "BH BANK",
    "BTE (ADP)",
    "WIFACK INT BANK"
]

banques_valides = [b for b in banques if b in prices.columns]

if len(banques_valides) < 2:
    st.error("Moins de deux banques valides trouvées.")
    st.write("Sociétés trouvées :", list(prices.columns))
    st.stop()

# ==============================
# Paramètres
# ==============================

st.sidebar.header("Paramètres")

selected_banques = st.sidebar.multiselect(
    "Choisir les banques",
    options=banques_valides,
    default=banques_valides[:min(6, len(banques_valides))]
)

rf = st.sidebar.number_input(
    "Taux sans risque annuel (%)",
    value=7.5,
    step=0.1
) / 100

if len(selected_banques) < 2:
    st.warning("Choisir au moins deux banques.")
    st.stop()

# ==============================
# Calculs financiers
# ==============================

selected_prices = prices[selected_banques].dropna(how="all").ffill()
returns = selected_prices.pct_change().dropna()

if returns.empty:
    st.error("Pas assez de données pour calculer les rendements.")
    st.stop()

mean_returns = returns.mean() * 252
volatility = returns.std() * np.sqrt(252)
cov_matrix = returns.cov() * 252
corr_matrix = returns.corr()

metrics = pd.DataFrame({
    "Rentabilité annualisée": mean_returns,
    "Volatilité annualisée": volatility,
    "Sharpe individuel": (mean_returns - rf) / volatility
})

metrics = metrics.replace([np.inf, -np.inf], np.nan).dropna()

n = len(selected_banques)
init = np.ones(n) / n

def port_return(w):
    return np.dot(w, mean_returns)

def port_vol(w):
    return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

def neg_sharpe(w):
    vol = port_vol(w)
    if vol == 0:
        return 999
    return -(port_return(w) - rf) / vol

def min_vol(w):
    return port_vol(w)

constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds = tuple((0, 1) for _ in range(n))

result_sharpe = minimize(
    neg_sharpe,
    init,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

if not result_sharpe.success:
    st.error("Erreur lors de l'optimisation Sharpe.")
    st.stop()

weights_sharpe = result_sharpe.x
ret_sharpe = port_return(weights_sharpe)
vol_sharpe = port_vol(weights_sharpe)
sharpe_ratio = (ret_sharpe - rf) / vol_sharpe

result_minvar = minimize(
    min_vol,
    init,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

if not result_minvar.success:
    st.error("Erreur lors de l'optimisation variance minimale.")
    st.stop()

weights_minvar = result_minvar.x
ret_minvar = port_return(weights_minvar)
vol_minvar = port_vol(weights_minvar)
sharpe_minvar = (ret_minvar - rf) / vol_minvar

weights_df = pd.DataFrame({
    "Banque": selected_banques,
    "Poids Sharpe max": weights_sharpe,
    "Poids variance minimale": weights_minvar
})

# ==============================
# Tabs
# ==============================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Vue générale",
    "Indicateurs",
    "Optimisation",
    "Frontière efficiente",
    "Export",
    "Aide investisseur"
])

# ==============================
# Tab 1
# ==============================

with tab1:
    st.subheader("Évolution des prix")

    fig_prices = px.line(
        selected_prices,
        title="Cours de clôture des banques sélectionnées"
    )
    fig_prices.update_layout(xaxis_title="Date", yaxis_title="Prix")
    st.plotly_chart(fig_prices, use_container_width=True)

    st.subheader("Rentabilité cumulée")

    cumulative_returns = (1 + returns).cumprod() - 1

    fig_cum = px.line(
        cumulative_returns,
        title="Rentabilité cumulée"
    )
    fig_cum.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_cum, use_container_width=True)

    st.subheader("Rendement vs Risque")

    risk_return = metrics.reset_index().rename(columns={"index": "Banque"})

    fig_scatter = px.scatter(
        risk_return,
        x="Volatilité annualisée",
        y="Rentabilité annualisée",
        text="Banque",
        size="Sharpe individuel",
        title="Comparaison rendement / risque"
    )
    fig_scatter.update_xaxes(tickformat=".0%")
    fig_scatter.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================
# Tab 2
# ==============================

with tab2:
    st.subheader("Indicateurs par banque")

    st.dataframe(
        metrics.style.format({
            "Rentabilité annualisée": "{:.2%}",
            "Volatilité annualisée": "{:.2%}",
            "Sharpe individuel": "{:.4f}"
        }),
        use_container_width=True
    )

    fig_ret = px.bar(
        metrics,
        y="Rentabilité annualisée",
        title="Rentabilité annualisée par banque"
    )
    fig_ret.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_ret, use_container_width=True)

    fig_vol = px.bar(
        metrics,
        y="Volatilité annualisée",
        title="Volatilité annualisée par banque"
    )
    fig_vol.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_vol, use_container_width=True)

    fig_sharpe = px.bar(
        metrics,
        y="Sharpe individuel",
        title="Ratio de Sharpe individuel par banque"
    )
    st.plotly_chart(fig_sharpe, use_container_width=True)

    st.subheader("Matrice variance-covariance")
    st.dataframe(cov_matrix, use_container_width=True)

    st.subheader("Matrice de corrélation")

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        title="Heatmap de corrélation"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ==============================
# Tab 3
# ==============================

with tab3:
    st.subheader("Portefeuille optimal Sharpe")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rentabilité", f"{ret_sharpe * 100:.2f}%")
    col2.metric("Risque", f"{vol_sharpe * 100:.2f}%")
    col3.metric("Sharpe", f"{sharpe_ratio:.4f}")

    st.subheader("Portefeuille à variance minimale")

    col4, col5, col6 = st.columns(3)
    col4.metric("Rentabilité", f"{ret_minvar * 100:.2f}%")
    col5.metric("Risque", f"{vol_minvar * 100:.2f}%")
    col6.metric("Sharpe", f"{sharpe_minvar:.4f}")

    st.subheader("Poids optimaux")

    st.dataframe(
        weights_df.style.format({
            "Poids Sharpe max": "{:.2%}",
            "Poids variance minimale": "{:.2%}"
        }),
        use_container_width=True
    )

    weights_long = weights_df.melt(
        id_vars="Banque",
        var_name="Portefeuille",
        value_name="Poids"
    )

    fig_weights = px.bar(
        weights_long,
        x="Banque",
        y="Poids",
        color="Portefeuille",
        barmode="group",
        title="Comparaison des poids optimaux"
    )
    fig_weights.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_weights, use_container_width=True)

    st.subheader("Camembert - Portefeuille Sharpe max")

    fig_pie_sharpe = px.pie(
        weights_df,
        names="Banque",
        values="Poids Sharpe max",
        title="Répartition du portefeuille optimal Sharpe"
    )
    st.plotly_chart(fig_pie_sharpe, use_container_width=True)

    st.subheader("Camembert - Portefeuille variance minimale")

    fig_pie_minvar = px.pie(
        weights_df,
        names="Banque",
        values="Poids variance minimale",
        title="Répartition du portefeuille à risque minimal"
    )
    st.plotly_chart(fig_pie_minvar, use_container_width=True)

# ==============================
# Tab 4
# ==============================

with tab4:
    st.subheader("Frontière efficiente")

    frontier_returns = []
    frontier_vols = []

    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)

    for target in target_returns:
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, target=target: port_return(w) - target}
        )

        result = minimize(
            min_vol,
            init,
            method="SLSQP",
            bounds=bounds,
            constraints=cons
        )

        if result.success:
            w = result.x
            frontier_returns.append(port_return(w))
            frontier_vols.append(port_vol(w))

    fig_frontier = go.Figure()

    fig_frontier.add_trace(go.Scatter(
        x=frontier_vols,
        y=frontier_returns,
        mode="lines",
        name="Frontière efficiente"
    ))

    fig_frontier.add_trace(go.Scatter(
        x=[vol_sharpe],
        y=[ret_sharpe],
        mode="markers",
        name="Sharpe max",
        marker=dict(size=14)
    ))

    fig_frontier.add_trace(go.Scatter(
        x=[vol_minvar],
        y=[ret_minvar],
        mode="markers",
        name="Variance minimale",
        marker=dict(size=14)
    ))

    fig_frontier.update_layout(
        title="Frontière efficiente de Markowitz",
        xaxis_title="Risque / Volatilité",
        yaxis_title="Rentabilité"
    )

    fig_frontier.update_xaxes(tickformat=".0%")
    fig_frontier.update_yaxes(tickformat=".0%")

    st.plotly_chart(fig_frontier, use_container_width=True)

# ==============================
# Tab 5
# ==============================

with tab5:
    st.subheader("Télécharger les résultats")

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        selected_prices.to_excel(writer, sheet_name="Prix")
        returns.to_excel(writer, sheet_name="Rendements")
        metrics.to_excel(writer, sheet_name="Indicateurs")
        cov_matrix.to_excel(writer, sheet_name="Covariance")
        corr_matrix.to_excel(writer, sheet_name="Correlation")
        weights_df.to_excel(writer, sheet_name="Poids", index=False)

    st.download_button(
        label="Télécharger le rapport Excel",
        data=output.getvalue(),
        file_name="rapport_markowitz_bvmt.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ==============================
# Tab 6
# ==============================

with tab6:
    st.subheader("📈 Aide à la décision pour investisseurs BVMT")

    st.info(
        "Cette section est informative. Elle ne constitue pas un conseil financier personnalisé."
    )

    ranking = metrics.copy()

    ranking["Score"] = (
        ranking["Sharpe individuel"].rank(ascending=False) +
        ranking["Rentabilité annualisée"].rank(ascending=False) +
        ranking["Volatilité annualisée"].rank(ascending=True)
    )

    ranking = ranking.sort_values("Score")

    st.subheader("Classement indicatif des banques")

    st.dataframe(
        ranking.style.format({
            "Rentabilité annualisée": "{:.2%}",
            "Volatilité annualisée": "{:.2%}",
            "Sharpe individuel": "{:.4f}",
            "Score": "{:.0f}"
        }),
        use_container_width=True
    )

    best_bank = ranking.index[0]

    st.success(f"🏆 Meilleure banque selon le modèle : {best_bank}")

    fig_score = px.bar(
        ranking,
        y="Score",
        title="Score indicatif des banques"
    )
    st.plotly_chart(fig_score, use_container_width=True)

    fig_best = px.bar(
        ranking,
        y=["Rentabilité annualisée", "Volatilité annualisée"],
        barmode="group",
        title="Rentabilité vs Risque par banque"
    )
    fig_best.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_best, use_container_width=True)

    st.subheader("Interprétation pour l’investisseur")

    st.write("""
    Ce tableau de bord aide un investisseur intéressé par la BVMT à analyser les banques cotées
    selon trois critères principaux :

    - la rentabilité annualisée ;
    - la volatilité, qui mesure le risque ;
    - le ratio de Sharpe, qui compare la rentabilité au risque.

    Le portefeuille optimal Sharpe cherche la meilleure combinaison rendement / risque.
    Le portefeuille à variance minimale cherche à réduire le risque au maximum.
    """)

    st.warning(
        "Avant tout investissement réel, il faut vérifier la liquidité du titre, "
        "les frais, la fiscalité, les conditions du compte titres et la réglementation en vigueur."
    )
