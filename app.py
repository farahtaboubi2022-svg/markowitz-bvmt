import io
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

# ==============================
# Configuration page
# ==============================

st.set_page_config(page_title="Markowitz BVMT", layout="wide")

st.title("📊 Tableau de bord Markowitz - BVMT")
st.write("Analyse financière et optimisation de portefeuille selon la théorie de Markowitz.")

# ==============================
# Chargement automatique des fichiers Excel
# ==============================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

excel_files = list(DATA_DIR.glob("*.xlsx"))

if not DATA_DIR.exists():
    st.error("Le dossier 'data' n'existe pas. Crée un dossier data au même niveau que app.py.")
    st.stop()

if not excel_files:
    st.error("Aucun fichier Excel trouvé dans le dossier data.")
    st.write("Chemin recherché :", DATA_DIR)
    st.write("Contenu du dossier du projet :", list(BASE_DIR.glob("*")))
    st.stop()

st.success(f"{len(excel_files)} fichier(s) Excel chargé(s) automatiquement.")

all_data = []

for file in excel_files:
    df = pd.read_excel(file)
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# ==============================
# Nettoyage des données
# ==============================

data.columns = data.columns.astype(str).str.strip()
data = data.loc[:, ~data.columns.duplicated()]

required_columns = ["SEANCE", "VALEUR", "CLOTURE"]

missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    st.error(f"Colonnes manquantes dans les fichiers Excel : {missing_columns}")
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
    st.error("Moins de deux banques valides trouvées dans les fichiers Excel.")
    st.write("Sociétés trouvées :", list(prices.columns))
    st.stop()

# ==============================
# Sidebar
# ==============================

st.sidebar.header("Paramètres")

selected_banques = st.sidebar.multiselect(
    "Choisir les banques",
    options=banques_valides,
    default=banques_valides[:min(5, len(banques_valides))]
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
    st.error("Erreur lors de l'optimisation du portefeuille Sharpe.")
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
    st.error("Erreur lors de l'optimisation du portefeuille à variance minimale.")
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
# Interface
# ==============================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Vue générale",
    "Indicateurs",
    "Optimisation",
    "Frontière efficiente",
    "Export"
])

with tab1:
    st.subheader("Évolution des prix")

    fig_prices = px.line(
        selected_prices,
        title="Cours de clôture des banques sélectionnées"
    )
    st.plotly_chart(fig_prices, use_container_width=True)

    st.subheader("Rentabilité cumulée")

    cumulative_returns = (1 + returns).cumprod() - 1

    fig_cum = px.line(
        cumulative_returns,
        title="Rentabilité cumulée"
    )
    fig_cum.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_cum, use_container_width=True)

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

    fig_vol = px.bar(
        metrics,
        y="Volatilité annualisée",
        title="Volatilité annualisée par banque"
    )
    fig_vol.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("Matrice variance-covariance")
    st.dataframe(cov_matrix, use_container_width=True)

    st.subheader("Matrice de corrélation")
    st.dataframe(corr_matrix, use_container_width=True)

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
