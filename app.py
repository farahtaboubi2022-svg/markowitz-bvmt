import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="Markowitz BVMT", layout="wide")

st.title("📊 Tableau de bord Markowitz - BVMT")

uploaded_files = st.file_uploader(
    "Importer les fichiers Excel BVMT (2021-2025)",
    type=["xlsx"],
    accept_multiple_files=True
)

if uploaded_files:

    all_data = []
    for file in uploaded_files:
        df = pd.read_excel(file)
        all_data.append(df)

    data = pd.concat(all_data, ignore_index=True)

    data.columns = data.columns.astype(str).str.strip()
    data = data.loc[:, ~data.columns.duplicated()]

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

    prices = data.pivot_table(
        index="Date",
        columns="Societe",
        values="Close",
        aggfunc="last"
    )

    prices = prices.sort_index().ffill()

    banques = [
        "BIAT","ATB","STB","BT","AMEN BANK",
        "UIB","UBCI","BH","BNA","ATTIJARI BANK"
    ]

    banques_valides = [b for b in banques if b in prices.columns]

    selected_prices = prices[banques_valides].dropna().ffill()

    returns = selected_prices.pct_change().dropna()

    rf = 0.075

    mean_returns = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    cov_matrix = returns.cov() * 252

    st.subheader("📊 Indicateurs")
    st.dataframe(pd.DataFrame({
        "Rentabilité": mean_returns,
        "Volatilité": volatility
    }))

    n = len(banques_valides)
    init = np.ones(n) / n

    def port_return(w):
        return np.dot(w, mean_returns)

    def port_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

    def neg_sharpe(w):
        return -(port_return(w) - rf) / port_vol(w)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))

    result = minimize(neg_sharpe, init, method="SLSQP",
                      bounds=bounds, constraints=constraints)

    weights = result.x

    ret = port_return(weights)
    vol = port_vol(weights)
    sharpe = (ret - rf) / vol

    st.subheader("🚀 Portefeuille optimal")
    st.dataframe(pd.DataFrame({
        "Banque": banques_valides,
        "Poids": weights
    }))

    st.write("Rentabilité :", round(ret*100,2), "%")
    st.write("Risque :", round(vol*100,2), "%")
    st.write("Sharpe :", round(sharpe,4))

    # Graphique prix
    fig = px.line(selected_prices)
    st.plotly_chart(fig, use_container_width=True)

    # Frontière efficiente
    frontier_returns = []
    frontier_vol = []

    for target in np.linspace(mean_returns.min(), mean_returns.max(), 30):
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w)-1},
            {"type": "eq", "fun": lambda w, target=target: port_return(w)-target}
        )

        res = minimize(port_vol, init, method="SLSQP",
                       bounds=bounds, constraints=cons)

        if res.success:
            frontier_returns.append(port_return(res.x))
            frontier_vol.append(port_vol(res.x))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=frontier_vol,
        y=frontier_returns,
        mode="lines",
        name="Frontière efficiente"
    ))

    fig2.add_trace(go.Scatter(
        x=[vol], y=[ret],
        mode="markers",
        name="Portefeuille optimal"
    ))

    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Importer les fichiers Excel BVMT pour commencer")
