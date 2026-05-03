import io
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="Markowitz BVMT Pro", layout="wide")

st.title("📊 Tableau de bord Markowitz BVMT - Version Pro")
st.write("Analyse financière avancée, optimisation de portefeuille et aide intelligente à la décision.")

# ==============================
# Cache - chargement rapide
# ==============================

@st.cache_data
def load_excel_files(file_paths):
    all_data = []

    for file_path in file_paths:
        file = Path(file_path)

        # Lecture de la première feuille seulement pour accélérer
        df = pd.read_excel(file)

        df["Source"] = file.name
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


@st.cache_data
def prepare_data(data):
    data = data.copy()

    data.columns = data.columns.astype(str).str.strip()
    data = data.loc[:, ~data.columns.duplicated()]

    required_columns = ["SEANCE", "VALEUR", "CLOTURE"]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        return None, missing_columns, list(data.columns)

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
        return None, ["Données vides après nettoyage"], []

    data["Année"] = data["Date"].dt.year

    prices = data.pivot_table(
        index="Date",
        columns="Societe",
        values="Close",
        aggfunc="last"
    )

    prices = prices.sort_index().ffill()

    return (data, prices), [], []


# ==============================
# Chargement fichiers Excel
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
    st.stop()

st.sidebar.success(f"{len(excel_files)} fichier(s) Excel détecté(s)")

try:
    data_raw = load_excel_files([str(f) for f in excel_files])
except Exception as e:
    st.error(f"Erreur lors du chargement Excel : {e}")
    st.stop()

prepared, errors, columns_found = prepare_data(data_raw)

if errors:
    st.error(f"Problème dans les données : {errors}")
    if columns_found:
        st.write("Colonnes trouvées :", columns_found)
    st.stop()

data, prices = prepared

# ==============================
# Infos sidebar
# ==============================

st.sidebar.write("📅 Période détectée")
st.sidebar.write(data["Date"].min(), "→", data["Date"].max())

st.sidebar.write("📊 Années disponibles")
st.sidebar.write(sorted(data["Année"].dropna().unique()))

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

capital = st.sidebar.number_input(
    "Capital à investir",
    value=10000,
    step=500
)

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

cumulative_returns = (1 + returns).cumprod() - 1
rolling_vol = returns.rolling(30).std() * np.sqrt(252)

drawdown = selected_prices / selected_prices.cummax() - 1
max_drawdown = drawdown.min()

var_95 = returns.quantile(0.05) * np.sqrt(252)
expected_shortfall = returns[returns.le(returns.quantile(0.05))].mean() * np.sqrt(252)

market_return = returns.mean(axis=1)

beta = {}
for col in returns.columns:
    cov = np.cov(returns[col], market_return)[0][1]
    var = np.var(market_return)
    beta[col] = cov / var if var != 0 else np.nan

beta = pd.Series(beta)

metrics = pd.DataFrame({
    "Rentabilité annualisée": mean_returns,
    "Volatilité annualisée": volatility,
    "Sharpe individuel": (mean_returns - rf) / volatility,
    "Max Drawdown": max_drawdown,
    "VaR 95%": var_95,
    "Expected Shortfall": expected_shortfall,
    "Beta marché": beta
})

metrics = metrics.replace([np.inf, -np.inf], np.nan).dropna()

# ==============================
# Optimisation Markowitz
# ==============================

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

result_minvar = minimize(
    min_vol,
    init,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

if not result_sharpe.success or not result_minvar.success:
    st.error("Erreur d’optimisation.")
    st.stop()

weights_sharpe = result_sharpe.x
weights_minvar = result_minvar.x

ret_sharpe = port_return(weights_sharpe)
vol_sharpe = port_vol(weights_sharpe)
sharpe_ratio = (ret_sharpe - rf) / vol_sharpe

ret_minvar = port_return(weights_minvar)
vol_minvar = port_vol(weights_minvar)
sharpe_minvar = (ret_minvar - rf) / vol_minvar

weights_df = pd.DataFrame({
    "Banque": selected_banques,
    "Poids Sharpe max": weights_sharpe,
    "Montant Sharpe max": weights_sharpe * capital,
    "Poids variance minimale": weights_minvar,
    "Montant variance minimale": weights_minvar * capital
})

# ==============================
# Recommandations
# ==============================

ranking = metrics.copy()

ranking["Score"] = (
    ranking["Sharpe individuel"].rank(ascending=False) +
    ranking["Rentabilité annualisée"].rank(ascending=False) +
    ranking["Volatilité annualisée"].rank(ascending=True) +
    ranking["Max Drawdown"].rank(ascending=False) +
    ranking["VaR 95%"].rank(ascending=False)
)

ranking = ranking.sort_values("Score")

best_bank = ranking.index[0]

def recommendation(row):
    if row["Sharpe individuel"] > 1 and row["Volatilité annualisée"] < metrics["Volatilité annualisée"].median():
        return "Très attractive"
    elif row["Sharpe individuel"] > 0.5:
        return "Intéressante"
    elif row["Volatilité annualisée"] > metrics["Volatilité annualisée"].median():
        return "Risque élevé"
    else:
        return "À surveiller"

ranking["Recommandation"] = ranking.apply(recommendation, axis=1)

# ==============================
# Interface
# ==============================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Vue générale",
    "Indicateurs",
    "Risques",
    "Optimisation",
    "Frontière efficiente",
    "Recommandations",
    "Simulation",
    "Export"
])

with tab1:
    st.subheader("Résumé global")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nombre de banques", len(selected_banques))
    col2.metric("Meilleure banque", best_bank)
    col3.metric("Sharpe max", f"{sharpe_ratio:.4f}")
    col4.metric("Capital simulé", f"{capital:,.0f}")

    fig_prices = px.line(selected_prices, title="Évolution des cours de clôture")
    fig_prices.update_layout(xaxis_title="Date", yaxis_title="Cours")
    st.plotly_chart(fig_prices, use_container_width=True)

    fig_cum = px.line(cumulative_returns, title="Rentabilité cumulée")
    fig_cum.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_cum, use_container_width=True)

    rr = metrics.reset_index().rename(columns={"index": "Banque"})

    fig_rr = px.scatter(
        rr,
        x="Volatilité annualisée",
        y="Rentabilité annualisée",
        size="Sharpe individuel",
        color="Sharpe individuel",
        text="Banque",
        title="Carte rendement / risque"
    )
    fig_rr.update_xaxes(tickformat=".0%")
    fig_rr.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_rr, use_container_width=True)

with tab2:
    st.subheader("Tableau des indicateurs")

    st.dataframe(
        metrics.style.format({
            "Rentabilité annualisée": "{:.2%}",
            "Volatilité annualisée": "{:.2%}",
            "Sharpe individuel": "{:.4f}",
            "Max Drawdown": "{:.2%}",
            "VaR 95%": "{:.2%}",
            "Expected Shortfall": "{:.2%}",
            "Beta marché": "{:.4f}"
        }),
        use_container_width=True
    )

    fig_ret = px.bar(metrics, y="Rentabilité annualisée", title="Rentabilité annualisée")
    fig_ret.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_ret, use_container_width=True)

    fig_vol = px.bar(metrics, y="Volatilité annualisée", title="Volatilité annualisée")
    fig_vol.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_vol, use_container_width=True)

    fig_sharpe = px.bar(metrics, y="Sharpe individuel", title="Ratio de Sharpe")
    st.plotly_chart(fig_sharpe, use_container_width=True)

    fig_beta = px.bar(metrics, y="Beta marché", title="Beta relatif au marché bancaire")
    st.plotly_chart(fig_beta, use_container_width=True)

with tab3:
    st.subheader("Analyse des risques")

    fig_drawdown = px.line(drawdown, title="Drawdown des banques sélectionnées")
    fig_drawdown.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_drawdown, use_container_width=True)

    fig_max_dd = px.bar(metrics, y="Max Drawdown", title="Max Drawdown par banque")
    fig_max_dd.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_max_dd, use_container_width=True)

    fig_var = px.bar(metrics, y="VaR 95%", title="Value at Risk 95%")
    fig_var.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_var, use_container_width=True)

    fig_es = px.bar(metrics, y="Expected Shortfall", title="Expected Shortfall")
    fig_es.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_es, use_container_width=True)

    fig_roll = px.line(rolling_vol, title="Volatilité glissante annualisée 30 jours")
    fig_roll.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_roll, use_container_width=True)

    fig_corr = px.imshow(corr_matrix, text_auto=True, title="Heatmap de corrélation")
    st.plotly_chart(fig_corr, use_container_width=True)

with tab4:
    st.subheader("Portefeuille Sharpe maximum")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rentabilité", f"{ret_sharpe:.2%}")
    c2.metric("Risque", f"{vol_sharpe:.2%}")
    c3.metric("Sharpe", f"{sharpe_ratio:.4f}")

    st.subheader("Portefeuille variance minimale")

    c4, c5, c6 = st.columns(3)
    c4.metric("Rentabilité", f"{ret_minvar:.2%}")
    c5.metric("Risque", f"{vol_minvar:.2%}")
    c6.metric("Sharpe", f"{sharpe_minvar:.4f}")

    st.subheader("Poids et montants à investir")

    st.dataframe(
        weights_df.style.format({
            "Poids Sharpe max": "{:.2%}",
            "Montant Sharpe max": "{:,.2f}",
            "Poids variance minimale": "{:.2%}",
            "Montant variance minimale": "{:,.2f}"
        }),
        use_container_width=True
    )

    weights_long = weights_df.melt(
        id_vars="Banque",
        value_vars=["Poids Sharpe max", "Poids variance minimale"],
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

    fig_pie1 = px.pie(weights_df, names="Banque", values="Poids Sharpe max", title="Répartition portefeuille Sharpe max")
    st.plotly_chart(fig_pie1, use_container_width=True)

    fig_pie2 = px.pie(weights_df, names="Banque", values="Poids variance minimale", title="Répartition portefeuille variance minimale")
    st.plotly_chart(fig_pie2, use_container_width=True)

with tab5:
    st.subheader("Frontière efficiente")

    frontier_returns = []
    frontier_vols = []

    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 60)

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

with tab6:
    st.subheader("🤖 Recommandations intelligentes")

    st.info("Cette analyse est indicative et ne constitue pas un conseil financier personnalisé.")

    st.success(f"🏆 Meilleure banque selon le modèle : {best_bank}")

    st.dataframe(
        ranking.style.format({
            "Rentabilité annualisée": "{:.2%}",
            "Volatilité annualisée": "{:.2%}",
            "Sharpe individuel": "{:.4f}",
            "Max Drawdown": "{:.2%}",
            "VaR 95%": "{:.2%}",
            "Expected Shortfall": "{:.2%}",
            "Beta marché": "{:.4f}",
            "Score": "{:.0f}"
        }),
        use_container_width=True
    )

    fig_score = px.bar(ranking, y="Score", color="Recommandation", title="Classement intelligent des banques")
    st.plotly_chart(fig_score, use_container_width=True)

    fig_reco = px.scatter(
        ranking.reset_index().rename(columns={"index": "Banque"}),
        x="Volatilité annualisée",
        y="Rentabilité annualisée",
        color="Recommandation",
        size="Sharpe individuel",
        text="Banque",
        title="Carte de recommandation"
    )
    fig_reco.update_xaxes(tickformat=".0%")
    fig_reco.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_reco, use_container_width=True)

    st.write("""
    Le modèle classe les banques selon la rentabilité, le risque, le Sharpe,
    le drawdown, la VaR et l’Expected Shortfall.
    """)

    st.warning(
        "Avant d’investir réellement, il faut vérifier la liquidité, les frais, "
        "la fiscalité, la réglementation et les conditions d’ouverture d’un compte titres."
    )

with tab7:
    st.subheader("💼 Simulation d’investissement")

    st.write(f"Capital simulé : **{capital:,.2f}**")

    invest_sharpe = weights_df[["Banque", "Montant Sharpe max"]].copy()

    st.subheader("Montants à investir - Sharpe max")
    st.dataframe(
        invest_sharpe.style.format({"Montant Sharpe max": "{:,.2f}"}),
        use_container_width=True
    )

    fig_invest1 = px.pie(
        invest_sharpe,
        names="Banque",
        values="Montant Sharpe max",
        title="Répartition du capital - Sharpe max"
    )
    st.plotly_chart(fig_invest1, use_container_width=True)

    invest_minvar = weights_df[["Banque", "Montant variance minimale"]].copy()

    st.subheader("Montants à investir - variance minimale")
    st.dataframe(
        invest_minvar.style.format({"Montant variance minimale": "{:,.2f}"}),
        use_container_width=True
    )

    fig_invest2 = px.pie(
        invest_minvar,
        names="Banque",
        values="Montant variance minimale",
        title="Répartition du capital - variance minimale"
    )
    st.plotly_chart(fig_invest2, use_container_width=True)

    profile = st.selectbox(
        "Profil investisseur",
        ["Prudent", "Équilibré", "Dynamique"]
    )

    if profile == "Prudent":
        st.success("Profil prudent : privilégier le portefeuille à variance minimale.")
    elif profile == "Équilibré":
        st.success("Profil équilibré : combiner portefeuille Sharpe max et variance minimale.")
    else:
        st.success("Profil dynamique : privilégier le portefeuille Sharpe max.")

with tab8:
    st.subheader("Télécharger le rapport Excel")

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        selected_prices.to_excel(writer, sheet_name="Prix")
        returns.to_excel(writer, sheet_name="Rendements")
        cumulative_returns.to_excel(writer, sheet_name="Rendements cumules")
        metrics.to_excel(writer, sheet_name="Indicateurs")
        ranking.to_excel(writer, sheet_name="Recommandations")
        cov_matrix.to_excel(writer, sheet_name="Covariance")
        corr_matrix.to_excel(writer, sheet_name="Correlation")
        weights_df.to_excel(writer, sheet_name="Poids", index=False)

    st.download_button(
        label="Télécharger le rapport complet",
        data=output.getvalue(),
        file_name="rapport_markowitz_bvmt_pro.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
