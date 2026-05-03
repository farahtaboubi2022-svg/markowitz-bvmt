import io
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

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
        
        try:
            df = pd.read_excel(file)
            df.columns = df.columns.str.strip()
            df["Source"] = file.stem
            all_data.append(df)
        except Exception as e:
            st.error(f"Erreur chargement {file.name}: {e}")
    
    if not all_data:
        return None
    
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

    data["Année"] = data["Date"].dt.year.astype(int)

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

excel_files = []
for file in BASE_DIR.iterdir():
    if file.is_file() and file.suffix.lower() in ['.xlsx', '.xls']:
        if not file.name.startswith('~') and not file.name.startswith('.'):
            excel_files.append(file)

if not excel_files:
    st.error("Aucun fichier Excel trouvé!")
    st.stop()

try:
    data_raw = load_excel_files(excel_files)
    if data_raw is None:
        st.error("Aucune donnée chargée.")
        st.stop()
except Exception as e:
    st.error(f"Erreur lors du chargement : {e}")
    st.stop()

prepared, errors, columns_found = prepare_data(data_raw)

if errors:
    st.error(f"Problème dans les données : {errors}")
    st.stop()

data, prices = prepared

if prices.empty:
    st.error("Aucune donnée de prix après traitement.")
    st.stop()

# ==============================
# Récupérer les années disponibles
# ==============================

annees_disponibles = sorted([int(a) for a in data["Année"].dropna().unique()])

if len(annees_disponibles) == 0:
    st.error("Aucune année détectée!")
    st.stop()

# ==============================
# Sidebar - Sélection de la période
# ==============================

st.sidebar.header("📅 Sélection de la période")

selected_annee = st.sidebar.selectbox(
    "Choisir l'année à analyser",
    options=annees_disponibles,
    index=len(annees_disponibles)-1
)

# ==============================
# Filtrer les données par année
# ==============================

data_filtered = data[data["Année"] == selected_annee].copy()

if data_filtered.empty:
    st.error(f"Aucune donnée pour l'année {selected_annee}")
    st.stop()

prices_filtered = data_filtered.pivot_table(
    index="Date",
    columns="Societe",
    values="Close",
    aggfunc="last"
).sort_index().ffill()

# ==============================
# Liste des banques
# ==============================

banques_liste = [
    "BIAT", "ATB", "STB", "BT", "AMEN BANK", "UIB", "UBCI", "BH",
    "BNA", "ATTIJARI BANK", "BH BANK", "BTE", "WIFACK INT BANK"
]

# Trouver les banques présentes
toutes_societes = sorted(prices_filtered.columns.tolist())
banques_disponibles = []

for b in banques_liste:
    for col in toutes_societes:
        if b.upper() in col.upper() or col.upper() in b.upper():
            if col not in banques_disponibles:
                banques_disponibles.append(col)

if not banques_disponibles:
    banques_disponibles = toutes_societes[:20]

# ==============================
# Sélection des sociétés
# ==============================

st.sidebar.header("⚙️ Sélection des actifs")

option_select = st.sidebar.radio(
    "Mode de sélection",
    ["Banques uniquement", "Toutes les sociétés", "Sélection manuelle"]
)

if option_select == "Banques uniquement":
    selected_societies = banques_disponibles
elif option_select == "Toutes les sociétés":
    selected_societies = toutes_societes[:30]  # Limite pour performance
else:
    selected_societies = st.sidebar.multiselect(
        "Choisir les sociétés",
        options=toutes_societes,
        default=toutes_societies[:min(10, len(toutes_societes))]
    )

if len(selected_societies) < 2:
    st.warning("Veuillez sélectionner au moins 2 sociétés")
    selected_societies = toutes_societes[:2]

# ==============================
# Paramètres financiers
# ==============================

st.sidebar.header("⚙️ Paramètres financiers")
rf = st.sidebar.number_input("Taux sans risque (%)", value=7.5, step=0.1) / 100
capital = st.sidebar.number_input("Capital (TND)", value=10000, step=1000)

# ==============================
# Calculs financiers
# ==============================

selected_prices = prices_filtered[selected_societies].dropna(how="all").ffill()
returns = selected_prices.pct_change().dropna()

# Annualisation
mean_returns = returns.mean() * 252
volatility = returns.std() * np.sqrt(252)
cov_matrix = returns.cov() * 252
corr_matrix = returns.corr()

# Calculs supplémentaires
cumulative_returns = (1 + returns).cumprod() - 1
rolling_vol = returns.rolling(20).std() * np.sqrt(252)
drawdown = selected_prices / selected_prices.cummax() - 1
max_drawdown = drawdown.min()
var_95 = returns.quantile(0.05) * np.sqrt(252)
expected_shortfall = returns[returns.le(returns.quantile(0.05))].mean() * np.sqrt(252)

# Beta par rapport au marché (moyenne du marché)
market_return = returns.mean(axis=1)
beta = {}
for col in returns.columns:
    cov = np.cov(returns[col], market_return)[0][1] if len(market_return) > 1 else 0
    var = np.var(market_return) if len(market_return) > 1 else 1
    beta[col] = cov / var if var != 0 else np.nan
beta = pd.Series(beta)

# Métriques individuelles
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

n = len(selected_societies)
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

# Portefeuille Sharpe max
result_sharpe = minimize(neg_sharpe, init, method="SLSQP", bounds=bounds, constraints=constraints)

# Portefeuille variance minimale
result_minvar = minimize(min_vol, init, method="SLSQP", bounds=bounds, constraints=constraints)

if result_sharpe.success and result_minvar.success:
    weights_sharpe = result_sharpe.x
    weights_minvar = result_minvar.x
    
    ret_sharpe = port_return(weights_sharpe)
    vol_sharpe = port_vol(weights_sharpe)
    sharpe_ratio = (ret_sharpe - rf) / vol_sharpe if vol_sharpe > 0 else 0
    
    ret_minvar = port_return(weights_minvar)
    vol_minvar = port_vol(weights_minvar)
    sharpe_minvar = (ret_minvar - rf) / vol_minvar if vol_minvar > 0 else 0
    
    weights_df = pd.DataFrame({
        "Société": selected_societies,
        "Poids Sharpe max": weights_sharpe,
        "Montant Sharpe max (TND)": weights_sharpe * capital,
        "Poids variance min": weights_minvar,
        "Montant variance min (TND)": weights_minvar * capital
    })
else:
    st.error("Erreur d'optimisation")
    st.stop()

# ==============================
# Classement intelligent
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
best_society = ranking.index[0]

def get_recommendation(row):
    median_vol = metrics["Volatilité annualisée"].median()
    if row["Sharpe individuel"] > 1 and row["Volatilité annualisée"] < median_vol:
        return "🟢 Très attractive"
    elif row["Sharpe individuel"] > 0.5:
        return "🟡 Intéressante"
    elif row["Volatilité annualisée"] > median_vol:
        return "🔴 Risque élevé"
    else:
        return "⚪ À surveiller"

ranking["Recommandation"] = ranking.apply(get_recommendation, axis=1)

# ==============================
# Frontière efficiente
# ==============================

frontier_returns = []
frontier_vols = []
target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 30)

for target in target_returns:
    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w, t=target: port_return(w) - t}
    )
    result = minimize(min_vol, init, method="SLSQP", bounds=bounds, constraints=cons)
    if result.success:
        frontier_returns.append(port_return(result.x))
        frontier_vols.append(port_vol(result.x))

# ==============================
# INTERFACE PRINCIPALE - 8 ONGLETS
# ==============================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Vue générale",
    "📈 Indicateurs",
    "⚠️ Risques",
    "🎯 Optimisation",
    "📉 Frontière efficiente",
    "🤖 Recommandations",
    "💼 Simulation",
    "📥 Export"
])

# ==================== TAB 1: VUE GÉNÉRALE ====================
with tab1:
    st.header(f"Vue générale - Année {selected_annee}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nombre d'actifs", len(selected_societies))
    col2.metric("Meilleur actif", best_society[:20])
    col3.metric("Sharpe optimal", f"{sharpe_ratio:.3f}")
    col4.metric("Capital simulé", f"{capital:,.0f} TND")
    
    # Évolution des cours
    st.subheader("📈 Évolution des cours 2021-2025")
    fig_prices = px.line(selected_prices, title="Cours de clôture")
    fig_prices.update_layout(xaxis_title="Date", yaxis_title="Prix (TND)", height=500)
    st.plotly_chart(fig_prices, use_container_width=True)
    
    # Rendements cumulés
    st.subheader("📊 Rendements cumulés")
    fig_cum = px.line(cumulative_returns, title="Performance cumulée")
    fig_cum.update_yaxes(tickformat=".0%")
    fig_cum.update_layout(height=400)
    st.plotly_chart(fig_cum, use_container_width=True)
    
    # Carte rendement/risque
    st.subheader("🗺️ Carte rendement/risque")
    rr_df = metrics.reset_index().rename(columns={"index": "Société"})
    fig_rr = px.scatter(
        rr_df,
        x="Volatilité annualisée",
        y="Rentabilité annualisée",
        size="Sharpe individuel",
        color="Sharpe individuel",
        text="Société",
        title="Rendement vs Risque"
    )
    fig_rr.update_xaxes(tickformat=".0%")
    fig_rr.update_yaxes(tickformat=".0%")
    fig_rr.update_layout(height=500)
    st.plotly_chart(fig_rr, use_container_width=True)

# ==================== TAB 2: INDICATEURS ====================
with tab2:
    st.header("📈 Indicateurs financiers détaillés")
    
    st.subheader("Tableau des métriques")
    st.dataframe(
        metrics.style.format({
            "Rentabilité annualisée": "{:.2%}",
            "Volatilité annualisée": "{:.2%}",
            "Sharpe individuel": "{:.3f}",
            "Max Drawdown": "{:.2%}",
            "VaR 95%": "{:.2%}",
            "Expected Shortfall": "{:.2%}",
            "Beta marché": "{:.3f}"
        }),
        use_container_width=True,
        height=400
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rentabilité annualisée")
        fig_ret = px.bar(metrics, y="Rentabilité annualisée", title="Rentabilité par actif")
        fig_ret.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_ret, use_container_width=True)
    
    with col2:
        st.subheader("Volatilité annualisée")
        fig_vol = px.bar(metrics, y="Volatilité annualisée", title="Risque par actif")
        fig_vol.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_vol, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Ratio de Sharpe individuel")
        fig_sharpe = px.bar(metrics, y="Sharpe individuel", title="Sharpe par actif")
        st.plotly_chart(fig_sharpe, use_container_width=True)
    
    with col4:
        st.subheader("Beta marché")
        fig_beta = px.bar(metrics, y="Beta marché", title="Sensibilité au marché")
        st.plotly_chart(fig_beta, use_container_width=True)

# ==================== TAB 3: RISQUES ====================
with tab3:
    st.header("⚠️ Analyse détaillée des risques")
    
    # Drawdown
    st.subheader("📉 Drawdown")
    fig_dd = px.line(drawdown, title="Drawdown des actifs")
    fig_dd.update_yaxes(tickformat=".0%")
    fig_dd.update_layout(height=400)
    st.plotly_chart(fig_dd, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Max Drawdown")
        fig_mdd = px.bar(metrics, y="Max Drawdown", title="Perte maximale")
        fig_mdd.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_mdd, use_container_width=True)
    
    with col2:
        st.subheader("Value at Risk (95%)")
        fig_var = px.bar(metrics, y="VaR 95%", title="VaR 95%")
        fig_var.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_var, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Expected Shortfall")
        fig_es = px.bar(metrics, y="Expected Shortfall", title="ES (CVaR)")
        fig_es.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_es, use_container_width=True)
    
    with col4:
        st.subheader("Volatilité glissante")
        fig_roll = px.line(rolling_vol, title="Volatilité rolling 20 jours")
        fig_roll.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_roll, use_container_width=True)
    
    # Heatmap de corrélation
    st.subheader("🔗 Matrice de corrélation")
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Corrélations entre actifs",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

# ==================== TAB 4: OPTIMISATION ====================
with tab4:
    st.header("🎯 Optimisation de portefeuille")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Portefeuille Sharpe Maximum")
        st.metric("Rentabilité", f"{ret_sharpe:.2%}")
        st.metric("Risque", f"{vol_sharpe:.2%}")
        st.metric("Ratio de Sharpe", f"{sharpe_ratio:.4f}")
    
    with col2:
        st.subheader("🛡️ Portefeuille Variance Minimale")
        st.metric("Rentabilité", f"{ret_minvar:.2%}")
        st.metric("Risque", f"{vol_minvar:.2%}")
        st.metric("Ratio de Sharpe", f"{sharpe_minvar:.4f}")
    
    # Tableau des poids
    st.subheader("📊 Allocation des portefeuilles")
    st.dataframe(
        weights_df.style.format({
            "Poids Sharpe max": "{:.2%}",
            "Montant Sharpe max (TND)": "{:,.2f}",
            "Poids variance min": "{:.2%}",
            "Montant variance min (TND)": "{:,.2f}"
        }),
        use_container_width=True
    )
    
    # Graphique des poids
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Portefeuille Sharpe Max")
        weights_show = weights_df[weights_df["Poids Sharpe max"] > 0.01].sort_values("Poids Sharpe max", ascending=False)
        if len(weights_show) > 0:
            fig_pie1 = px.pie(
                weights_show.head(10),
                names="Société",
                values="Poids Sharpe max",
                title="Répartition Sharpe max"
            )
            st.plotly_chart(fig_pie1, use_container_width=True)
    
    with col4:
        st.subheader("Portefeuille Variance Min")
        weights_show2 = weights_df[weights_df["Poids variance min"] > 0.01].sort_values("Poids variance min", ascending=False)
        if len(weights_show2) > 0:
            fig_pie2 = px.pie(
                weights_show2.head(10),
                names="Société",
                values="Poids variance min",
                title="Répartition variance min"
            )
            st.plotly_chart(fig_pie2, use_container_width=True)
    
    # Comparaison des poids
    st.subheader("Comparaison des allocations")
    weights_compare = weights_df.melt(
        id_vars="Société",
        value_vars=["Poids Sharpe max", "Poids variance min"],
        var_name="Portefeuille",
        value_name="Poids"
    )
    fig_compare = px.bar(
        weights_compare[weights_compare["Poids"] > 0.01],
        x="Société",
        y="Poids",
        color="Portefeuille",
        barmode="group",
        title="Comparaison des allocations"
    )
    fig_compare.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_compare, use_container_width=True)

# ==================== TAB 5: FRONTIÈRE EFFICIENTE ====================
with tab5:
    st.header("📉 Frontière efficiente de Markowitz")
    
    fig_frontier = go.Figure()
    
    # Frontière
    fig_frontier.add_trace(go.Scatter(
        x=frontier_vols,
        y=frontier_returns,
        mode="lines+markers",
        name="Frontière efficiente",
        line=dict(color="blue", width=2),
        marker=dict(size=5, color="lightblue")
    ))
    
    # Portefeuille Sharpe max
    fig_frontier.add_trace(go.Scatter(
        x=[vol_sharpe],
        y=[ret_sharpe],
        mode="markers",
        name="Sharpe max",
        marker=dict(size=15, color="red", symbol="star")
    ))
    
    # Portefeuille variance min
    fig_frontier.add_trace(go.Scatter(
        x=[vol_minvar],
        y=[ret_minvar],
        mode="markers",
        name="Variance min",
        marker=dict(size=15, color="green", symbol="triangle-up")
    ))
    
    # Actifs individuels
    fig_frontier.add_trace(go.Scatter(
        x=volatility,
        y=mean_returns,
        mode="markers",
        name="Actifs individuels",
        marker=dict(size=10, color="gray", symbol="circle"),
        text=selected_societies,
        hoverinfo="text+x+y"
    ))
    
    fig_frontier.update_layout(
        title="Frontière efficiente",
        xaxis_title="Risque (Volatilité annualisée)",
        yaxis_title="Rendement annualisé",
        height=600,
        hovermode="closest"
    )
    fig_frontier.update_xaxes(tickformat=".0%")
    fig_frontier.update_yaxes(tickformat=".0%")
    
    st.plotly_chart(fig_frontier, use_container_width=True)
    
    st.info("""
    **Interprétation de la frontière efficiente :**
    - La courbe bleue représente l'ensemble des portefeuilles optimaux
    - Le point rouge ⭐ est le portefeuille qui maximise le ratio de Sharpe
    - Le point vert ▲ est le portefeuille de variance minimale
    - Les points gris sont les actifs individuels
    """)

# ==================== TAB 6: RECOMMANDATIONS ====================
with tab6:
    st.header("🤖 Recommandations intelligentes")
    
    st.info("Cette analyse est basée sur les données historiques et ne constitue pas un conseil financier personnalisé.")
    
    st.success(f"🏆 **Meilleure société selon le modèle : {best_society}**")
    
    st.subheader("Classement des actifs")
    st.dataframe(
        ranking.style.format({
            "Rentabilité annualisée": "{:.2%}",
            "Volatilité annualisée": "{:.2%}",
            "Sharpe individuel": "{:.3f}",
            "Max Drawdown": "{:.2%}",
            "VaR 95%": "{:.2%}",
            "Expected Shortfall": "{:.2%}",
            "Beta marché": "{:.3f}",
            "Score": "{:.0f}"
        }),
        use_container_width=True,
        height=400
    )
    
    # Graphique du score
    fig_score = px.bar(
        ranking.reset_index().rename(columns={"index": "Société"}),
        x="Société",
        y="Score",
        color="Recommandation",
        title="Score de qualité par actif"
    )
    st.plotly_chart(fig_score, use_container_width=True)
    
    # Carte de recommandation
    st.subheader("Carte des recommandations")
    reco_df = ranking.reset_index().rename(columns={"index": "Société"})
    fig_reco = px.scatter(
        reco_df,
        x="Volatilité annualisée",
        y="Rentabilité annualisée",
        color="Recommandation",
        size="Sharpe individuel",
        text="Société",
        title="Recommandations basées sur rendement/risque"
    )
    fig_reco.update_xaxes(tickformat=".0%")
    fig_reco.update_yaxes(tickformat=".0%")
    fig_reco.update_layout(height=500)
    st.plotly_chart(fig_reco, use_container_width=True)
    
    st.warning("""
    ⚠️ **Avertissement :** 
    - Ces recommandations sont basées uniquement sur l'analyse quantitative
    - Avant d'investir, vérifiez la liquidité, les frais, la fiscalité et la réglementation
    - Consultez un conseiller financier professionnel
    """)

# ==================== TAB 7: SIMULATION ====================
with tab7:
    st.header("💼 Simulation d'investissement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portefeuille Sharpe Max")
        invest_sharpe = weights_df[weights_df["Montant Sharpe max (TND)"] > 0][["Société", "Montant Sharpe max (TND)"]].copy()
        invest_sharpe = invest_sharpe.sort_values("Montant Sharpe max (TND)", ascending=False)
        st.dataframe(
            invest_sharpe.style.format({"Montant Sharpe max (TND)": "{:,.2f}"}),
            use_container_width=True
        )
        if len(invest_sharpe) > 0:
            fig_inv1 = px.pie(
                invest_sharpe.head(10),
                names="Société",
                values="Montant Sharpe max (TND)",
                title="Répartition du capital - Sharpe max"
            )
            st.plotly_chart(fig_inv1, use_container_width=True)
    
    with col2:
        st.subheader("Portefeuille Variance Min")
        invest_minvar = weights_df[weights_df["Montant variance min (TND)"] > 0][["Société", "Montant variance min (TND)"]].copy()
        invest_minvar = invest_minvar.sort_values("Montant variance min (TND)", ascending=False)
        st.dataframe(
            invest_minvar.style.format({"Montant variance min (TND)": "{:,.2f}"}),
            use_container_width=True
        )
        if len(invest_minvar) > 0:
            fig_inv2 = px.pie(
                invest_minvar.head(10),
                names="Société",
                values="Montant variance min (TND)",
                title="Répartition du capital - Variance min"
            )
            st.plotly_chart(fig_inv2, use_container_width=True)
    
    # Profil investisseur
    st.subheader("Recommandation selon votre profil")
    profile = st.selectbox(
        "Quel est votre profil ?",
        ["Prudent", "Équilibré", "Dynamique"]
    )
    
    if profile == "Prudent":
        st.success("✅ **Recommandation :** Portefeuille à variance minimale")
        st.write("Ce portefeuille privilégie la préservation du capital avec un risque minimal.")
        st.metric("Capital à investir", f"{capital:,.0f} TND")
        st.metric("Rentabilité attendue", f"{ret_minvar:.2%}")
        st.metric("Risque attendu", f"{vol_minvar:.2%}")
    elif profile == "Équilibré":
        st.success("✅ **Recommandation :** Mixte (50% Sharpe max + 50% Variance min)")
        st.write("Ce portefeuille équilibre rendement et risque.")
        mix_ret = (ret_sharpe + ret_minvar) / 2
        mix_vol = (vol_sharpe + vol_minvar) / 2
        st.metric("Capital à investir", f"{capital:,.0f} TND")
        st.metric("Rentabilité attendue", f"{mix_ret:.2%}")
        st.metric("Risque attendu", f"{mix_vol:.2%}")
    else:
        st.success("✅ **Recommandation :** Portefeuille Sharpe maximum")
        st.write("Ce portefeuille recherche le meilleur rendement ajusté au risque.")
        st.metric("Capital à investir", f"{capital:,.0f} TND")
        st.metric("Rentabilité attendue", f"{ret_sharpe:.2%}")
        st.metric("Risque attendu", f"{vol_sharpe:.2%}")

# ==================== TAB 8: EXPORT ====================
with tab8:
    st.header("📥 Export des résultats")
    
    st.info("Téléchargez un rapport Excel complet contenant toutes les analyses.")
    
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            selected_prices.to_excel(writer, sheet_name="1_Prix")
            returns.to_excel(writer, sheet_name="2_Rendements_journaliers")
            cumulative_returns.to_excel(writer, sheet_name="3_Rendements_cumules")
            metrics.to_excel(writer, sheet_name="4_Metriques_individuelles")
            ranking.to_excel(writer, sheet_name="5_Classement")
            weights_df.to_excel(writer, sheet_name="6_Allocation_portefeuille", index=False)
            cov_matrix.to_excel(writer, sheet_name="7_Matrice_covariance")
            corr_matrix.to_excel(writer, sheet_name="8_Matrice_correlation")
            drawdown.to_excel(writer, sheet_name="9_Drawdown")
            rolling_vol.to_excel(writer, sheet_name="10_Volatilite_glissante")
            
            # Statistiques du portefeuille
            portfolio_stats = pd.DataFrame({
                "Métrique": ["Rentabilité Sharpe max", "Risque Sharpe max", "Sharpe ratio",
                            "Rentabilité variance min", "Risque variance min", "Sharpe variance min"],
                "Valeur": [f"{ret_sharpe:.2%}", f"{vol_sharpe:.2%}", f"{sharpe_ratio:.4f}",
                          f"{ret_minvar:.2%}", f"{vol_minvar:.2%}", f"{sharpe_minvar:.4f}"]
            })
            portfolio_stats.to_excel(writer, sheet_name="11_Stats_portefeuille", index=False)
        
        st.download_button(
            label="📥 Télécharger le rapport Excel complet",
            data=output.getvalue(),
            file_name=f"markowitz_bvmt_{selected_annee}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"Erreur lors de la création du rapport : {e}")
    
    st.success("Le rapport contient :")
    st.markdown("""
    - ✅ Prix historiques
    - ✅ Rendements journaliers et cumulés
    - ✅ Métriques individuelles (Sharpe, VaR, ES, Beta, Drawdown)
    - ✅ Classement et recommandations
    - ✅ Allocation optimale des portefeuilles
    - ✅ Matrices de covariance et corrélation
    - ✅ Analyse des risques (Drawdown, volatilité glissante)
    - ✅ Statistiques des portefeuilles optimisés
    """)

# ==================== SIDEBAR FOOTER ====================
st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ **Disclaimer**\n\n"
    "Analyse à but éducatif uniquement. "
    "Ne constitue pas un conseil en investissement."
)
