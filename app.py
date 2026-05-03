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

st.set_page_config(page_title="Markowitz BVMT - Tableau de bord", layout="wide")

st.title("📊 Tableau de bord Markowitz BVMT")
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
            pass
    
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
    selected_societies = banques_disponibles[:10]  # Limite à 10 pour performance
elif option_select == "Toutes les sociétés":
    selected_societies = toutes_societes[:10]
else:
    selected_societies = st.sidebar.multiselect(
        "Choisir les sociétés",
        options=toutes_societes,
        default=toutes_societies[:min(5, len(toutes_societes))]
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

if returns.empty or len(returns) < 5:
    st.error("Pas assez de données de rendements")
    st.stop()

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

# VaR
var_90 = returns.quantile(0.10) * np.sqrt(252)
var_95 = returns.quantile(0.05) * np.sqrt(252)
var_99 = returns.quantile(0.01) * np.sqrt(252)

# Expected Shortfall
expected_shortfall_95 = returns[returns.le(returns.quantile(0.05))].mean() * np.sqrt(252)

# Beta
market_return = returns.mean(axis=1)
beta = {}
for col in returns.columns:
    if len(market_return) > 1:
        cov = np.cov(returns[col], market_return)[0][1]
        var = np.var(market_return)
        beta[col] = cov / var if var != 0 else 1
    else:
        beta[col] = 1
beta = pd.Series(beta)

# Métriques
metrics = pd.DataFrame({
    "Rentabilité": mean_returns,
    "Risque": volatility,
    "Sharpe": (mean_returns - rf) / volatility,
    "Max Drawdown": max_drawdown,
    "VaR 90%": var_90,
    "VaR 95%": var_95,
    "VaR 99%": var_99,
    "ES 95%": expected_shortfall_95,
    "Beta": beta
})
metrics = metrics.replace([np.inf, -np.inf], np.nan).fillna(0)

# ==============================
# Optimisation simplifiée
# ==============================

n = len(selected_societies)
init = np.ones(n) / n

def port_return(w):
    return np.sum(w * mean_returns)

def port_vol(w):
    return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

def neg_sharpe(w):
    vol = port_vol(w)
    if vol < 0.0001:
        return 999
    return -(port_return(w) - rf) / vol

constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds = tuple((0, 1) for _ in range(n))

try:
    result = minimize(neg_sharpe, init, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        weights_opt = result.x
    else:
        weights_opt = init
except:
    weights_opt = init

ret_opt = port_return(weights_opt)
vol_opt = port_vol(weights_opt)
sharpe_opt = (ret_opt - rf) / vol_opt if vol_opt > 0 else 0

weights_df = pd.DataFrame({
    "Société": selected_societies,
    "Poids optimal": weights_opt,
    "Montant (TND)": weights_opt * capital
})
weights_df = weights_df[weights_df["Poids optimal"] > 0.001].sort_values("Poids optimal", ascending=False)

# ==============================
# Classement
# ==============================

ranking = metrics.copy()
ranking["Score"] = (
    ranking["Sharpe"].rank(ascending=False) +
    ranking["Rentabilité"].rank(ascending=False) +
    ranking["Risque"].rank(ascending=True) +
    ranking["Max Drawdown"].rank(ascending=False)
)
ranking = ranking.sort_values("Score")
best_society = ranking.index[0] if len(ranking) > 0 else "N/A"

def get_recommendation(row):
    if row["Sharpe"] > 1:
        return "🟢 Très attractive"
    elif row["Sharpe"] > 0.5:
        return "🟡 Intéressante"
    else:
        return "🟠 À surveiller"

ranking["Recommandation"] = ranking.apply(get_recommendation, axis=1)

# ==============================
# INTERFACE
# ==============================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Vue générale",
    "📈 Indicateurs",
    "⚠️ Risques",
    "🎯 Optimisation",
    "🤖 Recommandations"
])

with tab1:
    st.header(f"Vue générale - {selected_annee}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actifs", len(selected_societies))
    c2.metric("Meilleur", str(best_society)[:15])
    c3.metric("Sharpe", f"{sharpe_opt:.3f}")
    c4.metric("Capital", f"{capital:,.0f}")
    
    fig_prices = px.line(selected_prices, title="Cours")
    fig_prices.update_layout(height=400)
    st.plotly_chart(fig_prices, use_container_width=True)
    
    fig_cum = px.line(cumulative_returns, title="Rendements cumulés")
    fig_cum.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_cum, use_container_width=True)

with tab2:
    st.header("Indicateurs")
    st.dataframe(metrics.style.format("{:.2%}" if c in ["Rentabilité", "Risque", "Max Drawdown", "VaR 90%", "VaR 95%", "VaR 99%", "ES 95%"] else "{:.3f}" for c in metrics.columns), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_ret = px.bar(metrics, y="Rentabilité", title="Rentabilité")
        fig_ret.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_ret)
    with col2:
        fig_var = px.bar(metrics, y="VaR 95%", title="VaR 95%")
        fig_var.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_var)

with tab3:
    st.header("Analyse des risques")
    
    fig_dd = px.line(drawdown, title="Drawdown")
    fig_dd.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_dd)
    
    fig_corr = px.imshow(corr_matrix, text_auto=True, title="Corrélations", aspect="auto")
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr)

with tab4:
    st.header("Portefeuille optimal")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Rentabilité", f"{ret_opt:.2%}")
    c2.metric("Risque", f"{vol_opt:.2%}")
    c3.metric("Sharpe", f"{sharpe_opt:.3f}")
    
    st.subheader("Allocation")
    st.dataframe(weights_df.style.format({"Poids optimal": "{:.2%}", "Montant (TND)": "{:,.2f}"}), use_container_width=True)
    
    if len(weights_df) > 0:
        fig_pie = px.pie(weights_df.head(8), names="Société", values="Poids optimal", title="Répartition")
        st.plotly_chart(fig_pie)

with tab5:
    st.header("Recommandations")
    st.success(f"🏆 Meilleure société : {best_society}")
    st.dataframe(ranking[["Rentabilité", "Risque", "Sharpe", "Recommandation"]].style.format({"Rentabilité": "{:.2%}", "Risque": "{:.2%}", "Sharpe": "{:.3f}"}), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("⚠️ Analyse éducative uniquement")
