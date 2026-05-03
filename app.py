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

st.set_page_config(page_title="Markowitz BVMT", layout="wide")

st.title("📊 Tableau de bord Markowitz BVMT")
st.write("Analyse financière avancée, optimisation de portefeuille")

# ==============================
# Chargement intelligent - UNE ANNÉE À LA FOIS
# ==============================

@st.cache_data
def load_single_year(file_path):
    """Charge un seul fichier Excel à la fois"""
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Erreur chargement {file_path}: {e}")
        return None


@st.cache_data
def process_data(df, selected_annee):
    """Traite les données d'une année spécifique"""
    if df is None:
        return None, None
    
    # Nettoyage
    df.columns = df.columns.str.strip()
    
    required_columns = ["SEANCE", "VALEUR", "CLOTURE"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return None, None
    
    # Sélection des colonnes
    data = df[["SEANCE", "VALEUR", "CLOTURE"]].copy()
    data.columns = ["Date", "Societe", "Close"]
    
    # Conversion
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
        return None, None
    
    # Filtrer par année
    data["Année"] = data["Date"].dt.year
    data = data[data["Année"] == selected_annee]
    
    if data.empty:
        return None, None
    
    # Pivot table
    prices = data.pivot_table(
        index="Date",
        columns="Societe",
        values="Close",
        aggfunc="last"
    ).sort_index().ffill()
    
    return data, prices


# ==============================
# Interface - Sélection de l'année
# ==============================

st.sidebar.header("📅 Sélection")

# Lister les fichiers disponibles
BASE_DIR = Path(__file__).parent
excel_files = {}

for file in BASE_DIR.iterdir():
    if file.is_file() and file.suffix.lower() in ['.xlsx', '.xls']:
        if not file.name.startswith('~'):
            # Extraire l'année du nom
            years = re.findall(r'\b(20\d{2})\b', file.stem)
            if years:
                excel_files[int(years[0])] = file

if not excel_files:
    st.error("Aucun fichier Excel trouvé!")
    st.stop()

# Sélection de l'année
annees_disponibles = sorted(excel_files.keys())
selected_annee = st.sidebar.selectbox(
    "Choisir l'année",
    options=annees_disponibles,
    index=len(annees_disponibles)-1
)

st.sidebar.success(f"📊 Année {selected_annee} sélectionnée")

# ==============================
# Chargement UNIQUEMENT de l'année sélectionnée
# ==============================

with st.spinner(f"Chargement des données {selected_annee}..."):
    file_path = excel_files[selected_annee]
    raw_data = load_single_year(file_path)
    
    if raw_data is None:
        st.error(f"Impossible de charger {selected_annee}")
        st.stop()
    
    data, prices = process_data(raw_data, selected_annee)
    
    if data is None or prices.empty:
        st.error(f"Aucune donnée valide pour {selected_annee}")
        st.stop()

st.sidebar.write(f"📈 {len(prices)} sociétés trouvées")

# ==============================
# Sélection des sociétés
# ==============================

st.sidebar.header("⚙️ Configuration")

toutes_societes = sorted(prices.columns.tolist())
selected_societies = st.sidebar.multiselect(
    "Choisir les sociétés (max 8 pour performance)",
    options=toutes_societes,
    default=toutes_societies[:min(5, len(toutes_societes))]
)

if len(selected_societies) < 2:
    st.warning("Sélectionnez au moins 2 sociétés")
    selected_societies = toutes_societies[:2]

# Limiter à 8 pour performance
if len(selected_societies) > 8:
    selected_societies = selected_societies[:8]
    st.sidebar.warning("Limité à 8 sociétés pour performance")

# Paramètres
rf = st.sidebar.number_input("Taux sans risque (%)", value=7.5, step=0.5) / 100
capital = st.sidebar.number_input("Capital (TND)", value=10000, step=5000)

# ==============================
# Calculs financiers
# ==============================

with st.spinner("Calculs en cours..."):
    selected_prices = prices[selected_societies].dropna(how="all").ffill()
    returns = selected_prices.pct_change().dropna()
    
    if returns.empty or len(returns) < 5:
        st.error("Pas assez de données")
        st.stop()
    
    # Métriques
    mean_returns = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    cov_matrix = returns.cov() * 252
    corr_matrix = returns.corr()
    
    cumulative_returns = (1 + returns).cumprod() - 1
    drawdown = selected_prices / selected_prices.cummax() - 1
    max_drawdown = drawdown.min()
    var_95 = returns.quantile(0.05) * np.sqrt(252)
    
    # Optimisation
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
        result = minimize(neg_sharpe, init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200})
        weights_opt = result.x if result.success else init
    except:
        weights_opt = init
    
    ret_opt = port_return(weights_opt)
    vol_opt = port_vol(weights_opt)
    sharpe_opt = (ret_opt - rf) / vol_opt if vol_opt > 0 else 0
    
    weights_df = pd.DataFrame({
        "Société": selected_societies,
        "Poids": weights_opt,
        "Montant (TND)": weights_opt * capital
    })
    weights_df = weights_df[weights_df["Poids"] > 0.001].sort_values("Poids", ascending=False)

# ==============================
# Métriques individuelles
# ==============================

metrics_df = pd.DataFrame({
    "Société": selected_societies,
    "Rentabilité": mean_returns.values,
    "Risque": volatility.values,
    "Sharpe": (mean_returns - rf).values / volatility.values,
    "Drawdown": max_drawdown.values,
    "VaR 95%": var_95.values
})
metrics_df = metrics_df.replace([np.inf, -np.inf], np.nan).fillna(0)

best_idx = metrics_df["Sharpe"].idxmax()
best_society = metrics_df.loc[best_idx, "Société"]

# ==============================
# INTERFACE
# ==============================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Vue générale",
    "📈 Performance",
    "⚠️ Risques",
    "🎯 Portefeuille"
])

with tab1:
    st.header(f"Analyse {selected_annee}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sociétés", len(selected_societies))
    c2.metric("Meilleure", best_society[:15])
    c3.metric("Sharpe optimal", f"{sharpe_opt:.3f}")
    c4.metric("Capital", f"{capital:,.0f} TND")
    
    # Graphique des cours
    fig_prices = px.line(selected_prices, title="Évolution des cours")
    fig_prices.update_layout(height=450, xaxis_title="Date", yaxis_title="Prix (TND)")
    st.plotly_chart(fig_prices, use_container_width=True)
    
    # Métriques
    st.subheader("Métriques par société")
    st.dataframe(
        metrics_df.style.format({
            "Rentabilité": "{:.2%}",
            "Risque": "{:.2%}",
            "Sharpe": "{:.3f}",
            "Drawdown": "{:.2%}",
            "VaR 95%": "{:.2%}"
        }),
        use_container_width=True
    )

with tab2:
    st.header("Performance")
    
    # Rendements cumulés
    fig_cum = px.line(cumulative_returns, title="Rendements cumulés")
    fig_cum.update_yaxes(tickformat=".0%")
    fig_cum.update_layout(height=450)
    st.plotly_chart(fig_cum, use_container_width=True)
    
    # Comparaison rendement/risque
    fig_scatter = px.scatter(
        metrics_df,
        x="Risque",
        y="Rentabilité",
        size="Sharpe",
        color="Sharpe",
        text="Société",
        title="Rendement vs Risque"
    )
    fig_scatter.update_xaxes(tickformat=".0%")
    fig_scatter.update_yaxes(tickformat=".0%")
    fig_scatter.update_layout(height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.header("Analyse des risques")
    
    # Drawdown
    fig_dd = px.line(drawdown, title="Drawdown")
    fig_dd.update_yaxes(tickformat=".0%")
    fig_dd.update_layout(height=400)
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # VaR
    fig_var = px.bar(metrics_df, x="Société", y="VaR 95%", title="Value at Risk (95%)")
    fig_var.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_var, use_container_width=True)
    
    # Matrice de corrélation
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Corrélations",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

with tab4:
    st.header("Portefeuille optimal")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Rentabilité", f"{ret_opt:.2%}")
    c2.metric("Risque", f"{vol_opt:.2%}")
    c3.metric("Sharpe", f"{sharpe_opt:.3f}")
    
    st.subheader("Allocation recommandée")
    st.dataframe(
        weights_df.style.format({"Poids": "{:.2%}", "Montant (TND)": "{:,.2f}"}),
        use_container_width=True
    )
    
    if len(weights_df) > 0:
        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(weights_df.head(8), x="Société", y="Poids", title="Allocation", color="Poids")
            fig_bar.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            fig_pie = px.pie(weights_df.head(8), names="Société", values="Poids", title="Répartition")
            st.plotly_chart(fig_pie, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("⚠️ Analyse éducative uniquement")
