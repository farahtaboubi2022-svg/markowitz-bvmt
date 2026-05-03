import io
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy import stats
import streamlit as st

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Markowitz BVMT - Optimisation de Portefeuille",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style personnalisé
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.title("📊 Tableau de bord Markowitz BVMT")
st.markdown("### Optimisation de portefeuille - Analyse financière avancée")
st.markdown("---")

# ==============================
# LISTE DES BANQUES TUNISIENNES
# ==============================

BANQUES = [
    "BIAT", "ATB", "STB", "BT", "AMEN BANK", "UIB", "UBCI", "BH",
    "BNA", "ATTIJARI BANK", "BH BANK", "BTE", "WIFACK INT BANK"
]

# ==============================
# FONCTIONS DE CHARGEMENT
# ==============================

@st.cache_data
def load_excel_file(file_path, year):
    """Charge un fichier Excel et filtre les banques"""
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        
        # Vérification des colonnes
        required_cols = ["SEANCE", "VALEUR", "CLOTURE"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Colonnes manquantes dans {year}: {missing_cols}")
            return None
        
        # Sélection des colonnes utiles
        df = df[["SEANCE", "VALEUR", "CLOTURE"]].copy()
        df.columns = ["Date", "Societe", "Close"]
        
        # Nettoyage des dates
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["Date"])
        
        # Nettoyage des prix
        df["Close"] = df["Close"].astype(str).str.replace(",", ".", regex=False)
        df["Close"] = df["Close"].astype(str).str.replace(" ", "", regex=False)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])
        df = df[df["Close"] > 0]
        
        # Filtrage par année
        df["Annee"] = df["Date"].dt.year
        df = df[df["Annee"] == year]
        
        # Filtrage des banques uniquement
        df = df[df["Societe"].str.upper().isin([b.upper() for b in BANQUES])]
        
        if df.empty:
            st.warning(f"Aucune banque trouvée pour {year}")
            return None
        
        return df
    
    except Exception as e:
        st.error(f"Erreur chargement {year}: {str(e)}")
        return None


@st.cache_data
def prepare_prices(data):
    """Prépare la matrice des prix"""
    if data is None or data.empty:
        return None
    
    prices = data.pivot_table(
        index="Date",
        columns="Societe",
        values="Close",
        aggfunc="first"
    ).sort_index().ffill()
    
    return prices


# ==============================
# FONCTIONS D'OPTIMISATION
# ==============================

def calculate_metrics(returns, rf):
    """Calcule les métriques financières"""
    mean_returns = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    sharpe = (mean_returns - rf) / volatility
    return mean_returns, volatility, sharpe


def optimize_portfolio(mean_returns, cov_matrix, rf):
    """Optimisation du portefeuille (Sharpe max)"""
    n = len(mean_returns)
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
        result = minimize(
            neg_sharpe, init, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        weights = result.x if result.success else init
    except:
        weights = init
    
    ret_opt = port_return(weights)
    vol_opt = port_vol(weights)
    sharpe_opt = (ret_opt - rf) / vol_opt if vol_opt > 0 else 0
    
    return weights, ret_opt, vol_opt, sharpe_opt


def calculate_var(returns, confidence=0.95):
    """Calcule la Value at Risk"""
    return returns.quantile(1 - confidence) * np.sqrt(252)


def calculate_cvar(returns, confidence=0.95):
    """Calcule la Conditional Value at Risk (Expected Shortfall)"""
    var = returns.quantile(1 - confidence)
    return returns[returns <= var].mean() * np.sqrt(252)


def calculate_beta(returns, market_returns):
    """Calcule le beta par rapport au marché"""
    if len(market_returns) > 1:
        cov = np.cov(returns, market_returns)[0, 1]
        var = np.var(market_returns)
        return cov / var if var != 0 else 1
    return 1


# ==============================
# CHARGEMENT DES DONNÉES
# ==============================

BASE_DIR = Path(__file__).parent
YEARS = [2023, 2024, 2025]

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# Sélection de l'année
selected_year = st.sidebar.selectbox(
    "📅 Année à analyser",
    YEARS,
    index=len(YEARS)-1,
    help="Sélectionnez l'année pour l'analyse"
)

# Recherche du fichier
file_path = None
for f in BASE_DIR.iterdir():
    if f.is_file() and f.suffix.lower() in ['.xlsx', '.xls']:
        if str(selected_year) in f.stem:
            file_path = f
            break

if file_path is None:
    st.error(f"❌ Fichier pour {selected_year} non trouvé!")
    st.info(f"Assurez-vous d'avoir un fichier nommé {selected_year}.xlsx dans le dossier")
    st.stop()

# Chargement
with st.spinner(f"📂 Chargement des données {selected_year}..."):
    data = load_excel_file(file_path, selected_year)
    
    if data is None or data.empty:
        st.error(f"❌ Aucune donnée valide pour {selected_year}")
        st.stop()
    
    prices = prepare_prices(data)
    
    if prices is None or prices.empty:
        st.error("❌ Erreur lors de la préparation des prix")
        st.stop()

# Affichage des banques disponibles
st.sidebar.success(f"✅ {len(prices.columns)} banques chargées")
with st.sidebar.expander("🏦 Banques disponibles"):
    for bank in sorted(prices.columns):
        st.write(f"• {bank}")

# Sélection des banques
selected_banks = st.sidebar.multiselect(
    "🏦 Banques à analyser",
    options=sorted(prices.columns.tolist()),
    default=sorted(prices.columns.tolist()),
    help="Sélectionnez les banques pour l'analyse"
)

if len(selected_banks) < 2:
    st.warning("⚠️ Veuillez sélectionner au moins 2 banques")
    selected_banks = sorted(prices.columns.tolist())[:2]

# Paramètres financiers
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Paramètres financiers")

rf = st.sidebar.number_input(
    "Taux sans risque (%)",
    value=7.5,
    step=0.5,
    help="Taux des obligations d'État tunisiennes"
) / 100

capital = st.sidebar.number_input(
    "Capital à investir (TND)",
    value=10000,
    step=5000,
    help="Montant total à investir"
)

# ==============================
# CALCULS PRINCIPAUX
# ==============================

with st.spinner("🔄 Calculs en cours..."):
    # Filtrage des prix
    selected_prices = prices[selected_banks].dropna(how="all").ffill()
    
    # Calcul des rendements
    returns = selected_prices.pct_change().dropna()
    
    if returns.empty or len(returns) < 10:
        st.error("❌ Pas assez de données pour l'analyse")
        st.stop()
    
    # Métriques annualisées
    mean_returns, volatility, sharpe_individual = calculate_metrics(returns, rf)
    cov_matrix = returns.cov() * 252
    corr_matrix = returns.corr()
    
    # Rendements cumulés
    cumulative_returns = (1 + returns).cumprod() - 1
    
    # Drawdown
    cummax = selected_prices.cummax()
    drawdown = (selected_prices - cummax) / cummax
    max_drawdown = drawdown.min()
    
    # VaR et CVaR
    var_90 = calculate_var(returns, 0.90)
    var_95 = calculate_var(returns, 0.95)
    var_99 = calculate_var(returns, 0.99)
    cvar_95 = calculate_cvar(returns, 0.95)
    
    # Beta marché
    market_returns = returns.mean(axis=1)
    betas = pd.Series({
        bank: calculate_beta(returns[bank], market_returns)
        for bank in returns.columns
    })
    
    # Optimisation du portefeuille
    weights_opt, ret_opt, vol_opt, sharpe_opt = optimize_portfolio(
        mean_returns, cov_matrix, rf
    )
    
    # Portfolio VaR
    portfolio_returns = returns.dot(weights_opt)
    portfolio_var_95 = calculate_var(portfolio_returns, 0.95)
    
    # DataFrame des poids
    weights_df = pd.DataFrame({
        "Banque": selected_banks,
        "Poids optimal": weights_opt,
        "Montant (TND)": weights_opt * capital
    })
    weights_df = weights_df[weights_df["Poids optimal"] > 0.001]
    weights_df = weights_df.sort_values("Poids optimal", ascending=False).reset_index(drop=True)
    
    # DataFrame des métriques
    metrics_df = pd.DataFrame({
        "Banque": selected_banks,
        "Rentabilité annualisée": mean_returns.values,
        "Volatilité annualisée": volatility.values,
        "Ratio de Sharpe": sharpe_individual.values,
        "Drawdown max (%)": max_drawdown.values * 100,
        "VaR 95%": var_95.values * 100,
        "Beta": betas.values
    })
    
    # Classement des banques
    metrics_df["Score"] = (
        metrics_df["Ratio de Sharpe"].rank(ascending=False) +
        metrics_df["Rentabilité annualisée"].rank(ascending=False) +
        metrics_df["Volatilité annualisée"].rank(ascending=True) +
        metrics_df["Drawdown max (%)"].rank(ascending=True) +
        metrics_df["VaR 95%"].rank(ascending=True)
    )
    metrics_df = metrics_df.sort_values("Score").reset_index(drop=True)
    best_bank = metrics_df.iloc[0]["Banque"] if len(metrics_df) > 0 else "N/A"
    
    # Recommandations
    def get_recommendation(row):
        if row["Ratio de Sharpe"] > 1 and row["Volatilité annualisée"] < metrics_df["Volatilité annualisée"].median():
            return "🟢 Très attractive"
        elif row["Ratio de Sharpe"] > 0.5:
            return "🟡 Intéressante"
        elif row["Volatilité annualisée"] > metrics_df["Volatilité annualisée"].median():
            return "🔴 Risque élevé"
        else:
            return "⚪ À surveiller"
    
    metrics_df["Recommandation"] = metrics_df.apply(get_recommendation, axis=1)

# ==============================
# INTERFACE PRINCIPALE
# ==============================

st.header(f"📊 Analyse {selected_year} - Portefeuille bancaire optimal")

# KPIS
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("🏦 Banques", len(selected_banks), delta=None)
with col2:
    st.metric("📈 Rentabilité", f"{ret_opt:.2%}", delta_color="normal")
with col3:
    st.metric("⚠️ Risque", f"{vol_opt:.2%}", delta_color="inverse")
with col4:
    st.metric("🎯 Sharpe", f"{sharpe_opt:.3f}")
with col5:
    st.metric("🏆 Meilleure", best_bank[:15])

st.markdown("---")

# ========== TAB 1: VUE GÉNÉRALE ==========
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Vue générale",
    "📊 Métriques",
    "⚠️ Risques",
    "🎯 Optimisation",
    "🔗 Corrélations",
    "📥 Export"
])

with tab1:
    # Évolution des cours
    st.subheader("📈 Évolution des cours")
    fig_prices = px.line(
        selected_prices,
        title=f"Cours de clôture - {selected_year}",
        labels={"value": "Prix (TND)", "variable": "Banque"}
    )
    fig_prices.update_layout(height=450, hovermode="x unified")
    st.plotly_chart(fig_prices, use_container_width=True)
    
    # Rendements cumulés
    st.subheader("📊 Performance cumulée")
    fig_cum = px.line(
        cumulative_returns,
        title="Rendements cumulés",
        labels={"value": "Rendement", "variable": "Banque"}
    )
    fig_cum.update_yaxes(tickformat=".0%")
    fig_cum.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig_cum, use_container_width=True)
    
    # Carte rendement/risque
    st.subheader("🗺️ Carte rendement / risque")
    fig_scatter = px.scatter(
        metrics_df,
        x="Volatilité annualisée",
        y="Rentabilité annualisée",
        size="Ratio de Sharpe",
        color="Ratio de Sharpe",
        text="Banque",
        title="Positionnement des banques",
        labels={"Volatilité annualisée": "Risque", "Rentabilité annualisée": "Rendement"}
    )
    fig_scatter.update_xaxes(tickformat=".0%")
    fig_scatter.update_yaxes(tickformat=".0%")
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    # Tableau des métriques
    st.subheader("📊 Métriques détaillées par banque")
    st.dataframe(
        metrics_df.style.format({
            "Rentabilité annualisée": "{:.2%}",
            "Volatilité annualisée": "{:.2%}",
            "Ratio de Sharpe": "{:.3f}",
            "Drawdown max (%)": "{:.1f}%",
            "VaR 95%": "{:.1f}%",
            "Beta": "{:.2f}",
            "Score": "{:.0f}"
        }).background_gradient(subset=["Ratio de Sharpe"], cmap="RdYlGn"),
        use_container_width=True,
        height=400
    )
    
    # Graphiques individuels
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig_ret = px.bar(
            metrics_df,
            x="Banque",
            y="Rentabilité annualisée",
            title="Rentabilité par banque",
            color="Rentabilité annualisée",
            color_continuous_scale="Viridis"
        )
        fig_ret.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_ret, use_container_width=True)
        
        fig_sharpe = px.bar(
            metrics_df,
            x="Banque",
            y="Ratio de Sharpe",
            title="Ratio de Sharpe",
            color="Ratio de Sharpe",
            color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig_sharpe, use_container_width=True)
    
    with col_b:
        fig_vol = px.bar(
            metrics_df,
            x="Banque",
            y="Volatilité annualisée",
            title="Risque par banque",
            color="Volatilité annualisée",
            color_continuous_scale="OrRd"
        )
        fig_vol.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_vol, use_container_width=True)
        
        fig_beta = px.bar(
            metrics_df,
            x="Banque",
            y="Beta",
            title="Beta (risque systématique)",
            color="Beta",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_beta, use_container_width=True)

with tab3:
    # Drawdown
    st.subheader("📉 Drawdown (perte maximale)")
    fig_dd = px.area(
        drawdown,
        title="Drawdown des banques",
        labels={"value": "Perte (%)", "variable": "Banque"}
    )
    fig_dd.update_yaxes(tickformat=".0%")
    fig_dd.update_layout(height=450, hovermode="x unified")
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # VaR
    st.subheader("📊 Value at Risk (VaR) à 95%")
    fig_var = px.bar(
        metrics_df,
        x="Banque",
        y="VaR 95%",
        title="Perte maximale attendue (95% de confiance)",
        color="VaR 95%",
        color_continuous_scale="Reds"
    )
    fig_var.update_yaxes(tickformat=".1f")
    st.plotly_chart(fig_var, use_container_width=True)
    
    # Distribution des rendements
    st.subheader("📈 Distribution des rendements")
    selected_bank_var = st.selectbox("Choisir une banque", selected_banks, key="var_select")
    
    if selected_bank_var:
        returns_bank = returns[selected_bank_var].dropna()
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=returns_bank,
            name="Rendements",
            nbinsx=50,
            opacity=0.7,
            marker_color="lightblue"
        ))
        
        # Ajout des lignes VaR
        var_vals = [returns_bank.quantile(0.10), returns_bank.quantile(0.05), returns_bank.quantile(0.01)]
        colors = ["orange", "red", "darkred"]
        labels = ["VaR 90%", "VaR 95%", "VaR 99%"]
        
        for var_val, color, label in zip(var_vals, colors, labels):
            fig_dist.add_vline(
                x=var_val, line_dash="dash", line_color=color,
                annotation_text=f"{label}: {var_val:.2%}"
            )
        
        # Courbe de densité
        try:
            kde = stats.gaussian_kde(returns_bank)
            x_range = np.linspace(returns_bank.min(), returns_bank.max(), 100)
            y_range = kde(x_range) * len(returns_bank) * (returns_bank.max() - returns_bank.min()) / 50
            fig_dist.add_trace(go.Scatter(
                x=x_range, y=y_range, name="Densité", line=dict(color="blue", width=2)
            ))
        except:
            pass
        
        fig_dist.update_layout(
            title=f"Distribution des rendements - {selected_bank_var}",
            xaxis_title="Rendement journalier",
            yaxis_title="Fréquence",
            height=500
        )
        fig_dist.update_xaxes(tickformat=".1%")
        st.plotly_chart(fig_dist, use_container_width=True)

with tab4:
    # Portefeuille optimal
    st.subheader("🎯 Portefeuille optimal (Sharpe maximum)")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Rentabilité annualisée", f"{ret_opt:.2%}")
    with col_b:
        st.metric("Volatilité annualisée", f"{vol_opt:.2%}")
    with col_c:
        st.metric("Ratio de Sharpe", f"{sharpe_opt:.3f}")
    
    st.markdown("---")
    
    # Allocation
    col_pie, col_table = st.columns([1, 1.5])
    
    with col_pie:
        if len(weights_df) > 0:
            fig_pie = px.pie(
                weights_df,
                names="Banque",
                values="Poids optimal",
                title="Répartition du portefeuille",
                hole=0.3
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(height=450)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_table:
        st.dataframe(
            weights_df.style.format({
                "Poids optimal": "{:.2%}",
                "Montant (TND)": "{:,.2f}"
            }).bar(subset=["Poids optimal"], color="#2ecc71"),
            use_container_width=True,
            height=400
        )
    
    # Graphique d'allocation
    fig_alloc = px.bar(
        weights_df,
        x="Banque",
        y="Poids optimal",
        title="Allocation par banque",
        color="Poids optimal",
        color_continuous_scale="Viridis",
        text_auto=".1%"
    )
    fig_alloc.update_yaxes(tickformat=".0%")
    fig_alloc.update_layout(height=450)
    st.plotly_chart(fig_alloc, use_container_width=True)

with tab5:
    # Matrice de corrélation
    st.subheader("🔗 Matrice de corrélation")
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Corrélations entre banques",
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Heatmap colorée
    st.subheader("🎨 Matrice de covariance annualisée")
    fig_cov = px.imshow(
        cov_matrix,
        text_auto=True,
        aspect="auto",
        title="Covariance annualisée",
        color_continuous_scale="Viridis"
    )
    fig_cov.update_layout(height=600)
    st.plotly_chart(fig_cov, use_container_width=True)

with tab6:
    # Export
    st.subheader("📥 Export des résultats")
    st.info("Téléchargez un rapport Excel complet avec toutes les analyses")
    
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            selected_prices.to_excel(writer, sheet_name="1_Prix")
            returns.to_excel(writer, sheet_name="2_Rendements")
            cumulative_returns.to_excel(writer, sheet_name="3_Rendements_cumules")
            metrics_df.to_excel(writer, sheet_name="4_Metriques_banques", index=False)
            weights_df.to_excel(writer, sheet_name="5_Allocation_portefeuille", index=False)
            cov_matrix.to_excel(writer, sheet_name="6_Matrice_covariance")
            corr_matrix.to_excel(writer, sheet_name="7_Matrice_correlation")
            drawdown.to_excel(writer, sheet_name="8_Drawdown")
            
            # Portfolio stats
            portfolio_stats = pd.DataFrame({
                "Métrique": [
                    "Rentabilité annualisée", "Volatilité annualisée", "Ratio de Sharpe",
                    "VaR 95%", "Capital investi", "Nombre de banques",
                    "Taux sans risque", "Année analysée"
                ],
                "Valeur": [
                    f"{ret_opt:.2%}", f"{vol_opt:.2%}", f"{sharpe_opt:.3f}",
                    f"{portfolio_var_95:.2%}", f"{capital:,.0f} TND",
                    len(weights_df), f"{rf:.1%}", selected_year
                ]
            })
            portfolio_stats.to_excel(writer, sheet_name="9_Stats_portefeuille", index=False)
        
        st.download_button(
            label="📥 Télécharger le rapport Excel complet",
            data=output.getvalue(),
            file_name=f"markowitz_bvmt_{selected_year}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Erreur lors de la création du rapport: {e}")
    
    st.markdown("---")
    st.success("✅ Le rapport contient toutes les analyses:")
    st.markdown("""
    - 📈 Prix et rendements historiques
    - 📊 Métriques individuelles (Sharpe, VaR, Beta)
    - 🎯 Allocation optimale du portefeuille
    - 🔗 Matrices de covariance et corrélation
    - 📉 Analyse des risques (Drawdown)
    - 💼 Statistiques du portefeuille optimal
    """)

# ==============================
# RECOMMANDATIONS FINALES
# ==============================

st.markdown("---")
st.subheader("🤖 Recommandations intelligentes")

col_rec1, col_rec2, col_rec3 = st.columns(3)

with col_rec1:
    st.info(f"🏆 **Meilleure banque**\n\n{best_bank}\n\n*Basé sur le score composite*")

with col_rec2:
    if sharpe_opt > 1:
        st.success(f"📈 **Excellent ratio de Sharpe**\n\n{sharpe_opt:.3f}\n\nLe portefeuille offre un excellent rendement ajusté au risque")
    elif sharpe_opt > 0.5:
        st.info(f"📊 **Bon ratio de Sharpe**\n\n{sharpe_opt:.3f}\n\nLe portefeuille offre un bon rendement ajusté au risque")
    else:
        st.warning(f"⚠️ **Ratio de Sharpe faible**\n\n{sharpe_opt:.3f}\n\nLe rendement ne compense pas suffisamment le risque")

with col_rec3:
    st.warning("⚠️ **Avertissement**\n\nCette analyse est basée sur des données historiques et ne constitue pas un conseil en investissement")

# Disclaimer
st.markdown("---")
st.caption(
    "📊 **Méthodologie:** Analyse Markowitz | Annualisation: 252 jours | "
    "VaR 95% historique | Données: BVMT | © 2024"
)
