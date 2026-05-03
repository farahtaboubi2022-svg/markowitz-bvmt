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
            # Nettoyer les noms de colonnes
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


def get_all_societies(prices):
    """Récupère toutes les sociétés disponibles dans les données"""
    all_societies = sorted(prices.columns.tolist())
    return all_societies


# ==============================
# Chargement fichiers Excel
# ==============================

BASE_DIR = Path(__file__).parent

# Lister tous les fichiers Excel
excel_files = []
for file in BASE_DIR.iterdir():
    if file.is_file() and file.suffix.lower() in ['.xlsx', '.xls']:
        if not file.name.startswith('~') and not file.name.startswith('.'):
            excel_files.append(file)

if not excel_files:
    st.error("Aucun fichier Excel trouvé!")
    st.stop()

# Charger tous les fichiers
try:
    data_raw = load_excel_files(excel_files)
    if data_raw is None:
        st.error("Aucune donnée chargée.")
        st.stop()
except Exception as e:
    st.error(f"Erreur lors du chargement : {e}")
    st.stop()

# Préparer les données
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
# Sidebar - Sélection de l'année
# ==============================

st.sidebar.header("📅 Sélection de la période")

options_annees = [str(a) for a in annees_disponibles]
selected_annee_str = st.sidebar.selectbox(
    "Choisir l'année à analyser",
    options=options_annees,
    index=len(options_annees)-1 if options_annees else 0
)

selected_annee = int(selected_annee_str)

# Filtrer les données par année
data_filtered = data[data["Année"] == selected_annee].copy()

if data_filtered.empty:
    st.error(f"Aucune donnée pour l'année {selected_annee}")
    st.stop()

# Recréer les prix pour l'année sélectionnée
prices_filtered = data_filtered.pivot_table(
    index="Date",
    columns="Societe",
    values="Close",
    aggfunc="last"
).sort_index().ffill()

# ==============================
# Récupérer TOUTES les sociétés disponibles
# ==============================

toutes_societes = get_all_societies(prices_filtered)

st.sidebar.write(f"📊 **{len(toutes_societes)} sociétés disponibles en {selected_annee}**")

# ==============================
# Sidebar - Sélection des sociétés
# ==============================

st.sidebar.header("⚙️ Sélection des sociétés")

# Options de sélection
option_select = st.sidebar.radio(
    "Mode de sélection",
    ["Toutes les sociétés", "Sélection manuelle", "Banques uniquement", "Top 10 par capitalisation"]
)

if option_select == "Toutes les sociétés":
    selected_banques = toutes_societes
    st.sidebar.info(f"✅ {len(selected_banques)} sociétés sélectionnées")
    
elif option_select == "Sélection manuelle":
    selected_banques = st.sidebar.multiselect(
        f"Choisir les sociétés ({selected_annee})",
        options=toutes_societes,
        default=toutes_societes[:min(10, len(toutes_societes))]
    )
    
elif option_select == "Banques uniquement":
    # Liste des banques à rechercher
    banques_liste = [
        "BIAT", "ATB", "STB", "BT", "AMEN BANK", "UIB", "UBCI", "BH",
        "BNA", "ATTIJARI BANK", "BH BANK", "BTE (ADP)", "WIFACK INT BANK",
        "ATB", "BT", "BH BANK"
    ]
    
    # Filtrer les banques présentes
    selected_banques = []
    for b in banques_liste:
        matching = [col for col in toutes_societes if b.upper() in col.upper() or col.upper() in b.upper()]
        selected_banques.extend(matching)
    
    selected_banques = list(set(selected_banques))  # Supprimer les doublons
    selected_banques.sort()
    
    if not selected_banques:
        selected_banques = toutes_societes[:10]
        st.sidebar.warning("Aucune banque trouvée, affichage des 10 premières sociétés")
    else:
        st.sidebar.info(f"✅ {len(selected_banques)} banques trouvées")
    
elif option_select == "Top 10 par capitalisation":
    # Essayer d'obtenir la capitalisation si disponible
    if "CAPITAUX" in data_filtered.columns:
        caps = data_filtered.groupby("Societe")["CAPITAUX"].last().sort_values(ascending=False)
        selected_banques = caps.head(10).index.tolist()
    else:
        selected_banques = toutes_societes[:10]
    st.sidebar.info(f"✅ Top {len(selected_banques)} sociétés sélectionnées")

# Vérifier qu'on a au moins 2 sociétés
if len(selected_banques) < 2:
    st.warning("⚠️ Veuillez sélectionner au moins 2 sociétés pour l'analyse.")
    selected_banques = toutes_societes[:2]
    st.info(f"📊 {len(selected_banques)} sociétés sélectionnées par défaut")

# ==============================
# Paramètres d'analyse
# ==============================

st.sidebar.header("⚙️ Paramètres financiers")

rf = st.sidebar.number_input("Taux sans risque annuel (%)", value=7.5, step=0.1) / 100
capital = st.sidebar.number_input("Capital à investir (TND)", value=10000, step=1000)

# ==============================
# Calculs financiers
# ==============================

try:
    selected_prices = prices_filtered[selected_banques].dropna(how="all").ffill()
    
    if len(selected_prices) < 2:
        st.error("Pas assez de données de prix")
        st.stop()
    
    returns = selected_prices.pct_change().dropna()
    
    if returns.empty or len(returns) < 2:
        st.error("Pas assez de données de rendements")
        st.stop()
    
    # Calculs annualisés (252 jours de bourse)
    mean_returns = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    cov_matrix = returns.cov() * 252
    corr_matrix = returns.corr()
    
    cumulative_returns = (1 + returns).cumprod() - 1
    drawdown = selected_prices / selected_prices.cummax() - 1
    max_drawdown = drawdown.min()
    
    # Métriques individuelles
    metrics = pd.DataFrame({
        "Rentabilité annualisée": mean_returns,
        "Volatilité annualisée": volatility,
        "Sharpe individuel": (mean_returns - rf) / volatility,
        "Max Drawdown": max_drawdown
    })
    
    metrics = metrics.replace([np.inf, -np.inf], np.nan).dropna()
    
except Exception as e:
    st.error(f"Erreur lors des calculs : {e}")
    st.stop()

# ==============================
# Optimisation du portefeuille
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

constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds = tuple((0, 1) for _ in range(n))

try:
    result = minimize(neg_sharpe, init, method="SLSQP", bounds=bounds, constraints=constraints)
    
    if result.success:
        weights_optimal = result.x
        ret_optimal = port_return(weights_optimal)
        vol_optimal = port_vol(weights_optimal)
        sharpe_optimal = (ret_optimal - rf) / vol_optimal if vol_optimal > 0 else 0
        
        # Créer le dataframe des poids
        weights_df = pd.DataFrame({
            "Société": selected_banques,
            "Poids optimal": weights_optimal,
            "Montant à investir (TND)": weights_optimal * capital
        })
        
        # Filtrer les poids positifs et trier
        weights_df = weights_df[weights_df["Poids optimal"] > 0.001].sort_values("Poids optimal", ascending=False)
        
        # Statistiques du portefeuille
        portfolio_stats = {
            "Rentabilité annualisée": ret_optimal,
            "Volatilité annualisée": vol_optimal,
            "Ratio de Sharpe": sharpe_optimal,
            "Nombre d'actifs": len(weights_df)
        }
    else:
        st.error("Optimisation échouée")
        st.stop()
        
except Exception as e:
    st.error(f"Erreur lors de l'optimisation : {e}")
    st.stop()

# ==============================
# Interface principale
# ==============================

st.header(f"📊 Optimisation de portefeuille - Année {selected_annee}")

# Métriques principales
col1, col2, col3, col4 = st.columns(4)
col1.metric("📈 Rentabilité", f"{portfolio_stats['Rentabilité annualisée']:.2%}")
col2.metric("⚠️ Risque", f"{portfolio_stats['Volatilité annualisée']:.2%}")
col3.metric("🎯 Ratio de Sharpe", f"{portfolio_stats['Ratio de Sharpe']:.3f}")
col4.metric("🏦 Actifs sélectionnés", portfolio_stats["Nombre d'actifs"])

# Graphique des cours
st.subheader("📈 Évolution des cours")
fig_prices = px.line(selected_prices, title=f"Cours de clôture - {selected_annee}")
fig_prices.update_layout(xaxis_title="Date", yaxis_title="Prix (TND)", height=500)
st.plotly_chart(fig_prices, use_container_width=True)

# Graphique des rendements cumulés
st.subheader("📊 Performance cumulée")
fig_returns = px.line(cumulative_returns, title="Rendements cumulés")
fig_returns.update_yaxes(tickformat=".0%")
fig_returns.update_layout(height=400)
st.plotly_chart(fig_returns, use_container_width=True)

# Allocation du portefeuille
st.subheader("🎯 Allocation optimale du portefeuille")

col_left, col_right = st.columns([2, 1])

with col_left:
    # Graphique en barres des poids
    fig_allocation = px.bar(
        weights_df.head(15),
        x="Société",
        y="Poids optimal",
        title="Répartition du portefeuille",
        color="Poids optimal",
        color_continuous_scale="Viridis"
    )
    fig_allocation.update_yaxes(tickformat=".0%")
    fig_allocation.update_layout(height=500)
    st.plotly_chart(fig_allocation, use_container_width=True)

with col_right:
    # Graphique en camembert (top 10)
    if len(weights_df) > 1:
        fig_pie = px.pie(
            weights_df.head(10),
            names="Société",
            values="Poids optimal",
            title="Répartition (Top 10)"
        )
        fig_pie.update_layout(height=500)
        st.plotly_chart(fig_pie, use_container_width=True)

# Tableau détaillé
st.subheader("📋 Détail de l'allocation")
st.dataframe(
    weights_df.style.format({
        "Poids optimal": "{:.2%}",
        "Montant à investir (TND)": "{:,.2f}"
    }),
    use_container_width=True
)

# Métriques individuelles
st.subheader("📊 Métriques individuelles par société")
st.dataframe(
    metrics.style.format({
        "Rentabilité annualisée": "{:.2%}",
        "Volatilité annualisée": "{:.2%}",
        "Sharpe individuel": "{:.3f}",
        "Max Drawdown": "{:.2%}"
    }),
    use_container_width=True
)

# Matrice de corrélation
st.subheader("🔗 Matrice de corrélation")
fig_corr = px.imshow(
    corr_matrix,
    text_auto=True,
    aspect="auto",
    title="Corrélations entre les sociétés",
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1
)
fig_corr.update_layout(height=600)
st.plotly_chart(fig_corr, use_container_width=True)

# Frontière efficiente (simplifiée)
st.subheader("📉 Frontière efficiente")
try:
    frontier_returns = []
    frontier_risks = []
    
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 20)
    
    for target in target_returns:
        constraints_frontier = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: port_return(w) - t}
        )
        
        result_frontier = minimize(
            port_vol,
            init,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_frontier
        )
        
        if result_frontier.success:
            frontier_returns.append(port_return(result_frontier.x))
            frontier_risks.append(port_vol(result_frontier.x))
    
    fig_frontier = go.Figure()
    
    # Frontière
    fig_frontier.add_trace(go.Scatter(
        x=frontier_risks,
        y=frontier_returns,
        mode="lines+markers",
        name="Frontière efficiente",
        line=dict(color="blue", width=2),
        marker=dict(size=5)
    ))
    
    # Portefeuille optimal
    fig_frontier.add_trace(go.Scatter(
        x=[vol_optimal],
        y=[ret_optimal],
        mode="markers",
        name="Portefeuille optimal",
        marker=dict(size=15, color="red", symbol="star")
    ))
    
    fig_frontier.update_layout(
        title="Frontière efficiente de Markowitz",
        xaxis_title="Risque (Volatilité annualisée)",
        yaxis_title="Rendement annualisé",
        height=500
    )
    
    fig_frontier.update_xaxes(tickformat=".0%")
    fig_frontier.update_yaxes(tickformat=".0%")
    
    st.plotly_chart(fig_frontier, use_container_width=True)
    
except Exception as e:
    st.info("Frontière efficiente non disponible pour cette sélection")

# Téléchargement du rapport
st.subheader("📥 Export des résultats")

try:
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        selected_prices.to_excel(writer, sheet_name="Prix")
        returns.to_excel(writer, sheet_name="Rendements")
        cumulative_returns.to_excel(writer, sheet_name="Rendements_cumules")
        metrics.to_excel(writer, sheet_name="Metriques_individuelles")
        weights_df.to_excel(writer, sheet_name="Allocation_portefeuille", index=False)
        corr_matrix.to_excel(writer, sheet_name="Correlations")
        
        # Ajouter les statistiques du portefeuille
        stats_df = pd.DataFrame([portfolio_stats]).T
        stats_df.columns = ["Valeur"]
        stats_df.index = ["Rentabilité annualisée", "Volatilité annualisée", "Ratio de Sharpe", "Nombre d'actifs"]
        stats_df.to_excel(writer, sheet_name="Statistiques_portefeuille")
    
    st.download_button(
        label="📥 Télécharger le rapport Excel complet",
        data=output.getvalue(),
        file_name=f"markowitz_portfolio_{selected_annee}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
except Exception as e:
    st.error(f"Erreur lors de la création du rapport : {e}")

# Disclaimer
st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ **Disclaimer**\n\n"
    "Cette analyse est à but éducatif uniquement. "
    "Elle ne constitue pas un conseil en investissement. "
    "Consultez un professionnel avant toute décision d'investissement."
)
