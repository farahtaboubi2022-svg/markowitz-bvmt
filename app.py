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
            # Afficher le fichier en cours de chargement
            st.sidebar.write(f"📂 Chargement: {file.name}")
            
            # Lecture de la première feuille
            df = pd.read_excel(file)
            df["Source"] = file.stem  # Utiliser stem pour le nom du fichier
            all_data.append(df)
            
            st.sidebar.write(f"   ✅ {len(df)} lignes chargées")
            
        except Exception as e:
            st.sidebar.warning(f"⚠️ Erreur {file.name}: {e}")
    
    if not all_data:
        return None
    
    result = pd.concat(all_data, ignore_index=True)
    st.sidebar.success(f"📊 Total: {len(result)} lignes combinées")
    return result


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
    
    # Convertir en entier Python natif
    data["Année"] = data["Année"].astype(int)

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

# Afficher le chemin de recherche
st.sidebar.write("🔍 Recherche dans:", BASE_DIR)

# Liste tous les fichiers Excel du dossier
all_excel_files = []
for pattern in ['*.xlsx', '*.xls']:
    all_excel_files.extend(BASE_DIR.glob(pattern))

# Filtrer pour éviter les doublons et fichiers temporaires
excel_files = []
for file in all_excel_files:
    filename = str(file.name).lower()
    # Ignorer les fichiers temporaires et doublons
    if not filename.startswith('~') and not filename.startswith('.'):
        if file not in excel_files:
            excel_files.append(file)

# Afficher tous les fichiers trouvés
st.sidebar.write("📁 Fichiers trouvés:")
for f in excel_files:
    st.sidebar.write(f"   - {f.name}")

# Fonction pour extraire l'année du nom du fichier
def extract_year_from_filename(filename):
    try:
        # Chercher les années (2000-2099)
        years = re.findall(r'\b(20\d{2})\b', filename.stem)
        if years:
            return int(years[0])
        return 0
    except:
        return 0

# Trier par année
excel_files = sorted(excel_files, key=extract_year_from_filename)

if not excel_files:
    st.error("Aucun fichier Excel trouvé dans le dossier.")
    st.write("Chemin recherché :", BASE_DIR)
    st.write("Fichiers dans le dossier:", list(BASE_DIR.iterdir()))
    st.stop()

st.sidebar.success(f"📊 {len(excel_files)} fichier(s) Excel détecté(s)")

# Charger TOUS les fichiers
try:
    data_raw = load_excel_files(excel_files)
    if data_raw is None:
        st.error("Aucune donnée chargée depuis les fichiers Excel.")
        st.stop()
    
except Exception as e:
    st.error(f"Erreur lors du chargement Excel : {e}")
    st.stop()

# Préparer les données
prepared, errors, columns_found = prepare_data(data_raw)

if errors:
    st.error(f"Problème dans les données : {errors}")
    if columns_found:
        st.write("Colonnes trouvées :", columns_found)
        st.write("Premières lignes des données :")
        st.write(data_raw.head())
    st.stop()

data, prices = prepared

# Vérifier si prices est vide
if prices.empty:
    st.error("Aucune donnée de prix après traitement.")
    st.stop()

# ==============================
# Infos sidebar - Afficher TOUTES les années
# ==============================

st.sidebar.write("📅 Période détectée")
st.sidebar.write(data["Date"].min(), "→", data["Date"].max())

# Récupérer les années disponibles et les convertir en int Python
annees_disponibles = sorted([int(a) for a in data["Année"].dropna().unique()])
st.sidebar.write("📊 Années disponibles dans les données")
st.sidebar.write(annees_disponibles)

# Afficher le nombre de lignes par année
st.sidebar.write("📈 Lignes par année:")
for annee in annees_disponibles:
    nb_lignes = len(data[data["Année"] == annee])
    st.sidebar.write(f"   {annee}: {nb_lignes} lignes")

# ==============================
# Sélection de la période (année)
# ==============================

st.sidebar.header("📅 Sélection de la période")

# Convertir les années en liste de strings pour l'affichage
options_annees = [str(a) for a in annees_disponibles]
selected_annee_str = st.sidebar.selectbox(
    "Choisir l'année à analyser",
    options=options_annees,
    index=len(options_annees)-1 if options_annees else 0  # Dernière année par défaut
)

# Convertir la sélection en entier
selected_annee = int(selected_annee_str)

st.sidebar.info(f"📊 Analyse limitée à l'année {selected_annee}")

# Filtrer les données par année
data_filtered = data[data["Année"] == selected_annee].copy()

if data_filtered.empty:
    st.error(f"Aucune donnée trouvée pour l'année {selected_annee}")
    st.stop()

# Recréer les prix pour l'année sélectionnée
prices_filtered = data_filtered.pivot_table(
    index="Date",
    columns="Societe",
    values="Close",
    aggfunc="last"
).sort_index().ffill()

# ==============================
# Banques disponibles
# ==============================

# Afficher les sociétés disponibles pour l'année sélectionnée
societes_disponibles = sorted(prices_filtered.columns.tolist())
st.sidebar.write(f"🏦 Sociétés disponibles en {selected_annee}:")
for s in societes_disponibles[:10]:  # Limiter l'affichage
    st.sidebar.write(f"   - {s}")
if len(societes_disponibles) > 10:
    st.sidebar.write(f"   ... et {len(societes_disponibles)-10} autres")

# Liste des banques à rechercher
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

# Nettoyer les noms de colonnes
prices_filtered.columns = prices_filtered.columns.str.strip()

# Trouver les banques présentes dans les données pour l'année sélectionnée
banques_valides = []
for b in banques:
    # Recherche exacte et insensible à la casse
    matching_cols = [col for col in prices_filtered.columns if col.upper() == b.upper()]
    if matching_cols:
        banques_valides.append(matching_cols[0])

# Si aucune banque trouvée, utiliser toutes les sociétés
if len(banques_valides) < 2:
    st.warning(f"⚠️ Banques spécifiques non trouvées pour {selected_annee}. {len(prices_filtered.columns)} société(s) disponible(s).")
    banques_valides = list(prices_filtered.columns)
    if len(banques_valides) < 2:
        st.error(f"Pas assez de sociétés pour l'analyse en {selected_annee}.")
        st.stop()
    else:
        st.info(f"📊 Utilisation de toutes les sociétés disponibles ({len(banques_valides)})")

# ==============================
# Paramètres d'analyse
# ==============================

st.sidebar.header("⚙️ Paramètres d'analyse")

# Sélection des banques à analyser
selected_banques = st.sidebar.multiselect(
    f"Choisir les banques/sociétés à analyser ({selected_annee})",
    options=banques_valides,
    default=banques_valides[:min(6, len(banques_valides))]
)

rf = st.sidebar.number_input(
    "Taux sans risque annuel (%)",
    value=7.5,
    step=0.1
) / 100

capital = st.sidebar.number_input(
    "Capital à investir (TND)",
    value=10000,
    step=1000
)

if len(selected_banques) < 2:
    st.warning("⚠️ Veuillez sélectionner au moins deux banques/sociétés pour l'analyse.")
    st.stop()

# ==============================
# Calculs financiers pour l'année sélectionnée
# ==============================

try:
    selected_prices = prices_filtered[selected_banques].dropna(how="all").ffill()
    
    # Vérifier qu'on a assez de données
    if len(selected_prices) < 2:
        st.error(f"Pas assez de données de prix pour l'année {selected_annee}.")
        st.stop()
    
    returns = selected_prices.pct_change().dropna()
    
    if returns.empty or len(returns) < 2:
        st.error(f"Pas assez de données pour calculer les rendements en {selected_annee}.")
        st.stop()
    
    # Pour une analyse annuelle, on utilise les données quotidiennes
    # Annualisation: 252 jours de bourse
    mean_returns = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    cov_matrix = returns.cov() * 252
    corr_matrix = returns.corr()
    
    cumulative_returns = (1 + returns).cumprod() - 1
    rolling_vol = returns.rolling(20).std() * np.sqrt(252)  # 20 jours pour l'année
    
    drawdown = selected_prices / selected_prices.cummax() - 1
    max_drawdown = drawdown.min()
    
    var_95 = returns.quantile(0.05) * np.sqrt(252)
    expected_shortfall = returns[returns.le(returns.quantile(0.05))].mean() * np.sqrt(252)
    
    market_return = returns.mean(axis=1)
    
    beta = {}
    for col in returns.columns:
        cov = np.cov(returns[col], market_return)[0][1] if len(market_return) > 1 else 0
        var = np.var(market_return) if len(market_return) > 1 else 1
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
    
except Exception as e:
    st.error(f"Erreur lors des calculs financiers : {e}")
    st.stop()

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

try:
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
    
    weights_sharpe = result_sharpe.x
    weights_minvar = result_minvar.x
    
    ret_sharpe = port_return(weights_sharpe)
    vol_sharpe = port_vol(weights_sharpe)
    sharpe_ratio = (ret_sharpe - rf) / vol_sharpe if vol_sharpe > 0 else 0
    
    ret_minvar = port_return(weights_minvar)
    vol_minvar = port_vol(weights_minvar)
    sharpe_minvar = (ret_minvar - rf) / vol_minvar if vol_minvar > 0 else 0
    
    weights_df = pd.DataFrame({
        "Banque": selected_banques,
        "Poids Sharpe max": weights_sharpe,
        "Montant Sharpe max": weights_sharpe * capital,
        "Poids variance minimale": weights_minvar,
        "Montant variance minimale": weights_minvar * capital
    })
    
except Exception as e:
    st.error(f"Erreur lors de l'optimisation : {e}")
    st.stop()

# ==============================
# Recommandations
# ==============================

ranking = metrics.copy()

if not ranking.empty:
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
        median_vol = metrics["Volatilité annualisée"].median()
        if row["Sharpe individuel"] > 1 and row["Volatilité annualisée"] < median_vol:
            return "Très attractive"
        elif row["Sharpe individuel"] > 0.5:
            return "Intéressante"
        elif row["Volatilité annualisée"] > median_vol:
            return "Risque élevé"
        else:
            return "À surveiller"
    
    ranking["Recommandation"] = ranking.apply(recommendation, axis=1)
else:
    best_bank = "N/A"

# ==============================
# Interface principale
# ==============================

st.header(f"📊 Analyse pour l'année {selected_annee}")

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

with tab1:
    st.subheader(f"Résumé global - Année {selected_annee}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nombre de sociétés", len(selected_banques))
    col2.metric("Meilleure société", str(best_bank))
    col3.metric("Sharpe max", f"{float(sharpe_ratio):.4f}")
    col4.metric("Capital simulé", f"{float(capital):,.0f} TND")
    
    # Graphique des cours
    if not selected_prices.empty:
        fig_prices = px.line(selected_prices, title=f"Évolution des cours de clôture - {selected_annee}")
        fig_prices.update_layout(xaxis_title="Date", yaxis_title="Cours (TND)")
        st.plotly_chart(fig_prices, use_container_width=True)
    
    # Graphique des rendements cumulés
    if not cumulative_returns.empty:
        fig_cum = px.line(cumulative_returns, title=f"Rentabilité cumulée - {selected_annee}")
        fig_cum.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_cum, use_container_width=True)

with tab2:
    st.subheader(f"Tableau des indicateurs - {selected_annee}")
    
    if not metrics.empty:
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
    
    # Graphiques
    if not metrics["Rentabilité annualisée"].dropna().empty:
        fig_ret = px.bar(metrics, y="Rentabilité annualisée", title=f"Rentabilité annualisée - {selected_annee}")
        fig_ret.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_ret, use_container_width=True)
    
    if not metrics["Volatilité annualisée"].dropna().empty:
        fig_vol = px.bar(metrics, y="Volatilité annualisée", title=f"Volatilité annualisée - {selected_annee}")
        fig_vol.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_vol, use_container_width=True)

with tab3:
    st.subheader(f"Analyse des risques - {selected_annee}")
    
    if not drawdown.empty and not drawdown.isna().all().all():
        fig_drawdown = px.line(drawdown, title=f"Drawdown des sociétés - {selected_annee}")
        fig_drawdown.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_drawdown, use_container_width=True)
    
    if not corr_matrix.empty:
        fig_corr = px.imshow(corr_matrix, text_auto=True, title=f"Matrice de corrélation - {selected_annee}", aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)

with tab4:
    st.subheader(f"Portefeuille Sharpe maximum - {selected_annee}")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Rentabilité", f"{float(ret_sharpe):.2%}")
    c2.metric("Risque", f"{float(vol_sharpe):.2%}")
    c3.metric("Sharpe", f"{float(sharpe_ratio):.4f}")
    
    st.subheader(f"Portefeuille variance minimale - {selected_annee}")
    
    c4, c5, c6 = st.columns(3)
    c4.metric("Rentabilité", f"{float(ret_minvar):.2%}")
    c5.metric("Risque", f"{float(vol_minvar):.2%}")
    c6.metric("Sharpe", f"{float(sharpe_minvar):.4f}")
    
    st.subheader("Poids et montants à investir")
    
    if not weights_df.empty:
        st.dataframe(
            weights_df.style.format({
                "Poids Sharpe max": "{:.2%}",
                "Montant Sharpe max": "{:,.2f}",
                "Poids variance minimale": "{:.2%}",
                "Montant variance minimale": "{:,.2f}"
            }),
            use_container_width=True
        )

with tab5:
    st.subheader(f"Frontière efficiente - {selected_annee}")
    
    frontier_returns = []
    frontier_vols = []
    
    try:
        target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 30)
        
        for target in target_returns:
            cons = (
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, t=target: port_return(w) - t}
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
                frontier_returns.append(float(port_return(w)))
                frontier_vols.append(float(port_vol(w)))
        
        fig_frontier = go.Figure()
        
        if frontier_vols:
            fig_frontier.add_trace(go.Scatter(
                x=frontier_vols,
                y=frontier_returns,
                mode="lines",
                name="Frontière efficiente",
                line=dict(color="blue", width=2)
            ))
        
        fig_frontier.add_trace(go.Scatter(
            x=[float(vol_sharpe)],
            y=[float(ret_sharpe)],
            mode="markers",
            name="Sharpe max",
            marker=dict(size=14, color="red")
        ))
        
        fig_frontier.add_trace(go.Scatter(
            x=[float(vol_minvar)],
            y=[float(ret_minvar)],
            mode="markers",
            name="Variance minimale",
            marker=dict(size=14, color="green")
        ))
        
        fig_frontier.update_layout(
            title=f"Frontière efficiente de Markowitz - {selected_annee}",
            xaxis_title="Risque / Volatilité",
            yaxis_title="Rentabilité"
        )
        
        fig_frontier.update_xaxes(tickformat=".0%")
        fig_frontier.update_yaxes(tickformat=".0%")
        
        st.plotly_chart(fig_frontier, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Erreur lors du calcul de la frontière efficiente : {e}")

with tab6:
    st.subheader(f"🤖 Recommandations intelligentes - {selected_annee}")
    st.info("Cette analyse est indicative et ne constitue pas un conseil financier personnalisé.")
    
    if not ranking.empty:
        st.success(f"🏆 Meilleure société selon le modèle : {best_bank}")
        st.dataframe(ranking, use_container_width=True)

with tab7:
    st.subheader(f"💼 Simulation d'investissement - {selected_annee}")
    st.write(f"Capital simulé : **{capital:,.2f} TND**")
    
    if not weights_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portefeuille Sharpe max")
            st.dataframe(weights_df[["Banque", "Montant Sharpe max"]], use_container_width=True)
        
        with col2:
            st.subheader("Portefeuille Variance min")
            st.dataframe(weights_df[["Banque", "Montant variance minimale"]], use_container_width=True)

with tab8:
    st.subheader(f"Télécharger le rapport Excel - {selected_annee}")
    
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            selected_prices.to_excel(writer, sheet_name=f"Prix_{selected_annee}")
            returns.to_excel(writer, sheet_name=f"Rendements_{selected_annee}")
            metrics.to_excel(writer, sheet_name=f"Indicateurs_{selected_annee}")
            weights_df.to_excel(writer, sheet_name=f"Poids_{selected_annee}", index=False)
        
        st.download_button(
            label=f"📥 Télécharger le rapport {selected_annee}",
            data=output.getvalue(),
            file_name=f"rapport_markowitz_{selected_annee}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Erreur lors de la création du rapport : {e}")
