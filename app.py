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
            st.sidebar.write(f"📂 Chargement: {file.name}")
            
            # Lecture de la première feuille
            df = pd.read_excel(file)
            
            # NETTOYAGE IMPORTANT : Supprimer les espaces dans les noms de colonnes
            df.columns = df.columns.str.strip()
            
            df["Source"] = file.stem
            df["Fichier_Annee"] = file.stem
            all_data.append(df)
            
            st.sidebar.write(f"   ✅ {len(df)} lignes chargées")
            st.sidebar.write(f"   📋 Colonnes: {list(df.columns)[:5]}...")
            
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

    # Nettoyer les noms de colonnes (supprimer les espaces)
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

    # Afficher les statistiques par année
    st.sidebar.write("📊 **Statistiques par année:**")
    for annee in sorted(data["Année"].unique()):
        nb_lignes = len(data[data["Année"] == annee])
        nb_societes = data[data["Année"] == annee]["Societe"].nunique()
        st.sidebar.write(f"   {annee}: {nb_lignes} lignes, {nb_societes} sociétés")

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

# Lister TOUS les fichiers Excel
excel_files = []
for file in BASE_DIR.iterdir():
    if file.is_file() and file.suffix.lower() in ['.xlsx', '.xls']:
        if not file.name.startswith('~') and not file.name.startswith('.'):
            excel_files.append(file)

# Fonction pour extraire l'année
def extract_year_from_filename(filename):
    try:
        years = re.findall(r'\b(20\d{2})\b', filename.stem)
        if years:
            return int(years[0])
        return 0
    except:
        return 0

excel_files = sorted(excel_files, key=extract_year_from_filename)

st.sidebar.write(f"📊 **{len(excel_files)} fichiers Excel trouvés**")
for f in excel_files:
    annee = extract_year_from_filename(f)
    st.sidebar.write(f"   - {f.name} (année: {annee})")

if not excel_files:
    st.error("Aucun fichier Excel trouvé!")
    st.stop()

# Charger TOUS les fichiers
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
    if columns_found:
        st.write("Colonnes trouvées :", columns_found)
    st.stop()

data, prices = prepared

if prices.empty:
    st.error("Aucune donnée de prix après traitement.")
    st.stop()

# ==============================
# Récupérer les années disponibles
# ==============================

# Obtenir toutes les années présentes dans les données
annees_disponibles = sorted([int(a) for a in data["Année"].dropna().unique()])

st.sidebar.write("📅 **Période détectée**")
st.sidebar.write(f"   Du {data['Date'].min()} au {data['Date'].max()}")
st.sidebar.write(f"📊 **Années disponibles: {annees_disponibles}**")

# Afficher le nombre de lignes par année
st.sidebar.write("📈 **Détail par année:**")
for annee in annees_disponibles:
    nb_lignes = len(data[data["Année"] == annee])
    nb_societes = data[data["Année"] == annee]["Societe"].nunique()
    st.sidebar.write(f"   {annee}: {nb_lignes} lignes, {nb_societes} sociétés")

# ==============================
# Sélection de l'année
# ==============================

st.sidebar.header("📅 Sélection de la période")

if len(annees_disponibles) == 0:
    st.error("Aucune année détectée!")
    st.stop()

options_annees = [str(a) for a in annees_disponibles]
selected_annee_str = st.sidebar.selectbox(
    "Choisir l'année à analyser",
    options=options_annees,
    index=len(options_annees)-1 if options_annees else 0
)

selected_annee = int(selected_annee_str)
st.sidebar.info(f"📊 Analyse limitée à l'année {selected_annee}")

# Filtrer les données par année
data_filtered = data[data["Année"] == selected_annee].copy()

if data_filtered.empty:
    st.error(f"Aucune donnée pour l'année {selected_annee}")
    st.stop()

st.sidebar.success(f"✅ {len(data_filtered)} lignes pour {selected_annee}")

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

societes_disponibles = sorted(prices_filtered.columns.tolist())
st.sidebar.write(f"🏦 **{len(societes_disponibles)} sociétés disponibles en {selected_annee}**")

# Liste des banques
banques = [
    "BIAT", "ATB", "STB", "BT", "AMEN BANK", "UIB", "UBCI", "BH",
    "BNA", "ATTIJARI BANK", "BH BANK", "BTE (ADP)", "WIFACK INT BANK"
]

# Nettoyer les noms
prices_filtered.columns = prices_filtered.columns.str.strip()

# Trouver les banques présentes
banques_valides = []
for b in banques:
    matching_cols = [col for col in prices_filtered.columns if col.upper() == b.upper()]
    if matching_cols:
        banques_valides.append(matching_cols[0])

if len(banques_valides) < 2:
    st.warning(f"⚠️ {len(banques_valides)} banque(s) trouvée(s) sur {len(banques)}")
    st.info(f"Utilisation de toutes les sociétés disponibles ({len(societes_disponibles)})")
    banques_valides = societes_disponibles
    if len(banques_valides) < 2:
        st.error("Pas assez de sociétés pour l'analyse.")
        st.stop()

# ==============================
# Paramètres
# ==============================

st.sidebar.header("⚙️ Paramètres")

selected_banques = st.sidebar.multiselect(
    f"Sélectionner les sociétés à analyser ({selected_annee})",
    options=banques_valides,
    default=banques_valides[:min(8, len(banques_valides))]
)

rf = st.sidebar.number_input("Taux sans risque (%)", value=7.5, step=0.1) / 100
capital = st.sidebar.number_input("Capital (TND)", value=10000, step=1000)

if len(selected_banques) < 2:
    st.warning("Sélectionnez au moins 2 sociétés")
    st.stop()

# ==============================
# Calculs financiers
# ==============================

try:
    selected_prices = prices_filtered[selected_banques].dropna(how="all").ffill()
    
    if len(selected_prices) < 2:
        st.error("Pas assez de données")
        st.stop()
    
    returns = selected_prices.pct_change().dropna()
    
    if returns.empty or len(returns) < 2:
        st.error("Pas assez de rendements")
        st.stop()
    
    # Calculs annualisés
    mean_returns = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    cov_matrix = returns.cov() * 252
    corr_matrix = returns.corr()
    
    cumulative_returns = (1 + returns).cumprod() - 1
    drawdown = selected_prices / selected_prices.cummax() - 1
    max_drawdown = drawdown.min()
    
    metrics = pd.DataFrame({
        "Rentabilité annualisée": mean_returns,
        "Volatilité annualisée": volatility,
        "Sharpe individuel": (mean_returns - rf) / volatility,
        "Max Drawdown": max_drawdown
    })
    
    metrics = metrics.replace([np.inf, -np.inf], np.nan).dropna()
    
except Exception as e:
    st.error(f"Erreur calculs: {e}")
    st.stop()

# ==============================
# Optimisation
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

result_sharpe = minimize(neg_sharpe, init, method="SLSQP", bounds=bounds, constraints=constraints)

if result_sharpe.success:
    weights_sharpe = result_sharpe.x
    ret_sharpe = port_return(weights_sharpe)
    vol_sharpe = port_vol(weights_sharpe)
    sharpe_ratio = (ret_sharpe - rf) / vol_sharpe if vol_sharpe > 0 else 0
    
    weights_df = pd.DataFrame({
        "Société": selected_banques,
        "Poids optimal": weights_sharpe,
        "Montant à investir": weights_sharpe * capital
    })
    
    # Filtrer les poids positifs
    weights_df = weights_df[weights_df["Poids optimal"] > 0.001].sort_values("Poids optimal", ascending=False)
else:
    st.error("Optimisation échouée")
    st.stop()

# ==============================
# Affichage principal
# ==============================

st.header(f"📊 Analyse Markowitz - Année {selected_annee}")

# Métriques
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sociétés analysées", len(selected_banques))
col2.metric("Rentabilité portefeuille", f"{ret_sharpe:.2%}")
col3.metric("Risque portefeuille", f"{vol_sharpe:.2%}")
col4.metric("Ratio de Sharpe", f"{sharpe_ratio:.3f}")

# Graphique des cours
st.subheader(f"Évolution des cours - {selected_annee}")
fig_prices = px.line(selected_prices, title="Cours de clôture")
fig_prices.update_layout(xaxis_title="Date", yaxis_title="Prix (TND)")
st.plotly_chart(fig_prices, use_container_width=True)

# Graphique des rendements cumulés
st.subheader("Rendements cumulés")
fig_returns = px.line(cumulative_returns, title="Performance cumulée")
fig_returns.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_returns, use_container_width=True)

# Métriques individuelles
st.subheader("Métriques individuelles par société")
st.dataframe(
    metrics.style.format({
        "Rentabilité annualisée": "{:.2%}",
        "Volatilité annualisée": "{:.2%}",
        "Sharpe individuel": "{:.3f}",
        "Max Drawdown": "{:.2%}"
    }),
    use_container_width=True
)

# Portefeuille optimal
st.subheader("Portefeuille optimal (Sharpe maximum)")
st.dataframe(
    weights_df.style.format({
        "Poids optimal": "{:.2%}",
        "Montant à investir": "{:,.2f} TND"
    }),
    use_container_width=True
)

# Graphique des poids
if len(weights_df) > 0:
    fig_weights = px.bar(
        weights_df,
        x="Société",
        y="Poids optimal",
        title="Allocation du portefeuille optimal",
        color="Poids optimal"
    )
    fig_weights.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_weights, use_container_width=True)

# Matrice de corrélation
st.subheader("Matrice de corrélation")
fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Corrélations entre les sociétés")
st.plotly_chart(fig_corr, use_container_width=True)

# Téléchargement
try:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        selected_prices.to_excel(writer, sheet_name="Prix")
        returns.to_excel(writer, sheet_name="Rendements")
        metrics.to_excel(writer, sheet_name="Indicateurs")
        weights_df.to_excel(writer, sheet_name="Portefeuille_optimal", index=False)
        corr_matrix.to_excel(writer, sheet_name="Correlations")
    
    st.download_button(
        label="📥 Télécharger le rapport Excel",
        data=output.getvalue(),
        file_name=f"markowitz_{selected_annee}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
except Exception as e:
    st.error(f"Erreur création rapport: {e}")
