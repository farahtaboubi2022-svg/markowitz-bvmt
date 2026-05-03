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
            df["Source"] = file.stem
            df["Fichier_Annee"] = file.stem  # Garder l'année du fichier
            all_data.append(df)
            
            st.sidebar.write(f"   ✅ {len(df)} lignes, {df.columns.tolist()[:5]}...")
            
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
# Chargement fichiers Excel - DIAGNOSTIC
# ==============================

BASE_DIR = Path(__file__).parent

st.sidebar.write("🔍 **DIAGNOSTIC DE CHARGEMENT**")
st.sidebar.write(f"Chemin: {BASE_DIR}")

# Lister TOUS les fichiers dans le dossier
all_files = list(BASE_DIR.iterdir())
st.sidebar.write(f"📁 Total fichiers dans dossier: {len(all_files)}")

# Afficher tous les fichiers
st.sidebar.write("📄 Tous les fichiers:")
for f in all_files:
    st.sidebar.write(f"   - {f.name} (est fichier: {f.is_file()})")

# Chercher spécifiquement les fichiers Excel
excel_files = []
for file in all_files:
    if file.is_file():
        if file.suffix.lower() in ['.xlsx', '.xls']:
            excel_files.append(file)
            st.sidebar.write(f"   ✅ EXCEL trouvé: {file.name}")

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

st.sidebar.write(f"\n📊 **Fichiers Excel trouvés: {len(excel_files)}**")
for f in excel_files:
    annee = extract_year_from_filename(f)
    st.sidebar.write(f"   - {f.name} (année détectée: {annee})")

if not excel_files:
    st.error("Aucun fichier Excel trouvé dans le dossier!")
    st.write("Contenu du dossier:", list(BASE_DIR.iterdir()))
    st.stop()

# Charger TOUS les fichiers
try:
    data_raw = load_excel_files(excel_files)
    if data_raw is None:
        st.error("Aucune donnée chargée depuis les fichiers Excel.")
        st.stop()
    
except Exception as e:
    st.error(f"Erreur lors du chargement Excel : {e}")
    st.stop()

st.sidebar.write("\n📊 **Après chargement:**")
st.sidebar.write(f"Colonnes dans data_raw: {data_raw.columns.tolist()}")
st.sidebar.write(f"Premières lignes de data_raw:")
st.sidebar.dataframe(data_raw.head(3))

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
# DIAGNOSTIC DES ANNÉES
# ==============================

st.sidebar.write("\n📅 **DIAGNOSTIC DES DATES**")

# Afficher les dates uniques
dates_uniques = sorted(data["Date"].dropna().unique())
st.sidebar.write(f"Dates dans les données: {len(dates_uniques)} dates")
if dates_uniques:
    st.sidebar.write(f"Première date: {dates_uniques[0]}")
    st.sidebar.write(f"Dernière date: {dates_uniques[-1]}")

# Récupérer les années disponibles
data["Année"] = data["Date"].dt.year
annees_disponibles = sorted([int(a) for a in data["Année"].dropna().unique()])

st.sidebar.write(f"\n📊 **Années trouvées dans les données: {annees_disponibles}**")

# Afficher le nombre de lignes par année
st.sidebar.write("📈 Lignes par année:")
for annee in annees_disponibles:
    nb_lignes = len(data[data["Année"] == annee])
    st.sidebar.write(f"   {annee}: {nb_lignes} lignes")
    # Afficher les sociétés pour cette année
    societes_annee = data[data["Année"] == annee]["Societe"].unique()
    st.sidebar.write(f"      Sociétés: {len(societes_annee)} sociétés")
    if len(societes_annee) <= 5:
        st.sidebar.write(f"      {list(societes_annee)}")

# Si une seule année est trouvée, afficher un avertissement
if len(annees_disponibles) == 1:
    st.warning(f"⚠️ Une seule année trouvée dans les données: {annees_disponibles[0]}")
    st.info("Vérifiez que tous vos fichiers Excel contiennent des dates valides dans la colonne 'SEANCE'")
    
    # Afficher un aperçu des données brutes
    st.subheader("Aperçu des données chargées:")
    st.dataframe(data_raw.head(20))
    
    # Vérifier les sources
    if "Source" in data_raw.columns:
        st.subheader("Fichiers sources chargés:")
        st.write(data_raw["Source"].value_counts())

# ==============================
# Sélection de la période (année)
# ==============================

st.sidebar.header("📅 Sélection de la période")

if len(annees_disponibles) == 0:
    st.error("Aucune année détectée dans les données!")
    st.stop()

# Convertir les années en liste de strings pour l'affichage
options_annees = [str(a) for a in annees_disponibles]
selected_annee_str = st.sidebar.selectbox(
    "Choisir l'année à analyser",
    options=options_annees,
    index=len(options_annees)-1 if options_annees else 0
)

# Convertir la sélection en entier
selected_annee = int(selected_annee_str)

st.sidebar.info(f"📊 Analyse limitée à l'année {selected_annee}")

# Filtrer les données par année
data_filtered = data[data["Année"] == selected_annee].copy()

if data_filtered.empty:
    st.error(f"Aucune donnée trouvée pour l'année {selected_annee}")
    st.stop()

st.sidebar.success(f"✅ {len(data_filtered)} lignes pour l'année {selected_annee}")

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
st.sidebar.write(f"\n🏦 **Sociétés disponibles en {selected_annee}:**")
for s in societes_disponibles[:10]:
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
    matching_cols = [col for col in prices_filtered.columns if col.upper() == b.upper()]
    if matching_cols:
        banques_valides.append(matching_cols[0])
        st.sidebar.write(f"   ✅ Trouvé: {matching_cols[0]}")

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
# Suite de l'analyse (calculs, optimisation, etc.)
# ==============================

try:
    selected_prices = prices_filtered[selected_banques].dropna(how="all").ffill()
    
    if len(selected_prices) < 2:
        st.error(f"Pas assez de données de prix pour l'année {selected_annee}.")
        st.stop()
    
    returns = selected_prices.pct_change().dropna()
    
    if returns.empty or len(returns) < 2:
        st.error(f"Pas assez de données pour calculer les rendements en {selected_annee}.")
        st.stop()
    
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
        "Sharpe individuel": (mean_returns - rf) / volatility
    })
    
    metrics = metrics.replace([np.inf, -np.inf], np.nan).dropna()
    
except Exception as e:
    st.error(f"Erreur lors des calculs financiers : {e}")
    st.stop()

# Optimisation
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
weights_sharpe = result_sharpe.x
ret_sharpe = port_return(weights_sharpe)
vol_sharpe = port_vol(weights_sharpe)
sharpe_ratio = (ret_sharpe - rf) / vol_sharpe if vol_sharpe > 0 else 0

weights_df = pd.DataFrame({
    "Banque": selected_banques,
    "Poids Sharpe max": weights_sharpe,
    "Montant Sharpe max": weights_sharpe * capital
})

# ==============================
# Interface principale
# ==============================

st.header(f"📊 Analyse pour l'année {selected_annee}")

# Afficher les statistiques
col1, col2, col3, col4 = st.columns(4)
col1.metric("Nombre de sociétés", len(selected_banques))
col2.metric("Sharpe max", f"{sharpe_ratio:.4f}")
col3.metric("Capital simulé", f"{capital:,.0f} TND")
col4.metric("Année analysée", selected_annee)

# Graphique des cours
fig_prices = px.line(selected_prices, title=f"Évolution des cours - {selected_annee}")
st.plotly_chart(fig_prices, use_container_width=True)

# Tableau des métriques
st.subheader("Indicateurs")
st.dataframe(metrics.style.format({
    "Rentabilité annualisée": "{:.2%}",
    "Volatilité annualisée": "{:.2%}",
    "Sharpe individuel": "{:.4f}"
}))

# Poids du portefeuille
st.subheader("Portefeuille optimal - Sharpe maximum")
st.dataframe(weights_df.style.format({
    "Poids Sharpe max": "{:.2%}",
    "Montant Sharpe max": "{:,.2f}"
}))

# Téléchargement du rapport
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
    st.error(f"Erreur création rapport: {e}")
