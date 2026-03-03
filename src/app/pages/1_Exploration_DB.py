import sys
from pathlib import Path

# 1. Résolution dynamique des chemins (Architecture robuste)
root_path = Path(__file__).resolve().parent.parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st
import pandas as pd

# Importations locales
from src.data.mariadb_client import MariaDBClient
from src.detection_anomaly.detection_anomaly import CAHAnalyzer, SecurityOrchestrator

# Configuration de la page
st.set_page_config(page_title="Exploration DB & IA Cyber", page_icon="🛡️", layout="wide")
st.title("🛡️ Tableau de Bord SOC : Exploration & XAI")

# --- 1. Récupération des données avec Cache ---
@st.cache_data(ttl=600, show_spinner="Connexion à MariaDB...")
def load_firewall_logs():
    """Charge les logs depuis MariaDB avec un cache de 10 minutes."""
    db = MariaDBClient()
    return db.fetch_logs(table_name="FW", limit=1000)

try:
    df_logs = load_firewall_logs()
    
    # Affichage du tableau de données
    st.write(f"### 📋 Aperçu des logs réseau (Total : {len(df_logs)})")
    st.dataframe(df_logs, use_container_width=True)

    # --- 2. Section Analyse LLM ---
    st.divider()
    st.subheader("🤖 Analyse Experte d'Anomalies (Mistral)")
    st.info("Cette analyse combine une Classification Ascendante Hiérarchique (CAH) mathématique avec l'expertise d'un LLM pour qualifier les menaces.")

    # Bouton de déclenchement
    if st.button("🚀 Lancer l'analyse cyber experte", type="primary"):
        if df_logs.empty:
            st.warning("Le dataset est vide. Impossible de lancer l'analyse.")
        else:
            # Spinner pour patienter pendant l'inférence
            with st.spinner("Analyse topologique et requêtage de Mistral en cours... Veuillez patienter."):
                try:
                    # Instanciation de tes classes métiers
                    cah = CAHAnalyzer(df_logs)
                    orchestrator = SecurityOrchestrator(model_name="mistral-medium-latest")
                    
                    # Exécution du pipeline
                    report_path = orchestrator.run_analysis(cah)
                    
                    # Lecture du rapport Markdown généré
                    with open(report_path, "r", encoding="utf-8") as f:
                        markdown_report = f.read()
                    
                    st.success("Analyse terminée avec succès !")
                    
                    # 3. Affichage du rapport
                    with st.expander("📄 Voir le Rapport XAI Détaillé", expanded=True):
                        # L'attribut unsafe_allow_html=True est OBLIGATOIRE ici 
                        # car ton script génère une balise HTML <img> avec du base64
                        st.markdown(markdown_report, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"❌ Une erreur s'est produite lors de l'exécution du modèle : {e}")

except Exception as e:
    st.error(f"❌ Erreur de connexion à la base de données : {e}")