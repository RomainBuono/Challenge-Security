import streamlit as st

from src.app.utils import get_db_client
from src.data.mariadb_client import MariaDBClient

st.set_page_config(page_title="Exploration DB", layout="wide")
st.header("🗄️ Parcours des données")

db = get_db_client()

try:
    # 1. Récupération des métriques de volume
    total_db = db.count_all_logs(table_name="FW")

    # 2. Récupération de l'échantillon pour affichage
    limit = 50
    df = db.fetch_logs(table_name="FW", limit=limit)

    # 3. Affichage des compteurs
    col1, col2 = st.columns(2)
    col1.metric("Total en Base Cloud", f"{total_db:,}")
    col2.metric("Lignes affichées ci-dessous", f"{len(df):,}")

    st.divider()

    # 4. Affichage du tableau (renderDataTable)
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("La table est actuellement vide.")

except Exception as e:
    st.error(f"Erreur lors de l'exploration : {e}")
