# Dans un nouveau fichier src/app/utils.py (ou en haut de main.py)
import streamlit as st
from src.data.mariadb_client import MariaDBClient

@st.cache_resource
def get_db_client():
    return MariaDBClient()