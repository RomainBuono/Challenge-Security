import sys
from pathlib import Path

# 1. Résolution dynamique des chemins (Architecture robuste)
root_path = Path(__file__).resolve().parent.parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import io

import pandas as pd
import streamlit as st

# Importations locales
from src.data.mariadb_client import MariaDBClient
from src.detection_anomaly.detection_anomaly import CAHAnalyzer, SecurityOrchestrator

# Configuration de la page
st.set_page_config(page_title="Exploration DB & IA Cyber", page_icon="🛡️", layout="wide")
st.title("🛡️ Table des données")


# --- 1. Récupération des données avec Cache ---
@st.cache_data(ttl=300, show_spinner="Connexion à MariaDB...")
def load_firewall_logs(limit: int = 1000):
    """Charge les logs depuis MariaDB avec un cache de 5 minutes.

    Args:
        limit: nombre maximum de lignes à récupérer.
    """
    db = MariaDBClient()
    return db.fetch_logs(table_name="FW", limit=limit)


@st.cache_data(ttl=300)
def get_total_rows() -> int:
    try:
        return MariaDBClient().count_all_logs(table_name="FW")
    except Exception:
        return 0


try:
    # UI controls: choose how many rows to load/display
    col1, col2 = st.columns([2, 1])
    with col1:
        max_rows = get_total_rows()
        row_options = [100, 500, 1000, 2000, 5000]
        if max_rows and max_rows > 0 and max_rows not in row_options:
            row_options.append(max_rows)
        row_options = sorted(set(row_options))
        default = 100 if 100 in row_options else row_options[0]
        # present options with friendly labels (max value shown as-is)
        option_labels = [str(x) for x in row_options]
        selected_idx = st.selectbox(
            "Nombre de lignes à charger",
            options=list(range(len(option_labels))),
            format_func=lambda i: option_labels[i],
            index=option_labels.index(str(default)),
        )
        selected_limit = row_options[selected_idx]
    with col2:
        total_rows = get_total_rows()
        st.metric("Lignes en base", f"{total_rows:,}")

    df_logs = load_firewall_logs(limit=selected_limit)

    # Affichage du tableau de données
    st.write(f"### 📋 Aperçu des logs réseau — affichage {len(df_logs):,} lignes")
    st.dataframe(df_logs, width="stretch")

    # Téléchargement du DataFrame affiché
    if not df_logs.empty:
        csv_buffer = io.StringIO()
        df_logs.to_csv(csv_buffer, index=False)
        st.download_button(
            "📥 Télécharger le tableau affiché (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"firewall_logs_{len(df_logs)}.csv",
            mime="text/csv",
        )

    # Expander: description détaillée des colonnes (bloc par colonne)
    with st.expander("ℹ️ Description des colonnes"):
        if df_logs.empty:
            st.info("Aucun enregistrement à décrire.")
        else:
            # heuristiques simples pour expliquer quelques colonnes fréquentes
            explanations = {
                "id": "Identifiant unique du log (généralement auto-incrémenté).",
                "ipsrc": "Adresse IP source à l'origine du flux (format IPv4/IPv6).",
                "ipdst": "Adresse IP destination ciblée par le flux.",
                "ipsource": "Adresse IP source (synonyme).",
                "src_ip": "Adresse IP source (synonyme).",
                "ip": "Adresse IP (générique).",
                "proto": "Protocole réseau (ex: TCP, UDP, ICMP).",
                "action": "Action du firewall (ex: permit / deny).",
                "datetime": "Horodatage du flux (timestamp).",
                "time": "Horodatage du flux (timestamp).",
                "srcport": "Port source utilisé par l'initiateur du flux.",
                "dstport": "Port destination visé par le flux.",
                "bytes": "Nombre d'octets transférés (si disponible).",
                "packets": "Nombre de paquets transférés (si disponible).",
            }

            # Build transposed table: rows = metrics, columns = DB columns
            cols = list(df_logs.columns)
            types_row = {}
            uniques_row = {}
            examples_row = {}
            explanation_row = {}

            for c in cols:
                s = df_logs[c]
                types_row[c] = str(s.dtype)
                try:
                    uniques_row[c] = int(s.nunique(dropna=True))
                except Exception:
                    uniques_row[c] = "N/A"
                try:
                    exs = s.dropna().unique()[:2].tolist()
                    examples_row[c] = ", ".join(map(str, exs)) if exs else "(aucun)"
                except Exception:
                    examples_row[c] = "(aucun)"

                expl = explanations.get(c.lower())
                if not expl:
                    low = c.lower()
                    if "ip" in low and ("src" in low or "source" in low):
                        expl = "Adresse IP source."
                    elif "ip" in low and ("dst" in low or "dest" in low):
                        expl = "Adresse IP destination."
                    elif "port" in low:
                        expl = "Port réseau (source ou destination selon le nom)."
                    elif "time" in low or "date" in low:
                        expl = "Horodatage/temps associé au log."
                    elif "action" in low:
                        expl = "Résultat action (permit/deny) du firewall."
                    else:
                        expl = "Pas d'explication disponible — colonne spécifique au jeu de données."
                explanation_row[c] = expl

            transposed = pd.DataFrame(
                [types_row, uniques_row, examples_row, explanation_row],
                index=["Type", "Nb. Val. uniques", "2 exemples", "Explication"],
            )

            st.markdown(
                "**Tableau explicatif (lignes = métriques, colonnes = champs DB)**"
            )
            st.dataframe(transposed, width="stretch")

except Exception as e:
    st.error(f"❌ Erreur de connexion à la base de données : {e}")
