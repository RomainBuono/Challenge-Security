import streamlit as st

def main():
    st.set_page_config(
        page_title="Projet SISE-OPSIE 2026",
        page_icon="🛡️",
        layout="wide"
    )

    # Header principal conforme au sujet
    st.title("🛡️ Dashboard de l'état du SI")
    st.info("Projet SISE-OPSIE 2026 - Analyse de traces et traitement d'événements")

    # Structure en colonnes pour les KPIs futurs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Flux Totaux", "0", help="Nombre de lignes de log analysées")
    with col2:
        st.metric("Alertes (Deny)", "0", delta_color="inverse")
    with col3:
        st.metric("Règles Actives", "0")

    st.divider()

    # Section centrale d'attente
    st.warning("⚠️ **En attente de la création de visualisations et analyses**")
    
    with st.expander("Objectifs de la Web App", expanded=True):
        st.write("""
        * **Analyse descriptive** : Flux rejetés/autorisés par protocoles TCP/UDP[cite: 166].
        * **Exploration** : Parcours des données via renderDataTable[cite: 169].
        * **Interactivité** : Visualisation des IP sources et occurrences de destinations[cite: 170].
        * **Analytique** : Scénarios de Machine Learning et IA (Clustering/LLM)[cite: 178, 184].
        """)

if __name__ == "__main__":
    main()