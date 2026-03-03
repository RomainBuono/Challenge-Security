import sys
from pathlib import Path

# Calcul dynamique du chemin absolu vers la racine du projet (Challenge-Security)
root_path = Path(__file__).resolve().parent.parent.parent

# Ajout au PYTHONPATH s'il n'y est pas déjà
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.app.utils import get_db_client


def render_vulnerable_ports(db):
    """Tier 1: Threat Surface."""
    st.subheader("🚨 Surveillance des ports critiques")
    df_vuln = db.get_vulnerable_ports_stats()

    if not df_vuln.empty:
        df_vuln["dstport"] = (
            pd.to_numeric(df_vuln["dstport"], errors="coerce").fillna(0).astype(int)
        )

        port_mapping = {
            21: "FTP (21)",
            22: "SSH (22)",
            23: "Telnet (23)",
            80: "HTTP (80)",
            110: "POP3 (110)",
            445: "SMB (445)",
            3306: "MySQL (3306)",
            3389: "RDP (3389)",
            8080: "HTTP-alt (8080)",
        }

        df_vuln["port_name"] = df_vuln["dstport"].map(port_mapping)
        df_vuln = df_vuln.dropna(subset=["port_name"])

        if not df_vuln.empty:
            df_vuln["action"] = df_vuln["action"].str.capitalize()

            fig = px.bar(
                df_vuln,
                x="port_name",
                y="count",
                color="action",
                title="Taux Accepté / Refusé sur les ports ciblés",
                color_discrete_map={"Permit": "#2ecc71", "Deny": "#e74c3c"},
                barmode="stack",
                category_orders={"action": ["Permit", "Deny"]},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun trafic détecté sur les ports critiques spécifiés.")
    else:
        st.info("Aucune donnée disponible pour les ports critiques.")


def render_vue1_descriptive_analysis(db) -> None:
    """Tier 2: Descriptive Analysis (Vue 1)."""
    st.subheader("📈 Vue 1 : Analyse descriptive des flux")

    c1, c2 = st.columns(2)
    with c1:
        port_ranges = {
            "Tous les ports (0 - 65535)": (0, 65535),
            "System Ports (0 - 1023)": (0, 1023),
            "User Ports (1024 - 49151)": (1024, 49151),
            "Dynamic/Private Ports (49152 - 65535)": (49152, 65535),
        }
        selected_range = st.selectbox(
            "Plage de ports (RFC 6056) :", options=list(port_ranges.keys())
        )
        p_min, p_max = port_ranges[selected_range]

    with c2:
        rule_filter_enabled = st.checkbox("Activer le filtre par Règle Firewall")
        selected_rule = None
        if rule_filter_enabled:
            selected_rule = st.number_input(
                "ID de la règle (policyid) :", min_value=1, value=34, step=1
            )

    df_vue1 = db.get_vue1_data(rule_id=selected_rule, port_min=p_min, port_max=p_max)

    if df_vue1.empty:
        st.info("Aucune donnée correspondant à ces filtres.")
        return

    df_vue1["action"] = df_vue1["action"].str.capitalize()
    color_map = {"Permit": "#2ecc71", "Deny": "#e74c3c"}

    col_bar, col_time = st.columns([1, 2])

    with col_bar:
        df_proto = df_vue1.groupby(["proto", "action"])["count"].sum().reset_index()
        fig_bar = px.bar(
            df_proto,
            x="proto",
            y="count",
            color="action",
            barmode="group",
            title="Volume par Protocole",
            color_discrete_map=color_map,
            category_orders={"action": ["Permit", "Deny"]},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_time:
        df_time = (
            df_vue1.groupby(["time_window", "action"])["count"].sum().reset_index()
        )
        fig_time = px.line(
            df_time,
            x="time_window",
            y="count",
            color="action",
            markers=True,
            title="Répartition Temporelle (Activité par heure)",
            color_discrete_map=color_map,
            category_orders={"action": ["Permit", "Deny"]},
        )
        fig_time.update_layout(yaxis_rangemode="tozero")
        st.plotly_chart(fig_time, use_container_width=True)


def render_statistics_section(db):
    """Tier 3: Statistics and Anomalies."""
    st.subheader("📊 Vue 4 : Statistiques & Alertes d'Adressage")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 5 IPs sources émettrices**")
        df_top_ips = db.get_top_source_ips()
        st.dataframe(df_top_ips, hide_index=True, use_container_width=True)

    with c2:
        st.markdown("**Top 10 Ports Système (<1024) Autorisés**")
        df_top_ports = db.get_top_system_ports_permitted()
        st.dataframe(df_top_ports, hide_index=True, use_container_width=True)

    st.markdown("**Alertes : Adresses IP hors plan universitaire (159.84.x.x)**")
    df_external = db.get_external_ip_accesses()
    if not df_external.empty:
        st.warning(
            f"{len(df_external)} événements détectés depuis des sources externes."
        )
        st.dataframe(df_external, height=200, use_container_width=True)
    else:
        st.success("Aucun trafic externe détecté.")


def render_vue3_source_analysis(db) -> None:
    """Tier 3: Source Entity Behavior (Vue 3)."""
    st.subheader("🎯 Vue 3 : Analyse comportementale par IP Source")

    df_scatter = db.get_vue3_scatter_data()

    if not df_scatter.empty:
        fig = px.scatter(
            df_scatter,
            x="dest_count",
            y="total_flows",
            color="deny_ratio",
            color_continuous_scale="Reds",  # Une couleur foncée indique un fort taux de rejet
            hover_name="ipsrc",
            hover_data={
                "dest_count": True,
                "total_flows": True,
                "permit_count": True,
                "deny_count": True,
                "deny_ratio": ":.1f",  # Formatage à 1 décimale
            },
            labels={
                "dest_count": "Destinations contactées (Axe X)",
                "total_flows": "Volume de flux (Axe Y)",
                "deny_ratio": "Taux de DENY (%)",
                "permit_count": "Flux PERMIT",
                "deny_count": "Flux DENY",
            },
            title="Détection d'anomalies : Volume de requêtes vs Diversité des cibles",
        )

        fig.update_layout(coloraxis_colorbar=dict(title="Taux de rejet (%)"))
        fig.update_xaxes(tickformat="d", dtick=1)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("💡 Aide à l'interprétation de la Vue 3"):
            st.markdown("""
            * **Scanners de réseau :** Étirement horizontal vers la droite (beaucoup de cibles, peu de volume par cible).
            * **Attaques par inondation (DDoS/Flood) :** Étirement vertical vers le haut (énorme volume sur un nombre restreint de cibles).
            * **Couleur rouge :** L'IP se fait bloquer massivement, signalant un comportement potentiellement hostile identifié par les règles pare-feu.
            """)
    else:
        st.info("Aucune donnée disponible pour l'analyse par IP source.")


def render_port_scan_analysis(db) -> None:
    """Tier 3: Port Scan Detection."""
    st.subheader("🕵️‍♂️ Radar de Scans de Ports (Port Scanning)")

    df_ports = db.get_port_scan_data()

    if not df_ports.empty:
        # On filtre les IPs qui ne contactent qu'un seul port pour épurer le graphique visuellement
        df_suspects = df_ports[df_ports["distinct_ports"] > 1]

        if not df_suspects.empty:
            fig = px.scatter(
                df_suspects,
                x="distinct_ports",
                y="deny_ratio",
                size="total_flows",
                color="deny_ratio",
                color_continuous_scale="Reds",
                hover_name="ipsrc",
                hover_data={
                    "distinct_ports": True,
                    "total_flows": True,
                    "deny_ratio": ":.1f",
                    "permit_count": False,
                    "deny_count": False,
                },
                labels={
                    "distinct_ports": "Nombre de ports distincts ciblés (Axe X)",
                    "deny_ratio": "Taux de DENY (%) (Axe Y)",
                    "total_flows": "Volume total",
                },
                title="Identification des balayages réseau (Bulles = Volume de flux)",
            )

            # Quadrillage et présentation
            fig.update_xaxes(tickformat="d")
            fig.update_layout(coloraxis_colorbar=dict(title="Taux de rejet (%)"))

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("💡 Comment lire ce radar ?"):
                st.markdown("""
                * **Zone de danger (En haut à droite) :** Une IP qui cible un grand nombre de ports avec un fort taux de rejet effectue un **scan de ports agressif** (ex: Nmap).
                * **Taille de la bulle :** Indique l'intensité du scan. Une grosse bulle en haut à droite caractérise une attaque bruyante.
                """)
        else:
            st.info(
                "Aucun comportement de balayage multi-ports détecté pour le moment."
            )
    else:
        st.info("Aucune donnée disponible.")


def main() -> None:
    st.set_page_config(page_title="Dashboard Sécurité", layout="wide")
    db = get_db_client()
    st.title("🛡️ Dashboard de l'état du SI")

    try:
        stats = db.get_security_ratios()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Flux", f"{stats['total']:,}")
        k2.metric("Autorisés", f"{stats['accepted']:,}")
        k3.metric("Bloqués", f"{stats['rejected']:,}")
        k4.metric("Ratio d'Acceptation", f"{stats['ratio']:.1f}%")
        st.divider()

        render_vulnerable_ports(db)
        st.divider()

        render_vue1_descriptive_analysis(db)
        st.divider()
        render_vue3_source_analysis(db)
        st.divider()
        render_port_scan_analysis(db)
        st.divider()
        render_statistics_section(db)

    except Exception as e:
        st.error(f"Erreur d'exécution : {e}")


if __name__ == "__main__":
    main()
