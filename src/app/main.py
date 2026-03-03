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
    st.subheader("📈 Monitoring du trafic (Vue temporelle)")
    st.caption("💡 Utilisez la barre latérale, les filtres ci-dessous ou interagissez directement avec le graphique (Pan/Zoom) pour explorer les données.")

    min_dt, max_dt = db.get_time_bounds()
    
    time_range = st.slider(
        "Restriction de la plage temporelle :",
        min_value=min_dt.to_pydatetime(),
        max_value=max_dt.to_pydatetime(),
        value=(min_dt.to_pydatetime(), max_dt.to_pydatetime()),
        format="YYYY-MM-DD HH:mm:ss",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        port_ranges: dict[str, tuple[int, int]] = {
            "Tous les ports (0 - 65535)": (0, 65535),
            "System Ports (0 - 1023)": (0, 1023),
            "User Ports (1024 - 49151)": (1024, 49151),
            "Dynamic/Private Ports (49152 - 65535)": (49152, 65535),
        }
        selected_range = st.selectbox("Plage de ports :", options=list(port_ranges.keys()))
        p_min, p_max = port_ranges[selected_range] if selected_range else (0, 65535)

    with c2:
        granularity_map: dict[str, str] = {
            "Par seconde": "second",
            "Par minute": "minute",
            "Par heure": "hour"
        }
        selected_granularity = st.selectbox("Granularité temporelle :", options=list(granularity_map.keys()), index=1)
        granularity = granularity_map[selected_granularity] if selected_granularity else "minute"

    with c3:
        rule_filter_enabled = st.checkbox("Activer le filtre par Règle Firewall")
        selected_rule = None
        if rule_filter_enabled:
            selected_rule = int(st.number_input("ID de la règle :", min_value=1, value=34, step=1))

    df_vue1 = db.get_vue1_data(
        rule_id=selected_rule, 
        port_min=p_min, 
        port_max=p_max, 
        granularity=granularity,
        start_time=time_range[0].strftime("%Y-%m-%d %H:%M:%S"),
        end_time=time_range[1].strftime("%Y-%m-%d %H:%M:%S")
    )

    if df_vue1.empty:
        st.info("Aucune donnée correspondant à ces filtres.")
        return

    df_vue1["action"] = df_vue1["action"].str.capitalize()
    df_time = df_vue1.groupby(["time_window", "action"], as_index=False)["count"].sum()

    fig_time = px.bar(
        df_time,
        x="time_window",
        y="count",
        color="action",
        color_discrete_map={"Permit": "#2ecc71", "Deny": "#e74c3c"},
        category_orders={"action": ["Permit", "Deny"]},
        barmode="stack"
    )
    
    fig_time.update_layout(
        xaxis_title="Temps",
        yaxis_title="Volume de requêtes",
        hovermode="x unified",
        legend_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0),
        bargap=0.05
    )
    
    fig_time.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
    
    fig_time.update_xaxes(
        showgrid=False,
        range=[time_range[0], time_range[1]],
        constrain='domain'
    )

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

        # Inversion : Monitoring temporel en premier
        render_vue1_descriptive_analysis(db)
        st.divider()
        
        # Ports vulnérables en second
        render_vulnerable_ports(db)
        st.divider()
    
        render_port_scan_analysis(db)
        st.divider()
        
        render_statistics_section(db)

    except Exception as e:
        st.error(f"Erreur d'exécution : {e}")

if __name__ == "__main__":
    main()
