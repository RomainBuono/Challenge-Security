import time
import warnings
from pathlib import Path
from functools import lru_cache
from collections import Counter

import requests
import polars as pl
import numpy as np
import folium
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.express as px

# Désactivation des warnings
warnings.filterwarnings("ignore")

class GeoSecurityConfig:
    """Configuration centralisée de l'analyse géographique et temporelle."""
    
    COLORS = {
        "bg": "#0d1117", "bg_panel": "#161b22", "grid": "#21262d",
        "text": "#e6edf3", "muted": "#8b949e", "border": "#30363d",
        "accent": "#58a6ff", "danger": "#f85149", "warning": "#d29922",
        "success": "#3fb950", "purple": "#bc8cff"
    }

    # Dictionnaire des ports connus
    KNOWN_PORTS = {
        21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
        80: "HTTP", 110: "POP3", 143: "IMAP", 443: "HTTPS", 445: "SMB",
        135: "RPC", 139: "NetBIOS", 993: "IMAPS", 995: "POP3S", 3389: "RDP"
    }

    # Référentiel des fuseaux horaires par pays (Offset UTC)
    TIMEZONES = {
        "France": 1, "Germany": 1, "Spain": 1, "Italy": 1, "United Kingdom": 0,
        "United States": -5, "Canada": -5, "Brazil": -3, "China": 8, "Japan": 9,
        "India": 5.5, "Russia": 3, "Australia": 10, "South Africa": 2
        # Tu peux étendre cette liste selon tes besoins
    }


class AdvancedSecurityAnalyzer:
    """Analyseur de logs avec géolocalisation et séries temporelles (Niveau Production)."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.df: pl.DataFrame | None = None
        self.ip_locations: dict[str, dict] = {}
        self._setup_style()

    def _setup_style(self) -> None:
        """Application du thème sombre cohérent pour Matplotlib et Seaborn."""
        c = GeoSecurityConfig.COLORS
        plt.rcParams.update({
            "figure.facecolor": c["bg"], "axes.facecolor": c["bg"],
            "axes.edgecolor": c["border"], "text.color": c["text"],
            "axes.labelcolor": c["text"], "xtick.color": c["muted"],
            "ytick.color": c["muted"], "grid.color": c["grid"],
            "grid.linestyle": "--", "grid.linewidth": 0.6,
        })
        # Forcer Seaborn à respecter notre palette sombre
        sns.set_theme(style="darkgrid", rc={"axes.facecolor": c["bg_panel"], "figure.facecolor": c["bg"], "text.color": c["text"]})

    def load_data_lazy(self) -> None:
        """
        Extraction optimisée (Lazy Evaluation).
        Plutôt que de lire ligne par ligne en Python, on laisse le moteur Rust
        de Polars filtrer le fichier de 2Go à la volée pendant la lecture.
        """
        if not self.log_path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {self.log_path}")

        print("⏳ Chargement Lazy (filtrage à la volée)...")
        
        column_names = ["timestamp", "src_ip", "dst_ip", "proto", "src_port", 
                        "dst_port", "rule_id", "action", "interface", "col_9", "proto_num"]

        # Création du plan d'exécution Lazy
        lazy_query = (
            pl.scan_csv(self.log_path, separator=";", has_header=False, new_columns=column_names, infer_schema_length=0)
            .filter(
                pl.col("timestamp").str.starts_with("2025-11") |
                pl.col("timestamp").str.starts_with("2025-12") |
                pl.col("timestamp").str.starts_with("2026-01") |
                pl.col("timestamp").str.starts_with("2026-02")
            )
            .select(pl.exclude(["col_9", "proto_num"])) # Exclusion directe des colonnes inutiles
            .with_columns([
                pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False),
                pl.col("src_port").cast(pl.Int32, strict=False),
                pl.col("dst_port").cast(pl.Int32, strict=False),
                pl.col("rule_id").cast(pl.Int32, strict=False),
                pl.col("proto").str.to_uppercase().str.strip_chars().alias("protocol_clean")
            ])
            .with_columns([
                pl.col("timestamp").dt.hour().alias("hour"),
                pl.col("timestamp").dt.date().alias("date"),
                pl.col("timestamp").dt.weekday().alias("weekday") # 1=Lun, 7=Dim
            ])
            .drop_nulls(subset=["timestamp", "src_ip"])
        )

        # Exécution du graphe de calcul (Multi-threadé en Rust)
        self.df = lazy_query.collect()
        print(f"✅ Chargement terminé : {self.df.height:,} lignes conservées en mémoire.")

    @staticmethod
    @lru_cache(maxsize=200)
    def _fetch_geolocation(ip: str) -> dict | None:
        """
        Appel API géolocalisation.
        @lru_cache permet de ne jamais requêter 2 fois la même IP.
        """
        try:
            response = requests.get(f"http://ip-api.com/json/{ip}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    return {
                        "ip": ip, "country": data.get("country", "Unknown"),
                        "city": data.get("city", "Unknown"), "lat": data.get("lat", 0),
                        "lon": data.get("lon", 0), "isp": data.get("isp", "Unknown")
                    }
        except Exception as e:
            print(f"⚠️ Erreur géoloc pour {ip}: {e}")
        return None

    def geolocate_top_ips(self) -> None:
        """Récupère les coordonnées des 20 IPs sources et 20 destinations les plus actives."""
        print("🌍 Géolocalisation des acteurs principaux en cours...")
        
        top_src = self.df["src_ip"].value_counts().sort("count", descending=True).head(20)
        top_dst = self.df["dst_ip"].value_counts().sort("count", descending=True).head(20)
        
        unique_ips_to_fetch = set(top_src["src_ip"].to_list() + top_dst["dst_ip"].to_list())
        
        for i, ip in enumerate(unique_ips_to_fetch):
            loc = self._fetch_geolocation(ip)
            if loc:
                self.ip_locations[ip] = loc
            # Délai uniquement si l'IP n'était pas en cache pour respecter la limite API
            time.sleep(1.4) 
            
        print(f"✅ {len(self.ip_locations)} adresses IP localisées avec succès.")

    # =========================================================
    # VISUALISATIONS & CARTOGRAPHIE
    # =========================================================

    def plot_static_map_png(self) -> None:
        """
        Génère une carte statique PNG en utilisant Plotly (Native/Vectorisé).
        Pas de navigateur Headless, pas de HTML temporaire, exécution instantanée.
        """
        if not self.ip_locations:
            print("❌ Cartographie annulée : Aucune donnée de géolocalisation.")
            return

        print("🗺️  Génération de la carte statique Plotly en cours...")
        
        top_src_df = self.df["src_ip"].value_counts().sort("count", descending=True).head(20)
        
        map_data = []
        for row in top_src_df.iter_rows(named=True):
            ip = row["src_ip"]
            count = row["count"]
            loc = self.ip_locations.get(ip)
            
            if loc:
                map_data.append({
                    "Adresse IP": ip, "Nombre de hits": count,
                    "Latitude": loc["lat"], "Longitude": loc["lon"], "Pays": loc["country"]
                })

        if not map_data:
            print("⚠️ Aucune des IPs n'a pu être placée sur la carte (IPs privées ?).")
            return

        plot_df = pl.DataFrame(map_data)

        fig = px.scatter_mapbox(
            plot_df, lat="Latitude", lon="Longitude",
            size="Nombre de hits", color="Nombre de hits",
            color_continuous_scale="Reds", hover_name="Adresse IP",
            hover_data=["Pays"], size_max=30, zoom=1.2,
            title="<b>Cartographie des 20 IP sources les plus agressives</b>"
        )

        fig.update_layout(
            mapbox_style="carto-positron",
            margin={"r": 0, "t": 50, "l": 0, "b": 0}, title_font_size=18, title_x=0.5,
            coloraxis_colorbar=dict(
                title="<b>Volume de trafic</b>", thicknessmode="pixels", thickness=15,
                lenmode="pixels", len=300, yanchor="bottom", y=0.1, x=0.02
            )
        )

        try:
            fig.write_image("map_sources_ip.png", width=1200, height=700, scale=2)
            print(" 💾 map_sources_ip.png générée nativement avec succès (haute résolution).")
        except Exception as e:
            print(f"❌ Erreur lors de l'export d'image Plotly : {e}")

    def plot_temporal_analysis(self) -> None:
        """Analyse temporelle avancée (Heure UTC vs Locale)."""
        top_5_ips = self.df["src_ip"].value_counts().sort("count", descending=True).head(5)["src_ip"].to_list()
        df_top5 = self.df.filter(pl.col("src_ip").is_in(top_5_ips))

        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)
        colors = [GeoSecurityConfig.COLORS["danger"], GeoSecurityConfig.COLORS["warning"], 
                  GeoSecurityConfig.COLORS["accent"], GeoSecurityConfig.COLORS["success"], 
                  GeoSecurityConfig.COLORS["purple"]]

        # 1. Distribution Globale par Heure
        ax1 = fig.add_subplot(gs[0, 0])
        hourly_dist = df_top5.group_by("hour").agg(pl.len().alias("count")).sort("hour")
        ax1.bar(hourly_dist["hour"], hourly_dist["count"], color=GeoSecurityConfig.COLORS["accent"], alpha=0.8)
        ax1.set_title("Distribution des Connexions par Heure (UTC)", fontweight="bold")
        ax1.set_xticks(range(0, 24)); ax1.grid(axis="y", alpha=0.3)

        # 2. Profils Horaires Individuels (Lignes)
        ax2 = fig.add_subplot(gs[0, 1])
        for i, ip in enumerate(top_5_ips):
            ip_hourly = df_top5.filter(pl.col("src_ip") == ip).group_by("hour").agg(pl.len().alias("count")).sort("hour")
            # Compléter les heures manquantes avec 0 (astuce NumPy)
            full_day = np.zeros(24)
            for h, c in zip(ip_hourly["hour"], ip_hourly["count"]):
                full_day[h] = c
            ax2.plot(range(24), full_day, marker="o", linewidth=2, label=ip, color=colors[i])
            
        ax2.set_title("Profils Horaires - Top 5 IP Sources", fontweight="bold")
        ax2.legend(); ax2.set_xticks(range(0, 24)); ax2.grid(True, alpha=0.3)

        # 3. Heatmap d'Activité
        ax3 = fig.add_subplot(gs[1, :])
        heatmap_matrix = np.zeros((5, 24))
        for i, ip in enumerate(top_5_ips):
            ip_hourly = df_top5.filter(pl.col("src_ip") == ip).group_by("hour").agg(pl.len().alias("count"))
            for h, c in zip(ip_hourly["hour"], ip_hourly["count"]):
                heatmap_matrix[i, h] = c

        sns.heatmap(heatmap_matrix, yticklabels=top_5_ips, xticklabels=range(24), 
                    cmap="Reds", ax=ax3, cbar_kws={"label": "Connexions"})
        ax3.set_title("Heatmap Horaire des Top 5 IPs", fontweight="bold")

        plt.savefig("temporal_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(" 💾 temporal_analysis.png générée.")

    def plot_tcp_deep_dive(self) -> None:
        """Analyse matricielle TCP : Ports vs Actions."""
        df_tcp = self.df.filter(pl.col("protocol_clean") == "TCP")
        
        # Pivot Polars natif remplaçant le groupby.unstack()
        top_ports = df_tcp["dst_port"].value_counts().sort("count", descending=True).head(15)["dst_port"].to_list()
        
        pivot_matrix = (
            df_tcp.filter(pl.col("dst_port").is_in(top_ports))
            .group_by(["dst_port", "action"])
            .agg(pl.len().alias("count"))
            .pivot(values="count", index="dst_port", on="action", aggregate_function="sum")
            .fill_null(0)
        )
        
        # Formatage pour Seaborn
        actions = [col for col in pivot_matrix.columns if col != "dst_port"]
        matrix_data = pivot_matrix.select(actions).to_numpy()
        port_labels = [f"{p} ({GeoSecurityConfig.KNOWN_PORTS.get(p, 'Unk')})" for p in pivot_matrix["dst_port"]]

        plt.figure(figsize=(12, 8))
        sns.heatmap(matrix_data, annot=True, fmt=".0f", cmap="YlOrRd", 
                    yticklabels=port_labels, xticklabels=actions)
        
        plt.title("Répartition des Actions par Port de Destination (TCP)", fontweight="bold")
        plt.tight_layout()
        plt.savefig("tcp_port_actions.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(" 💾 tcp_port_actions.png générée.")

    def run_pipeline(self) -> None:
        """Orchestrateur global."""
        print("\n🚀 Lancement du Pipeline de Sécurité Avancé...")
        self.geolocate_top_ips()
        self.plot_static_map_png() # ✅ Appel de la bonne méthode
        self.plot_temporal_analysis()
        self.plot_tcp_deep_dive()
        print("\n✅ Analyse terminée avec succès. Rapports disponibles dans le dossier courant.")

if __name__ == "__main__":
    analyzer = AdvancedSecurityAnalyzer("log_export.log")
    analyzer.load_data_lazy()
    analyzer.run_pipeline()