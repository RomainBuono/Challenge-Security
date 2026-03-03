import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# Désactivation des warnings non critiques pour la lisibilité de la console
warnings.filterwarnings("ignore")

class FirewallConfig:
    """Configuration statique : sépare les paramètres (couleurs, ports) de la logique métier."""
    
    COLORS = {
        "bg": "#0d1117", "bg_panel": "#161b22", "grid": "#21262d",
        "text": "#e6edf3", "muted": "#8b949e", "border": "#30363d",
        "accent": "#58a6ff", "danger": "#f85149", "warning": "#d29922",
        "success": "#3fb950", "purple": "#bc8cff"
    }

    TCP_PORTS = {20, 21, 22, 23, 25, 80, 110, 143, 443, 445, 465, 587, 993, 995, 1433, 1521, 3306, 3389, 5432, 5900, 6379, 8080, 8443, 27017}
    UDP_PORTS = {53, 67, 68, 69, 123, 161, 162, 500, 514, 1194, 4500, 5060, 5353, 1900, 4789}
    KNOWN_PORTS = {
        21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
        80: "HTTP", 110: "POP3", 143: "IMAP", 443: "HTTPS", 445: "SMB",
        1433: "MSSQL", 1521: "Oracle", 3306: "MySQL", 3389: "RDP",
        5432: "PgSQL", 5900: "VNC", 6379: "Redis", 8080: "HTTP-alt",
        8443: "HTTPS-alt", 27017: "MongoDB",
    }


class FirewallLogAnalyzer:
    """Analyseur de logs réseau de niveau production."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.df: pl.DataFrame | None = None
        self._setup_style()

    def _setup_style(self) -> None:
        """Applique le thème sombre global à Matplotlib."""
        c = FirewallConfig.COLORS
        plt.rcParams.update({
            "figure.facecolor": c["bg"], "axes.facecolor": c["bg"],
            "axes.edgecolor": c["border"], "text.color": c["text"],
            "axes.labelcolor": c["text"], "xtick.color": c["muted"],
            "ytick.color": c["muted"], "grid.color": c["grid"],
            "grid.linestyle": "--", "grid.linewidth": 0.6,
            "font.family": "monospace",
        })

    def load_and_preprocess(self) -> None:
        """Pipeline ETL : Extraction, Transformation par Regex et Typage, Chargement."""
        if not self.log_path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {self.log_path}")

        print("⏳ Chargement et parsing regex...")
        df_raw = pl.read_csv(self.log_path, separator=";", has_header=False, new_columns=["raw"], infer_schema_length=0)

        df = df_raw.with_columns([
            pl.col("raw").str.extract(r'^(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})', 1).alias("timestamp_raw"),
            pl.col("raw").str.extract(r'SRC=([\d.]+)', 1).alias("src_ip"),
            pl.col("raw").str.extract(r'DST=([\d.]+)', 1).alias("dst_ip"),
            pl.col("raw").str.extract(r'PROTO=(\S+)', 1).str.to_uppercase().alias("protocol"),
            pl.col("raw").str.extract(r'SPT=(\d+)', 1).cast(pl.Int32, strict=False).alias("src_port"),
            pl.col("raw").str.extract(r'DPT=(\d+)', 1).cast(pl.Int32, strict=False).alias("dst_port"),
            pl.col("raw").str.extract(r'RULE=(\S+)', 1).alias("rule_id"),
            pl.col("raw").str.extract(r'\]\s*(DENY|PERMIT|ALLOW|BLOCK|DROP|REJECT)\b', 1).str.to_uppercase().alias("action"),
            pl.col("raw").str.extract(r'\bIN=(\S+)', 1).alias("interface"),
            pl.col("raw").str.extract(r'FW=(\d+)', 1).cast(pl.Int32, strict=False).alias("protocol_num"),
        ])

        # Gestion temporelle vectorisée (Polars natif)
        df = (
            df.with_columns(
                pl.concat_str([pl.lit("2025 "), pl.col("timestamp_raw")])
                .str.strptime(pl.Datetime, format="%Y %b %d %H:%M:%S", strict=False)
                .alias("timestamp")
            )
            .with_columns(
                pl.when(pl.col("timestamp").dt.month() <= 2)
                .then(pl.col("timestamp").dt.offset_by("1y"))
                .otherwise(pl.col("timestamp"))
                .alias("timestamp")
            )
            .filter(pl.col("src_ip").is_not_null() & pl.col("action").is_not_null() & pl.col("timestamp").is_not_null())
            .drop(["raw", "timestamp_raw"])
            .with_columns([
                pl.col("timestamp").dt.hour().alias("hour"),
                pl.col("timestamp").dt.date().alias("date"),
                pl.col("timestamp").dt.weekday().alias("weekday"), # 1=Lundi, 7=Dimanche
            ])
        )

        # Filtre période
        self.df = df.filter(
            ((pl.col("timestamp").dt.year() == 2025) & (pl.col("timestamp").dt.month() >= 11)) |
            ((pl.col("timestamp").dt.year() == 2026) & (pl.col("timestamp").dt.month() <= 2))
        )
        self._deduce_protocols()
        
        d_min, d_max = self.df["timestamp"].min(), self.df["timestamp"].max()
        duree_h = (d_max - d_min).total_seconds() / 3600 if d_max and d_min else 0
        print(f"✅ Prétraitement terminé : {self.df.height:,} évènements conservés ({duree_h:.0f}h analysées).")

    def _deduce_protocols(self) -> None:
        """Déduction performante du protocole (remplace le lambda map_elements)."""
        self.df = self.df.with_columns(
            pl.when(pl.col("protocol").is_in(["TCP", "UDP", "ICMP"])).then(pl.col("protocol"))
            .when((pl.col("protocol") == "6") | (pl.col("protocol_num") == 6)).then(pl.lit("TCP"))
            .when((pl.col("protocol") == "17") | (pl.col("protocol_num") == 17)).then(pl.lit("UDP"))
            .when((pl.col("protocol") == "1") | (pl.col("protocol_num") == 1)).then(pl.lit("ICMP"))
            .when(pl.col("dst_port").is_in(list(FirewallConfig.TCP_PORTS))).then(pl.lit("TCP"))
            .when(pl.col("dst_port").is_in(list(FirewallConfig.UDP_PORTS))).then(pl.lit("UDP"))
            .otherwise(pl.lit("AUTRE"))
            .alias("protocol_clean")
        )

    # =========================================================
    # VISUALISATIONS GRAPHIQUES
    # =========================================================

    def plot_top_rules_generic(self, protocol: str | None, top_n: int, color_key: str, title: str, filename: str) -> None:
        """Q1, Q3, Q4 - Méthode factorisée (DRY) pour les barres horizontales de classements."""
        data = self.df if protocol is None else self.df.filter(pl.col("protocol_clean") == protocol)
        rules = data.group_by("rule_id").agg(pl.len().alias("hits")).sort("hits", descending=True).head(top_n)
        labels, hits = rules["rule_id"].to_list()[::-1], rules["hits"].to_list()[::-1]
        
        fig, ax = plt.subplots(figsize=(11, 6))
        cmap = LinearSegmentedColormap.from_list("custom", [FirewallConfig.COLORS["bg_panel"], FirewallConfig.COLORS[color_key]])
        colors = [cmap(i / max(len(labels)-1, 1)) for i in range(len(labels))]

        bars = ax.barh(labels, hits, color=colors, edgecolor=FirewallConfig.COLORS["bg"], linewidth=0.5)
        for bar, cnt in zip(bars, hits):
            ax.text(bar.get_width() + max(hits)*0.005, bar.get_y() + bar.get_height()/2,
                    f"{cnt:,}", va="center", fontsize=9, color=FirewallConfig.COLORS["muted"])

        ax.set_title(title, pad=14, fontweight="bold")
        ax.set_xlabel("Nombre de hits"); ax.set_ylabel("Rule ID")
        ax.grid(axis="x"); ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f" 💾 {filename}")

    def plot_protocol_distribution(self) -> None:
        """Q2 - Histogramme de distribution des protocoles détectés."""
        counts = self.df["protocol_clean"].value_counts().sort("count", descending=True)
        protos, cnts = counts["protocol_clean"].to_list(), counts["count"].to_list()
        
        color_map = {"TCP": "accent", "UDP": "success", "ICMP": "warning", "AUTRE": "muted"}
        p_colors = [FirewallConfig.COLORS.get(color_map.get(p, "purple"), FirewallConfig.COLORS["purple"]) for p in protos]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(protos, cnts, color=p_colors, width=0.5, edgecolor=FirewallConfig.COLORS["bg"])
        for bar, cnt in zip(bars, cnts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cnts)*0.01,
                    f"{cnt:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_title("Q2 — Distribution des protocoles détectés", pad=14, fontweight="bold")
        ax.set_ylabel("Nombre d'événements")
        ax.grid(axis="y"); ax.set_axisbelow(True)
        
        ax.text(0.98, 0.97, "Méthode de déduction :\n① Champ PROTO\n② Numéro IANA/FW\n③ Port DST",
                 transform=ax.transAxes, ha="right", va="top", fontsize=8, color=FirewallConfig.COLORS["muted"],
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=FirewallConfig.COLORS["bg_panel"], edgecolor=FirewallConfig.COLORS["border"]))
        
        plt.tight_layout()
        plt.savefig("q2_protocoles.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(" 💾 q2_protocoles.png")

    def plot_rule_port_heatmap(self) -> None:
        """Q5a - Heatmap Rule x Port DST optimisée."""
        df_tcp = self.df.filter((pl.col("protocol_clean") == "TCP") & pl.col("dst_port").is_not_null())
        top10_rules = df_tcp.group_by("rule_id").agg(pl.len().alias("n")).sort("n", descending=True).head(10)["rule_id"].to_list()
        top10_ports = df_tcp.group_by("dst_port").agg(pl.len().alias("n")).sort("n", descending=True).head(10)["dst_port"].to_list()
        
        port_labels = [f"{p}\n({FirewallConfig.KNOWN_PORTS.get(p, '?')})" for p in top10_ports]

        # Calcul matriciel via le moteur Polars
        pivot_df = (
            df_tcp.filter(pl.col("rule_id").is_in(top10_rules) & pl.col("dst_port").is_in(top10_ports))
            .group_by(["rule_id", "dst_port"]).agg(pl.len().alias("n"))
            .pivot(values="n", index="rule_id", on="dst_port", aggregate_function="sum")
            .fill_null(0)
        )
        
        heat_matrix = np.zeros((len(top10_rules), len(top10_ports)))
        for i, rule in enumerate(top10_rules):
            row = pivot_df.filter(pl.col("rule_id") == rule)
            if not row.is_empty():
                for j, port in enumerate(top10_ports):
                    if str(port) in row.columns:
                        heat_matrix[i, j] = row[str(port)][0]

        fig, ax = plt.subplots(figsize=(14, 7))
        cmap_h = LinearSegmentedColormap.from_list("threat", [FirewallConfig.COLORS["bg"], "#1f3a5f", FirewallConfig.COLORS["danger"]])
        im = ax.imshow(heat_matrix, aspect="auto", cmap=cmap_h)

        ax.set_xticks(range(len(top10_ports))); ax.set_xticklabels(port_labels, fontsize=8)
        ax.set_yticks(range(len(top10_rules))); ax.set_yticklabels(top10_rules, fontsize=9)
        ax.set_title("Q5a — Heatmap : Rule ID × Port destination — TCP", pad=14, fontweight="bold")
        ax.set_xlabel("Port de destination"); ax.set_ylabel("Rule ID")
        plt.colorbar(im, ax=ax, label="Nb événements", shrink=0.8)

        max_val = heat_matrix.max()
        for i in range(len(top10_rules)):
            for j in range(len(top10_ports)):
                val = int(heat_matrix[i, j])
                if val > 0:
                    ax.text(j, i, f"{val:,}", ha="center", va="center", fontsize=7,
                            color="white" if val > max_val * 0.45 else FirewallConfig.COLORS["muted"])

        plt.tight_layout()
        plt.savefig("q5a_heatmap_rule_port.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(" 💾 q5a_heatmap_rule_port.png")

    def plot_rule_action_stacked(self) -> None:
        """Q5b - Barres empilées Rule x Action."""
        df_tcp = self.df.filter(pl.col("protocol_clean") == "TCP")
        top10_rules = df_tcp.group_by("rule_id").agg(pl.len().alias("n")).sort("n", descending=True).head(10)["rule_id"].to_list()

        pivot_actions = (
            df_tcp.filter(pl.col("rule_id").is_in(top10_rules))
            .group_by(["rule_id", "action"]).agg(pl.len().alias("n"))
            .pivot(values="n", index="rule_id", on="action", aggregate_function="sum")
            .fill_null(0)
            .with_row_index("idx").sort(pl.col("rule_id").replace({r: i for i, r in enumerate(top10_rules)}))
        )

        action_cols = [c for c in pivot_actions.columns if c not in ["rule_id", "idx"]]
        color_action = {
            "ALLOW": FirewallConfig.COLORS["success"], "PERMIT": FirewallConfig.COLORS["success"],
            "DENY": FirewallConfig.COLORS["danger"], "BLOCK": FirewallConfig.COLORS["danger"],
            "DROP": FirewallConfig.COLORS["danger"], "REJECT": FirewallConfig.COLORS["danger"]
        }

        fig, ax = plt.subplots(figsize=(13, 6))
        x, bottom = np.arange(len(top10_rules)), np.zeros(len(top10_rules))

        for action in action_cols:
            vals = pivot_actions[action].to_numpy()
            col = color_action.get(action, FirewallConfig.COLORS["purple"])
            ax.bar(x, vals, bottom=bottom, label=action, color=col, alpha=0.85, edgecolor=FirewallConfig.COLORS["bg"], linewidth=0.5)
            bottom += vals

        ax.set_xticks(x); ax.set_xticklabels(top10_rules, rotation=30, ha="right", fontsize=9)
        ax.set_title("Q5b — Top 10 règles TCP : répartition DENY / ALLOW par règle", pad=14, fontweight="bold")
        ax.set_ylabel("Nombre d'événements")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.2)
        ax.grid(axis="y"); ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig("q5b_rule_action_stacked.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(" 💾 q5b_rule_action_stacked.png")

    def plot_tcp_spike_analysis(self, spike_start: str = "2025-11-18", spike_end: str = "2025-11-24") -> None:
        """Q5c - Figure multi-panneaux (v2 - 5 subplots) d'analyse du pic d'attaques TCP."""
        df_tcp = self.df.filter((pl.col("protocol_clean") == "TCP") & pl.col("action").is_in(["DENY", "BLOCK", "DROP", "REJECT"]))
        dt_start, dt_end = datetime.strptime(spike_start, "%Y-%m-%d"), datetime.strptime(spike_end, "%Y-%m-%d")

        df_spk = df_tcp.filter(pl.col("timestamp").is_between(dt_start, dt_end))
        df_bsl = df_tcp.filter(~pl.col("timestamp").is_between(dt_start, dt_end))

        # Calcul Comparaison Règles
        r_spk = df_spk.group_by("rule_id").agg(pl.len().alias("spk")).sort("spk", descending=True).head(10)
        r_bsl = df_bsl.group_by("rule_id").agg(pl.len().alias("bsl"))
        comp_r = r_spk.join(r_bsl, on="rule_id", how="left").with_columns(pl.col("bsl").fill_null(1))
        comp_r = comp_r.with_columns((pl.col("spk") / pl.col("bsl")).alias("rat")).sort("spk", descending=True)

        # Calcul Comparaison Ports
        p_spk = df_spk.filter(pl.col("dst_port").is_not_null()).group_by("dst_port").agg(pl.len().alias("spk")).sort("spk", descending=True).head(12)
        p_bsl = df_bsl.filter(pl.col("dst_port").is_not_null()).group_by("dst_port").agg(pl.len().alias("bsl"))
        comp_p = p_spk.join(p_bsl, on="dst_port", how="left").with_columns(pl.col("bsl").fill_null(1))
        comp_p = comp_p.with_columns((pl.col("spk") / pl.col("bsl")).alias("rat")).sort("spk", descending=True)

        top5_rules = comp_r.head(5)["rule_id"].to_list()
        daily_rules = df_tcp.filter(pl.col("rule_id").is_in(top5_rules)).group_by(["date", "rule_id"]).agg(pl.len().alias("n")).sort("date")

        # --- Initialisation de la figure 5 panneaux ---
        fig = plt.figure(figsize=(20, 18))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.38)
        fig.suptitle(f"PLOT 5c — Analyse du pic de blocage TCP ({spike_start} → {spike_end})", fontsize=16, fontweight="bold", y=0.96)

        # Panneau A : Volume Règles
        ax_a = fig.add_subplot(gs[0, 0])
        y_pos = range(len(comp_r))
        ax_a.barh(y_pos, comp_r["bsl"].to_list()[::-1], color=FirewallConfig.COLORS["muted"], alpha=0.5, label="Baseline")
        ax_a.barh(y_pos, comp_r["spk"].to_list()[::-1], color=FirewallConfig.COLORS["danger"], alpha=0.85, label="Pic")
        ax_a.set_yticks(y_pos); ax_a.set_yticklabels(comp_r["rule_id"].to_list()[::-1], fontsize=8)
        ax_a.set_title("A — Volume règles : Pic vs Baseline", fontweight="bold"); ax_a.legend(); ax_a.grid(axis="x")

        # Panneau B : Ratio Règles
        ax_b = fig.add_subplot(gs[0, 1])
        ratios_r = comp_r["rat"].to_list()[::-1]
        cmap_amp = LinearSegmentedColormap.from_list("amp", [FirewallConfig.COLORS["warning"], FirewallConfig.COLORS["danger"]])
        bars_b = ax_b.barh(y_pos, ratios_r, color=[cmap_amp(min(r/max(ratios_r+[1]), 1.0)) for r in ratios_r])
        for bar, r in zip(bars_b, ratios_r):
            ax_b.text(bar.get_width() + max(ratios_r)*0.01, bar.get_y() + bar.get_height()/2, f"×{r:.1f}", va="center", fontsize=8, color=FirewallConfig.COLORS["danger"] if r>5 else FirewallConfig.COLORS["warning"])
        ax_b.set_yticks(y_pos); ax_b.set_yticklabels(comp_r["rule_id"].to_list()[::-1], fontsize=8)
        ax_b.axvline(1, color=FirewallConfig.COLORS["muted"], linestyle="--"); ax_b.set_title("B — Amplification Règles (×N)", fontweight="bold"); ax_b.grid(axis="x")

        # Panneau C : Volume Ports
        ax_c = fig.add_subplot(gs[1, 0])
        yp_pos = range(len(comp_p))
        p_labels = [f"{p}\n({FirewallConfig.KNOWN_PORTS.get(p,'?')})" for p in comp_p["dst_port"].to_list()[::-1]]
        ax_c.barh(yp_pos, comp_p["bsl"].to_list()[::-1], color=FirewallConfig.COLORS["muted"], alpha=0.5)
        ax_c.barh(yp_pos, comp_p["spk"].to_list()[::-1], color=FirewallConfig.COLORS["warning"], alpha=0.85)
        ax_c.set_yticks(yp_pos); ax_c.set_yticklabels(p_labels, fontsize=8)
        ax_c.set_title("C — Volume ports DST ciblés", fontweight="bold"); ax_c.grid(axis="x")

        # Panneau D : Ratio Ports
        ax_d = fig.add_subplot(gs[1, 1])
        ratios_p = comp_p["rat"].to_list()[::-1]
        cmap_pw = LinearSegmentedColormap.from_list("pw", ["#3a2a10", FirewallConfig.COLORS["warning"]])
        bars_d = ax_d.barh(yp_pos, ratios_p, color=[cmap_pw(min(r/max(ratios_p+[1]), 1.0)) for r in ratios_p])
        for bar, r in zip(bars_d, ratios_p):
            ax_d.text(bar.get_width() + max(ratios_p)*0.01, bar.get_y() + bar.get_height()/2, f"×{r:.1f}", va="center", fontsize=8, color=FirewallConfig.COLORS["warning"] if r>3 else FirewallConfig.COLORS["muted"])
        ax_d.set_yticks(yp_pos); ax_d.set_yticklabels(p_labels, fontsize=8)
        ax_d.axvline(1, color=FirewallConfig.COLORS["muted"], linestyle="--"); ax_d.set_title("D — Amplification Ports (×N)", fontweight="bold"); ax_d.grid(axis="x")

        # Panneau E : Timeline Top 5 règles
        ax_e = fig.add_subplot(gs[2, :])
        all_dates = sorted(daily_rules["date"].cast(pl.String).unique().to_list())
        rule_colors = [FirewallConfig.COLORS["danger"], FirewallConfig.COLORS["warning"], FirewallConfig.COLORS["accent"], FirewallConfig.COLORS["success"], FirewallConfig.COLORS["purple"]]
        
        for idx, rule in enumerate(top5_rules):
            rd = daily_rules.filter(pl.col("rule_id") == rule).sort("date")
            dates_r = rd["date"].cast(pl.String).to_list()
            cnts_r = rd["n"].to_list()
            x_idx = [all_dates.index(d) for d in dates_r]
            ax_e.plot(x_idx, cnts_r, color=rule_colors[idx], linewidth=1.8, marker="o", markersize=4, label=f"Rule {rule}")
            ax_e.fill_between(x_idx, cnts_r, alpha=0.07, color=rule_colors[idx])

        if spike_start in all_dates:
            x0 = all_dates.index(spike_start)
            x1 = all_dates.index(min(spike_end, all_dates[-1]))
            ax_e.axvspan(x0, x1, color=FirewallConfig.COLORS["danger"], alpha=0.08)
            ax_e.axvline(x0, color=FirewallConfig.COLORS["danger"], linestyle=":")
            ax_e.axvline(x1, color=FirewallConfig.COLORS["danger"], linestyle=":")

        step = max(1, len(all_dates)//14)
        ax_e.set_xticks(range(0, len(all_dates), step)); ax_e.set_xticklabels([all_dates[i] for i in range(0, len(all_dates), step)], rotation=30, ha="right", fontsize=8)
        ax_e.set_title("E — Timeline Top 5 règles du pic (zone rouge = pic)", fontweight="bold")
        ax_e.legend(); ax_e.grid()

        plt.savefig("q5c_focus_pic_deny.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(" 💾 q5c_focus_pic_deny.png")

    def plot_bonus_timeline_and_heatmap(self) -> None:
        """Bonus 1 (Timeline quotidienne) & Bonus 2 (Heatmap Jour x Heure)."""
        df_blocks = self.df.filter(pl.col("protocol_clean") == "TCP").filter(pl.col("action").is_in(["DENY", "BLOCK", "DROP", "REJECT"]))
        
        # Timeline
        daily = df_blocks.group_by("date").agg(pl.len().alias("n")).sort("date")
        dates, d_cnts = [str(d) for d in daily["date"].to_list()], daily["n"].to_list()
        
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(range(len(dates)), d_cnts, alpha=0.2, color=FirewallConfig.COLORS["danger"])
        ax.plot(range(len(dates)), d_cnts, color=FirewallConfig.COLORS["danger"], marker="o", markersize=3)
        step = max(1, len(dates)//12)
        ax.set_xticks(range(0, len(dates), step)); ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=30, ha="right", fontsize=8)
        ax.set_title("Évolution quotidienne des blocages TCP", pad=12, fontweight="bold"); ax.grid()
        plt.tight_layout(); plt.savefig("timeline_blocages.png", dpi=150, bbox_inches="tight"); plt.close()
        print(" 💾 timeline_blocages.png")

        # Heatmap Jour x Heure optimisée avec Numpy
        DAYS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
        pivot_time = df_blocks.group_by(["weekday", "hour"]).agg(pl.len().alias("n"))
        
        heat2 = np.zeros((7, 24))
        for row in pivot_time.iter_rows(named=True):
            d, h, n = row["weekday"], row["hour"], row["n"]
            # En Polars, dt.weekday() -> 1 (Lundi) à 7 (Dimanche). On décale à 0-6 pour Numpy.
            if 1 <= d <= 7 and 0 <= h < 24:
                heat2[d - 1, h] = n

        fig2, ax2 = plt.subplots(figsize=(14, 5))
        cmap_t = LinearSegmentedColormap.from_list("t", [FirewallConfig.COLORS["bg"], "#1f3a5f", FirewallConfig.COLORS["danger"]])
        im2 = ax2.imshow(heat2, aspect="auto", cmap=cmap_t)
        ax2.set_xticks(range(0, 24, 2)); ax2.set_xticklabels(range(0, 24, 2), fontsize=8)
        ax2.set_yticks(range(7)); ax2.set_yticklabels(DAYS, fontsize=9)
        ax2.set_title("Heatmap blocages : Jour de semaine × Heure UTC", pad=12, fontweight="bold")
        ax2.set_xlabel("Heure (UTC)")
        plt.colorbar(im2, ax=ax2, label="Nb blocages", shrink=0.8)
        
        plt.tight_layout(); plt.savefig("heatmap_jour_heure.png", dpi=150, bbox_inches="tight"); plt.close()
        print(" 💾 heatmap_jour_heure.png")

    def run_all_analyses(self) -> None:
        """Méthode chef d'orchestre : exécute toutes les visualisations."""
        print("\n🚀 Lancement de la génération des rapports graphiques...")
        
        # 1. Appel du Top 100 dans la console
        self.print_top_100_ports()
        
        # 2. Génération des graphiques
        self.plot_top_rules_generic(None, 15, "danger", "Q1 — Top 15 règles (Tous protocoles)", "q1_top_regles.png")
        self.plot_protocol_distribution()
        self.plot_top_rules_generic("UDP", 10, "success", "Q3 — Top 10 règles (UDP)", "q3_top10_udp.png")
        self.plot_top_rules_generic("TCP", 5, "danger", "Q4 — Top 5 règles (TCP)", "q4_top5_tcp.png")
        
        self.plot_rule_port_heatmap()
        self.plot_rule_action_stacked()
        self.plot_tcp_spike_analysis()
        self.plot_bonus_timeline_and_heatmap()
        
        print("\n✅ Analyse complète terminée. Tous les exports PNG sont générés.")

    def print_top_100_ports(self) -> None:
        """Calcule et affiche dans la console les 100 ports de destination les plus ciblés."""
        print("\n🔍 TOP 100 DES PORTS DE DESTINATION LES PLUS SOLLICITÉS :")
        print("-" * 60)
        
        # 1. Calcul vectorisé ultra-rapide avec Polars (bien indenté)
        top_100 = (
            self.df.filter(pl.col("dst_port").is_not_null())
            .group_by("dst_port")
            .agg(pl.len().alias("hits"))
            .sort("hits", descending=True)
            .head(100)
        )
        
        # 2. Astuce de Lead Data : Forcer Polars à afficher les 100 lignes
        # Par défaut, Polars tronque l'affichage console à 8 lignes pour ne pas spammer.
        # Le context manager "pl.Config" modifie ce comportement temporairement.
        with pl.Config(tbl_rows=100):
            print(top_100)

if __name__ == "__main__":
    try:
        analyzer = FirewallLogAnalyzer("log_brute.log")
        analyzer.load_and_preprocess()
        analyzer.run_all_analyses()
    except Exception as e:
        print(f"❌ Erreur critique lors de l'exécution : {e}")