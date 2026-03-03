"""
Page 3 — Détection d'anomalies par Machine Learning
=====================================================
Utilise le module ``detection_anomaly`` (CAH + Isolation Forest / LOF)
et propose trois modes :
  • Isolation Forest  (choix manuel)
  • Local Outlier Factor  (choix manuel)
  • Automatique  (Mistral décide via les métriques topologiques)
"""

import io
from datetime import timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.app.utils import get_db_client
from src.detection_anomaly.detection_anomaly import (
    CAHAnalyzer,
    SecurityOrchestrator,
)

# ──────────────────────────────────────────────────────────────
# Config Streamlit
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Machine Learning — Détection d'anomalies", layout="wide")
st.title("🤖 Détection d'anomalies dans les logs réseau")
st.markdown(
    "Analyse topologique (CAH Ward) puis détection d'anomalies via **Isolation Forest** "
    "ou **Local Outlier Factor**. Le mode *Automatique* laisse **Mistral** choisir "
    "l'algorithme le plus adapté aux propriétés statistiques du jeu de données."
)

# ──────────────────────────────────────────────────────────────
# Contrôles en haut de page
# ──────────────────────────────────────────────────────────────

# Récupérer le nombre total de logs en BDD pour borner le slider
_db_client = get_db_client()
try:
    _MAX_LOGS_DB = _db_client.count_all_logs(table_name="FW")
except Exception:
    _MAX_LOGS_DB = 10000

ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.5, 2])

with ctrl_col1:
    TABLE_NAME = st.selectbox("Table de logs", ["FW"], index=0)
    _default_limit = min(5000, _MAX_LOGS_DB)
    LIMIT = st.slider(
        "Nombre de logs à analyser (ML)",
        min_value=500,
        max_value=max(500, _MAX_LOGS_DB),
        value=_default_limit,
        step=500,
        help=f"Taille de l'échantillon. La BDD contient **{_MAX_LOGS_DB:,}** logs.",
    )

with ctrl_col2:
    MODEL_CHOICE = st.selectbox(
        "Modèle de détection",
        options=[
            "🤖 Automatique (Mistral)",
            "🌲 Isolation Forest",
            "📍 Local Outlier Factor",
        ],
        index=0,
    )
    GENERATE_REPORT = st.checkbox(
        "📝 Générer le rapport expert SOC (Mistral)",
        value=False,
        help="Appelle Mistral pour produire une synthèse SOC et des recommandations.",
    )

IF_CONTAMINATION = None
LOF_CONTAMINATION = 0.05
LOF_NEIGHBORS = 20

with ctrl_col3:
    if "Isolation Forest" in MODEL_CHOICE:
        st.markdown("**⚙️ Hyperparamètres — Isolation Forest**")
        IF_CONTAMINATION = st.select_slider(
            "Contamination",
            options=["auto", 0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
            value="auto",
            help="Proportion estimée d'anomalies. 'auto' laisse sklearn décider.",
        )
    elif "Local Outlier" in MODEL_CHOICE:
        st.markdown("**⚙️ Hyperparamètres — Local Outlier Factor**")
        LOF_NEIGHBORS = st.slider("n_neighbors", 5, 50, 20, step=5)
        LOF_CONTAMINATION = st.select_slider(
            "Contamination",
            options=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
            value=0.05,
        )
    else:
        st.info(
            "Mode automatique : Mistral choisira l'algorithme et ses paramètres par défaut."
        )

RUN_BTN = st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)

# Persister l'état de l'analyse dans session_state pour éviter
# qu'un changement de widget (granularité, slider…) ne perde les résultats.
if RUN_BTN:
    st.session_state["ml_run_requested"] = True
    # Invalider les résultats précédents pour forcer un recalcul
    for _k in [
        "ml_df_raw",
        "ml_scores",
        "ml_score_values",
        "ml_analyzer",
        "ml_metrics",
        "ml_algo_label",
        "ml_algo_reason",
        "ml_model_display",
        "ml_n_anomalies",
        "ml_n_normal",
        "ml_fraud_rate",
        "ml_model_choice",
        "ml_table",
        "ml_limit",
    ]:
        st.session_state.pop(_k, None)

st.divider()


# ──────────────────────────────────────────────────────────────
# Helpers : génération des graphiques
# ──────────────────────────────────────────────────────────────


def _dendrogram_to_figure(analyzer: CAHAnalyzer, cut: float) -> plt.Figure:
    """Renvoie un objet Figure matplotlib du dendrogramme (pas de fichier disque)."""
    Z = analyzer._get_linkage()
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    dendrogram(
        Z,
        truncate_mode="lastp",
        p=30,
        show_leaf_counts=True,
        color_threshold=cut,
        leaf_rotation=45.0,
        leaf_font_size=9.0,
        ax=ax,
    )

    ax.axhline(
        y=cut,
        c="crimson",
        linestyle="--",
        linewidth=2,
        label=f"Seuil clusters (d={cut:.1f})",
    )
    ax.set_title(
        "Dendrogramme CAH — Structure des logs réseau",
        fontsize=14,
        pad=15,
        fontweight="bold",
        color="white",
    )
    ax.set_xlabel(
        "(n) = nb logs regroupés | chiffre seul = log isolé",
        fontsize=10,
        color="lightgray",
    )
    ax.set_ylabel("Distance de fusion (Ward)", fontsize=10, color="lightgray")
    ax.tick_params(colors="lightgray")
    ax.legend(
        loc="upper right",
        fontsize=9,
        facecolor="#1e1e1e",
        edgecolor="gray",
        labelcolor="white",
    )
    fig.tight_layout()
    return fig


def _build_anomaly_plots(
    df: pd.DataFrame, scores: np.ndarray, score_values: np.ndarray | None
) -> dict[str, go.Figure]:
    """Construit un dict de figures Plotly pour les différentes vues des anomalies."""
    df = df.copy()
    df["anomaly"] = np.where(scores == -1, "Anomalie", "Normal")
    figs: dict[str, go.Figure] = {}

    color_map = {"Normal": "#2ecc71", "Anomalie": "#e74c3c"}

    # 1. Répartition Normal / Anomalie
    counts = df["anomaly"].value_counts().reset_index()
    counts.columns = ["label", "count"]
    figs["repartition"] = px.pie(
        counts,
        names="label",
        values="count",
        color="label",
        color_discrete_map=color_map,
        title="Répartition Normal / Anomalie",
        hole=0.4,
    )

    # 2. Anomalies par protocole (si colonne proto existe)
    if "proto" in df.columns:
        proto_anom = df.groupby(["proto", "anomaly"]).size().reset_index(name="count")
        figs["proto"] = px.bar(
            proto_anom,
            x="proto",
            y="count",
            color="anomaly",
            color_discrete_map=color_map,
            barmode="group",
            title="Anomalies par protocole",
        )

    # 3. Anomalies par action (permit / deny)
    if "action" in df.columns:
        action_anom = df.groupby(["action", "anomaly"]).size().reset_index(name="count")
        figs["action"] = px.bar(
            action_anom,
            x="action",
            y="count",
            color="anomaly",
            color_discrete_map=color_map,
            barmode="group",
            title="Anomalies par action firewall",
        )

    # 4. Série temporelle des anomalies
    if "datetime" in df.columns:
        df_time = df.copy()
        df_time["datetime"] = pd.to_datetime(df_time["datetime"], errors="coerce")
        df_time = df_time.dropna(subset=["datetime"])
        if not df_time.empty:
            df_time["hour"] = df_time["datetime"].dt.floor("h")
            ts = df_time.groupby(["hour", "anomaly"]).size().reset_index(name="count")
            figs["timeline"] = px.area(
                ts,
                x="hour",
                y="count",
                color="anomaly",
                color_discrete_map=color_map,
                title="Distribution temporelle des anomalies",
            )

    # 5. Top 15 IPs sources les plus anomaliques
    if "ipsrc" in df.columns:
        src_anom = (
            df[df["anomaly"] == "Anomalie"]["ipsrc"]
            .value_counts()
            .head(15)
            .reset_index()
        )
        src_anom.columns = ["ipsrc", "count"]
        if not src_anom.empty:
            figs["top_src"] = px.bar(
                src_anom,
                x="count",
                y="ipsrc",
                orientation="h",
                title="Top 15 IPs sources flaggées comme anomalies",
                color_discrete_sequence=["#e74c3c"],
            )
            figs["top_src"].update_layout(yaxis={"categoryorder": "total ascending"})

    # 6. Top 15 IPs destination les plus anomaliques
    if "ipdst" in df.columns:
        dst_anom = (
            df[df["anomaly"] == "Anomalie"]["ipdst"]
            .value_counts()
            .head(15)
            .reset_index()
        )
        dst_anom.columns = ["ipdst", "count"]
        if not dst_anom.empty:
            figs["top_dst"] = px.bar(
                dst_anom,
                x="count",
                y="ipdst",
                orientation="h",
                title="Top 15 IPs destination dans les anomalies",
                color_discrete_sequence=["#e67e22"],
            )
            figs["top_dst"].update_layout(yaxis={"categoryorder": "total ascending"})

    # 7. Ports destination ciblés par les anomalies (treemap)
    if "dstport" in df.columns:
        port_anom = (
            df[df["anomaly"] == "Anomalie"]["dstport"]
            .value_counts()
            .head(20)
            .reset_index()
        )
        port_anom.columns = ["dstport", "count"]
        if not port_anom.empty:
            port_anom["dstport"] = port_anom["dstport"].astype(str)
            port_anom["label"] = "Port " + port_anom["dstport"] + " (" + port_anom["count"].astype(str) + ")"
            figs["ports"] = px.treemap(
                port_anom,
                path=["label"],
                values="count",
                title="Top 20 ports destination dans les flux anomaliques",
                color="count",
                color_continuous_scale=["#d2b4de", "#8e44ad", "#4a235a"],
            )
            figs["ports"].update_traces(textinfo="label+value")
            figs["ports"].update_layout(coloraxis_showscale=False)

    # 8. Scatter : srcport vs dstport coloré par anomalie
    if "srcport" in df.columns and "dstport" in df.columns:
        df_scatter = df.copy()
        df_scatter["srcport"] = pd.to_numeric(df_scatter["srcport"], errors="coerce")
        df_scatter["dstport"] = pd.to_numeric(df_scatter["dstport"], errors="coerce")
        df_scatter = df_scatter.dropna(subset=["srcport", "dstport"])
        # Sous-échantillonner pour performance Plotly
        if len(df_scatter) > 3000:
            df_scatter = df_scatter.sample(3000, random_state=42)
        if not df_scatter.empty:
            figs["scatter_ports"] = px.scatter(
                df_scatter,
                x="srcport",
                y="dstport",
                color="anomaly",
                color_discrete_map=color_map,
                opacity=0.5,
                title="Port src vs Port dst (coloré par anomalie)",
                hover_data=[
                    c
                    for c in ["ipsrc", "ipdst", "proto", "action"]
                    if c in df_scatter.columns
                ],
            )

    return figs


# ──────────────────────────────────────────────────────────────
# Exécution principale
# ──────────────────────────────────────────────────────────────

if not st.session_state.get("ml_run_requested", False):
    st.info(
        "Configurez les paramètres ci-dessus puis cliquez sur **Lancer l'analyse**."
    )
    st.stop()

# --- Chargement des données (cache en session_state) ---
if "ml_df_raw" not in st.session_state:
    with st.spinner("📡 Récupération des logs depuis la base de données…"):
        db = get_db_client()
        _df = db.fetch_logs(table_name=TABLE_NAME, limit=LIMIT)
    if _df.empty:
        st.error("Aucun log récupéré. Vérifiez la connexion à MariaDB.")
        st.stop()
    st.session_state["ml_df_raw"] = _df
    st.session_state["ml_table"] = TABLE_NAME
    st.session_state["ml_limit"] = LIMIT
    st.session_state["ml_model_choice"] = MODEL_CHOICE

df_raw = st.session_state["ml_df_raw"]

st.success(
    f"**{len(df_raw):,}** logs chargés depuis `{st.session_state.get('ml_table', TABLE_NAME)}`."
)

# --- Analyse CAH (cache en session_state) ---
if "ml_analyzer" not in st.session_state:
    with st.spinner("🔬 Analyse topologique (CAH Ward) en cours…"):
        _analyzer = CAHAnalyzer(df_raw)
        st.session_state["ml_analyzer"] = _analyzer
        st.session_state["ml_metrics"] = _analyzer.get_metrics()

analyzer = st.session_state["ml_analyzer"]
metrics = st.session_state["ml_metrics"]

# ────── Affichage des métriques CAH ──────
st.header("📐 Métriques topologiques (CAH)")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Logs analysés", f"{metrics.n_samples:,}")
m2.metric("Corr. Cophénétique", f"{metrics.cophenetic_corr:.4f}")
m3.metric("Dist. fusion max", f"{metrics.max_fusion_dist:.1f}")
m4.metric("Singletons", f"{metrics.singleton_count}")
m5.metric("Kurtosis global", f"{metrics.global_kurtosis:.2f}")
m6.metric("Hétérog. densité", f"{metrics.density_heterogeneity:.4f}")

with st.expander("� Guide de lecture des métriques CAH"):
    st.markdown("""
| Métrique | Signification | Comment l'interpréter |
|----------|--------------|----------------------|
| **Logs analysés** | Taille de l'échantillon soumis à l'analyse | Plus l'échantillon est grand, plus les patterns rares sont détectables. |
| **Corr. Cophénétique** | Fidélité du dendrogramme par rapport aux distances réelles (0 → 1) | **> 0.75** = bonne structure hiérarchique ; **< 0.75** = les clusters sont flous, LOF sera plus adapté. |
| **Dist. fusion max** | Distance de Ward à laquelle les deux derniers groupes fusionnent | Plus elle est **élevée**, plus il existe des groupes très dissemblables (outliers potentiels). |
| **Singletons** | Nombre de logs qui restent isolés dans l'arbre de fusion | Beaucoup de singletons (> 10 %) → les données contiennent de nombreux comportements uniques (scans, IPs rares). |
| **Kurtosis global** | Mesure de l'épaisseur des queues de distribution | **> 3** = queues lourdes (valeurs extrêmes fréquentes, IF adapté) ; **≤ 3** = distribution plus homogène. |
| **Hétérog. densité** | Coefficient de variation des distances intra-cluster | Valeur **élevée** = clusters de densités très différentes → anomalies globales bien séparées. |
""")

with st.expander("�🔎 Entropie par feature"):
    ent_df = pd.DataFrame(
        list(metrics.feature_entropy.items()),
        columns=["Feature", "Entropie"],
    ).sort_values("Entropie", ascending=False)
    st.dataframe(ent_df, hide_index=True, use_container_width=True)

# ────── Dendrogramme ──────
st.header("🌿 Dendrogramme CAH")
Z = analyzer._get_linkage()
max_d = float(Z[-1, 2])
cut = 0.7 * max_d

fig_dendro = _dendrogram_to_figure(analyzer, cut)
st.pyplot(fig_dendro)
plt.close(fig_dendro)

with st.expander("💡 Guide de lecture du dendrogramme"):
    st.markdown("""
- **Axe Y** : distance de fusion (Ward euclidienne). Plus la jonction est haute, plus les groupes fusionnés sont dissemblables.
- **Ligne rouge pointillée** : seuil de coupure à 70 % de la distance max. Les branches en dessous forment les clusters principaux.
- **Branches isolées très hautes** : potentiels *outliers* globaux (singletons).
- **(n)** sur l'axe X : nombre de logs regroupés dans cette feuille.
""")

# ────── Décision du modèle ──────
st.header("🧠 Choix du modèle de détection")

algo_label = None
algo_reason = None

if "Automatique" in MODEL_CHOICE:
    # Cache la décision Mistral pour ne pas rappeler l'API à chaque rerun
    if "ml_algo_label" not in st.session_state:
        with st.spinner("🤖 Mistral analyse les métriques pour choisir l'algorithme…"):
            try:
                orchestrator = SecurityOrchestrator(model_name="mistral-medium-latest")
                decision = orchestrator._decide_algorithm(metrics)
                st.session_state["ml_algo_label"] = decision.algorithm
                st.session_state["ml_algo_reason"] = decision.reason
            except Exception as e:
                st.warning(
                    f"Erreur lors de l'appel Mistral : {e}. Fallback sur Isolation Forest."
                )
                st.session_state["ml_algo_label"] = "IF"
                st.session_state["ml_algo_reason"] = (
                    "Fallback automatique — erreur Mistral"
                )

    algo_label = st.session_state["ml_algo_label"]
    algo_reason = st.session_state["ml_algo_reason"]

    chosen_name = "Isolation Forest" if algo_label == "IF" else "Local Outlier Factor"
    st.info(f"**Mistral a choisi : {chosen_name}**")

    with st.expander("💡 Pourquoi Mistral a choisi cet algorithme ?", expanded=False):
        st.markdown(f"""
**Algorithme retenu** : {chosen_name}

**Justification de Mistral :**
> {algo_reason}

**Rappel des heuristiques transmises au LLM :**

| Condition | Algorithme | Raison |
|-----------|-----------|--------|
| Corrélation cophénétique < 0.75 | LOF | Hiérarchie floue → structure locale |
| Singletons > 10% des logs | LOF | Isolats locaux nombreux |
| Hétérogénéité de densité élevée | IF | Anomalies globales bien séparées |
| Kurtosis global > 3 | IF | Distribution à queues lourdes |

**Valeurs observées sur vos données :**
- Corrélation cophénétique : **{metrics.cophenetic_corr:.4f}** {"⚠️ < 0.75" if metrics.cophenetic_corr < 0.75 else "✅ ≥ 0.75"}
- Singletons : **{metrics.singleton_count}** / {metrics.n_samples} ({metrics.singleton_count / metrics.n_samples * 100:.1f}%) {"⚠️ > 10%" if metrics.singleton_count / metrics.n_samples > 0.10 else "✅ ≤ 10%"}
- Kurtosis global : **{metrics.global_kurtosis:.4f}** {"⚠️ > 3" if metrics.global_kurtosis > 3 else "✅ ≤ 3"}
- Hétérogénéité de densité : **{metrics.density_heterogeneity:.4f}**
""")

elif "Isolation Forest" in MODEL_CHOICE:
    algo_label = "IF"
    algo_reason = "Choix manuel de l'utilisateur."
    st.info("**Modèle sélectionné : Isolation Forest** (choix manuel)")

else:
    algo_label = "LOF"
    algo_reason = "Choix manuel de l'utilisateur."
    st.info("**Modèle sélectionné : Local Outlier Factor** (choix manuel)")

# ────── Entraînement & prédiction ──────
st.header("⚡ Détection d'anomalies")

if "ml_scores" not in st.session_state:
    with st.spinner("Entraînement du modèle en cours…"):
        if algo_label == "IF":
            contamination = IF_CONTAMINATION if IF_CONTAMINATION is not None else "auto"
            _model = IsolationForest(contamination=contamination, random_state=42)
            _model_display = f"Isolation Forest (contamination={contamination})"
        else:
            _model = LocalOutlierFactor(
                n_neighbors=LOF_NEIGHBORS, contamination=LOF_CONTAMINATION
            )
            _model_display = f"Local Outlier Factor (n_neighbors={LOF_NEIGHBORS}, contamination={LOF_CONTAMINATION})"

        _scores = _model.fit_predict(analyzer.X_scaled)

        if algo_label == "IF":
            _score_values = _model.decision_function(analyzer.X_scaled)
        else:
            _score_values = _model.negative_outlier_factor_

        st.session_state["ml_scores"] = _scores
        st.session_state["ml_score_values"] = _score_values
        st.session_state["ml_model_display"] = _model_display
        st.session_state["ml_algo_label"] = algo_label
        st.session_state["ml_algo_reason"] = algo_reason
        st.session_state["ml_n_anomalies"] = int((_scores == -1).sum())
        st.session_state["ml_n_normal"] = int((_scores == 1).sum())
        st.session_state["ml_fraud_rate"] = (
            st.session_state["ml_n_anomalies"] / len(_scores)
        ) * 100

scores = st.session_state["ml_scores"]
score_values = st.session_state["ml_score_values"]
model_display = st.session_state["ml_model_display"]
n_anomalies = st.session_state["ml_n_anomalies"]
n_normal = st.session_state["ml_n_normal"]
fraud_rate = st.session_state["ml_fraud_rate"]

st.markdown(f"**Modèle utilisé** : `{model_display}`")

k1, k2, k3 = st.columns(3)
k1.metric("Flux normaux", f"{n_normal:,}", delta=None)
k2.metric(
    "Anomalies détectées",
    f"{n_anomalies:,}",
    delta=f"{fraud_rate:.2f}%",
    delta_color="inverse",
)
k3.metric("Taux de contamination", f"{fraud_rate:.2f}%")

with st.expander("💡 Guide de lecture des KPIs de détection"):
    st.markdown("""
| KPI | Signification | Repères |
|-----|--------------|--------|
| **Flux normaux** | Nombre de logs classés comme comportement habituel par le modèle | Doit représenter la grande majorité (> 90 %). Un chiffre trop bas indique un modèle trop sensible. |
| **Anomalies détectées** | Nombre de logs jugés atypiques (score = −1) | À mettre en rapport avec le taux de contamination attendu. |
| **Taux de contamination** | Pourcentage d'anomalies dans l'échantillon | **< 1 %** = très sélectif ; **1–5 %** = zone typique en cybersécurité ; **> 10 %** = probable sur-détection. |
""")

# ────── Visualisations ──────
st.header("📊 Analyse des résultats")

with st.spinner("Génération des graphiques…"):
    plots = _build_anomaly_plots(df_raw, scores, score_values)

# Ligne 1 : Répartition + Protocoles
col1, col2 = st.columns(2)
with col1:
    if "repartition" in plots:
        st.plotly_chart(plots["repartition"], use_container_width=True)
with col2:
    if "proto" in plots:
        st.plotly_chart(plots["proto"], use_container_width=True)
    elif "action" in plots:
        st.plotly_chart(plots["action"], use_container_width=True)

# Ligne 2 : Action + Timeline
col3, col4 = st.columns(2)
with col3:
    if "action" in plots and "proto" in plots:
        st.plotly_chart(plots["action"], use_container_width=True)
with col4:
    if "timeline" in plots:
        st.plotly_chart(plots["timeline"], use_container_width=True)

# Ligne 3 : Scatter Ports
if "scatter_ports" in plots:
    st.plotly_chart(plots["scatter_ports"], use_container_width=True)

# Ligne 4 : Top IPs
col7, col8 = st.columns(2)
with col7:
    if "top_src" in plots:
        st.plotly_chart(plots["top_src"], use_container_width=True)
with col8:
    if "top_dst" in plots:
        st.plotly_chart(plots["top_dst"], use_container_width=True)

# Ligne 5 : Ports
if "ports" in plots:
    st.plotly_chart(plots["ports"], use_container_width=True)

# ────── Graphique temporel filtrable des attaques ──────
st.header("🕐 Anomalies sur une période choisie")

df_viz = df_raw.copy()
df_viz["anomaly"] = np.where(scores == -1, "Anomalie", "Normal")

_UNIT_MAP = {
    "Seconde": {
        "freq": "s",
        "step": timedelta(seconds=1),
        "fmt": "YYYY-MM-DD HH:mm:ss",
    },
    "Minute": {"freq": "min", "step": timedelta(minutes=1), "fmt": "YYYY-MM-DD HH:mm"},
    "Heure": {"freq": "h", "step": timedelta(hours=1), "fmt": "YYYY-MM-DD HH:mm"},
    "Jour": {"freq": "D", "step": timedelta(days=1), "fmt": "YYYY-MM-DD"},
}

if "datetime" in df_viz.columns:
    df_viz["datetime"] = pd.to_datetime(df_viz["datetime"], errors="coerce")
    df_viz = df_viz.dropna(subset=["datetime"])

    if not df_viz.empty:
        _min_dt = df_viz["datetime"].min().to_pydatetime().replace(microsecond=0)
        _max_dt = df_viz["datetime"].max().to_pydatetime().replace(microsecond=0)
        if _max_dt < df_viz["datetime"].max().to_pydatetime():
            _max_dt += timedelta(seconds=1)

        with st.container(border=True):
            unit_col, slider_col = st.columns([1, 4])

            with unit_col:
                time_unit = st.selectbox(
                    "Granularité",
                    options=list(_UNIT_MAP.keys()),
                    index=1,  # Minute par défaut
                    key="ml_time_unit",
                )
            unit_cfg = _UNIT_MAP[time_unit]

            with slider_col:
                st.caption(f"Plage disponible : {_min_dt} → {_max_dt}")
                if _min_dt == _max_dt:
                    st.info("Les données ne couvrent qu'un seul instant.")
                    start_time, end_time = _min_dt, _max_dt
                else:
                    start_time, end_time = st.slider(
                        "Plage temporelle",
                        min_value=_min_dt,
                        max_value=_max_dt,
                        value=(_min_dt, _max_dt),
                        step=unit_cfg["step"],
                        format=unit_cfg["fmt"],
                        key="anomaly_time_range",
                    )

            mask = (df_viz["datetime"] >= pd.Timestamp(start_time)) & (
                df_viz["datetime"] <= pd.Timestamp(end_time)
            )
            df_period = df_viz[mask]

            if not df_period.empty:
                df_period_anom = df_period[df_period["anomaly"] == "Anomalie"]
                n_period = len(df_period)
                n_period_anom = len(df_period_anom)

                pa1, pa2, pa3 = st.columns(3)
                pa1.metric("Logs sur la période", f"{n_period:,}")
                pa2.metric("Anomalies sur la période", f"{n_period_anom:,}")
                pa3.metric(
                    "Taux d'anomalie",
                    f"{n_period_anom / n_period * 100:.2f}%" if n_period else "N/A",
                )

                df_period_ts = df_period.copy()
                df_period_ts["bucket"] = df_period_ts["datetime"].dt.floor(
                    unit_cfg["freq"]
                )
                ts_agg = (
                    df_period_ts.groupby(["bucket", "anomaly"])
                    .size()
                    .reset_index(name="count")
                )
                fig_period = px.bar(
                    ts_agg,
                    x="bucket",
                    y="count",
                    color="anomaly",
                    color_discrete_map={"Normal": "#2ecc71", "Anomalie": "#e74c3c"},
                    barmode="stack",
                    title=f"Flux normaux vs anomalies — granularité : {time_unit.lower()}",
                    labels={"bucket": time_unit, "count": "Nombre de flux"},
                )
                fig_period.update_layout(
                    xaxis_title=time_unit,
                    yaxis_title="Nombre de flux",
                    legend_title="Classification",
                )
                st.plotly_chart(fig_period, use_container_width=True)

                with st.expander("💡 Aide à l'interprétation"):
                    st.markdown("""
- **Pics rouges isolés** : bursts d'activité anormale — à corréler avec des scans ou tentatives d'intrusion.
- **Bandes rouges persistantes** : activité malveillante soutenue (ex : exfiltration, brute-force lent).
- **Absence de rouge** : période calme ou modèle peu sensible (ajustez la contamination).
- Utilisez le slider pour zoomer sur un incident spécifique et changez la granularité pour affiner.
""")
            else:
                st.warning("Aucun log sur la période sélectionnée.")
else:
    st.info("Colonne `datetime` absente — graphique temporel indisponible.")

# ────── Tableau des logs anomaux ──────
st.header("🔍 Échantillon des logs anomaliques")

df_anomalies = df_raw[scores == -1].copy()

if not df_anomalies.empty:
    st.markdown(
        f"**{len(df_anomalies):,}** lignes flaggées — affichage des 100 premières :"
    )
    st.dataframe(df_anomalies.head(100), use_container_width=True, height=400)

    # Téléchargement CSV
    csv_buffer = io.StringIO()
    df_anomalies.to_csv(csv_buffer, index=False)
    st.download_button(
        "📥 Télécharger toutes les anomalies (CSV)",
        data=csv_buffer.getvalue(),
        file_name="anomalies_detectees.csv",
        mime="text/csv",
    )
else:
    st.success("Aucune anomalie détectée avec ces paramètres.")

# ────── Visualisations avancées ──────
if not df_anomalies.empty:
    st.header("🔬 Analyses avancées")

    adv_tabs = st.tabs(
        [
            "🗓️ Heatmap horaire",
            "🕸️ Graphe réseau",
            "🔀 Sankey",
            "📉 Score temporel",
        ]
    )

    # ── TAB 1 : Heatmap jour × heure ─────────────────────────────
    with adv_tabs[0]:
        if "datetime" in df_anomalies.columns:
            _hm = df_anomalies.copy()
            _hm["datetime"] = pd.to_datetime(_hm["datetime"], errors="coerce")
            _hm = _hm.dropna(subset=["datetime"])
            if not _hm.empty:
                _hm["day_of_week"] = _hm["datetime"].dt.day_name()
                _hm["hour"] = _hm["datetime"].dt.hour
                hm_pivot = (
                    _hm.groupby(["day_of_week", "hour"])
                    .size()
                    .reset_index(name="count")
                )
                day_order = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
                hm_pivot["day_of_week"] = pd.Categorical(
                    hm_pivot["day_of_week"],
                    categories=day_order,
                    ordered=True,
                )
                hm_full = hm_pivot.pivot_table(
                    index="day_of_week", columns="hour", values="count", fill_value=0
                )
                # Réordonner les jours ET forcer les 24 heures (0-23)
                hm_full = hm_full.reindex(
                    index=[d for d in day_order if d in hm_full.index],
                    columns=range(24),
                    fill_value=0,
                )
                fig_hm = px.imshow(
                    hm_full,
                    labels=dict(
                        x="Heure de la journée",
                        y="Jour de la semaine",
                        color="Nb anomalies",
                    ),
                    color_continuous_scale="Reds",
                    title="Heatmap des anomalies — Jour × Heure",
                    aspect="auto",
                )
                fig_hm.update_layout(xaxis=dict(dtick=1))
                st.plotly_chart(fig_hm, use_container_width=True)
                with st.expander("💡 Interprétation"):
                    st.markdown(
                        "Les zones **rouge foncé** signalent des créneaux récurrents d'activité anormale. "
                        "Des attaques automatisées (bots, scans) apparaissent souvent la nuit ou le weekend."
                    )
            else:
                st.info("Pas de données temporelles exploitables.")
        else:
            st.info("Colonne `datetime` absente.")

    # ── TAB 2 : Graphe réseau IP src → IP dst ────────────────────
    with adv_tabs[1]:
        if "ipsrc" in df_anomalies.columns and "ipdst" in df_anomalies.columns:
            _net = (
                df_anomalies.groupby(["ipsrc", "ipdst"])
                .size()
                .reset_index(name="flows")
                .sort_values("flows", ascending=False)
                .head(80)  # Top 80 liens pour lisibilité
            )
            if not _net.empty:
                all_ips = list(set(_net["ipsrc"].tolist() + _net["ipdst"].tolist()))
                ip_to_idx = {ip: i for i, ip in enumerate(all_ips)}

                node_labels = all_ips
                edge_x, edge_y = [], []
                # Layout circulaire simple
                import math as _math

                n_nodes = len(all_ips)
                pos = {
                    ip: (
                        _math.cos(2 * _math.pi * i / n_nodes),
                        _math.sin(2 * _math.pi * i / n_nodes),
                    )
                    for i, ip in enumerate(all_ips)
                }
                for _, row in _net.iterrows():
                    x0, y0 = pos[row["ipsrc"]]
                    x1, y1 = pos[row["ipdst"]]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

                edge_trace = go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(width=0.8, color="#888"),
                    hoverinfo="none",
                )
                # Calcul du degré pour la taille des nœuds
                degree = {ip: 0 for ip in all_ips}
                for _, row in _net.iterrows():
                    degree[row["ipsrc"]] += row["flows"]
                    degree[row["ipdst"]] += row["flows"]

                node_x = [pos[ip][0] for ip in all_ips]
                node_y = [pos[ip][1] for ip in all_ips]
                node_size = [max(8, min(40, degree[ip])) for ip in all_ips]
                node_color = [degree[ip] for ip in all_ips]
                node_text = [f"{ip}<br>Flux: {degree[ip]}" for ip in all_ips]

                node_trace = go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    marker=dict(
                        size=node_size,
                        color=node_color,
                        colorscale="YlOrRd",
                        showscale=True,
                        colorbar=dict(title="Flux"),
                    ),
                    text=[ip.split(".")[-1] for ip in all_ips],  # Dernier octet
                    textposition="top center",
                    textfont=dict(size=7),
                    hovertext=node_text,
                    hoverinfo="text",
                )
                fig_net = go.Figure(
                    data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Graphe réseau IP src → IP dst (anomalies)",
                        showlegend=False,
                        xaxis=dict(
                            showgrid=False, zeroline=False, showticklabels=False
                        ),
                        yaxis=dict(
                            showgrid=False, zeroline=False, showticklabels=False
                        ),
                        height=600,
                    ),
                )
                st.plotly_chart(fig_net, use_container_width=True)
                with st.expander("💡 Interprétation"):
                    st.markdown(
                        "Les **nœuds gros et rouges** sont des IPs impliquées dans de nombreux flux anormaux (pivots / relais). "
                        "Les clusters denses signalent des campagnes coordonnées."
                    )
            else:
                st.info("Pas assez de données pour le graphe réseau.")
        else:
            st.info("Colonnes `ipsrc` / `ipdst` absentes.")

    # ── TAB 3 : Sankey IP src → Port dst → Action ────────────────
    with adv_tabs[2]:
        if all(c in df_anomalies.columns for c in ["ipsrc", "dstport", "action"]):
            _sk = df_anomalies.copy()
            # Top 10 IPs src + top 10 ports pour lisibilité
            top_src = _sk["ipsrc"].value_counts().head(10).index.tolist()
            top_ports = _sk["dstport"].value_counts().head(10).index.tolist()
            _sk = _sk[_sk["ipsrc"].isin(top_src) & _sk["dstport"].isin(top_ports)]

            if not _sk.empty:
                _sk["dstport"] = _sk["dstport"].astype(str)
                _sk["action"] = _sk["action"].astype(str).str.strip().str.lower()

                # Nœuds : IPs, ports, actions
                ips_list = sorted(_sk["ipsrc"].unique())
                ports_list = sorted(_sk["dstport"].unique())
                actions_list = sorted(_sk["action"].unique())
                all_labels = (
                    ips_list
                    + [f"Port {p}" for p in ports_list]
                    + [a.capitalize() for a in actions_list]
                )

                label_idx = {label: i for i, label in enumerate(all_labels)}

                sources, targets, values = [], [], []
                # IP → Port
                for (ip, port), grp in _sk.groupby(["ipsrc", "dstport"]):
                    sources.append(label_idx[ip])
                    targets.append(label_idx[f"Port {port}"])
                    values.append(len(grp))
                # Port → Action
                for (port, action), grp in _sk.groupby(["dstport", "action"]):
                    targets.append(label_idx[action.capitalize()])
                    sources.append(label_idx[f"Port {port}"])
                    values.append(len(grp))

                # Couleurs des liens : teinte douce selon la source
                _link_colors = []
                for s in sources:
                    if s < len(ips_list):
                        _link_colors.append(
                            "rgba(231,76,60,0.35)"
                        )  # IP → rouge translucide
                    else:
                        _link_colors.append(
                            "rgba(155,89,182,0.35)"
                        )  # Port → violet translucide

                # Couleurs des nœuds par catégorie
                _node_colors = (
                    ["#e74c3c"] * len(ips_list)
                    + ["#9b59b6"] * len(ports_list)
                    + [
                        "#2ecc71" if "permit" in a else "#e67e22"
                        for a in actions_list
                    ]
                )

                fig_sankey = go.Figure(
                    go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            label=all_labels,
                            color=_node_colors,
                        ),
                        link=dict(
                            source=sources,
                            target=targets,
                            value=values,
                            color=_link_colors,
                        ),
                        textfont=dict(
                            color="white",
                            size=13,
                            family="monospace",
                        ),
                    )
                )
                fig_sankey.update_layout(
                    title=dict(
                        text="Sankey : IP source → Port destination → Action (anomalies)",
                        font=dict(color="white"),
                    ),
                    font=dict(color="white", size=12),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=550,
                )
                st.plotly_chart(fig_sankey, use_container_width=True)
                with st.expander("💡 Interprétation"):
                    st.markdown(
                        "Ce diagramme montre les **chaînes d'attaque** : d'où vient le flux (IP), "
                        "quel port est ciblé, et si le firewall a autorisé ou bloqué. "
                        "Les flux épais rouges vers *Permit* sont les plus préoccupants."
                    )
            else:
                st.info("Pas assez de données pour le Sankey.")
        else:
            st.info("Colonnes `ipsrc`, `dstport`, `action` requises.")

    # ── TAB 4 : Courbe d'évolution du score dans le temps ────────
    with adv_tabs[3]:
        if "datetime" in df_raw.columns and score_values is not None:
            _sc = df_raw.copy()
            _sc["datetime"] = pd.to_datetime(_sc["datetime"], errors="coerce")
            _sc = _sc.dropna(subset=["datetime"])
            _sc["score"] = score_values[: len(_sc)]  # Align

            if not _sc.empty:
                _sc["hour"] = _sc["datetime"].dt.floor("h")
                score_ts = (
                    _sc.groupby("hour")["score"].agg(["mean", "std"]).reset_index()
                )
                score_ts.columns = ["hour", "mean_score", "std_score"]
                score_ts["std_score"] = score_ts["std_score"].fillna(0)
                score_ts["upper"] = score_ts["mean_score"] + score_ts["std_score"]
                score_ts["lower"] = score_ts["mean_score"] - score_ts["std_score"]

                fig_score = go.Figure()
                fig_score.add_trace(
                    go.Scatter(
                        x=score_ts["hour"],
                        y=score_ts["upper"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                    )
                )
                fig_score.add_trace(
                    go.Scatter(
                        x=score_ts["hour"],
                        y=score_ts["lower"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(231,76,60,0.2)",
                        name="± 1 écart-type",
                    )
                )
                fig_score.add_trace(
                    go.Scatter(
                        x=score_ts["hour"],
                        y=score_ts["mean_score"],
                        mode="lines+markers",
                        line=dict(color="#e74c3c", width=2),
                        name="Score moyen",
                    )
                )
                fig_score.update_layout(
                    title="Évolution du score d'anomalie dans le temps",
                    xaxis_title="Heure",
                    yaxis_title="Score d'anomalie (moyen)",
                    height=450,
                )
                st.plotly_chart(fig_score, use_container_width=True)
                with st.expander("💡 Interprétation"):
                    st.markdown(
                        "Un **score qui chute** (ligne descendante) indique une dégradation : les logs deviennent "
                        "plus anormaux. La **bande rouge** montre la dispersion — une bande large = comportement hétérogène."
                    )
            else:
                st.info("Pas de données temporelles exploitables.")
        else:
            st.info("Colonne `datetime` ou scores continus absents.")

# ────── Rapport expert Mistral (optionnel) ──────
if GENERATE_REPORT and not df_anomalies.empty:
    st.header("📝 Rapport d'expertise SOC (Mistral)")

    with st.spinner("🤖 Mistral rédige le rapport d'expertise…"):
        try:
            orchestrator = SecurityOrchestrator(model_name="mistral-medium-latest")

            label = "Isolation Forest" if algo_label == "IF" else "Local Outlier Factor"

            # ── Construire un contexte riche pour Mistral ──
            _rpt_proto = (
                df_anomalies["proto"].value_counts().to_dict()
                if "proto" in df_anomalies.columns
                else "N/A"
            )
            _rpt_action = (
                df_anomalies["action"].value_counts().to_dict()
                if "action" in df_anomalies.columns
                else "N/A"
            )
            _rpt_top_src = (
                df_anomalies["ipsrc"].value_counts().head(10).to_dict()
                if "ipsrc" in df_anomalies.columns
                else "N/A"
            )
            _rpt_top_dst_ip = (
                df_anomalies["ipdst"].value_counts().head(10).to_dict()
                if "ipdst" in df_anomalies.columns
                else "N/A"
            )
            _rpt_top_dstport = (
                df_anomalies["dstport"].value_counts().head(10).to_dict()
                if "dstport" in df_anomalies.columns
                else "N/A"
            )
            _rpt_top_srcport = (
                df_anomalies["srcport"].value_counts().head(10).to_dict()
                if "srcport" in df_anomalies.columns
                else "N/A"
            )

            # Distribution temporelle
            _rpt_time = "N/A"
            if "datetime" in df_anomalies.columns:
                _tmp = pd.to_datetime(
                    df_anomalies["datetime"], errors="coerce"
                ).dropna()
                if not _tmp.empty:
                    _rpt_time = (
                        f"Période : {_tmp.min()} → {_tmp.max()} | "
                        f"Heures les plus actives : {_tmp.dt.hour.value_counts().head(5).to_dict()} | "
                        f"Jours les plus actifs : {_tmp.dt.day_name().value_counts().head(5).to_dict()}"
                    )

            # Ratio permit/deny global (pas seulement anomalies)
            _rpt_action_global = (
                df_raw["action"].value_counts().to_dict()
                if "action" in df_raw.columns
                else "N/A"
            )

            # Statistiques de score d'anomalie
            _rpt_score_stats = "N/A"
            if score_values is not None:
                _sv = pd.Series(score_values)
                _rpt_score_stats = (
                    f"min={_sv.min():.4f}, Q1={_sv.quantile(0.25):.4f}, "
                    f"médiane={_sv.median():.4f}, Q3={_sv.quantile(0.75):.4f}, max={_sv.max():.4f}"
                )

            # Bytes / paquets si disponibles
            _rpt_bytes = "N/A"
            for _bcol in ["bytes", "bytessent", "bytesrecv", "sentbyte", "rcvdbyte"]:
                if _bcol in df_anomalies.columns:
                    _rpt_bytes = df_anomalies[_bcol].describe().to_dict()
                    break
            _rpt_pkts = "N/A"
            for _pcol in ["packets", "sentpkt", "rcvdpkt"]:
                if _pcol in df_anomalies.columns:
                    _rpt_pkts = df_anomalies[_pcol].describe().to_dict()
                    break

            # Entropie résumée
            _rpt_entropy = (
                {k: round(v, 3) for k, v in metrics.feature_entropy.items()}
                if metrics.feature_entropy
                else "N/A"
            )

            expert_prompt = f"""Tu es un expert SOC senior. Produis un rapport structuré en Markdown.
Commence DIRECTEMENT par la première section. Pas de préambule, pas de titre général.

**Contexte technique :**
- Table analysée : {TABLE_NAME}, {metrics.n_samples} logs au total
- Algorithme utilisé : {label} (raison : {algo_reason})
- Corrélation Cophénétique : {metrics.cophenetic_corr:.4f}
- Singletons : {metrics.singleton_count} / {metrics.n_samples}
- Kurtosis global : {metrics.global_kurtosis:.4f}
- Hétérogénéité de densité : {metrics.density_heterogeneity:.4f}
- Entropie par feature : {_rpt_entropy}
- Anomalies détectées : {n_anomalies} / {len(df_raw)} ({fraud_rate:.2f}%)
- Distribution des scores d'anomalie : {_rpt_score_stats}

**Répartition globale (tous les logs) :**
- Actions firewall (tous logs) : {_rpt_action_global}

**Détails des anomalies :**
- Protocoles : {_rpt_proto}
- Actions firewall (anomalies) : {_rpt_action}
- Top 10 IPs sources : {_rpt_top_src}
- Top 10 IPs destinations : {_rpt_top_dst_ip}
- Top 10 ports destination : {_rpt_top_dstport}
- Top 10 ports source : {_rpt_top_srcport}
- Distribution temporelle des anomalies : {_rpt_time}
- Statistiques bytes/trafic : {_rpt_bytes}
- Statistiques paquets : {_rpt_pkts}

**Structure OBLIGATOIRE (respecte exactement ces 5 sections dans cet ordre) :**

## 1. Choix du modèle
Pourquoi {label} a été retenu pour ce jeu de données ? Explique en t'appuyant sur les métriques topologiques (corrélation cophénétique, kurtosis, hétérogénéité de densité, singletons). Compare brièvement avec l'alternative ({("LOF" if algo_label == "IF" else "IF")}) et justifie pourquoi elle était moins adaptée ici.

## 2. Résultats du modèle
Analyse les résultats de la détection : taux d'anomalie, distribution temporelle, répartition permit/deny dans les anomalies vs le global, patterns observés. Sois factuel et appuie-toi sur les chiffres fournis.

## 3. Recommandations — IPs suspectes
Liste et commente les IPs sources et destinations les plus suspectes. Pour chaque IP (ou groupe d'IPs), indique : le volume de flux anormaux, le comportement observé (scan, brute-force, exfiltration, C2, etc.), et l'action recommandée (blocage, surveillance, investigation).

## 4. Recommandations — Ports et Protocoles
Identifie les ports destination et protocoles les plus représentés dans les anomalies. Explique leur signification (services connus, ports exotiques), le risque associé, et les mesures à prendre (règles firewall, segmentation, etc.).

## 5. Pistes d'analyse supplémentaire
Propose des axes d'investigation complémentaires : corrélations à vérifier, logs supplémentaires à collecter, features à ajouter au modèle, ajustements d'hyperparamètres, sources de threat intelligence à croiser, etc.
"""
            res = orchestrator.client.chat.complete(
                model=orchestrator.model,
                messages=[{"role": "user", "content": expert_prompt}],
            )
            report_md = res.choices[0].message.content.strip()

            # Nettoyage défensif
            report_md = report_md.replace("```markdown", "").replace("```", "").strip()
            lines = report_md.splitlines()
            start_idx = next(
                (i for i, line in enumerate(lines) if line.strip().startswith("## ")),
                0,
            )
            report_md = "\n".join(lines[start_idx:])

            with st.container(border=True):
                st.markdown(report_md)

                st.download_button(
                    "📥 Télécharger le rapport (Markdown)",
                    data=report_md,
                    file_name="rapport_soc_anomalies.md",
                    mime="text/markdown",
                )

        except Exception as e:
            st.error(f"Erreur lors de la génération du rapport : {e}")

elif GENERATE_REPORT and df_anomalies.empty:
    st.info("Aucune anomalie détectée — le rapport expert n'est pas pertinent.")

# ────── Footer ──────
st.divider()
st.caption(
    "Pipeline : CAH Ward → Métriques topologiques → "
    f"{'Mistral (décision automatique)' if 'Automatique' in MODEL_CHOICE else 'Choix manuel'} → "
    f"{'Isolation Forest' if algo_label == 'IF' else 'Local Outlier Factor'} → Analyse des résultats"
)
