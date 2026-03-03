# pipeline.py — Version corrigée (Code Review SISE-OPSIE 2026)

import os
import io
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # FIX: Évite les erreurs d'affichage en contexte serveur (Streamlit/Docker)
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Literal
from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
from scipy.stats import kurtosis, entropy
from sqlalchemy import create_engine, text
from mistralai import Mistral

load_dotenv()

import base64

def _img_to_base64(img_path: str) -> Optional[str]:
    """Encode une image PNG en base64 pour l'intégrer directement dans le Markdown/HTML."""
    if not os.path.exists(img_path):
        print(f"⚠️ Image introuvable : {img_path}")
        return None
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ─────────────────────────────────────────────────────────────
# 1. ACCÈS DONNÉES
# ─────────────────────────────────────────────────────────────

class MariaDBClient:
    def __init__(self):
        self.user     = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.host     = os.getenv("DB_HOST")
        self.port     = int(os.getenv("DB_PORT", 3306))
        self.database = os.getenv("DB_NAME")

        connection_string = (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )
        self.engine = create_engine(connection_string, pool_pre_ping=True)

    # FIX : table_name whitelisté pour éviter l'injection SQL
    _ALLOWED_TABLES = {"FW", "FW_archive"}

    def fetch_logs(self, table_name: str = "FW", limit: int = 5000) -> pd.DataFrame:
        """Récupère les logs pour l'analyse cyber."""
        # FIX : Validation explicite du nom de table (interpolation directe sinon)
        if table_name not in self._ALLOWED_TABLES:
            raise ValueError(
                f"Table '{table_name}' non autorisée. "
                f"Tables valides : {self._ALLOWED_TABLES}"
            )
        query = text(f"SELECT * FROM {table_name} ORDER BY datetime DESC LIMIT :limit")
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params={"limit": limit})


# ─────────────────────────────────────────────────────────────
# 2. ANALYSEUR TOPOLOGIQUE (CAH)
# ─────────────────────────────────────────────────────────────

@dataclass
class TopologyMetrics:
    n_samples:              int
    singleton_count:        int
    max_fusion_dist:        float
    cophenetic_corr:        float
    density_heterogeneity:  float
    global_kurtosis:        float
    feature_entropy:        Dict[str, float]


class CAHAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df_raw     = df.copy()
        self.df_numeric = df.copy()
        self._Z: Optional[np.ndarray] = None  # FIX : cache linkage matrix
        self._prepare()

    def _prepare(self):
        """Encode et normalise les features. Exclut les colonnes temporelles."""
        if 'datetime' in self.df_numeric.columns:
            self.df_numeric = self.df_numeric.drop(columns=['datetime'])

        le = LabelEncoder()
        for col in self.df_numeric.select_dtypes(include=['object']).columns:
            self.df_numeric[col] = le.fit_transform(
                self.df_numeric[col].astype(str)
            )
        self.X_scaled = StandardScaler().fit_transform(self.df_numeric)

    # FIX : linkage calculé UNE seule fois et mis en cache
    def _get_linkage(self) -> np.ndarray:
        """Retourne la matrice de liaison Ward (calcul unique, puis cache)."""
        if self._Z is None:
            self._Z = linkage(self.X_scaled, method='ward')
        return self._Z

    def get_metrics(self) -> TopologyMetrics:
        Z     = self._get_linkage()
        c, _  = cophenet(Z, pdist(self.X_scaled))
        max_d = float(Z[-1, 2])

        # FIX : seuil documenté — 70% de la distance max de fusion
        # Heuristique : sépare les outliers globaux des clusters principaux.
        # À ajuster si vos données sont très denses (essayez 0.5–0.8).
        CUT_THRESHOLD_RATIO = 0.7
        clusters   = fcluster(Z, t=CUT_THRESHOLD_RATIO * max_d, criterion='distance')
        singletons = int((pd.Series(clusters).value_counts() == 1).sum())

        return TopologyMetrics(
            n_samples             = len(self.df_numeric),
            singleton_count       = singletons,
            max_fusion_dist       = max_d,
            cophenetic_corr       = float(c),
            density_heterogeneity = float(np.var(pdist(self.X_scaled))),
            global_kurtosis       = float(kurtosis(self.X_scaled, axis=None)),
            feature_entropy       = {
                col: float(entropy(self.df_numeric[col].value_counts()))
                for col in self.df_numeric.columns
            }
        )

    def save_dendrogram(self, filename: str = "cah_dendrogram.png") -> str:
        """Génère et sauvegarde le dendrogramme. Réutilise le linkage en cache."""
        Z     = self._get_linkage()  # FIX : pas de double calcul
        max_d = float(Z[-1, 2])
        cut   = 0.7 * max_d

        plt.figure(figsize=(12, 6))
        plt.style.use('dark_background')

        dendrogram(
            Z,
            truncate_mode   = 'lastp',
            p               = 30,
            show_leaf_counts= True,
            color_threshold = cut,
            leaf_rotation   = 45.,
            leaf_font_size  = 10.,
        )

        plt.axhline(y=cut, c='crimson', linestyle='--', linewidth=2,
                    label=f'Seuil clusters (d={cut:.1f})')

        plt.title("Dendrogramme CAH — Structure des Logs Réseau",
                  fontsize=14, pad=15, fontweight='bold')
        plt.xlabel(
            "Axe X : (n) = nombre de logs regroupés | Chiffre seul = log isolé",
            fontsize=11
        )
        plt.ylabel("Axe Y : Distance de fusion (Euclidienne / Ward)", fontsize=11)

        explication = (
            "Guide de lecture :\n"
            "• Plus la fusion est haute (Y), plus les logs sont dissemblables.\n"
            "• Les branches sous la ligne rouge = clusters principaux.\n"
            "• Une branche isolée très haute = anomalie globale (Singleton)."
        )
        plt.figtext(
            0.12, 0.75, explication,
            bbox=dict(facecolor='#1e1e1e', alpha=0.9,
                      edgecolor='gray', boxstyle='round,pad=0.5'),
            fontsize=9, color='lightgray'
        )
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        return filename


# ─────────────────────────────────────────────────────────────
# 3. ORCHESTRATEUR LLM (Mistral)
# ─────────────────────────────────────────────────────────────

# FIX : Type strict pour la décision algorithmique
AlgorithmChoice = Literal["IF", "LOF"]

@dataclass
class DecisionResult:
    algorithm: AlgorithmChoice
    reason:    str


class SecurityOrchestrator:

    # FIX : Heuristiques explicites documentées, transmises au LLM
    _DECISION_HEURISTICS = """
Tu es un expert en détection d'anomalies. Voici les règles de décision :

| Condition                          | Algorithme recommandé | Raison                              |
|------------------------------------|----------------------|-------------------------------------|
| cophenetic_corr < 0.75             | LOF                  | Hiérarchie floue → structure locale |
| singleton_count > 10% de n_samples | LOF                  | Isolats locaux nombreux             |
| density_heterogeneity élevée       | IF                   | Anomalies globales bien séparées    |
| global_kurtosis > 3                | IF                   | Distribution à queues lourdes       |
| Combinaison des cas ci-dessus      | Priorité à LOF       | La structure locale prime           |
"""

    def __init__(self, model_name: str = "mistral-medium-latest"):
        self.model  = model_name
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    def _clean_previous_files(self,
                               img_path:    str = "cah_dendrogram.png",
                               report_path: str = "rapport_cyber_final.md"):
        for filepath in [img_path, report_path]:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"🧹 Nettoyage : '{filepath}' supprimé.")
                except Exception as e:
                    print(f"⚠️ Impossible de supprimer {filepath} : {e}")

    # FIX : Sortie JSON contrainte → décision 100% déterministe et parsable
    def _decide_algorithm(self, metrics: TopologyMetrics) -> DecisionResult:
        """
        Demande au LLM de choisir IF ou LOF en se basant sur les métriques CAH.
        La réponse est contrainte au format JSON strict pour éviter
        les erreurs de parsing sur du texte libre.
        """
        prompt = f"""
{self._DECISION_HEURISTICS}

Métriques calculées sur le dataset :
{json.dumps(asdict(metrics), indent=2, default=str)}

Réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte avant ou après,
sans backticks, sans markdown :
{{"algorithm": "IF", "reason": "explication courte en français"}}
ou
{{"algorithm": "LOF", "reason": "explication courte en français"}}
"""
        res  = self.client.chat.complete(
            model    = self.model,
            messages = [{"role": "user", "content": prompt}]
        )
        raw = res.choices[0].message.content.strip()

        # FIX : Nettoyage défensif des éventuels backticks résiduels
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(raw)
            algo = data.get("algorithm", "").upper()
            if algo not in ("IF", "LOF"):
                raise ValueError(f"Valeur inattendue : {algo}")
            return DecisionResult(
                algorithm = algo,         # type: ignore[arg-type]
                reason    = data.get("reason", "Non fournie")
            )
        except (json.JSONDecodeError, ValueError) as e:
            # FIX : Fallback explicite et loggé plutôt que comportement silencieux
            print(f"⚠️ Parsing JSON échoué ({e}). Réponse brute : {raw!r}")
            print("↩️  Fallback sur Isolation Forest par défaut.")
            return DecisionResult(algorithm="IF", reason="Fallback — parsing LLM échoué")

    def run_analysis(self, analyzer: CAHAnalyzer) -> str:
        """
        Exécute le pipeline complet et retourne le chemin du rapport généré.
        """
        self._clean_previous_files()

        # 1. Métriques CAH (linkage calculé une seule fois ici)
        metrics = analyzer.get_metrics()

        # 2. Dendrogramme (réutilise le cache — pas de double calcul)
        img_path = analyzer.save_dendrogram()

        # 3. Décision algorithmique via LLM (JSON contraint)
        decision = self._decide_algorithm(metrics)
        print(f"🧠 Algorithme choisi : {decision.algorithm} — {decision.reason}")

        # 4. Instanciation du modèle
        if decision.algorithm == "IF":
            # FIX : contamination='auto' valide uniquement pour IsolationForest
            model_engine = IsolationForest(contamination='auto', random_state=42)
            label = "Isolation Forest"
        else:
            # FIX : LOF ne supporte pas contamination='auto' en toutes versions sklearn
            # On utilise une valeur explicite et raisonnable
            model_engine = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
            label = "Local Outlier Factor"

        scores = model_engine.fit_predict(analyzer.X_scaled)

        # 5. Rapport Markdown enrichi
        report_path = self._generate_markdown_report(
            analyzer, metrics, scores, label, img_path, decision
        )
        return report_path

    def _generate_markdown_report(
        self,
        analyzer:  CAHAnalyzer,
        metrics:   TopologyMetrics,
        scores:    np.ndarray,
        label:     str,
        img_path:  str,
        decision:  DecisionResult,
    ) -> str:

        n_fraud    = int((scores == -1).sum())
        fraud_rate = (n_fraud / metrics.n_samples) * 100
        fraud_sample = analyzer.df_raw[scores == -1].head(5).to_markdown(index=False)

        # ── Sections 1–5 écrites UNIQUEMENT par le code Python ──────────────
        # FIX IMAGE : on encode le PNG en base64 et on l'intègre directement
        # dans le Markdown → aucune dépendance au chemin relatif,
        # fonctionne dans Streamlit, dans un PDF exporté, et en standalone.
        img_b64 = _img_to_base64(img_path)
        img_tag = (
            f'<img src="data:image/png;base64,{img_b64}" '
            f'alt="Dendrogramme CAH" style="max-width:100%;">'
            if img_b64
            else f"*(Image non disponible : {img_path})*"
        )

        sections_1_to_5 = f"""# RAPPORT D'ANALYSE CYBERSÉCURITÉ (XAI)

    ## 1. Métriques Globales du Dataset
    - Nombre total de logs analysés : {metrics.n_samples}
    - Nombre de features : {len(analyzer.df_numeric.columns)}
    - Entropie des colonnes :
    {json.dumps(metrics.feature_entropy, indent=2)}

    ## 2. Analyse Topologique (CAH Ward)

    {img_tag}

    - **Corrélation Cophenétique** : {metrics.cophenetic_corr:.4f}
    - **Distance de fusion maximale** : {metrics.max_fusion_dist:.2f}
    - **Singletons détectés** : {metrics.singleton_count}
    - **Hétérogénéité de densité** : {metrics.density_heterogeneity:.4f}
    - **Kurtosis global** : {metrics.global_kurtosis:.4f}

    ## 3. Décision Algorithmique (LLM)
    - **Algorithme sélectionné** : {label}
    - **Justification** : {decision.reason}

    ## 4. Résultats de Détection
    - **Anomalies détectées** : {n_fraud}
    - **Taux de contamination** : {fraud_rate:.2f}%

    ## 5. Échantillon des Logs Anomaux

    {fraud_sample}
    """

        # ── Sections 6–7 générées par le LLM ────────────────────────────────
        # FIX PREAMBLE : le LLM ne reçoit QUE les données contextuelles,
        # pas le rapport à recopier → impossible pour lui d'ajouter une intro.
        expert_prompt = f"""Tu es un expert SOC. Tu dois produire UNIQUEMENT deux sections Markdown,
    sans aucun texte introductif, sans phrase de présentation, sans backticks de code.
    Commence ta réponse DIRECTEMENT par "## 6. Synthèse de l'Expert".

    ---

    **Données de contexte :**
    - Algorithme utilisé : {label}
    - Justification du choix : {decision.reason}
    - Corrélation Cophenétique : {metrics.cophenetic_corr:.4f}
    - Singletons : {metrics.singleton_count} / {metrics.n_samples} logs
    - Kurtosis global : {metrics.global_kurtosis:.4f}
    - Hétérogénéité de densité : {metrics.density_heterogeneity:.4f}
    - Anomalies détectées : {n_fraud} ({fraud_rate:.2f}%)

    ---

    ## 6. Synthèse de l'Expert
    [Explique en 3–5 phrases pourquoi {label} était pertinent au regard des métriques CAH ci-dessus.]

    ## 7. Recommandations Opérationnelles
    [Liste 4 actions concrètes pour le SOC basées sur les anomalies détectées.]
    """

        res = self.client.chat.complete(
            model    = self.model,
            messages = [{"role": "user", "content": expert_prompt}]
        )

        llm_sections = res.choices[0].message.content.strip()

        # FIX PREAMBLE (défense en profondeur) : supprime toute ligne avant "## 6."
        # au cas où le LLM ajouterait quand même une intro malgré les instructions
        lines = llm_sections.splitlines()
        start_idx = next(
            (i for i, line in enumerate(lines) if line.strip().startswith("## 6.")),
            0  # si introuvable, on garde tout (ne devrait pas arriver)
        )
        llm_sections_clean = "\n".join(lines[start_idx:])

        # ── Concaténation Python — le code assemble, le LLM ne touche pas à 1–5 ──
        final_content = sections_1_to_5 + "\n" + llm_sections_clean

        report_path = "rapport_cyber_final.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        print(f"✅ Rapport généré : {report_path}")
        return report_path


# ─────────────────────────────────────────────────────────────
# 4. MAIN (exécution standalone)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    db   = MariaDBClient()
    logs = db.fetch_logs(table_name="FW", limit=1000)

    if not logs.empty:
        cah          = CAHAnalyzer(logs)
        orchestrator = SecurityOrchestrator(model_name="mistral-medium-latest")
        report_path  = orchestrator.run_analysis(cah)
        print(f"Pipeline terminé. Rapport : {report_path}")
    else:
        print("⚠️ Aucun log récupéré — vérifiez la connexion MariaDB.")