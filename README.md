# 🛡️ SISE-Challenge-Security

> **Plateforme d'analyse cybersécurité en temps réel**  
> Détection d'anomalies par Machine Learning, géolocalisation d'IPs et monitoring de logs firewall

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Vue d'ensemble

**SISE-Challenge-Security** est un projet issu d'un challenge sprint d'une journée et demi dont le but est une solution complète d'analyse de logs de sécurité réseau. Notre réalisation combine visualisation interactive, machine learning et intelligence artificielle pour fournir aux équipes SOC (Security Operations Center) des outils avancés de détection d'anomalies et de monitoring de menaces.

### 🎯 Fonctionnalités principales

- **📊 Dashboard interactif** : Monitoring temps réel des flux réseau avec KPIs de sécurité
- **🤖 Détection d'anomalies IA** : Algorithmes ML (Isolation Forest, LOF) avec sélection automatique par Mistral AI
- **🗺️ Géolocalisation d'IPs** : Cartographie des sources d'attaque avec enrichissement géographique
- **📋 Exploration de données** : Interface intuitive pour analyser les logs avec exports CSV
- **📝 Rapports SOC** : Génération automatique de rapports d'expertise via Mistral AI

---

## 🏗️ Architecture technique

```
├── 🐍 Backend Python
│   ├── MariaDB/MySQL (logs réseau)
│   ├── SQLAlchemy (ORM)
│   └── Scikit-learn (ML)
├── 🌐 Frontend Web
│   ├── Streamlit (interface)
│   ├── Plotly (graphiques)
│   └── Folium (cartes)
├── 🤖 Intelligence Artificielle
│   ├── Mistral AI (rapports)
│   └── Algorithmes ML (anomalies)
└── 🐳 Déploiement
    ├── Docker + Docker Compose
    └── UV (gestionnaire de dépendances)
```

### Stack technologique

- **Backend** : Python 3.13+, pandas, scikit-learn, MariaDB
- **Frontend** : Streamlit, Plotly, Folium, matplotlib  
- **Database** : MariaDB/MySQL avec PyMySQL
- **ML/AI** : Isolation Forest, Local Outlier Factor, Mistral AI
- **Geolocation** : API ip-api.com
- **Containerization** : Docker, UV package manager

---

## 🚀 Installation et démarrage

### Prérequis

- Python 3.13+
- MariaDB server d'où proviennent les données
- Clé API Mistral AI (optionnelle mais vivement recommandé, pour les rapports automatiques)

### 1. Clonage et dépendances

```bash
git clone https://github.com/OlivierBOROT/SISE-Challenge-Security.git
cd SISE-Challenge-Security

# Installation des dépendances avec UV (recommandé)
uv sync
```

### 2. Configuration de la base de données

Créez un fichier `.env` à la racine :

```env
# Configuration MariaDB
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=security_logs

# API Mistral (optionnel)
MISTRAL_API_KEY=your_mistral_api_key
```

### 3. Lancement de l'application

```bash
# Méthode recommandée avec UV
uv run python -m streamlit run src/app/main.py

# Ou classiquement avec pip
pip install -r requirements.txt
streamlit run src/app/main.py
```

L'application sera accessible sur `http://localhost:8501`

### 4. Déploiement Docker (optionnel)

```bash
# Construction de l'image
docker build -f docker/dockerfile -t sise-security .

# Lancement du conteneur  
docker run -p 8501:8501 --env-file .env sise-security
```

---

## 📱 Utilisation

### Interface principale

1. **🏠 Dashboard** : Vue d'ensemble des métriques de sécurité
   - KPIs temps réel (flux total, autorisés, bloqués)
   - Monitoring des ports critiques (FTP, SSH, RDP, etc.)
   - Détection des scans de ports avec radar visuel

2. **📋 Exploration DB** : Analyse interactive des logs
   - Sélecteur de volume de données
   - Tableau explicatif des colonnes
   - Export CSV des données affichées

3. **🗺️ Maps** : Géolocalisation des adresses IP
   - Visualisation géographique des sources
   - Enrichissement automatique des coordonnées

4. **🤖 Machine Learning** : Détection d'anomalies avancée
   - **Mode Automatique** : Mistral AI choisit l'algorithme optimal
   - **Mode Manuel** : Isolation Forest ou Local Outlier Factor
   - Analyse topologique (Classification Ascendante Hiérarchique)
   - Analyse des anomalies trouvées
   - Génération de rapports SOC automatisés

### Workflow recommandé

1. **Monitoring** : Consultez le dashboard pour identifier les tendances
2. **Investigation** : Utilisez les cartes pour localiser les sources suspectes  
3. **Analyse ML** : Lancez la détection d'anomalies sur les périodes critiques
4. **Rapport** : Générez un rapport SOC pour documentation et actions

---

## 📚 Structure du projet

```
SISE-Challenge-Security/
├── 📁 src/
│   ├── 📁 app/                    # Application Streamlit
│   │   ├── 📄 main.py            # Page principale (dashboard)
│   │   ├── 📄 utils.py           # Utilitaires partagés
│   │   ├── 📁 pages/             # Pages de l'application
│   │   │   ├── 📄 1_Exploration_DB.py    # Exploration de données
│   │   │   ├── 📄 2_Maps.py             # Géolocalisation
│   │   │   ├── 📄 3_machine_learning.py # Détection ML
│   │   │   └── 📄 4_A_propos.py         # Informations projet
│   │   └── 📁 services/          # Services métier
│   │       ├── 📄 geo_service.py       # Géolocalisation IP
│   │       └── 📄 map_service.py       # Service cartographique
│   ├── 📁 data/                  # Couche d'accès aux données  
│   │   └── 📄 mariadb_client.py  # Client base de données
│   └── 📁 detection_anomaly/     # Module de détection ML
│       └── 📄 detection_anomaly.py     # Algorithmes et orchestration
├── 📁 dataviz/                   # Scripts d'analyse standalone
├── 📁 docker/                    # Configuration Docker
├── 📁 data/brute/               # Données brutes (logs)
├── 📄 pyproject.toml            # Configuration UV/Python
└── 📄 README.md                 # Ce fichier
```

---

## 🔬 Algorithmes de détection

### 1. Classification Ascendante Hiérarchique (CAH)
- **Objectif** : Analyse topologique des logs pour comprendre la structure des données
- **Méthode** : Ward linkage avec métriques de Cophénéticité
- **Utilisation** : Pré-analyse pour guider le choix d'algorithme ML

### 2. Isolation Forest
- **Principe** : Isolation des anomalies par partitionnement aléatoire
- **Avantage** : Efficace sur datasets volumineux, peu de faux positifs
- **Cas d'usage** : Détection d'intrusions, comportements aberrants

### 3. Local Outlier Factor (LOF)  
- **Principe** : Détection basée sur la densité locale des points
- **Avantage** : Excellent pour les anomalies contextuelles
- **Cas d'usage** : Scans de ports, comportements inhabituels

### 4. Sélection automatique (Mistral AI)
- **Intelligence** : Analyse des métriques topologiques (kurtosis, hétérogénéité)
- **Décision** : Choix optimal de l'algorithme selon les caractéristiques des données
- **Traçabilité** : Justification explicite des choix algorithmiques

## 📊 Données et exemples

### Format des logs supportés
Le système supporte les logs de firewall avec les champs suivants :

| Champ | Type | Description |
|-------|------|-------------|
| `id` | int | Identifiant unique du log |
| `datetime` | timestamp | Horodatage du flux |
| `ipsrc` | string | Adresse IP source |
| `ipdst` | string | Adresse IP destination |
| `srcport` | int | Port source |
| `dstport` | int | Port destination |
| `proto` | string | Protocole (TCP/UDP/ICMP) |
| `action` | string | Action firewall (permit/deny) |

### Exemple de jeu de données
```sql
INSERT INTO FW (datetime, ipsrc, ipdst, srcport, dstport, proto, action) VALUES 
('2024-03-01 14:30:15', '192.168.1.100', '8.8.8.8', 54231, 53, 'UDP', 'permit'),
('2024-03-01 14:30:16', '10.0.0.50', '159.84.1.10', 12345, 22, 'TCP', 'deny');
```

## 👥 Contributeurs

- BETARD Martin
- BOROT Olivier
- BUONO Romain
- REY-COQUAIS Constantin
- SLAOUI Salah Eddine

<div align="center">

[⬆️ Retour en haut](#%EF%B8%8F-sise-challenge-security)

</div>