import os
import re

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()


class MariaDBClient:
    def __init__(self):
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.host = os.getenv("DB_HOST")
        self.port = os.getenv("DB_PORT", "3306")
        self.database = os.getenv("DB_NAME")

        # Utilisation de pymysql pour la compatibilité MariaDB/MySQL [cite: 9]
        connection_string = f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        self.engine = create_engine(connection_string, pool_pre_ping=True)

    @staticmethod
    def _is_valid_identifier(identifier: str) -> bool:
        return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", identifier or ""))

    def _validate_table(self, table_name: str) -> None:
        if not self._is_valid_identifier(table_name):
            raise ValueError(f"Nom de table invalide: {table_name}")

        if table_name not in self.list_tables():
            raise ValueError(
                f"La table '{table_name}' est introuvable dans '{self.database}'."
            )

    def _validate_columns(self, table_name: str, columns: list[str]) -> None:
        inspector = inspect(self.engine)
        existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
        invalid = [col for col in columns if col not in existing_columns]
        if invalid:
            raise ValueError(
                f"Colonnes invalides dans '{table_name}': {invalid}. Colonnes disponibles: {sorted(existing_columns)}"
            )

    def list_tables(self):
        """Liste les tables disponibles pour le débuggage."""
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def list_columns(self, table_name: str) -> list[str]:
        """Retourne les colonnes d'une table."""
        self._validate_table(table_name)
        inspector = inspect(self.engine)
        return [col["name"] for col in inspector.get_columns(table_name)]

    def execute_query(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        """Exécute une requête SQL paramétrée et retourne un DataFrame."""
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(text(sql), conn, params=params or {})
        except SQLAlchemyError as exc:
            raise ValueError(
                f"Erreur SQL lors de l'exécution de la requête: {exc}"
            ) from exc

    def fetch_table(
        self,
        table_name: str,
        columns: list[str] | None = None,
        where_clause: str | None = None,
        params: dict | None = None,
        order_by: str | None = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        """Récupère une table avec filtres optionnels et limite."""
        self._validate_table(table_name)

        selected_cols = columns or ["*"]
        if selected_cols != ["*"]:
            for col in selected_cols:
                if not self._is_valid_identifier(col):
                    raise ValueError(f"Nom de colonne invalide: {col}")
            self._validate_columns(table_name, selected_cols)

        if order_by and not self._is_valid_identifier(order_by):
            raise ValueError(f"order_by invalide: {order_by}")

        col_sql = ", ".join(selected_cols)
        query = f"SELECT {col_sql} FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        if order_by:
            query += f" ORDER BY {order_by} DESC"
        query += f" LIMIT {int(limit)}"

        safe_params = dict(params or {})
        return self.execute_query(query, safe_params)

    def fetch_logs(
        self, table_name: str = "Logs_fw", limit: int = 5000
    ) -> pd.DataFrame:
        """Récupère les logs en respectant les colonnes du projet[cite: 80, 87]."""
        self._validate_table(table_name)
        columns = self.list_columns(table_name)
        order_by = "datetime" if "datetime" in columns else columns[0]
        return self.fetch_table(table_name=table_name, order_by=order_by, limit=limit)

    def fetch_metric_by_location(
        self,
        table_name: str,
        location_column: str,
        metric_column: str,
        agg: str = "SUM",
        where_clause: str | None = None,
        params: dict | None = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        """
        Prépare des données agrégées pour les cartes (ex: pays -> nb événements).
        """
        self._validate_table(table_name)
        self._validate_columns(table_name, [location_column, metric_column])

        agg_upper = agg.upper()
        allowed_aggs = {"SUM", "COUNT", "AVG", "MIN", "MAX"}
        if agg_upper not in allowed_aggs:
            raise ValueError(
                f"Aggregation non supportée: {agg}. Choix: {sorted(allowed_aggs)}"
            )

        query = (
            f"SELECT {location_column} AS location, "
            f"{agg_upper}({metric_column}) AS metric "
            f"FROM {table_name}"
        )
        if where_clause:
            query += f" WHERE {where_clause}"
        query += " GROUP BY location ORDER BY metric DESC LIMIT :limit"

        safe_params = dict(params or {})
        safe_params["limit"] = limit
        return self.execute_query(query, safe_params)

    def fetch_points_for_map(
        self,
        table_name: str,
        lat_column: str = "latitude",
        lon_column: str = "longitude",
        metric_column: str | None = None,
        where_clause: str | None = None,
        params: dict | None = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        """
        Retourne des points exploitables par map_service (lat/lon + métrique optionnelle).
        """
        self._validate_table(table_name)

        cols = [lat_column, lon_column]
        if metric_column:
            cols.append(metric_column)
        self._validate_columns(table_name, cols)

        selected_cols = cols
        query = f"SELECT {', '.join(selected_cols)} FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        query += (
            f" AND {lat_column} IS NOT NULL AND {lon_column} IS NOT NULL"
            if where_clause
            else f" WHERE {lat_column} IS NOT NULL AND {lon_column} IS NOT NULL"
        )
        query += " LIMIT :limit"

        safe_params = dict(params or {})
        safe_params["limit"] = limit
        return self.execute_query(query, safe_params)

    # --- MÉTHODES D'EXPLORATION (Utilisées par la page Exploration DB) ---

    def fetch_logs(self, table_name: str = "FW", limit: int = 5000) -> pd.DataFrame:
        """Récupère un échantillon de logs bruts pour l'affichage tableau."""
        query = text(f"SELECT * FROM {table_name} ORDER BY datetime DESC LIMIT :limit")
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params={"limit": limit})

    def count_all_logs(self, table_name: str = "FW") -> int:
        """Compte le volume total sans charger les lignes."""
        query = text(f"SELECT COUNT(*) FROM {table_name}")
        with self.engine.connect() as conn:
            return conn.execute(query).scalar()

    # --- MÉTHODES DE SERVICE (Utilisées par le Dashboard) ---

    def get_last_sync_date(self, table_name: str = "FW"):
        """Récupère la date du log le plus récent."""
        query = text(f"SELECT MAX(datetime) FROM {table_name}")
        with self.engine.connect() as conn:
            return conn.execute(query).scalar()

    @st.cache_data(ttl=600)  # Cache de 10 minutes
    def get_security_ratios(_self) -> dict:
        """Calcule les indicateurs Permit/Deny directement en SQL."""
        query = text("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN LOWER(action) = 'permit' THEN 1 ELSE 0 END) as accepted,
                SUM(CASE WHEN LOWER(action) = 'deny' THEN 1 ELSE 0 END) as rejected
            FROM FW
        """)
        # CORRECTION ICI : _self au lieu de self
        with _self.engine.connect() as conn:
            res = conn.execute(query).mappings().first()
            total = res["total"] or 0
            acc = res["accepted"] or 0
            rej = res["rejected"] or 0
            ratio = (acc / total * 100) if total > 0 else 0
            return {"total": total, "accepted": acc, "rejected": rej, "ratio": ratio}

    def get_protocol_distribution(self) -> pd.DataFrame:
        """Agrégation SQL par protocole."""
        query = text("SELECT proto, COUNT(*) as count FROM FW GROUP BY proto")
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn)

    @st.cache_data(ttl=3600)
    def get_top_rules(_self, proto: str, limit: int = 10) -> pd.DataFrame:
        """Classement des règles (policyid) par protocole."""
        query = text("""
            SELECT policyid, COUNT(*) as count 
            FROM FW 
            WHERE UPPER(proto) = :proto 
            GROUP BY policyid 
            ORDER BY count DESC 
            LIMIT :limit
        """)
        # CORRECTION ICI : _self au lieu de self
        with _self.engine.connect() as conn:
            return pd.read_sql(
                query, conn, params={"proto": proto.upper(), "limit": limit}
            )

    @st.cache_data(ttl=3600)
    def get_rfc6056_distribution(_self) -> pd.DataFrame:
        query = text("""
            SELECT 
                UPPER(proto) AS proto, 
                LOWER(action) AS action, 
                CASE 
                    WHEN dstport BETWEEN 0 AND 1023 THEN 'System Ports'
                    WHEN dstport BETWEEN 1024 AND 49151 THEN 'User Ports'
                    WHEN dstport BETWEEN 49152 AND 65535 THEN 'Dynamic Ports'
                    ELSE 'Other'
                END AS port_category,
                COUNT(*) AS count
            FROM FW
            GROUP BY UPPER(proto), LOWER(action), port_category
        """)
        with _self.engine.connect() as conn:
            return pd.read_sql(query, conn)

    @st.cache_data(ttl=60)  # Reduced TTL to 1 minute for active testing
    def get_vulnerable_ports_stats(_self) -> pd.DataFrame:
        """Fetch permit/deny ratios for specific vulnerable ports."""
        # Enforce string types for strict matching
        target_ports = ("21", "22", "23", "80", "110", "445", "3306", "3389", "8080")

        query = text(f"""
            SELECT 
                TRIM(dstport) AS dstport, 
                TRIM(LOWER(action)) AS action, 
                COUNT(*) AS count
            FROM FW
            WHERE TRIM(dstport) IN {target_ports}
            GROUP BY TRIM(dstport), TRIM(LOWER(action))
        """)
        with _self.engine.connect() as conn:
            return pd.read_sql(query, conn)

    @st.cache_data(ttl=3600)
    def get_top_source_ips(_self, limit: int = 5) -> pd.DataFrame:
        """Req 4: TOP 5 IPs sources les plus émettrices."""
        query = text("""
            SELECT ipsrc, COUNT(*) as flow_count 
            FROM FW 
            GROUP BY ipsrc 
            ORDER BY flow_count DESC 
            LIMIT :limit
        """)
        with _self.engine.connect() as conn:
            return pd.read_sql(query, conn, params={"limit": limit})

    @st.cache_data(ttl=3600)
    def get_top_system_ports_permitted(_self, limit: int = 10) -> pd.DataFrame:
        """Req 4: TOP 10 des ports inférieurs à 1024 avec un accès autorisé."""
        query = text("""
            SELECT dstport, COUNT(*) as count 
            FROM FW 
            WHERE dstport <= 1024 AND LOWER(action) = 'permit' 
            GROUP BY dstport 
            ORDER BY count DESC 
            LIMIT :limit
        """)
        with _self.engine.connect() as conn:
            return pd.read_sql(query, conn, params={"limit": limit})

    @st.cache_data(ttl=3600)
    def get_external_ip_accesses(_self) -> pd.DataFrame:
        """Req 4: Lister les accès des adresses non inclues dans 159.84.x.x."""
        query = text("""
            SELECT datetime, ipsrc, ipdst, dstport, LOWER(action) as action 
            FROM FW 
            WHERE ipsrc NOT LIKE '159.84.%'
            ORDER BY datetime DESC
            LIMIT 1000
        """)
        with _self.engine.connect() as conn:
            return pd.read_sql(query, conn)

    @st.cache_data(ttl=60)
    def get_vue1_data(
        _self, rule_id: int = None, port_min: int = 0, port_max: int = 65535
    ) -> pd.DataFrame:
        """
        Retrieves aggregated flow data for Vue 1, supporting temporal distribution,
        protocol grouping, and rule/port filtering.
        """
        base_query = """
            SELECT 
                DATE_FORMAT(datetime, '%Y-%m-%d %H:00:00') AS time_window,
                UPPER(proto) AS proto,
                LOWER(action) AS action,
                COUNT(*) AS count
            FROM FW
            WHERE dstport BETWEEN :port_min AND :port_max
        """
        params = {"port_min": port_min, "port_max": port_max}

        if rule_id is not None:
            base_query += " AND policyid = :rule_id"
            params["rule_id"] = rule_id

        base_query += (
            " GROUP BY time_window, UPPER(proto), LOWER(action) ORDER BY time_window"
        )

        with _self.engine.connect() as conn:
            df = pd.read_sql(text(base_query), conn, params=params)
            if not df.empty:
                df["time_window"] = pd.to_datetime(df["time_window"])
            return df

    @st.cache_data(ttl=300)
    def get_vue3_scatter_data(_self, limit: int = 2000) -> pd.DataFrame:
        """
        Agrège les données par IP source pour la détection de scanners.
        """
        query = text("""
            SELECT 
                ipsrc,
                COUNT(DISTINCT ipdst) AS dest_count,
                COUNT(*) AS total_flows,
                SUM(CASE WHEN LOWER(TRIM(action)) = 'permit' THEN 1 ELSE 0 END) AS permit_count,
                SUM(CASE WHEN LOWER(TRIM(action)) = 'deny' THEN 1 ELSE 0 END) AS deny_count
            FROM FW
            GROUP BY ipsrc
            ORDER BY total_flows DESC
            LIMIT :limit
        """)

        with _self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"limit": limit})
            if not df.empty:
                # Calcul du ratio pour l'échelle de couleur
                df["deny_ratio"] = (df["deny_count"] / df["total_flows"]) * 100
            return df

    @st.cache_data(ttl=300)
    def get_port_scan_data(_self, limit: int = 1000) -> pd.DataFrame:
        """
        Agrège les données par IP source pour détecter le balayage de ports (Port Scanning).
        """
        query = text("""
            SELECT 
                ipsrc,
                COUNT(DISTINCT dstport) AS distinct_ports,
                COUNT(*) AS total_flows,
                SUM(CASE WHEN LOWER(TRIM(action)) = 'permit' THEN 1 ELSE 0 END) AS permit_count,
                SUM(CASE WHEN LOWER(TRIM(action)) = 'deny' THEN 1 ELSE 0 END) AS deny_count
            FROM FW
            GROUP BY ipsrc
            ORDER BY distinct_ports DESC
            LIMIT :limit
        """)

        with _self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"limit": limit})
            if not df.empty:
                df["deny_ratio"] = (df["deny_count"] / df["total_flows"]) * 100
            return df
