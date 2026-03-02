import os
import re

import pandas as pd
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
        """Utile pour débugger et voir quelles tables existent réellement."""
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
        query += " LIMIT :limit"

        safe_params = dict(params or {})
        safe_params["limit"] = limit
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
