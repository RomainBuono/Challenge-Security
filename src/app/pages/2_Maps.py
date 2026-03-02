import pandas as pd
import streamlit as st

from src.app.services.map_service import map_service
from src.data.mariadb_client import MariaDBClient


def _find_default_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _numeric_columns(df_columns: list[str], df_dtypes: dict) -> list[str]:
    numeric = []
    for col in df_columns:
        dtype = str(df_dtypes.get(col, ""))
        if any(token in dtype for token in ["int", "float", "double", "decimal"]):
            numeric.append(col)
    return numeric


def _datetime_candidates(columns: list[str], dtypes: dict) -> list[str]:
    candidates = []
    for col in columns:
        col_lower = col.lower()
        dtype = str(dtypes.get(col, "")).lower()
        if any(
            token in col_lower for token in ["date", "time", "timestamp", "datetime"]
        ):
            candidates.append(col)
            continue
        if "datetime" in dtype or "date" in dtype:
            candidates.append(col)
    return candidates


st.set_page_config(page_title="Maps", layout="wide")
st.header("🗺️ Cartographie des logs")
st.caption("Créez une carte selon la métrique et le type de visualisation.")

db_client = MariaDBClient()
maps = map_service()

try:
    tables = db_client.list_tables()
    if not tables:
        st.warning("Aucune table disponible en base.")
        st.stop()

    default_table = "FW" if "FW" in tables else tables[0]

    col_a, col_b, col_c = st.columns([1.2, 1, 1])
    with col_a:
        selected_table = st.selectbox(
            "Table", options=tables, index=tables.index(default_table)
        )
    with col_b:
        map_type = st.selectbox(
            "Type de carte",
            options=["points", "bubble", "choropleth"],
            index=1,
        )
    with col_c:
        row_limit = st.slider(
            "Nombre de lignes", min_value=200, max_value=20000, value=5000, step=200
        )

    table_columns = db_client.list_columns(selected_table)
    preview_df = db_client.fetch_table(
        table_name=selected_table, columns=table_columns, limit=min(row_limit, 1500)
    )

    numeric_cols = _numeric_columns(
        table_columns, preview_df.dtypes.to_dict() if not preview_df.empty else {}
    )
    if not numeric_cols:
        st.warning("Aucune colonne numérique détectée pour la métrique.")
        st.stop()

    metric_col = st.selectbox("Métrique sur la carte", options=numeric_cols, index=0)

    # Filtre temporel (bornes récupérées depuis la base)
    datetime_cols = _datetime_candidates(
        table_columns, preview_df.dtypes.to_dict() if not preview_df.empty else {}
    )
    selected_time_col = None
    time_where_clause = None
    time_params = {}

    if datetime_cols:
        default_time_col = (
            _find_default_column(
                datetime_cols, ["datetime", "timestamp", "date", "event_time"]
            )
            or datetime_cols[0]
        )

        selected_time_col = st.selectbox(
            "Colonne temporelle (plage)",
            options=datetime_cols,
            index=datetime_cols.index(default_time_col),
        )

        bounds_query = (
            f"SELECT MIN({selected_time_col}) AS min_bound_val, "
            f"MAX({selected_time_col}) AS max_bound_val "
            f"FROM {selected_table} WHERE {selected_time_col} IS NOT NULL"
        )
        bounds_df = db_client.execute_query(bounds_query)

        min_available = None
        max_available = None
        if not bounds_df.empty:
            min_raw = bounds_df.iloc[0]["min_bound_val"]
            max_raw = bounds_df.iloc[0]["max_bound_val"]
            if min_raw is not None and max_raw is not None:
                parsed_min = pd.to_datetime(min_raw, errors="coerce")
                parsed_max = pd.to_datetime(max_raw, errors="coerce")
                if not pd.isna(parsed_min) and not pd.isna(parsed_max):
                    min_available = parsed_min
                    max_available = parsed_max

        if min_available is not None and max_available is not None:
            st.caption(f"Plage disponible en base: {min_available} → {max_available}")

            if min_available == max_available:
                st.info(
                    "La base ne contient qu'un seul instant pour cette colonne temporelle."
                )
                start_time, end_time = min_available, max_available
            else:
                start_time, end_time = st.slider(
                    "Plage temporelle",
                    min_value=min_available.to_pydatetime(),
                    max_value=max_available.to_pydatetime(),
                    value=(
                        min_available.to_pydatetime(),
                        max_available.to_pydatetime(),
                    ),
                    format="YYYY-MM-DD HH:mm:ss",
                )

            time_where_clause = f"{selected_time_col} BETWEEN :start_time AND :end_time"
            time_params = {
                "start_time": pd.to_datetime(start_time),
                "end_time": pd.to_datetime(end_time),
            }

    if map_type in {"points", "bubble"}:
        lat_default = _find_default_column(
            table_columns, ["latitude", "lat", "src_lat", "dst_lat"]
        )
        lon_default = _find_default_column(
            table_columns, ["longitude", "lon", "lng", "src_lon", "dst_lon"]
        )

        geo_col_1, geo_col_2, geo_col_3 = st.columns(3)
        with geo_col_1:
            lat_col = st.selectbox(
                "Colonne latitude",
                options=table_columns,
                index=table_columns.index(lat_default)
                if lat_default in table_columns
                else 0,
            )
        with geo_col_2:
            lon_col = st.selectbox(
                "Colonne longitude",
                options=table_columns,
                index=table_columns.index(lon_default)
                if lon_default in table_columns
                else min(1, len(table_columns) - 1),
            )
        with geo_col_3:
            size_metric = st.selectbox(
                "Taille des points (optionnel)",
                options=["(auto)"] + numeric_cols,
                index=0,
            )

        required = [lat_col, lon_col, metric_col]
        if size_metric != "(auto)" and size_metric not in required:
            required.append(size_metric)

        df_map = db_client.fetch_table(
            table_name=selected_table,
            columns=required,
            where_clause=time_where_clause,
            params=time_params,
            limit=row_limit,
        )

        if map_type == "points":
            fig = maps.create_points_map(
                df=df_map,
                lat_col=lat_col,
                lon_col=lon_col,
                metric_col=metric_col,
                title=f"Carte points - {metric_col}",
            )
        else:
            selected_metric_for_size = (
                metric_col if size_metric == "(auto)" else size_metric
            )
            fig = maps.create_metric_bubble_map(
                df=df_map,
                lat_col=lat_col,
                lon_col=lon_col,
                metric_col=selected_metric_for_size,
                hover_cols=[metric_col],
                title=f"Carte bulles - taille={selected_metric_for_size}",
            )

    else:
        location_default = _find_default_column(
            table_columns, ["country", "pays", "country_code", "iso3"]
        )
        location_col = st.selectbox(
            "Colonne de zone (pays/code)",
            options=table_columns,
            index=table_columns.index(location_default)
            if location_default in table_columns
            else 0,
        )
        agg_mode = st.selectbox(
            "Agrégation", options=["SUM", "COUNT", "AVG", "MAX", "MIN"], index=0
        )

        agg_df = db_client.fetch_metric_by_location(
            table_name=selected_table,
            location_column=location_col,
            metric_column=metric_col,
            agg=agg_mode,
            where_clause=time_where_clause,
            params=time_params,
            limit=500,
        )

        fig = maps.create_choropleth_map(
            df=agg_df,
            location_col="location",
            metric_col="metric",
            location_mode="country names",
            title=f"Choroplèthe - {agg_mode}({metric_col})",
        )

    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Aperçu des données utilisées"):
        if map_type == "choropleth":
            st.dataframe(agg_df.head(100), use_container_width=True)
        else:
            st.dataframe(df_map.head(100), use_container_width=True)

except (ValueError, KeyError, TypeError) as e:
    st.error(f"Erreur lors du rendu de la carte: {e}")
