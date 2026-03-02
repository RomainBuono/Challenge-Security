"""Service de cartographie pour créer des cartes sur n'importe quelle métrique."""

from typing import Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class map_service:
    """Service de création de cartes (points, bulles, choroplèthes)."""

    def create_points_map(
        self,
        df: pd.DataFrame,
        lat_col: str,
        lon_col: str,
        metric_col: str | None = None,
        hover_name: str | None = None,
        title: str = "Carte des points",
    ) -> go.Figure:
        """Carte de points géographiques avec taille/couleur optionnelle selon une métrique."""
        self._validate_columns(df, [lat_col, lon_col], "create_points_map")

        clean_df = df.dropna(subset=[lat_col, lon_col]).copy()
        if clean_df.empty:
            return self._empty_map("Aucune donnée géographique disponible")

        fig = px.scatter_geo(
            clean_df,
            lat=lat_col,
            lon=lon_col,
            color=metric_col if metric_col and metric_col in clean_df.columns else None,
            size=metric_col if metric_col and metric_col in clean_df.columns else None,
            hover_name=hover_name if hover_name in clean_df.columns else None,
            projection="natural earth",
            title=title,
        )
        fig.update_layout(height=650)
        return fig

    def create_metric_bubble_map(
        self,
        df: pd.DataFrame,
        lat_col: str,
        lon_col: str,
        metric_col: str,
        hover_cols: list[str] | None = None,
        title: str = "Carte bulle par métrique",
    ) -> go.Figure:
        """Carte bulle pour visualiser une métrique quantitative sur des points."""
        self._validate_columns(
            df, [lat_col, lon_col, metric_col], "create_metric_bubble_map"
        )

        clean_df = df.dropna(subset=[lat_col, lon_col, metric_col]).copy()
        if clean_df.empty:
            return self._empty_map("Aucune donnée suffisante pour la carte bulle")

        valid_hover_cols = [c for c in (hover_cols or []) if c in clean_df.columns]

        fig = px.scatter_geo(
            clean_df,
            lat=lat_col,
            lon=lon_col,
            size=metric_col,
            color=metric_col,
            hover_data=valid_hover_cols,
            projection="natural earth",
            title=title,
            size_max=32,
        )
        fig.update_layout(height=650)
        return fig

    def create_choropleth_map(
        self,
        df: pd.DataFrame,
        location_col: str,
        metric_col: str,
        location_mode: Literal[
            "country names", "ISO-3", "USA-states"
        ] = "country names",
        title: str = "Carte choroplèthe",
    ) -> go.Figure:
        """Carte choroplèthe pour des agrégations par zone (pays, codes ISO, états)."""
        self._validate_columns(df, [location_col, metric_col], "create_choropleth_map")

        clean_df = df.dropna(subset=[location_col, metric_col]).copy()
        if clean_df.empty:
            return self._empty_map("Aucune donnée agrégée disponible")

        fig = px.choropleth(
            clean_df,
            locations=location_col,
            locationmode=location_mode,
            color=metric_col,
            hover_name=location_col,
            color_continuous_scale="Reds",
            title=title,
        )
        fig.update_layout(height=650)
        return fig

    def create_map_for_metric(
        self,
        df: pd.DataFrame,
        metric_col: str,
        map_type: Literal["points", "bubble", "choropleth"],
        lat_col: str | None = None,
        lon_col: str | None = None,
        location_col: str | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Méthode générique: crée une carte pour n'importe quelle métrique."""
        if map_type in {"points", "bubble"}:
            if not lat_col or not lon_col:
                raise ValueError(
                    "lat_col et lon_col sont requis pour map_type 'points' ou 'bubble'."
                )

            if map_type == "points":
                return self.create_points_map(
                    df=df,
                    lat_col=lat_col,
                    lon_col=lon_col,
                    metric_col=metric_col,
                    title=title or f"Carte des points - {metric_col}",
                )

            return self.create_metric_bubble_map(
                df=df,
                lat_col=lat_col,
                lon_col=lon_col,
                metric_col=metric_col,
                title=title or f"Carte bulle - {metric_col}",
            )

        if map_type == "choropleth":
            if not location_col:
                raise ValueError("location_col est requis pour map_type 'choropleth'.")

            return self.create_choropleth_map(
                df=df,
                location_col=location_col,
                metric_col=metric_col,
                title=title or f"Carte choroplèthe - {metric_col}",
            )

        raise ValueError("map_type doit être 'points', 'bubble' ou 'choropleth'.")

    @staticmethod
    def to_html(fig: go.Figure) -> str:
        """Convertit une figure Plotly en HTML."""
        return fig.to_html()

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required_cols: list[str], ctx: str) -> None:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes pour {ctx}: {missing}")

    @staticmethod
    def _empty_map(message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(height=450)
        return fig
