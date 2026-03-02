"""Service de cartographie pour créer des cartes sur n'importe quelle métrique."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class map_service:
    """Service de création de cartes (points, bulles, choroplèthes)."""

    _SIZE_MAX = 32
    _FLAT_COLOR = "steelblue"

    def create_points_map(
        self,
        df: pd.DataFrame,
        lat_col: str = "lat",
        lon_col: str = "lon",
        metric_col: str | None = None,
        hover_name: str | None = None,
        title: str = "Carte des points",
        log_scale: bool = False,
    ) -> go.Figure:
        """Carte de points géographiques avec taille/couleur optionnelle selon une métrique."""
        self._validate_columns(df, [lat_col, lon_col], "create_points_map")

        clean_df = df.dropna(subset=[lat_col, lon_col]).copy()
        if clean_df.empty:
            return self._empty_map("Aucune donnée géographique disponible")

        size_col = self._apply_log_scale(clean_df, metric_col, log_scale)

        fig = px.scatter_geo(
            clean_df,
            lat=lat_col,
            lon=lon_col,
            color=metric_col if metric_col and metric_col in clean_df.columns else None,
            size=size_col,
            hover_name=hover_name
            if hover_name and hover_name in clean_df.columns
            else None,
            projection="natural earth",
            title=title,
        )
        fig.update_layout(height=650)
        return fig

    def create_metric_bubble_map(
        self,
        df: pd.DataFrame,
        lat_col: str = "lat",
        lon_col: str = "lon",
        metric_col: str | None = None,
        color_col: str | None = None,
        hover_cols: list[str] | None = None,
        title: str = "Carte bulle par métrique",
        log_scale: bool = False,
        color_continuous_scale: str = "YlOrRd",
    ) -> go.Figure:
        """Carte bulle pour visualiser une métrique quantitative sur des points."""
        self._validate_columns(df, [lat_col, lon_col], "create_metric_bubble_map")

        clean_df = df.dropna(subset=[lat_col, lon_col]).copy()
        if clean_df.empty:
            return self._empty_map("Aucune donnée suffisante pour la carte bulle")

        # Taille optionnelle
        size_col = None
        if metric_col and metric_col in clean_df.columns:
            clean_df = clean_df.dropna(subset=[metric_col])
            size_col = self._apply_log_scale(clean_df, metric_col, log_scale)

        valid_hover_cols = [c for c in (hover_cols or []) if c in clean_df.columns]
        hover_data_dict = {c: True for c in valid_hover_cols}

        # Masquer _size_log du hover quand log scale est actif
        if size_col and size_col != metric_col:
            hover_data_dict[size_col] = False

        # Couleur : uniquement si explicitement demandée
        effective_color = None
        if color_col and color_col in clean_df.columns:
            effective_color = color_col

        # color_continuous_scale only for numeric color columns
        ccs = None
        if effective_color and effective_color in clean_df.columns:
            if clean_df[effective_color].dtype.kind in "iuf":
                ccs = color_continuous_scale

        fig = px.scatter_geo(
            clean_df,
            lat=lat_col,
            lon=lon_col,
            size=size_col,
            color=effective_color,
            color_continuous_scale=ccs,
            hover_data=hover_data_dict if hover_data_dict else None,
            hover_name=valid_hover_cols[0] if valid_hover_cols else None,
            projection="natural earth",
            title=title,
            size_max=self._SIZE_MAX,
        )

        # Si pas de couleur, forcer une couleur unie
        if effective_color is None:
            for trace in fig.data:
                trace.marker.color = self._FLAT_COLOR
                trace.showlegend = False

        # Masquer les légendes Plotly (rendues en HTML à côté de la carte)
        fig.update_layout(
            height=650,
            coloraxis_showscale=False,
            showlegend=False,
        )
        return fig

    # Palettes proposées à l'utilisateur
    COLOR_SCALES = [
        "Plasma",
        "Reds",
        "YlOrRd",
        "Inferno",
        "Viridis",
        "Turbo",
        "Hot",
        "Electric",
        "Magma",
        "Cividis",
    ]

    def create_choropleth_map(
        self,
        df: pd.DataFrame,
        location_col: str,
        metric_col: str,
        location_mode: Literal[
            "country names", "ISO-3", "USA-states"
        ] = "country names",
        title: str = "Carte choroplèthe",
        color_continuous_scale: str = "YlOrRd",
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
            color_continuous_scale=color_continuous_scale,
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

    @classmethod
    def compute_size_legend(
        cls,
        df: pd.DataFrame,
        metric_col: str,
        size_col: str,
        log_scale: bool,
        n_levels: int = 4,
    ) -> list[tuple[str, float]]:
        """Renvoie une liste de (label, diamètre_px) pour la légende de taille.

        Les diamètres reproduisent la logique de ``px.scatter_geo(size_max=...)``.
        """
        vals = df[size_col].dropna()
        if vals.empty or vals.max() == 0:
            return []

        vmax = float(vals.max())
        vmin = float(vals.min())
        size_max = cls._SIZE_MAX

        if vmin == vmax:
            rep = [vmax]
        else:
            rep = list(np.linspace(vmin, vmax, n_levels))

        items: list[tuple[str, float]] = []
        for v in rep:
            diameter = max(4, size_max * float(np.sqrt(v / vmax))) if vmax > 0 else 8
            orig = float(np.expm1(v)) if log_scale else float(v)
            label = f"{int(round(orig)):,}" if orig >= 1 else f"{orig:.2f}"
            items.append((label, diameter))
        return items

    @staticmethod
    def color_scale_to_css(scale_name: str, direction: str = "to top") -> str:
        """Convertit un nom de palette Plotly en gradient CSS linéaire."""
        import plotly.colors as pc

        try:
            colors = pc.get_colorscale(scale_name)
        except Exception:
            return f"{direction}, #440154, #fde725"  # fallback Viridis

        stops = ", ".join(f"{c[1]} {c[0] * 100:.0f}%" for c in colors)
        return f"{direction}, {stops}"

    @staticmethod
    def _apply_log_scale(
        df: pd.DataFrame,
        metric_col: str | None,
        log_scale: bool,
    ) -> str | None:
        """Ajoute une colonne `_size_log` si log_scale est activé, retourne le nom de la colonne de taille."""
        if not metric_col or metric_col not in df.columns:
            return None
        if not log_scale:
            return metric_col
        # log1p pour gérer les zéros ; on normalise ensuite pour que size_max fonctionne
        df["_size_log"] = np.log1p(df[metric_col].clip(lower=0))
        return "_size_log"

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
