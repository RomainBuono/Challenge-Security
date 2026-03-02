"""Service de géolocalisation d'adresses IP via ip-api.com (batch)."""

from __future__ import annotations

import ipaddress
import math
import time

import pandas as pd
import requests

# ── ip-api.com free-tier ────────────────────────────────────────────
_BATCH_URL = "http://ip-api.com/batch"
_SINGLE_URL = "http://ip-api.com/json"
_BATCH_SIZE = 100  # maximum par requête batch
_RATE_PAUSE = 1.5  # pause entre batches (15 req/min ≈ 1 req/s)

# Champs retournés par l'API
_FIELDS = "status,message,query,lat,lon,country,regionName,city,isp"


class GeoService:
    """Géolocalise des IPs via ip-api.com avec cache en mémoire."""

    def __init__(self) -> None:
        self._cache: dict[str, dict | None] = {}

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def is_private_ip(ip: str) -> bool:
        """Retourne True si l'adresse est privée / réservée / non-routable."""
        try:
            return ipaddress.ip_address(ip).is_private
        except ValueError:
            return True  # adresse invalide → non géolocalisable

    # ── batch geolocation ────────────────────────────────────────────

    def geolocate_ips(self, ips: list[str]) -> dict[str, dict | None]:
        """
        Géolocalise une liste d'IPs.

        Retourne un dict  ip → {lat, lon, country, regionName, city, isp}
        ou  ip → None  si l'IP est privée / échouée.
        """
        # IPs publiques uniques non encore en cache
        to_resolve: list[str] = []
        for ip in set(ips):
            if ip in self._cache:
                continue
            if self.is_private_ip(ip):
                self._cache[ip] = None
            else:
                to_resolve.append(ip)

        if to_resolve:
            self._batch_request(to_resolve)

        return {ip: self._cache.get(ip) for ip in ips}

    def _batch_request(self, ips: list[str]) -> None:
        """Envoie les requêtes batch par paquets de 100."""
        n_batches = math.ceil(len(ips) / _BATCH_SIZE)
        for i in range(n_batches):
            chunk = ips[i * _BATCH_SIZE : (i + 1) * _BATCH_SIZE]
            payload = [{"query": ip, "fields": _FIELDS} for ip in chunk]
            try:
                resp = requests.post(_BATCH_URL, json=payload, timeout=10)
                resp.raise_for_status()
                results = resp.json()
                for entry in results:
                    ip_key = entry.get("query", "")
                    if entry.get("status") == "success":
                        self._cache[ip_key] = {
                            "lat": entry["lat"],
                            "lon": entry["lon"],
                            "country": entry.get("country", ""),
                            "region": entry.get("regionName", ""),
                            "city": entry.get("city", ""),
                            "isp": entry.get("isp", ""),
                        }
                    else:
                        self._cache[ip_key] = None
            except requests.RequestException:
                # En cas d'erreur réseau, marquer les IPs du chunk comme None
                for ip_key in chunk:
                    self._cache.setdefault(ip_key, None)

            # Pause rate-limit entre batches (sauf dernier)
            if i < n_batches - 1:
                time.sleep(_RATE_PAUSE)

    # ── DataFrame enrichment ─────────────────────────────────────────

    def enrich_dataframe(
        self,
        df: pd.DataFrame,
        ip_col: str,
    ) -> pd.DataFrame:
        """
        Ajoute les colonnes ``latitude``, ``longitude``, ``country``,
        ``region``, ``city``, ``isp`` au DataFrame à partir de *ip_col*.

        Les lignes dont l'IP n'a pas pu être résolue sont supprimées.
        """
        if ip_col not in df.columns:
            raise ValueError(f"Colonne '{ip_col}' absente du DataFrame.")

        unique_ips = df[ip_col].dropna().unique().tolist()
        geo_map = self.geolocate_ips(unique_ips)

        geo_rows: list[dict] = []
        for ip in df[ip_col]:
            info = geo_map.get(ip)
            if info is not None:
                geo_rows.append(info)
            else:
                geo_rows.append(
                    {
                        "lat": None,
                        "lon": None,
                        "country": None,
                        "region": None,
                        "city": None,
                        "isp": None,
                    }
                )

        geo_df = pd.DataFrame(geo_rows, index=df.index)
        enriched = pd.concat([df, geo_df], axis=1)
        enriched = enriched.dropna(subset=["lat", "lon"])
        return enriched
