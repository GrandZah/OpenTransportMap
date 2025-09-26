import json, requests
from collections.abc import Callable
from pathlib import Path
import logging
from typing import Dict
import geopandas as gpd

from transport_posters.load_configs import OVERPASS_URL, CONFIG_PATHS

logger = logging.getLogger(__name__)


def download_osm(bbox: Dict[str, float], query_fn: Callable[[Dict[str, float]], str], cache_file: Path) -> Dict:
    """
    Get data from OSM and saving in cache_file.

    :param bbox: Dict contains bounds in lon/lat
    :param query_fn: Function, which contains request from OSM
    :param cache_file: Save response from OSM in this file
    :return: data from response
    """
    if cache_file.exists():
        logger.info("Reading OSM from cache %s", cache_file)
        return json.loads(cache_file.read_text())
    logger.info("Requesting OSM data from Overpass …")
    resp = requests.post(OVERPASS_URL, data={"data": query_fn(bbox)})
    resp.raise_for_status()
    data = resp.json()
    cache_file.write_text(json.dumps(data))
    logger.info("Saved OSM JSON to cache")
    return data


def download_to_cache(stops_gdf: gpd.GeoDataFrame, platforms_gdf: gpd.GeoDataFrame, routes_gdf: gpd.GeoDataFrame,
                      edges_gdf: gpd.GeoDataFrame, area_id: int) -> None:
    """
    Saves a GeoDataFrame for stops, routes, and edges in Parquet format.
    """
    out_dir = Path(CONFIG_PATHS.data_processed_dir / f"{area_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    stops_gdf.to_parquet(out_dir / f"stops.parquet", index=False)
    platforms_gdf.to_parquet(out_dir / f"platforms.parquet", index=False)
    routes_gdf.to_parquet(out_dir / f"routes.parquet", index=False)
    edges_gdf.to_parquet(out_dir / f"edges.parquet", index=False)

    stops_gdf.to_file(out_dir / f"stops.geojson", driver="GeoJSON")
    platforms_gdf.to_file(out_dir / f"platforms.geojson", driver="GeoJSON")
    routes_gdf.to_file(out_dir / f"routes.geojson", driver="GeoJSON")
    edges_gdf.to_file(out_dir / f"edges.geojson", driver="GeoJSON")
