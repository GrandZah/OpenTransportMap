import logging
from pathlib import Path
from typing import Union
import geopandas as gpd
from pyproj import CRS
import json, requests
from collections.abc import Callable
from typing import Dict

from transport_posters.data_transport.build_edges import build_edges_pairwise
from transport_posters.data_transport.сity_route_database import CityRouteDatabase
from transport_posters.data_transport.parser_osm import build_query_by_bbox, parse_osm
from transport_posters.logger import log_function_call
from transport_posters.utils.utils_generate import get_bbox_with_buf
from transport_posters.load_configs import OVERPASS_URL, CONFIG_PATHS

logger = logging.getLogger(__name__)
TRANSPORT_DEGREE_BUFFER = 0.01
CRSLike = Union[int, str, dict, CRS]


@log_function_call
def get_bus_layers(area_id: int, from_parser="OSM") -> CityRouteDatabase:
    """Downloads/reads the cache, parses it, and returns CityRouteDatabase."""
    raw_cache_path = CONFIG_PATHS.data_raw_dir / f"{area_id}_transport.json"
    bbox = get_bbox_with_buf(area_id, buf=TRANSPORT_DEGREE_BUFFER)
    data = _download_osm(bbox, build_query_by_bbox, raw_cache_path)
    stops_gdf, platforms_gdf, routes_gdf, stops_dists_map = parse_osm(data)
    edges_gdf = build_edges_pairwise(routes_gdf, stops_dists_map)
    _download_to_cache(stops_gdf, platforms_gdf, routes_gdf, edges_gdf, area_id)
    return CityRouteDatabase(stops_gdf, platforms_gdf, routes_gdf, edges_gdf)


@log_function_call
def get_from_cache_bus_layers(area_id: int, from_parser="OSM") -> CityRouteDatabase:
    """
     Loads a GeoDataFrame from the Parquet cache if the files exist. Otherwise, it loads from the source.
     All columns containing numpy.ndarray are converted to lists.
     """
    parquet_cache_paths = _cache_path(area_id, "parquet")

    if all([v.exists() for k, v in parquet_cache_paths.items()]):
        stops_gdf = gpd.read_parquet(parquet_cache_paths["stops"])
        platforms_gdf = gpd.read_parquet(parquet_cache_paths["platforms"])
        routes_gdf = gpd.read_parquet(parquet_cache_paths["routes"])
        edges_gdf = gpd.read_parquet(parquet_cache_paths["edges"])

        for gdf in (stops_gdf, platforms_gdf, routes_gdf, edges_gdf):
            for col in gdf.columns:
                sample = gdf[col].iat[0] if not gdf.empty else None
                if hasattr(sample, 'tolist'):
                    gdf[col] = gdf[col].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x)

        return CityRouteDatabase(stops_gdf, platforms_gdf, routes_gdf, edges_gdf)

    return get_bus_layers(area_id)


def _download_osm(bbox: Dict[str, float], query_fn: Callable[[Dict[str, float]], str], cache_file: Path) -> Dict:
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


def _download_to_cache(stops_gdf: gpd.GeoDataFrame, platforms_gdf: gpd.GeoDataFrame, routes_gdf: gpd.GeoDataFrame,
                       edges_gdf: gpd.GeoDataFrame, area_id: int) -> None:
    """
    Saves a GeoDataFrame for stops, routes, and edges in Parquet format.
    """
    parquet_cache_paths = _cache_path(area_id, "parquet")

    stops_gdf.to_parquet(parquet_cache_paths["stops"], index=False)
    platforms_gdf.to_parquet(parquet_cache_paths["platforms"], index=False)
    routes_gdf.to_parquet(parquet_cache_paths["routes"], index=False)
    edges_gdf.to_parquet(parquet_cache_paths["edges"], index=False)


def _cache_path(area_id: int, format: str="parquet"):
    cache_paths = {}
    out_dir = Path(CONFIG_PATHS.data_processed_dir / f"{area_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_paths["stops"] = out_dir / f"stops.{format}"
    cache_paths["platforms"] = out_dir / f"platforms.{format}"
    cache_paths["routes"] = out_dir / f"routes.{format}"
    cache_paths["edges"] = out_dir / f"edges.{format}"
    return cache_paths
