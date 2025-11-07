import logging
from pathlib import Path
import geopandas as gpd

from transport_posters.load_configs import CONFIG_PATHS
from transport_posters.logger import log_function_call
from transport_posters.utils.utils_generate import get_bbox_gdf_with_buf
from transport_posters.utils.utils_projection import ensure_epsg4326
from .get_layers import get_layers, LayersMap
from .get_style_layers import get_style_layers

logger = logging.getLogger(__name__)


@log_function_call
def get_data_map_by_area_id(area_id: int, style_layers_name: str, buf: float = 0) -> LayersMap:
    """"
    Get layers of map by area_id bound bbox and style_layers_name.

    additional argument = buf - add to boundary in lon/lat type.
    """
    bbox_gdf = get_bbox_gdf_with_buf(area_id, buf=buf)
    layers = get_data_map_by_bbox_gdf(area_id, bbox_gdf, style_layers_name)
    return layers


@log_function_call
def get_data_map_by_bbox_gdf(area_id: int, bbox_gdf: gpd.GeoDataFrame, style_layers_name: str) -> LayersMap:
    """
    Get layers of map by style_layers_name and save their style and data in LayersMap.

    bbox_gdf - should be in EPSG:4326
    """
    ensure_epsg4326(bbox_gdf)

    style_dir = CONFIG_PATHS.style_dir
    style_layers = get_style_layers(Path(f"{style_dir}/{style_layers_name}.json"))

    data_map_dir = CONFIG_PATHS.data_map_dir
    parquet_path = data_map_dir

    layers = get_layers(style_layers, parquet_path, bbox_gdf, area_id)
    return layers
