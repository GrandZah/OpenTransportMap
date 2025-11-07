import json
import logging
import webbrowser
import geopandas as gpd
from shapely.geometry import box
import osmnx as ox
from pathlib import Path
from osmnx._errors import InsufficientResponseError
from typing import Optional, Dict, Tuple
import shapely

from load_configs import CONFIG_PATHS

logger = logging.getLogger(__name__)

def get_bbox_with_buf(area_id: int, buf: float = 0) -> Dict[str, float]:
    """
     Autodetects the type of the OSM element (relation, way or node) by the prefix:
     R<id> → relation
     W<id> → way
     N<id> → node
     and returns its bbox with the buf - in degrees (lon/lat).

     Return dict with boundary in lon/lan = EPSG:4326
    """
    cache_file = CONFIG_PATHS.data_cache_dir / "osm_bbox_cache.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    bounds: Optional[Tuple[float, ...]] = cache.get(str(area_id))

    if bounds is None:
        for prefix in ("R", "W", "N"):
            try:
                gdf = ox.geocoder.geocode_to_gdf(f"{prefix}{area_id}", by_osmid=True)
                bounds = tuple(map(float, gdf.total_bounds))
                cache[str(area_id)] = list(bounds)
                cache_file.write_text(json.dumps(cache, separators=(",", ":")))
                break
            except InsufficientResponseError:
                continue
        else:
            logger.exception(f"Without in cache and not found by osmnx for area_id: {area_id}")
            raise RuntimeError(f"Without in cache and not found by osmnx for area_id: {area_id}")

    minx, miny, maxx, maxy = bounds
    return {"west": minx - buf, "south": miny - buf, "east": maxx + buf, "north": maxy + buf}


def get_bbox_gdf_with_buf(area_id: int, buf: float = 0) -> gpd.GeoDataFrame:
    """
    Return a GeoDataFrame with the bounding box geometry of a city area (EPSG:4326).

    buf - in degrees (lon/lat)
    """
    bbox_map = get_bbox_with_buf(area_id, buf=buf)
    tuple_coordinate = (bbox_map['west'], bbox_map['south'], bbox_map['east'], bbox_map['north'])
    bbox_gdf = gpd.GeoDataFrame({'geometry': [box(*tuple_coordinate)]}, crs=4326)
    return bbox_gdf


def get_local_projection_by_area_id(area_id: int) -> str:
    """Try to estimate local projection by object in OSM with id-area_id"""
    bbox_gdf = get_bbox_gdf_with_buf(area_id)
    local_crs = bbox_gdf.to_crs(4326).estimate_utm_crs()
    local_projection = local_crs.to_string()
    return local_projection


def expand_gdf_bounds_in_meters(gdf_obj: gpd.GeoDataFrame, buffer_m: float = 0.0) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with bounding box expanded by a buffer (in meters) and reprojected to EPSG:4326."""
    g = gdf_obj.to_crs(gdf_obj.estimate_utm_crs())

    minx, miny, maxx, maxy = g.total_bounds
    if buffer_m:
        minx -= buffer_m
        miny -= buffer_m
        maxx += buffer_m
        maxy += buffer_m

    bbox = shapely.geometry.box(minx, miny, maxx, maxy)
    return gpd.GeoDataFrame(geometry=[bbox], crs=g.crs).to_crs(4326)


def expand_gdf_bounds_in_degrees(bbox_gdf, buf: float = 0):
    """Return a GeoDataFrame with bounding box expanded by a buffer (in degrees) and reprojected to EPSG:4326."""
    bounds = tuple(map(float, bbox_gdf.total_bounds))
    minx, miny, maxx, maxy = bounds
    tuple_coordinate = (minx - buf, miny - buf, maxx + buf, maxy + buf)
    bbox_gdf = gpd.GeoDataFrame({'geometry': [box(*tuple_coordinate)]}, crs=4326)
    return bbox_gdf


def generate_gallery(save_dir: Path) -> None:
    """
     Generates a simple HTML gallery for all image files in save_dir
     and opens it in the browser.
     Supported formats: SVG, PNG, JPG, JPEG, GIF, WEBP.
    """
    supported_exts = {".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp"}
    files = sorted(f for f in save_dir.iterdir() if f.suffix.lower() in supported_exts)

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <title>Rendered Gallery</title>",
        "  <style>",
        "    body { font-family: sans-serif; margin: 20px; }",
        "    .item { display: inline-block; margin: 10px; text-align: center; }",
        "    img, object { max-width: 200px; height: auto; border: 1px solid #ccc; }",
        "  </style>",
        "</head>",
        "<body>",
        f"<h1>Gallery: {save_dir.name}</h1>"
    ]

    for img_path in files:
        ext = img_path.suffix.lower()
        if ext == ".svg":
            html_parts.append(
                f"<div class='item'><object data='{img_path.name}' type='image/svg+xml'></object><br><small>{img_path.stem}</small></div>"
            )
        else:
            html_parts.append(
                f"<div class='item'><img src='{img_path.name}' alt='{img_path.stem}'><br><small>{img_path.stem}</small></div>"
            )

    html_parts.append("</body></html>")

    index_file = save_dir / "index.html"
    index_file.write_text("\n".join(html_parts), encoding="utf-8")
    webbrowser.open(index_file.as_uri())
