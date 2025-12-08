import matplotlib.pyplot as plt
import logging
import os
import geopandas as gpd
import shapely
from shapely.geometry import Point

from transport_posters.render_map.render_basemap import render_basemap
from transport_posters.render_map.render_labels_for_layers import render_labels_for_layers
from transport_posters.utils.forbidden import ForbiddenCollector
from transport_posters.utils.utils_rendering import fit_bbox_to_aspect, settings_ax
from transport_posters.data_map.get_data_map import get_data_map_by_bbox_gdf, LayersMap
from transport_posters.data_map.get_layers import reproject_all
from transport_posters.logger import log_function_call
from transport_posters.load_configs import CONFIG_PATHS, CONFIG_RENDER, PAPER_SIZES_INCH
from transport_posters.utils.utils_generate import get_local_projection_by_area_id


logger = logging.getLogger(__name__)
LOCAL_MAP_RADIUS = 900
COORDS_CENTRE = (59.220501, 39.891523)
FIGSIZE_DETAILED = [40, 40]

@log_function_call(enable_timing=True)
def all_detailed_city(args):
    save_dir = CONFIG_PATHS.output_composed_img_dir / f"city_{args.area_id}"
    save_dir.mkdir(parents=True, exist_ok=True)

    local_projection = get_local_projection_by_area_id(args.area_id)
    detailed_out_path = os.path.join(save_dir, f"detailed_map_ALL_CITY_{COORDS_CENTRE}_{args.area_id}_2.png")

    local_point = _centre_to_local_point(COORDS_CENTRE, local_projection)
    stop_bbox_gdf = _get_bbox_gdf_from_point(local_point, local_projection, LOCAL_MAP_RADIUS)
    if args.render_map:
        detailed_layers = get_data_map_by_bbox_gdf(args.area_id, stop_bbox_gdf.to_crs(4326), CONFIG_RENDER["detailed_layers_name_v_all"])
        detailed_layers = reproject_all(detailed_layers, local_projection)
    else:
        detailed_layers = None

    render_detailed_map(args, detailed_layers, stop_bbox_gdf, detailed_out_path, figsize_poster=FIGSIZE_DETAILED)

@log_function_call
def render_detailed_map(args, layers: LayersMap, bbox_gdf: gpd.GeoDataFrame, out_path: str, figsize_poster=None):
    """Render a detailed stop map with routes, stops and walking overlay."""
    figsize = figsize_poster or PAPER_SIZES_INCH.get(getattr(args, "paper", "A4"), PAPER_SIZES_INCH["A4"])

    dpi = CONFIG_RENDER.get("dpi", 150)
    target_aspect = figsize[0] / figsize[1]
    bleed = CONFIG_RENDER.get("bleed", 0.0)
    fitted_bbox = fit_bbox_to_aspect(bbox_gdf, aspect=target_aspect, bleed=bleed)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    settings_ax(ax, fitted_bbox)

    forbidden = ForbiddenCollector()

    if args.render_map:
        render_basemap(ax, layers, fitted_bbox)
    if args.render_map:
        render_labels_for_layers(ax, layers, fitted_bbox, forbidden=forbidden)

    fig.savefig(out_path, pad_inches=0)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _get_bbox_gdf_from_point(point, local_proj, buffer_m=0):
    """Create bbox GeoDataFrame around a point in local CRS."""
    minx = maxx = point.x
    miny = maxy = point.y

    if buffer_m:
        minx -= buffer_m
        miny -= buffer_m
        maxx += buffer_m
        maxy += buffer_m

    bbox = shapely.geometry.box(minx, miny, maxx, maxy)
    return gpd.GeoDataFrame(geometry=[bbox], crs=local_proj)

def _centre_to_local_point(coords_centre, local_proj, src_crs="EPSG:4326"):
    """Convert (lat, lon) coordinates to a shapely Point in the target CRS."""
    lat, lon = coords_centre
    gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs=src_crs)
    gdf_local = gdf.to_crs(local_proj)
    return gdf_local.geometry.iloc[0]