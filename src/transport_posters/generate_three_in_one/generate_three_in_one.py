import logging
import os
import geopandas as gpd
import shapely

from transport_posters.data_map.get_data_map import get_data_map_by_bbox_gdf
from transport_posters.data_map.get_layers import reproject_all
from transport_posters.data_transport.get_bus_layers import get_from_cache_bus_layers
from transport_posters.data_transport.сity_route_database import CityRouteDatabase
from transport_posters.logger import log_function_call
from transport_posters.utils.utils import slugify
from .compose_img_to_poster import compose_img_to_poster
from .render_detailed_map import render_detailed_map
from transport_posters.load_configs import CONFIG_PATHS, CONFIG_RENDER
from transport_posters.utils.utils_generate import get_bbox_gdf_with_buf, get_local_projection_by_area_id, \
    expand_gdf_bounds_in_degrees
from .render_far_plan import render_far_plan
from .render_middle_transit_map import render_middle_transit_map

logger = logging.getLogger(__name__)

TRANSIT_MAP_RADIUS = 2750
LOCAL_MAP_RADIUS = 500
DEGREES_BUF = 0.08


def get_stop_bbox_gdf(stop_row, local_proj, buffer_m=0):
    minx = maxx = stop_row.geometry.x
    miny = maxy = stop_row.geometry.y
    if buffer_m:
        minx -= buffer_m
        miny -= buffer_m
        maxx += buffer_m
        maxy += buffer_m

    bbox = shapely.geometry.box(minx, miny, maxx, maxy)
    return gpd.GeoDataFrame(geometry=[bbox], crs=local_proj)


@log_function_call(enable_timing=True)
def generate_three_in_one(args):
    save_dir = CONFIG_PATHS.output_composed_img_dir / f"city_{args.area_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    city_bbox_gdf_data = get_bbox_gdf_with_buf(args.area_id, DEGREES_BUF)

    local_projection = get_local_projection_by_area_id(args.area_id)

    ctx_map: CityRouteDatabase = get_from_cache_bus_layers(int(args.area_id))
    ctx_map = ctx_map.reproject_all(local_projection)

    stops_gdf = ctx_map.stops_gdf
    stops_gdf = _choose_stops(stops_gdf, city_bbox_gdf_data.to_crs(local_projection))
    stops_gdf_iterate = stops_gdf.head(args.limit) if args.limit else stops_gdf
    logger.info("Rendering %d stops …", len(stops_gdf_iterate))

    city_bbox_gdf_render = expand_gdf_bounds_in_degrees(city_bbox_gdf_data, -DEGREES_BUF)

    if args.render_map:
        general_layers = get_data_map_by_bbox_gdf(args.area_id, city_bbox_gdf_data, CONFIG_RENDER["general_layers_name"])
        general_layers = reproject_all(general_layers, local_projection)
    else:
        general_layers = None

    for _, stop_row in stops_gdf_iterate.iterrows():
        transit_out_path = os.path.join(save_dir, f"transit_map_{stop_row.stop_id}_{slugify(stop_row['name'])}.png")
        _prepare_for_transit_map_and_render(args, stop_row, ctx_map, general_layers, local_projection, transit_out_path)


    if args.render_map:
        far_layers = get_data_map_by_bbox_gdf(args.area_id, city_bbox_gdf_data, CONFIG_RENDER["far_layers_name"])
        far_layers = reproject_all(far_layers, local_projection)
    else:
        far_layers = None

    for _, stop_row in stops_gdf_iterate.iterrows():
        far_plan_out_path = os.path.join(save_dir, f"far_plan_{stop_row.stop_id}_{slugify(stop_row['name'])}.png")
        _prepare_for_far_plan_and_render(args, stop_row, ctx_map, far_layers, local_projection, far_plan_out_path,
                                         city_bbox_gdf_render.to_crs(local_projection))

    for _, stop_row in stops_gdf_iterate.iterrows():
        transit_out_path = os.path.join(save_dir, f"transit_map_{stop_row.stop_id}_{slugify(stop_row['name'])}.png")
        detailed_out_path = os.path.join(save_dir, f"detailed_map_{stop_row.stop_id}_{slugify(stop_row['name'])}.png")
        far_plan_out_path = os.path.join(save_dir, f"far_plan_{stop_row.stop_id}_{slugify(stop_row['name'])}.png")
        poster_out_path = os.path.join(save_dir, f"poster_{stop_row.stop_id}_{slugify(stop_row['name'])}.png")

        stop_bbox_gdf = get_stop_bbox_gdf(stop_row, local_projection, LOCAL_MAP_RADIUS)
        if args.render_map:
            detailed_layers = get_data_map_by_bbox_gdf(args.area_id, stop_bbox_gdf.to_crs(4326),
                                                       CONFIG_RENDER["detailed_layers_name"])
            detailed_layers = reproject_all(detailed_layers, local_projection)
        else:
            detailed_layers = None

        _prepare_for_detailed_map_and_render(args, stop_row, ctx_map, detailed_layers, local_projection,
                                             detailed_out_path)

        compose_img_to_poster(transit_out_path, detailed_out_path, far_plan_out_path, poster_out_path)


def _prepare_for_transit_map_and_render(args, stop_row, ctx_map, layers, local_projection, transit_map_out_path):
    stop_bbox_gdf = get_stop_bbox_gdf(stop_row, local_projection, TRANSIT_MAP_RADIUS)
    figsize = [20, 20]
    render_middle_transit_map(stop_row, ctx_map, layers, stop_bbox_gdf, args, transit_map_out_path,
                              figsize_poster=figsize)


def _prepare_for_detailed_map_and_render(args, stop_row, ctx_map, layers, local_projection, detailed_map_out_path):
    stop_bbox_gdf = get_stop_bbox_gdf(stop_row, local_projection, LOCAL_MAP_RADIUS)
    figsize = [10, 10]
    render_detailed_map(stop_row, ctx_map, layers, stop_bbox_gdf, args, detailed_map_out_path, figsize_poster=figsize)


def _prepare_for_far_plan_and_render(args, stop_row, ctx_map, layers, local_projection, far_plan_out_path,
                                     city_bbox_gdf):
    stop_bbox_gdf = get_stop_bbox_gdf(stop_row, local_projection, TRANSIT_MAP_RADIUS)
    figsize = [10, 10]
    render_far_plan(stop_row, ctx_map, layers, city_bbox_gdf, stop_bbox_gdf, args, far_plan_out_path,
                    figsize_poster=figsize)


def _choose_stops(stops_gdf: gpd.GeoDataFrame, bbox_gdf: gpd.GeoDataFrame):
    if stops_gdf.crs != bbox_gdf.crs:
        bbox_gdf = bbox_gdf.to_crs(stops_gdf.crs)

    # inside = gpd.clip(stops_gdf, bbox_gdf)
    # stops_gdf = inside[inside['routes'].apply(len) > 4]
    return stops_gdf
