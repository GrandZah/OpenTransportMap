import logging
import os
import matplotlib.pyplot as plt
from pandas import Series
import geopandas as gpd

from transport_posters.data_map.get_data_map import get_data_map_by_bbox_gdf
from transport_posters.data_map.get_layers import LayersMap, reproject_all
from transport_posters.data_transport.get_bus_layers import get_from_cache_bus_layers
from transport_posters.data_transport.сity_route_database import CityRouteDatabase
from transport_posters.load_configs import CONFIG_PATHS, CONFIG_RENDER
from transport_posters.logger import log_function_call
from transport_posters.render_map.render_basemap import render_basemap
from transport_posters.render_map.render_labels_for_layers import render_labels_for_layers
from transport_posters.render_transport.render_bus_lines import render_bus_lines
from transport_posters.utils.utils import slugify
from transport_posters.utils.utils_generate import generate_gallery, get_local_projection_by_area_id, \
    expand_gdf_bounds_in_meters

logger = logging.getLogger(__name__)


@log_function_call
def generate_maps(args):
    save_dir = CONFIG_PATHS.output_maps_dir / f"city_{args.area_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    local_projection = get_local_projection_by_area_id(args.area_id)

    ctx_map: CityRouteDatabase = get_from_cache_bus_layers(int(args.area_id))
    routes_bbox_gdf = expand_gdf_bounds_in_meters(ctx_map.routes_gdf, 1000)

    ctx_map = ctx_map.reproject_all(local_projection)
    local_routes_bbox_gdf = routes_bbox_gdf.copy().to_crs(local_projection)

    layers = get_data_map_by_bbox_gdf(args.area_id, routes_bbox_gdf, CONFIG_RENDER["general_layers_name"])
    layers = reproject_all(layers, local_projection)

    stops_gdf = ctx_map.stops_gdf
    stops_gdf_iterate = stops_gdf.head(args.limit) if args.limit else stops_gdf
    logger.info(f"Rendering {len(stops_gdf_iterate)} stops …")
    for _, stop_row in stops_gdf_iterate.iterrows():
        _render_stop_map(stop_row, ctx_map, layers, local_routes_bbox_gdf, args, save_dir)

    if args.gallery and save_dir:
        generate_gallery(save_dir)


def _render_stop_map(stop_row: Series, ctx_map: CityRouteDatabase, layers: LayersMap,
                     bbox_gdf: gpd.GeoDataFrame, args, save_dir: str):
    fig, ax = plt.subplots(figsize=CONFIG_RENDER["figsize"], dpi=CONFIG_RENDER.get("dpi", 150))

    ax.set_aspect('equal')
    minx, miny, maxx, maxy = bbox_gdf.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_axis_off()

    if args.render_map:
        render_basemap(ax, layers, bbox_gdf)
    if args.render_routes:
        render_bus_lines(ax, stop_row, ctx_map)
    if args.render_map:
        render_labels_for_layers(ax, layers, bbox_gdf)

    out_path = os.path.join(save_dir, f"{stop_row.stop_id}_{slugify(stop_row['name'])}.png")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
