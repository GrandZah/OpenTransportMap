import logging
import matplotlib.pyplot as plt
import geopandas as gpd
from pandas import Series

from data_map.get_data_map import LayersMap
from data_transport.сity_route_database import CityRouteDatabase
from load_configs import CONFIG_RENDER, PAPER_SIZES_INCH
from logger import log_function_call
from render_map.render_basemap import render_basemap
from render_map.render_labels_for_layers import render_labels_for_layers
from render_map.render_walk_access import render_walk_5min_focus_gap
from render_transport.render_bus_lines_v2 import render_bus_lines_v2
from utils.forbidden import ForbiddenCollector
from utils.utils_rendering import fit_bbox_to_aspect

logger = logging.getLogger(__name__)


def _settings_ax(ax, bbox_gdf):
    minx, miny, maxx, maxy = bbox_gdf.total_bounds
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_axis_off()
    ax.margins(0)


@log_function_call
def render_middle_transit_map(stop_row: Series, ctx_map: CityRouteDatabase, layers: LayersMap,
                              bbox_gdf: gpd.GeoDataFrame, args, out_path: str, figsize_poster=None):
    """Render a medium-scale transit map with routes and walking overlay."""
    figsize = figsize_poster or PAPER_SIZES_INCH.get(getattr(args, "paper", "A2"), PAPER_SIZES_INCH["A1"])

    dpi = CONFIG_RENDER.get("dpi", 150)
    target_aspect = figsize[0] / figsize[1]
    bleed = CONFIG_RENDER.get("bleed", 0.0)
    fitted_bbox = fit_bbox_to_aspect(bbox_gdf, aspect=target_aspect, bleed=bleed)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    _settings_ax(ax, fitted_bbox)

    forbidden = ForbiddenCollector()

    if args.render_map:
        render_basemap(ax, layers, fitted_bbox)
    if args.render_routes:
        render_bus_lines_v2(ax, stop_row, ctx_map, forbidden=forbidden)
    if args.render_map:
        render_labels_for_layers(ax, layers, fitted_bbox, forbidden_px=forbidden.geoms)
        render_walk_5min_focus_gap(ax, stop_row, color="#d7263d",
                                   dash_on_off=(36.0, 22.0), dash_offset=6.0, gap_width_deg=0, label_text="")
    fig.savefig(out_path, pad_inches=0)
    plt.close(fig)
    logger.info("Saved %s", out_path)
