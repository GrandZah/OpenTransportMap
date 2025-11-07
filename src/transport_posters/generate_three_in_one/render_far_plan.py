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
from render_transport.render_bus_lines_v2 import render_bus_lines_v2_only_last
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
def render_far_plan(stop_row: Series, ctx_map: CityRouteDatabase, layers: LayersMap,
                    bbox_gdf: gpd.GeoDataFrame, transit_bbox, args, out_path: str, figsize_poster=None):
    """Render a far-scale map with only terminal routes and boundaries."""
    figsize = figsize_poster or PAPER_SIZES_INCH.get(getattr(args, "paper", "A4"), PAPER_SIZES_INCH["A4"])

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
        render_bus_lines_v2_only_last(ax, stop_row, ctx_map, bbox_gdf, forbidden=forbidden)
    if args.render_map:
        render_labels_for_layers(ax, layers, fitted_bbox, forbidden_px=forbidden.geoms)
        draw_gdf_boundaries_dashed(ax, transit_bbox)

    fig.savefig(out_path, pad_inches=0)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def draw_gdf_boundaries_dashed(
        ax,
        gdf: gpd.GeoDataFrame,
        color: str = "black",
        linewidth: float = 1.8,
        dash_on: float = 28.0,
        dash_off: float = 14.0,
        alpha: float = 0.35,
        zorder: int = 3,
        joinstyle: str = "round",
        capstyle: str = "round",
) -> None:
    """Plot only the boundaries of *gdf* as a semi-transparent long dash line.
    Works for Polygon/MultiPolygon/LineString. ``dash_on``/``dash_off`` are in pt."""
    if gdf is None or gdf.empty:
        return

    gdf.plot(
        ax=ax,
        facecolor="none",
        edgecolor=color,
        linewidth=linewidth,
        linestyle=(0, (dash_on, dash_off)),
        alpha=alpha,
        zorder=zorder,
    )
