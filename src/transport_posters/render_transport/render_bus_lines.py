import logging
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from pandas import Series
from shapely.geometry import LineString, Point
import geopandas as gpd
import matplotlib.patheffects as pe

from data_transport.сity_route_database import CityRouteDatabase
from logger import log_function_call
from render_transport.render_stop_point import render_stop_point


@dataclass(frozen=True)
class RenderStyle:
    LINE_COLOR: str = "#34a853"
    LINE_WIDTH_PT: float = 3
    CURRENT_STOP_COLOR: str = "#d7263d"


RENDER_BUS_LINES_STYLE = RenderStyle()
logger = logging.getLogger(__name__)


@log_function_call
def render_bus_lines(ax, stop_row: pd.Series, ctx_map: CityRouteDatabase) -> None:
    id2ref = ctx_map.id2ref
    platforms_map = ctx_map.platforms_map

    stop_id = int(stop_row.stop_id)
    stop_pt: Point = stop_row.geometry
    platform_row = platforms_map.get(stop_id)
    if platform_row.empty:
        logger.warning("Platform %s not found in platforms_map", stop_id)
        return
    platform_pt = platform_row.iloc[0].geometry

    route_ids = get_routes_without_last_stop(stop_row, ctx_map.routes_gdf, stop_row.routes)
    logger.debug("Stop %s serves routes: %s", stop_id, route_ids)

    all_geometry, visited_stops = _get_geometry_routes(ctx_map, route_ids, stop_id)
    _draw_routes(ax, all_geometry)

    render_stop_point(ax, stop_pt, platform_pt, stop_row['short_name'], id2ref, route_ids, route_ids,
                      RENDER_BUS_LINES_STYLE.CURRENT_STOP_COLOR)
    _render_stops(ax, visited_stops, ctx_map.stops_gdf, platforms_map, id2ref, route_ids)


def _get_geometry_routes(ctx_map, route_ids, stop_id):
    routes_map = ctx_map.routes_map
    edges_map = ctx_map.edges_map

    visited_stops = set()
    all_geometry = []
    for rid in route_ids:
        route_sel = routes_map.get(rid)
        if route_sel.empty:
            logger.warning("Route %s not found in routes_gdf", rid)
        route = route_sel.iloc[0]
        rout_stops = route["stop_seq"]
        idx = rout_stops.index(stop_id)
        visited_stops.update(rout_stops[idx + 1:])

        edges_route = edges_map.get(rid)
        if edges_route.empty:
            logger.warning("No edges for route %s", rid)
            continue

        start = edges_route[edges_route.seq_from == stop_id]
        if start.empty:
            logger.warning("Stop %s not found in edges of route %s", stop_id, rid)
            continue
        start_idx = int(start.edge_idx.iloc[0])

        onward = edges_route[edges_route.edge_idx >= start_idx].sort_values("edge_idx")
        all_geometry.append(onward)

    return all_geometry, visited_stops


def _draw_routes(ax, all_geometry):
    lines = []
    for onward in all_geometry:
        geoms = getattr(onward, "geometry", onward)
        for geom in geoms:
            if isinstance(geom, LineString):
                lines.append(geom)
            else:
                logger.warning("Route has not LineString part")
    if not lines:
        logger.debug("No LineString segments to draw")
        return
    paths = [np.asarray(g.coords) for g in lines]
    w = RENDER_BUS_LINES_STYLE.LINE_WIDTH_PT
    ax.add_collection(LineCollection(
        paths,
        linewidths=w,
        capstyle="round",
        joinstyle="round",
        color=RENDER_BUS_LINES_STYLE.LINE_COLOR,
        path_effects=[pe.Stroke(linewidth=w + 2, foreground="white"), pe.Normal()],
    ))
    logger.debug("Drawn %d segments for stop", len(paths))


def _render_stops(ax, visited_stops: set, stops_gdf: gpd.GeoDataFrame, platforms_map: Dict[int, gpd.GeoDataFrame],
                  id2ref: Dict[int, str], route_ids: List[int]) -> None:
    for vis_stop_id in visited_stops:
        cur_stop_row = stops_gdf[stops_gdf.stop_id == vis_stop_id]
        if cur_stop_row.empty:
            continue
        cur_stop_row = cur_stop_row.iloc[0]
        cur_stop_pt: Point = cur_stop_row.geometry

        platform_row = platforms_map.get(vis_stop_id)
        if platform_row.empty:
            logger.warning("Platform %s not found in platforms_map", vis_stop_id)
            continue
        platform = platform_row.iloc[0]
        platform_pt = platform.geometry

        cur_route_ids = cur_stop_row.routes

        render_stop_point(ax, cur_stop_pt, platform_pt, str(cur_stop_row["short_name"]), id2ref, route_ids,
                          cur_route_ids, "white")


def get_routes_without_last_stop(stop_row: Series, routes_gdf: gpd.GeoDataFrame, route_ids: List[int]) -> List[int]:
    routes_without_last_stop = []
    for rid in route_ids:
        route_sel = routes_gdf[routes_gdf.route_id == rid]
        if route_sel.empty:
            logger.warning("Route %s not found in routes_gdf", rid)
            continue
        route = route_sel.iloc[0]

        rout_stops = route["stop_seq"]

        if stop_row.stop_id == rout_stops[-1]:
            logger.debug("Stop %s is the final stops of routes: %s", stop_row.stop_id, route_ids)
            continue

        routes_without_last_stop.append(rid)

    return routes_without_last_stop
