import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Iterable
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, box
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
import geopandas as gpd

from data_transport.get_bus_layers import CityRouteDatabase
from logger import log_function_call

from render_transport.render_stop_point_v2 import (
    StopLabelInput,
    layout_and_render_stop_pairs,
)
from utils.forbidden import ForbiddenCollector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderStyle:
    LINE_COLOR: str = "#34a853"
    LINE_WIDTH_PT: float = 3.0
    CURRENT_STOP_COLOR: str = "#d7263d"
    STOP_EDGE: str = "#34a853"
    STOP_SIZE_PT2: float = 45.0
    STOP_SIZE_PT2_SMALL: float = 16


RENDER_BUS_LINES_STYLE = RenderStyle()


def _get_routes_without_last_stop(stop_row: pd.Series,
                                  routes_gdf: gpd.GeoDataFrame,
                                  route_ids: List[int]) -> List[int]:
    routes_without_last_stop = []
    for rid in route_ids:
        route_sel = routes_gdf[routes_gdf.route_id == rid]
        if route_sel.empty:
            logger.warning("Route %s not found in routes_gdf", rid)
            continue
        route = route_sel.iloc[0]
        stops_seq = route["stop_seq"]
        if not stops_seq or stop_row.stop_id == stops_seq[-1]:
            continue
        routes_without_last_stop.append(rid)
    return routes_without_last_stop


def _get_last_stops_for_routes(ctx_map: CityRouteDatabase,
                               route_ids: Iterable[int],
                               exclude: Set[int] | None = None) -> Set[int]:
    """Return IDs of terminal stops for each route in *route_ids*.

    The "last" stop is the ``seq_to`` of the segment with maximal
    ``shape_dist`` if present, otherwise with maximal ``edge_idx``.
    """
    last_stops: Set[int] = set()
    ex = exclude or set()

    for rid in route_ids:
        edges = ctx_map.edges_map.get(rid)
        if edges is None or edges.empty:
            continue

        last_row = edges.loc[edges["edge_idx"].idxmax()]

        last_stop_id = int(last_row["seq_to"])
        if last_stop_id not in ex:
            last_stops.add(last_stop_id)

    return last_stops


def _get_last_stops_for_routes_in_bbox(
        ctx_map: CityRouteDatabase,
        route_ids: Iterable[int],
        bbox_gdf: gpd.GeoDataFrame,
        exclude: Set[int] | None = None,
        use_intersects: bool = True,
) -> Set[int]:
    """Return IDs of terminal stops for each route in *route_ids* within *bbox_gdf*.

    - Route order uses ``shape_dist`` if any non-zero values exist, otherwise
      ``edge_idx``.
    - "Last in bbox" is the first stop from the end whose geometry intersects
      (or, if *use_intersects* is False, lies within) the union of
      *bbox_gdf*.
    - *exclude* lists stop_ids to ignore (e.g. the current stop).
    """
    ex = exclude or set()
    result: Set[int] = set()

    minx, miny, maxx, maxy = map(float, bbox_gdf.total_bounds)
    # bbox_geom = box(minx, miny, maxx, maxy)

    stops_idx = ctx_map.stops_gdf.set_index("stop_id")["geometry"]

    for rid in route_ids:
        edges = ctx_map.edges_map.get(rid)
        if edges is None or edges.empty:
            continue

        cols = edges.columns

        if "shape_dist" in cols and edges["shape_dist"].notna().any():
            order_col = "shape_dist"
        else:
            order_col = "edge_idx"

        ordered = edges.sort_values(order_col, ascending=True)

        seq_from_first = int(ordered.iloc[0]["seq_from"])
        seq_tos = [int(x) for x in ordered["seq_to"].tolist()]
        route_stop_seq: list[int] = [seq_from_first]
        for sid in seq_tos:
            if not route_stop_seq or route_stop_seq[-1] != sid:
                route_stop_seq.append(sid)

        picked: int | None = None
        for sid in reversed(route_stop_seq):
            if sid in ex:
                continue

            if sid not in stops_idx.index:
                continue
            pt = stops_idx.loc[sid]
            if pt is None:
                continue
            # inside = bbox_geom.intersects(pt) if use_intersects else pt.within(bbox_geom)
            inside = (minx <= pt.x <= maxx) and (miny <= pt.y <= maxy)
            if inside:
                picked = sid
                break

        if picked is not None:
            result.add(picked)

    return result


def _get_geometry_routes(ctx_map: CityRouteDatabase,
                         route_ids: List[int],
                         stop_id: int) -> Tuple[List[gpd.GeoSeries], Set[int]]:
    routes_map = ctx_map.routes_map
    edges_map = ctx_map.edges_map
    visited_stops: Set[int] = set()
    all_geometry = []

    for rid in route_ids:
        route_sel = routes_map.get(rid)
        if route_sel.empty:
            logger.warning("Route %s not found in routes_map", rid)
            continue
        route = route_sel.iloc[0]
        stop_seq = route["stop_seq"]
        if stop_id not in stop_seq:
            logger.warning("Stop %s not in route %s stop_seq", stop_id, rid)
            continue

        idx = stop_seq.index(stop_id)
        visited_stops.update(stop_seq[idx + 1:])

        edges_route = edges_map.get(rid)
        if edges_route.empty:
            logger.warning("No edges for route %s", rid)
            continue

        start_edge = edges_route[edges_route.seq_from == stop_id]
        if start_edge.empty:
            logger.warning("Stop %s not found in edges of route %s", stop_id, rid)
            continue
        start_idx = int(start_edge.edge_idx.iloc[0])

        onward = edges_route[edges_route.edge_idx >= start_idx].sort_values("edge_idx")
        all_geometry.append(onward)

    return all_geometry, visited_stops


def _draw_routes(ax, all_geometry: List[gpd.GeoSeries]) -> None:
    lines = []
    for onward in all_geometry:
        geoms = getattr(onward, "geometry", onward)
        for geom in geoms:
            if isinstance(geom, LineString):
                lines.append(geom)
            else:
                logger.warning("Route has a non-LineString part")
    if not lines:
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
        zorder=3,
    ))


@log_function_call
def render_bus_lines_v2(ax,
                        stop_row: pd.Series,
                        ctx_map: CityRouteDatabase,
                        time_budget_sec: float =  0.5,
                        forbidden: 'ForbiddenCollector|None' = None) -> None:
    """Draw bus lines and lay out labels globally, mirroring E/W sides."""
    id2ref: Dict[int, str] = ctx_map.id2ref
    platforms_map: Dict[int, gpd.GeoDataFrame] = ctx_map.platforms_map

    stop_id = int(stop_row.stop_id)
    stop_pt: Point = stop_row.geometry

    platform_row = platforms_map.get(stop_id)
    if platform_row.empty:
        logger.warning("Platform %s not found in platforms_map", stop_id)
        return
    platform_pt: Point = platform_row.iloc[0].geometry

    route_ids = _get_routes_without_last_stop(stop_row, ctx_map.routes_gdf, stop_row.routes)
    logger.debug("Stop %s serves routes: %s", stop_id, route_ids)

    all_geometry, visited_stops = _get_geometry_routes(ctx_map, route_ids, stop_id)
    _draw_routes(ax, all_geometry)

    inputs: List[StopLabelInput] = []

    cur_bus_nums = [id2ref.get(rid, str(rid)) for rid in route_ids]
    inputs.append(StopLabelInput(
        stop_id=stop_id,
        stop_point=stop_pt,
        platform_point=platform_pt,
        stop_name=str(stop_row['short_name']),
        bus_nums=cur_bus_nums,
        stop_face=RENDER_BUS_LINES_STYLE.CURRENT_STOP_COLOR,
        stop_edge=RENDER_BUS_LINES_STYLE.STOP_EDGE,
        stop_size_pt2=RENDER_BUS_LINES_STYLE.STOP_SIZE_PT2,
    ))

    for vis_stop_id in visited_stops:
        cur_stop_row = ctx_map.stops_gdf[ctx_map.stops_gdf.stop_id == vis_stop_id]
        if cur_stop_row.empty:
            continue
        cur_stop_row = cur_stop_row.iloc[0]
        cur_stop_pt: Point = cur_stop_row.geometry

        platform_row2 = platforms_map.get(vis_stop_id)
        if platform_row2.empty:
            logger.warning("Platform %s not found in platforms_map", vis_stop_id)
            continue
        platform_pt2: Point = platform_row2.iloc[0].geometry

        cur_route_ids = cur_stop_row.routes
        visible_rids = list(set(route_ids) & set(cur_route_ids))
        if not visible_rids:
            continue

        bus_nums = [id2ref.get(rid, str(rid)) for rid in visible_rids]

        inputs.append(StopLabelInput(
            stop_id=vis_stop_id,
            stop_point=cur_stop_pt,
            platform_point=platform_pt2,
            stop_name=str(cur_stop_row['short_name']),
            bus_nums=bus_nums,
            stop_face="white",
            stop_edge=RENDER_BUS_LINES_STYLE.STOP_EDGE,
            stop_size_pt2=RENDER_BUS_LINES_STYLE.STOP_SIZE_PT2,
        ))

    if not inputs:
        logger.debug("No stops to label")
        return

    layout_and_render_stop_pairs(
        ax=ax,
        stops=inputs,
        time_budget_sec=time_budget_sec,
        forbidden=forbidden
    )


@log_function_call
def render_bus_lines_v2_only_last(ax,
                                  stop_row: pd.Series,
                                  ctx_map: CityRouteDatabase,
                                  bbox_gdf,
                                  time_budget_sec: float = 0.5,
                                  forbidden: 'ForbiddenCollector|None' = None
                                  ) -> None:
    """Draw bus lines and labels globally (mirrored E/W) keeping only routes
    for which this stop is terminal."""
    id2ref: Dict[int, str] = ctx_map.id2ref
    platforms_map: Dict[int, gpd.GeoDataFrame] = ctx_map.platforms_map

    stop_id = int(stop_row.stop_id)
    stop_pt: Point = stop_row.geometry

    platform_row = platforms_map.get(stop_id)
    if platform_row.empty:
        logger.warning("Platform %s not found in platforms_map", stop_id)
        return
    platform_pt: Point = platform_row.iloc[0].geometry

    route_ids = _get_routes_without_last_stop(stop_row, ctx_map.routes_gdf, stop_row.routes)
    logger.debug("Stop %s serves routes: %s", stop_id, route_ids)

    all_geometry, visited_stops = _get_geometry_routes(ctx_map, route_ids, stop_id)
    _draw_routes(ax, all_geometry)

    inputs: List[StopLabelInput] = []

    inputs.append(StopLabelInput(
        stop_id=stop_id,
        stop_point=stop_pt,
        platform_point=platform_pt,
        stop_name="",
        bus_nums=[],
        stop_face=RENDER_BUS_LINES_STYLE.CURRENT_STOP_COLOR,
        stop_edge=RENDER_BUS_LINES_STYLE.STOP_EDGE,
        stop_size_pt2=RENDER_BUS_LINES_STYLE.STOP_SIZE_PT2,
    ))

    last_stops = _get_last_stops_for_routes_in_bbox(ctx_map, route_ids, bbox_gdf, exclude={stop_id})

    for vis_stop_id in last_stops:
        cur_stop_row = ctx_map.stops_gdf[ctx_map.stops_gdf.stop_id == vis_stop_id]
        if cur_stop_row.empty:
            continue
        cur_stop_row = cur_stop_row.iloc[0]
        cur_stop_pt: Point = cur_stop_row.geometry

        platform_row2 = platforms_map.get(vis_stop_id)
        if platform_row2.empty:
            logger.warning("Platform %s not found in platforms_map", vis_stop_id)
            continue
        platform_pt2: Point = platform_row2.iloc[0].geometry

        cur_route_ids = cur_stop_row.routes
        visible_rids = list(set(route_ids) & set(cur_route_ids))
        visible_rids = _get_routes_with_last_stop(cur_stop_row, visible_rids, ctx_map.routes_map)
        if not visible_rids:
            continue

        bus_nums = [id2ref.get(rid, str(rid)) for rid in visible_rids]

        inputs.append(StopLabelInput(
            stop_id=vis_stop_id,
            stop_point=cur_stop_pt,
            platform_point=platform_pt2,
            stop_name=str(cur_stop_row['short_name']),
            bus_nums=bus_nums,
            stop_face="white",
            stop_edge=RENDER_BUS_LINES_STYLE.STOP_EDGE,
            stop_size_pt2=RENDER_BUS_LINES_STYLE.STOP_SIZE_PT2,
        ))

    for vis_stop_id in (visited_stops - last_stops):
        cur_stop_row = ctx_map.stops_gdf[ctx_map.stops_gdf.stop_id == vis_stop_id]
        if cur_stop_row.empty:
            continue
        cur_stop_row = cur_stop_row.iloc[0]
        cur_stop_pt: Point = cur_stop_row.geometry

        platform_row2 = platforms_map.get(vis_stop_id)
        if platform_row2.empty:
            logger.warning("Platform %s not found in platforms_map", vis_stop_id)
            continue
        platform_pt2: Point = platform_row2.iloc[0].geometry

        inputs.append(StopLabelInput(
            stop_id=vis_stop_id,
            stop_point=cur_stop_pt,
            platform_point=platform_pt2,
            stop_name="",
            bus_nums=[],
            stop_face="white",
            stop_edge=RENDER_BUS_LINES_STYLE.STOP_EDGE,
            stop_size_pt2=RENDER_BUS_LINES_STYLE.STOP_SIZE_PT2_SMALL,
        ))

    if not inputs:
        logger.debug("No stops to label")
        return

    layout_and_render_stop_pairs(
        ax=ax,
        stops=inputs,
        time_budget_sec=time_budget_sec,
        forbidden=forbidden
    )


def _get_routes_with_last_stop(stop_row, route_rids, routes_map):
    new_rids = []
    for route_rid in route_rids:
        route_sel = routes_map.get(route_rid)
        if route_sel.empty:
            logger.warning("Route %s not found in routes_gdf", route_rid)
        route = route_sel.iloc[0]
        rout_stops = route["stop_seq"]
        if rout_stops[-1] == stop_row.stop_id:
            new_rids.append(route_rid)
    return new_rids
