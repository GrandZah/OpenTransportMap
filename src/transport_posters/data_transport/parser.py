import logging

import pyproj
from pyproj import Geod
from typing import Dict, List, Tuple, Optional
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import substring, transform, linemerge

from logger import log_function_call
from utils.util_short_name import shorten_stop_name

logger = logging.getLogger(__name__)


class OSMParseError(Exception):
    pass


class NoRoutesFound(OSMParseError):
    pass


class InvalidGeometryType(OSMParseError):
    def __init__(self, route_id: int, geom_type: str, expected_types: str):
        super().__init__(
            f"Route {route_id}: has invalid geometry type '{geom_type}'. Expected type :{expected_types}")


class GeometryMissing(OSMParseError):
    def __init__(self, route_id: int, way_id: int):
        super().__init__(f"Route {route_id}: way {way_id} lacks geometry")


class EmptyRouteGeometry(OSMParseError):
    def __init__(self, route_id: int):
        super().__init__(f"Without route {route_id}, cause: Empty merged_geom")


class RouteEndpointMismatch(OSMParseError):
    def __init__(self, route_id: int, stop_id: int, where: str, dist_m: float):
        super().__init__(f"Route {route_id}: {where} stop {stop_id} is too far ({dist_m:.1f} m) from route {where}")


class StopSequenceTooShort(OSMParseError):
    def __init__(self, route_id: int, count: int):
        super().__init__(
            f"Route {route_id}: too few stops ({count}). Please verify the stop count for this route in OpenStreetMap")


class StopProjectionFailed(OSMParseError):
    def __init__(self, route_id: int, reason: str | None = None):
        super().__init__(f"Route {route_id} likely has incorrect stop sequence. Please verify in OpenStreetMap")


class NonIncreasingStopDistances(OSMParseError):
    def __init__(self, route_id: int, dists: List[float]):
        super().__init__(
            f"Route {route_id} has non-increasing stop distances. Likely incorrect stop sequence."
            f" Please verify in OpenStreetMap. Current distances: {dists}")


class MissingRefTag(OSMParseError):
    def __init__(self, route_id: int):
        super().__init__(f"Remove route {route_id}, cause: no ref tag")


class RouteHasNoGeometry(OSMParseError):
    def __init__(self, route_id: int, stop_count: int):
        super().__init__(f"Route {route_id} has not geometry and have {stop_count} stops; skip")


@log_function_call
def parse_osm(data: Dict, *, strict: bool = False) \
        -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, Dict[int, List[float]]]:
    """
    Parse data from OSM. Take only validated routes and stops.

    Returns (stops_gdf, platforms_gdf, routes_gdf, stops_dists_map)."""

    ways: Dict[int, Dict] = {}
    relations: List[Dict] = []
    stop_nodes: Dict[int, Dict] = {}

    for el in data.get("elements", []):
        t = el.get("type")
        if t == "way":
            ways[el["id"]] = el
        elif t == "relation":
            relations.append(el)
        elif t == "node":
            if el.get("tags", {}).get("public_transport") == "platform":
                stop_nodes[el["id"]] = el

    route_rows: List[Dict] = []
    stop_to_refs: Dict[int, List[str]] = {}
    stops_dists_map: Dict[int, List[float]] = {}
    errors: List[Exception] = []

    for rel in relations:
        tags = rel.get("tags", {})
        if tags.get("route") != "bus":
            continue
        try:
            route_row = _parse_one_route(rel, ways, stop_nodes, stops_dists_map)
            if route_row:
                route_rows.append(route_row)
                for sid in route_row["stop_seq"]:
                    if sid in stop_nodes:
                        stop_to_refs.setdefault(sid, []).append(route_row["route_id"])
        except OSMParseError as e:
            if strict:
                errors.append(e)
            else:
                logger.warning(str(e))
                continue

    if strict and errors:
        raise ExceptionGroup("OSM parse errors", errors)

    if not route_rows:
        logger.exception(f"Data did not contain right routes")
        raise NoRoutesFound("route_rows is empty — parser OSM search nothing")

    routes_gdf = gpd.GeoDataFrame(route_rows, crs="EPSG:4326", geometry="geometry")

    logger.debug("Start adding stop_rows")
    stop_rows: List[Dict] = []
    platform_rows = []
    for sid, node in stop_nodes.items():
        tags = node.get("tags", {})
        name = tags.get("name", "").strip()
        if not name:
            name = str(sid)
        if sid not in stop_to_refs:
            continue
        stop_rows.append(
            {
                "stop_id": int(sid),
                "name": str(name),
                "short_name": shorten_stop_name(name),
                "routes": sorted(set(stop_to_refs[sid])),
                "geometry": Point(node["lon"], node["lat"]),
            }
        )
        platform_rows.append(
            {
                "stop_id": int(sid),
                "name": str(name),
                "geometry": Point(node["lon"], node["lat"])
            }
        )

    stops_gdf = gpd.GeoDataFrame(stop_rows, crs="EPSG:4326", geometry="geometry")
    platforms_gdf = gpd.GeoDataFrame(platform_rows, crs="EPSG:4326", geometry="geometry")
    logger.debug("End adding stop_rows")

    stops_gdf = _find_projection_for_stops(stops_gdf, routes_gdf)

    logger.info("Number of stops in stops_gdf: %d", len(stops_gdf))
    logger.info("Number of platforms in platforms_gdf: %d", len(platforms_gdf))
    logger.info("Number of routes in routes_gdf: %d", len(routes_gdf))
    return stops_gdf, platforms_gdf, routes_gdf, stops_dists_map


def _parse_one_route(rel: Dict, ways: Dict[int, Dict], stop_nodes: Dict[int, Dict],
                     stops_dists_map: Dict[int, List[float]]) -> Optional[Dict]:
    tags = rel.get("tags", {})
    route_id = (rel["id"])
    segments = []
    stop_seq: List[int] = []

    for m in rel.get("members", []):
        if m.get("type") == "node":
            sid = m["ref"]
            if sid in stop_nodes:
                stop_seq.append(sid)

    for m in rel.get("members", []):
        if m.get("type") == "way":
            w = ways.get(m["ref"])
            if not w or "geometry" not in w:
                raise GeometryMissing(route_id=route_id, way_id=m["ref"])
            cords = [[pt["lon"], pt["lat"]] for pt in w["geometry"]]

            if not segments:
                if not stop_seq:
                    continue
                stop_node = stop_nodes[stop_seq[0]]
                stop_node = [stop_node["lon"], stop_node["lat"]]
                d_start = find_dist_btw_pt(stop_node, cords[0])
                d_end = find_dist_btw_pt(stop_node, cords[-1])
                if d_end < d_start:
                    cords.reverse()
            else:
                last_node = segments[-1]
                d_start = find_dist_btw_pt(last_node, cords[0])
                d_end = find_dist_btw_pt(last_node, cords[-1])
                if d_end < d_start:
                    cords.reverse()

            segments.extend(cords)

    if len(stop_seq) == 0 or not segments:
        raise RouteHasNoGeometry(route_id, len(stop_seq))

    merged_geom = LineString(segments)
    if merged_geom.is_empty:
        raise EmptyRouteGeometry(route_id)

    _validate_on_bad_route(route_id, stop_seq, stop_nodes, merged_geom)
    _add_stop_seq_dists_of_route_in_map(route_id, stop_seq, stop_nodes, merged_geom, stops_dists_map)

    if "ref" not in tags:
        raise MissingRefTag(route_id)

    return {
        "route_id": int(rel["id"]),
        "ref": str(tags["ref"]),
        "stop_seq": stop_seq,
        "geometry": merged_geom,
    }


def _validate_on_bad_route(route_id: int, stop_seq: List[int], stop_nodes: Dict[int, Dict],
                           route_geom: LineString) -> None:
    TOLERANCE_M = 100
    first_stop_sid = stop_seq[0]
    last_stop_sid = stop_seq[-1]
    first_stop = stop_nodes[first_stop_sid]
    last_stop = stop_nodes[last_stop_sid]
    cords = list(route_geom.coords)
    dist_start = find_dist_btw_pt([first_stop["lon"], first_stop["lat"]], cords[0])
    dist_end = find_dist_btw_pt([last_stop["lon"], last_stop["lat"]], cords[-1])

    if dist_start > TOLERANCE_M:
        raise RouteEndpointMismatch(route_id, first_stop_sid, "start", dist_start)

    if dist_end > TOLERANCE_M:
        raise RouteEndpointMismatch(route_id, last_stop_sid, "end", dist_end)


def _add_stop_seq_dists_of_route_in_map(route_id: int, stop_seq: List[int], stop_nodes: Dict[int, Dict],
                                        route_geom: LineString, stops_dists_map: Dict[int, List[float]]) -> None:
    """ Count stop_dists_map of route in meter and find error in OSM data"""
    local_crs = gpd.GeoSeries([route_geom], crs="EPSG:4326").estimate_utm_crs()
    local_proj = local_crs.to_string()
    project_to_local_proj = pyproj.Transformer.from_crs(pyproj.CRS("EPSG:4326"), pyproj.CRS(local_proj),
                                                        always_xy=True).transform

    route_local_geom = transform(project_to_local_proj, route_geom)

    stop_ids: List[int] = stop_seq
    if len(stop_ids) < 2:
        raise StopSequenceTooShort(route_id, len(stop_seq))

    dists: List[float] = []
    pos = 0.0
    full_tail: LineString = route_local_geom

    try:
        for sid in stop_ids:
            node = stop_nodes[sid]
            pt_local = transform(project_to_local_proj, Point(node["lon"], node["lat"]))
            try:
                pt_proj = full_tail.interpolate(full_tail.project(pt_local))
            except:
                logger.warning(
                    f"non-lineal geometry: full_tail.geom_type=%s, pt_local.geom_type=%s, route_id={route_id}, {full_tail}",
                    full_tail.geom_type, pt_local.geom_type
                )
                raise InvalidGeometryType(route_id, full_tail.geom_type, " ".join(["LineString", "MultiLineString"]))

            d = _find_and_get_first_projection_dist(full_tail, pt_proj)
            pos = pos + d
            dists.append(pos)
            full_tail = substring(full_tail, d, full_tail.length)
    except (KeyError, IndexError, ValueError) as e:
        raise StopProjectionFailed(route_id) from e

    if not all(dists[i] < dists[i + 1] for i in range(len(dists) - 1)):
        raise NonIncreasingStopDistances(route_id, dists)

    stops_dists_map[route_id] = dists


def _find_and_get_first_projection_dist(line: LineString, pt: Point, tol: float = 1e-6) -> float:
    cum = 0.0
    for a, b in zip(line.coords[:-1], line.coords[1:]):
        seg = LineString([a, b])
        d_loc = seg.project(pt)
        if seg.interpolate(d_loc).distance(pt) <= tol:
            return cum + d_loc
        cum += seg.length
    raise ValueError("Point is not on the line (tol might be too small).")


def find_dist_btw_pt(pt1, pt2) -> float:
    """ For point in EPSG:4326 projection get distance between them"""
    geode = Geod(ellps="WGS84")

    lon1, lat1 = pt1
    lon2, lat2 = pt2
    _, _, d_end = geode.inv(lon1, lat1, lon2, lat2)
    return d_end


def _find_projection_for_stops(stops_gdf: gpd.GeoDataFrame, routes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    TOL_METERS = 50

    route_geoms_m = routes_gdf.set_index("route_id")["geometry"].to_dict()

    new_geoms = []
    for idx, stop in stops_gdf.iterrows():
        pt = stop.geometry
        first_route = stop.routes[0]
        line = route_geoms_m[first_route]
        proj_pt = line.interpolate(line.project(pt))
        dist_between_stop_and_projection = find_dist_btw_pt([pt.x, pt.y], [proj_pt.x, proj_pt.y])
        if dist_between_stop_and_projection <= TOL_METERS:
            new_geoms.append(proj_pt)
        else:
            logger.warning(
                f"Distance between stop and projection is {dist_between_stop_and_projection} in stop with id {stop.stop_id}")
            new_geoms.append(pt)

    stops_gdf["geometry"] = new_geoms
    return stops_gdf
