import logging
from typing import List, Dict
import geopandas as gpd
from shapely.geometry.linestring import LineString
from shapely.ops import substring

from logger import log_function_call

logger = logging.getLogger(__name__)


@log_function_call
def build_edges_pairwise(routes_gdf: gpd.GeoDataFrame,
                         stops_dists_map: Dict[int, List[float]]
                         ) -> gpd.GeoDataFrame:
    """
    Divides each route into segments between adjacent stops,
    correctly handling cases where the same line point occurs multiple times (rings, "pockets", duplicate coordinates).

    Output: GeoDataFrame with fields
    edge_id | route_id | seq_from | seq_to | edge_idx | length_m | shape_dist | geometry
    """
    local_crs = routes_gdf.estimate_utm_crs()
    local_proj = local_crs.to_string()
    routes_m = routes_gdf.to_crs(local_proj)

    rows: List[Dict] = []

    for r in routes_m.itertuples():
        line_full: LineString = r.geometry
        stop_ids: List[int] = r.stop_seq
        dists: List[float] = stops_dists_map[r.route_id]

        cum = 0.0
        for idx, (sid0, sid1, d0, d1) in enumerate(zip(stop_ids[:-1], stop_ids[1:], dists[:-1], dists[1:])):
            if d1 <= d0:
                continue

            seg = substring(line_full, d0, d1)
            seg_len = seg.length

            rows.append(dict(
                edge_id=f"{r.route_id}_{idx}",
                route_id=r.route_id,
                seq_from=sid0,
                seq_to=sid1,
                edge_idx=idx,
                length_m=seg_len,
                shape_dist=cum,
                geometry=seg,
            ))
            cum += seg_len

    edges_gdf = gpd.GeoDataFrame(rows, crs=local_proj)
    logger.info("Number of edges in edges_gdf: %d", len(edges_gdf))
    return edges_gdf.to_crs(routes_gdf.crs)
