import logging
from dataclasses import dataclass
from typing import List, Set, Tuple
import math

import pandas as pd
import geopandas as gpd
from shapely.geometry import box, MultiPoint, Point

import matplotlib.font_manager as fm
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle

from data_transport.get_bus_layers import CityRouteDatabase
from load_configs import FONT_INTER_BOLD
from render_transport.stop_layout_opt import RectPX
from utils.forbidden import ForbiddenCollector


@dataclass(frozen=True)
class RenderStopConfig:
    STOP_SIZE_PT2: float = 300.0
    STOP_EDGE_PT: float = 1.2
    STOP_EDGE_COLOR: str = "#34a853"
    STOP_FACE_COLOR: str = "white"

    FONT_SIZE_LABEL: float = 18.0
    NUMBER_PAD_PT: float = 1.6

    MERGE_DIST_M: float = 25.0

    HERE_PIN_PX: int = 48
    HERE_PIN_FILL_COLOR: str = "#E4323B"
    HERE_PIN_EDGE_COLOR: str = "#7B1E24"
    HERE_PIN_HALO_LW_PX: float = 5.0
    HERE_PIN_EDGE_LW_PX: float = 1.8
    HERE_PIN_INNER_ALPHA: float = 0.04
    HERE_PIN_DOT_REL_Y: float = 0.36
    HERE_PIN_DOT_REL_R: float = 0.22
    PIN_ZORDER: int = 3


@dataclass(frozen=True)
class RenderStyle:
    LINE_COLOR: str = "#34a853"
    LINE_WIDTH_PT: float = 3.0
    CURRENT_STOP_COLOR: str = "#d7263d"


CFG = RenderStopConfig()
STYLE = RenderStyle()
custom_font = fm.FontProperties(fname=FONT_INTER_BOLD)
logger = logging.getLogger(__name__)


def render_stops(ax, stop_row: pd.Series, ctx_map: CityRouteDatabase, bbox: gpd.GeoDataFrame,
                 forbidden: 'ForbiddenCollector|None' = None) -> None:
    """Render the red 'you are here' pin at the current stop and numbered circles for other stops.
    Nearby stops are merged into a single marker (cluster centroid)."""
    logger.debug("Start render_stops")

    if stop_row.empty:
        logger.warning("Empty stop_row")
        return

    stop_id = int(stop_row.stop_id)
    origin: Point = stop_row.geometry

    _draw_here_pin(ax, origin, forbidden=forbidden)

    stop_ids = get_stops_in_bbox(ctx_map, bbox) - {stop_id}
    _render_stops_numbered(ax, stop_ids, ctx_map.stops_gdf, origin, forbidden=forbidden)

    logger.debug("End render_stops")


def get_stops_in_bbox(ctx_map: CityRouteDatabase, bbox: gpd.GeoDataFrame) -> Set[int]:
    if bbox.empty or ctx_map.stops_gdf.empty:
        return set()
    minx, miny, maxx, maxy = bbox.total_bounds
    poly = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs=ctx_map.stops_gdf.crs).iloc[0]
    mask = ctx_map.stops_gdf.within(poly)
    return set(ctx_map.stops_gdf.loc[mask, "stop_id"])


def _render_stops_numbered(ax, stop_ids: Set[int], stops_gdf: gpd.GeoDataFrame, origin: Point,
                           forbidden: 'ForbiddenCollector|None' = None) -> None:
    if not stop_ids:
        return

    sub = stops_gdf[stops_gdf.stop_id.isin(stop_ids)].copy()
    sub["__dist"] = sub.geometry.distance(origin)

    clusters = _cluster_stops(sub, CFG.MERGE_DIST_M)

    reps: List[Tuple[Point, float]] = []
    for idxs in clusters:
        block = sub.loc[idxs]
        centroid = MultiPoint([p for p in block.geometry]).centroid
        dmin = float(block["__dist"].min())
        reps.append((centroid, dmin))

    reps.sort(key=lambda t: t[1])

    for n, (pt, _) in enumerate(reps, start=1):
        _draw_stop_circle(ax, pt, forbidden=forbidden)
        _draw_stop_number(ax, pt, str(n), forbidden=forbidden)


def _draw_stop_circle(ax, pt: Point,
                      forbidden: 'ForbiddenCollector|None' = None) -> None:
    """Draw a stop circle with a crisp soft shadow in pixels (scale-independent)."""
    r_pt = math.sqrt(max(CFG.STOP_SIZE_PT2, 1e-6) / math.pi)
    px_per_pt = ax.figure.dpi / 72.0
    r_px = r_pt * px_per_pt

    ax.scatter(
        pt.x, pt.y,
        s=CFG.STOP_SIZE_PT2,
        facecolor=CFG.STOP_FACE_COLOR,
        edgecolor=CFG.STOP_EDGE_COLOR,
        linewidths=CFG.STOP_EDGE_PT,
        zorder=5.0,
    )

    if forbidden is not None:
        ppp = ax.figure.dpi / 72.0
        pad = 1.0 * ppp
        cx_px, cy_px = ax.transData.transform((pt.x, pt.y))
        forbidden.add_circle(cx_px, cy_px, r_px, buffer_px=0.0)


def _draw_stop_number(ax, pt: Point, text: str,
                      forbidden: 'ForbiddenCollector|None' = None) -> None:
    fs = _fit_fontsize_in_circle(text, CFG.FONT_SIZE_LABEL, CFG.STOP_SIZE_PT2,
                                 edge_pt=CFG.STOP_EDGE_PT, pad_pt=CFG.NUMBER_PAD_PT)

    tp = TextPath((0, 0), text, size=fs, prop=custom_font)
    bb = tp.get_extents()
    centered = Affine2D().translate(-(bb.x0 + bb.x1) / 2.0, -(bb.y0 + bb.y1) / 2.0)
    trans = centered + Affine2D().translate(pt.x, pt.y) + ax.transData
    patch = PathPatch(tp, transform=trans, fc=CFG.STOP_EDGE_COLOR, ec='none', zorder=6, antialiased=True)
    ax.add_patch(patch)

    if forbidden is not None:
        ppp = ax.figure.dpi / 72.0
        pad = 1.0 * ppp

        renderer = ax.figure.canvas.get_renderer()
        if renderer is None:
            ax.figure.canvas.draw()
            renderer = ax.figure.canvas.get_renderer()

        win_bb = patch.get_window_extent(renderer=renderer)
        rectPx = RectPX(win_bb.x0, win_bb.y0, win_bb.x1, win_bb.y1)
        forbidden.add_rect(rectPx, buffer_px=pad)


def _cluster_stops(sub_gdf: gpd.GeoDataFrame, merge_dist_m: float) -> List[List[int]]:
    """Simple threshold-based union-find O(n^2). Suitable for tens of points within ~500 m."""
    idxs = list(sub_gdf.index)
    pts = [sub_gdf.at[i, "geometry"] for i in idxs]
    parent = {i: i for i in idxs}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    n = len(idxs)
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            if pi.distance(pts[j]) <= merge_dist_m:
                union(idxs[i], idxs[j])

    groups = {}
    for i in idxs:
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


def _fit_fontsize_in_circle(text: str,
                            base_fs: float,
                            area_pt2: float,
                            edge_pt: float,
                            pad_pt: float) -> float:
    """Find a font size that fits the text's real glyph bbox (TextPath) into the circle.
    Inner circle diameter = 2 * (r - edge/2 - pad)."""
    r_pt = math.sqrt(max(area_pt2, 1e-6) / math.pi)
    diam_avail = 2.0 * max(0.0, r_pt - edge_pt * 0.5 - pad_pt)

    lo, hi = 6.0, max(6.0, base_fs)
    for _ in range(14):
        mid = (lo + hi) / 2.0
        tp = TextPath((0, 0), text, size=mid, prop=custom_font)
        bb = tp.get_extents()
        if max(bb.width, bb.height) <= diam_avail:
            lo = mid
        else:
            hi = mid
    return lo


def _draw_here_pin(ax, pt: Point,
                   forbidden: 'ForbiddenCollector|None' = None) -> None:
    """Draw a map pin ('you are here') at pt on axes ax. Size is fixed in pixels (zoom-independent)."""
    size_px = CFG.HERE_PIN_PX
    pin_color = CFG.HERE_PIN_FILL_COLOR
    pin_edge = CFG.HERE_PIN_EDGE_COLOR
    halo_lw_px = CFG.HERE_PIN_HALO_LW_PX
    edge_lw_px = CFG.HERE_PIN_EDGE_LW_PX
    inner_alpha = CFG.HERE_PIN_INNER_ALPHA
    dot_rel_y = CFG.HERE_PIN_DOT_REL_Y
    dot_rel_r = CFG.HERE_PIN_DOT_REL_R
    z = CFG.PIN_ZORDER

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx_per_px = (x1 - x0) / ax.bbox.width
    dy_per_px = (y1 - y0) / ax.bbox.height

    sx = dx_per_px * size_px
    sy = dy_per_px * size_px

    halo_lw = dx_per_px * halo_lw_px
    edge_lw = dx_per_px * edge_lw_px

    verts = [
        (0.0, 0.82),
        (0.55, 0.82),
        (0.68, 0.25),
        (0.00, -0.75),
        (-0.68, 0.25),
        (-0.55, 0.82),
        (0.0, 0.82),
    ]
    codes = [Path.MOVETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CURVE4, Path.CURVE4, Path.CURVE4]
    pin_path = Path(verts, codes)

    TF = (Affine2D()
          .scale(sx, sy)
          .translate(pt.x, pt.y)
          + ax.transData)

    halo = PathPatch(pin_path, facecolor="none", edgecolor="white",
                     lw=halo_lw, antialiased=True, joinstyle="round",
                     transform=TF, zorder=z)
    ax.add_patch(halo)

    pin = PathPatch(pin_path, facecolor=pin_color, edgecolor=pin_edge,
                    lw=edge_lw, antialiased=True, joinstyle="round",
                    transform=TF, zorder=z)
    ax.add_patch(pin)

    inner_TF = (Affine2D()
                .scale(0.92 * sx, 0.92 * sy)
                .translate(pt.x - 0.03 * sx, pt.y + 0.03 * sy)
                + ax.transData)
    inner = PathPatch(pin_path, facecolor="white", edgecolor="none",
                      alpha=inner_alpha, transform=inner_TF, zorder=z + 2)
    ax.add_patch(inner)

    dot_r = dot_rel_r
    dot_y = dot_rel_y
    hole = Circle((pt.x, pt.y + dot_y * sy),
                  radius=dot_r * sx,
                  facecolor="white", edgecolor=pin_edge, lw=edge_lw,
                  transform=ax.transData, zorder=z)
    ax.add_patch(hole)

    if forbidden is not None:
        ppp = ax.figure.dpi / 72.0
        pad_px = 1.0 * ppp

        renderer = ax.figure.canvas.get_renderer() or (ax.figure.canvas.draw() or ax.figure.canvas.get_renderer())
        bb = halo.get_window_extent(renderer)
        r = RectPX(bb.x0, bb.y0, bb.x1, bb.y1)
        forbidden.add_rect(r, buffer_px=pad_px)
