from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Optional
import math
import logging

import matplotlib.font_manager as fm
from matplotlib.textpath import TextPath
from matplotlib.transforms import Bbox
from matplotlib import patheffects as pe

from shapely.geometry import Point

from load_configs import FONT_INTER_BOLD
from utils.forbidden import ForbiddenCollector
from utils.utils import natural_key

from render_transport.stop_layout_opt import (
    RectPX, CandidatePair,
    precompute_costs, optimize_all, cost_breakdown_per_stop
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderStopConfig:
    """Rendering configuration for stop labels and route capsules."""

    STOP_SIZE: float = 45.0
    STOP_COLOR: str = "#34a853"

    FONT_SIZE_LABEL: float = 24.0
    FONT_SIZE_BUS: float = 18.0

    BUS_BOX_STYLE: str = "round"
    BUS_BOX_PAD_FRAC: float = 0.2
    BUS_BOX_FACE: str = "white"
    BUS_BOX_EDGE: str = "gray"
    BUS_BOX_ALPHA: float = 0.8
    BUS_BOX_LINEWIDTH: float = 0.8

    ROUTES_SEP: str = ". "
    ROUTES_MAX_LINES: int = 3
    ROUTES_WIDTH_MIN_PT: float = 200.0
    ROUTES_WIDTH_FACTOR: float = 1.2

    LABEL_ROUTES_GAP_PT: float = 2.0
    ANCHOR_GAP_PT: float = 8.0
    GRID_SNAP_PX: int = 4

    MAX_ITERS: int = 20000
    W_OVERLAP: float = 50.0
    W_CORRIDOR: float = 80.0
    W_OFFSET: float = 0.02
    W_LINES: float = 2.0
    W_ROUTES_NEAR: float = 0.5
    W_OCCLUDE_STOP: float = 50.0
    STOP_BUFFER_PT: float = 3.0
    W_SIDE_COHESION: float = 10.0
    SIDE_NEIGHBOR_RADIUS_PX: float = 160.0

    NAME_WIDTH_REF_PX: float = 240.0
    ROUTES_FAR_LEN_BOOST: float = 0.6
    ROUTES_NEAR_FACTOR_MAX: float = 3.0

    HOTK_2OPT: int = 8
    HOTK_3OPT: int = 4
    DEBUG_TOPK_REPORT: int = 6


config_render_stop = RenderStopConfig()
custom_font = fm.FontProperties(fname=FONT_INTER_BOLD)


@dataclass
class StopLabelInput:
    """Input data for laying out a stop label."""
    stop_id: int
    stop_point: Point
    platform_point: Point
    stop_name: str
    bus_nums: List[str]
    stop_face: str
    stop_edge: str
    stop_size_pt2: float


def _ppp(ax) -> float:
    return ax.figure.dpi / 72.0


def _data_to_px(ax, x: float, y: float) -> Tuple[float, float]:
    X, Y = ax.transData.transform((x, y))
    return float(X), float(Y)


def _axis_bbox_px(ax) -> RectPX:
    ax.figure.canvas.draw()
    rend = ax.figure.canvas.get_renderer()
    bbox: Bbox = ax.get_window_extent(renderer=rend)
    return RectPX(bbox.x0, bbox.y0, bbox.x1, bbox.y1)

@lru_cache(maxsize=8192)
def _text_wh_pt(text: str, size_pt: float) -> Tuple[float, float]:
    if not text:
        return 0.0, size_pt
    tp = TextPath((0, 0), text, prop=custom_font, size=size_pt)
    bb = tp.get_extents()
    return bb.width, bb.height


def _snap(v_px: float, step: int) -> float:
    return round(v_px / step) * step


def _point_to_rect_dist_px(px: float, py: float, r: RectPX) -> float:
    dx = 0.0 if r.l <= px <= r.r else (r.l - px if px < r.l else px - r.r)
    dy = 0.0 if r.b <= py <= r.t else (r.b - py if py < r.b else py - r.t)
    return math.hypot(dx, dy)


def _make_name_rect_sign(px_anchor: Tuple[float, float],
                         w_pt: float, h_pt: float,
                         sign: int,
                         gap_pt: float, ppp: float, snap: int) -> RectPX:
    """Name beside the point: vertically centered; horizontal side via *sign*."""
    w_px = w_pt * ppp
    h_px = h_pt * ppp
    X, Y = px_anchor
    gap_px = gap_pt * ppp

    l = _snap(X + sign * gap_px - (0.5 * (1 - sign)) * w_px, snap)
    r = l + w_px
    b = Y - 0.5 * h_px
    t = b + h_px
    return RectPX(l, b, r, t)


def _corridor_between(bottom_of_name_px: float,
                      top_of_box_px: float,
                      left_px: float, right_px: float) -> RectPX:
    mid_y = 0.5 * (bottom_of_name_px + top_of_box_px)
    return RectPX(left_px, mid_y - 1.0, right_px, mid_y + 1.0)


def _prepare_routes_lines(
        bus_nums: List[str],
        size_pt: float,
        max_width_pt: float,
        target_width_pt: float
) -> Tuple[List[str], float, float]:
    """Wrap route numbers preserving order and balancing line widths."""
    uniq_sorted = sorted(set(bus_nums), key=natural_key)
    sep = config_render_stop.ROUTES_SEP

    memo = {}

    def w_of(parts: List[str]) -> float:
        key = tuple(parts)
        if key not in memo:
            s = sep.join(parts)
            w, _ = _text_wh_pt(s, size_pt)
            memo[key] = w
        return memo[key]

    if not uniq_sorted:
        _, line_h = _text_wh_pt("Hg", size_pt)
        return [], 0.0, 0.0

    if w_of(uniq_sorted) <= max_width_pt:
        lines = [sep.join(uniq_sorted)]
    else:
        best = None
        n = len(uniq_sorted)

        def score(ws: List[float]) -> float:
            over = sum(max(0.0, w - max_width_pt) for w in ws)
            bal = sum((w / max(1.0, target_width_pt)) ** 3 for w in ws)
            return 1000.0 * over + bal

        for i in range(1, n):
            l1, l2 = uniq_sorted[:i], uniq_sorted[i:]
            w1, w2 = w_of(l1), w_of(l2)
            s = score([w1, w2])
            cand = ([sep.join(l1), sep.join(l2)], [w1, w2])
            if best is None or s < best[0]:
                best = (s, cand)

        if config_render_stop.ROUTES_MAX_LINES >= 3:
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    l1, l2, l3 = uniq_sorted[:i], uniq_sorted[i:j], uniq_sorted[j:]
                    w1, w2, w3 = w_of(l1), w_of(l2), w_of(l3)
                    s = score([w1, w2, w3])
                    cand = ([sep.join(l1), sep.join(l2), sep.join(l3)], [w1, w2, w3])
                    if best is None or s < best[0]:
                        best = (s, cand)

        lines, ws = best[1]
        if max(ws) > max_width_pt and len(lines) >= 2:
            last = lines[-1].split(sep)
            mid = max(1, len(last) // 2)
            lines[-1:] = [sep.join(last[:mid]), sep.join(last[mid:])]

    _, line_h = _text_wh_pt("Hg", size_pt)
    total_w = max((_text_wh_pt(s, size_pt)[0] for s in lines), default=0.0)
    total_h = len(lines) * line_h
    return lines[:config_render_stop.ROUTES_MAX_LINES], total_w, total_h


def _candidates_for_stop(ax,
                         stop_idx: int,
                         s: StopLabelInput) -> Tuple[List[CandidatePair], List[str]]:
    name_w_pt, name_h_pt = _text_wh_pt(s.stop_name, config_render_stop.FONT_SIZE_LABEL)
    max_routes_w_pt = max(config_render_stop.ROUTES_WIDTH_MIN_PT,
                          config_render_stop.ROUTES_WIDTH_FACTOR * name_w_pt)

    routes_lines, routes_w_pt, routes_h_pt = _prepare_routes_lines(
        s.bus_nums, config_render_stop.FONT_SIZE_BUS,
        max_routes_w_pt, name_w_pt
    )

    pad_pt = config_render_stop.BUS_BOX_PAD_FRAC * config_render_stop.FONT_SIZE_BUS
    lw_pt = float(config_render_stop.BUS_BOX_LINEWIDTH)
    routes_w_pt_box = routes_w_pt + 2 * pad_pt + lw_pt
    routes_h_pt_box = routes_h_pt + 2 * pad_pt + lw_pt

    ppp = _ppp(ax)
    Xs, Ys = _data_to_px(ax, s.stop_point.x, s.stop_point.y)
    snap = config_render_stop.GRID_SNAP_PX
    gap_pair_pt = config_render_stop.ANCHOR_GAP_PT
    gap_inner_pt = config_render_stop.LABEL_ROUTES_GAP_PT

    dx_box2text_pt = pad_pt + 0.5 * lw_pt
    dy_box2text_pt = pad_pt + 0.5 * lw_pt

    r_pt = math.sqrt(max(1e-6, s.stop_size_pt2) / math.pi)
    stop_radius_px = (r_pt + config_render_stop.STOP_BUFFER_PT) * ppp

    def name_offset_from_rect(rect_px: RectPX) -> Tuple[float, float]:
        cy = 0.5 * (rect_px.b + rect_px.t)
        return (rect_px.l - Xs) / ppp, (cy - Ys) / ppp

    def routes_text_offset_from_box(left_px: float, top_px: float) -> Tuple[float, float]:
        x_text_px = left_px + dx_box2text_pt * ppp
        y_text_px = top_px - dy_box2text_pt * ppp
        return (x_text_px - Xs) / ppp, (y_text_px - Ys) / ppp

    cands: List[CandidatePair] = []
    for sign in (+1, -1):
        name_rect = _make_name_rect_sign((Xs, Ys), name_w_pt, name_h_pt,
                                         sign, gap_pair_pt, ppp, snap)

        top_box_px = name_rect.b - gap_inner_pt * ppp
        w_box_px = routes_w_pt_box * ppp

        def mk_candidate(left_box_px: float, is_near: bool) -> CandidatePair:
            right_box_px = left_box_px + w_box_px
            routes_box = RectPX(left_box_px,
                                top_box_px - routes_h_pt_box * ppp,
                                right_box_px,
                                top_box_px)
            name_off = name_offset_from_rect(name_rect)
            routes_off = routes_text_offset_from_box(left_box_px, top_box_px)
            corridor = _corridor_between(name_rect.b, top_box_px,
                                         min(name_rect.l, left_box_px),
                                         max(name_rect.r, right_box_px))
            union_rect = RectPX.union(name_rect, routes_box)
            anchor_dist = _point_to_rect_dist_px(Xs, Ys, union_rect)
            routes_dist = _point_to_rect_dist_px(Xs, Ys, routes_box)

            return CandidatePair(
                stop_idx=stop_idx,
                name_offset_pt=name_off,
                routes_text_offset_pt=routes_off,
                name_rect_px=name_rect,
                routes_box_rect_px=routes_box,
                corridor_rect_px=corridor,
                pair_union_rect_px=union_rect,
                side_sign=sign,
                is_near_edge=is_near,
                point_px=(Xs, Ys),
                stop_radius_px=stop_radius_px,
                lines_used=len(routes_lines),
                anchor_dist_px=anchor_dist,
                routes_dist_px=routes_dist
            )

        left_near = _snap(name_rect.l if sign > 0 else name_rect.r - w_box_px, snap)
        cands.append(mk_candidate(left_near, is_near=True))

        left_far = _snap(name_rect.r - w_box_px if sign > 0 else name_rect.l, snap)
        cands.append(mk_candidate(left_far, is_near=False))

    return cands, routes_lines


def _rect_inside(box, r, pad=0):
    return (r.l >= box.l + pad and r.b >= box.b + pad and
            r.r <= box.r - pad and r.t <= box.t - pad)


def layout_and_render_stop_pairs(ax, stops, time_budget_sec: float = 0.5, debug: bool = False,
                                 forbidden: 'ForbiddenCollector|None' = None) -> None:
    """Layout stop/name pairs and render them onto *ax*."""

    ax.figure.canvas.draw()

    if debug:
        logger.info(f"layout: start, stops={len(stops)}, time_budget={time_budget_sec:.3f}s")

    per_stop_cands: List[List[CandidatePair]] = []
    routes_text_lines: List[List[str]] = []
    axbox = _axis_bbox_px(ax)

    for i, s in enumerate(stops):
        cands, lines = _candidates_for_stop(ax, i, s)

        good = [c for c in cands
                if _rect_inside(axbox, c.name_rect_px) and _rect_inside(axbox, c.routes_box_rect_px)]

        per_stop_cands.append(good)
        routes_text_lines.append(lines)

    idx_nonempty = [i for i, c in enumerate(per_stop_cands) if c]
    selection: List[Optional[int]] = [None] * len(stops)

    if idx_nonempty:

        cand_ne = [per_stop_cands[i] for i in idx_nonempty]
        pre_ne = precompute_costs(cand_ne, config_render_stop)
        sel_ne = optimize_all(cand_ne, pre_ne, time_budget_sec, config_render_stop, debug=debug)

        for j, i in enumerate(idx_nonempty):
            selection[i] = sel_ne[j]

    for i, s in enumerate(stops):
        k = selection[i]
        if k is None:
            continue

        c = per_stop_cands[i][k]
        routes_text = "\n".join(routes_text_lines[i]) if routes_text_lines[i] else ""

        ax.scatter([s.stop_point.x], [s.stop_point.y],
                   s=s.stop_size_pt2, facecolor=s.stop_face,
                   edgecolor=s.stop_edge, linewidths=1.0, zorder=5)

        ax.annotate(
            s.stop_name,
            xy=(s.stop_point.x, s.stop_point.y),
            xytext=c.name_offset_pt,
            textcoords="offset points",
            ha="left", va="center",
            fontsize=config_render_stop.FONT_SIZE_LABEL,
            fontproperties=custom_font,
            zorder=6,
            clip_on=True,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")]
        )

        ax.annotate(
            routes_text,
            xy=(s.stop_point.x, s.stop_point.y),
            xytext=c.routes_text_offset_pt,
            textcoords="offset points",
            ha="left", va="top",
            fontsize=config_render_stop.FONT_SIZE_BUS,
            fontproperties=custom_font,
            zorder=6,
            clip_on=True,
            bbox=dict(
                boxstyle=f"{config_render_stop.BUS_BOX_STYLE},pad={config_render_stop.BUS_BOX_PAD_FRAC}",
                facecolor=config_render_stop.BUS_BOX_FACE,
                edgecolor=config_render_stop.BUS_BOX_EDGE,
                alpha=config_render_stop.BUS_BOX_ALPHA,
                linewidth=config_render_stop.BUS_BOX_LINEWIDTH
            )
        )

        if forbidden is not None:
            ppp = ax.figure.dpi / 72.0
            pad = 1.0 * ppp

            forbidden.add_rect(c.name_rect_px, buffer_px=pad)
            forbidden.add_rect(c.routes_box_rect_px, buffer_px=pad)

            forbidden.add_rect(c.corridor_rect_px, buffer_px=0.5 * ppp)

            forbidden.add_circle(c.point_px[0], c.point_px[1], c.stop_radius_px, buffer_px=0.0)
