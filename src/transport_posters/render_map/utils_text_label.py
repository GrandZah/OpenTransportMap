from typing import Optional, Dict
import numpy as np
from matplotlib.text import Text
from shapely.geometry import LineString, box
from shapely.affinity import rotate
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from rtree.index import Index as _RTIndex
from math import cos, sin, radians
from shapely.geometry import Polygon as _Poly

from data_map.get_style_layers import LabelSpec
from utils.utils_rendering import points_per_meter, meters_to_px

CONFIG_RENDER_LABELS = {

    "coef_text_wight": 0.65,

    "margin_min_m": 0.25,
    "margin_text_h_coeff": 0.30,
    "margin_halo_coeff": 0.50,

    "line_tol_min_m": 0.40,
    "line_tol_text_h_coeff": 0.60,

    "repeat_min_px": 80.0,
    "repeat_max_px": 1000.0,
}


class _LabelCollider:
    """Store occupied regions (in px) and quickly check intersections."""

    def __init__(self):
        self._geoms = []
        self._rt = _RTIndex() if _RTIndex else None
        self._next_id = 0

    def conflicts(self, poly_px) -> bool:
        if self._rt:
            hits = list(self._rt.intersection(poly_px.bounds))
            if not hits:
                return False
            return any(self._geoms[i].intersects(poly_px) for i in hits)
        else:
            return any(g.intersects(poly_px) for g in self._geoms)

    def add(self, poly_px):
        if self._rt:
            self._rt.insert(self._next_id, poly_px.bounds)
        self._geoms.append(poly_px)
        self._next_id += 1


def _text_height_m(spec: LabelSpec, ppm: Optional[float]) -> float:
    if ppm is None:
        return 0.0
    return spec.size_pt / max(ppm, 1e-6)


def _label_margin_m(spec: LabelSpec, ppm: Optional[float]) -> float:
    if ppm is None:
        return CONFIG_RENDER_LABELS["margin_min_m"]
    text_h = _text_height_m(spec, ppm)
    halo_h = getattr(spec, "halo_width", 0) / max(ppm, 1e-6)
    return max(
        CONFIG_RENDER_LABELS["margin_min_m"],
        CONFIG_RENDER_LABELS["margin_text_h_coeff"] * text_h + CONFIG_RENDER_LABELS["margin_halo_coeff"] * halo_h,
    )


def _text_width_m(ax: plt.Axes, s: str, size_pt: float, ppm: Optional[float] = None) -> float:
    width_pt = size_pt * CONFIG_RENDER_LABELS["coef_text_wight"] * max(1, len(s))
    if ppm is None:
        ppm = points_per_meter(ax)
    if ppm <= 0:
        return 0.0
    return width_pt / ppm


def _text_footprint(cx: float, cy: float, w_m: float, h_m: float,
                    angle_deg: float, align: Optional[str]) -> _Poly:
    a = (align or "center").lower()
    if a == "left":
        minx, maxx = cx, cx + w_m
    elif a == "right":
        minx, maxx = cx - w_m, cx
    else:
        minx, maxx = cx - 0.5 * w_m, cx + 0.5 * w_m
    miny, maxy = cy - 0.5 * h_m, cy + 0.5 * h_m
    rect = box(minx, miny, maxx, maxy)
    if angle_deg:
        rect = rotate(rect, angle_deg, origin=(cx, cy), use_radians=False)
    return rect


def _text_footprint_px(ax: plt.Axes, cx: float, cy: float,
                       w_m: float, h_m: float, angle_deg: float,
                       align: Optional[str], ppm: Optional[float]) -> _Poly:
    cx_px, cy_px = ax.transData.transform((cx, cy))

    w_px = meters_to_px(ax, w_m, ppm=ppm)
    h_px = meters_to_px(ax, h_m, ppm=ppm)

    th = radians(angle_deg or 0.0)
    ux, uy = cos(th), sin(th)
    vx, vy = -uy, ux

    a = (align or "center").lower()
    if a == "left":
        cx_px += 0.5 * w_px * ux
        cy_px += 0.5 * w_px * uy
    elif a == "right":
        cx_px -= 0.5 * w_px * ux
        cy_px -= 0.5 * w_px * uy

    dxw, dyw = 0.5 * w_px * ux, 0.5 * w_px * uy
    dxh, dyh = 0.5 * h_px * vx, 0.5 * h_px * vy
    pts = [
        (cx_px - dxw - dxh, cy_px - dyw - dyh),
        (cx_px + dxw - dxh, cy_px + dyw - dyh),
        (cx_px + dxw + dxh, cy_px + dyw + dyh),
        (cx_px - dxw + dxh, cy_px - dyw + dyh),
    ]
    return _Poly(pts)


def _footprint_fits(bbox_geom, footprint, margin_m: float) -> bool:
    safe = bbox_geom.buffer(-margin_m)
    return (not safe.is_empty) and safe.covers(footprint)


class _NameDeduper:
    def __init__(self, min_px: float, ax: plt.Axes):
        self.min_px = float(min_px)
        self.ax = ax
        self._by_name: Dict[str, list[tuple[float, float]]] = {}

    def allow(self, name: str, x: float, y: float) -> bool:
        pts = self._by_name.setdefault(name, [])
        if not pts:
            pts.append((x, y))
            return True
        x0, y0 = self.ax.transData.transform((x, y))
        for (px, py) in pts:
            x1, y1 = self.ax.transData.transform((px, py))
            if np.hypot(x0 - x1, y0 - y1) < self.min_px:
                return False
        pts.append((x, y))
        return True


def _draw_label(ax: plt.Axes, x: float, y: float, text: str, spec: LabelSpec) -> Text | None:
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None

    kw = dict(
        x=x, y=y, s=t, fontsize=spec.size_pt, color=spec.color, fontproperties=spec.fontproperties, weight=spec.weight,
        alpha=spec.alpha, zorder=spec.zorder, va="center", rotation_mode="anchor",
        ha={"center": "center", "left": "left", "right": "right"}.get(spec.align, "center")
    )

    if spec.italic:
        kw["style"] = "italic"

    txt = ax.text(**kw)
    txt.set_path_effects([
        pe.Stroke(linewidth=spec.halo_width, foreground=spec.halo_color, alpha=spec.alpha),
        pe.Normal(),
    ])
    return txt


def _straight_segment(ls: LineString, s0: float, s1: float, tol_m: float):
    """Chord between points at distances s0..s1; return ok, segment, p0, p1."""
    p0 = ls.interpolate(s0, normalized=False)
    p1 = ls.interpolate(s1, normalized=False)
    seg = LineString([(p0.x, p0.y), (p1.x, p1.y)])
    ok = ls.buffer(tol_m, cap_style=2).covers(seg)
    return ok, seg, p0, p1
