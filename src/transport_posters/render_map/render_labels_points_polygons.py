from shapely.geometry import Point
import matplotlib.pyplot as plt
import logging
import geopandas as gpd
from typing import List, Tuple

from transport_posters.data_map.get_style_layers import LabelSpec
from transport_posters.render_map.utils_text_label import (
    _label_margin_m,
    _text_width_m,
    _text_height_m,
    _text_footprint,
    _text_footprint_px,
    _LabelCollider,
    _draw_label,
    _footprint_fits,
)

logger = logging.getLogger(__name__)


def _wrap_text_compact(ax, text, size_pt, ppm, max_width_m, max_lines=3):
    """
    Balanced layout for short labels (3 to 10 words), without duplication or caching.
    Priority: 1 line > 2 lines > 3 lines (and so on, but no more than 3).
    If a non-overflow option is found for k lines, options with a higher k are not considered.

    Penalties:
    - underfill: (slack ^ 2), the last line is not penalized (natural "flagpole");
    - overflow: (overflow^2) * OVER_PEN (very expensive, only as a backup).
    """
    words = text.split()
    n = len(words)
    if n == 0:
        return []
    if max_lines < 1:
        max_lines = 1

    def line_w(i, j):

        return _text_width_m(ax, " ".join(words[i:j]), size_pt, ppm=ppm)

    P = 2.0
    OVER = 100.0
    LAST_W = 0.0

    def pen(width, is_last):
        slack = max_width_m - width
        if slack >= 0:
            return (slack ** P) * (LAST_W if is_last else 1.0)
        else:
            return (abs(slack) ** P) * OVER

    def assemble(split):

        lines, start = [], 0
        for end in split:
            lines.append(" ".join(words[start:end]))
            start = end
        return lines

    one = " ".join(words)
    if line_w(0, n) <= max_width_m or max_lines == 1:
        return [one]

    best_lines = None
    best_cost = float("inf")

    def try_k(k, require_fit=True):
        nonlocal best_lines, best_cost
        if k < 2:
            return False
        found_fit = False

        if k == 2:
            for i in range(1, n):
                w1, w2 = line_w(0, i), line_w(i, n)
                c = pen(w1, False) + pen(w2, True)
                fits = (w1 <= max_width_m and w2 <= max_width_m)
                if require_fit and not fits:
                    continue
                if not require_fit and not fits:
                    pass
                if c < best_cost:
                    best_cost = c
                    best_lines = assemble((i, n))
                found_fit = found_fit or fits

        elif k >= 3:

            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    w1, w2, w3 = line_w(0, i), line_w(i, j), line_w(j, n)
                    c = pen(w1, False) + pen(w2, False) + pen(w3, True)
                    fits = (w1 <= max_width_m and w2 <= max_width_m and w3 <= max_width_m)
                    if require_fit and not fits:
                        continue
                    if c < best_cost:
                        best_cost = c
                        best_lines = assemble((i, j, n))
                    found_fit = found_fit or fits
        return found_fit

    if max_lines >= 2 and try_k(2, require_fit=True):
        return best_lines

    if max_lines >= 3 and n >= 3 and try_k(3, require_fit=True):
        return best_lines

    best_lines, best_cost = None, float("inf")
    if max_lines >= 2:
        try_k(2, require_fit=False)
    if max_lines >= 3 and n >= 3:
        try_k(3, require_fit=False)

    return best_lines if best_lines else [one]


def _text_block_size_m(ax: plt.Axes, lines: List[str], size_pt: float, ppm: float,
                       line_spacing: float = 1.10) -> Tuple[float, float]:
    """Width and height of a multi-line block (m), based on the maximum line length and the number of lines."""
    if not lines:
        return 0.0, 0.0
    w_m = max(_text_width_m(ax, ln, size_pt, ppm) for ln in lines)
    h_line = _text_height_m(LabelSpec(size_pt=size_pt), ppm)
    h_m = h_line * (len(lines) * line_spacing)
    return w_m, h_m


def _place_centroid_like(
        ax: plt.Axes,
        gdf: gpd.GeoDataFrame,
        spec: LabelSpec,
        field: str,
        stats: dict,
        bbox_geom,
        ppm,
        *,
        collider: _LabelCollider,
        get_anchor
):
    """
    General logic for placing point/centroid labels:
    - take the anchor (point);
    - transfer the text "compactly" into several lines under the target width;
    - calculate the footprint of the block and check the bbox + collisions.
    """
    margin_m = _label_margin_m(spec, ppm)
    line_spacing = float(getattr(spec, "line_spacing", 1.10))
    max_lines = int(getattr(spec, "max_lines", 3))

    bxmin, bymin, bxmax, bymax = bbox_geom.bounds
    bbox_w_m = max(0.0, bxmax - bxmin)
    wrap_width_m = float(getattr(spec, "wrap_width_m", 0.10 * bbox_w_m))

    for geom, val in zip(gdf.geometry, gdf[field]):
        if geom.is_empty:
            stats["empty_geom"] += 1
            continue

        p: Point = get_anchor(geom)
        text = None if val is None else str(val).strip()
        if not text:
            stats["empty_text"] += 1
            stats["attempted"] += 1
            continue

        lines = _wrap_text_compact(ax, text, spec.size_pt, ppm, max_width_m=wrap_width_m, max_lines=max_lines)
        if not lines:
            stats["empty_text"] += 1
            stats["attempted"] += 1
            continue

        w_m, h_m = _text_block_size_m(ax, lines, spec.size_pt, ppm, line_spacing=line_spacing)

        fp = _text_footprint(p.x, p.y, w_m, h_m, angle_deg=0.0, align=getattr(spec, "align", "center"))
        if not _footprint_fits(bbox_geom, fp, margin_m):
            stats["bbox_cross"] = stats.get("bbox_cross", 0) + 1
            stats["attempted"] += 1
            continue

        fp_px = _text_footprint_px(ax, p.x, p.y, w_m, h_m, 0.0, getattr(spec, "align", "center"), ppm)
        if collider.conflicts(fp_px):
            stats["overlap_skip"] = stats.get("overlap_skip", 0) + 1
            stats["attempted"] += 1
            continue

        txt = _draw_label(ax, p.x, p.y, "\n".join(lines), spec)
        if txt:
            try:
                txt.set_linespacing(line_spacing)
            except Exception:
                pass
            collider.add(fp_px)
            stats["placed"] += 1
        stats["attempted"] += 1


def label_points(ax: plt.Axes, gdf: gpd.GeoDataFrame, spec: LabelSpec, field: str,
                 stats: dict, bbox_geom, ppm, *, collider: _LabelCollider):
    """Public version: signatures for points. General logic + anchor = point itself or representative_point()."""

    def _anchor(geom):
        return geom if isinstance(geom, Point) else geom.representative_point()

    _place_centroid_like(ax, gdf, spec, field, stats, bbox_geom, ppm, collider=collider, get_anchor=_anchor)


def label_polygons(ax: plt.Axes, gdf: gpd.GeoDataFrame, spec: LabelSpec, field: str,
                   stats: dict, bbox_geom, ppm, *, collider: _LabelCollider):
    """Public version: polygon signatures (by centroid/representative point)."""

    def _anchor(geom):
        return geom.representative_point()

    _place_centroid_like(ax, gdf, spec, field, stats, bbox_geom, ppm, collider=collider, get_anchor=_anchor)
