import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import matplotlib.pyplot as plt

from transport_posters.data_map.get_style_layers import LabelSpec
from transport_posters.render_map.utils_text_label import _label_margin_m, CONFIG_RENDER_LABELS, _NameDeduper, _text_width_m, \
    _text_height_m, _straight_segment, _text_footprint, _footprint_fits, _text_footprint_px, _draw_label, _LabelCollider
from transport_posters.utils.utils_rendering import meters_to_px


def label_lines(ax: plt.Axes, gdf: gpd.GeoDataFrame, spec: LabelSpec, field: str,
                stats: dict, bbox_geom, *, ppm: float, collider: _LabelCollider):
    margin_m = _label_margin_m(spec, ppm)

    min_px = float(np.clip(
        meters_to_px(ax, spec.repeat_m, ppm=ppm),
        CONFIG_RENDER_LABELS["repeat_min_px"],
        CONFIG_RENDER_LABELS["repeat_max_px"]
    ))
    dedupe = _NameDeduper(min_px=min_px, ax=ax)

    for geom, name in zip(gdf.geometry, gdf[field]):
        if geom.is_empty:
            stats["empty_geom"] += 1
            continue

        if isinstance(geom, MultiLineString):
            visible_parts = []
            for ls in geom.geoms:
                vis = ls.intersection(bbox_geom)
                if isinstance(vis, LineString) and not vis.is_empty:
                    visible_parts.append(vis)
            if not visible_parts:
                continue
            ls_visible = max(visible_parts, key=lambda ls: ls.length if not ls.is_empty else 0.0)
        elif isinstance(geom, LineString):
            vis = geom.intersection(bbox_geom)
            if not isinstance(vis, LineString) or vis.is_empty:
                continue
            ls_visible = vis
        else:
            stats["not_linestr"] += 1
            continue

        text = "" if name is None else str(name).strip()
        if not text:
            stats["empty_text"] += 1
            continue

        w_m = _text_width_m(ax, text, spec.size_pt, ppm=ppm)
        need_len = max(spec.min_len_m, w_m)

        L = float(ls_visible.length)
        if L < need_len:
            stats["too_short_m"] += 1
            continue

        text_h_m = _text_height_m(spec, ppm)
        tol_m = max(CONFIG_RENDER_LABELS["line_tol_text_h_coeff"] * text_h_m,
                    CONFIG_RENDER_LABELS["line_tol_min_m"])

        mid = 0.5 * need_len
        s_candidates = [mid, L * 0.5, L - mid]

        placed_here = False
        for s in s_candidates:
            if s < mid or s > L - mid:
                continue
            ok, seg, p0, p1 = _straight_segment(ls_visible, s - mid, s + mid, tol_m)
            if not ok:
                continue

            pc = ls_visible.interpolate(s, normalized=False)
            cx, cy = float(pc.x), float(pc.y)
            x0, y0 = ax.transData.transform((p0.x, p0.y))
            x1, y1 = ax.transData.transform((p1.x, p1.y))
            angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
            if angle > 90: angle -= 180
            if angle < -90: angle += 180
            angle = float(np.clip(angle, -spec.max_angle, spec.max_angle))

            fp = _text_footprint(cx, cy, w_m, text_h_m, angle_deg=angle, align=getattr(spec, "align", "center"))
            if not _footprint_fits(bbox_geom, fp, margin_m):
                stats["bbox_cross"] = stats.get("bbox_cross", 0) + 1
                continue

            if not dedupe.allow(text, cx, cy):
                stats["dedup_skip"] += 1
                continue

            fp_px = _text_footprint_px(ax, cx, cy, w_m, text_h_m, angle, getattr(spec, "align", "center"), ppm)
            if collider.conflicts(fp_px):
                stats["overlap_skip"] = stats.get("overlap_skip", 0) + 1
                continue

            txt = _draw_label(ax, cx, cy, text, spec)
            if txt:
                txt.set_rotation(angle)
                collider.add(fp_px)
                stats["placed"] += 1
                stats["attempted"] += 1
                placed_here = True

        if not placed_here:
            stats["no_window"] += 1
            stats["attempted"] += 1
