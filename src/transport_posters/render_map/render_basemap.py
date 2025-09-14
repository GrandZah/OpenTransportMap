import logging
import os
from functools import lru_cache
from typing import Dict, Any, Optional, Iterable

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, base as shapely_base
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from PIL import Image

from data_map.get_data_map import LayersMap
from logger import log_function_call
from utils.utils_rendering import meters_to_points

logger = logging.getLogger(__name__)


def _poly_to_path(p: Polygon) -> Path:
    """Convert a shapely Polygon to a Matplotlib Path (exterior + interiors)."""
    if p.is_empty:
        return Path(np.zeros((0, 2)), np.zeros((0,), dtype=np.uint8))

    v = np.asarray(p.exterior.coords)
    c = np.full(len(v), Path.LINETO, dtype=np.uint8)
    c[0] = Path.MOVETO
    path = Path(v, c)

    for ring in p.interiors:
        vr = np.asarray(ring.coords)
        cr = np.full(len(vr), Path.LINETO, dtype=np.uint8)
        cr[0] = Path.MOVETO
        path = Path.make_compound_path(path, Path(vr, cr))

    return path


def _geoms_to_pathpatch(geoms: Iterable[shapely_base.BaseGeometry], ax, **patch_kwargs) -> PathPatch:
    """Combine (Multi)Polygons into a single compound PathPatch in data coords."""
    comp: Optional[Path] = None
    for g in geoms:
        if g is None or g.is_empty:
            continue
        if isinstance(g, Polygon):
            p = _poly_to_path(g)
        elif isinstance(g, MultiPolygon):
            pp: Optional[Path] = None
            for sub in g.geoms:
                p2 = _poly_to_path(sub)
                pp = p2 if pp is None else Path.make_compound_path(pp, p2)
            p = pp
        else:
            continue
        comp = p if comp is None else Path.make_compound_path(comp, p)

    if comp is None:
        comp = Path(np.zeros((0, 2)), np.zeros((0,), dtype=np.uint8))

    return PathPatch(comp, transform=ax.transData, **patch_kwargs)


@lru_cache(maxsize=32)
def _load_rgba_image_cached(path: str, mtime: float) -> Image.Image:
    """Load image from disk, convert to RGBA. Cached by (path, mtime)."""
    return Image.open(path).convert("RGBA")


def _load_rgba_image(path: str) -> Image.Image:
    """Public loader: uses file mtime in cache key and returns a *copy*."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        logger.exception("Failed to stat image file: %s", path)
        raise
    return _load_rgba_image_cached(path, mtime).copy()


def _draw_pattern_image_for_layer(
        ax,
        geoms: Iterable[shapely_base.BaseGeometry],
        image_path: str,
        *,
        px_per_meter: float = 1.0,
        angle: float = 0.0,
        alpha: float = 0.9,
        zorder: float = 0,
        oversample: float = 1.10,
        max_dim: int = 8192,
        max_pixels: int = 32_000_000,
) -> None:
    """
    Render a repeated image pattern for a *whole layer* with a single imshow.

    Strategy:
      1) Compute union bbox of `geoms` → (minx, miny, maxx, maxy) in EPSG:3857 (meters).
      2) Choose output raster size (pixels) via `px_per_meter` with sane caps.
      3) Make a *single* tiled RGBA image to exactly cover that bbox.
      4) ax.imshow(..., extent=...) once; then clip by a *single* compound PathPatch.
    """
    polys: list[shapely_base.BaseGeometry] = []
    minx = miny = maxx = maxy = None
    for g in geoms:
        if g is None or g.is_empty:
            continue
        if not isinstance(g, (Polygon, MultiPolygon)):
            continue
        gx0, gy0, gx1, gy1 = g.bounds
        if not np.isfinite([gx0, gy0, gx1, gy1]).all() or gx0 == gx1 or gy0 == gy1:
            continue
        polys.append(g)
        if minx is None:
            minx, miny, maxx, maxy = gx0, gy0, gx1, gy1
        else:
            minx = min(minx, gx0)
            miny = min(miny, gy0)
            maxx = max(maxx, gx1)
            maxy = max(maxy, gy1)

    if not polys:
        return

    (x0p, y0p) = ax.transData.transform((minx, miny))
    (x1p, y1p) = ax.transData.transform((maxx, maxy))
    W_disp = int(max(2, abs(x1p - x0p) * oversample))
    H_disp = int(max(2, abs(y1p - y0p) * oversample))

    width_m = maxx - minx
    height_m = maxy - miny
    out_W = int(max(2, min(max_dim, max(W_disp, width_m * px_per_meter))))
    out_H = int(max(2, min(max_dim, max(H_disp, height_m * px_per_meter))))

    if out_W * out_H > max_pixels:
        scale = (max_pixels / (out_W * out_H)) ** 0.5
        out_W = max(2, int(out_W * scale))
        out_H = max(2, int(out_H * scale))

    base_img = _load_rgba_image(image_path)
    if angle:
        base_img = base_img.rotate(float(angle), expand=True, resample=Image.Resampling.BICUBIC)

    tile_w_px, tile_h_px = base_img.width, base_img.height
    tile_w_px = min(tile_w_px, out_W)
    tile_h_px = min(tile_h_px, out_H)
    if tile_w_px < 1 or tile_h_px < 1:
        return

    if (base_img.width, base_img.height) != (tile_w_px, tile_h_px):
        base_img = base_img.resize((tile_w_px, tile_h_px), resample=Image.Resampling.BICUBIC)
    tile_arr = np.asarray(base_img)
    ny = (out_H + tile_arr.shape[0] - 1) // tile_arr.shape[0]
    nx = (out_W + tile_arr.shape[1] - 1) // tile_arr.shape[1]
    big = np.tile(tile_arr, (ny, nx, 1))[:out_H, :out_W, :]

    im = ax.imshow(
        big,
        extent=[minx, maxx, miny, maxy],
        origin="lower",
        alpha=float(alpha),
        zorder=float(zorder),
        interpolation="nearest",
    )

    clip_patch = _geoms_to_pathpatch(polys, ax, facecolor="none")
    im.set_clip_path(clip_patch.get_path(), clip_patch.get_transform())


@log_function_call
def render_basemap(ax, layers: LayersMap, bbox: gpd.GeoDataFrame):
    """
    Render layers on the given Matplotlib Axes within bbox (GeoDataFrame in EPSG:3857).
    Supports layer.gtype in {"polygon","line","point"} with optional polygon fill modes.
    """
    for name, geo_layers in layers.items():
        gdf = geo_layers.gdf
        if gdf is None or gdf.empty:
            continue

        clipped = gdf.clip(bbox)
        if clipped.empty:
            continue

        color = geo_layers.color
        gtype = (geo_layers.gtype or "").lower()
        alpha = float(geo_layers.alpha if geo_layers.alpha is not None else 1.0)
        zorder = float(geo_layers.zorder if geo_layers.zorder is not None else 0.0)

        match gtype:
            case "polygon":
                fill: Optional[Dict[str, Any]] = getattr(geo_layers, "fill", None)
                edgecolor = getattr(geo_layers, "edgecolor", None) or "none"
                edgewidth = float(getattr(geo_layers, "edgewidth", None) or 0.0)

                mode = (fill or {}).get("mode", "solid")
                if not fill or mode == "solid":
                    clipped.plot(ax=ax, facecolor=color, alpha=alpha,
                                 edgecolor=edgecolor, linewidth=edgewidth, zorder=zorder)
                    continue
                if mode == "empty":
                    if edgecolor != "none" and edgewidth > 0:
                        clipped.plot(ax=ax, facecolor="none", edgecolor=edgecolor,
                                     linewidth=edgewidth, zorder=zorder)
                    continue

                if edgecolor != "none" and edgewidth > 0:
                    clipped.plot(ax=ax, facecolor="none", edgecolor=edgecolor,
                                 linewidth=edgewidth, zorder=zorder + 0.1)

                if mode in {"pattern_image", "pattern", "image"}:
                    img_path = (fill.get("path") or fill.get("image") or fill.get("img"))
                    if not img_path:
                        logger.warning("Pattern mode without image path in layer %s", name)
                        continue
                    ppm = float(fill.get("px_per_meter") or fill.get("ppm") or 1.0)
                    angle = float(fill.get("angle", 0.0))
                    alpha_fill = float(fill.get("alpha", alpha))
                    try:
                        _draw_pattern_image_for_layer(
                            ax,
                            clipped.geometry,
                            image_path=img_path,
                            px_per_meter=ppm,
                            angle=angle,
                            alpha=alpha_fill,
                            zorder=zorder,
                        )
                    except Exception:
                        logger.exception("Pattern fill failed for layer %s", name)
                else:
                    raise ValueError(f"Unknown fill mode: {mode!r}")

            case "line":
                lw = meters_to_points(ax, geo_layers.linewidth or 5.0)
                clipped.plot(ax=ax, linewidth=lw, color=color, alpha=alpha, zorder=zorder)

            case "point":
                pass

            case _:
                raise ValueError(f"Unknown gtype: {gtype!r}")
