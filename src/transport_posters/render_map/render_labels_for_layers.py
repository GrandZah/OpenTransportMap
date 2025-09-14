from shapely.ops import unary_union
import matplotlib.pyplot as plt
import logging
from typing import Optional
import geopandas as gpd

from data_map.get_layers import LayersMap, GeoLayer
from data_map.get_style_layers import LabelSpec
from logger import log_function_call
from render_map.render_labels_lines import label_lines
from render_map.render_labels_points_polygons import label_points, label_polygons
from render_map.utils_text_label import _LabelCollider
from utils.utils_rendering import points_per_meter

logger = logging.getLogger(__name__)


def _require_field(gdf: gpd.GeoDataFrame, field: str, layer_name: str):
    if field not in gdf.columns:
        raise KeyError(f"label({layer_name}): omitted field '{field}' in columns {list(gdf.columns)[:20]}")


def _validation_data(spec: Optional[LabelSpec], layer: GeoLayer, bbox_gdf: gpd.GeoDataFrame, layer_name: str = "?"):
    if spec is None:
        raise RuntimeError("spec is None")

    gdf: Optional[gpd.GeoDataFrame] = getattr(layer, "gdf", None)
    if gdf is None or gdf.empty:
        raise RuntimeError(f"label: gdf empty")

    _require_field(gdf, spec.field, layer_name)

    clipped = gdf.clip(bbox_gdf)
    if clipped.empty:
        raise RuntimeError(f"label: clipped empty")
    return clipped


def _render_labels_for_layer(ax: plt.Axes, layer: GeoLayer, bbox_gdf: gpd.GeoDataFrame,
                             layer_name: str = "?", *, ppm: float, collider: _LabelCollider):
    spec: Optional[LabelSpec] = getattr(layer, "label", None)
    try:
        clipped = _validation_data(spec, layer, bbox_gdf, layer_name)
    except Exception as e:
        logger.warning(f"Skip layer {layer_name}, cause:{e}")
        return

    bbox_geom = unary_union(bbox_gdf.geometry)

    placement = spec.placement
    if placement == "auto":
        placement = "line" if layer.gtype == "line" else ("centroid" if layer.gtype == "polygon" else "point")

    stats = {
        "attempted": 0, "placed": 0, "empty_geom": 0, "empty_text": 0, "not_linestr": 0,
        "too_short_m": 0, "no_window": 0, "dedup_skip": 0, "bbox_cross": 0
    }

    logger.debug("label(%s): placement=%s field=%s features=%d", layer_name, placement, spec.field, len(clipped))

    match placement:
        case "line":
            label_lines(ax, clipped, spec, spec.field, stats, bbox_geom, ppm=ppm, collider=collider)
        case "point":
            label_points(ax, clipped, spec, spec.field, stats, bbox_geom, ppm=ppm, collider=collider)
        case "centroid" | "polygon":
            label_polygons(ax, clipped, spec, spec.field, stats, bbox_geom, ppm=ppm, collider=collider)
        case _:
            logger.warning("unknown placement %s", placement)
            return

    logger.info("label(%s): stats=%s", layer_name, stats)


@log_function_call
def render_labels_for_layers(ax: plt.Axes, layers: LayersMap, bbox_gdf: gpd.GeoDataFrame,
                             forbidden_px: list | None = None):
    """Render text labels for map layers within the specified bounding box."""
    ppm = points_per_meter(ax)
    collider = _LabelCollider()

    if forbidden_px:
        for poly_px in forbidden_px:
            collider.add(poly_px)

    items = [(name, layer) for name, layer in layers.items() if getattr(layer, "label", None)]
    items.sort(key=lambda kv: getattr(kv[1].label, "zorder", 0), reverse=True)

    for name, layer in items:
        _render_labels_for_layer(ax, layer, bbox_gdf, layer_name=name, ppm=ppm, collider=collider)
