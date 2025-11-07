import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

from matplotlib.font_manager import FontProperties

from transport_posters.data_transport.get_bus_layers import CONFIG_PATHS
from transport_posters.logger import log_function_call

logger = logging.getLogger(__name__)


@dataclass
class LabelSpec:
    """
    Information about rules for layer labels
    """
    field: str = "name"
    placement: str = "auto"
    size_pt: float = 10.0
    color: str = "#0b1d2a"
    halo_color: str = "#ffffff"
    halo_width: float = 2.0
    fontproperties: FontProperties = FontProperties(family="DejaVu Sans")
    weight: str = "regular"
    italic: bool = False
    alpha: float = 1.0
    zorder: float = 3.0
    min_len_m: float = 80.0
    max_angle: float = 90.0
    align: str = "center"
    repeat_m: float = 1200.0
    straight_tol_m: float = 6.0


@dataclass
class BaseLayer:
    """Base information about a layer."""
    color: str
    gtype: str
    zorder: float
    linewidth: Optional[float] = field(default=None, kw_only=True)
    alpha: Optional[float] = field(default=None, kw_only=True)
    fill: Optional[Dict[str, Any]] = field(default=None, kw_only=True)
    label: Optional[LabelSpec] = field(default=None, kw_only=True)


@dataclass
class StyleGeoLayer(BaseLayer):
    source: str
    tags: Dict[str, Any]
    custom_filter: Optional[str]


ALLOWED_TYPES = {
    "point": {"Point", "MultiPoint"},
    "line": {"LineString", "MultiLineString"},
    "polygon": {"Polygon", "MultiPolygon"},
}

_ALLOWED_GTYPES = ALLOWED_TYPES.keys()

StyleLayersMap = Dict[str, StyleGeoLayer]


@log_function_call
def get_style_layers(style_path: Path) -> StyleLayersMap:
    """
    Get StyleLayersMap in style_path - expected JSON file
    """
    try:
        text = style_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        folder = style_path.parent
        files = ", ".join(f.name for f in folder.iterdir() if f.is_file())
        logger.error(
            f"File with style name «{style_path.name}» not found. Available styles in folder: «{folder}»: {files}")
        raise

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Error while parsing JSON in «{style_path}»: {e}")
        raise

    layers: StyleLayersMap = {}
    style_name = style_path.name
    for source, group in raw.items():
        if not isinstance(group, dict):
            continue
        for name, entry in group.items():
            layer_name = style_name + "_" + name

            if not isinstance(entry, dict):
                logger.warning(f"In style file {style_name} should be dict type for name: {name}")
                continue
            if not {"tags", "color", "gtype", "zorder"}.issubset(entry):
                logger.warning(f"In style file {style_name} should be tags, color, gtype, zorder for name: {name}")
                continue
            gtype = str(entry["gtype"]).lower()
            if gtype not in _ALLOWED_GTYPES:
                logger.warning(f"In style file {style_name} unknown gtype: {gtype}")
                continue
            line_width = float((entry["width"]) if "width" in entry else 1) if gtype == "line" else None
            layers[layer_name] = StyleGeoLayer(
                source=source,
                tags=entry["tags"],
                color=str(entry["color"]),
                gtype=gtype,
                zorder=float(entry["zorder"]),
                linewidth=line_width,
                alpha=entry.get("alpha"),
                fill=entry.get("fill"),
                custom_filter=entry.get("custom_filter"),
                label=_parse_label(entry.get("label")))

    logger.info(f"Load style in path {style_path}")
    _resolve_asset_paths(layers)
    return layers


def _resolve_asset_paths(layers: StyleLayersMap) -> None:
    """
    In JSON files we contain path - for example - "path": "assets/patterns/grass_lawn_01.png", so this function resolve
    this paths to real by adding path CONFIG_PATHS.style_dir to begin of path.
    """
    base = CONFIG_PATHS.style_dir
    for lyr in layers.values():
        if lyr.fill and lyr.fill.get("mode") == "pattern_image":
            p = lyr.fill.get("path")
            if p:
                p = Path(p)
                if not p.is_absolute():
                    p = (base / p).resolve()
                lyr.fill["path"] = str(p)
        if lyr.fill and lyr.fill.get("icon_path"):
            p = Path(lyr.fill["icon_path"])
            if not p.is_absolute():
                p = (base / p).resolve()
            lyr.fill["icon_path"] = str(p)


def _parse_fontproperties(entry: Optional[Dict[str, Any]]) -> FontProperties:
    base = CONFIG_PATHS.style_dir

    path_font = entry.get("fontproperties", None)
    if path_font:
        path_font = Path(str(path_font))
        if not path_font.is_absolute():
            path_font = (base / path_font).resolve()
        if not path_font.is_file():
            logger.warning(f"Font not found: {path_font}")
            return FontProperties(family="DejaVu Sans")

    return FontProperties(fname=path_font)


def _parse_label(entry: Optional[Dict[str, Any]]) -> Optional[LabelSpec]:
    if not entry or not isinstance(entry, dict):
        return None
    try:
        return LabelSpec(
            field=str(entry.get("field", "name")),
            placement=str(entry.get("placement", "auto")),
            size_pt=float(entry.get("size_pt", 10)),
            color=str(entry.get("color", "#0b1d2a")),
            halo_color=str(entry.get("halo_color", "#ffffff")),
            halo_width=float(entry.get("halo_width", 2.0)),
            fontproperties=_parse_fontproperties(entry),
            weight=str(entry.get("weight", "regular")),
            italic=bool(entry.get("italic", False)),
            alpha=float(entry.get("alpha", 1.0)),
            zorder=float(entry.get("zorder", 3.0)),
            min_len_m=float(entry.get("min_len_m", 80.0)),
            max_angle=float(entry.get("max_angle", 90.0)),
            align=str(entry.get("align", "center")),
        )
    except Exception as e:
        logger.warning(f"Label parse error: {e}")
        return None
