from typing import List, Dict, Tuple
from dataclasses import dataclass
import matplotlib.font_manager as fm
from load_configs import FONT_INTER_BOLD
from utils.utils import natural_key


@dataclass(frozen=True)
class RenderStopConfig:
    """Rendering parameters for a stop symbol and its labels."""

    STOP_SIZE: float = 45
    STOP_COLOR: str = "#34a853"
    FONT_SIZE_LABEL: float = 20
    FONT_SIZE_BUS: float = 12
    LABEL_OFFSET_PTS: float = 0.3
    BUS_GAP_PTS: float = 3.0

    BUS_BOX_STYLE: str = "round"
    BUS_BOX_PAD_FRAC: float = 0.2
    BUS_BOX_FACE: str = "white"
    BUS_BOX_EDGE: str = "gray"
    BUS_BOX_ALPHA: float = 0.8
    BUS_BOX_LINEWIDTH: float = 0.8


config_render_stop = RenderStopConfig()
custom_font = fm.FontProperties(fname=FONT_INTER_BOLD)


def _bus_box_align(ox: float, ha: str) -> Tuple[float, dict]:
    fs = config_render_stop.FONT_SIZE_BUS
    pad_frac = config_render_stop.BUS_BOX_PAD_FRAC
    lw = float(config_render_stop.BUS_BOX_LINEWIDTH)
    shift = pad_frac * fs + 0.5 * lw
    ox2 = ox + shift if ha == "left" else ox - shift if ha == "right" else ox
    bbox_kw = {
        "boxstyle": f"{config_render_stop.BUS_BOX_STYLE},pad={pad_frac}",
        "facecolor": config_render_stop.BUS_BOX_FACE,
        "edgecolor": config_render_stop.BUS_BOX_EDGE,
        "alpha": config_render_stop.BUS_BOX_ALPHA,
        "linewidth": lw,
    }
    return ox2, bbox_kw


def _compute_label_offset(platform_pt, stop_pt, offset_pts: float) -> Tuple[float, float, str, str]:
    dx = stop_pt.x - platform_pt.x
    side = 1 if dx >= 0 else -1
    ha = "left" if side > 0 else "right"
    return side * offset_pts, 0.0, ha, "center"


def render_stop_point(ax, cur_stop_pt, platform_pt,
                      cur_stop_name: str, id2ref: Dict[int, str],
                      stop_route_ids: List[int], cur_stop_route_ids: List[int],
                      cur_stop_colour: str = "white") -> None:
    """Render a single stop with its label and route numbers."""
    x0, y0 = cur_stop_pt.x, cur_stop_pt.y

    gap = max(2.0, config_render_stop.LABEL_OFFSET_PTS * config_render_stop.FONT_SIZE_LABEL)
    ox, oy, ha, va = _compute_label_offset(platform_pt, cur_stop_pt, gap)

    ax.scatter(x0, y0, s=config_render_stop.STOP_SIZE,
               facecolor=cur_stop_colour, edgecolor=config_render_stop.STOP_COLOR, zorder=5)

    ax.annotate(cur_stop_name, xy=(x0, y0), xytext=(ox, oy), textcoords="offset points",
                fontsize=config_render_stop.FONT_SIZE_LABEL, fontproperties=custom_font,
                ha=ha, va=va, zorder=6, clip_on=True)

    bus_nums = [id2ref.get(rid, str(rid)) for rid in set(stop_route_ids) & set(cur_stop_route_ids)]
    if not bus_nums:
        return

    sorted_bus = ", ".join(sorted(set(bus_nums), key=natural_key))
    down = 0.5 * config_render_stop.FONT_SIZE_LABEL + config_render_stop.BUS_GAP_PTS
    ox_bus, bbox_kw = _bus_box_align(ox, ha)

    ax.annotate(sorted_bus, xy=(x0, y0), xytext=(ox_bus, -down), textcoords="offset points",
                fontsize=config_render_stop.FONT_SIZE_BUS, fontproperties=custom_font,
                ha=ha, va="top", zorder=6, bbox=bbox_kw, clip_on=True)
