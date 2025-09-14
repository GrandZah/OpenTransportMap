import numpy as np
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, Wedge
from shapely.geometry import Point


def _arc_xy(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 900):
    """Return arc coordinates from angle a0 to a1 in radians."""
    t = np.linspace(a0, a1, n, endpoint=True)
    return cx + r * np.cos(t), cy + r * np.sin(t)


def _plot_dashed_arc(ax, cx, cy, r, a0, a1, *,
                     lw=1.0, color="#2d5f8a", alpha=1.0, zorder=2.0,
                     dash_on_off=(28.0, 18.0), dash_offset=0.0):
    """Draw a dashed circular arc with an optional phase offset."""
    x, y = _arc_xy(cx, cy, r, a0, a1)
    (ln,) = ax.plot(x, y, color=color, lw=lw, alpha=alpha,
                    zorder=zorder, solid_capstyle="round")

    ln.set_linestyle((dash_offset, dash_on_off))
    return ln


def _text_halo(txt, lw=2.2, fg="white", alpha=0.96):
    txt.set_path_effects([pe.Stroke(linewidth=lw, foreground=fg, alpha=alpha), pe.Normal()])


def render_walk_5min_focus_gap(
        ax,
        stop_row,
        minutes: float = 5.0,
        walk_speed_mps: float = 1.35,
        color: str = "#2d5f8a",
        inside_alpha: float = 0.055,
        outer_glow_levels: int = 5,
        outer_glow_span: float = 0.22,
        outer_glow_alpha_max: float = 0.06,
        dash_on_off=(28.0, 18.0),
        dash_offset: float = 0.0,
        gap_center_deg: float = 90.0,
        gap_width_deg: float = 28.0,
        label_text: str = "~5 minutes",
):
    """
    Focus mask 5 minutes:
    - outer long dotted line with white casing;
    - clean break under the signature (no dotted line in the window);
    - inside — light white veil; outside — soft dark glow.
    """
    pt: Point = stop_row.geometry
    cx, cy = float(pt.x), float(pt.y)
    r = float(minutes) * 60.0 * float(walk_speed_mps)

    WASH_Z, OUT_Z, LABEL_Z = 1.62, 2.02, 3.45

    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor="none",
                        alpha=inside_alpha, zorder=WASH_Z))

    r_out_max = r * (1.0 + outer_glow_span)
    radii = np.linspace(r, r_out_max, outer_glow_levels + 1)
    for i in range(outer_glow_levels):
        rin, rout = radii[i], radii[i + 1]
        a = outer_glow_alpha_max * (1.0 - i / outer_glow_levels)
        ax.add_patch(Wedge((cx, cy), rout, 0, 360, width=rout - rin,
                           facecolor="black", edgecolor="none",
                           alpha=a, zorder=WASH_Z))

    c = np.deg2rad(gap_center_deg)
    h = np.deg2rad(gap_width_deg / 2.0)

    _plot_dashed_arc(
        ax,
        cx,
        cy,
        r,
        c + h,
        c - h + 2 * np.pi,
        lw=2.6,
        color="white",
        alpha=0.92,
        zorder=OUT_Z - 0.001,
        dash_on_off=dash_on_off,
        dash_offset=dash_offset,
    )

    _plot_dashed_arc(
        ax,
        cx,
        cy,
        r,
        c + h,
        c - h + 2 * np.pi,
        lw=1.1,
        color=color,
        alpha=0.72,
        zorder=OUT_Z,
        dash_on_off=dash_on_off,
        dash_offset=dash_offset,
    )

    tx, ty = cx + r * np.cos(c), cy + r * np.sin(c)
    txt = ax.text(tx, ty, label_text,
                  ha="center", va="center",
                  fontsize=10, color=color, zorder=LABEL_Z,
                  rotation=0, rotation_mode="anchor")
    _text_halo(txt)

    ax.set_aspect("equal")
