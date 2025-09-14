import shapely
import geopandas as gpd
from typing import Optional
from matplotlib import pyplot as plt


def points_per_meter(ax: plt.Axes) -> float:
    """Return the average number of display points per meter for the given axis."""
    fig = ax.figure
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    data_w_m = float(abs(x1 - x0))
    data_h_m = float(abs(y1 - y0))
    if data_w_m <= 0 or data_h_m <= 0:
        return 0.0

    bbox = ax.get_position()
    fig_w_in, fig_h_in = fig.get_size_inches()
    width_pts = bbox.width * fig_w_in * 72.0
    height_pts = bbox.height * fig_h_in * 72.0

    ppm_x = width_pts / data_w_m
    ppm_y = height_pts / data_h_m
    return (ppm_x + ppm_y) * 0.5


def meters_to_points(ax: plt.Axes, meters: float, ppm: Optional[float] = None) -> float:
    """Convert meters to display points using cached or computed ppm value."""
    if ppm is None:
        ppm = points_per_meter(ax)
    return meters * ppm


def meters_to_px(ax: plt.Axes, m: float, ppm: Optional[float] = None) -> float:
    """Convert meters to pixels for the axis using display resolution."""
    pts = meters_to_points(ax, m, ppm=ppm)
    return pts * ax.figure.dpi / 72.0


def fit_bbox_to_aspect(bbox_gdf: gpd.GeoDataFrame, aspect: float, bleed: float = 0.0) -> gpd.GeoDataFrame:
    """Return a bounding box adjusted to the target aspect ratio with optional bleed."""
    minx, miny, maxx, maxy = map(float, bbox_gdf.total_bounds)
    w, h = (maxx - minx), (maxy - miny)
    data_aspect = w / h
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2

    if data_aspect > aspect:
        target_h = w / aspect
        add = (target_h - h) / 2
        miny, maxy = cy - h / 2 - add, cy + h / 2 + add
    else:
        target_w = h * aspect
        add = (target_w - w) / 2
        minx, maxx = cx - w / 2 - add, cx + w / 2 + add

    if bleed > 0:
        dx, dy = (maxx - minx) * bleed, (maxy - miny) * bleed
        minx, maxx, miny, maxy = minx - dx, maxx + dx, miny - dy, maxy + dy

    return gpd.GeoDataFrame(geometry=[shapely.geometry.box(minx, miny, maxx, maxy)], crs=bbox_gdf.crs)
