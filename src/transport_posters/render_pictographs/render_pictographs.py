import logging
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import geopandas as gpd


from transport_posters.load_configs import CONFIG_PATHS, CONFIG_RENDER, CRSLike
from transport_posters.logger import log_function_call

logger = logging.getLogger(__name__)


@log_function_call
def render_pictographs(ax, df_pictographs, size=250):
    """
    Render pictographs on the given Matplotlib Axes within bbox (GeoDataFrame in EPSG:3857).
    """
    for _, row in df_pictographs.iterrows():
        _draw_pictographs(ax, row,size=size)

def _draw_pictographs(ax, row: pd.Series, size: float = 250) -> None:
    """Draw a PNG image on the given axis while preserving its aspect ratio."""
    path_name = row["img_path"]
    img = _get_image(path_name)
    if img is None:
        return

    img_height_px, img_width_px = img.shape[0], img.shape[1]
    aspect = img_width_px / img_height_px

    height_m = size
    width_m = size * aspect

    x, y = row.geometry.x, row.geometry.y
    half_width = width_m / 2
    half_height = height_m / 2

    extent = [x - half_width, x + half_width, y - half_height, y + half_height]
    ax.imshow(img, extent=extent, aspect="equal", zorder=10, interpolation="bilinear")

def _get_image(path_name: str):
    full_path = Path(CONFIG_PATHS.style_pictographs_img_dir) / path_name
    if not full_path.exists():
        logger.warning(f"Not found file {full_path}")
        return None
    return plt.imread(full_path)


def _load_pictographs_gdf(csv_filename: str) -> gpd.GeoDataFrame | None:
    """Load pictographs from CSV into a GeoDataFrame in EPSG:4326. Returns None if the file does not exist."""
    df_path = Path(CONFIG_PATHS.style_pictographs_dir) / csv_filename
    if not df_path.exists():
        logger.warning(f"File not found: {df_path}")
        return None

    df = pd.read_csv(df_path)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    return gdf


def _get_bbox_geom_4326(bbox: gpd.GeoDataFrame):
    """Convert bbox to EPSG:4326 and get a single union geometry."""
    bbox_4326 = bbox.to_crs("EPSG:4326")
    return bbox_4326.union_all()


def get_df_pictographs_in_bbox(bbox: gpd.GeoDataFrame, csv_filename: str) -> gpd.GeoDataFrame | None:
    """
    Get pictographs from 'csv_filename' that are located inside the given bbox.
    Pictographs are stored in a CSV and converted to a GeoDataFrame in EPSG:4326.
    """
    gdf = _load_pictographs_gdf(csv_filename)
    if gdf is None:
        return None

    bbox_geom = _get_bbox_geom_4326(bbox)
    gdf_in_bbox = gdf[gdf.within(bbox_geom)]
    return gdf_in_bbox


def get_df_pictographs_outside_bbox(bbox: gpd.GeoDataFrame, csv_filename: str) -> gpd.GeoDataFrame | None:
    """
    Get pictographs from 'csv_filename' that are located outside the given bbox.
    Pictographs are stored in a CSV and converted to a GeoDataFrame in EPSG:4326.
    """
    gdf = _load_pictographs_gdf(csv_filename)
    if gdf is None:
        return None

    bbox_geom = _get_bbox_geom_4326(bbox)
    gdf_outside_bbox = gdf[~gdf.within(bbox_geom)]
    return gdf_outside_bbox

def reproject_to_local_projection(df_pictographs: gpd.GeoDataFrame, target_crs: CRSLike) -> gpd.GeoDataFrame:
    """
    Reproject Point in df_pictographs to target_crs.
    """
    return df_pictographs.to_crs(target_crs)