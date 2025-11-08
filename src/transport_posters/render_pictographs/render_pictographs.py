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
def render_pictographs(ax, df_pictographs):
    """
    Render pictographs on the given Matplotlib Axes within bbox (GeoDataFrame in EPSG:3857).
    """
    for _, row in df_pictographs.iterrows():
        _draw_pictographs(ax, row)

# def _draw_pictographs(ax, row: pd.Series) -> None:
#     """ Draw img on ax """
#     path_name = row["img_path"]
#     img = _get_image(path_name)
#     if img is None:
#         return
#     x, y = row.geometry.x, row.geometry.y
#     imagebox = OffsetImage(img, zoom=0.5)
#     ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0)
#     ax.add_artist(ab)

def _draw_pictographs(ax, row: pd.Series) -> None:
    """ Draw img on ax """
    path_name = row["img_path"]
    img = _get_image(path_name)
    if img is None:
        return
    name = row.get("name_pictograph", "").strip().rstrip(",").strip()
    x, y = row.geometry.x, row.geometry.y
    SIZE_METERS = 250
    # GAP_METERS = 10.0
    half_size = SIZE_METERS / 2
    extent = [x - half_size, x + half_size, y - half_size, y + half_size]
    ax.imshow(img, extent=extent, aspect='equal', zorder=10, interpolation='bilinear')
    # if name:
    #     text_y = y - half_size - GAP_METERS
    #     ax.text(
    #         x, text_y, name, ha='center', va='top', fontsize=9, fontweight='bold',
    #         bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="none", alpha=0.8),
    #         zorder=11, clip_on=False
    #     )

def _get_image(path_name: str):
    full_path = Path(CONFIG_PATHS.style_pictographs_img_dir) / path_name
    if not full_path.exists():
        logger.warning(f"Not found file {full_path}")
        return None
    return plt.imread(full_path)



def get_df_pictographs_in_bbox(bbox: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    """
    Get pictographs from "csv_path", which in (GeoDataFrame in EPSG:4326).
    """
    if "pictographs_csv" not in CONFIG_RENDER:
        return None
    csv_filename = CONFIG_RENDER["pictographs_csv"]
    df_path = Path(CONFIG_PATHS.style_pictographs_dir) / csv_filename
    if not df_path.exists():
        logger.warning(f"File not found: {df_path}")
        return None
    df = pd.read_csv(df_path)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    )
    bbox_4326 = bbox.to_crs("EPSG:4326")
    gdf_in_bbox = gdf[gdf.within(bbox_4326.union_all())]
    return gdf_in_bbox

def reproject_to_local_projection(df_pictographs: gpd.GeoDataFrame, target_crs: CRSLike) -> gpd.GeoDataFrame:
    """
    Reproject Point in df_pictographs to target_crs.
    """
    return df_pictographs.to_crs(target_crs)