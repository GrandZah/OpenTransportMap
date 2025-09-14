from typing import Union
import geopandas as gpd
from pyproj import CRS


def ensure_epsg4326(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject the GeoDataFrame to EPSG:4326 if it uses a different CRS."""
    return ensure_crs(gdf, 4326)


def ensure_crs(gdf: gpd.GeoDataFrame, target_crs: Union[int, str, CRS]) -> gpd.GeoDataFrame:
    """Ensure the GeoDataFrame has a valid CRS and reproject it to the target CRS if it differs."""
    if gdf.crs is None:
        raise ValueError(f"GeoDataFrame without CRS (expected {target_crs}).")

    target = CRS.from_user_input(target_crs)
    return gdf if CRS(gdf.crs).equals(target) else gdf.to_crs(target)
