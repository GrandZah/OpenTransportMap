import numbers
from typing import Dict, Union
import geopandas as gpd
from dataclasses import dataclass, field
import pandas as pd
from pyproj import CRS

CRSLike = Union[int, str, dict, CRS]


@dataclass
class CityRouteDatabase:
    """
    stops_gdf: {
        "stop_id":int, OSM id stop
        "name":str, name of stop
        "short_name":str, shorten name of stop by utils/util_short_name
        "routes":set, route_ids which stopped in this stops
        "geometry":Point, place stop}

    platforms_gdf: {
        "stop_id":int, OSM id stop
        "name":str, name of stop
        "geometry":Point, place platform}

    routes_gdf: {
        "route_id":int, OSM id route
        "ref":str, name of route
        "stop_seq":List[int], sequence of stop_ids on which transport stops, in correct order
        "geometry":LineString, geometry of the route}

    edges_gdf:{
        "edge_id":str, f"{route_id}_{idx}" - in route with id route_id. Part of this route - between stops - idx.
        "route_id":int, OSM id route
        "seq_from":int, stop_id of stop from which this edge
        "seq_to":int, stop_id of stop to which this edge
        "edge_idx":int, edge id in this route
        "length_m":int, length of edge part in meters
        "shape_dist":int, cumulative length of route in meters
        "geometry", geometry of edge part}
    """
    stops_gdf: gpd.GeoDataFrame
    platforms_gdf: gpd.GeoDataFrame
    routes_gdf: gpd.GeoDataFrame
    edges_gdf: gpd.GeoDataFrame
    platforms_map: Dict[int, gpd.GeoDataFrame] = field(init=False)
    routes_map: Dict[int, gpd.GeoDataFrame] = field(init=False)
    edges_map: Dict[int, gpd.GeoDataFrame] = field(init=False)
    id2ref: Dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        self.__rebuild_maps()

    def reproject_all(
            self,
            target_crs: CRSLike,
            *,
            inplace: bool = False,
            allow_none: bool = False
    ) -> "CityRouteDatabase":
        """
        Reproject all layers (stops, platforms, routes, edges) into the target CRS.

        Parameters
        ----------
        target_crs : int | str | dict | pyproj.CRS
            Target CRS (e.g., 3857, "EPSG:3857", CRS.from_epsg(3857)).
        inplace : bool, default False
            If True, modifies this object in place and returns it.
            If False, returns a new CityRouteDatabase instance.
        allow_none : bool, default False
            If a GeoDataFrame has crs == None:
              - False: raises ValueError.
              - True: assigns the target CRS without reprojection.

        Returns
        -------
        CityRouteDatabase
        """
        tgt = CRS.from_user_input(target_crs)

        def _convert(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
            if gdf.crs is None:
                if not allow_none:
                    raise ValueError(
                        "One of the GeoDataFrames has crs=None. "
                        "Define the source CRS or call to_crs(..., allow_none=True) "
                        "to assign the target CRS without reprojection."
                    )
                return gpd.GeoDataFrame(gdf, geometry=gdf.geometry.name, crs=tgt)
            if CRS.from_user_input(gdf.crs) == tgt:
                return gdf
            return gdf.to_crs(tgt)

        if inplace:
            self.stops_gdf = _convert(self.stops_gdf)
            self.platforms_gdf = _convert(self.platforms_gdf)
            self.routes_gdf = _convert(self.routes_gdf)
            self.edges_gdf = _convert(self.edges_gdf)
            self.__rebuild_maps()
            return self

        return CityRouteDatabase(
            stops_gdf=_convert(self.stops_gdf),
            platforms_gdf=_convert(self.platforms_gdf),
            routes_gdf=_convert(self.routes_gdf),
            edges_gdf=_convert(self.edges_gdf),
        )

    def __rebuild_maps(self) -> None:
        def _int_key(key: object, col: str) -> int:
            if key is None or (isinstance(key, float) and pd.isna(key)):
                raise ValueError(f"{col}: NaN/None is not allowed as a group key")
            if isinstance(key, numbers.Integral):
                return int(key)
            if isinstance(key, float) and key.is_integer():
                return int(key)
            if isinstance(key, str) and key.isdigit():
                return int(key)
            raise TypeError(f"{col} must be int-like, got {type(key).__name__}: {key!r}")

        platforms_map: dict[int, gpd.GeoDataFrame] = {}
        for key, df in self.platforms_gdf.groupby("stop_id", sort=False, dropna=False):
            ikey: int = _int_key(key, "stop_id")
            platforms_map[ikey] = gpd.GeoDataFrame(df.copy(), geometry=df.geometry.name, crs=df.crs)
        self.platforms_map = platforms_map

        routes_map: dict[int, gpd.GeoDataFrame] = {}
        for key, df in self.routes_gdf.groupby("route_id", sort=False, dropna=False):
            ikey: int = _int_key(key, "route_id")
            routes_map[ikey] = gpd.GeoDataFrame(df.copy(), geometry=df.geometry.name, crs=df.crs)
        self.routes_map = routes_map

        edges_map: dict[int, gpd.GeoDataFrame] = {}
        for key, df in self.edges_gdf.groupby("route_id", sort=False, dropna=False):
            ikey: int = _int_key(key, "route_id")
            edges_map[ikey] = gpd.GeoDataFrame(df.copy(), geometry=df.geometry.name, crs=df.crs)
        self.edges_map = edges_map

        if "route_id" not in self.routes_gdf or "ref" not in self.routes_gdf:
            raise KeyError("routes_gdf must contain 'route_id' and 'ref' columns")

        route_ids = pd.to_numeric(self.routes_gdf["route_id"], errors="raise").astype("int64")
        refs = self.routes_gdf["ref"].astype("string")
        self.id2ref = dict(zip(route_ids.tolist(), refs.tolist()))
