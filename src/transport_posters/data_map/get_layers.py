import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable, Any, Union
import time
import geopandas as gpd
import osmnx as ox
from osmnx._errors import InsufficientResponseError
from pyproj import CRS
from shapely.geometry import box
import math
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc

from transport_posters.logger import log_function_call
from transport_posters.utils.utils_projection import ensure_epsg4326
from .get_style_layers import StyleLayersMap, StyleGeoLayer, BaseLayer, ALLOWED_TYPES

logger = logging.getLogger(__name__)
_PARQUET_TILE_Z = 12


def _fetch_osmnx(region, *, style: "StyleGeoLayer", name: str, **_kwargs) -> Optional[gpd.GeoDataFrame]:
    try:
        g = ox.features_from_polygon(region, style.tags)
        if g is not None and not g.empty:
            logger.info(f"Layer '{name}': fetched from osmnx")
            return g
        logger.info(f"Layer '{name}': osmnx returned empty")
        return None
    except Exception as e:
        logger.warning(f"Error while fetching layer from polygon in layer:{name}, with error :{e}")
        return None


def _fetch_osmnx_edges(region, *, style: "StyleGeoLayer", name: str, **_kwargs) -> Optional[gpd.GeoDataFrame]:
    cf = getattr(style, "custom_filter", None)
    if not cf:
        logger.warning(f"Layer '{name}': custom_filter is not provided for source 'ox.graph_edges_from_polygon'")
        return None
    try:
        G = ox.graph_from_polygon(region, custom_filter=cf, simplify=False, retain_all=True)
        if G is None:
            logger.info(f"Layer '{name}': graph_from_polygon returned None")
            return None
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        if edges is not None and not edges.empty:
            logger.info(f"Layer '{name}': fetched edges from graph (n={len(edges)})")
            return edges
        logger.info(f"Layer '{name}': edges GeoDataFrame is empty")
        return None
    except Exception as e:
        logger.warning(f"Layer '{name}': graph edges fetch failed: {e}")
        return None


def _fetch_by_area(_region, *, style: "StyleGeoLayer", name: str,
                   area_id: Optional[int] = None, req_poly=None, **_kwargs) -> Optional[gpd.GeoDataFrame]:
    if area_id is None:
        logger.info(f"Layer '{name}': skipped, area_id not provided")
        return None
    try:
        g = load_by_area_id(area_id, style.gtype)
        if g is None or g.empty:
            logger.info(f"Layer '{name}': by_area_id returned empty")
            return None
        try:
            if req_poly is not None:
                g = gpd.clip(g, req_poly)
        except Exception as e:
            logger.warning(f"Layer '{name}': clip failed: {e}")
        logger.info(f"Layer '{name}': fetched by area_id")
        return g
    except Exception as e:
        logger.warning(f"Layer '{name}': failed load_by_area_id: {e}")
        return None


SOURCE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ox.features_from_bbox": {
        "fetcher": _fetch_osmnx,
        "unique_id": "osm_id",
    },
    "ox.graph_edges_from_polygon": {
        "fetcher": _fetch_osmnx_edges,
        "unique_id": "osmid",
    },
    "by_area_id": {
        "fetcher": _fetch_by_area,
        "unique_id": "id",
    },
}

CRSLike = Union[int, str, dict, CRS]


@dataclass
class GeoLayer(BaseLayer):
    """
    Contains gdf of layer and data from BaseLayer
    """
    gdf: gpd.GeoDataFrame

    @classmethod
    def from_style(cls, style_layer: StyleGeoLayer, gdf: gpd.GeoDataFrame) -> "GeoLayer":
        base_names = BaseLayer.__annotations__.keys()
        base_kwargs = {name: getattr(style_layer, name) for name in base_names}
        return cls(gdf=gdf, **base_kwargs)

    def reproject(self, target_crs: CRSLike, *, inplace: bool = False, allow_none: bool = False) -> "GeoLayer":
        tgt = CRS.from_user_input(target_crs)
        gdf = self.gdf
        if gdf.crs is None:
            if not allow_none:
                raise ValueError(
                    "GeoLayer.gdf has crs=None. Define source CRS or use allow_none=True to assign without transformation.")
            new_gdf = gpd.GeoDataFrame(gdf, geometry=gdf.geometry.name, crs=tgt)
        else:
            src = CRS.from_user_input(gdf.crs)
            new_gdf = gdf if src == tgt else gdf.to_crs(tgt)
        if inplace:
            self.gdf = new_gdf
            return self
        return replace(self, gdf=new_gdf)


LayersMap = Dict[str, GeoLayer]


def reproject_all(layers: LayersMap, target_crs: CRSLike, *, inplace: bool = False,
                  allow_none: bool = False) -> LayersMap:
    """
    Reproject all layers to target_crs with parameters inplace and allow_none.
    """
    if inplace:
        for layer in layers.values():
            layer.reproject(target_crs, inplace=True, allow_none=allow_none)
        return layers
    return {name: layer.reproject(target_crs, inplace=False, allow_none=allow_none) for name, layer in layers.items()}


def _parq_layer_dir(base: Path, name: str) -> Path:
    """Layer dataset folder: base/<layer_name>/tile=Z-X-Y/part-*.parquet"""
    return base / name


def _parq_cov_path(base: Path, name: str) -> Path:
    """Layer coverage file: base/__cov__<layer>.parquet (one line with a union polygon, EPSG:4326)"""
    return base / f"__cov__{name}.parquet"


def _read_cov_parquet(base: Path, name: str):
    p = _parq_cov_path(base, name)
    if not p.exists():
        return None
    tbl = pq.read_table(p)
    if tbl.num_rows == 0 or "geometry_wkb" not in tbl.column_names:
        return None
    pdf = tbl.to_pandas()
    geom = gpd.GeoSeries.from_wkb(pdf["geometry_wkb"], crs="EPSG:4326")
    if geom.empty:
        return None
    return geom.union_all()


def _write_cov_parquet(base: Path, name: str, poly):
    p = _parq_cov_path(base, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, geometry="geometry", crs="EPSG:4326")
    tbl = pa.table({"geometry_wkb": pa.array(gdf.geometry.to_wkb())})
    pq.write_table(tbl, p)


def _merge_cov(base: Path, name: str, add_poly):
    cur = _read_cov_parquet(base, name)
    if cur is None:
        _write_cov_parquet(base, name, add_poly)
    else:
        _write_cov_parquet(base, name, cur.union(add_poly))


def _lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    xtile = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return z, xtile, ytile


def _bbox_tiles(w, s, e, n, z):
    z0, x0, y0 = _lonlat_to_tile(w, n, z)
    z1, x1, y1 = _lonlat_to_tile(e, s, z)
    xs = range(min(x0, x1), max(x0, x1) + 1)
    ys = range(min(y0, y1), max(y0, y1) + 1)
    return [f"{z}-{x}-{y}" for x in xs for y in ys]


def _tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Tile boundaries in lon/lat (EPSG:4326) as (west, south, east, north)."""
    n = 2 ** z
    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_w, lat_s, lon_e, lat_n


def _to_pa_array(series: pd.Series) -> pa.Array:
    """Safely converts pandas.Series to Arrow, normalizing object types to strings."""
    if pd.api.types.is_integer_dtype(series):
        return pa.array(series.astype("Int64"), from_pandas=True)
    if pd.api.types.is_float_dtype(series):
        return pa.array(series.astype("float64"), from_pandas=True)
    if pd.api.types.is_bool_dtype(series):
        return pa.array(series.astype("boolean"), from_pandas=True)

    def _norm(x):
        if pd.isna(x):
            return None
        if isinstance(x, (bytes, bytearray)):
            try:
                return x.decode("utf-8")
            except Exception:
                return x.hex()
        return str(x)

    return pa.array(series.map(_norm), type=pa.string())


def _write_part_parquet(base: Path, layer_name: str, gdf: gpd.GeoDataFrame, unique_id: Optional[str]):
    """
     Places each feature in all tiles (tile=Z-X-Y) whose boundaries intersect its bbox.
     We do dedup by unique_id INSIDE the tile.
     """
    if gdf is None or gdf.empty:
        return

    layer_dir = _parq_layer_dir(base, layer_name)
    layer_dir.mkdir(parents=True, exist_ok=True)

    src_crs = gdf.crs
    pdf4326 = gdf.to_crs(4326) if src_crs and CRS.from_user_input(src_crs) != CRS.from_epsg(4326) else gdf.copy()

    b = pdf4326.geometry.bounds
    W, S, E, N = float(b["minx"].min()), float(b["miny"].min()), float(b["maxx"].max()), float(b["maxy"].max())

    tiles = _bbox_tiles(W, S, E, N, _PARQUET_TILE_Z)
    if not tiles:
        return

    geom_wkb_all = pa.array(pdf4326.geometry.to_wkb())
    minx_all = pa.array(b["minx"].astype(np.float64).tolist())
    miny_all = pa.array(b["miny"].astype(np.float64).tolist())
    maxx_all = pa.array(b["maxx"].astype(np.float64).tolist())
    maxy_all = pa.array(b["maxy"].astype(np.float64).tolist())

    non_geom_cols = {c: _to_pa_array(pdf4326[c]) for c in pdf4326.columns if c != pdf4326.geometry.name}

    ts = int(time.time())

    for tile in tiles:
        z_str, x_str, y_str = tile.split("-")
        tw, tsouth, te, tnorth = _tile_bounds(int(z_str), int(x_str), int(y_str))

        mask = (b["minx"] <= te) & (b["miny"] <= tnorth) & (b["maxx"] >= tw) & (b["maxy"] >= tsouth)
        if not mask.any():
            continue

        idx = np.flatnonzero(mask.to_numpy())
        if unique_id and unique_id in pdf4326.columns:
            tile_dir = layer_dir / f"tile={tile}"
            ex_ids = set()
            if tile_dir.exists():
                dset_tile = ds.dataset(tile_dir, format="parquet")
                if dset_tile.schema.get_field_index(unique_id) != -1:
                    ex_ids = set(dset_tile.to_table(columns=[unique_id])[unique_id].to_pylist())

            if ex_ids:
                mask_ids = ~pdf4326.iloc[idx][unique_id].isin(ex_ids)
                if not mask_ids.any():
                    continue
                idx = idx[mask_ids.to_numpy()]

        if len(idx) == 0:
            continue

        idx_arr = pa.array(idx, type=pa.int64())

        cols = {}
        for c, arr in non_geom_cols.items():
            cols[c] = pc.take(arr, idx_arr)

        cols["geometry_wkb"] = pc.take(geom_wkb_all, idx_arr)
        cols["minx"] = pc.take(minx_all, idx_arr)
        cols["miny"] = pc.take(miny_all, idx_arr)
        cols["maxx"] = pc.take(maxx_all, idx_arr)
        cols["maxy"] = pc.take(maxy_all, idx_arr)
        cols["tile"] = pa.array([tile] * len(idx), type=pa.string())

        subtbl = pa.table(cols)

        tdir = layer_dir / f"tile={tile}"
        tdir.mkdir(parents=True, exist_ok=True)
        out = tdir / f"part-{ts}.parquet"
        pq.write_table(subtbl, out)


def _read_layer_parquet(base: Path, layer_name: str, bbox: Tuple[float, float, float, float]) -> Optional[
    gpd.GeoDataFrame]:
    """Read a layer from a parquet dataset with pruning by tile and pushdown by bbox columns."""
    layer_dir = _parq_layer_dir(base, layer_name)
    if not layer_dir.exists():
        return None

    w, s, e, n = bbox
    tiles = _bbox_tiles(w, s, e, n, _PARQUET_TILE_Z)
    dset = ds.dataset(layer_dir, format="parquet", partitioning="hive")

    f_tiles = ds.field("tile").isin(tiles)
    f_bbox = (ds.field("minx") <= e) & (ds.field("miny") <= n) & \
             (ds.field("maxx") >= w) & (ds.field("maxy") >= s)

    tbl = dset.to_table(filter=f_tiles & f_bbox)
    if tbl.num_rows == 0:
        return None

    pdf = tbl.to_pandas()
    if "geometry_wkb" not in pdf.columns:
        return None
    geom = gpd.GeoSeries.from_wkb(pdf.pop("geometry_wkb"), crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(pdf, geometry=geom, crs="EPSG:4326")
    return gpd.clip(gdf, box(w, s, e, n))


@log_function_call
def get_layers(style_layers: StyleLayersMap, parquet_path: Path,
               bbox_gdf: gpd.GeoDataFrame, area_id: Optional[int] = None) -> LayersMap:
    """
     Reads/completes the cache of layers in the format of Parquet datasets.
     parquet_path is a directory that contains:
     <parquet_path>/<layer_name>/tile=Z-X-Y/part-*.parquet
     <parquet_path>/__cov__<layer>.parquet

     bbox_gdf: gpd.GeoDataFrame - should be in EPSG:4326
     """
    ensure_epsg4326(bbox_gdf)
    west, south, east, north = bbox_gdf.total_bounds
    bbox = (west, south, east, north)

    layers: LayersMap = {}
    parquet_path.mkdir(parents=True, exist_ok=True)
    req_poly = box(*bbox)

    def _fetch_layer(name: str, s: StyleGeoLayer) -> Optional[gpd.GeoDataFrame]:
        cov = _read_cov_parquet(parquet_path, name)
        covered = (cov is not None and cov.covers(req_poly))

        if covered:
            logger.info(f"Layer '{name}': loaded from cache (parquet, coverage ok)")
            return _read_layer_parquet(parquet_path, name, bbox)

        missing = req_poly if cov is None else req_poly.difference(cov)
        if (missing is None) or missing.is_empty:
            return _read_layer_parquet(parquet_path, name, bbox)

        src = SOURCE_REGISTRY.get(s.source)
        if not src:
            logger.info(f"Layer '{name}': unknown source '{s.source}', skipped")
            return _read_layer_parquet(parquet_path, name, bbox)

        fetcher: Callable = src["fetcher"]
        try:
            new_part = fetcher(
                missing,
                style=s,
                name=name,
                area_id=area_id,
                req_poly=req_poly,
            )
        except Exception as e:
            logger.warning(f"Layer '{name}': fetch failed in missing area: {e}")
            return _read_layer_parquet(parquet_path, name, bbox)

        if new_part is None or new_part.empty:
            logger.info(f"Layer '{name}': fetch empty, return old parquet (if any)")
            return _read_layer_parquet(parquet_path, name, bbox)

        unique_id = SOURCE_REGISTRY[s.source].get("unique_id")
        _write_part_parquet(parquet_path, name, new_part, unique_id=unique_id)
        _merge_cov(parquet_path, name, req_poly if cov is None else req_poly.union(cov))

        return _read_layer_parquet(parquet_path, name, bbox)

    for name, s in style_layers.items():
        gdf = _fetch_layer(name, s)
        if gdf is None or gdf.empty:
            logger.debug(f"Layer '{name}' is empty, skipping")
            continue
        if s.gtype in ALLOWED_TYPES:
            gdf = gdf[gdf.geom_type.isin(ALLOWED_TYPES[s.gtype])]
        if gdf.empty:
            logger.debug(f"Layer '{name}' is empty after type filtering, skipping")
            continue
        layers[name] = GeoLayer.from_style(s, gdf)
    return layers


def load_by_area_id(area_id: int, gtype: str) -> gpd.GeoDataFrame:
    """Get area_id_gdf by OSM area_id and gtype of object"""
    prefixes = ['R', 'W', 'N']
    for prefix in prefixes:
        try:
            area_id_gdf = ox.geocoder.geocode_to_gdf(f"{prefix}{area_id}", by_osmid=True)
            break
        except InsufficientResponseError:
            continue
    else:
        raise ValueError(f"No OSM element found for id '{area_id}' as relation (R), way (W) or node (N)")
    geom_types = set(area_id_gdf.geom_type)
    if gtype not in ALLOWED_TYPES:
        raise ValueError(f"Unknown gtype '{gtype}'. Expected one of {list(ALLOWED_TYPES.keys())}")
    if not geom_types.issubset(ALLOWED_TYPES[gtype]):
        logger.warning(
            f"OSM element '{area_id}' has geometry types={geom_types}, which do not match expected '{gtype}' types {ALLOWED_TYPES[gtype]}")
        raise ValueError(f"Area id '{area_id}' does not match expected geometry type '{gtype}'")
    return area_id_gdf
