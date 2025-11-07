import logging

from transport_posters.data_transport.get_bus_layers import get_bus_layers, get_from_cache_bus_layers
from transport_posters.data_transport.сity_route_database import CityRouteDatabase
from transport_posters.load_configs import load_config_paths
from transport_posters.logger import configure_root_logger

CONFIG_PATHS = load_config_paths()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    configure_root_logger()

    AREA_OSM_REL_ID = 1327509

    ctx_map: CityRouteDatabase = get_from_cache_bus_layers(AREA_OSM_REL_ID)
    stops = ctx_map.stops_gdf
    platforms = ctx_map.platforms_gdf
    routes = ctx_map.routes_gdf
    edges = ctx_map.edges_gdf

    out_dir = CONFIG_PATHS.data_processed_dir
    stops.to_file(out_dir / f"{AREA_OSM_REL_ID}_stops.geojson", driver="GeoJSON")
    platforms.to_file(out_dir / f"{AREA_OSM_REL_ID}_platforms.geojson", driver="GeoJSON")
    routes.to_file(out_dir / f"{AREA_OSM_REL_ID}_routes.geojson", driver="GeoJSON")
    edges.to_file(out_dir / f"{AREA_OSM_REL_ID}_edges.geojson", driver="GeoJSON")

    logger.info("Saved %d stops, %d routes, %d edges", len(stops), len(routes), len(edges))
