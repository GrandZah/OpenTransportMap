from matplotlib import pyplot as plt

from data_map.get_data_map import get_data_map_by_bbox_gdf
from load_configs import CONFIG_PATHS
from logger import configure_root_logger
from render_map.render_basemap import render_basemap
from utils.utils_generate import get_bbox_gdf_with_buf


def local_rendering_map(figsize, save_dir, area_id, style_layers_name, layers, bbox):
    fig, ax = plt.subplots(figsize=figsize)
    render_basemap(ax, layers, bbox)
    fig.savefig(
        f"{save_dir}/{area_id}_{style_layers_name}_styled_map_cropped_fixed.svg",
        format="svg",
        bbox_inches="tight",
    )


def main(area_id, style_layers_name):
    bbox_gdf = get_bbox_gdf_with_buf(area_id)
    layers = get_data_map_by_bbox_gdf(area_id, bbox_gdf, style_layers_name)
    local_rendering_map((40, 40), CONFIG_PATHS.debug_dir, area_id, style_layers_name, layers, bbox_gdf)


if __name__ == '__main__':
    configure_root_logger()
    main(4022079, "general_city_layers")
