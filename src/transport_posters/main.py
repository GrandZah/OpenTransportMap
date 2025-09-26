import argparse

from transport_posters.generate_maps.generate_maps import generate_maps
from transport_posters.generate_three_in_one.generate_three_in_one import generate_three_in_one
from transport_posters.logger import configure_root_logger


def main():
    """
    CLI enter - parse argument and run function of generate img.
    """
    configure_root_logger()

    parser = argparse.ArgumentParser(
        description="Generate per-stop SVG maps with forward bus segments from given stop.")
    parser.add_argument("--area-id", required=True, help="OSM area id (for basemap layers)")
    parser.add_argument("--limit", type=int, help="Limit number of stops to render")
    parser.add_argument("--render-map", action="store_true", help="Draw pastel basemap")
    parser.add_argument("--render-routes", action="store_true", help="Draw bus lines")
    parser.add_argument("--gallery", action="store_true", help="Generate and open an HTML gallery of all rendered SVGs")
    args = parser.parse_args()

    generate_maps(args)
    generate_three_in_one(args)


if __name__ == "__main__":
    main()
