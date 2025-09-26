from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConfigPaths:
    base_dir: Path
    data_cache_dir: Path
    data_raw_dir: Path
    data_processed_dir: Path
    data_map_dir: Path
    style_dir: Path
    output_logs_dir: Path
    output_maps_dir: Path
    output_composed_img_dir: Path
    style_assets_fonts_dir: Path

    @classmethod
    def from_base(cls, base: Path) -> "ConfigPaths":
        return cls(
            base_dir=base,
            data_cache_dir=base / "data" / "cache",
            data_raw_dir=base / "data" / "raw",
            data_processed_dir=base / "data" / "processed",
            data_map_dir=base / "data" / "map",
            style_dir=base / "style",
            output_logs_dir=base / "output" / "logs",
            output_maps_dir=base / "output" / "maps",
            output_composed_img_dir=base / "output" / "composed_img",
            style_assets_fonts_dir=base / "style" / "assets" / "fonts"
        )

    def ensure_dirs(self):
        for value in self.__dict__.values():
            if isinstance(value, Path):
                value.mkdir(parents=True, exist_ok=True)


def load_config_paths() -> ConfigPaths:
    base = Path(__file__).resolve().parents[2]
    paths = ConfigPaths.from_base(base)
    paths.ensure_dirs()
    return paths


def load_config_render() -> dict:
    return {
        "area_id": 1327509,
        "figsize": (20, 20),
        "general_layers_name": "general_city_layers",
        "far_layers_name": "far_city_layers_labeled",
        "detailed_layers_name": "detailed_city_layers_v2"
    }


CONFIG_RENDER = load_config_render()
CONFIG_PATHS = load_config_paths()

FONT_INTER_BOLD = CONFIG_PATHS.style_assets_fonts_dir / "Inter-Bold.otf"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

PAPER_SIZES_INCH = {
    "A5": (5.8, 8.3),
    "A4": (8.3, 11.7),
    "A3": (11.7, 16.5),
    "A2": (16.5, 23.4),
    "A1": (23.4, 33.1),
    "A0": (33.1, 46.8),
    "Letter": (8.5, 11),
    "Legal": (8.5, 14),
}
