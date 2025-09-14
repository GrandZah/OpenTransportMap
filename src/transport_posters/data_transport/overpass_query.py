import textwrap
from typing import Dict


def build_query_by_bbox(bbox: Dict[str, float]) -> str:
    """We only request relation route=bus, related ways, and stop_position nodes."""
    s, w, n, e = bbox["south"], bbox["west"], bbox["north"], bbox["east"]
    return textwrap.dedent(
        f"""
        [out:json][timeout:180];
        (
          relation["type"="route"]["route"="bus"]({s},{w},{n},{e});
        )->.routes;
        (.routes; >>;)->.r_ways;
        node["public_transport"="stop_position"]({s},{w},{n},{e});
        out body;
        .routes out tags;
        .r_ways out body geom;
    """
    ).strip()
