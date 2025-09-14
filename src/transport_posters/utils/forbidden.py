from dataclasses import dataclass, field
from typing import List
from shapely.geometry import Polygon as _Poly, Point as _Pt
from render_transport.stop_layout_opt import RectPX


@dataclass
class ForbiddenCollector:
    """Collect polygons representing areas where new elements cannot be placed."""

    geoms: List[_Poly] = field(default_factory=list)

    @staticmethod
    def rectpx_to_poly(r: RectPX) -> _Poly:
        """Convert a RectPX to a shapely polygon defined by its pixel bounds."""
        return _Poly([(r.l, r.b), (r.r, r.b), (r.r, r.t), (r.l, r.t)])

    def add_rect(self, r: RectPX, buffer_px: float = 0.0):
        """Add a buffered rectangle to the forbidden geometry list."""
        g = self.rectpx_to_poly(r)
        if buffer_px > 0:
            g = g.buffer(buffer_px, cap_style=3, join_style=2)
        self.geoms.append(g)

    def add_circle(self, cx_px: float, cy_px: float, radius_px: float, buffer_px: float = 0.0):
        """Add a circle centered at (cx_px, cy_px) to the forbidden geometry list."""
        g = _Pt(cx_px, cy_px).buffer(radius_px + buffer_px, cap_style=1, join_style=1)
        self.geoms.append(g)

    def extend(self, polys: List[_Poly]):
        """Extend the forbidden list with existing polygons."""
        self.geoms.extend(polys)
