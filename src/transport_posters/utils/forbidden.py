from dataclasses import dataclass, field
from typing import List

from shapely.geometry import Polygon as _Poly, Point as _Pt, MultiPolygon as _MultiPoly
from matplotlib.axes import Axes
from matplotlib.patches import Polygon as MplPolygon

from transport_posters.render_transport.stop_layout_opt import RectPX


@dataclass
class ForbiddenCollector:
    """Collect polygons representing areas where new elements cannot be placed."""

    geoms: List[_Poly] = field(default_factory=list)

    @staticmethod
    def rectpx_to_poly(r: RectPX) -> _Poly:
        """Convert a RectPX to a shapely polygon defined by its pixel bounds."""
        return _Poly([(r.l, r.b), (r.r, r.b), (r.r, r.t), (r.l, r.t)])

    def add_rect(self, r: RectPX, buffer_px: float = 0.0) -> None:
        """Add a buffered rectangle to the forbidden geometry list."""
        g = self.rectpx_to_poly(r)
        if buffer_px > 0:
            g = g.buffer(buffer_px, cap_style=3, join_style=2)
        self.geoms.append(g)

    def add_circle(self, cx_px: float, cy_px: float, radius_px: float, buffer_px: float = 0.0) -> None:
        """Add a circle centered at (cx_px, cy_px) to the forbidden geometry list."""
        g = _Pt(cx_px, cy_px).buffer(radius_px + buffer_px, cap_style=1, join_style=1)
        self.geoms.append(g)

    def add_poly(self, poly: _Poly) -> None:
        """Add an arbitrary polygon to the forbidden geometry list."""
        self.geoms.append(poly)

    def extend(self, polys: List[_Poly]) -> None:
        """Extend the forbidden list with existing polygons."""
        self.geoms.extend(polys)

    def is_free_geom(self, geom: _Poly) -> bool:
        """Return True if `geom` does not intersect any collected forbidden geometry."""
        for g in self.geoms:
            if geom.intersects(g):
                return False
        return True

    def is_rect_free(self, r: RectPX, buffer_px: float = 0.0) -> bool:
        """Check whether the given rectangle is free (does not intersect forbidden geometry)."""
        geom = self.rectpx_to_poly(r)
        if buffer_px > 0:
            geom = geom.buffer(buffer_px)
        return self.is_free_geom(geom)

    def reserve_rect_if_free(self, r: RectPX, buffer_px: float = 0.0) -> bool:
        """
        Check if rectangle is free and, if so, add it to forbidden geometry.
        Returns True if the rectangle was reserved (i.e. area was free).
        """
        if self.is_rect_free(r, buffer_px=buffer_px):
            self.add_rect(r, buffer_px=buffer_px)
            return True
        return False

    def plot_boxes(self, ax: Axes, **patch_kwargs) -> None:
        if not self.geoms:
            return

        inv = ax.transData.inverted()

        if "facecolor" not in patch_kwargs and "fc" not in patch_kwargs:
            patch_kwargs.setdefault("facecolor", "none")
        if "edgecolor" not in patch_kwargs and "ec" not in patch_kwargs:
            patch_kwargs.setdefault("edgecolor", "red")
        patch_kwargs.setdefault("linewidth", 0.8)
        patch_kwargs.setdefault("alpha", 0.7)
        patch_kwargs.setdefault("zorder", 99)

        for geom in self.geoms:
            if geom.is_empty:
                continue

            if isinstance(geom, _MultiPoly):
                polys = list(geom.geoms)
            else:
                polys = [geom]

            for poly in polys:
                xs, ys = poly.exterior.coords.xy
                pts_disp = list(zip(xs, ys))
                pts_data = [inv.transform(p) for p in pts_disp]
                patch = MplPolygon(pts_data, **patch_kwargs)
                ax.add_patch(patch)
