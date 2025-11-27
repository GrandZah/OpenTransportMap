from dataclasses import dataclass, field
from typing import List
from rtree.index import Index as _RtreeIndex
from shapely.geometry import Polygon as _Poly, Point as _Pt, MultiPolygon as _MultiPoly
from matplotlib.axes import Axes
from matplotlib.patches import Polygon as MplPolygon

from transport_posters.render_transport.stop_layout_opt import RectPX



@dataclass
class ForbiddenCollector:
    """Collect polygons representing areas where new elements cannot be placed."""

    geoms: List[_Poly] = field(default_factory=list)
    _index: _RtreeIndex = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize R-tree index and sync it with existing geometries."""
        self._index = _RtreeIndex()
        for idx, geom in enumerate(self.geoms):
            self._index.insert(idx, geom.bounds)

    @staticmethod
    def rectpx_to_poly(r: "RectPX") -> _Poly:
        """Convert a RectPX to a pixel-space polygon."""
        return _Poly([(r.l, r.b), (r.r, r.b), (r.r, r.t), (r.l, r.t)])

    def _add_geom(self, geom: _Poly) -> None:
        """Append geometry and insert its bounds into the R-tree."""
        idx = len(self.geoms)
        self.geoms.append(geom)
        self._index.insert(idx, geom.bounds)

    def add_rect(self, r: "RectPX", buffer_px: float = 0.0) -> None:
        """Add a buffered rectangle to the forbidden list."""
        geom = self.rectpx_to_poly(r)
        if buffer_px > 0.0:
            geom = geom.buffer(
                buffer_px,
                cap_style="square",
                join_style="mitre",
            )
        self._add_geom(geom)

    def add_circle(self, cx_px: float, cy_px: float, radius_px: float, buffer_px: float = 0.0) -> None:
        """Add a circular forbidden region in pixel coordinates."""
        effective_radius = radius_px + buffer_px
        geom = _Pt(cx_px, cy_px).buffer(
            effective_radius,
            cap_style="round",
            join_style="round",
        )
        self._add_geom(geom)

    def add_poly(self, poly: _Poly) -> None:
        """Add an arbitrary polygon to the forbidden list."""
        self._add_geom(poly)

    def extend(self, polys: List[_Poly]) -> None:
        """Extend the forbidden list with multiple polygons."""
        for poly in polys:
            self._add_geom(poly)

    def is_free_geom(self, geom: _Poly) -> bool:
        """Return True if `geom` does not intersect any forbidden geometry."""
        if not self.geoms:
            return True

        candidate_ids = list(self._index.intersection(geom.bounds))
        for idx in candidate_ids:
            if geom.intersects(self.geoms[idx]):
                return False
        return True

    def is_rect_free(self, r: "RectPX", buffer_px: float = 0.0) -> bool:
        """Check whether a rectangle is free of forbidden geometry."""
        geom = self.rectpx_to_poly(r)
        if buffer_px > 0.0:
            geom = geom.buffer(
                buffer_px,
                cap_style="square",
                join_style="mitre",
            )
        return self.is_free_geom(geom)

    def reserve_rect_if_free(self, r: "RectPX", buffer_px: float = 0.0) -> bool:
        """
        Reserve a rectangle if it does not intersect existing forbidden regions.

        Returns True if the rectangle was free and added, False otherwise.
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
