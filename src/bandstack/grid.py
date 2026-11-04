"""Grid definitions for the Bandstack: one frozen GridSpec aligns every layer.

Rules implemented here (target-grid decision, 2026-10):

- default CRS: Belgian Lambert 72 (EPSG:31370), switchable via a small
  preset registry;
- extent: Study Area (Flemish Region, NUTS BE2) bounding box plus a buffer,
  snapped *outward* to whole-kilometre Lambert 72 lines
- square cells only, north-up, axis-aligned;
- the Study Area mask is rasterized with ``all_touched=True`` (no sterile
  border ring after warping).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine

REPO_ROOT = Path(__file__).resolve().parents[2]  # Two directories up from this file
STUDY_AREA_GEOJSON = REPO_ROOT / "data" / "raw" / "gisco" / "be2_flemish_region_4326.geojson"
BUFFER_M = 1000.0
LAMBERT72 = CRS.from_epsg(31370)
ETRS89_LAEA = CRS.from_epsg(3035)


@dataclass(frozen=True)
class GridSpec:
    """A frozen raster grid: CRS + affine transform + dimensions.

    Every layer-producing function takes a GridSpec as its target geometry,
    so all bands are aligned by construction.
    """

    crs: CRS
    transform: Affine  # transforms a raster (col, row) to real-world coordinates (x,y)
    width: int  # how many columns the raster has
    height: int  # how many rows the raster has

    def __post_init__(self) -> None:
        """Validate the invariants every Bandstack layer relies on."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        t = self.transform
        if t.b != 0.0 or t.d != 0.0:
            raise ValueError(
                "transform must be north-up and axis-aligned (b = d = 0); "
                "rotated or sheared grids would break cell-centre alignment"
            )
        if t.a <= 0.0 or t.e >= 0.0:
            raise ValueError(
                "transform must have a > 0 and e < 0 "
                "(x grows east, y grows south from the top-left origin)"
            )
        if t.a != -t.e:
            raise ValueError(
                "grid supports square cells only (a == -e, i.e. pixel width "
                "equals pixel height); rectangular cells would break integer "
                "aggregation and distance computations"
            )

    @property
    def res(self) -> float:
        """Cell size in CRS units (valid because cells are square)."""
        return self.transform.a

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape ``(height, width)`` for numpy/rasterio."""
        return (self.height, self.width)

    @property
    def origin(self) -> tuple[float, float]:
        """Top-left corner ``(x, y)`` in CRS coordinates."""
        return (self.transform.c, self.transform.f)

    @property
    def bounds(self) -> tuple[float, float, float]:
        """Full extent as ``(west, south, east, north)`` in CRS coordinates."""
        x0, y0 = self.origin
        return (x0, y0 - self.height * self.res, x0 + self.width * self.res, y0)

    def aggregation_factor(self, finer: GridSpec) -> int:
        """Integer ``f`` such that ``f x f`` cells of ``finer`` tile exactly into
        one cell of this grid (exact nesting, no resampling).

        Used in bandstack assembly: derived predictors are computed
        on the 10 m Linda-anchored master grid, then folded onto the 200 m
        Bandstack grid — here ``f = 20`` — as a pure integer reshape
        (per-class fractions, mean of distances). Raises instead of silently
        resampling whenever the grids do not nest exactly.
        """
        if finer.crs != self.crs:
            raise ValueError("grids must share a CRS")
        ratio = self.res / finer.res
        f = round(ratio)
        if f < 1 or not math.isclose(ratio, f, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"resolutions do not nest integer-wise: {self.res} / {finer.res}")
        dx = self.transform.c - finer.transform.c
        dy = self.transform.f - finer.transform.f
        if dx % finer.res != 0.0 or dy % finer.res != 0.0:
            raise ValueError("origins are off-lattice; aggregation would resample")
        if self.width * self.res != finer.width * finer.res:
            raise ValueError("grids do not span the same width")
        if self.height * self.res != finer.height * finer.res:
            raise ValueError("grids do not span the same height")
        return f

    def mask(self, geometry: Path | str | None = None, all_touched: bool = True) -> np.ndarray:
        """Boolean Study Area mask on this grid (read-only, cached)."""
        return study_area_mask(self, geometry, all_touched)


def from_bounds_snapped(
    crs: CRS,
    bounds: Sequence[float],
    res: float,
    snap: float,
    buffer_m: float = 0.0,
) -> GridSpec:
    """Build a GridSpec covering ``bounds`` (padded by ``buffer_m``) with cell
    edges snapped outward to multiples of ``snap``.

    Snapping matters because every resolution we use (10 m master, 200 m
    Bandstack) must share one lattice anchored on whole-kilometre lines: the
    200 m grid then nests exactly 20:1 into the 10 m master grid, and both
    land on Linda's native lattice, so loading Linda is a pure clip.
    Snapping *outward* (floor on the low edges, ceil on the high edges)
    guarantees the buffered extent is fully covered, and makes the grid a
    deterministic function of the mask geometry — small changes in the input
    bounds leave the GridSpec unchanged.
    """
    if snap < res or snap % res != 0.0:
        raise ValueError("snap must be a multiple of res")
    left, bottom, right, top = bounds
    x0 = math.floor((left - buffer_m) / snap) * snap
    x1 = math.ceil((right + buffer_m) / snap) * snap
    y0 = math.floor((bottom - buffer_m) / snap) * snap
    y1 = math.ceil((top + buffer_m) / snap) * snap
    width = int(round((x1 - x0) / res))
    height = int(round((y1 - y0) / res))
    return GridSpec(
        crs=crs, transform=Affine(res, 0.0, x0, 0.0, -res, y1), width=width, height=height
    )


@lru_cache(maxsize=8)
def _study_area_bounds(path: Path, crs: CRS) -> tuple[float, float, float]:
    """Read the Study Area polygon and return its bbox ``(w, s, e, n)`` in ``crs``
    (cached: file I/O plus reprojection is expensive)."""
    b = gpd.read_file(path).to_crs(crs).total_bounds
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


@lru_cache(maxsize=8)
def _study_area_geometries(path: Path, crs: CRS) -> tuple:
    """Read the Study Area polygons reprojected to ``crs`` (cached)."""
    return tuple(gpd.read_file(path).to_crs(crs).geometry)


@lru_cache(maxsize=32)
def _mask(grid: GridSpec, path: Path, all_touched: bool) -> np.ndarray:
    """Rasterize the Study Area onto ``grid`` (cached; result is read-only)."""
    geoms = _study_area_geometries(path, grid.crs)
    arr = rasterize(
        ((g, 1) for g in geoms),
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    )
    mask = arr.astype(bool)
    mask.flags.writeable = False
    return mask


def study_area_mask(
    grid: GridSpec,
    geometry: Path | str | None = None,
    all_touched: bool = True,
) -> np.ndarray:
    """Rasterize the Study Area polygon onto ``grid``.

    Cached per (grid, geometry, all_touched); the returned array is read-only
    — copy it before modifying.
    """
    path = (Path(geometry) if geometry is not None else STUDY_AREA_GEOJSON).resolve()
    return _mask(grid, path, all_touched)


def _lambert72(
    res: float = 200.0,
    buffer_m: float = BUFFER_M,
    mask_geometry: Path | str | None = None,
) -> GridSpec:
    """Default preset: Lambert 72, whole-km snap — shares Linda's native lattice."""
    path = Path(mask_geometry) if mask_geometry is not None else STUDY_AREA_GEOJSON
    bounds = _study_area_bounds(path.resolve(), LAMBERT72)
    return from_bounds_snapped(LAMBERT72, bounds, res, snap=1000.0, buffer_m=buffer_m)


def _eea(
    res: float = 1000.0,
    buffer_m: float = BUFFER_M,
    mask_geometry: Path | str | None = None,
) -> GridSpec:
    """EPSG:3035 preset, whole-km snap. Approximate EEA reference-grid
    alignment; verify against the official EEA grid before first real use."""
    path = Path(mask_geometry) if mask_geometry is not None else STUDY_AREA_GEOJSON
    bounds = _study_area_bounds(path.resolve(), ETRS89_LAEA)
    return from_bounds_snapped(ETRS89_LAEA, bounds, res, snap=1000.0, buffer_m=buffer_m)


PRESETS: dict[str, Callable[..., GridSpec]] = {"lambert72": _lambert72, "eea": _eea}


def from_preset(name: str, **kwargs) -> GridSpec:
    """Build a GridSpec by preset ``name`` (see ``PRESETS``); ``kwargs`` pass
    through to the preset factory (e.g. ``res=``, ``buffer_m=``)."""
    try:
        factory = PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown grid preset {name!r}; available: {sorted(PRESETS)}") from None
    return factory(**kwargs)
