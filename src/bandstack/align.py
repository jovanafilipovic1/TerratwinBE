"""Warp a single-band raster onto a GridSpec: the one resampling of the chain.

Contract:

- the source is read as published (e.g. ISRIC int16) with its nodata declared;
- exactly one warp onto the target GridSpec (bilinear for continuous data);
- the decode factor is applied after the warp (linear, exact under bilinear);
- the Study Area mask is applied here, on the target grid — never in the
  source, because masking a raster before resampling bleeds nodata into
  border cells;
- the result is float32 in decoded physical units, NaN outside the Study
  Area and wherever the source has no data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from bandstack.grid import GridSpec


def warp_band(
    src_path: Path | str,
    grid: GridSpec,
    *,
    resampling: Resampling = Resampling.bilinear,
    decode_factor: float = 1.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return one source band warped onto ``grid`` as decoded, masked float32.

    ``mask`` defaults to the Study Area mask on ``grid`` (read-only, cached);
    pass a custom boolean array of ``grid.shape`` to override.
    """
    if mask is None:
        mask = grid.mask()
    dst = np.full(grid.shape, np.nan, dtype="float32")
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    if decode_factor != 1.0:
        dst /= np.float32(decode_factor)
    dst[~mask] = np.nan
    return dst
