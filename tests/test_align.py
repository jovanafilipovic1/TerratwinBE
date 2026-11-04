"""Tests for bandstack.align: synthetic-raster warps with known answers."""

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import transform_bounds

from bandstack.align import warp_band
from bandstack.grid import GridSpec

L72 = CRS.from_epsg(31370)


def _grid() -> GridSpec:
    """200 m grid over (0, 0)–(4000, 4000) in Lambert 72: 20×20 cells."""
    return GridSpec(L72, Affine(200.0, 0, 0.0, 0, -200.0, 4000.0), 20, 20)


def _write_tif(path, arr, transform, crs, nodata=None):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)


def _ramp_source(path):
    """40×40 int16 source at 100 m over the same extent; value = column index."""
    arr = np.tile(np.arange(40, dtype="int16"), (40, 1))
    _write_tif(path, arr, Affine(100.0, 0, 0.0, 0, -100.0, 4000.0), L72)
    return arr


def test_bilinear_reproduces_linear_ramp(tmp_path):
    src = tmp_path / "ramp.tif"
    _ramp_source(src)
    mask = np.ones((20, 20), dtype=bool)
    out = warp_band(src, _grid(), mask=mask)
    # Target col c centres at x = 100 + 200c -> source coordinate 0.5 + 2c:
    # bilinear of the ramp between columns 2c and 2c+1 is exactly 2c + 0.5.
    expected = np.tile(2 * np.arange(20, dtype="float64") + 0.5, (20, 1))
    assert np.allclose(out[:, 1:19], expected[:, 1:19], atol=1e-4)
    # GDAL renormalizes kernel weights at the source-raster edge; production
    # windows are padded (PAD_M) precisely so target cells never sit there.
    assert np.allclose(out[:, [0, 19]], expected[:, [0, 19]], atol=0.3)


def test_decode_factor_divides(tmp_path):
    src = tmp_path / "ramp.tif"
    _ramp_source(src)
    mask = np.ones((20, 20), dtype=bool)
    out = warp_band(src, _grid(), decode_factor=10, mask=mask)
    expected = np.tile(2 * np.arange(20, dtype="float64") + 0.5, (20, 1)) / 10.0
    assert np.allclose(out[:, 1:19], expected[:, 1:19], atol=1e-4)


def test_source_nodata_propagates_as_nan(tmp_path):
    src = tmp_path / "hole.tif"
    arr = np.ones((40, 40), dtype="int16")
    arr[:, 10:16] = -32768  # 600 m-wide block of nodata
    _write_tif(src, arr, Affine(100.0, 0, 0.0, 0, -100.0, 4000.0), L72, nodata=-32768)
    mask = np.ones((20, 20), dtype=bool)
    out = warp_band(src, _grid(), mask=mask)
    # Target cols whose 2x2 kernel lies fully inside source cols 10..15 -> NaN.
    assert np.isnan(out[:, 5:8]).all()
    # Everything else keeps the constant value 1.
    assert np.isfinite(out[:, :5]).all() and np.allclose(out[:, :5], 1.0)
    assert np.isfinite(out[:, 8:]).all() and np.allclose(out[:, 8:], 1.0)


def test_mask_outside_is_nan(tmp_path):
    src = tmp_path / "ramp.tif"
    _ramp_source(src)
    mask = np.zeros((20, 20), dtype=bool)
    mask[:, :10] = True  # west half only
    out = warp_band(src, _grid(), mask=mask)
    assert np.isnan(out[:, 10:]).all()
    assert np.isfinite(out[:, :10]).all()


def test_cross_crs_warp_of_constant(tmp_path):
    src = tmp_path / "const_4326.tif"
    w, s, e, n = transform_bounds(L72, CRS.from_epsg(4326), 0.0, 0.0, 4000.0, 4000.0)
    res = 0.001
    width = int(np.ceil((e - w) / res)) + 2
    height = int(np.ceil((n - s) / res)) + 2
    transform = Affine(res, 0.0, w, 0.0, -res, n)
    _write_tif(src, np.full((height, width), 7.0, dtype="float32"), transform, CRS.from_epsg(4326))
    mask = np.ones((20, 20), dtype=bool)
    out = warp_band(src, _grid(), mask=mask)
    assert np.isfinite(out).all()
    assert np.allclose(out, 7.0)
