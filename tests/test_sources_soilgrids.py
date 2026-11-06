"""Tests for the SoilGrids Source adapter (no network)."""

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import bounds as window_bounds

from bandstack.grid import from_preset
from bandstack.sources import soilgrids as sg


def test_inventory_is_31_unique_layers():
    assert len(sg.LAYERS) == 31
    assert len(set(sg.LAYERS)) == 31
    assert sg.LAYERS[-1] == "ocs_0-30cm_mean"
    assert all(layer.split("_", 1)[0] in sg.UNITS for layer in sg.LAYERS)


def test_decode_factors():
    assert sg.decode_factor("phh2o_0-5cm_mean") == 10
    assert sg.decode_factor("bdod_0-5cm_mean") == 100
    assert sg.decode_factor("clay_15-30cm_mean") == 1
    assert sg.decode_factor("ocs_0-30cm_mean") == 10


def test_window_outward_covers_and_is_tight():
    transform = Affine(250.0, 0.0, 0.0, 0.0, -250.0, 0.0)
    req = (123456.7, -5666666.6, 234567.8, -5555555.5)
    win = sg.window_outward(transform, req)
    assert win.col_off == int(win.col_off) and win.width == int(win.width)
    assert win.row_off == int(win.row_off) and win.height == int(win.height)
    w, s, e, n = window_bounds(win, transform)
    assert w <= req[0] and s <= req[1] and e >= req[2] and n >= req[3]
    assert w >= req[0] - 250.0 - 1e-9
    assert s >= req[1] - 250.0 - 1e-9
    assert e <= req[2] + 250.0 + 1e-9
    assert n <= req[3] + 250.0 + 1e-9


def _write_tif(path, bounds_wsen):
    w, s, e, n = bounds_wsen
    res = 100.0
    transform = Affine(res, 0.0, w, 0.0, -res, n)
    width, height = int((e - w) / res), int((n - s) / res)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs="EPSG:31370",
        transform=transform,
    ) as dst:
        dst.write(np.zeros((height, width), dtype="uint8"), 1)


def test_file_covers(tmp_path):
    tif = tmp_path / "a.tif"
    _write_tif(tif, (0.0, 0.0, 1000.0, 1000.0))
    assert sg.file_covers(tif, (100.0, 100.0, 900.0, 900.0))
    assert not sg.file_covers(tif, (-100.0, 0.0, 1000.0, 1000.0))
    assert not sg.file_covers(tif, (0.0, 0.0, 1100.0, 1000.0))


def test_required_bounds_pads_the_grid_extent():
    grid = from_preset("lambert72", res=200)
    w, s, e, n = sg.required_bounds_igh(grid)
    w0, s0, e0, n0 = transform_bounds(grid.crs, sg.CRS_IGH, *grid.bounds, densify_pts=101)
    assert w <= w0 and s <= s0 and e >= e0 and n >= n0
    assert (w0 - w) >= sg.PAD_M - 1e-6 and (e - e0) >= sg.PAD_M - 1e-6


def test_window_outward_respects_each_vrt_grids_origin():
    # The published property VRTs form two IGH grid families with origins
    # 750 m (3 px) apart. The same required bounds must therefore map to
    # windows offset by exactly 3 columns — one window reused across both
    # families was the 2026-11 misregistration bug.
    bdod_soc = Affine(250.0, 0.0, -19949000.0, 0.0, -250.0, 8361000.0)
    nine_props = Affine(250.0, 0.0, -19949750.0, 0.0, -250.0, 8361000.0)
    req = (1267000.0, 2925000.0, 1276000.0, 3020000.0)
    w_b = sg.window_outward(bdod_soc, req)
    w_n = sg.window_outward(nine_props, req)
    assert w_n.col_off - w_b.col_off == 3
    for w, t in ((w_b, bdod_soc), (w_n, nine_props)):
        b = window_bounds(w, t)
        assert b[0] <= req[0] and b[1] <= req[1] and b[2] >= req[2] and b[3] >= req[3]
