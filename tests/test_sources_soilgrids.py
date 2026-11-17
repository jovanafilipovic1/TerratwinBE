"""Tests for the SoilGrids Source adapter (no network)."""

from rasterio.warp import transform_bounds

from bandstack.grid import from_preset
from bandstack.sources import soilgrids as sg


def test_inventory_is_31_unique_layers():
    """10 properties x 3 depths + ocs (0–30 cm only) = 31 unique layers, and
    every layer's property has a unit/decode entry — a typo'd name would fetch
    the wrong file or crash at decode."""
    assert len(sg.LAYERS) == 31
    assert len(set(sg.LAYERS)) == 31
    assert sg.LAYERS[-1] == "ocs_0-30cm_mean"
    assert all(layer.split("_", 1)[0] in sg.UNITS for layer in sg.LAYERS)


def test_decode_factors():
    """Decode factors match ISRIC's documented integer encoding (pH x10, bdod
    x100, clay x1, ocs x10) — a transcription slip would silently rescale."""
    assert sg.decode_factor("phh2o_0-5cm_mean") == 10
    assert sg.decode_factor("bdod_0-5cm_mean") == 100
    assert sg.decode_factor("clay_15-30cm_mean") == 1
    assert sg.decode_factor("ocs_0-30cm_mean") == 10


def test_required_bounds_pads_the_grid_extent():
    """The fetch window is the grid extent reprojected to IGH (recomputed here
    independently) plus PAD_M on every side, so border cells' bilinear kernels
    are fully populated by the raw rectangle alone."""
    grid = from_preset("lambert72", res=200)
    w, s, e, n = sg.required_bounds_igh(grid)
    w0, s0, e0, n0 = transform_bounds(grid.crs, sg.CRS_IGH, *grid.bounds, densify_pts=101)
    assert w <= w0 and s <= s0 and e >= e0 and n >= n0
    assert (w0 - w) >= sg.PAD_M - 1e-6 and (e - e0) >= sg.PAD_M - 1e-6
