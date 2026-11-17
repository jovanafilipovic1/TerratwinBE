"""Tests for the SoilGrids Source adapter (no network)."""

import json

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform_bounds

from bandstack.grid import from_preset
from bandstack.rawstore import window_outward
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


_SYNTHETIC_RAW = {  # raw int16 values: every property plausible, clay+silt+sand = 1000 g/kg
    "bdod": 125,
    "cec": 120,
    "cfvo": 5,
    "clay": 330,
    "nitrogen": 20,
    "ocd": 10,
    "ocs": 55,
    "phh2o": 55,
    "sand": 340,
    "silt": 330,
    "soc": 12,
}


def _synthetic_raw_store(raw_dir, grid):
    """A constant-valued, gap-free fake raw store on the IGH grid, covering
    the window ``fetch`` would fetch, with a raw Record whose reference
    fractions (all 1.0) the reference-consistency check can reproduce."""
    req = sg.required_bounds_igh(grid)
    t = Affine(sg.RES_M, 0.0, req[0], 0.0, -sg.RES_M, req[3])
    win = window_outward(t, req)
    profile = dict(
        driver="GTiff",
        height=int(win.height),
        width=int(win.width),
        count=1,
        dtype="int16",
        crs=sg.CRS_IGH,
        transform=t @ Affine.translation(win.col_off, win.row_off),
        nodata=sg.NODATA,
    )
    for layer in sg.LAYERS:
        arr = np.full(
            (int(win.height), int(win.width)), _SYNTHETIC_RAW[layer.split("_", 1)[0]], dtype="int16"
        )
        with rasterio.open(raw_dir / f"{layer}.tif", "w", **profile) as dst:
            dst.write(arr, 1)
    (raw_dir / "_metadata.json").write_text(
        json.dumps({"layers": [{"name": layer, "valid_frac_inside": 1.0} for layer in sg.LAYERS]})
    )


def test_ensure_aligned_end_to_end_from_a_synthetic_raw_store(tmp_path):
    """ensure_aligned runs the whole aligned side — warp, per-band stats,
    every verification check, Provenance Record — from any raw store into any
    output dir: the tmp-dir testability the hardcoded output path made
    impossible. Constant plausible input must pass every check."""
    grid = from_preset("lambert72")
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "aligned"
    raw_dir.mkdir()
    _synthetic_raw_store(raw_dir, grid)

    checks = sg.ensure_aligned(grid, raw_dir=raw_dir, out_dir=out_dir)

    failed = [c for c in checks if not c["ok"]]
    assert not failed, failed
    assert (out_dir / sg.ALIGNED_NAME).exists()

    with open(out_dir / "_metadata.json") as f:
        record = json.load(f)
    assert record["source"].startswith("SoilGrids")
    assert record["grid"]["width"] == grid.width and record["grid"]["height"] == grid.height
    assert [b["layer"] for b in record["bands"]] == sg.LAYERS
    assert record["verification"] == checks
    with rasterio.open(out_dir / sg.ALIGNED_NAME) as ds:
        ph = ds.read(sg.LAYERS.index("phh2o_0-5cm_mean") + 1)
    inside = grid.mask()
    assert np.isfinite(ph[inside]).all()
    assert np.allclose(ph[inside], 5.5)  # raw 55, decode /10, warp of a constant is exact
    assert np.isnan(ph[~inside]).all()
