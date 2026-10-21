"""Tests for bandstack.grid: frozen numbers, lattice sharing, nesting, masks."""

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine

from bandstack.grid import GridSpec, from_bounds_snapped, from_preset

LINDA_ORIGIN = (18000.0, 248000.0)
LINDA_RES = 10.0


def test_lambert72_master_grid_frozen_numbers():
    g = from_preset("lambert72", res=10)
    assert g.crs == CRS.from_epsg(31370)
    assert g.origin == (21000.0, 245000.0)
    assert (g.width, g.height) == (23900, 9300)
    assert g.res == 10.0
    assert g.shape == (9300, 23900)
    assert g.bounds == (21000.0, 152000.0, 260000.0, 245000.0)


def test_lambert72_bandstack_grid_frozen_numbers():
    g = from_preset("lambert72", res=200)
    assert g.origin == (21000.0, 245000.0)
    assert (g.width, g.height) == (1195, 465)
    assert g.width * g.height == 555_675


def test_master_grid_shares_lindas_lattice():
    g = from_preset("lambert72", res=10)
    assert (g.transform.c - LINDA_ORIGIN[0]) % LINDA_RES == 0.0
    assert (LINDA_ORIGIN[1] - g.transform.f) % LINDA_RES == 0.0


def test_master_nests_into_bandstack():
    master = from_preset("lambert72", res=10)
    band = from_preset("lambert72", res=200)
    assert band.aggregation_factor(master) == 20


def test_snap_is_outward_and_stable():
    crs = CRS.from_epsg(31370)
    on_line = from_bounds_snapped(crs, (21000.0, 152000.0, 260000.0, 245000.0), res=10, snap=1000.0)
    assert on_line.origin == (21000.0, 245000.0)
    assert (on_line.width, on_line.height) == (23900, 9300)
    shifted = from_bounds_snapped(crs, (21001.0, 152001.0, 259999.0, 244999.0), res=10, snap=1000.0)
    assert shifted == on_line


def test_gridspec_rejects_non_square_cells():
    with pytest.raises(ValueError, match="square"):
        GridSpec(CRS.from_epsg(31370), Affine(10, 0, 0, 0, -20, 0), 5, 5)


def test_gridspec_rejects_rotated_transform():
    with pytest.raises(ValueError, match="north-up"):
        GridSpec(CRS.from_epsg(31370), Affine(10, 2, 0, 2, -10, 0), 5, 5)


def test_snap_must_be_multiple_of_res():
    with pytest.raises(ValueError, match="multiple"):
        from_bounds_snapped(CRS.from_epsg(31370), (0, 0, 100, 100), res=30, snap=1000.0)


def test_aggregation_factor_rejects_off_lattice():
    band = from_preset("lambert72", res=200)
    off = GridSpec(band.crs, Affine(10, 0, 21005.0, 0, -10, 245000.0), 100, 100)
    with pytest.raises(ValueError):
        band.aggregation_factor(off)


def test_aggregation_factor_rejects_coarser_input():
    master = from_preset("lambert72", res=10)
    band = from_preset("lambert72", res=200)
    with pytest.raises(ValueError):
        master.aggregation_factor(band)


def test_bandstack_mask_shape_and_extent():
    g = from_preset("lambert72", res=200)
    m = g.mask()
    assert m.shape == g.shape
    assert m.dtype == np.bool_
    assert 300_000 < int(m.sum()) < 400_000


def test_bandstack_mask_is_readonly_and_cached():
    g = from_preset("lambert72", res=200)
    assert g.mask() is g.mask()
    with pytest.raises(ValueError):
        g.mask()[0, 0] = True


def test_bandstack_mask_has_brussels_hole():
    g = from_preset("lambert72", res=200)
    m = g.mask()
    col0, col1 = int((146_000 - 21_000) // 200), int((158_000 - 21_000) // 200)
    row0, row1 = int((245_000 - 183_000) // 200), int((245_000 - 168_000) // 200)
    box = m[row0:row1, col0:col1]
    assert box.any()
    assert not box.all()


def test_eea_preset_builds_3035_grid():
    g = from_preset("eea", res=1000)
    assert g.crs == CRS.from_epsg(3035)
    assert g.transform.c % 1000.0 == 0.0
    assert g.transform.f % 1000.0 == 0.0


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="preset"):
        from_preset("mercat")
