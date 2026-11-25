"""Tests for the Linda Source adapter (no network)."""

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from bandstack.grid import GridSpec, from_preset
from bandstack.sources import linda

# The publisher's SLD for lu_landgebruik_vlaa_2025_v3, fetched live
# 2026-11-19 (identical to the copy inside the distribution zip). The
# transcription in linda.CLASSES must reproduce it exactly.
_PUBLISHED_SLD = """<?xml version="1.0" encoding="UTF-8"?>
<StyledLayerDescriptor xmlns="http://www.opengis.net/sld" xmlns:sld="http://www.opengis.net/sld"
  xmlns:ogc="http://www.opengis.net/ogc" xmlns:gml="http://www.opengis.net/gml" version="1.0.0">
  <UserLayer>
    <sld:UserStyle>
      <sld:Name>lu_landgebruik_vlaa_2025_v3</sld:Name>
      <sld:FeatureTypeStyle>
        <sld:Rule>
          <sld:RasterSymbolizer>
            <sld:ColorMap type="values">
              <sld:ColorMapEntry color="#ff0000" quantity="1" label="Huizen en tuinen"/>
              <sld:ColorMapEntry color="#8400a8" quantity="2" label="Industrie"/>
              <sld:ColorMapEntry color="#ff00c5" quantity="3" label="Commerciële doeleinden"/>
              <sld:ColorMapEntry color="#002673" quantity="4" label="Diensten"/>
              <sld:ColorMapEntry color="#686868" quantity="5" label="Transportinfrastructuur"/>
              <sld:ColorMapEntry color="#ffaa00" quantity="6" label="Recreatie"/>
              <sld:ColorMapEntry color="#a87000" quantity="7"
                label="Landbouwgebouwen en -infrastructuur"/>
              <sld:ColorMapEntry color="#cccccc" quantity="8" label="Overige bebouwde terreinen"/>
              <sld:ColorMapEntry color="#828282" quantity="9" label="Overige onbebouwde terreinen"/>
              <sld:ColorMapEntry color="#dfe6a9" quantity="10" label="Actieve groeves"/>
              <sld:ColorMapEntry color="#df73ff" quantity="11" label="Luchthavens"/>
              <sld:ColorMapEntry color="#267300" quantity="12" label="Bos"/>
              <sld:ColorMapEntry color="#ffffbe" quantity="13" label="Akker"/>
              <sld:ColorMapEntry color="#a3ff73" quantity="14" label="Grasland in landbouwgebruik"/>
              <sld:ColorMapEntry color="#897044" quantity="15" label="Struikgewas"/>
              <sld:ColorMapEntry color="#ffd37f" quantity="16" label="Braakliggend en duinen"/>
              <sld:ColorMapEntry color="#005ce6" quantity="17" label="Water"/>
              <sld:ColorMapEntry color="#00a884" quantity="18" label="Moeras"/>
              <sld:ColorMapEntry color="#82ca5b" quantity="19" label="Overige graslanden"/>
            </sld:ColorMap>
          </sld:RasterSymbolizer>
        </sld:Rule>
      </sld:FeatureTypeStyle>
    </sld:UserStyle>
  </UserLayer>
</StyledLayerDescriptor>
"""


def test_transcribed_legend_matches_the_published_sld():
    """CLASSES is a faithful transcription of the publisher's SLD (labels and
    colors): a typo'd label or color would mislabel or miscolor every
    downstream product, and the fetch-time live-SLD check would fail."""
    assert linda.parse_sld(_PUBLISHED_SLD) == linda.CLASSES


def test_legend_is_19_contiguous_classes():
    """19 classes with contiguous codes 1–19 and #rrggbb colors — the codes
    are the raster's raw values, so gaps or extras mean a broken legend."""
    assert sorted(linda.CLASSES) == list(range(1, 20))
    for label, color in linda.CLASSES.values():
        assert label
        assert color.startswith("#") and len(color) == 7


def test_required_bounds_is_the_full_published_raster_for_our_grid():
    """The padded master-grid window strictly contains the published 2025
    extent, so the clamped window must equal the published raster exactly —
    the raw store keeps every published pixel and demands no more."""
    grid = from_preset("lambert72", res=10)
    assert linda.required_bounds(grid) == linda.PUBLISHED_BOUNDS


def test_required_bounds_clamps_each_side_independently():
    """A small grid whose padded extent pokes past the published edge on one
    side is clamped only there and padded normally on the other sides."""
    small = GridSpec(
        crs=linda.LAMBERT72,
        transform=Affine(10.0, 0.0, 255000.0, 0.0, -10.0, 244000.0),
        width=200,
        height=100,
    )  # x 255000–257000, y 243000–244000 — well inside
    w, s, e, n = linda.required_bounds(small)
    assert (w, s) == (255000.0 - 500.0, 243000.0 - 500.0)  # padded, not clamped
    assert e == 257000.0 + 500.0  # clamped at the published east edge


def _synthetic_published_raster(path, bounds, values):
    """A small Linda-shaped published raster: 10 m Lambert 72, int32, nodata
    −9999, values cycling through the given class codes."""
    w, s, e, n = bounds
    width, height = int((e - w) / 10), int((n - s) / 10)
    rng = np.random.default_rng(1)
    arr = rng.choice(values, size=(height, width)).astype("int32")
    profile = dict(
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="int32",
        crs=linda.LAMBERT72,
        transform=Affine(10.0, 0.0, w, 0.0, -10.0, n),
        nodata=linda.NODATA,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return arr


@pytest.fixture
def fake_published(monkeypatch, tmp_path):
    """A fake whole-file download (small raster placed near Leuven, inside
    the Study Area) plus a stubbed legend check — everything else in
    fetch_raw runs for real, network aside."""
    bounds = (174000.0, 173000.0, 174600.0, 173500.0)  # 60 x 50 cells
    raster = tmp_path / "published.tiff"
    expected = _synthetic_published_raster(raster, bounds, values=[1, 12, 13, 14, 17])
    monkeypatch.setattr(linda, "_cached_download", lambda: (raster, "0" * 64))
    monkeypatch.setattr(linda, "verify_legend", lambda: None)
    return bounds, raster, expected


def test_fetch_raw_end_to_end_from_a_fake_download(fake_published, tmp_path):
    """fetch_raw runs the whole raw-store process — window read, class-code
    assert, fetch gate, fractions, Provenance Record — against a synthetic
    published raster: the store must reproduce it exactly."""
    bounds, raster, expected = fake_published
    grid = from_preset("lambert72", res=10)
    raw_dir = tmp_path / "raw"

    written = linda.fetch_raw(grid, raw_dir=raw_dir, published_bounds=bounds)

    assert written == [linda.LAYER]
    out = raw_dir / f"{linda.LAYER}.tif"
    with rasterio.open(out) as ds:
        assert ds.crs == linda.LAMBERT72
        assert ds.nodata == linda.NODATA
        stored = ds.read(1)
        assert tuple(round(v, 6) for v in ds.bounds) == bounds
    assert np.array_equal(stored, expected)

    record = json.loads((raw_dir / "_metadata.json").read_text())
    assert record["layer_inventory"] == [linda.LAYER]
    assert record["distribution"]["file_last_modified"] == "0" * 64  # the sha vintage
    assert record["value_encoding"]["classes"]["13"] == {"label": "Akker", "color": "#ffffbe"}
    assert record["decode_factors"] == {}
    layers = record["layers"]
    assert layers[0]["name"] == linda.LAYER
    assert 0.0 < layers[0]["valid_frac_inside"] <= 1.0


def test_fetch_raw_is_idempotent(fake_published, tmp_path):
    """A second run keeps the stored file (nothing rewritten) and re-runs the
    gate against the same download; the recorded per-layer fetched_at
    survives from the first run."""
    bounds, _, _ = fake_published
    grid = from_preset("lambert72", res=10)
    raw_dir = tmp_path / "raw"

    linda.fetch_raw(grid, raw_dir=raw_dir, published_bounds=bounds)
    first_at = json.loads((raw_dir / "_metadata.json").read_text())["layers"][0]["fetched_at"]

    written = linda.fetch_raw(grid, raw_dir=raw_dir, published_bounds=bounds)
    assert written == []
    second = json.loads((raw_dir / "_metadata.json").read_text())["layers"]
    assert second[0]["fetched_at"] == first_at


def test_fetch_raw_rejects_unknown_class_codes(monkeypatch, tmp_path):
    """A value outside the legend (e.g. 20) fails the fetch loudly instead of
    silently storing an unlabeled class."""
    bounds = (174000.0, 173000.0, 174600.0, 173500.0)
    raster = tmp_path / "published.tiff"
    _synthetic_published_raster(raster, bounds, values=[1, 20])
    monkeypatch.setattr(linda, "_cached_download", lambda: (raster, "0" * 64))
    monkeypatch.setattr(linda, "verify_legend", lambda: None)

    with pytest.raises(RuntimeError, match="unexpected class codes"):
        linda.fetch_raw(
            from_preset("lambert72", res=10), raw_dir=tmp_path / "raw", published_bounds=bounds
        )
