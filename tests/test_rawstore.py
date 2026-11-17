"""Tests for the generic Raw Store process: ensure, gate, predicates (no network)."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine
from rasterio.windows import bounds as window_bounds
from shapely.geometry import box

from bandstack import rawstore as rs

NODATA = -99
TRANSFORM = Affine(100.0, 0.0, 0.0, 0.0, -100.0, 200.0)  # 4x2 grid, bounds (0, 0, 400, 200)
REQ = (50.0, 50.0, 350.0, 150.0)


def _vals(mask: np.ndarray) -> np.ndarray:
    """int16 values matching a boolean valid mask (True = data)."""
    return np.where(mask, 1, NODATA).astype("int16")


def _fetch_one_for(masks: dict[str, np.ndarray]):
    """A fake Source's live-read primitive: one tiny synthetic layer per call."""

    def fetch_one(layer: str) -> tuple[np.ndarray, dict]:
        arr = _vals(masks[layer])
        profile = dict(
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype="int16",
            crs="EPSG:31370",
            transform=TRANSFORM,
            nodata=NODATA,
        )
        return arr, profile

    return fetch_one


def _geometry(tmp_path: Path) -> Path:
    """A polygon covering the whole synthetic grid — the 'inside' mask."""
    path = tmp_path / "area.geojson"
    gdf = gpd.GeoDataFrame(geometry=[box(-1000.0, -1000.0, 1000.0, 1000.0)], crs="EPSG:31370")
    gdf.to_file(path, driver="GeoJSON")
    return path


def _record_prose() -> dict:
    return dict(
        source="Fake Source",
        distribution_note="note",
        base_url="https://example",
        statistic="mean",
        grid={"crs_proj4": "+proj=foo"},
        value_encoding="int16",
        decode_factors={"a": 1, "b": 10},
    )


def _run(
    tmp_path: Path,
    masks: dict[str, np.ndarray],
    *,
    vintage: str | None = "Mon, 02 Jun 2020 06:02:00 GMT",
    fetch_one=None,
    **kwargs,
):
    return rs.ensure(
        layers=list(masks),
        raw_dir=tmp_path / "raw",
        required_bounds=REQ,
        fetch_one=fetch_one if fetch_one is not None else _fetch_one_for(masks),
        nodata=NODATA,
        vintage=lambda: vintage,
        record=_record_prose(),
        mask_geometry=_geometry(tmp_path),
        **kwargs,
    )


def _record(tmp_path: Path) -> dict:
    with open(tmp_path / "raw" / "_metadata.json") as f:
        return json.load(f)


# ---------------------------------------------------------------- predicates


def test_group_by_mask_partitions_by_bit_identity():
    """Layers with bit-identical valid masks group together in first-occurrence
    order — the grouping that lets one live read certify a whole batch."""
    a = np.array([True, True, False, False])
    b = np.array([True, True, False, False])
    c = np.array([True, False, False, False])
    groups = rs.group_by_mask({"l1": a, "l2": b, "l3": c})
    assert groups == [["l1", "l2"], ["l3"]]


def test_reference_fraction_violations_flags_drift_beyond_tolerance():
    """The fraction gate flags layers whose valid fraction drifted beyond tol
    from the recorded reference; layers absent from the reference are ignored."""
    measured = {"a": 0.9000, "b": 0.8000}
    reference = {"a": 0.8950, "b": 0.7000, "c": 0.5}
    assert rs.reference_fraction_violations(measured, reference, tol=0.01) == [
        "b: 0.8000 vs reference 0.7000"
    ]


def test_next_fetched_at_written_now_kept_previous_unknown_none():
    """Per-layer timestamps stay truthful across idempotent runs: now if
    (re)written, inherited if the file was kept, None if kept with no history."""
    previous = {"kept": {"fetched_at": "2026-10-26T00:00:00+00:00"}}
    assert rs.next_fetched_at("written", {"written"}, previous, "now") == "now"
    assert rs.next_fetched_at("kept", {"written"}, previous, "now") == "2026-10-26T00:00:00+00:00"
    assert rs.next_fetched_at("unknown", {"written"}, previous, "now") is None


def test_raw_reference_fractions_reads_the_sidecar(tmp_path):
    """The raw Provenance Record's per-layer valid_frac_inside entries parse
    into the reference dict the fraction gate compares against."""
    (tmp_path / "_metadata.json").write_text(
        json.dumps({"layers": [{"name": "x", "valid_frac_inside": 0.5}]})
    )
    assert rs.raw_reference_fractions(tmp_path) == {"x": 0.5}


def test_window_outward_covers_and_is_tight():
    """Outward rounding yields whole pixels that fully cover the requested
    bounds (no data lost at a partial edge) yet overshoot by at most one pixel
    per side (no wasted bandwidth)."""
    transform = Affine(250.0, 0.0, 0.0, 0.0, -250.0, 0.0)
    req = (123456.7, -5666666.6, 234567.8, -5555555.5)
    win = rs.window_outward(transform, req)
    assert win.col_off == int(win.col_off) and win.width == int(win.width)
    assert win.row_off == int(win.row_off) and win.height == int(win.height)
    w, s, e, n = window_bounds(win, transform)
    assert w <= req[0] and s <= req[1] and e >= req[2] and n >= req[3]
    assert w >= req[0] - 250.0 - 1e-9
    assert s >= req[1] - 250.0 - 1e-9
    assert e <= req[2] + 250.0 + 1e-9
    assert n <= req[3] + 250.0 + 1e-9


def test_window_outward_respects_each_vrt_grids_origin():
    """Regression for the 2026-11 misregistration bug: one Source may publish
    layer files on offset native grids (here: origins 750 m = 3 px apart), so
    the same required bounds must yield windows offset by exactly 3 columns —
    one window reused across both families was the bug."""
    family_a = Affine(250.0, 0.0, -19949000.0, 0.0, -250.0, 8361000.0)
    family_b = Affine(250.0, 0.0, -19949750.0, 0.0, -250.0, 8361000.0)
    req = (1267000.0, 2925000.0, 1276000.0, 3020000.0)
    w_a = rs.window_outward(family_a, req)
    w_b = rs.window_outward(family_b, req)
    assert w_b.col_off - w_a.col_off == 3
    for win, t in ((w_a, family_a), (w_b, family_b)):
        b = window_bounds(win, t)
        assert b[0] <= req[0] and b[1] <= req[1] and b[2] >= req[2] and b[3] >= req[3]


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
    """file_covers is the keep-vs-refetch test of idempotent fetching: a file
    counts as reusable only if it fully contains the required bounds."""
    tif = tmp_path / "a.tif"
    _write_tif(tif, (0.0, 0.0, 1000.0, 1000.0))
    assert rs.file_covers(tif, (100.0, 100.0, 900.0, 900.0))
    assert not rs.file_covers(tif, (-100.0, 0.0, 1000.0, 1000.0))
    assert not rs.file_covers(tif, (0.0, 0.0, 1100.0, 1000.0))


# ----------------------------------------------------------------- the gate


def test_gate_clean_fetch_passes_without_refetch():
    """When every on-disk mask matches the live read, the gate passes with
    zero refetches — the no-false-positives case."""
    batch_a = np.array([True, True, False, False])
    batch_b = np.array([True, False, False, False])
    masks = {"A1": batch_a, "A2": batch_a.copy(), "B1": batch_b}
    refetched: list[str] = []
    verified = rs.verify_raw_store(
        list(masks),
        load_mask=masks.__getitem__,
        read_live=lambda rep: _vals(masks[rep]),
        refetch=refetched.append,
        nodata=NODATA,
    )
    assert refetched == []
    assert verified == masks


def test_gate_group_artifact_refetches_only_that_group():
    """A fetch artifact (valid live, nodata on disk — a simulated failed tile
    read) refetches only the affected mask group and then verifies clean."""
    truth_a = np.ones(12, dtype=bool)
    truth_b = np.array([True, True, True, False] * 3)
    artifact_b = truth_b.copy()
    artifact_b[0:3] = False  # valid live, nodata on disk: proven fetch artifacts
    truth = {"A1": truth_a, "A2": truth_a, "B1": truth_b, "B2": truth_b}
    on_disk = {"A1": truth_a, "A2": truth_a.copy(), "B1": artifact_b, "B2": artifact_b.copy()}
    refetched: list[str] = []

    def refetch(layer: str) -> None:
        refetched.append(layer)
        on_disk[layer] = truth[layer]

    verified = rs.verify_raw_store(
        list(on_disk),
        load_mask=lambda layer: on_disk[layer],
        read_live=lambda rep: _vals(truth[rep]),
        refetch=refetch,
        nodata=NODATA,
    )
    assert sorted(refetched) == ["B1", "B2"]
    for layer, mask in verified.items():
        assert mask.tolist() == truth[layer].tolist()


def test_gate_ghost_cells_fail_the_group_too():
    """The reverse disagreement (valid on disk, nodata live) also refetches:
    either the disk copy is stale or the live read itself glitched — both are
    defects, never a pass."""
    live_truth = np.array([True, True, False, False])
    disk = np.array([True, True, True, False])  # 'valid' where the live read says nodata
    on_disk = {"B1": disk}
    refetched: list[str] = []

    def refetch(layer: str) -> None:
        refetched.append(layer)
        on_disk[layer] = live_truth

    rs.verify_raw_store(
        ["B1"],
        load_mask=lambda layer: on_disk[layer],
        read_live=lambda rep: _vals(live_truth),
        refetch=refetch,
        nodata=NODATA,
    )
    assert refetched == ["B1"]


def test_gate_fails_loudly_after_max_attempts():
    """An artifact that survives every refetch raises after exactly
    max_attempts tries — never a silent pass, never an infinite loop."""
    truth = np.ones(8, dtype=bool)
    artifact = truth.copy()
    artifact[:4] = False
    on_disk = {"B1": artifact}
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="fetch verification failed for B1"):
        rs.verify_raw_store(
            ["B1"],
            load_mask=lambda layer: on_disk[layer],
            read_live=lambda rep: _vals(truth),
            refetch=calls.append,
            nodata=NODATA,
            max_attempts=3,
        )
    assert len(calls) == 3


# -------------------------------------------------------------------- ensure


def test_ensure_fetches_into_a_raw_store_and_writes_the_record(tmp_path):
    """A fresh run writes every layer, passes the gate, and assembles the
    Provenance Record: generic skeleton in canonical order, adapter prose in
    its slots, per-layer fractions and fetched_at bookkeeping."""
    full = np.ones((2, 4), dtype=bool)
    masks = {"a": full, "b": full.copy()}
    written = _run(tmp_path, masks)
    assert written == ["a", "b"]
    assert (tmp_path / "raw" / "a.tif").exists()

    meta = _record(tmp_path)
    assert list(meta) == [
        "source",
        "run_at",
        "distribution",
        "base_url",
        "layer_inventory",
        "statistic",
        "grid",
        "nodata",
        "value_encoding",
        "decode_factors",
        "verification",
        "layers",
    ]
    assert meta["source"] == "Fake Source"
    assert meta["nodata"] == NODATA
    assert meta["layer_inventory"] == ["a", "b"]
    assert meta["distribution"]["file_last_modified"] == "Mon, 02 Jun 2020 06:02:00 GMT"
    assert meta["verification"]["mask_groups"] == [["a", "b"]]
    assert meta["verification"]["refetched_by_gate"] == []
    layer = meta["layers"][0]
    assert list(layer) == [
        "name",
        "fetched_at",
        "window_bounds",
        "valid_frac_window",
        "valid_frac_inside",
    ]
    assert layer["valid_frac_inside"] == 1.0
    assert layer["window_bounds"] == [0.0, 0.0, 400.0, 200.0]


def test_ensure_is_idempotent_and_preserves_fetched_at(tmp_path):
    """A second run keeps covering files (only the gate's representative is
    read live) and the Record inherits each kept layer's fetched_at."""
    full = np.ones((2, 4), dtype=bool)
    masks = {"a": full, "b": full.copy()}
    _run(tmp_path, masks)
    first = {rec["name"]: rec["fetched_at"] for rec in _record(tmp_path)["layers"]}
    assert all(ts is not None for ts in first.values())

    calls: list[str] = []
    fetch_one = _fetch_one_for(masks)

    def counting_fetch_one(layer):
        calls.append(layer)
        return fetch_one(layer)

    written = _run(tmp_path, masks, fetch_one=counting_fetch_one)
    assert written == []
    assert calls == ["a"]  # one live read for the single mask group
    second = {rec["name"]: rec["fetched_at"] for rec in _record(tmp_path)["layers"]}
    assert second == first


def test_ensure_force_refetches_every_layer(tmp_path):
    """force=True re-reads every layer even when the files cover the bounds."""
    full = np.ones((2, 4), dtype=bool)
    masks = {"a": full, "b": full.copy()}
    _run(tmp_path, masks)
    calls: list[str] = []
    fetch_one = _fetch_one_for(masks)

    def counting_fetch_one(layer):
        calls.append(layer)
        return fetch_one(layer)

    written = _run(tmp_path, masks, force=True, fetch_one=counting_fetch_one)
    assert written == ["a", "b"]
    assert calls == ["a", "b", "a"]  # two fetches + one gate representative read


def test_ensure_gate_refetches_disk_artifacts_and_records_them(tmp_path):
    """A nodata hole punched into a kept file (simulating a failed tile read)
    is refetched by the gate and listed in the Record's refetched_by_gate."""
    full = np.ones((2, 4), dtype=bool)
    masks = {"a": full, "b": full.copy()}
    _run(tmp_path, masks)

    path = tmp_path / "raw" / "a.tif"
    with rasterio.open(path) as ds:
        profile = ds.profile.copy()
        arr = ds.read(1)
    arr[0, 0] = NODATA
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)

    written = _run(tmp_path, masks)
    assert written == ["a"]
    with rasterio.open(path) as ds:
        assert (ds.read(1) != NODATA).all()  # hole repaired by the refetch
    assert _record(tmp_path)["verification"]["refetched_by_gate"] == ["a"]


def test_ensure_fraction_gate_raises_on_drift_with_unchanged_vintage(tmp_path):
    """When the live source serves sparser data than the recorded reference
    and the vintage is unchanged, the run fails loudly — the drift check the
    gate's live read alone cannot provide."""
    full = np.ones((2, 4), dtype=bool)
    _run(tmp_path, {"a": full, "b": full.copy()})

    sparse = full.copy()
    sparse[0, 0] = False  # 7/8 valid inside vs recorded 1.0
    with pytest.raises(RuntimeError, match="valid-fraction gate failed"):
        _run(tmp_path, {"a": sparse, "b": sparse.copy()})


def test_ensure_fraction_gate_silent_when_vintage_changed(tmp_path):
    """A changed distribution vintage deactivates the reference comparison —
    the fractions are simply re-recorded as the new reference."""
    full = np.ones((2, 4), dtype=bool)
    _run(tmp_path, {"a": full, "b": full.copy()})

    sparse = full.copy()
    sparse[0, 0] = False
    _run(tmp_path, {"a": sparse, "b": sparse.copy()}, vintage="Tue, 17 Nov 2026 08:00:00 GMT")
    meta = _record(tmp_path)
    assert meta["layers"][0]["valid_frac_inside"] == 0.875
    assert meta["distribution"]["file_last_modified"] == "Tue, 17 Nov 2026 08:00:00 GMT"


def test_ensure_writes_nothing_when_file_does_not_cover(tmp_path):
    """A fetch_one whose file does not cover the required bounds fails loudly
    at write time — the post-write coverage assertion."""
    full = np.ones((2, 4), dtype=bool)
    narrow = Affine(100.0, 0.0, 0.0, 0.0, -100.0, 200.0)

    def narrow_fetch_one(layer):
        arr = _vals(full)
        profile = dict(
            driver="GTiff",
            height=2,
            width=2,  # covers only (0, 0, 200, 200) — not REQ
            count=1,
            dtype="int16",
            crs="EPSG:31370",
            transform=narrow,
            nodata=NODATA,
        )
        return arr, profile

    with pytest.raises(RuntimeError, match="does not cover the required bounds"):
        _run(tmp_path, {"a": full, "b": full.copy()}, fetch_one=narrow_fetch_one)
