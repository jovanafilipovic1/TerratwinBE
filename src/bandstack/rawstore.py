"""The generic Raw Store process: idempotent fetch, fetch gate, fraction
gate, and Provenance Record — behind one call, for any Source.

A Source adapter supplies everything Source-specific: its layer inventory,
one live-read primitive ``fetch_one(layer) -> (array, profile)`` — the layer
as published on its native grid plus the rasterio write profile describing
it —, its nodata sentinel, the required bounds the store must cover, a
distribution-vintage hook, and the prose embedded in its Provenance Record.
This module owns the process around them:

- idempotent fetch: existing files whose extent covers the required bounds
  are kept (their per-layer ``fetched_at`` survives); ``force=True``
  re-reads every layer;
- the fetch gate: GDAL writes a failed tile read as nodata,
  indistinguishable from a published gap at write time. After every fetch,
  each layer's valid mask is compared cell-for-cell with a second, live
  read of one representative per mask group (published masks are often
  shared within a batch, so one comparison certifies the whole group). A
  cell valid live but nodata on disk is a proven fetch artifact: the
  affected group is refetched, at most ``max_attempts`` times per layer,
  then the run fails loudly;
- the fraction gate: when a previous Record exists and the distribution
  vintage is unchanged, per-layer valid fractions inside the Study Area
  must reproduce the recorded reference within ``REF_TOL`` — the drift
  check that catches artifacts the gate's live read cannot see;
- the raw Provenance Record (``_metadata.json`` beside the store):
  overwritten per run, the authoritative record of what this run did.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

from bandstack.grid import GridSpec, study_area_mask

MAX_VERIFY_ATTEMPTS = 3
REF_TOL = 0.01


def window_outward(transform: rasterio.Affine, req: tuple[float, float, float, float]) -> Window:
    """Integer window on a ``transform`` grid covering ``req`` bounds, rounded
    outward (floor on low edges, ceil on high edges) so no data is lost."""
    # On the raster described by this transform, which pixels cover this
    # geographic area? The answer comes in fractional pixels (e.g. columns
    # 12.34 to 61.20).
    win = from_bounds(*req, transform)
    col0 = math.floor(win.col_off)
    row0 = math.floor(win.row_off)
    col1 = math.ceil(win.col_off + win.width)
    row1 = math.ceil(win.row_off + win.height)
    return Window(col0, row0, col1 - col0, row1 - row0)


def file_covers(path: Path, req: tuple[float, float, float, float], eps: float = 1e-6) -> bool:
    """True if the raster at ``path`` covers the required bounds (w, s, e, n)."""
    with rasterio.open(path) as ds:
        w, s, e, n = ds.bounds
    return w <= req[0] + eps and s <= req[1] + eps and e >= req[2] - eps and n >= req[3] - eps


def _load_mask(path: Path, nodata: int | float) -> np.ndarray:
    """Valid mask (True = data) of one raw-store layer file."""
    with rasterio.open(path) as ds:
        arr = ds.read(1)
    return arr != nodata


def group_by_mask(masks: dict[str, np.ndarray]) -> list[list[str]]:
    """Partition layers into groups with bit-identical valid masks.

    Groups keep first-occurrence order, layers keep input order — so each
    group's representative (``group[0]``) is deterministic.
    """
    groups: dict[str, list[str]] = {}
    for layer, mask in masks.items():
        key = hashlib.sha1(mask.tobytes()).hexdigest()
        groups.setdefault(key, []).append(layer)
    return list(groups.values())


def verify_raw_store(
    layers: Sequence[str],
    *,
    load_mask: Callable[[str], np.ndarray],
    read_live: Callable[[str], np.ndarray],
    refetch: Callable[[str], None],
    nodata: int | float,
    max_attempts: int = MAX_VERIFY_ATTEMPTS,
) -> dict[str, np.ndarray]:
    """Certify every layer's on-disk valid mask against a live second read.

    Layers are grouped by bit-identical masks; one representative per group
    is read live. The distribution is static, so the live read is ground
    truth: a cell valid live but nodata on disk is a proven fetch artifact
    (failed tile read), a cell valid on disk but nodata live means the live
    read itself failed. Either disagreement refetches the whole group (all
    members share the mask that failed). After ``max_attempts`` failures
    attributed to a layer the run fails loudly.

    Returns the verified masks per layer.
    """
    attempts: dict[str, int] = dict.fromkeys(layers, 0)
    masks = {layer: load_mask(layer) for layer in layers}
    while True:
        failing: set[str] = set()
        for group in group_by_mask(masks):
            rep = group[0]
            live = read_live(rep) != nodata
            disk = masks[rep]
            n_artifact = int((live & ~disk).sum())
            n_ghost = int((disk & ~live).sum())
            if n_artifact or n_ghost:
                print(
                    f"gate: {rep} (group of {len(group)}): {n_artifact} artifact, "
                    f"{n_ghost} ghost cells vs live read -> refetch group"
                )
                failing.update(group)
        if not failing:
            return masks
        for layer in failing:
            attempts[layer] += 1
            if attempts[layer] > max_attempts:
                raise RuntimeError(
                    f"fetch verification failed for {layer} after {max_attempts} attempts; "
                    "the live source disagrees with every refetch — investigate before "
                    "trusting the raw store"
                )
        for layer in sorted(failing):
            refetch(layer)
            masks[layer] = load_mask(layer)


def reference_fraction_violations(
    measured: dict[str, float],
    reference: dict[str, float],
    tol: float = REF_TOL,
) -> list[str]:
    """Layers whose measured valid-inside fraction drifted from the recorded
    reference by more than ``tol`` (fraction gate)."""
    return [
        f"{layer}: {measured[layer]:.4f} vs reference {reference[layer]:.4f}"
        for layer in measured
        if layer in reference and abs(measured[layer] - reference[layer]) > tol
    ]


def raw_reference_fractions(raw_dir: Path) -> dict[str, float]:
    """Per-layer valid-inside-Study-Area fractions recorded by the last
    verified fetch (raw Provenance Record)."""
    path = raw_dir / "_metadata.json"
    with open(path) as f:
        meta = json.load(f)
    return {rec["name"]: rec["valid_frac_inside"] for rec in meta.get("layers", [])}


def next_fetched_at(
    layer: str,
    written: set[str],
    previous: dict[str, dict],
    run_at: str,
) -> str | None:
    """Per-layer fetch timestamp: now if (re)written this run; the previous
    record's value if the file was kept; None if kept with no known history."""
    if layer in written:
        return run_at
    return previous.get(layer, {}).get("fetched_at")


def _load_previous_records(raw_dir: Path) -> tuple[dict[str, dict], str | None]:
    """Per-layer records and distribution file date from the existing raw
    Provenance Record, if there is one ({} if absent or unreadable)."""
    path = raw_dir / "_metadata.json"
    if not path.exists():
        return {}, None
    try:
        with open(path) as f:
            meta = json.load(f)
    except json.JSONDecodeError:
        return {}, None
    layers = {rec["name"]: rec for rec in meta.get("layers", []) if "name" in rec}
    return layers, meta.get("distribution", {}).get("file_last_modified")


def ensure(
    *,
    layers: Sequence[str],
    raw_dir: Path,
    required_bounds: tuple[float, float, float, float],
    fetch_one: Callable[[str], tuple[np.ndarray, dict]],
    nodata: int | float,
    vintage: Callable[[], str | None],
    record: dict,
    env: dict[str, str] | None = None,
    bounds_key: str = "window_bounds",
    mask_geometry: Path | str | None = None,
    force: bool = False,
    max_attempts: int = MAX_VERIFY_ATTEMPTS,
) -> list[str]:
    """Ensure a verified Raw Store at ``raw_dir`` and write its Provenance Record.

    The Source adapter supplies: ``layers`` (inventory, in Record order),
    ``fetch_one`` (one live read, returning the undecoded array and its
    rasterio write profile), ``nodata`` (the source's published sentinel),
    ``required_bounds`` (what every file must cover, in the source's native
    CRS), ``vintage`` (distribution file date, or None), and ``record`` —
    the Source-specific prose embedded verbatim, with keys ``source``,
    ``distribution_note``, ``base_url``, ``statistic``, ``grid``,
    ``value_encoding``, ``decode_factors``. ``bounds_key`` names the
    per-layer Record key holding each raw file's native-CRS bounds;
    ``mask_geometry`` overrides the Study Area polygon used for the
    inside-fractions.

    Idempotent: existing files whose extent covers ``required_bounds`` are
    kept (their per-layer ``fetched_at`` survives). ``force=True`` re-reads
    every layer. Every run ends with the fetch gate, so kept files are
    re-certified against the live source, never trusted on extent alone.
    Returns the list of layers (re)written, gate rewrites included.
    """
    req = required_bounds
    raw_dir.mkdir(parents=True, exist_ok=True)
    previous, prev_date = _load_previous_records(raw_dir)
    run_at = datetime.now(UTC).isoformat(timespec="seconds")
    written: list[str] = []

    def fetch_and_write(layer: str) -> None:
        arr, profile = fetch_one(layer)
        out_path = raw_dir / f"{layer}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr, 1)
        if not file_covers(out_path, req):
            raise RuntimeError(f"{layer}: written window does not cover the required bounds")
        if layer not in written:
            written.append(layer)

    gate_rewrites: list[str] = []

    def gate_refetch(layer: str) -> None:
        fetch_and_write(layer)
        gate_rewrites.append(layer)
        print(f"gate refetch {layer}")

    with rasterio.Env(**(env or {})):
        for layer in layers:
            out_path = raw_dir / f"{layer}.tif"
            if out_path.exists() and not force and file_covers(out_path, req):
                continue
            fetch_and_write(layer)
            print(f"fetched {layer}")

        masks = verify_raw_store(
            layers,
            load_mask=lambda layer: _load_mask(raw_dir / f"{layer}.tif", nodata),
            read_live=lambda layer: fetch_one(layer)[0],
            refetch=gate_refetch,
            nodata=nodata,
            max_attempts=max_attempts,
        )

    dist_date = vintage()

    # Per-layer valid fractions: window-wide, and inside the Study Area —
    # the latter is the recorded reference for later fetches and the
    # align-time consistency check. Each layer's Study Area mask is
    # rasterized on that layer's own grid (native grids may differ per
    # batch within one source).
    layer_fractions: dict[str, dict] = {}
    for layer in layers:
        with rasterio.open(raw_dir / f"{layer}.tif") as ds:
            layer_grid = GridSpec(
                crs=ds.crs, transform=ds.transform, width=ds.width, height=ds.height
            )
            layer_bounds = list(ds.bounds)
        inside = study_area_mask(layer_grid, geometry=mask_geometry)
        layer_fractions[layer] = {
            bounds_key: layer_bounds,
            "valid_frac_window": float(masks[layer].mean()),
            "valid_frac_inside": float(masks[layer][inside].sum() / inside.sum()),
        }
    measured_inside = {layer: layer_fractions[layer]["valid_frac_inside"] for layer in layers}
    reference = {
        layer: rec["valid_frac_inside"]
        for layer, rec in previous.items()
        if "valid_frac_inside" in rec
    }
    if reference and dist_date and prev_date == dist_date:
        violations = reference_fraction_violations(measured_inside, reference)
        if violations:
            raise RuntimeError("valid-fraction gate failed: " + "; ".join(violations))

    groups = group_by_mask(masks)
    print(
        f"gate passed: {len(groups)} mask group(s), sizes "
        f"{[len(g) for g in groups]}; every mask certified against a live second read"
    )

    metadata = {
        "source": record["source"],
        "run_at": run_at,
        "distribution": {
            "file_last_modified": dist_date,
            "note": record["distribution_note"],
        },
        "base_url": record["base_url"],
        "layer_inventory": list(layers),
        "statistic": record["statistic"],
        "grid": record["grid"],
        "nodata": nodata,
        "value_encoding": record["value_encoding"],
        "decode_factors": record["decode_factors"],
        "verification": {
            "method": (
                "live second read of one representative per mask group, compared "
                "cell-for-cell; group members refetched on mismatch"
            ),
            "mask_groups": [list(g) for g in groups],
            "refetched_by_gate": gate_rewrites,
        },
        "layers": [
            {
                "name": layer,
                "fetched_at": next_fetched_at(layer, set(written), previous, run_at),
                **layer_fractions[layer],
            }
            for layer in layers
        ],
    }
    with open(raw_dir / "_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    n_fetched = len(written)
    print(f"raw store ready ({n_fetched} written, {len(layers)} verified)")
    return written
