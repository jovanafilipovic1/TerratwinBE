"""SoilGrids Source: inventory, raw fetch, fetch gate, alignment, verification.

Raw store bounding policy: the fetch window is the target-grid extent
(Study Area bbox + buffer, Lambert 72) transformed into SoilGrids' native
IGH grid, padded and snapped *outward* to whole source pixels. No polygon
mask is applied at fetch time — every pixel is real published data, so the
bilinear warp kernels of border cells are fully populated. The Study Area
mask is applied once, at align time, on the target grid.

Fetch verification gate (ADR 0001): GDAL writes a failed tile read as
nodata, indistinguishable from a published gap at write time. After every
fetch, each layer's valid mask is therefore compared cell-for-cell with a
second, live read of one representative layer per mask group (published
masks are shared within a property batch, so one comparison certifies the
whole group). A cell valid live but nodata on disk is a proven fetch
artifact: the affected group is refetched, at most ``MAX_VERIFY_ATTEMPTS``
times per layer, then the run fails loudly. Verified per-layer valid
fractions are recorded in the raw Provenance Record as the reference for
later fetches (ADR 0001 decision 1).
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, transform, transform_bounds
from rasterio.windows import Window, from_bounds

from bandstack.align import warp_band
from bandstack.grid import REPO_ROOT, GridSpec, study_area_mask

BASE_URL = "https://files.isric.org/soilgrids/latest/data"
CRS_IGH = CRS.from_proj4("+proj=igh +R=6371000 +units=m +no_defs")
NODATA = -32768
RES_M = 250.0
PAD_M = 500.0

RAW_DIR = REPO_ROOT / "data" / "raw" / "soilgrids"
ALIGNED_DIR = REPO_ROOT / "data" / "aligned" / "soilgrids"
ALIGNED_NAME = "soilgrids_lambert72_200m.tif"

# DEPTHS = ["0-5", "5-15", "15-30", "30-60", "60-100", "100-200"]
DEPTHS = ["0-5", "5-15", "15-30"]
PROPERTIES = ["bdod", "cec", "cfvo", "clay", "nitrogen", "ocd", "phh2o", "sand", "silt", "soc"]
LAYERS = [f"{p}_{d}cm_mean" for p in PROPERTIES for d in DEPTHS] + ["ocs_0-30cm_mean"]

UNITS: dict[str, tuple[int, str, str]] = {
    "bdod": (100, "g/cm³", "bulk density"),
    "cec": (10, "cmolc/kg", "cation exchange capacity"),
    "cfvo": (1, "g/kg", "coarse fragments"),
    "clay": (1, "g/kg", "clay fraction"),
    "nitrogen": (100, "g/kg", "total nitrogen"),
    "ocd": (10, "g/dm³", "organic carbon density"),
    "ocs": (10, "t/ha", "organic carbon stock"),
    "phh2o": (10, "pH", "pH (H2O)"),
    "sand": (1, "g/kg", "sand fraction"),
    "silt": (1, "g/kg", "silt fraction"),
    "soc": (10, "g/kg", "soil organic carbon"),
}

PLAUSIBLE: dict[str, tuple[float, float]] = {
    "bdod": (0.2, 2.5),
    "cec": (0.0, 300.0),
    "cfvo": (0.0, 1000.0),
    "clay": (0.0, 1000.0),
    "nitrogen": (0.0, 30.0),
    "ocd": (0.0, 300.0),
    "ocs": (0.0, 1500.0),
    "phh2o": (3.0, 10.0),
    "sand": (0.0, 1000.0),
    "silt": (0.0, 1000.0),
    "soc": (0.0, 300.0),
}

# Approximate city center coordinates
CITIES: dict[str, tuple[float, float]] = {
    "Brussels": (4.35, 50.85),
    "Leuven": (4.70, 50.88),
    "Ghent": (3.72, 51.05),
    "Antwerp": (4.40, 51.22),
    "Bruges": (3.22, 51.21),
    "Hasselt": (5.34, 50.93),
}

# Verification gates (ADR 0001): every layer's valid mask is compared with a
# live second read (fetch gate), and its valid-inside fraction must reproduce
# the Provenance Record reference within REF_TOL (fraction gate).
REF_TOL = 0.01
MAX_VERIFY_ATTEMPTS = 3
INTERP_TOL = 0.05
CLOSURE_MEDIAN_MAX = 10.0

# Network settings for the fetch. GDAL reads the Source over the network with /vsicurl
GDAL_ENV = dict(
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="2",
    GDAL_HTTP_TIMEOUT="60",
    VSI_CACHE="TRUE",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.vrt,.ovr",
)


def vrt_url(layer: str) -> str:
    prop = layer.split("_", 1)[0]
    return f"/vsicurl/{BASE_URL}/{prop}/{layer}.vrt"


def decode_factor(layer: str) -> int:
    return UNITS[layer.split("_", 1)[0]][0]


def required_bounds_igh(grid: GridSpec, pad_m: float = PAD_M) -> tuple[float, float, float, float]:
    """Grid extent in the native IGH grid, padded by ``pad_m`` on all sides."""
    w, s, e, n = transform_bounds(grid.crs, CRS_IGH, *grid.bounds, densify_pts=101)
    return (w - pad_m, s - pad_m, e + pad_m, n + pad_m)


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


def _live_window(
    layer: str, req: tuple[float, float, float, float]
) -> tuple[Window, rasterio.Affine]:
    """(window, transform) of ``layer`` over the required bounds, on the
    layer's *own* VRT grid: the published property VRTs do not share one grid
    — the bdod/soc family and the 9-property family differ by 750 m (3 px) in
    origin — so every layer must be windowed on its own transform."""
    with rasterio.open(vrt_url(layer)) as src:
        win = window_outward(src.transform, req)
        return win, src.window_transform(win)


def _read_live(layer: str, req: tuple[float, float, float, float]) -> np.ndarray:
    """One live read of ``layer`` over the required bounds (int16, undecoded)."""
    win, _ = _live_window(layer, req)
    with rasterio.open(vrt_url(layer)) as src:
        return src.read(1, window=win).astype("int16")


def _read_mask(path: Path) -> np.ndarray:
    """Valid mask (True = data) of one raw-store layer file."""
    with rasterio.open(path) as ds:
        arr = ds.read(1)
    return arr != NODATA


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
            live = read_live(rep) != NODATA
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
                    "trusting the raw store (ADR 0001 decision 1)"
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
    reference by more than ``tol`` (ADR 0001 decision 1 fraction gate)."""
    return [
        f"{layer}: {measured[layer]:.4f} vs reference {reference[layer]:.4f}"
        for layer in measured
        if layer in reference and abs(measured[layer] - reference[layer]) > tol
    ]


def raw_reference_fractions(raw_dir: Path | None = None) -> dict[str, float]:
    """Per-layer valid-inside-Study-Area fractions recorded by the last
    verified fetch (raw Provenance Record)."""
    path = (raw_dir if raw_dir is not None else RAW_DIR) / "_metadata.json"
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


def _load_previous_records() -> tuple[dict[str, dict], str | None]:
    """Per-layer records and distribution file date from the existing raw
    Provenance Record, if there is one ({} if absent or unreadable)."""
    path = RAW_DIR / "_metadata.json"
    if not path.exists():
        return {}, None
    try:
        with open(path) as f:
            meta = json.load(f)
    except json.JSONDecodeError:
        return {}, None
    layers = {rec["name"]: rec for rec in meta.get("layers", []) if "name" in rec}
    return layers, meta.get("distribution", {}).get("file_last_modified")


def distribution_file_date() -> str | None:
    """``Last-Modified`` of one layer VRT: the vintage of the frozen 'latest'
    distribution (ADR 0001 decision 5)."""
    prop = LAYERS[0].split("_", 1)[0]
    req = urllib.request.Request(f"{BASE_URL}/{prop}/{LAYERS[0]}.vrt", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.headers.get("Last-Modified")
    except OSError as exc:
        print(f"warning: could not read the distribution file date: {exc}")
        return None


def fetch_raw(grid: GridSpec, *, force: bool = False) -> list[str]:
    """Ensure the raw store covers ``grid`` and passes the fetch gate.

    Idempotent: existing files whose extent covers the required IGH bounds
    are kept (their per-layer ``fetched_at`` survives). ``force=True``
    re-reads every layer from the distribution. Every run ends with the
    verification gate (see module docstring), so kept files are re-certified
    against the live source, never trusted on extent alone. Returns the
    list of layers (re)written.
    """
    req: tuple[float, float, float, float] = required_bounds_igh(grid)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    previous, prev_date = _load_previous_records()
    run_at = datetime.now(UTC).isoformat(timespec="seconds")
    written: list[str] = []

    def fetch_layer(layer: str) -> None:
        arr = _read_live(layer, req)
        _, win_transform = _live_window(layer, req)
        profile = dict(
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype="int16",
            crs=CRS_IGH,
            transform=win_transform,
            nodata=NODATA,
            compress="deflate",
        )
        out_path = RAW_DIR / f"{layer}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr, 1)
        if not file_covers(out_path, req):
            raise RuntimeError(f"{layer}: written window does not cover the required bounds")
        if layer not in written:
            written.append(layer)

    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(vrt_url(layer=LAYERS[0])) as first:
            win = window_outward(first.transform, req)  # which pixels cover our rectangle?
        print(
            f"raw window ≈{int(win.width)}×{int(win.height)} px "
            f"(≈{win.width * RES_M / 1000:.0f}×{win.height * RES_M / 1000:.0f} km, "
            f"unmasked rectangle, per-layer windows on each VRT's own grid)"
        )
        for layer in LAYERS:
            out_path = RAW_DIR / f"{layer}.tif"
            if out_path.exists() and not force and file_covers(out_path, req):
                continue
            fetch_layer(layer)
            print(f"fetched {layer}")

        gate_rewrites: list[str] = []

        def gate_refetch(layer: str) -> None:
            fetch_layer(layer)
            gate_rewrites.append(layer)
            print(f"gate refetch {layer}")

        masks = verify_raw_store(
            LAYERS,
            load_mask=lambda layer: _read_mask(RAW_DIR / f"{layer}.tif"),
            read_live=lambda layer: _read_live(layer, req),
            refetch=gate_refetch,
        )

    dist_date = distribution_file_date()

    # Per-layer valid fractions: window-wide, and inside the Study Area —
    # the latter is the recorded reference for later fetches and the
    # align-time consistency check (ADR 0001 decisions 1–2). Each layer's
    # Study Area mask is rasterized on that layer's own grid: the two
    # published grid families are offset by 3 px.
    layer_fractions: dict[str, dict] = {}
    for layer in LAYERS:
        with rasterio.open(RAW_DIR / f"{layer}.tif") as ds:
            layer_grid = GridSpec(
                crs=CRS_IGH, transform=ds.transform, width=ds.width, height=ds.height
            )
            layer_bounds = list(ds.bounds)
        inside = study_area_mask(layer_grid)
        layer_fractions[layer] = {
            "window_bounds_igh": layer_bounds,
            "valid_frac_window": float(masks[layer].mean()),
            "valid_frac_inside": float(masks[layer][inside].sum() / inside.sum()),
        }
    measured_inside = {layer: layer_fractions[layer]["valid_frac_inside"] for layer in LAYERS}
    reference = {
        layer: rec["valid_frac_inside"]
        for layer, rec in previous.items()
        if "valid_frac_inside" in rec
    }
    if reference and dist_date and prev_date == dist_date:
        violations = reference_fraction_violations(measured_inside, reference)
        if violations:
            raise RuntimeError(
                "valid-fraction gate failed (ADR 0001 decision 1): " + "; ".join(violations)
            )

    groups = group_by_mask(masks)
    print(
        f"gate passed: {len(groups)} mask group(s), sizes "
        f"{[len(g) for g in groups]}; every mask certified against a live second read"
    )

    metadata = {
        "source": "SoilGrids 2.0 (ISRIC), 'latest' WebDAV distribution",
        "run_at": run_at,
        "distribution": {
            "file_last_modified": dist_date,
            "note": "'latest' is a frozen snapshot; this date is its vintage",
        },
        "base_url": BASE_URL,
        "layer_inventory": LAYERS,
        "statistic": "mean",
        "grid": {
            "crs_proj4": CRS_IGH.to_proj4(),
            "resolution_m": RES_M,
            "bounding_policy": (
                "unmasked rectangle: target-grid extent transformed to IGH, "
                f"padded {PAD_M:g} m, snapped outward to whole pixels; per-layer "
                "windows on each VRT's own grid (two published grid families, "
                "origins 750 m apart — see per-layer window_bounds_igh)"
            ),
        },
        "nodata": NODATA,
        "value_encoding": "ISRIC int16, undecoded",
        "decode_factors": {
            prop: UNITS[prop][0] for prop in sorted({layer.split("_", 1)[0] for layer in LAYERS})
        },
        "verification": {
            "method": (
                "live second read of one representative per mask group, compared "
                "cell-for-cell (ADR 0001 decision 1); group members refetched on "
                "mismatch"
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
            for layer in LAYERS
        ],
    }
    with open(RAW_DIR / "_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    n_fetched = len(written)
    print(f"raw store ready ({n_fetched} written, {len(LAYERS)} verified)")
    return written


def align(
    grid: GridSpec,
    *,
    raw_dir: Path | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> tuple[Path, list[dict]]:
    """Warp every raw layer onto ``grid`` into one multiband float32 GeoTIFF.

    Bands carry the published layer names as descriptions; values are decoded
    physical units with NaN outside the Study Area. Returns the output path
    and per-band statistics for verification.
    """
    raw_dir = raw_dir if raw_dir is not None else RAW_DIR
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ALIGNED_DIR / ALIGNED_NAME
    mask = grid.mask()
    stats: list[dict] = []
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=grid.height,
        width=grid.width,
        count=len(LAYERS),
        dtype="float32",
        crs=grid.crs,
        transform=grid.transform,
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        bigtiff="if_safer",
    ) as dst:
        for i, layer in enumerate(LAYERS, start=1):
            arr = warp_band(
                raw_dir / f"{layer}.tif",
                grid,
                resampling=resampling,
                decode_factor=decode_factor(layer),
                mask=mask,
            )
            dst.write(arr, i)
            dst.set_band_description(i, layer)
            inside = mask & np.isfinite(arr)
            valid = arr[inside]
            stats.append(
                {
                    "layer": layer,
                    "unit": UNITS[layer.split("_", 1)[0]][1],
                    "valid_inside": int(valid.size),
                    "nan_inside_frac": float(1.0 - valid.size / mask.sum()),
                    "min": float(valid.min()),
                    "max": float(valid.max()),
                }
            )
    print(f"aligned stack written: {out_path} ({len(LAYERS)} bands)")
    return out_path, stats


def _raw_bilinear(layer: str, x: float, y: float) -> float:
    """Textbook bilinear of the raw store at an IGH coordinate, mirroring the
    warp kernel. Partial kernels (some of the 2×2 pixels nodata) are
    renormalized over the valid pixels — this is what the GDAL warper does
    (verified empirically 2026-11-04); NaN only when the whole kernel is
    nodata. Uses centre-based fractional indices — rasterio affines are
    edge-based, pixel centres sit at col + 0.5."""
    with rasterio.open(RAW_DIR / f"{layer}.tif") as ds:
        fc, fr = ~ds.transform * (x, y)
        uc, ur = fc - 0.5, fr - 0.5
        c0, r0 = math.floor(uc), math.floor(ur)
        vals = ds.read(1, window=Window(c0, r0, 2, 2)).astype("float64")
    valid = vals != NODATA
    if not valid.any():
        return math.nan
    dx, dy = uc - c0, ur - r0
    w = np.array([[(1 - dx) * (1 - dy), dx * (1 - dy)], [(1 - dx) * dy, dx * dy]])
    ww = w * valid
    return float((vals * ww).sum() / ww.sum() / decode_factor(layer))


def verify_aligned(
    path: Path,
    stats: list[dict],
    grid: GridSpec,
) -> list[dict]:
    """Run the post-align checks; returns a list of {name, ok, detail}."""
    checks: list[dict] = []

    with rasterio.open(path) as ds:
        fingerprint_ok = (
            ds.crs == grid.crs
            and ds.transform == grid.transform
            and ds.width == grid.width
            and ds.height == grid.height
            and list(ds.descriptions) == LAYERS
        )
    checks.append(
        {
            "name": "grid fingerprint",
            "ok": bool(fingerprint_ok),
            "detail": "CRS/transform/shape/band names",
        }
    )

    ph_idx = LAYERS.index("phh2o_0-5cm_mean")
    mask = grid.mask()
    with rasterio.open(path) as ds:
        ph = ds.read(ph_idx + 1)
    spread = max(s["valid_inside"] for s in stats) - min(s["valid_inside"] for s in stats)

    # Reference consistency (ADR 0001 decision 1): every band's valid fraction
    # inside the Study Area must reproduce the raw store's verified reference.
    # Replaces the refuted NaN=sea gates, which passed the artifacted fetch.
    try:
        references = raw_reference_fractions()
    except (OSError, KeyError, json.JSONDecodeError):
        references = {}
    measured = {s["layer"]: 1.0 - s["nan_inside_frac"] for s in stats}
    if references:
        violations = reference_fraction_violations(measured, references)
        worst = max(
            abs(measured[layer] - references[layer]) for layer in measured if layer in references
        )
        ok = not violations
        detail = (
            f"max deviation {worst:.4f} (tolerance ±{REF_TOL}); "
            f"valid-count spread across bands {spread}"
        )
        if violations:
            detail += "; " + "; ".join(violations)
    else:
        ok = False
        detail = (
            "raw Provenance Record lacks per-layer reference fractions — "
            "run a verified fetch (python -m bandstack --force)"
        )
    checks.append({"name": "reference consistency", "ok": ok, "detail": detail})

    # Modelling domain (ADR 0001 decision 3): cells where all bands are
    # finite. Informational — the domain is defined as exactly these cells.
    with rasterio.open(path) as ds:
        all_finite = np.ones(grid.shape, dtype=bool)
        for i in range(1, ds.count + 1):
            all_finite &= np.isfinite(ds.read(i))
    domain_frac = float(all_finite[mask].sum() / mask.sum())
    checks.append(
        {
            "name": "modelling domain",
            "ok": True,
            "all_bands_finite_frac": domain_frac,
            "detail": (
                f"all {len(LAYERS)} bands finite in {domain_frac:.1%} of "
                "Study Area cells (informational)"
            ),
        }
    )

    # City pH: compare the aligned value of the cell containing each city
    # against a manual bilinear of the raw store at that cell centre's IGH
    # position — an apples-to-apples test of the whole warp chain.
    lons = [lon for lon, _ in CITIES.values()]
    lats = [lat for _, lat in CITIES.values()]
    xs_t, ys_t = transform("EPSG:4326", grid.crs, lons, lats)
    rows, cols = rasterio.transform.rowcol(grid.transform, list(xs_t), list(ys_t))
    city_lines = []
    city_ok = True
    for (city, _), r, c in zip(CITIES.items(), rows, cols, strict=True):
        a = float(ph[r, c])
        cx = grid.transform.c + (c + 0.5) * grid.res
        cy = grid.transform.f - (r + 0.5) * grid.res
        xi, yi = transform(grid.crs, CRS_IGH, [cx], [cy])
        ref = _raw_bilinear("phh2o_0-5cm_mean", xi[0], yi[0])
        if city == "Brussels":
            ok = math.isnan(a)
            city_lines.append(f"{city}: aligned {a:.2f} (expected NaN — outside Study Area)")
        elif math.isnan(ref):
            ok = math.isnan(a)
            city_lines.append(f"{city}: aligned {a:.2f} (raw kernel touches water)")
        else:
            ok = abs(a - ref) <= INTERP_TOL
            city_lines.append(f"{city}: aligned {a:.2f} vs raw-interp {ref:.2f}")
        city_ok &= ok
    checks.append(
        {
            "name": "city pH spot check",
            "ok": city_ok,
            "detail": "; ".join(city_lines) + f" (tolerance ±{INTERP_TOL})",
        }
    )

    range_bad = []
    for s in stats:
        prop = s["layer"].split("_", 1)[0]
        lo, hi = PLAUSIBLE[prop]
        if not (lo <= s["min"] and s["max"] <= hi):
            range_bad.append(f"{s['layer']} [{s['min']:.2f}, {s['max']:.2f}] not in [{lo}, {hi}]")
    checks.append(
        {
            "name": "physical plausibility",
            "ok": not range_bad,
            "detail": (
                "; ".join(range_bad) if range_bad else "all bands within plausible decoded ranges"
            ),
        }
    )

    band_idx = {layer: i + 1 for i, layer in enumerate(LAYERS)}
    closure_lines = []
    closure_ok = True
    with rasterio.open(path) as ds:
        for depth in DEPTHS:
            parts = [ds.read(band_idx[f"{p}_{depth}cm_mean"]) for p in ("clay", "silt", "sand")]
            total = sum(np.where(np.isfinite(p), p, 0.0) for p in parts)
            valid = np.isfinite(parts[0]) & np.isfinite(parts[1]) & np.isfinite(parts[2])
            dev = np.abs(total[valid] - 1000.0)
            med, p99 = float(np.median(dev)), float(np.percentile(dev, 99))
            closure_ok &= med < CLOSURE_MEDIAN_MAX
            closure_lines.append(f"{depth}cm: median {med:.1f}, p99 {p99:.1f} g/kg")
    checks.append(
        {
            "name": "texture closure",
            "ok": closure_ok,
            "detail": "; ".join(closure_lines) + f" (gate: median < {CLOSURE_MEDIAN_MAX:g})",
        }
    )

    return checks
