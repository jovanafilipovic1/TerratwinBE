"""SoilGrids Source adapter: inventory, live reads, alignment, verification.

Public interface: ``fetch_raw`` (ensure a verified raw store, thin call into
``bandstack.rawstore``) and ``ensure_aligned`` (align → verify → Provenance
Record in one call).

Raw store bounding policy: the fetch window is the target-grid extent
(Study Area bbox + buffer, Lambert 72) transformed into SoilGrids' native
IGH grid, padded and snapped *outward* to whole source pixels. No polygon
mask is applied at fetch time — every pixel is real published data, so the
bilinear warp kernels of border cells are fully populated. The Study Area
mask is applied once, at align time, on the target grid.

The fetch process around this adapter — idempotent fetch, the fetch gate
(live second read of one representative per mask group), the fraction
gate, and the raw Provenance Record — is owned by ``bandstack.rawstore``;
this module supplies the SoilGrids specifics through its interface.
"""

from __future__ import annotations

import json
import math
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, transform, transform_bounds
from rasterio.windows import Window

from bandstack.align import warp_band
from bandstack.grid import GridSpec
from bandstack.rawstore import (
    REF_TOL,
    ensure,
    raw_reference_fractions,
    reference_fraction_violations,
    window_outward,
)

BASE_URL = "https://files.isric.org/soilgrids/latest/data"
CRS_IGH = CRS.from_proj4("+proj=igh +R=6371000 +units=m +no_defs")
NODATA = -32768
RES_M = 250.0
PAD_M = 500.0

REPO_ROOT = Path(__file__).resolve().parents[3]
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

# Verification tolerances (align-time checks): city spot check vs raw
# interpolation, and the texture-closure median gate.
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


def distribution_file_date() -> str | None:
    """``Last-Modified`` of one layer VRT: the vintage of the frozen 'latest'
    distribution (frozen snapshot)."""
    prop = LAYERS[0].split("_", 1)[0]
    req = urllib.request.Request(f"{BASE_URL}/{prop}/{LAYERS[0]}.vrt", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.headers.get("Last-Modified")
    except OSError as exc:
        print(f"warning: could not read the distribution file date: {exc}")
        return None


def fetch_raw(grid: GridSpec, *, force: bool = False) -> list[str]:
    """Ensure the SoilGrids raw store covers ``grid`` and passes the fetch gate.

    Thin adapter over ``rawstore.ensure``: the generic process (idempotent
    fetch, gate, fraction gate, Provenance Record) lives there; everything
    here is SoilGrids — live reads windowed on each layer's own IGH grid,
    the nodata sentinel, and the Record prose. Returns the list of layers
    (re)written.
    """
    req: tuple[float, float, float, float] = required_bounds_igh(grid)

    def fetch_one(layer: str) -> tuple[np.ndarray, dict]:
        """One live read of ``layer`` over ``req`` (int16, undecoded), on the
        layer's *own* VRT grid: the published property VRTs do not share one
        grid — the bdod/soc family and the 9-property family differ by 750 m
        (3 px) in origin — so every layer must be windowed on its own
        transform."""
        with rasterio.open(vrt_url(layer)) as src:
            win = window_outward(src.transform, req)
            arr = src.read(1, window=win).astype("int16")
            profile = dict(
                driver="GTiff",
                height=arr.shape[0],
                width=arr.shape[1],
                count=1,
                dtype="int16",
                crs=CRS_IGH,
                transform=src.window_transform(win),
                nodata=NODATA,
                compress="deflate",
            )
        return arr, profile

    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(vrt_url(LAYERS[0])) as first:
            win = window_outward(first.transform, req)  # which pixels cover our rectangle?
        print(
            f"raw window ≈{int(win.width)}×{int(win.height)} px "
            f"(≈{win.width * RES_M / 1000:.0f}×{win.height * RES_M / 1000:.0f} km, "
            f"unmasked rectangle, per-layer windows on each VRT's own grid)"
        )

    return ensure(
        layers=LAYERS,
        raw_dir=RAW_DIR,
        required_bounds=req,
        fetch_one=fetch_one,
        nodata=NODATA,
        vintage=distribution_file_date,
        env=GDAL_ENV,
        bounds_key="window_bounds_igh",
        force=force,
        record=dict(
            source="SoilGrids 2.0 (ISRIC), 'latest' WebDAV distribution",
            distribution_note="'latest' is a frozen snapshot; this date is its vintage",
            base_url=BASE_URL,
            statistic="mean",
            grid={
                "crs_proj4": CRS_IGH.to_proj4(),
                "resolution_m": RES_M,
                "bounding_policy": (
                    "unmasked rectangle: target-grid extent transformed to IGH, "
                    f"padded {PAD_M:g} m, snapped outward to whole pixels; per-layer "
                    "windows on each VRT's own grid (two published grid families, "
                    "origins 750 m apart — see per-layer window_bounds_igh)"
                ),
            },
            value_encoding="ISRIC int16, undecoded",
            decode_factors={
                prop: UNITS[prop][0]
                for prop in sorted({layer.split("_", 1)[0] for layer in LAYERS})
            },
        ),
    )


def align(
    grid: GridSpec,
    *,
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> tuple[Path, list[dict]]:
    """Warp every raw layer onto ``grid`` into one multiband float32 GeoTIFF.

    Bands carry the published layer names as descriptions; values are decoded
    physical units with NaN outside the Study Area. Returns the output path
    and per-band statistics for verification.
    """
    raw_dir = raw_dir if raw_dir is not None else RAW_DIR
    out_dir = out_dir if out_dir is not None else ALIGNED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ALIGNED_NAME
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


def _raw_bilinear(layer: str, x: float, y: float, *, raw_dir: Path) -> float:
    """Textbook bilinear of the raw store at an IGH coordinate, mirroring the
    warp kernel. Partial kernels (some of the 2×2 pixels nodata) are
    renormalized over the valid pixels — this is what the GDAL warper does
    (verified empirically 2026-11-04); NaN only when the whole kernel is
    nodata. Uses centre-based fractional indices — rasterio affines are
    edge-based, pixel centres sit at col + 0.5."""
    with rasterio.open(raw_dir / f"{layer}.tif") as ds:
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
    *,
    raw_dir: Path | None = None,
) -> list[dict]:
    """Run the post-align checks; returns a list of {name, ok, detail}."""
    raw_dir = raw_dir if raw_dir is not None else RAW_DIR
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

    # Reference consistency: every band's valid fraction
    # inside the Study Area must reproduce the raw store's verified reference.
    # Replaces the refuted NaN=sea gates, which passed the artifacted fetch.
    try:
        references = raw_reference_fractions(raw_dir)
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

    # Modelling domain: cells where all bands are
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
        ref = _raw_bilinear("phh2o_0-5cm_mean", xi[0], yi[0], raw_dir=raw_dir)
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


def _grid_fingerprint(grid: GridSpec) -> dict:
    """The target grid as recorded in the Aligned Source Provenance Record."""
    return {
        "crs": grid.crs.to_string(),
        "transform": [
            grid.transform.a,
            grid.transform.b,
            grid.transform.c,
            grid.transform.d,
            grid.transform.e,
            grid.transform.f,
        ],
        "width": grid.width,
        "height": grid.height,
    }


def ensure_aligned(
    grid: GridSpec,
    *,
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
) -> list[dict]:
    """Align, verify, and record the SoilGrids Aligned Source in one call.

    Warps the raw store onto ``grid`` (``align``), prints the per-band
    summary, runs the post-align checks (``verify_aligned``), and writes the
    Aligned Source Provenance Record beside the stack — the Record is owned
    by the code that aligns. Returns the checks; the caller decides what to
    do with failures.
    """
    path, stats = align(grid, raw_dir=raw_dir, out_dir=out_dir)

    print("\nper-band summary (inside Study Area mask):")
    for s in stats:
        print(
            f"  {s['layer']:26s} valid {1 - s['nan_inside_frac']:6.1%} "
            f"[{s['min']:8.2f}, {s['max']:8.2f}] {s['unit']}"
        )

    checks = verify_aligned(path, stats, grid, raw_dir=raw_dir)

    sidecar = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "grid": _grid_fingerprint(grid),
        "source": "SoilGrids 2.0 (ISRIC) via data/raw/soilgrids (see raw _metadata.json)",
        "resampling": "bilinear (one warp per layer, native IGH -> target grid)",
        "encoding": "float32, decoded physical units, NaN nodata",
        "mask": "Study Area (GISCO NUTS 2021 BE2), all_touched=True, applied at align",
        "bands": stats,
        "verification": checks,
    }
    sidecar_path = path.parent / "_metadata.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"\nsidecar written: {sidecar_path}")
    return checks
