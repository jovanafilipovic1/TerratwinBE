"""Linda ("Landgebruik Vlaanderen") Source adapter: inventory, live reads.

Public interface: ``fetch_raw`` (ensure a verified raw store, thin call into
``bandstack.rawstore``). Alignment is a later step; it does not exist yet.

Linda is Flanders' actual-land-use raster: one band, one state year, whose
cell *values* are categorical class codes (1–19) with the legend published
as an SLD style — there are no per-property bands and nothing to decode.

Raw store bounding policy: the ADR-0005 window is the target-grid extent
padded 500 m and clamped to what the source publishes — VPO trimmed the
2025 raster to the Flemish Region bbox, which publishes *less* than the
padded window on the west, east and south. The clamped window therefore
covers the full published raster, so the raw store is the whole file: every
pixel is real published data and no polygon mask is applied at fetch time.

Access route: a whole-file download of the published GeoTIFF. The
distribution platform serves no HTTP Range requests (so no windowed
``/vsicurl`` reads), the WCS GetCoverage endpoint is broken (subset-parser
bug, 78 MB output cap) and answers are rate-limited — one download per
run, cached in a process-local temporary file that both the fetch and the
gate's live read use. The server exposes no ``Last-Modified``/ETag on the
version-pinned filename, so the distribution vintage is the sha256 of the
download: identical bytes under the same name must reproduce it, and any
drift fails the fraction gate.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS

from bandstack.grid import GridSpec
from bandstack.rawstore import ensure, window_outward

STATE_YEAR = 2025
LAYER = "lu_landgebruik_vlaa_2025_v3"

DATAPLATFORM_URL = "https://dataplatform.omgeving.vlaanderen.be/public/mercatornet"
DOWNLOAD_URL = f"{DATAPLATFORM_URL}/{LAYER}/{LAYER}?format=tiff"

WMS_URL = "https://www.mercator.vlaanderen.be/raadpleegdienstenmercatorpubliek/ows"
GETSTYLES_URL = (
    f"{WMS_URL}?service=WMS&version=1.1.1&request=GetStyles&layers=lu:{LAYER}&sldver=1.0.0"
)

LAMBERT72 = CRS.from_epsg(31370)
NODATA = -9999
RES_M = 10.0
PAD_M = 500.0

# The published 2025 v3 grid, pinned: the fetch asserts the download sits
# exactly on it (VPO trimmed this raster to the Flemish Region bbox; the
# 2022 state carried the wider house grid, origin (18 000, 248 000)).
PUBLISHED_BOUNDS = (22000.0, 153000.0, 259000.0, 245000.0)

# Class legend as published in the source's SLD style (verified against the
# live WMS GetStyles at every fetch). code -> (label, color).
CLASSES: dict[int, tuple[str, str]] = {
    1: ("Huizen en tuinen", "#ff0000"),
    2: ("Industrie", "#8400a8"),
    3: ("Commerciële doeleinden", "#ff00c5"),
    4: ("Diensten", "#002673"),
    5: ("Transportinfrastructuur", "#686868"),
    6: ("Recreatie", "#ffaa00"),
    7: ("Landbouwgebouwen en -infrastructuur", "#a87000"),
    8: ("Overige bebouwde terreinen", "#cccccc"),
    9: ("Overige onbebouwde terreinen", "#828282"),
    10: ("Actieve groeves", "#dfe6a9"),
    11: ("Luchthavens", "#df73ff"),
    12: ("Bos", "#267300"),
    13: ("Akker", "#ffffbe"),
    14: ("Grasland in landbouwgebruik", "#a3ff73"),
    15: ("Struikgewas", "#897044"),
    16: ("Braakliggend en duinen", "#ffd37f"),
    17: ("Water", "#005ce6"),
    18: ("Moeras", "#00a884"),
    19: ("Overige graslanden", "#82ca5b"),
}

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw" / "linda"


def required_bounds(
    grid: GridSpec,
    *,
    pad_m: float = PAD_M,
    published_bounds: tuple[float, float, float, float] = PUBLISHED_BOUNDS,
) -> tuple[float, float, float, float]:
    """The ADR-0005 fetch window for Linda: the target-grid extent padded
    ``pad_m``, clamped to what the source publishes (Linda 2025 publishes
    less than the padded window, and a fetch must not treat the unpublished
    margin as a coverage error)."""
    w, s, e, n = grid.bounds
    pw, ps, pe, pn = published_bounds
    req = (max(w - pad_m, pw), max(s - pad_m, ps), min(e + pad_m, pe), min(n + pad_m, pn))
    if not (req[0] < req[2] and req[1] < req[3]):
        raise ValueError(
            f"grid extent {grid.bounds} does not reach the published raster {published_bounds}"
        )
    return req


def parse_sld(xml: str) -> dict[int, tuple[str, str]]:
    """Extract the class legend (code -> (label, color)) from the source's
    SLD style document."""
    entries: dict[int, tuple[str, str]] = {}
    for e in ET.fromstring(xml).iter("{http://www.opengis.net/sld}ColorMapEntry"):
        entries[int(e.get("quantity"))] = (e.get("label"), e.get("color", "").lower())
    return entries


def live_legend(attempts: int = 3, delay_s: float = 10.0) -> dict[int, tuple[str, str]]:
    """The legend as currently served by the publisher's WMS GetStyles. The
    OWS endpoint is intermittently unavailable (503/timeout observed); retry
    a few times before failing the fetch."""
    req = urllib.request.Request(GETSTYLES_URL, headers={"User-Agent": "bandstack/0.1"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return parse_sld(resp.read().decode())
        except OSError as exc:
            if attempt == attempts:
                raise
            print(f"legend fetch failed ({exc}); retry {attempt}/{attempts - 1} in {delay_s:g}s")
            time.sleep(delay_s)
    raise AssertionError("unreachable")


def verify_legend() -> None:
    """Fail loudly unless the transcribed legend matches the publisher's
    live style: the class codes are the raw values, so a drifted legend
    would silently mislabel every downstream product."""
    live = live_legend()
    if live != CLASSES:
        raise RuntimeError(
            f"Linda legend drifted from the publisher's live SLD: {live} vs transcribed {CLASSES}"
        )
    print(f"legend verified against live SLD: {len(CLASSES)} classes")


_TMP_DIR: tempfile.TemporaryDirectory | None = None
_DOWNLOAD_CACHE: dict[str, tuple[Path, str]] = {}


def _cached_download() -> tuple[Path, str]:
    """The published GeoTIFF downloaded once per process, with its sha256.

    No Range support on the platform, so windowed reads are impossible —
    every reader (fetch, gate, vintage) shares this one download.
    """
    if DOWNLOAD_URL in _DOWNLOAD_CACHE:
        return _DOWNLOAD_CACHE[DOWNLOAD_URL]

    global _TMP_DIR
    _TMP_DIR = tempfile.TemporaryDirectory(prefix="linda_")
    path = Path(_TMP_DIR.name) / f"{LAYER}.tiff"
    digest = hashlib.sha256()
    req = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "bandstack/0.1"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(path, "wb") as f:
        for chunk in iter(lambda: resp.read(1 << 20), b""):
            digest.update(chunk)
            f.write(chunk)
    _DOWNLOAD_CACHE[DOWNLOAD_URL] = (path, digest.hexdigest())
    return _DOWNLOAD_CACHE[DOWNLOAD_URL]


def vintage() -> str | None:
    """Distribution vintage = sha256 of the published GeoTIFF download. The
    version-pinned name must serve identical bytes forever; a changed hash
    under the same name is drift and fails the fraction gate."""
    try:
        return _cached_download()[1]
    except OSError as exc:
        print(f"warning: could not hash the published file: {exc}")
        return None


def _assert_published_grid(
    ds: rasterio.DatasetReader, bounds: tuple[float, float, float, float]
) -> None:
    """The download must sit exactly on the pinned native grid."""
    if (
        ds.crs != LAMBERT72
        or ds.res != (RES_M, RES_M)
        or tuple(round(v, 6) for v in ds.bounds) != tuple(round(v, 6) for v in bounds)
    ):
        raise RuntimeError(
            f"{LAYER}: download grid changed — crs {ds.crs}, res {ds.res}, "
            f"bounds {tuple(ds.bounds)} (expected EPSG:31370, {RES_M:g} m, {bounds})"
        )


def fetch_raw(
    grid: GridSpec,
    *,
    raw_dir: Path | None = None,
    published_bounds: tuple[float, float, float, float] | None = None,
    force: bool = False,
) -> list[str]:
    """Ensure the Linda raw store covers ``grid`` and passes the fetch gate.

    Thin adapter over ``rawstore.ensure``: the generic process (idempotent
    fetch, gate, fraction gate, Provenance Record) lives there; everything
    here is Linda — the legend check against the live SLD, one whole-file
    download per run shared by fetch and gate, and the class-code assert.
    Returns the list of layers (re)written.
    """
    verify_legend()
    pub = PUBLISHED_BOUNDS if published_bounds is None else published_bounds
    req = required_bounds(grid, published_bounds=pub)

    def fetch_one(layer: str) -> tuple[np.ndarray, dict]:
        path, _ = _cached_download()
        with rasterio.open(path) as src:
            _assert_published_grid(src, pub)
            win = window_outward(src.transform, req)
            arr = src.read(1, window=win)
            profile = dict(
                driver="GTiff",
                height=arr.shape[0],
                width=arr.shape[1],
                count=1,
                dtype="int32",
                crs=src.crs,
                transform=src.window_transform(win),
                nodata=NODATA,
                compress="deflate",
            )
        observed = set(np.unique(arr).tolist()) - {NODATA}
        unknown = observed - set(CLASSES)
        if unknown:
            raise RuntimeError(
                f"{layer}: unexpected class codes {sorted(unknown)} (legend: 1–{max(CLASSES)})"
            )
        return arr, profile

    with rasterio.open(_cached_download()[0]) as src:
        win = window_outward(src.transform, req)
        print(
            f"raw window ≈{int(win.width)}×{int(win.height)} px of the published "
            f"{src.width}×{src.height} (whole published raster; padded window clamped "
            "to what the source publishes)"
        )

    return ensure(
        layers=[LAYER],
        raw_dir=raw_dir if raw_dir is not None else RAW_DIR,
        required_bounds=req,
        fetch_one=fetch_one,
        nodata=NODATA,
        vintage=vintage,
        bounds_key="window_bounds_l72",
        force=force,
        record=dict(
            source=(
                f"Landgebruik - Vlaanderen - toestand {STATE_YEAR} v3 "
                "(Departement Omgeving, MercatorNet)"
            ),
            distribution_note=(
                "version-pinned dataset name; the platform serves no Last-Modified/ETag, "
                "so the vintage is the sha256 of the downloaded GeoTIFF"
            ),
            base_url=DOWNLOAD_URL,
            statistic=f"n/a — categorical land use, single state year ({STATE_YEAR})",
            grid={
                "crs": "EPSG:31370",
                "resolution_m": RES_M,
                "bounding_policy": (
                    "full published raster: the target-grid extent padded "
                    f"{PAD_M:g} m, clamped to the published extent (the source "
                    "publishes less than the padded window on the west, east and "
                    "south); no polygon mask at fetch time"
                ),
            },
            value_encoding={
                "values": "int32 class codes as published, undecoded",
                "nodata": NODATA,
                "legend_check": "transcribed legend verified against the live WMS GetStyles SLD",
                "classes": {
                    str(code): {"label": label, "color": color}
                    for code, (label, color) in sorted(CLASSES.items())
                },
            },
            decode_factors={},
        ),
    )
