# Bandstack catalog

This README documents the published files and the sources behind them. Two
files ship alongside it:

- `be2_flemish_region_4326.geojson` — the Study Area geometry
- `soilgrids_lambert72_200m.tif` — SoilGrids soil properties for the Study
  Area on the project's common grid

Each source (SoilGrids, Linda, LUCAS, climate, ...) is fetched into a raw
store on its native grid, then warped onto one common 200 m Belgian Lambert 72
grid masked to the Study Area. The published GeoTIFF holds the 31 SoilGrids
layers of that aligned product; further sources will be merged into the full
bandstack later (see [Planned sources](#planned-sources-not-yet-fetched)).

## The published GeoTIFF

`soilgrids_lambert72_200m.tif` is a single multiband GeoTIFF: one band per
SoilGrids layer, band names embedded as GDAL band descriptions (visible in
QGIS and most tools).

| | |
|---|---|
| Content | 31 layers: 10 soil properties at depths 0–5, 5–15, 15–30 cm, plus `ocs` (0–30 cm only); mean statistic only |
| Grid | EPSG:31370 (Belgian Lambert 72), 200 m square cells, 1195 × 465 columns × rows (x 21000–260000, y 152000–245000) |
| Encoding | float32 in decoded physical units — **no scaling needed**; NaN = nodata |
| Band order | Alphabetical by property (`bdod` … `soc`), depths in order within each property, `ocs_0-30cm_mean` appended last. Band 1 = `bdod_0-5cm_mean`, band 31 = `ocs_0-30cm_mean` |
| Units | Per property in the [Properties](#properties) table below |
| NaN meaning | Outside the polygon: Study Area mask. Inside (~17.7 % of cells): gaps SoilGrids itself publishes as nodata (built-up, water) — a permanent source characteristic, not an error |
| Caveats | Bilinearly resampled from 250 m native cells — effective resolution is 250 m despite the 200 m grid. Vintage: ISRIC "latest" snapshot (directory listing dated 2023-02-01) |

Band names programmatically:

```python
import rasterio

with rasterio.open("soilgrids_lambert72_200m.tif") as src:
    print(src.descriptions)  # ('bdod_0-5cm_mean', ..., 'ocs_0-30cm_mean')
```

## Study Area geometry

| | |
|---|---|
| Role | Mask geometry for every raster source — the exact polygon the published GeoTIFF is masked to; never a bandstack layer |
| Definition | **Flemish Region** (NUTS `BE2`), excluding Brussels|
| Source | GISCO NUTS 2021, scale 1:1M, level 1, EPSG:4326 |
| URL | `https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_01M_2021_4326_LEVL_1.geojson` |
| License | European Commission reuse policy, attribution required; boundaries © EuroGeographics |
| Local | `data/raw/gisco/be2_flemish_region_4326.geojson` (filtered from the cached full LEVL-1 file) |

## SoilGrids 2.0 (ISRIC)

| | |
|---|---|
| What | Global soil property maps at 250 m, fitted by ensemble regression kriging (Hengl et al. 2017; Poggio et al. 2021) |
| URL | `https://files.isric.org/soilgrids/latest/data/` |
| Version | "latest" distribution, directory listing dated 2023-02-01; exact fetch timestamp in `_metadata.json` |
| Access | Windowed `/vsicurl` reads of the published COG VRTs |
| Native grid | Interrupted Goode Homolosine on a **sphere** (R = 6371000 m), 250 m square cells|
| Values | int16, exactly as published; decode factors below |
| License | CC-BY 4.0 (ISRIC) |
| Local | `data/raw/soilgrids/` — GeoTIFFs (mean statistic only); layer inventory and fetch timestamp in the `_metadata.json` sidecar |

### Properties

The "Decoded unit" column is what the published GeoTIFF contains. The factors
apply to ISRIC's raw int16 storage in `data/raw/soilgrids/` only — the
published GeoTIFF is already decoded; do not rescale it.

| Property | Description | Decoded unit | Factor |
|---|---|---|---|
| bdod | Bulk density of the fine earth fraction | g/cm³ | 100 |
| cec | Cation exchange capacity | cmolc/kg | 10 |
| cfvo | Coarse fragments (> 2 mm) | g/kg | 1 |
| clay | Clay fraction (< 2 µm) | g/kg | 1 |
| nitrogen | Total nitrogen | g/kg | 100 |
| ocd | Organic carbon density | g/dm³ | 10 |
| ocs | Organic carbon stock (0–30 cm only) | t/ha | 10 |
| phh2o | pH measured in H₂O | pH | 10 |
| sand | Sand fraction (50–2000 µm) | g/kg | 1 |
| silt | Silt fraction (2–50 µm) | g/kg | 1 |
| soc | Soil organic carbon content | g/kg | 10 |

### Depths and statistics

- Depths fetched: 0–5, 5–15, 15–30 cm, plus `ocs` (0–30 cm only). **Decision**:
  the biodiversity target samples are taken at 0–30 cm depth, so the
  deeper published standard depths (30–60, 60–100, 100–200 cm) were not fetched
- Statistics published: mean, Q0.05, Q0.5, Q0.95, uncertainty — **only mean is
  fetched**


## Landgebruik Vlaanderen — toestand 2025 ("Linda")

| | |
|---|---|
| What | Actual land use per 10 m cell — one band, cell values are categorical class codes, 19 classes (legend below) |
| URL | `https://dataplatform.omgeving.vlaanderen.be/public/mercatornet/lu_landgebruik_vlaa_2025_v3/lu_landgebruik_vlaa_2025_v3?format=tiff` |
| Version | `lu_landgebruik_vlaa_2025_v3` — version-pinned name, published 2026-06; the platform serves no Last-Modified/ETag, so the Provenance Record pins the download's sha256 as the vintage |
| Access | Whole-file GeoTIFF download. **Decision**: no WCS (GetCoverage broken on the endpoint and capped at 78 MB) and no windowed `/vsicurl` reads (no Range support); answers are rate-limited, so the fetch downloads once per run |
| Native grid | EPSG:31370, 10 m, 23 700 × 9 200, origin (22 000, 245 000) — extent trimmed to the Flemish Region bbox (earlier state years carry a wider house grid) |
| Values | int32 class codes 1–19, undecoded; nodata −9999 |
| License | Flemish open data — free reuse with attribution (Departement Omgeving / MercatorNet) |
| Local | `data/raw/linda/` — the full published raster (the padded fetch window exceeds the published extent, so nothing is cut); inventory, sha and fractions in the `_metadata.json` sidecar |

### Classes

Fetched as published (Dutch labels); English glosses are ours, for reading
only. The legend is verified against the publisher's live SLD style at
every fetch.

| Code | Label | Gloss |
|---|---|---|
| 1 | Huizen en tuinen | Houses and gardens |
| 2 | Industrie | Industry |
| 3 | Commerciële doeleinden | Commercial |
| 4 | Diensten | Services |
| 5 | Transportinfrastructuur | Transport infrastructure |
| 6 | Recreatie | Recreation |
| 7 | Landbouwgebouwen en -infrastructuur | Farm buildings and infrastructure |
| 8 | Overige bebouwde terreinen | Other built-up land |
| 9 | Overige onbebouwde terreinen | Other unbuilt land |
| 10 | Actieve groeves | Active quarries |
| 11 | Luchthavens | Airports |
| 12 | Bos | Forest |
| 13 | Akker | Arable land |
| 14 | Grasland in landbouwgebruik | Grassland in agricultural use |
| 15 | Struikgewas | Shrubs |
| 16 | Braakliggend en duinen | Bare land and dunes |
| 17 | Water | Water |
| 18 | Moeras | Marsh |
| 19 | Overige graslanden | Other grasslands |

### State year

**Decision**: fetch the single latest state year only (2025 v3), the
phase-1 scope; earlier state years (2013–2022) are not fetched. Distance
and fraction predictors will be derived from this one state; a multi-year
land-use history is a possible later extension.


## Planned sources (not yet fetched)

| Source | Content | Native grid | Status |
|---|---|---|---|
| LUCAS-derived EU rasters (ESDAC) | Topsoil properties interpolated from LUCAS points | EPSG:3035, 500 m | planned — Predictor candidates |
| Climate | temporal climate predictors | varies, ~1 km+ | planned |
...
