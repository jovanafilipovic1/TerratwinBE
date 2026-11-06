# TerratwinBE
The Belgian use case of the TerraTwin project: Impact of Compound Flooding on Soil-Related Biodiversity under Extreme Climate Change Scenarios in the Flemish Region

## Project phases
1. **Predictor data & bandstack** — automated pipeline building a 200 m
   Bandstack of soil-biodiversity Predictor variables for the Study Area
   (the Flemish Region, NUTS BE2) on a Belgian Lambert 72 grid
   (land cover, soil composition, distance-to-land-use, ...).
2. ML model predicting biodiversity classes/indices (later).
3. Flood modelling (later).

## Repository layout

- `data/` — raw and aligned data stores; see [data/README.md](data/README.md)
  for the Catalog (contents are gitignored, the catalog is not)
- `notebooks/` — per-source exploration notebooks
- `src/bandstack/` — the pipeline: fetches Sources into the raw store, aligns
  them onto the common grid, and assembles the Bandstack
- `tests/` — pytest suite (fetch gate, grid, alignment, sources)

## Setup
Requires [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

```bash
uv run python -m bandstack
```

Add `--force` to re-read every raw layer from its Source even if the stored
files already cover the required bounds.
