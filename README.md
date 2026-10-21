# TerratwinBE
The Belgian use case of the TerraTwin project: Impact of Compound Flooding on Soil-Related Biodiversity under Extreme Climate Change Scenarios in Flanders

## Project phases
1. **Predictor data & bandstack** — automated pipeline building a ~200 m
   bandstack of soil-biodiversity predictor variables for Flanders
   (land cover, soil composition, distance-to-land-use, ...).
2. ML model predicting biodiversity classes/indices (later).
3. Flood modelling (later).

## Repository layout

- `data/` — raw data store; see [data/README.md](data/README.md) for the
  source catalog (contents are gitignored, the catalog is not)
- `notebooks/` — per-source exploration and fetch notebooks
- `src/bandstack/` — pipeline code that maps the environmental predictors onto a common 200m grid

## Setup
Requires [uv](https://docs.astral.sh/uv/).
