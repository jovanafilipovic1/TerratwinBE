# TerratwinBE

The Belgian use case of the TerraTwin project: predicting soil-related
biodiversity from environmental predictors, and assessing compound-flooding
impact on it under extreme climate change scenarios.

## Language

**Study Area**:
The geographic region covered by the bandstack and models: the Flemish
Region only (NUTS BE2), excluding Brussels — "Flanders" in the institutional
sense includes Brussels and is therefore the wrong extent.
_Avoid_: Belgium ("Belgian" describes our role in the TerraTwin consortium, not the extent); Flanders when meaning the Flemish Region plus Brussels

**Source**:
An external dataset fetched into the raw store on its native grid, e.g.
SoilGrids or Linda. Its layers become Predictors only by an explicit
selection decision.
_Avoid_: dataset (vague), provider

**Bandstack**:
The single stack of raster layers on one common grid covering the Study Area;
one band per Predictor. The deliverable of project phase 1. Assembled by
merging every Aligned Source; no individual source file is itself the
Bandstack.
_Avoid_: cube, raster stack

**Aligned Source**:
One Source's layers warped onto the common grid, ready to be merged into the
Bandstack. An intermediate product of assembly, not the deliverable.
_Avoid_: bandstack as a name for per-source files

**Predictor**:
An environmental variable used as model input, e.g. soil pH at 0–5 cm or
distance to nearest road. Static predictors change on geologic timescales;
temporal predictors vary within the modelling period.
_Avoid_: feature (reserved for the ML modelling phase), covariate, band (a band is a Predictor's slot in the Bandstack)

**Raw Store**:
The store of fetched Sources, each on its native grid, exactly as published
and undecoded. Alignment onto the common grid happens later; a raw store is
never masked or resampled in place.
_Avoid_: raw data (vague), downloads

**Fetch Artifact**:
A nodata cell written by our own fetch, not a gap in the published Source
— e.g. a miswindowed read or a failed tile read. Invisible at write time;
caught only by the fetch gate (live second read, cell-for-cell) or a drift
from the Provenance Record reference fractions. A verified re-fetch
removes it.
_Avoid_: gap (reserved for published gaps)

**Published Gap**:
A cell the Source itself publishes as nodata (e.g. SoilGrids' built-up and
water masking). A permanent Source characteristic, recorded once per layer
in the Provenance Record; re-fetching never fills it, and the modelling
domain excludes it wherever any Predictor has one.
_Avoid_: fetch artifact (that one is ours and fixable)

**Catalog**:
The human-maintained inventory of Sources (`data/README.md`): what we fetch
and why, and licenses. Records
decisions only — counts, dates, and inventories live in the Provenance
Records, which the Catalog cites.
_Avoid_: metadata (the machine records are Provenance Records), data dictionary

**Provenance Record**:
The machine-written `_metadata.json` sidecar beside a raw store or Aligned
Source: the authoritative record of what one fetch or align run produced —
grid, layers, decode factors, per-band statistics, verification. Overwritten
by the next run; deliberately not a history.
_Avoid_: catalog (that is the human file), README
