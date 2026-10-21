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

**Bandstack**:
The single stack of raster layers on one common grid covering the Study Area;
one band per Predictor. The deliverable of project phase 1.
_Avoid_: cube, raster stack

**Predictor**:
An environmental variable used as model input, e.g. soil pH at 0–5 cm or
distance to nearest road. Static predictors change on geologic timescales;
temporal predictors vary within the modelling period.
_Avoid_: feature (reserved for the ML modelling phase), covariate, band (a band is a Predictor's slot in the Bandstack)
