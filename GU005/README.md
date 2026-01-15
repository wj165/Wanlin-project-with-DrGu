# GU005_ClassPose
ClassPose single-patch pipeline: WSI patch → GeoJSON → instance overlay

# GU005_ClassPose

Single-patch ClassPose pipeline for digital pathology.

## Scope
- Extract a single patch from WSI
- Run ClassPose
- Convert outputs to GeoJSON
- Rasterize polygons to instance masks
- Generate StarDist-style instance overlays

## Notes
- Current focus: visualization and output interpretation
- QuPath metadata is optional and not required for overlay



## Current observation

When visualizing ClassPose outputs as instance overlays on a single WSI patch,
the resulting instances appear as relatively coarse, block-like regions rather
than fine nucleus-level contours. This likely reflects the granularity of the
polygons produced by ClassPose rather than an issue with the overlay pipeline
itself.

An example comparison with a StarDist-style overlay is provided in `examples/`.

