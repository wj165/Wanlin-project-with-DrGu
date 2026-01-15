GU003 — StarDist → QuPath Pipeline

Author: Wanlin Jiang
Mentor: Dr. Quincy Gu

Project Overview

This repository contains the working pipeline for:

✔ Running StarDist on histopathology ROIs
✔ Extracting nuclei polygons
✔ Exporting polygons to GeoJSON
✔ Importing GeoJSON into QuPath for visualization and analysis

Two-step (split) pipeline works 100% correctly.

Repository Structure
GU003-Stardist-QuPath/
│
├── stardist_step1_predict.py      # Step 1 — Run StarDist & save raw polygons
├── save_stardist_geojson.py       # Step 2 — Convert polygons → GeoJSON
│
├── stardist_output1.geojson       # Final verified GeoJSON output
├── DHMC_0039_ROI.tif              # Example ROI used for validation
│
└── README.md                      # Project documentation

Two-Step Pipeline
Step 1 — Run StarDist on ROI
python stardist_step1_predict.py


Outputs:

Raw polygon coordinates (Python dict)

Number of detected nuclei

Saved polygon data

Step 2 — Convert to GeoJSON
python save_stardist_geojson.py


Outputs:

Valid GeoJSON file (stardist_output1.geojson)

Import successfully into QuPath



## Known issue (for debugging)

When importing exported GeoJSON into QuPath, all objects may collapse
to the top-left corner of the image. This is likely due to a coordinate
system mismatch (pixel vs WSI space, origin, or scaling).

See `stardist_oneclick_Wanlin_v3.py` for the reproducible example.




# GU003 – StarDist nucleus segmentation (WSI batch pipeline)

This repository contains the GU003 StarDist pipeline for nucleus segmentation
on H&E whole-slide images (WSI), with validated GeoJSON export and centroid mapping.

## Main notebook (recommended entry point)

- **notebooks/GU003_StarDist_100_patch_geojson.ipynb**

Batch pipeline:
- Randomly sample ~100 patches from one WSI
- Exclude white-background patches
- Run StarDist instance segmentation
- Export nucleus polygons to GeoJSON
- Compute centroids and map them back to WSI level-0 coordinates
- Python-only QC visualization (no QuPath required)

## Supporting notebook

- **notebooks/GU003_StarDist_single_patch_geojson.ipynb**

Single-patch validation used to establish correct preprocessing,
model settings, and GeoJSON alignment.

## Model & settings

- Model: StarDist2D (`2D_versatile_he`)
- Patch size: 512 × 512
- Preprocessing: RGB channel-wise normalization
- Output: Instance-level polygons + centroids (GeoJSON)

## Downstream usage

- QuPath import
- Nucleus morphology feature extraction
- Spatial analysis at WSI scale




---

## GU003 Task 1 – Single-patch cell-level feature extraction

In addition to GeoJSON export and QuPath visualization, this repository now includes a validated pipeline for **cell-level feature extraction from a single image patch**.

Pipeline:
StarDist GeoJSON → polygon-to-mask → scikit-image regionprops → cell-level CSV

Key files:
- `GU003_StarDist_single_patch_cell_features.ipynb`
- `cell_features_single_patch.csv`

The extracted features include area, perimeter, centroid, bounding box, equivalent diameter, eccentricity, solidity, extent, and derived ratios.  
This output serves as the baseline for future spatial feature computation and multi-patch scaling.

