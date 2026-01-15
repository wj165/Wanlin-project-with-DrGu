# Variable Mapping: Notebook → Scripts

This document records the 1:1 mapping between variables used in the
exploratory notebook (`wsi_patch_extraction_colab.ipynb`) and the
script-level interfaces defined in `run_inference.py`.

The purpose of this file is to ensure reproducibility and transparency
during notebook-to-script migration.

---

## Source Notebook

- notebooks/wsi_patch_extraction_colab.ipynb

---

## run_patch_inference

| Notebook Variable | Script Argument | Notes |
|-------------------|-----------------|-------|
| slide             | image           | OpenSlide / WSI handle |
| patch             | image           | extracted image tile |
| tile_size         | config["tile_size"] | patch size in pixels |
| level             | config["level"] | pyramid level used |
| device            | config["device"] | CPU / GPU selection |

---

## aggregate_to_wsi

| Notebook Variable | Script Argument | Notes |
|-------------------|-----------------|-------|
| patch_coords      | patch_outputs  | patch-level spatial outputs |
| wsi_metadata      | metadata       | WSI coordinate system |
| downsample        | metadata       | level-to-level scaling |
| slide_dimensions  | metadata       | full-resolution size |

---

## Notes

- Variable names reflect current usage in the notebook.
- Mappings may be refined as logic is migrated into scripts.
- No computational logic is implemented at this stage.
