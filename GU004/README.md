# CellSAM WSI Pipeline (GU004) — Research Prototype

This repository hosts a research-stage workflow for developing and validating
a CellSAM-based pipeline for whole-slide image (WSI) processing.

The current goal is to enable **reproducible experimentation and method sharing**
rather than providing an industrialized or production-ready system.

---

## Overview

This project explores a patch-based workflow for WSI analysis using CellSAM,
with emphasis on:

- Patch-level cell instance segmentation
- Mapping patch-level predictions back to the WSI coordinate space
- Exporting results in GeoJSON format for downstream visualization and analysis
  (e.g. in QuPath)

---

## Status

🚧 **Active research development**

Current focus:
- Exploratory development and validation
- Transition from notebooks to script-based structure (in progress)
- Improving reproducibility and clarity for collaboration and review

This repository is expected to evolve as experiments and design decisions
are finalized.

---

## Planned Workflow (Conceptual)

1. Load WSI or extracted regions / patches
2. Run CellSAM-based inference at the patch level
3. Aggregate predictions into WSI-level coordinates
4. Export results as QuPath-compatible GeoJSON



## Repository Structure (Planned)

```text
notebooks/    # exploratory analysis and validation
scripts/      # reproducible research scripts (WIP)
configs/      # configuration files for experiments
data/         # example inputs and outputs (optional)








Notes

This is a research prototype, not a finalized pipeline.

Interfaces, file formats, and scripts may change during development.

Documentation will be expanded as the workflow stabilizes.


