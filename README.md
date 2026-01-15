# Wanlin–DrGu Cell Segmentation Projects

This repository consolidates a series of WSI-based image analysis projects
conducted with Dr. Gu, organized as GU001–GU005.  
Each GU project corresponds to a clearly defined dataset, goal, and set of deliverables.
Together, they form a progressive pipeline from patch-level classification
to instance- and semantic-aware nuclei segmentation with GeoJSON export.

---

## Project Overview

| Project ID | Start Day | Name | Dataset | Main Goal | Dr. Gu Split | Your Deliverables |
|----------|----------|------|--------|-----------|-------------|------------------|
| **GU001** | 11/25/25 | Kidney: Papillary vs ccRCC | DHMC Kidney | Classification pipeline | 84/21 train/val each + 45/398 test | Split table, scripts, baseline model, slides |
| **GU002** | 11/25/25 | Lung: Solid vs Acinar | DHMC Lung | Pattern classification | Solid 28/8/15; Acinar 28/8/23 | Patch pipeline, baseline results, slides |
| **GU003** | 11/25/25 | WSI StarDist + GeoJSON | Lung WSI | Nuclei segmentation → GeoJSON | 1 WSI only | StarDist results, GeoJSON export attempt, error logs, GitHub repo, slides |
| **GU004** | 12/16/25 | WSI CellSAM + GeoJSON | Lung WSI | Generalist nuclei segmentation → GeoJSON (baseline comparison) | 1 WSI only (same as GU003) | CellSAM results, GeoJSON export, StarDist vs CellSAM visual comparison, inference notes, GitHub repo, slides |
| **GU005** | 01/03/26 | WSI ClassPose + Semantic GeoJSON | Lung WSI | Semantic-aware nuclei segmentation (instance + cell type) → GeoJSON | 1 WSI only (same as GU003/GU004) | ClassPose inference results, cell-type–annotated GeoJSON, centroid alignment checks, CellSAM vs ClassPose qualitative comparison, export schema notes, GitHub repo, slides |

---

## Repository Structure

```text
Wanlin-DrGu-CellSegmentation/
├── GU001-kidney/
├── GU002-lung/
├── GU003-stardist/
├── GU004-cellsam/
├── GU005-classpose/
├── common/
├── docs/
└── README.md
# Wanlin-project-with-DrGu
Unified repository for WSI-based cell segmentation projects with Dr. Gu






Each GU00X-* folder contains code, notes, and results specific to that project.

common/ is reserved for shared utilities (e.g., patch extraction, visualization, GeoJSON helpers).

docs/ contains cross-project documentation and pipeline summaries.

Scope and Notes

GU003–GU005 are intentionally restricted to 1 WSI only, following Dr. Gu’s guidance,
and are designed for method comparison and pipeline validation, not performance benchmarking.

The focus of later GU projects progressively shifts from:

instance segmentation (StarDist),

to generalist nuclei segmentation (CellSAM),

to semantic-aware nuclei segmentation (ClassPose).

GeoJSON export compatibility and downstream usability are treated as first-order deliverables.

Status

All projects are organized for reproducibility, comparison, and discussion.
Results are intended for internal review, method evaluation, and presentation slides.
