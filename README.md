# Wanlin–DrGu Cell Segmentation Projects

This repository consolidates a series of WSI-based image analysis projects
conducted with Dr. Gu, organized as GU001–GU006.  
Each GU project corresponds to a clearly defined dataset, goal, and set of deliverables.
Together, they form a progressive pipeline from patch-level classification
to instance- and semantic-aware nuclei segmentation with GeoJSON export,
and downstream feature-based modeling.

---

## Project Overview

| Project ID | Start Day | Name | Dataset | Main Goal | Dr. Gu Split | Your Deliverables |
|------------|-----------|------|---------|-----------|--------------|-------------------|
| **GU001** | 11/25/25 | Kidney: Papillary vs ccRCC | DHMC Kidney | Classification pipeline | **For Papillary RCC:**<br>84 slides → Training<br>21 slides → Validation<br>45 slides → Testing<br><br>**For Clear Cell RCC:**<br>105 slides → Training + Validation (84 / 21)<br>398 slides → Testing | Split table, scripts, baseline model, slides |
| **GU002** | 11/25/25 | Lung: Solid vs Acinar | DHMC Lung | Pattern classification | Solid 28/8/15; Acinar 28/8/23 (Training / Validation / Test) | Patch pipeline, baseline results, slides |
| **GU003** | 11/25/25 | WSI StarDist + GeoJSON | Lung WSI | Nuclei segmentation → GeoJSON | 1 WSI only | StarDist results, GeoJSON export attempt, error logs, GitHub repo, slides |
| **GU004** | 12/16/25 | WSI CellSAM + GeoJSON | Lung WSI | Generalist nuclei segmentation → GeoJSON (baseline comparison) | 1 WSI only (same as GU003) | CellSAM results, GeoJSON export, StarDist vs CellSAM visual comparison, inference notes, GitHub repo, slides |
| **GU005** | 01/03/26 | WSI ClassPose + Semantic GeoJSON | Lung WSI | Semantic-aware nuclei segmentation (instance + cell type) → GeoJSON | 1 WSI only (same as GU003 / GU004) | ClassPose inference results, cell-type–annotated GeoJSON, centroid alignment checks, CellSAM vs ClassPose qualitative comparison, export schema notes, GitHub repo, slides |
| **GU006** | 03/20/26 | Feature + UMAP (Patch-level Modeling) | Lung WSI (GeoJSON from GU003–GU005) | Cell feature expansion (G003 + legacy + radiomics) + UMAP-based aggregation for patch-level classification | Same as GU001 / GU002 (classification splits) | Updated features.py (radiomics integration), updated aggregation.py (UMAP), patch-level feature extraction, baseline vs GU006 comparison (ROC-AUC), slides |

---

## Repository Structure

```text
Wanlin-DrGu-CellSegmentation/
├── GU001-kidney/
├── GU002-lung/
├── GU003-stardist/
├── GU004-cellsam/
├── GU005-classpose/
├── GU006-feature-umap/
├── common/
├── docs/
└── README.md



Each GU00X-* folder contains code, notes, and results specific to that project.

common/ is reserved for shared utilities (e.g., patch extraction, visualization, GeoJSON helpers).

docs/ contains cross-project documentation and pipeline summaries.

Scope and Notes

GU003–GU005 are intentionally restricted to 1 WSI only, following Dr. Gu’s guidance,
and are designed for method comparison and pipeline validation, not performance benchmarking.

GU006 extends these pipelines into downstream modeling by focusing on feature engineering
and aggregation strategies rather than segmentation itself.

The focus of later GU projects progressively shifts from:

instance segmentation (StarDist),

to generalist nuclei segmentation (CellSAM),

to semantic-aware nuclei segmentation (ClassPose),

to feature-based modeling with radiomics and representation learning (GU006). https://pyradiomics.readthedocs.io/en/latest/

GeoJSON export compatibility and downstream usability are treated as first-order deliverables.

Status

All projects are organized for reproducibility, comparison, and discussion.
Results are intended for internal review, method evaluation, and presentation slides.



