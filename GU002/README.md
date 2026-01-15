# GU002 – Lung Cancer Histology Classification (Solid vs Acinar)

This repository contains dataset preparation and splitting code for a lung cancer
histopathology classification task focusing on **solid** versus **acinar** growth patterns.

---

## Dataset Overview

- Source: Institutional lung cancer whole-slide image (WSI) dataset
- Labels are derived from histologic subtype annotations
- Each row corresponds to one WSI
- Image identifiers are provided in the `File Name` column

### Included Histologic Classes

- **solid**
- **acinar**

Other histologic subtypes (e.g. lepidic, papillary, micropapillary) were excluded
from this experiment.

---

## Train / Validation / Test Split

A class-balanced and reproducible dataset split was generated with a fixed random seed.

### Split Counts

| Class  | Train | Validation | Test | Total |
|------|-------|------------|------|-------|
| solid | 28 | 8 | 15 | 51 |
| acinar | 28 | 8 | 23 | 59 |

- Stratified by histologic class
- No overlap between train, validation, and test sets
- Each WSI appears in exactly one split

The finalized split is stored in:

data/GU002_lung_solid_vs_acinar_train_val_test_split.xlsx



---

## File Structure

GU002-lung/
├── data/
│ ├── Lung_MetaData_Release_1.csv
│ └── GU002_lung_solid_vs_acinar_train_val_test_split.xlsx
├── notebooks/
│ └── lung_solid_vs_acinar_train_val_test_split.ipynb
└── README.md



---

## Notes

- The `Class` column represents the histologic growth pattern.
- The `File Name` column corresponds to the WSI identifier used for downstream analysis.
- This dataset split is intended to be reused consistently across all downstream modeling experiments.
