---
name: MRI data directory structure
description: Detailed organization of /data/MRI_data including Xiangya and multi-center data, naming inconsistencies, and annotation formats
type: project
---

## Data Root: /data/MRI_data/

### 1. Xiangya Second Hospital (湘雅二医院) — Main Dataset

#### Raw DICOM: MRI/
- `Positive LNM/`: 328 patients (317 Y_ prefix, 8 N_ prefix, 3 mixed)
- `Negative LNM/`: 328 patients (327 N_ prefix, 1 Y_ prefix: Y_705_ZTR)
- Structure: `{patient_id}/{MRI|mri}/{sequence}/{slice}.dcm`

#### Converted Images: MRI_img/
- `Positive LNM/`: 328 patients
- `Negative/`: 328 patients
- Structure: `{patient_id}/{MRI|mri}/{sequence}/{nnnnn}.jpg`

#### Box Annotations (rectangle): MRI_json/
- `Positive LNM/`: 25 patients
- `Negative LNM/`: 163 patients
- Format: LabelMe JSON with `shape_type: "rectangle"`, label: `cancer`
- Contains embedded `imageData` (base64)

#### Segmentation Annotations (polygon, reviewed): MRI_json_reviewed/
- `positive/`: 208 patients
- `negative/`: 159 patients (note: `N_11_ZX（delete）` marked for deletion)
- Format: LabelMe JSON with `shape_type: "polygon"`, labels: `cancer`, `lymph node`
- Structure mirrors MRI/ with same 11 sequence folders

#### Supplement Data: MRI_supplement/
- `Positive_LNM/`: 97 patients (DICOM)
- `Negative LNM/`: 145 patients (DICOM)
- `MRI_img/MRI_img/`: 94 positive patients (images)
- `MRI_img/Negative_LNM/`: 140 negative patients (images)

#### Supplement Annotations: MRI_json/
- `Poistive_LNM_addition_json/`: 135 patients (note: typo "Poistive")
- `Negative_LNM_addition_json/`: 121 patients

### 2. Multi-center Data (4 hospitals, compressed)

Located in `多中心原始数据分类图像/`:
- 南华大学附属第二医院: 3 tar.gz
- 常德市一: 2 tar.gz
- 衡阳市中心医院: 3 tar.gz
- 郴州市一: 2 tar.gz

Processed versions in:
- `MRI/多中心原始数据_排序后(image)/` — sorted by hospital
- `MRI_img/多中心原始数据/` — converted images

### 3. Per-patient Sequence Structure (11 sequences)

| Sequence | Naming Variants |
|----------|----------------|
| T1 | `T1`, `T1WI` |
| T2 | `T2`, `T2WI` |
| ADC | `ADC` |
| DWI | `DWI` |
| DCE phase 1-7 | `D1-D7`, `增强1-增强7`, `T1-DYNA1~T1-DYNA7`, `T1-DYN1~T1-DYN7` |

### 4. Naming Inconsistencies

| Issue | Details |
|-------|---------|
| Subdirectory case | `MRI/` vs `mri/` (Pos: 290 MRI + 38 mri; Neg: 247 MRI + 81 mri) |
| T1/T2 naming | Neg: 221 T1 + 92 T1WI; 224 T2 + 90 T2WI |
| DCE naming | 3 patterns: D1-D7 (majority), 增强1-7, T1-DYN* |
| Positive LNM directory | Contains 8 N_-prefixed patients |
| Negative LNM directory | Contains 1 Y_-prefixed patient (Y_705_ZTR) |
| Supplement folder typo | `Poistive_LNM_addition_json` (should be Positive) |

**Why:** Understanding the data layout and inconsistencies is critical for building a correct data loading pipeline.
**How to apply:** Use the standardized directory at `/data/MRI_data/standardized/` for training. Load `dataset.csv` for patient metadata and paths.

### 5. Standardized Data (created 2026-04-08)

Location: `/data/MRI_data/standardized/`
Script: `scripts/standardize_data.py`

Structure:
```
standardized/
├── images/{positive,negative}/{patient_id}/{T1,T2,ADC,DWI,DCE1-DCE7}/  (symlinks)
├── annotations/{positive,negative}/{patient_id}/{sequence}/             (symlinks)
├── dataset.csv
└── standardize_report.json
```

Stats:
- 644 patients processed (324 positive, 320 negative), 12 skipped (empty)
- 627 patients with complete 11 sequences
- 357 patients with reviewed segmentation annotations
- File formats: 638k JPG + 419k PNG (mixed, both linked)
- Standard sequence names: T1, T2, ADC, DWI, DCE1-DCE7

CSV columns: patient_id, label, {seq}_path, {seq}_num_slices (x11), has_annotation, annotation_path

