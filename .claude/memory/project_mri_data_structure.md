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
- `positive/`: 208 patients (6 dirs have parenthetical notes in name)
- `positive_1210_MRI/`: 135 patients (supplement positive, reviewed annotations)
- `negative/`: 159 patients (3 dirs have parenthetical notes, incl. N_11_ZX（delete）)
- `negative_1113/negative_1113/`: 159 patients (duplicate of negative/)
- `positive_已修改/`: empty
- Total unique annotated: **502** (343 positive + 159 negative)
- Format: LabelMe JSON with `shape_type: "polygon"`, labels: `cancer`, `lymph node`
- Structure: `{patient_id}/{MRI|mri|MR}/{sequence}/` with co-located .jpg + .json files

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

### 5. Standardized Data (rebuilt 2026-04-15, annotation-driven)

Location: `/data/MRI_data/standardized/`
Script: `scripts/standardize_data.py`

Structure:
```
standardized/
├── images/{positive,negative}/{patient_id}/{T1,T2,ADC,DWI,DCE1-DCE7}/  (symlinks)
├── annotations/{positive,negative}/{patient_id}/{sequence}/             (symlinks to .json)
├── dataset.csv
├── features/BiomedCLIP/  (preserved from prior run)
└── standardize_report.json
```

Stats:
- **502 patients** (343 positive, 159 negative), 0 skipped — annotation-driven (only annotated patients included)
- All 502 have has_annotation=1; 501 have actual annotation JSON files (N_26_CL annotation dir is empty)
- Image sources: MRI_img/ (main) + MRI_supplement/MRI_img/MRI_img/ (74 supplement patients)
- 9 annotation dirs with special characters mapped to clean patient_id
- File formats: 740k JPG + 87k PNG
- Standard sequence names: T1, T2, ADC, DWI, DCE1-DCE7
- Additional sequence mapping: PH1-PH7 → DCE1-DCE7

CSV columns: patient_id, label, {seq}_path, {seq}_num_slices (x11), has_annotation, annotation_path

