---
name: Breast cancer lymph node metastasis prediction
description: Patient-level classification task using multi-sequence breast MRI with lesion segmentation annotations
type: project
---

## Task
Patient-level breast cancer lymph node metastasis prediction (binary classification: positive/negative).

## Dataset
- Source: Xiangya Second Hospital (湘雅二医院)
- Positive (metastasis): 320 patients
- Negative (no metastasis): 328 patients
- Balanced dataset

## Data Structure
- Each patient has multiple MRI sequences: T1, T2, DCE, ADC, DWI
- DCE contains 7 sub-sequences (D1-D7) representing temporal dynamics
- Each sequence consists of continuous slices (like video frames)

## Annotations
- Frame-level lesion segmentation in JSON format
- Includes both tumor and lymph node annotations

## Technical Approach
1. Locate tumor region using segmentation annotations (ROI extraction)
2. Extract imaging features from lesion regions
3. Fuse information across multiple sequences
4. Output patient-level metastasis prediction

**Why:** Clinical need for non-invasive prediction of lymph node metastasis status in breast cancer patients.
**How to apply:** All model design, data pipeline, and experiment decisions should serve this end-to-end pipeline.
