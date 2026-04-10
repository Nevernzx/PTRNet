#!/usr/bin/env python3
"""
Standardize MRI data directory structure.

Creates a unified directory layout with consistent sequence naming
using symlinks, and generates a training-ready dataset.csv.

Source: /data/MRI_data/ (Xiangya Second Hospital main dataset only)
Output: /data/MRI_data/standardized/
"""

import os
import csv
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional

# === Configuration ===
DATA_ROOT = Path("/data/MRI_data")
IMG_POSITIVE_DIR = DATA_ROOT / "MRI_img" / "Positive LNM"
IMG_NEGATIVE_DIR = DATA_ROOT / "MRI_img" / "Negative"
ANN_POSITIVE_DIR = DATA_ROOT / "MRI_json_reviewed" / "positive"
ANN_NEGATIVE_DIR = DATA_ROOT / "MRI_json_reviewed" / "negative"
OUTPUT_DIR = DATA_ROOT / "standardized"

STANDARD_SEQUENCES = ["T1", "T2", "ADC", "DWI", "DCE1", "DCE2", "DCE3", "DCE4", "DCE5", "DCE6", "DCE7"]

# Mapping from original sequence folder names to standard names
SEQUENCE_MAP = {
    # T1 variants
    "T1": "T1", "T1WI": "T1", "T1Wi": "T1",
    # T2 variants
    "T2": "T2", "T2WI": "T2", "t2WI": "T2",
    # ADC variants
    "ADC": "ADC", "adc": "ADC", "ＡＤＣ": "ADC",
    # DWI variants
    "DWI": "DWI", "dwi": "DWI", "DW": "DWI",
    # DCE1
    "D1": "DCE1", "增强1": "DCE1", "T1-DYN1": "DCE1", "T1-DYNA1": "DCE1",
    "动态增强1": "DCE1", "zengqiang１": "DCE1", "zq1": "DCE1", "T1-D1": "DCE1",
    # DCE2
    "D2": "DCE2", "增强2": "DCE2", "T1-DYN2": "DCE2", "T1-DYNA2": "DCE2",
    "动态增强2": "DCE2", "zengqiang２": "DCE2", "zq2": "DCE2", "T1-D2": "DCE2",
    # DCE3
    "D3": "DCE3", "增强3": "DCE3", "T1-DYN3": "DCE3", "T1-DYNA3": "DCE3",
    "动态增强3": "DCE3", "zengqiang３＼": "DCE3", "zengqiang3、": "DCE3", "zq3": "DCE3",
    # DCE4
    "D4": "DCE4", "增强4": "DCE4", "T1-DYN4": "DCE4", "T1-DYNA4": "DCE4",
    "动态增强4": "DCE4", "zengqiang４": "DCE4", "zq4": "DCE4",
    # DCE5
    "D5": "DCE5", "增强5": "DCE5", "T1-DYN5": "DCE5", "T1-DYNA5": "DCE5",
    "动态增强5": "DCE5", "zengqiang５": "DCE5", "zq5": "DCE5",
    # DCE6
    "D6": "DCE6", "增强6": "DCE6", "T1-DYN6": "DCE6", "T1-DYNA6": "DCE6",
    "动态增强6": "DCE6", "zengqiang６": "DCE6", "zq6": "DCE6",
    # DCE7
    "D7": "DCE7", "增强7": "DCE7", "T1-DYN7": "DCE7", "T1-DYNA7": "DCE7",
    "动态增强7": "DCE7", "zengqiang７": "DCE7", "zq7": "DCE7",
}


def find_mri_subdir(patient_dir: Path) -> Optional[Path]:
    """Find MRI/mri subdirectory (case-insensitive)."""
    for name in ("MRI", "mri"):
        sub = patient_dir / name
        if sub.is_dir():
            return sub
    return None


def map_sequences(mri_dir: Path) -> Dict[str, Path]:
    """Map original sequence folders to standard names, skipping _jpeg dirs."""
    mapped = {}
    if not mri_dir or not mri_dir.is_dir():
        return mapped

    for seq_dir in sorted(mri_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        name = seq_dir.name
        # Skip _jpeg duplicate directories
        if name.endswith("_jpeg"):
            continue
        standard = SEQUENCE_MAP.get(name)
        if standard:
            mapped[standard] = seq_dir
    return mapped


def create_symlinks(src_dir: Path, dst_dir: Path) -> int:
    """Create symlinks for all files in src_dir under dst_dir. Returns file count."""
    if not src_dir.is_dir():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src_dir.iterdir()):
        if f.is_file():
            link_path = dst_dir / f.name
            if not link_path.exists():
                link_path.symlink_to(f.resolve())
            count += 1
    return count


def should_skip_annotation_dir(name: str) -> bool:
    """Skip annotation patient dirs with delete markers."""
    return "delete" in name.lower() or "（" in name


def process_patient(
    patient_id: str,
    label: int,
    img_dir: Path,
    ann_base_dir: Path,
    out_img_base: Path,
    out_ann_base: Path,
    report: dict,
):
    """Process a single patient: create symlinks and return CSV row data."""
    label_str = "positive" if label == 1 else "negative"

    # Find MRI subdir in image directory
    mri_subdir = find_mri_subdir(img_dir)
    if mri_subdir is None:
        report["skipped_patients"].append({"id": patient_id, "reason": "no MRI subdir"})
        return None

    # Map sequences
    seq_map = map_sequences(mri_subdir)
    if not seq_map:
        report["skipped_patients"].append({"id": patient_id, "reason": "empty sequences"})
        return None

    # Track original naming for report
    for seq_dir in sorted(mri_subdir.iterdir()):
        if seq_dir.is_dir() and not seq_dir.name.endswith("_jpeg"):
            report["original_naming"][seq_dir.name] = report["original_naming"].get(seq_dir.name, 0) + 1

    # Create image symlinks
    out_patient_img = out_img_base / label_str / patient_id
    slice_counts = {}
    for std_name in STANDARD_SEQUENCES:
        src = seq_map.get(std_name)
        if src:
            dst = out_patient_img / std_name
            count = create_symlinks(src, dst)
            slice_counts[std_name] = count
            # Track file format
            for f in src.iterdir():
                if f.is_file():
                    ext = f.suffix.lower()
                    report["file_formats"][ext] = report["file_formats"].get(ext, 0) + 1
        else:
            slice_counts[std_name] = 0

    # Find and process annotations
    has_annotation = False
    ann_patient_dir = None

    # Try to find annotation directory for this patient
    if ann_base_dir.is_dir():
        for d in ann_base_dir.iterdir():
            if d.is_dir() and d.name == patient_id:
                ann_patient_dir = d
                break

    out_patient_ann = out_ann_base / label_str / patient_id
    if ann_patient_dir and not should_skip_annotation_dir(ann_patient_dir.name):
        ann_mri_subdir = find_mri_subdir(ann_patient_dir)
        if ann_mri_subdir:
            ann_seq_map = map_sequences(ann_mri_subdir)
            if ann_seq_map:
                has_annotation = True
                for std_name, src in ann_seq_map.items():
                    dst = out_patient_ann / std_name
                    create_symlinks(src, dst)

    # Build CSV row
    row = {
        "patient_id": patient_id,
        "label": label,
    }
    for seq in STANDARD_SEQUENCES:
        seq_path = out_patient_img / seq
        row[f"{seq}_path"] = str(seq_path) if slice_counts[seq] > 0 else ""
        row[f"{seq}_num_slices"] = slice_counts[seq]
    row["has_annotation"] = int(has_annotation)
    row["annotation_path"] = str(out_patient_ann) if has_annotation else ""

    report["processed_patients"] += 1
    if has_annotation:
        report["annotated_patients"] += 1

    return row


def main():
    print("=" * 60)
    print("MRI Data Standardization")
    print("=" * 60)

    # Clean output directory if exists
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
        print(f"Cleaned existing output: {OUTPUT_DIR}")

    out_img = OUTPUT_DIR / "images"
    out_ann = OUTPUT_DIR / "annotations"
    out_img.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)

    report = {
        "processed_patients": 0,
        "annotated_patients": 0,
        "skipped_patients": [],
        "original_naming": {},
        "file_formats": {},
        "positive_count": 0,
        "negative_count": 0,
    }

    csv_rows = []

    # Process Positive LNM patients
    print(f"\nProcessing Positive LNM from: {IMG_POSITIVE_DIR}")
    for patient_dir in sorted(IMG_POSITIVE_DIR.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        row = process_patient(
            patient_id, 1, patient_dir, ANN_POSITIVE_DIR,
            out_img, out_ann, report,
        )
        if row:
            csv_rows.append(row)
            report["positive_count"] += 1

    # Process Negative LNM patients
    print(f"Processing Negative LNM from: {IMG_NEGATIVE_DIR}")
    for patient_dir in sorted(IMG_NEGATIVE_DIR.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        row = process_patient(
            patient_id, 0, patient_dir, ANN_NEGATIVE_DIR,
            out_img, out_ann, report,
        )
        if row:
            csv_rows.append(row)
            report["negative_count"] += 1

    # Write dataset.csv
    csv_path = OUTPUT_DIR / "dataset.csv"
    fieldnames = ["patient_id", "label"]
    for seq in STANDARD_SEQUENCES:
        fieldnames.extend([f"{seq}_path", f"{seq}_num_slices"])
    fieldnames.extend(["has_annotation", "annotation_path"])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Write report
    report_path = OUTPUT_DIR / "standardize_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"{'=' * 60}")
    print(f"Output directory:    {OUTPUT_DIR}")
    print(f"Patients processed:  {report['processed_patients']}")
    print(f"  Positive:          {report['positive_count']}")
    print(f"  Negative:          {report['negative_count']}")
    print(f"  With annotations:  {report['annotated_patients']}")
    print(f"Patients skipped:    {len(report['skipped_patients'])}")
    for s in report["skipped_patients"]:
        print(f"  - {s['id']}: {s['reason']}")
    print(f"\nFile formats:")
    for ext, count in sorted(report["file_formats"].items()):
        print(f"  {ext}: {count} files")
    print(f"\nFiles created:")
    print(f"  {csv_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
