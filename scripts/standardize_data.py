#!/usr/bin/env python3
"""
Standardize MRI data directory structure.

Creates a unified directory layout with consistent sequence naming
using symlinks, and generates a training-ready dataset.csv.

Only includes patients that have reviewed segmentation annotations.

Source: /data/MRI_data/
Output: /data/MRI_data/standardized/
"""

import os
import csv
import json
import re
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# === Configuration ===
DATA_ROOT = Path("/data/MRI_data")
OUTPUT_DIR = DATA_ROOT / "standardized"

# Annotation sources (drives which patients are included)
ANN_SOURCES = [
    {
        "dir": DATA_ROOT / "MRI_json_reviewed" / "positive",
        "label": 1,
        "name": "positive",
    },
    {
        "dir": DATA_ROOT / "MRI_json_reviewed" / "positive_1210_MRI",
        "label": 1,
        "name": "positive_1210_MRI",
    },
    {
        "dir": DATA_ROOT / "MRI_json_reviewed" / "negative",
        "label": 0,
        "name": "negative",
    },
]

# Image search paths (tried in order)
IMG_SEARCH_POSITIVE = [
    DATA_ROOT / "MRI_img" / "Positive LNM",
    DATA_ROOT / "MRI_supplement" / "MRI_img" / "MRI_img",
]
IMG_SEARCH_NEGATIVE = [
    DATA_ROOT / "MRI_img" / "Negative",
]

STANDARD_SEQUENCES = [
    "T1", "T2", "ADC", "DWI",
    "DCE1", "DCE2", "DCE3", "DCE4", "DCE5", "DCE6", "DCE7",
]

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
    "PH1": "DCE1",
    # DCE2
    "D2": "DCE2", "增强2": "DCE2", "T1-DYN2": "DCE2", "T1-DYNA2": "DCE2",
    "动态增强2": "DCE2", "zengqiang２": "DCE2", "zq2": "DCE2", "T1-D2": "DCE2",
    "PH2": "DCE2",
    # DCE3
    "D3": "DCE3", "增强3": "DCE3", "T1-DYN3": "DCE3", "T1-DYNA3": "DCE3",
    "动态增强3": "DCE3", "zengqiang３＼": "DCE3", "zengqiang3、": "DCE3", "zq3": "DCE3",
    "PH3": "DCE3",
    # DCE4
    "D4": "DCE4", "增强4": "DCE4", "T1-DYN4": "DCE4", "T1-DYNA4": "DCE4",
    "动态增强4": "DCE4", "zengqiang４": "DCE4", "zq4": "DCE4",
    "PH4": "DCE4",
    # DCE5
    "D5": "DCE5", "增强5": "DCE5", "T1-DYN5": "DCE5", "T1-DYNA5": "DCE5",
    "动态增强5": "DCE5", "zengqiang５": "DCE5", "zq5": "DCE5",
    "PH5": "DCE5",
    # DCE6
    "D6": "DCE6", "增强6": "DCE6", "T1-DYN6": "DCE6", "T1-DYNA6": "DCE6",
    "动态增强6": "DCE6", "zengqiang６": "DCE6", "zq6": "DCE6",
    "PH6": "DCE6",
    # DCE7
    "D7": "DCE7", "增强7": "DCE7", "T1-DYN7": "DCE7", "T1-DYNA7": "DCE7",
    "动态增强7": "DCE7", "zengqiang７": "DCE7", "zq7": "DCE7",
    "PH7": "DCE7",
}


def clean_patient_id(dirname: str) -> str:
    """Extract clean patient_id from annotation directory name.

    Strips parenthetical notes like '(一大片）' or '（delete）'.
    """
    # Remove anything starting from the first '(' or '（'
    cleaned = re.split(r"[（(]", dirname, maxsplit=1)[0]
    return cleaned.strip()


def find_mri_subdir(patient_dir: Path) -> Optional[Path]:
    """Find MRI/mri/MR subdirectory (case-insensitive)."""
    for name in ("MRI", "mri", "MR"):
        sub = patient_dir / name
        if sub.is_dir():
            return sub
    return None


def find_image_dir(patient_id: str, label: int) -> Optional[Path]:
    """Find the image directory for a patient across multiple search paths."""
    search_paths = IMG_SEARCH_POSITIVE if label == 1 else IMG_SEARCH_NEGATIVE
    for base in search_paths:
        candidate = base / patient_id
        if candidate.is_dir():
            return candidate
    return None


def map_sequences(mri_dir: Path) -> Dict[str, Path]:
    """Map original sequence folders to standard names."""
    mapped = {}
    if not mri_dir or not mri_dir.is_dir():
        return mapped

    for seq_dir in sorted(mri_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        name = seq_dir.name
        if name.endswith("_jpeg"):
            continue
        standard = SEQUENCE_MAP.get(name)
        if standard and standard not in mapped:
            mapped[standard] = seq_dir
    return mapped


def create_image_symlinks(src_dir: Path, dst_dir: Path) -> int:
    """Create symlinks for image files (.jpg/.png) in src_dir. Returns count."""
    if not src_dir.is_dir():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            link_path = dst_dir / f.name
            if not link_path.exists():
                link_path.symlink_to(f.resolve())
            count += 1
    return count


def create_annotation_symlinks(src_dir: Path, dst_dir: Path) -> int:
    """Create symlinks for annotation files (.json) in src_dir. Returns count."""
    if not src_dir.is_dir():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src_dir.iterdir()):
        if f.is_file() and f.suffix.lower() == ".json":
            link_path = dst_dir / f.name
            if not link_path.exists():
                link_path.symlink_to(f.resolve())
            count += 1
    return count


def process_patient(
    patient_id: str,
    ann_dirname: str,
    label: int,
    ann_patient_dir: Path,
    out_img_base: Path,
    out_ann_base: Path,
    report: dict,
) -> Optional[dict]:
    """Process a single annotated patient."""
    label_str = "positive" if label == 1 else "negative"

    # Find image directory
    img_dir = find_image_dir(patient_id, label)
    if img_dir is None:
        report["skipped_patients"].append({
            "id": patient_id,
            "ann_dir": ann_dirname,
            "reason": "no image directory found",
        })
        return None

    # Find MRI subdir in image directory
    img_mri = find_mri_subdir(img_dir)
    if img_mri is None:
        report["skipped_patients"].append({
            "id": patient_id,
            "reason": "no MRI/mri/MR subdir in image directory",
        })
        return None

    # Map image sequences
    img_seq_map = map_sequences(img_mri)
    if not img_seq_map:
        report["skipped_patients"].append({
            "id": patient_id,
            "reason": "no recognizable sequences in image directory",
        })
        return None

    # Find MRI subdir in annotation directory
    ann_mri = find_mri_subdir(ann_patient_dir)
    if ann_mri is None:
        report["skipped_patients"].append({
            "id": patient_id,
            "reason": "no MRI/mri/MR subdir in annotation directory",
        })
        return None

    ann_seq_map = map_sequences(ann_mri)

    # Create image symlinks
    out_patient_img = out_img_base / label_str / patient_id
    slice_counts = {}
    for std_name in STANDARD_SEQUENCES:
        src = img_seq_map.get(std_name)
        if src:
            dst = out_patient_img / std_name
            count = create_image_symlinks(src, dst)
            slice_counts[std_name] = count
            for f in src.iterdir():
                if f.is_file():
                    ext = f.suffix.lower()
                    report["file_formats"][ext] = report["file_formats"].get(ext, 0) + 1
        else:
            slice_counts[std_name] = 0

    # Create annotation symlinks
    out_patient_ann = out_ann_base / label_str / patient_id
    ann_count = 0
    for std_name in STANDARD_SEQUENCES:
        src = ann_seq_map.get(std_name)
        if src:
            dst = out_patient_ann / std_name
            ann_count += create_annotation_symlinks(src, dst)

    # Track sequence naming
    for seq_dir in sorted(img_mri.iterdir()):
        if seq_dir.is_dir() and not seq_dir.name.endswith("_jpeg"):
            report["original_naming"][seq_dir.name] = report["original_naming"].get(seq_dir.name, 0) + 1

    # Build CSV row
    row = {"patient_id": patient_id, "label": label}
    for seq in STANDARD_SEQUENCES:
        seq_path = out_patient_img / seq
        row[f"{seq}_path"] = str(seq_path) if slice_counts[seq] > 0 else ""
        row[f"{seq}_num_slices"] = slice_counts[seq]
    row["has_annotation"] = 1
    row["annotation_path"] = str(out_patient_ann)

    report["processed_patients"] += 1
    report["annotation_json_files"] += ann_count
    if patient_id != ann_dirname:
        report["renamed_patients"].append({"original": ann_dirname, "cleaned": patient_id})

    return row


def main():
    print("=" * 60)
    print("MRI Data Standardization (annotation-driven, 502 patients)")
    print("=" * 60)

    # Clean output directory but preserve features/
    features_dir = OUTPUT_DIR / "features"
    features_tmp = None
    if features_dir.exists():
        features_tmp = DATA_ROOT / "_features_backup_tmp"
        if features_tmp.exists():
            shutil.rmtree(features_tmp)
        shutil.move(str(features_dir), str(features_tmp))
        print(f"Backed up features/ temporarily")

    # Remove old standardized content
    for item in ["images", "annotations", "dataset.csv", "standardize_report.json"]:
        p = OUTPUT_DIR / item
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink()
    print(f"Cleaned old standardized content")

    # Restore features/
    out_img = OUTPUT_DIR / "images"
    out_ann = OUTPUT_DIR / "annotations"
    out_img.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)

    if features_tmp and features_tmp.exists():
        shutil.move(str(features_tmp), str(features_dir))
        print(f"Restored features/")

    report = {
        "processed_patients": 0,
        "annotation_json_files": 0,
        "skipped_patients": [],
        "renamed_patients": [],
        "original_naming": {},
        "file_formats": {},
        "positive_count": 0,
        "negative_count": 0,
        "source_counts": {},
    }

    csv_rows = []
    seen_patients = set()  # Avoid duplicates across sources

    for source in ANN_SOURCES:
        ann_base = source["dir"]
        label = source["label"]
        source_name = source["name"]
        source_count = 0

        if not ann_base.is_dir():
            print(f"WARNING: annotation dir not found: {ann_base}")
            continue

        print(f"\nProcessing annotations from: {ann_base}")

        for ann_patient_dir in sorted(ann_base.iterdir()):
            if not ann_patient_dir.is_dir():
                continue

            ann_dirname = ann_patient_dir.name
            patient_id = clean_patient_id(ann_dirname)

            if patient_id in seen_patients:
                report["skipped_patients"].append({
                    "id": patient_id,
                    "ann_dir": ann_dirname,
                    "reason": f"duplicate (already processed from earlier source)",
                })
                continue
            seen_patients.add(patient_id)

            row = process_patient(
                patient_id, ann_dirname, label,
                ann_patient_dir, out_img, out_ann, report,
            )
            if row:
                csv_rows.append(row)
                source_count += 1
                if label == 1:
                    report["positive_count"] += 1
                else:
                    report["negative_count"] += 1

        report["source_counts"][source_name] = source_count
        print(f"  -> {source_count} patients processed")

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
    print(f"Patients skipped:    {len(report['skipped_patients'])}")
    for s in report["skipped_patients"]:
        print(f"  - {s['id']}: {s['reason']}")
    if report["renamed_patients"]:
        print(f"Renamed directories: {len(report['renamed_patients'])}")
        for r in report["renamed_patients"]:
            print(f"  - {r['original']} -> {r['cleaned']}")
    print(f"Annotation JSON files linked: {report['annotation_json_files']}")
    print(f"\nSource breakdown:")
    for name, count in report["source_counts"].items():
        print(f"  {name}: {count}")
    print(f"\nFile formats:")
    for ext, count in sorted(report["file_formats"].items()):
        print(f"  {ext}: {count} files")
    print(f"\nFiles created:")
    print(f"  {csv_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
