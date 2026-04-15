#!/usr/bin/env python3
"""
Extract BiomedCLIP features from standardized breast MRI data.

For each patient, extracts ViT patch embeddings from every slice of each sequence,
generates mask_attention from segmentation annotations, and saves all sequences
into a single .pth file for fast training-time loading.

Input:  /data/MRI_data/standardized/ (images + annotations + dataset.csv)
Output: /data/MRI_data/standardized/features/BiomedCLIP/{patient_id}.pth

Each .pth is a dict keyed by sequence name, e.g.:
  {
    "T1":   {"patch_embed": (S,14,14,D), "mask_attention": (S,197), "slices_weight": (S,)},
    "T2":   {"patch_embed": ..., "mask_attention": ..., "slices_weight": ...},
    "DCE1": ...,
    
    ...
  }
"""

from PIL import Image, ImageDraw
import torch
import os
import numpy as np
import csv
import json
import argparse
import re
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

STANDARD_SEQUENCES = ["T1", "T2", "ADC", "DWI", "DCE1", "DCE2", "DCE3", "DCE4", "DCE5", "DCE6", "DCE7"]

SEQUENCE_MAP = {
    "T1": "T1", "T1WI": "T1", "T1Wi": "T1",
    "T2": "T2", "T2WI": "T2", "t2WI": "T2",
    "ADC": "ADC", "adc": "ADC", "ＡＤＣ": "ADC",
    "DWI": "DWI", "dwi": "DWI", "DW": "DWI",
    "D1": "DCE1", "增强1": "DCE1", "T1-DYN1": "DCE1", "T1-DYNA1": "DCE1",
    "动态增强1": "DCE1", "zengqiang１": "DCE1", "zq1": "DCE1", "T1-D1": "DCE1",
    "D2": "DCE2", "增强2": "DCE2", "T1-DYN2": "DCE2", "T1-DYNA2": "DCE2",
    "动态增强2": "DCE2", "zengqiang２": "DCE2", "zq2": "DCE2", "T1-D2": "DCE2",
    "D3": "DCE3", "增强3": "DCE3", "T1-DYN3": "DCE3", "T1-DYNA3": "DCE3",
    "动态增强3": "DCE3", "zengqiang３＼": "DCE3", "zengqiang3、": "DCE3", "zq3": "DCE3",
    "D4": "DCE4", "增强4": "DCE4", "T1-DYN4": "DCE4", "T1-DYNA4": "DCE4",
    "动态增强4": "DCE4", "zengqiang４": "DCE4", "zq4": "DCE4",
    "D5": "DCE5", "增强5": "DCE5", "T1-DYN5": "DCE5", "T1-DYNA5": "DCE5",
    "动态增强5": "DCE5", "zengqiang５": "DCE5", "zq5": "DCE5",
    "D6": "DCE6", "增强6": "DCE6", "T1-DYN6": "DCE6", "T1-DYNA6": "DCE6",
    "动态增强6": "DCE6", "zengqiang６": "DCE6", "zq6": "DCE6",
    "D7": "DCE7", "增强7": "DCE7", "T1-DYN7": "DCE7", "T1-DYNA7": "DCE7",
    "动态增强7": "DCE7", "zengqiang７": "DCE7", "zq7": "DCE7",
}

# Reviewed Xiangya annotations only: 208 positive + 159 negative + 135 positive_1210_MRI = 502 patients.
ANNOTATION_SOURCE_CONFIG = {
    1: [
        ("standardized_reviewed", "standardized"),
        ("reviewed_positive", "/data/MRI_data/MRI_json_reviewed/positive"),
        ("reviewed_positive_1210", "/data/MRI_data/MRI_json_reviewed/positive_1210_MRI"),
    ],
    0: [
        ("standardized_reviewed", "standardized"),
        ("reviewed_negative", "/data/MRI_data/MRI_json_reviewed/negative"),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract BiomedCLIP features from standardized MRI data")
    parser.add_argument("--data_root", type=str, default="/data/MRI_data/standardized",
                        help="Path to standardized data directory")
    parser.add_argument("--output_dir", type=str, default="/data/MRI_data/standardized/features/BiomedCLIP",
                        help="Output directory for extracted features")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (cuda:0, cuda:1, cpu)")
    parser.add_argument("--sequences", type=str, nargs="+", default=STANDARD_SEQUENCES,
                        help="Sequences to extract features from")
    parser.add_argument("--resume", action="store_true",
                        help="Skip patients whose features already exist")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for feature extraction (increase to use more VRAM)")
    parser.add_argument("--update_masks_only", action="store_true",
                        help="Only refresh mask_attention and slices_weight in existing .pth files")
    parser.add_argument("--skip_deleted_annotations", action="store_true", default=True,
                        help="Ignore annotation directories explicitly marked as delete")
    parser.add_argument("--dry_run", action="store_true",
                        help="Scan and report mask updates without writing files")
    return parser.parse_args()


def load_model(device):
    import open_clip

    model, _, preprocess_val = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    model.to(device)
    model.eval()
    return model, preprocess_val


def load_annotation_mask(ann_dir, slice_name, img_h=None, img_w=None, img_path=None):
    """Load segmentation polygon from JSON and render as binary mask."""
    # Try matching json filename: strip image extension, add .json
    stem = Path(slice_name).stem
    json_path = os.path.join(ann_dir, stem + ".json")
    if not os.path.exists(json_path):
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    if img_h is None:
        img_h = data.get("imageHeight")
    if img_w is None:
        img_w = data.get("imageWidth")
    if (img_h is None or img_w is None) and img_path and os.path.exists(img_path):
        with Image.open(img_path) as img:
            img_w, img_h = img.size
    if img_h is None or img_w is None:
        return None

    mask_img = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask_img)
    for shape in data.get("shapes", []):
        points = [(float(x), float(y)) for x, y in shape["points"]]
        if shape["shape_type"] == "polygon":
            if len(points) < 3:
                continue
            draw.polygon(points, outline=255, fill=255)
        elif shape["shape_type"] == "rectangle":
            if len(points) < 2:
                continue
            x1, y1 = points[0]
            x2, y2 = points[1]
            draw.rectangle([x1, y1, x2, y2], outline=255, fill=255)

    return np.array(mask_img, dtype=np.uint8)


def normalize_patient_id(name):
    return re.split(r"[（(]", name, 1)[0].strip()


def should_skip_annotation_dir(name, skip_deleted_annotations):
    return skip_deleted_annotations and "delete" in name.lower()


def find_mri_subdir(patient_dir):
    for name in ("MRI", "mri", "MR", "mr"):
        sub = patient_dir / name
        if sub.is_dir():
            return sub

    if patient_dir.is_dir():
        child_names = [child.name for child in patient_dir.iterdir() if child.is_dir()]
        if any(name in SEQUENCE_MAP for name in child_names):
            return patient_dir
    return None


def map_sequences(mri_dir):
    mapped = {}
    if not mri_dir or not mri_dir.is_dir():
        return mapped

    for seq_dir in sorted(mri_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        if seq_dir.name.endswith("_jpeg"):
            continue
        std_name = SEQUENCE_MAP.get(seq_dir.name)
        if std_name:
            mapped[std_name] = seq_dir
    return mapped


def build_annotation_index(skip_deleted_annotations=True):
    annotation_index = {0: {}, 1: {}}

    for label, sources in ANNOTATION_SOURCE_CONFIG.items():
        for source_priority, (source_name, source_path) in enumerate(sources):
            if source_path == "standardized":
                continue

            root = Path(source_path)
            if not root.is_dir():
                continue

            for patient_dir in sorted(root.iterdir()):
                if not patient_dir.is_dir():
                    continue
                if should_skip_annotation_dir(patient_dir.name, skip_deleted_annotations):
                    continue

                norm_id = normalize_patient_id(patient_dir.name)
                annotation_index[label].setdefault(norm_id, []).append(
                    (source_priority, source_name, patient_dir)
                )

    for label in annotation_index:
        for patient_id in annotation_index[label]:
            annotation_index[label][patient_id].sort(key=lambda item: item[0])

    return annotation_index


def resolve_annotation_sequence_dirs(row, annotation_index):
    patient_id = row["patient_id"]
    label = int(row["label"])
    candidates = []

    ann_base = row.get("annotation_path", "")
    if ann_base:
        ann_base_path = Path(ann_base)
        if ann_base_path.is_dir():
            candidates.append((-1, "standardized_reviewed", ann_base_path))

    candidates.extend(annotation_index[label].get(patient_id, []))

    seq_to_dir = {}
    for _, source_name, patient_dir in candidates:
        if source_name == "standardized_reviewed":
            for seq in STANDARD_SEQUENCES:
                ann_dir = patient_dir / seq
                if ann_dir.is_dir() and seq not in seq_to_dir:
                    seq_to_dir[seq] = ann_dir
            continue

        ann_mri_subdir = find_mri_subdir(patient_dir)
        ann_seq_map = map_sequences(ann_mri_subdir)
        for seq, ann_dir in ann_seq_map.items():
            if seq in seq_to_dir:
                continue
            if any(file.name.endswith(".json") for file in ann_dir.iterdir() if file.is_file()):
                seq_to_dir[seq] = ann_dir

    return seq_to_dir


def compute_patch_mask(mask, input_size=224, patch_size=16):
    """Convert pixel-level mask to 14x14 patch-level mask with CLS token prefix.

    Returns tensor of shape (197,): [1.0, patch_0_0, patch_0_1, ..., patch_13_13]
    CLS token is always set to 1.0.
    """
    grid_size = input_size // patch_size  # 14
    resized = Image.fromarray(mask).resize((input_size, input_size), resample=Image.NEAREST)
    mask_tensor = torch.from_numpy(np.array(resized, dtype=np.uint8))

    result = [1.0]  # CLS token
    for i in range(0, input_size, patch_size):
        for j in range(0, input_size, patch_size):
            patch = mask_tensor[i:i + patch_size, j:j + patch_size]
            result.append(1.0 if torch.sum(patch) > 0 else 0.0)

    return torch.tensor(result)


def build_sequence_masks(img_dir, ann_dir):
    if not os.path.isdir(img_dir):
        return None

    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
    ])
    if len(img_files) == 0:
        return None

    all_mask_attention = []
    all_slices_weight = []

    for img_name in img_files:
        img_path = os.path.join(img_dir, img_name)
        mask = None
        if ann_dir and os.path.isdir(ann_dir):
            mask = load_annotation_mask(ann_dir, img_name, img_path=img_path)

        if mask is not None and np.sum(mask) > 0:
            patch_mask = compute_patch_mask(mask)
            all_mask_attention.append(patch_mask.unsqueeze(0))
            all_slices_weight.append(torch.tensor([1.0]))
        else:
            no_mask = torch.zeros(1, 197)
            no_mask[0, 0] = 1.0
            all_mask_attention.append(no_mask)
            all_slices_weight.append(torch.tensor([0.0]))

    return {
        "mask_attention": torch.cat(all_mask_attention, dim=0),
        "slices_weight": torch.cat(all_slices_weight, dim=0),
    }


def extract_sequence_features(model, preprocess_val, device, img_dir, ann_dir, has_ann, batch_size=64):
    """Extract features for all slices in one sequence directory using Batch Processing."""
    if not os.path.isdir(img_dir):
        return None

    # Get sorted image files
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
    ])

    if len(img_files) == 0:
        return None

    all_input_tensors = []
    all_masks_info = []

    # ==========================================
    # 1. 预加载所有图像和 Mask (在 CPU 上进行)
    # ==========================================
    for img_name in img_files:
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        # Preprocess image
        input_tensor = preprocess_val(img)  # 形状: (3, 224, 224)
        all_input_tensors.append(input_tensor)

        # Load mask if available
        if has_ann and ann_dir and os.path.isdir(ann_dir):
            mask = load_annotation_mask(ann_dir, img_name, img_h, img_w)
        else:
            mask = None
        all_masks_info.append(mask)

    # 将列表堆叠成一个大张量 形状: (S, 3, 224, 224)
    stacked_inputs = torch.stack(all_input_tensors)
    total_slices = stacked_inputs.shape[0]

    all_patch_embed = []

    # ==========================================
    # 2. 批量输入模型提取特征 (在 GPU 上并行)
    # ==========================================
    with torch.no_grad():
        for i in range(0, total_slices, batch_size):
            # 切片取出一个 Batch，并送入 GPU
            batch_inputs = stacked_inputs[i:i + batch_size].to(device)
            
            # 批量提取特征: 返回形状 (B, 197, D)
            features = model.visual.trunk.forward_features(batch_inputs)
            
            # 处理 Patch tokens (跳过 CLS)
            B, _, D = features.shape
            patch_embed = features[:, 1:, :].reshape(B, 14, 14, D)  # (B, 14, 14, D)
            
            # 存回 CPU，防止 GPU 显存由于存储整个序列的历史结果而爆满
            all_patch_embed.append(patch_embed.cpu())

    # ==========================================
    # 3. 处理 Mask 注意力和权重
    # ==========================================
    all_mask_attention = []
    all_slices_weight = []
    
    for mask in all_masks_info:
        if mask is not None and np.sum(mask) > 0:
            patch_mask = compute_patch_mask(mask)
            all_mask_attention.append(patch_mask.unsqueeze(0))
            all_slices_weight.append(torch.tensor([1.0]))
        else:
            no_mask = torch.zeros(1, 197)
            no_mask[0, 0] = 1.0  # CLS token
            all_mask_attention.append(no_mask)
            all_slices_weight.append(torch.tensor([0.0]))

    # 合并并返回结果
    result = {
        "patch_embed": torch.cat(all_patch_embed, dim=0),         # (S, 14, 14, D)
        "mask_attention": torch.cat(all_mask_attention, dim=0),   # (S, 197)
        "slices_weight": torch.cat(all_slices_weight, dim=0),     # (S,)
    }
    return result


def update_masks_only(args, patients):
    annotation_index = build_annotation_index(skip_deleted_annotations=args.skip_deleted_annotations)

    processed = 0
    updated = 0
    missing_feature = 0
    any_mask_patients = 0
    warning_count = 0

    for row in patients:
        patient_id = row["patient_id"]
        out_path = os.path.join(args.output_dir, f"{patient_id}.pth")
        if not os.path.exists(out_path):
            missing_feature += 1
            continue

        ann_seq_dirs = resolve_annotation_sequence_dirs(row, annotation_index)
        patient_data = torch.load(out_path, map_location="cpu")
        patient_has_any_mask = False
        patient_changed = False

        for seq in args.sequences:
            if seq not in patient_data:
                continue

            img_dir = row.get(f"{seq}_path", "")
            if not img_dir:
                continue

            try:
                mask_result = build_sequence_masks(img_dir, ann_seq_dirs.get(seq))
            except Exception as exc:
                warning_count += 1
                print(
                    f"Warning: failed to build masks for patient={patient_id}, seq={seq}: {exc}",
                    flush=True,
                )
                continue
            if mask_result is None:
                continue

            new_mask_attention = mask_result["mask_attention"]
            new_slices_weight = mask_result["slices_weight"]
            old_mask_attention = patient_data[seq].get("mask_attention")
            old_slices_weight = patient_data[seq].get("slices_weight")

            patient_data[seq]["mask_attention"] = new_mask_attention
            patient_data[seq]["slices_weight"] = new_slices_weight

            if torch.sum(new_slices_weight) > 0:
                patient_has_any_mask = True

            if (
                old_mask_attention is None
                or old_slices_weight is None
                or old_mask_attention.shape != new_mask_attention.shape
                or old_slices_weight.shape != new_slices_weight.shape
                or not torch.equal(old_mask_attention, new_mask_attention)
                or not torch.equal(old_slices_weight, new_slices_weight)
            ):
                patient_changed = True

        if patient_has_any_mask:
            any_mask_patients += 1

        if patient_changed:
            updated += 1
            if not args.dry_run:
                torch.save(patient_data, out_path)

        processed += 1
        if processed % 20 == 0:
            print(
                f"[{processed}/{len(patients)}] Updated masks for {updated} patients, "
                f"patients with any mask={any_mask_patients}, missing_feature={missing_feature}, warnings={warning_count}",
                flush=True,
            )

    print(
        f"\nMask update done! Processed: {processed}, Updated: {updated}, Missing features: {missing_feature}, warnings: {warning_count}",
        flush=True,
    )
    print(f"Patients with any mask after update: {any_mask_patients}", flush=True)
    if args.dry_run:
        print("Dry run enabled, no feature files were written.", flush=True)


def main():
    args = parse_args()
    data_root = args.data_root
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset CSV
    csv_path = os.path.join(data_root, "dataset.csv")
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        patients = list(reader)
    print(f"Loaded {len(patients)} patients from {csv_path}", flush=True)

    if args.update_masks_only:
        update_masks_only(args, patients)
        return

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    print(f"Loading BiomedCLIP model on {device}...")
    model, preprocess_val = load_model(device)
    print("Model loaded.", flush=True)

    processed = 0
    skipped = 0

    for idx, row in enumerate(patients):
        patient_id = row["patient_id"]
        label = int(row["label"])
        has_ann = int(row["has_annotation"])
        ann_base = row["annotation_path"]

        out_path = os.path.join(output_dir, f"{patient_id}.pth")

        # Resume support
        if args.resume and os.path.exists(out_path):
            skipped += 1
            continue

        patient_data = {}
        for seq in args.sequences:
            img_dir = row.get(f"{seq}_path", "")
            if not img_dir:
                continue

            # Annotation directory for this sequence
            ann_dir = os.path.join(ann_base, seq) if has_ann and ann_base else None

            # 传入 batch_size 参数
            result = extract_sequence_features(
                model, preprocess_val, device, img_dir, ann_dir, has_ann, batch_size=args.batch_size
            )
            if result is not None:
                patient_data[seq] = result

        if patient_data:
            torch.save(patient_data, out_path)

        processed += 1
        if processed % 10 == 0:
            print(f"[{processed}/{len(patients)}] Processed {patient_id} "
                  f"(label={label}, sequences={len(patient_data)})", flush=True)

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}", flush=True)
    print(f"Features saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
