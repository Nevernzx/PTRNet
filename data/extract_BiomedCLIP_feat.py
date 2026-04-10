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

import open_clip
from PIL import Image
import torch
import os
import numpy as np
import cv2
import csv
import json
import argparse
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

STANDARD_SEQUENCES = ["T1", "T2", "ADC", "DWI", "DCE1", "DCE2", "DCE3", "DCE4", "DCE5", "DCE6", "DCE7"]


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
    return parser.parse_args()


def load_model(device):
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    model.to(device)
    model.eval()
    return model, preprocess_val


def load_annotation_mask(ann_dir, slice_name, img_h, img_w):
    """Load segmentation polygon from JSON and render as binary mask."""
    # Try matching json filename: strip image extension, add .json
    stem = Path(slice_name).stem
    json_path = os.path.join(ann_dir, stem + ".json")
    if not os.path.exists(json_path):
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for shape in data.get("shapes", []):
        points = np.array(shape["points"], dtype=np.int32)
        if shape["shape_type"] == "polygon":
            cv2.fillPoly(mask, [points], 255)
        elif shape["shape_type"] == "rectangle":
            x1, y1 = points[0]
            x2, y2 = points[1]
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)

    return mask


def compute_patch_mask(mask, input_size=224, patch_size=16):
    """Convert pixel-level mask to 14x14 patch-level mask with CLS token prefix.

    Returns tensor of shape (197,): [1.0, patch_0_0, patch_0_1, ..., patch_13_13]
    CLS token is always set to 1.0.
    """
    grid_size = input_size // patch_size  # 14
    resized = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    mask_tensor = torch.from_numpy(resized)

    result = [1.0]  # CLS token
    for i in range(0, input_size, patch_size):
        for j in range(0, input_size, patch_size):
            patch = mask_tensor[i:i + patch_size, j:j + patch_size]
            result.append(1.0 if torch.sum(patch) > 0 else 0.0)

    return torch.tensor(result)


def extract_sequence_features(model, preprocess_val, device, img_dir, ann_dir, has_ann):
    """Extract features for all slices in one sequence directory."""
    if not os.path.isdir(img_dir):
        return None

    # Get sorted image files
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
    ])

    if len(img_files) == 0:
        return None

    all_patch_embed = []
    all_mask_attention = []
    all_slices_weight = []

    for img_name in img_files:
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        # Preprocess and extract features
        input_tensor = preprocess_val(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.visual.trunk.extract_patch_embed(input_tensor)
        # features shape: (1, 197, D) -> take patch tokens (skip CLS)
        N, _, D = features.shape
        patch_embed = features[:, 1:, :].reshape(N, 14, 14, D)  # (1, 14, 14, D)
        all_patch_embed.append(patch_embed.cpu())

        # Process annotation mask
        if has_ann and ann_dir and os.path.isdir(ann_dir):
            mask = load_annotation_mask(ann_dir, img_name, img_h, img_w)
        else:
            mask = None

        if mask is not None and np.sum(mask) > 0:
            patch_mask = compute_patch_mask(mask)
            all_mask_attention.append(patch_mask.unsqueeze(0))
            all_slices_weight.append(torch.tensor([1.0]))
        else:
            # No annotation for this slice
            no_mask = torch.zeros(1, 197)
            no_mask[0, 0] = 1.0  # CLS token
            all_mask_attention.append(no_mask)
            all_slices_weight.append(torch.tensor([0.0]))

    result = {
        "patch_embed": torch.cat(all_patch_embed, dim=0),        # (S, 14, 14, D)
        "mask_attention": torch.cat(all_mask_attention, dim=0),   # (S, 197)
        "slices_weight": torch.cat(all_slices_weight, dim=0),     # (S,)
    }
    return result


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
    print(f"Loaded {len(patients)} patients from {csv_path}")

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    print(f"Loading BiomedCLIP model on {device}...")
    model, preprocess_val = load_model(device)
    print("Model loaded.")

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

            result = extract_sequence_features(
                model, preprocess_val, device, img_dir, ann_dir, has_ann
            )
            if result is not None:
                patient_data[seq] = result

        if patient_data:
            torch.save(patient_data, out_path)

        processed += 1
        if processed % 10 == 0:
            print(f"[{processed}/{len(patients)}] Processed {patient_id} "
                  f"(label={label}, sequences={len(patient_data)})")

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}")
    print(f"Features saved to: {output_dir}")


if __name__ == "__main__":
    main()
