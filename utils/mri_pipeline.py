import csv
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from utils.dataset_ANCA import ANCAdataset, STANDARD_SEQUENCES
from utils.dice_score import dice_coeff


def move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(move_to_device(value, device) for value in data)
    return data


def mri_collate_fn(batch):
    """Custom collate that pads variable-length MRI sequences across a batch.

    Each sample from ANCAdataset (standardized_mri mode) is:
        (label, x_categ, x_numer, img_data_dict, inner_mask_dict, inter_mask_dict)
    where the dicts are keyed by sequence name with variable-shaped tensors.

    Returns:
        labels:           (B,)
        x_categ:          (B, 4)
        x_numer:          (B, 23)
        padded_img:       dict[seq] -> (B, S_max_seq, 14, 14, 768)   zero-padded
        padded_inner:     dict[seq] -> (B, S_max_seq, 197)            zero-padded
        padded_inter:     dict[seq] -> (B, S_max_seq)                 zero-padded
        seq_presence:     (B, num_sequences) bool
        seq_num_slices:   (B, num_sequences) long
    """
    labels = torch.stack([s[0] for s in batch])
    x_categ = torch.stack([s[1] for s in batch])
    x_numer = torch.stack([s[2] for s in batch])

    B = len(batch)
    sequences = STANDARD_SEQUENCES
    num_seq = len(sequences)

    seq_presence = torch.zeros(B, num_seq, dtype=torch.bool)
    seq_num_slices = torch.zeros(B, num_seq, dtype=torch.long)

    # First pass: determine max slices per sequence and presence
    max_slices = {}
    for seq_idx, seq in enumerate(sequences):
        s_max = 0
        for b in range(B):
            img_dict = batch[b][3]
            if seq in img_dict:
                seq_presence[b, seq_idx] = True
                n = img_dict[seq].shape[0]
                seq_num_slices[b, seq_idx] = n
                s_max = max(s_max, n)
        if s_max > 0:
            max_slices[seq] = s_max

    # Second pass: pad and stack
    padded_img = {}
    padded_inner = {}
    padded_inter = {}

    for seq_idx, seq in enumerate(sequences):
        if seq not in max_slices:
            continue
        s_max = max_slices[seq]

        img_list = []
        inner_list = []
        inter_list = []
        for b in range(B):
            img_dict = batch[b][3]
            if seq in img_dict:
                pe = img_dict[seq]           # (S, 14, 14, 768)
                ma = batch[b][4][seq]        # (S, 197)
                sw = batch[b][5][seq]        # (S,)
                n = pe.shape[0]
                if n < s_max:
                    pe = torch.cat([pe, pe.new_zeros(s_max - n, *pe.shape[1:])], dim=0)
                    ma = torch.cat([ma, ma.new_zeros(s_max - n, *ma.shape[1:])], dim=0)
                    sw = torch.cat([sw, sw.new_zeros(s_max - n)], dim=0)
                img_list.append(pe)
                inner_list.append(ma)
                inter_list.append(sw)
            else:
                # Patient doesn't have this sequence — fill with zeros
                sample_img = batch[0][3]
                ref_seq = next(iter(sample_img))
                ref = sample_img[ref_seq]
                img_list.append(ref.new_zeros(s_max, *ref.shape[1:]))
                inner_list.append(torch.zeros(s_max, 197))
                inter_list.append(torch.zeros(s_max))

        padded_img[seq] = torch.stack(img_list, dim=0)      # (B, S_max, 14, 14, 768)
        padded_inner[seq] = torch.stack(inner_list, dim=0)   # (B, S_max, 197)
        padded_inter[seq] = torch.stack(inter_list, dim=0)   # (B, S_max)

    return (labels, x_categ, x_numer, padded_img, padded_inner, padded_inter,
            seq_presence, seq_num_slices)


def prepare_batch(batch, device):
    labels, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask, seq_presence, seq_num_slices = batch
    return (
        labels.to(device),
        x_categ.to(device),
        x_numer.to(device),
        move_to_device(img_data, device),
        move_to_device(inner_slice_mask, device),
        move_to_device(inter_slice_mask, device),
        seq_presence.to(device),
        seq_num_slices.to(device),
    )


def build_dataloaders(args):
    dataset_kwargs = dict(
        root=args.feature_dir,
        csv_path=args.dataset_csv,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_slices=getattr(args, "max_slices", 96),
    )

    train_dataset = ANCAdataset(TrainValTest="train", **dataset_kwargs)
    val_dataset = ANCAdataset(TrainValTest="val", **dataset_kwargs)
    test_dataset = ANCAdataset(TrainValTest="test", **dataset_kwargs)

    collate_fn = mri_collate_fn if train_dataset.dataset_mode == "standardized_mri" else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    class_counts = {label: 0 for label in range(2)}
    for item in train_dataset.items:
        label = int(item["label"])
        class_counts[label] = class_counts.get(label, 0) + 1

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_eval": train_eval_loader,
        "meta": {
            "dataset_mode": train_dataset.dataset_mode,
            "has_tabular_features": train_dataset.has_tabular_features,
            "class_counts": class_counts,
        },
    }


def _compute_segmentation_dice(pred_masks, inner_slice_mask, inter_slice_mask, seq_presence):
    """Compute dice scores from batched predictions.

    pred_masks: dict[seq] -> (B, S, H, W) or (B_sub, S, H, W)
    inner_slice_mask: dict[seq] -> (B, S, 197)
    inter_slice_mask: dict[seq] -> (B, S)
    seq_presence: (B, num_seq) bool
    """
    dice_scores = []

    for seq_name, pred_mask in pred_masks.items():
        gt_mask = inner_slice_mask[seq_name]
        slice_weight = inter_slice_mask[seq_name]

        # pred_mask: (B_sub, S, H, W), gt_mask: (B, S, 197), slice_weight: (B, S)
        grid_h, grid_w = pred_mask.shape[-2:]
        B_sub = pred_mask.shape[0]

        for b in range(B_sub):
            valid_slices = slice_weight[b] > 0
            if not torch.any(valid_slices):
                continue
            gt_b = gt_mask[b][:, 1:].reshape(-1, grid_h, grid_w).float()
            pred_binary = (pred_mask[b][valid_slices] >= 0.5).float()
            dice_scores.append(dice_coeff(pred_binary, gt_b[valid_slices], reduce_batch_first=True).item())

    return dice_scores


def resolve_mri_model_mode(requested_model_mode, meta):
    has_tabular_features = meta.get("has_tabular_features", True)
    if requested_model_mode in {"tabular", "union"} and not has_tabular_features:
        return "image", (
            f"Requested model_mode={requested_model_mode} but dataset_mode={meta.get('dataset_mode')} "
            "has no tabular features. Falling back to image."
        )
    return requested_model_mode, None


def compute_class_weights(class_counts, num_classes=2):
    ordered_counts = [int(class_counts.get(label, 0)) for label in range(num_classes)]
    total = float(sum(ordered_counts))
    if total <= 0:
        raise ValueError("class_counts must contain at least one sample.")

    weights = []
    for count in ordered_counts:
        if count <= 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_classification_criterion(class_counts, device, num_classes=2):
    weights = compute_class_weights(class_counts, num_classes=num_classes).to(device)
    return nn.CrossEntropyLoss(weight=weights), weights


def compute_classification_metrics(labels, probs, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    preds = (probs >= threshold).astype(np.int64)

    acc = accuracy_score(labels, preds)
    sen = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")

    return {
        "accuracy": acc,
        "sensitivity": sen,
        "specificity": spe,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "predictions": preds,
    }


def evaluate_model(model, dataloader, device, split_name, criterion=None, output_dir=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()

    all_labels = []
    all_probs = []
    patient_ids = []
    running_loss = 0.0
    dice_scores = []

    with torch.no_grad():
        for batch in dataloader:
            labels, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask, seq_presence, seq_num_slices = prepare_batch(batch, device)
            outputs = model(
                x_categ,
                x_numer,
                img_data,
                inner_slice_mask,
                inter_slice_mask,
                seq_presence=seq_presence,
                seq_num_slices=seq_num_slices,
                mode="test",
                return_aux=True,
            )
            if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict):
                logits, aux = outputs
            else:
                logits, aux = outputs, {}
            probs = torch.softmax(logits, dim=1)[:, 1]

            running_loss += criterion(logits, labels).item()
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

            # Collect patient IDs for each sample in the batch
            batch_size = labels.shape[0]
            dataset = dataloader.dataset
            for _ in range(batch_size):
                patient_ids.append(dataset.getFileName())

            pred_masks = aux.get("pred_masks")
            if isinstance(pred_masks, dict):
                dice_scores.extend(_compute_segmentation_dice(
                    pred_masks, inner_slice_mask, inter_slice_mask, seq_presence))

    metrics = compute_classification_metrics(all_labels, all_probs)
    metrics["loss"] = running_loss / max(1, len(dataloader))
    metrics["dice"] = float(np.mean(dice_scores)) if dice_scores else None

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_csv = os.path.join(output_dir, f"{split_name}_predictions.csv")
        with open(out_csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["patient_id", "prob_positive", "pred_label", "label"])
            for patient_id, prob, pred, label in zip(
                patient_ids,
                all_probs,
                metrics["predictions"],
                all_labels,
            ):
                writer.writerow([patient_id, prob, int(pred), label])

        metrics["prediction_csv"] = out_csv

    return metrics
