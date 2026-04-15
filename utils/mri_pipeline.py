import csv
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from utils.dataset_ANCA import ANCAdataset
from utils.dice_score import dice_coeff


def move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(move_to_device(value, device) for value in data)
    return data


def prepare_batch(batch, device):
    labels, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask = batch
    return (
        labels.to(device),
        x_categ.to(device),
        x_numer.to(device),
        move_to_device(img_data, device),
        move_to_device(inner_slice_mask, device),
        move_to_device(inter_slice_mask, device),
    )


def build_dataloaders(args):
    if args.batch_size != 1:
        raise ValueError("MRI feature training currently supports batch_size=1 only.")

    dataset_kwargs = dict(
        root=args.feature_dir,
        csv_path=args.dataset_csv,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    train_dataset = ANCAdataset(TrainValTest="train", **dataset_kwargs)
    val_dataset = ANCAdataset(TrainValTest="val", **dataset_kwargs)
    test_dataset = ANCAdataset(TrainValTest="test", **dataset_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
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


def _compute_segmentation_dice(pred_masks, inner_slice_mask, inter_slice_mask):
    dice_scores = []

    for seq_name, pred_mask in pred_masks.items():
        gt_mask = inner_slice_mask[seq_name]
        slice_weight = inter_slice_mask[seq_name]

        if gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        if slice_weight.dim() == 2:
            slice_weight = slice_weight.squeeze(0)

        valid_slices = slice_weight > 0
        if not torch.any(valid_slices):
            continue

        grid_h, grid_w = pred_mask.shape[-2:]
        gt_mask = gt_mask[:, 1:].reshape(pred_mask.shape[0], grid_h, grid_w).float()
        pred_binary = (pred_mask[valid_slices] >= 0.5).float()
        dice_scores.append(dice_coeff(pred_binary, gt_mask[valid_slices], reduce_batch_first=True).item())

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
            labels, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask = prepare_batch(batch, device)
            outputs = model(
                x_categ,
                x_numer,
                img_data,
                inner_slice_mask,
                inter_slice_mask,
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
            patient_ids.append(dataloader.dataset.getFileName())

            pred_masks = aux.get("pred_masks")
            if isinstance(pred_masks, dict):
                dice_scores.extend(_compute_segmentation_dice(pred_masks, inner_slice_mask, inter_slice_mask))

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
