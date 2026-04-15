import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())

from models.mri_feature_models import build_model
from utils.mri_pipeline import (
    build_dataloaders,
    build_weighted_classification_criterion,
    evaluate_model,
    resolve_mri_model_mode,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate MRI patient-level classifier from BiomedCLIP features")
    parser.add_argument("--saveName", type=str, default="mri_biomedclip")
    parser.add_argument("--feature_dir", type=str, default="/data/MRI_data/standardized/features/BiomedCLIP")
    parser.add_argument("--dataset_csv", type=str, default="/data/MRI_data/standardized/dataset.csv")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--target_depth", type=int, default=96)
    parser.add_argument("--model_mode", type=str, default="image", choices=["tabular", "image", "union"])
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=3407)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--weight_path", type=str, default="log/mri_biomedclip/model_final.pt")
    return parser.parse_args()


def format_metrics(prefix, metrics):
    dice_value = "nan" if metrics["dice"] is None else f"{metrics['dice']:.4f}"
    cm = metrics["confusion_matrix"].tolist()
    return (
        f"{prefix}: "
        f"loss={metrics['loss']:.4f}, "
        f"auc={metrics['auc']:.4f}, "
        f"acc={metrics['accuracy']:.4f}, "
        f"sen={metrics['sensitivity']:.4f}, "
        f"spe={metrics['specificity']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"dice={dice_value}, "
        f"cm={cm}"
    )


def main():
    args = get_arguments()
    set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join("log", args.saveName)
    os.makedirs(save_dir, exist_ok=True)

    print(args)
    print(f"Using device: {device}")

    dataloaders = build_dataloaders(args)
    requested_model_mode = args.model_mode
    effective_model_mode, model_mode_note = resolve_mri_model_mode(requested_model_mode, dataloaders["meta"])
    args.model_mode = effective_model_mode
    print(f"Requested model_mode={requested_model_mode}, effective model_mode={effective_model_mode}")
    if model_mode_note is not None:
        print(model_mode_note)

    model = build_model(args).to(device)
    model.load_state_dict(torch.load(args.weight_path, map_location=device))
    print(f"Loaded weights from: {args.weight_path}")
    criterion, class_weights = build_weighted_classification_criterion(
        dataloaders["meta"]["class_counts"],
        device,
    )
    class_counts = dataloaders["meta"]["class_counts"]
    print(
        "Train class counts: "
        f"negative={class_counts.get(0, 0)}, positive={class_counts.get(1, 0)}"
    )
    print(
        "Class weights: "
        f"class_0={class_weights[0].item():.6f}, class_1={class_weights[1].item():.6f}"
    )

    split_names = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split_name in split_names:
        loader_key = "train_eval" if split_name == "train" else split_name
        metrics = evaluate_model(
            model,
            dataloaders[loader_key],
            device,
            split_name,
            criterion=criterion,
            output_dir=save_dir,
        )
        print(format_metrics(split_name, metrics))


if __name__ == "__main__":
    main()
