import argparse
import math
import os
import random
import sys

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

sys.path.append(os.getcwd())

from models.mri_feature_models import build_model
from utils.config import Logger
from utils.mri_pipeline import (
    build_dataloaders,
    build_weighted_classification_criterion,
    evaluate_model,
    prepare_batch,
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
    parser = argparse.ArgumentParser(description="Train MRI patient-level classifier from BiomedCLIP features")
    parser.add_argument("--saveName", type=str, default="mri_biomedclip")
    parser.add_argument("--feature_dir", type=str, default="/data/MRI_data/standardized/features/BiomedCLIP")
    parser.add_argument("--dataset_csv", type=str, default="/data/MRI_data/standardized/dataset.csv")
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--base_lr", type=float, default=3e-4)
    parser.add_argument("--init_lr", type=float, default=1e-7)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_warmup", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--target_depth", type=int, default=96)
    parser.add_argument("--seg_loss_weight", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--model_mode", type=str, default="image", choices=["tabular", "image", "union"])
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=3407)
    return parser.parse_args()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    raise RuntimeError("Optimizer has no parameter groups.")


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def build_epoch_lr_schedule(num_epochs, init_lr, base_lr, num_warmup):
    if num_epochs <= 0:
        return []

    num_warmup = max(0, int(num_warmup))
    lrs = []
    for epoch in range(num_epochs):
        if num_warmup > 0 and epoch < num_warmup:
            if num_warmup == 1:
                lr = init_lr
            else:
                progress = epoch / float(num_warmup - 1)
                lr = base_lr + (init_lr - base_lr) * (math.cos(math.pi * progress) + 1.0) / 2.0
        elif num_warmup == 0:
            if epoch == 0:
                lr = base_lr
            else:
                decay_epochs = max(1, num_epochs - 1)
                progress = epoch / float(decay_epochs)
                lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))
        else:
            decay_epochs = num_epochs - num_warmup
            if decay_epochs <= 0:
                lr = base_lr
            else:
                if num_warmup == 1:
                    progress = (epoch - num_warmup) / float(max(1, decay_epochs - 1))
                else:
                    progress = (epoch - num_warmup + 1) / float(decay_epochs)
                progress = min(max(progress, 0.0), 1.0)
                lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))
        lrs.append(lr)
    return lrs


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


def plot_progress_safe(epochs, train_losses, val_losses, val_f1s, val_aucs, lrs, save_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    ax.plot(epochs, train_losses, label="train_loss", color="tab:blue")
    ax.plot(epochs, val_losses, label="val_loss", color="tab:red")
    ax2.plot(epochs, val_f1s, label="val_f1", color="tab:green")
    ax2.plot(epochs, val_aucs, label="val_auc", color="tab:orange")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2.set_ylabel("metric")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "progress.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.plot(epochs, lrs, label="lr", color="tab:purple")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "lr.png"))
    plt.close(fig)


def train_one_epoch(model, dataloader, optimizer, criterion, device, args, epoch):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_seg_loss = 0.0

    tbar = tqdm(dataloader)
    for step, batch in enumerate(tbar):
        labels, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask = prepare_batch(batch, device)

        optimizer.zero_grad()

        if args.model_mode == "tabular":
            logits = model(x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask, mode="train")
            cls_loss = criterion(logits, labels)
            seg_loss = cls_loss.new_zeros(())
            loss = cls_loss
        else:
            logits, seg_loss = model(x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask, mode="train")
            cls_loss = criterion(logits, labels)
            loss = cls_loss + args.seg_loss_weight * seg_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_cls_loss += cls_loss.item()
        running_seg_loss += seg_loss.item() if torch.is_tensor(seg_loss) else float(seg_loss)

        tbar.set_description(
            f"Epoch {epoch}/{args.num_epochs - 1}, "
            f"step {step}/{len(dataloader) - 1}, "
            f"loss={loss.item():.4f}, "
            f"cls={cls_loss.item():.4f}, "
            f"seg={running_seg_loss / (step + 1):.4f}, "
            f"lr={get_lr(optimizer):.6f}"
        )

    num_batches = max(1, len(dataloader))
    return {
        "loss": running_loss / num_batches,
        "cls_loss": running_cls_loss / num_batches,
        "seg_loss": running_seg_loss / num_batches,
    }


def main():
    args = get_arguments()
    set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.savePath = os.path.join("log", args.saveName)
    os.makedirs(args.savePath, exist_ok=True)
    Logger(os.path.join(args.savePath, "train.log"))

    print(args)
    print(f"Using device: {device}")

    dataloaders = build_dataloaders(args)
    requested_model_mode = args.model_mode
    effective_model_mode, model_mode_note = resolve_mri_model_mode(requested_model_mode, dataloaders["meta"])
    args.model_mode = effective_model_mode
    print(
        f"Dataset split sizes: "
        f"train={len(dataloaders['train'].dataset)}, "
        f"val={len(dataloaders['val'].dataset)}, "
        f"test={len(dataloaders['test'].dataset)}"
    )
    print(f"Requested model_mode={requested_model_mode}, effective model_mode={effective_model_mode}")
    if model_mode_note is not None:
        print(model_mode_note)

    model = build_model(args).to(device)
    print(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params * 1e-6:.3f} M")
    print(f"Total params: {total_params * 1e-6:.3f} M")

    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    lr_schedule = build_epoch_lr_schedule(
        num_epochs=args.num_epochs,
        init_lr=args.init_lr,
        base_lr=args.base_lr,
        num_warmup=args.num_warmup,
    )
    if not lr_schedule:
        raise ValueError("num_epochs must be positive.")
    set_lr(optimizer, lr_schedule[0])

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
    print(f"Initial lr={get_lr(optimizer):.6f}")

    epochs = []
    train_losses = []
    val_losses = []
    val_f1s = []
    val_aucs = []
    lrs = []

    best_val_auc = -float("inf")
    best_val_acc = -float("inf")
    best_epoch = -1
    best_test_metrics = None

    for epoch in range(args.num_epochs):
        print("-" * 10)
        print(f"Train Epoch {epoch}/{args.num_epochs - 1}")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            args=args,
            epoch=epoch,
        )
        current_lr = get_lr(optimizer)
        print(
            f"Train: loss={train_metrics['loss']:.4f}, "
            f"cls_loss={train_metrics['cls_loss']:.4f}, "
            f"seg_loss={train_metrics['seg_loss']:.4f}, "
            f"lr={current_lr:.6f}"
        )

        if epoch + 1 < len(lr_schedule):
            set_lr(optimizer, lr_schedule[epoch + 1])

        if (epoch + 1) % args.eval_every != 0:
            continue

        val_metrics = evaluate_model(
            model,
            dataloaders["val"],
            device,
            "val",
            criterion=criterion,
            output_dir=args.savePath,
        )
        print(format_metrics("Val", val_metrics))

        test_metrics = evaluate_model(
            model,
            dataloaders["test"],
            device,
            "test",
            criterion=criterion,
            output_dir=args.savePath,
        )
        print(format_metrics("Test", test_metrics))

        epochs.append(epoch + 1)
        train_losses.append(train_metrics["loss"])
        val_losses.append(val_metrics["loss"])
        val_f1s.append(val_metrics["f1"])
        val_aucs.append(val_metrics["auc"] * 100.0)
        lrs.append(current_lr)
        plot_progress_safe(epochs, train_losses, val_losses, val_f1s, val_aucs, lrs, args.savePath)

        better_auc = val_metrics["auc"] > best_val_auc
        same_auc = np.isclose(val_metrics["auc"], best_val_auc)
        better_acc = val_metrics["accuracy"] > best_val_acc
        if better_auc or (same_auc and better_acc):
            best_val_auc = val_metrics["auc"]
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            best_test_metrics = test_metrics
            torch.save(model.state_dict(), os.path.join(args.savePath, "model_final.pt"))
            print(f"Saved best checkpoint at epoch {epoch}")

    print("-" * 20)
    print(f"Best epoch: {best_epoch}")
    print(f"Best val auc: {best_val_auc:.4f}")
    print(f"Best val acc: {best_val_acc:.4f}")
    if best_test_metrics is not None:
        print(format_metrics("Best test", best_test_metrics))
    print("-" * 20)


if __name__ == "__main__":
    main()
