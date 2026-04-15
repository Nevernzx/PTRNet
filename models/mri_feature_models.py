from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dice_score import dice_loss


STANDARD_SEQUENCES = [
    "T1",
    "T2",
    "ADC",
    "DWI",
    "DCE1",
    "DCE2",
    "DCE3",
    "DCE4",
    "DCE5",
    "DCE6",
    "DCE7",
]


class FCN(nn.Module):
    def __init__(self, in_channels=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return self.conv(x).sigmoid()


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim=768, d_model=256, target_depth=96):
        super().__init__()
        self.target_depth = target_depth
        self.mask_head = FCN(in_channels=input_dim)
        self.conv3 = nn.Sequential(
            nn.Conv3d(input_dim, 128, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(128, d_model, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

    def forward(self, patch_embed, mask_attention, slices_weight):
        if patch_embed.dim() == 5:
            patch_embed = patch_embed.squeeze(0)
        if mask_attention.dim() == 3:
            mask_attention = mask_attention.squeeze(0)
        if slices_weight.dim() == 2:
            slices_weight = slices_weight.squeeze(0)

        num_slices, grid_h, grid_w, _ = patch_embed.shape
        patch_tokens = patch_embed.permute(0, 3, 1, 2).contiguous().float()

        pred_mask = self.mask_head(patch_tokens).squeeze(1)
        gt_mask = mask_attention[:, 1:].reshape(num_slices, grid_h, grid_w).float()
        valid_slices = slices_weight > 0

        if torch.any(valid_slices):
            seg_loss = dice_loss(pred_mask, gt_mask, valid_mask=valid_slices)
        else:
            seg_loss = pred_mask.new_zeros(())

        weighted_tokens = patch_tokens * pred_mask.unsqueeze(1)
        volume = weighted_tokens.unsqueeze(0).permute(0, 2, 1, 3, 4)
        volume = F.interpolate(
            volume,
            size=(self.target_depth, grid_h, grid_w),
            mode="trilinear",
            align_corners=False,
        )

        seq_feat = self.conv3(volume).flatten(1)
        return seq_feat, pred_mask, seg_loss


class SequenceAttentionPool(nn.Module):
    def __init__(self, d_model, num_sequences):
        super().__init__()
        self.sequence_embed = nn.Embedding(num_sequences, d_model)
        self.score_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, seq_feats, seq_indices):
        seq_tokens = seq_feats + self.sequence_embed(seq_indices)
        attn_scores = self.score_head(seq_tokens).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=0)
        pooled_feat = torch.sum(seq_tokens * attn_weights.unsqueeze(-1), dim=0, keepdim=True)
        return pooled_feat, attn_weights


class TabularProjector(nn.Module):
    def __init__(self, output_dim, input_dim=27, hidden_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_categ, x_numer):
        if x_categ.dim() == 1:
            x_categ = x_categ.unsqueeze(0)
        if x_numer.dim() == 1:
            x_numer = x_numer.unsqueeze(0)

        features = torch.cat([x_categ.float(), x_numer.float()], dim=-1)
        if torch.sum(features.abs()) == 0:
            return features.new_zeros((features.shape[0], self.output_dim))
        return self.net(features)


class MRIImageBackbone(nn.Module):
    def __init__(self, input_dim=768, d_model=256, target_depth=96, sequences=None):
        super().__init__()
        self.sequences = sequences or STANDARD_SEQUENCES
        self.sequence_to_idx = {name: idx for idx, name in enumerate(self.sequences)}
        self.sequence_encoder = SequenceEncoder(input_dim=input_dim, d_model=d_model, target_depth=target_depth)
        self.sequence_pool = SequenceAttentionPool(d_model=d_model, num_sequences=len(self.sequences))

    def forward(self, img_data, inner_slice_mask, inter_slice_mask, mode="test"):
        seq_features = []
        seq_indices = []
        pred_masks = OrderedDict()
        seg_losses = []

        for seq_name in self.sequences:
            if seq_name not in img_data:
                continue

            seq_feat, pred_mask, seg_loss = self.sequence_encoder(
                img_data[seq_name],
                inner_slice_mask[seq_name],
                inter_slice_mask[seq_name],
            )
            seq_features.append(seq_feat.squeeze(0))
            seq_indices.append(self.sequence_to_idx[seq_name])
            pred_masks[seq_name] = pred_mask
            seg_losses.append(seg_loss)

        if not seq_features:
            raise RuntimeError("No valid MRI sequences were found in the loaded feature file.")

        seq_features = torch.stack(seq_features, dim=0)
        seq_indices = torch.tensor(seq_indices, dtype=torch.long, device=seq_features.device)
        pooled_feat, seq_weights = self.sequence_pool(seq_features, seq_indices)

        if seg_losses:
            seg_loss = torch.stack(seg_losses).mean()
        else:
            seg_loss = pooled_feat.new_zeros(())

        if mode == "vis":
            return pred_masks

        return pooled_feat, seg_loss, pred_masks, seq_weights


class image_model(nn.Module):
    def __init__(self, d_model=256, hidden_dim=128, input_dim=768, target_depth=96, sequences=None):
        super().__init__()
        self.backbone = MRIImageBackbone(
            input_dim=input_dim,
            d_model=d_model,
            target_depth=target_depth,
            sequences=sequences,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask, mode="test", return_aux=False):
        if mode == "vis":
            return self.backbone(img_data, inner_slice_mask, inter_slice_mask, mode="vis")

        image_feat, loss_extra, pred_masks, seq_weights = self.backbone(
            img_data,
            inner_slice_mask,
            inter_slice_mask,
            mode=mode,
        )
        logits = self.classifier(image_feat)
        if mode == "train":
            return logits, loss_extra
        if return_aux:
            return logits, {
                "pred_masks": pred_masks,
                "seq_weights": seq_weights,
                "seg_loss": loss_extra,
            }
        return logits


class union_model(nn.Module):
    def __init__(self, d_model=256, hidden_dim=128, input_dim=768, target_depth=96, sequences=None):
        super().__init__()
        self.backbone = MRIImageBackbone(
            input_dim=input_dim,
            d_model=d_model,
            target_depth=target_depth,
            sequences=sequences,
        )
        self.tabular_encoder = TabularProjector(output_dim=d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask, mode="test", return_aux=False):
        if mode == "vis":
            return self.backbone(img_data, inner_slice_mask, inter_slice_mask, mode="vis")

        image_feat, loss_extra, pred_masks, seq_weights = self.backbone(
            img_data,
            inner_slice_mask,
            inter_slice_mask,
            mode=mode,
        )
        tabular_feat = self.tabular_encoder(x_categ, x_numer)
        logits = self.classifier(torch.cat([image_feat, tabular_feat], dim=-1))

        if mode == "train":
            return logits, loss_extra
        if return_aux:
            return logits, {
                "pred_masks": pred_masks,
                "seq_weights": seq_weights,
                "seg_loss": loss_extra,
            }
        return logits


class tabular_model(nn.Module):
    def __init__(self, d_model=256, hidden_dim=128):
        super().__init__()
        self.tabular_encoder = TabularProjector(output_dim=d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask, mode="test", return_aux=False):
        tabular_feat = self.tabular_encoder(x_categ, x_numer)
        logits = self.classifier(tabular_feat)
        if mode == "train":
            return logits
        if return_aux:
            return logits, {
                "pred_masks": None,
                "seq_weights": None,
                "seg_loss": logits.new_zeros(()),
            }
        return logits


def build_model(args):
    model_mode = getattr(args, "model_mode", "union")
    input_dim = getattr(args, "input_dim", 768)
    target_depth = getattr(args, "target_depth", 96)
    d_model = getattr(args, "d_model", 256)
    hidden_dim = getattr(args, "hidden_dim", 128)

    if model_mode == "image":
        return image_model(
            d_model=d_model,
            hidden_dim=hidden_dim,
            input_dim=input_dim,
            target_depth=target_depth,
        )
    if model_mode == "tabular":
        return tabular_model(d_model=d_model, hidden_dim=hidden_dim)
    return union_model(
        d_model=d_model,
        hidden_dim=hidden_dim,
        input_dim=input_dim,
        target_depth=target_depth,
    )
