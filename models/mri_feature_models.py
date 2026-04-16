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

    def forward(self, patch_embed, mask_attention, slices_weight, num_slices=None):
        """
        Args:
            patch_embed:    (B, S, H, W, C) or (S, H, W, C)
            mask_attention: (B, S, 197) or (S, 197)
            slices_weight:  (B, S) or (S,)
            num_slices:     (B,) long — actual (unpadded) slice count per sample, or None
        Returns:
            seq_feat:  (B, d_model)
            pred_mask: (B, S, H, W)
            seg_loss:  scalar
        """
        # Ensure batch dimension
        if patch_embed.dim() == 4:
            patch_embed = patch_embed.unsqueeze(0)
            mask_attention = mask_attention.unsqueeze(0)
            slices_weight = slices_weight.unsqueeze(0)

        B, S, grid_h, grid_w, C = patch_embed.shape

        # (B*S, C, H, W)
        patch_tokens = patch_embed.reshape(B * S, grid_h, grid_w, C).permute(0, 3, 1, 2).contiguous().float()

        # Predict masks: (B*S, 1, H, W) -> (B, S, H, W)
        pred_mask = self.mask_head(patch_tokens).reshape(B, S, grid_h, grid_w)
        gt_mask = mask_attention[:, :, 1:].reshape(B, S, grid_h, grid_w).float()

        # Compute dice loss per sample, only on valid (unpadded + annotated) slices
        seg_losses = []
        for b in range(B):
            n = num_slices[b].item() if num_slices is not None else S
            valid = slices_weight[b, :n] > 0
            if torch.any(valid):
                seg_losses.append(dice_loss(pred_mask[b, :n], gt_mask[b, :n], valid_mask=valid))
        if seg_losses:
            seg_loss = torch.stack(seg_losses).mean()
        else:
            seg_loss = pred_mask.new_zeros(())

        # Build slice validity mask to zero out padded slices: (B, 1, S, 1, 1)
        if num_slices is not None:
            slice_valid = torch.arange(S, device=patch_embed.device).unsqueeze(0) < num_slices.unsqueeze(1)  # (B, S)
            slice_valid = slice_valid.float().reshape(B, S, 1, 1, 1)
        else:
            slice_valid = 1.0

        # Weighted tokens: (B, S, C, H, W)
        weighted_tokens = patch_tokens.reshape(B, S, C, grid_h, grid_w) * pred_mask.unsqueeze(2)
        # Apply slice validity mask
        weighted_tokens = weighted_tokens * slice_valid

        # (B, C, S, H, W) for 3D conv
        volume = weighted_tokens.permute(0, 2, 1, 3, 4)
        volume = F.interpolate(
            volume,
            size=(self.target_depth, grid_h, grid_w),
            mode="trilinear",
            align_corners=False,
        )

        seq_feat = self.conv3(volume).flatten(1)  # (B, d_model)
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

    def forward(self, seq_feats, seq_indices, seq_mask=None):
        """
        Args:
            seq_feats:   (B, num_seq, d_model) or (num_seq, d_model)
            seq_indices: (num_seq,) long — sequence index for embedding lookup
            seq_mask:    (B, num_seq) bool — True for present sequences, or None
        Returns:
            pooled_feat: (B, d_model)
            attn_weights: (B, num_seq)
        """
        unbatched = seq_feats.dim() == 2
        if unbatched:
            seq_feats = seq_feats.unsqueeze(0)
            if seq_mask is not None:
                seq_mask = seq_mask.unsqueeze(0)

        # seq_indices: (num_seq,) -> embedding: (1, num_seq, d_model), broadcast over B
        seq_tokens = seq_feats + self.sequence_embed(seq_indices).unsqueeze(0)
        attn_scores = self.score_head(seq_tokens).squeeze(-1)  # (B, num_seq)

        if seq_mask is not None:
            attn_scores = attn_scores.masked_fill(~seq_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, num_seq)
        # Replace NaN from all-masked rows (shouldn't happen, but safe)
        attn_weights = attn_weights.nan_to_num(0.0)
        pooled_feat = torch.sum(seq_tokens * attn_weights.unsqueeze(-1), dim=1)  # (B, d_model)

        if unbatched:
            pooled_feat = pooled_feat.squeeze(0)
            attn_weights = attn_weights.squeeze(0)

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

    def forward(self, img_data, inner_slice_mask, inter_slice_mask, mode="test",
                seq_presence=None, seq_num_slices=None):
        """
        Args:
            img_data:         dict[seq] -> (B, S, H, W, C) padded tensors
            inner_slice_mask: dict[seq] -> (B, S, 197)
            inter_slice_mask: dict[seq] -> (B, S)
            seq_presence:     (B, num_sequences) bool — which seqs each patient has
            seq_num_slices:   (B, num_sequences) long — actual slice counts
        """
        # Determine batch size from first available sequence
        first_key = next(iter(img_data))
        B = img_data[first_key].shape[0]
        device = img_data[first_key].device
        num_seq = len(self.sequences)

        # Storage: (B, num_seq, d_model), allocated after first encoder call
        all_seq_feats = None
        pred_masks = OrderedDict()
        seg_losses = []
        seq_indices_list = []

        # Build presence mask if not provided (legacy single-sample path)
        if seq_presence is None:
            seq_presence = torch.zeros(B, num_seq, dtype=torch.bool, device=device)
            for seq_idx, seq_name in enumerate(self.sequences):
                if seq_name in img_data:
                    seq_presence[:, seq_idx] = True

        for seq_idx, seq_name in enumerate(self.sequences):
            if seq_name not in img_data:
                continue

            # Get patients that have this sequence
            has_seq = seq_presence[:, seq_idx]  # (B,)
            if not torch.any(has_seq):
                continue

            # Extract data for patients that have this sequence
            pe = img_data[seq_name][has_seq]            # (B_sub, S, H, W, C)
            ma = inner_slice_mask[seq_name][has_seq]    # (B_sub, S, 197)
            sw = inter_slice_mask[seq_name][has_seq]    # (B_sub, S)

            ns = None
            if seq_num_slices is not None:
                ns = seq_num_slices[has_seq, seq_idx]   # (B_sub,)

            seq_feat, pred_mask, seg_loss = self.sequence_encoder(pe, ma, sw, num_slices=ns)

            if all_seq_feats is None:
                d_model = seq_feat.shape[-1]
                all_seq_feats = torch.zeros(B, num_seq, d_model, device=device)

            all_seq_feats[has_seq, seq_idx] = seq_feat
            pred_masks[seq_name] = pred_mask
            seg_losses.append(seg_loss)
            seq_indices_list.append(seq_idx)

        if not seq_indices_list:
            raise RuntimeError("No valid MRI sequences were found in the loaded feature file.")

        # Build index tensor for the sequences that exist in the batch
        seq_indices = torch.tensor(seq_indices_list, dtype=torch.long, device=device)  # (num_present_seq,)

        # Gather only the sequence slots that are present in the data
        batch_seq_feats = all_seq_feats[:, seq_indices_list, :]  # (B, num_present_seq, d_model)
        batch_seq_mask = seq_presence[:, seq_indices_list]        # (B, num_present_seq) bool

        pooled_feat, seq_weights = self.sequence_pool(
            batch_seq_feats, seq_indices, seq_mask=batch_seq_mask
        )

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

    def forward(self, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask,
                mode="test", return_aux=False, seq_presence=None, seq_num_slices=None):
        if mode == "vis":
            return self.backbone(img_data, inner_slice_mask, inter_slice_mask, mode="vis",
                                 seq_presence=seq_presence, seq_num_slices=seq_num_slices)

        image_feat, loss_extra, pred_masks, seq_weights = self.backbone(
            img_data,
            inner_slice_mask,
            inter_slice_mask,
            mode=mode,
            seq_presence=seq_presence,
            seq_num_slices=seq_num_slices,
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

    def forward(self, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask,
                mode="test", return_aux=False, seq_presence=None, seq_num_slices=None):
        if mode == "vis":
            return self.backbone(img_data, inner_slice_mask, inter_slice_mask, mode="vis",
                                 seq_presence=seq_presence, seq_num_slices=seq_num_slices)

        image_feat, loss_extra, pred_masks, seq_weights = self.backbone(
            img_data,
            inner_slice_mask,
            inter_slice_mask,
            mode=mode,
            seq_presence=seq_presence,
            seq_num_slices=seq_num_slices,
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

    def forward(self, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask,
                mode="test", return_aux=False, seq_presence=None, seq_num_slices=None):
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
